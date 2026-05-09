# Combo variant of 2d0172a1 (index 31)
# combo_spec: D14/0+D15/0+D16/1
# Name: 边缘控制像素+强制垂直排列+规范化后镜像

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import find_components, get_bg, label_mask, paste


"""
design_analysis:
  a_concretization_and_trace:
    - 原题先从粗糙 frame/blob 图恢复包含树，再按原始 row/col band 关系生成规范矩形框+点图。
    - 本融合版先读边缘控制像素，再忽略其颜色；主体规范化阶段强制所有 frame 内的直接子节点都改为
      纵向堆叠，最后再按控制边指定的镜像轴翻转整张规范图。
    - 具体例子：若外框里原本是左右并排的 inner-frame 与 blob，外框外右侧还有一个 blob，
      原题会输出“内层兄弟仍大致横向关系 + 外侧点在右边”；本题先把内层兄弟改成上下排列，
      再若控制像素在左/右边，则整张规范图做左右镜像，于是外侧点跑到另一边，内层上下栈保持不变。
      若控制像素在上/下边，则做上下镜像，外侧点会换到上下相反的一侧。
  b_complexity_calibration:
    - 控制边只编码两种镜像轴：左/右边 => 左右镜像；上/下边 => 上下镜像。这样有可见分支，
      但自由度仍然很小。
    - 为避免 D15 退化成“左右等价、上下等价”的弱提示，generator 让 outside blob 与控制边同侧，
      这样镜像后的最终位置会稳定跑到对侧，四条边都各自可观察。
    - 为避免 D14 退化，generator 偏置输入 sibling 为 horizontal/diagonal，并要求新规范图
      与原题规范图不同。
    - 为避免 D16 退化，generator 强制存在一个与目标镜像轴同向的 outside blob，并拒绝
      “镜像前后完全相同”的规范图。
  c_implicit_assumption_audit:
    - 原 main() 假设唯一非背景色就是主体前景；这里先识别边界控制色并擦除，否则它会被误判为 blob。
    - 原 _layout_group() 假设 sibling 的 row/col overlap 决定 canonical 布局；融合后这个前提失效，
      必须改为按原始 top-row 排序的固定纵向堆叠。
    - 原 main() 在得到 canonical mask 后直接返回；融合后还必须追加镜像步骤。
    - 控制色识别依赖“主体前景不触边”；generator 明确给主体保留至少 1 格边距。
    - 镜像步骤依赖控制边只出现一条边且不在角上；generator 用同色 1-2 个非角落边缘像素来保证。
  d_placement_legality:
    - frame/blob 的粗糙输入仍用显式 gap 和 margin 放置，避免重叠。
    - outside blob 以正间距贴在 root frame 一侧，但整体绘制后仍与网格边界留出至少 1 格背景，
      给控制像素留出专用边界槽位。
    - 所有候选样本都经过新 main() 回算校验。
  e_visual_observability:
    - 输入中可见第三种颜色的小边缘标记；输出中可见 mirror 造成的外侧点换边/换上下。
    - 同时，frame 内部若原本横排或斜排，输出会变成纵排，确保 D14 也肉眼可见。
    - 样本筛选要求 verticalized canonical 与 legacy canonical 不同，且所选镜像轴确实改变结果。
  f_visual_cleanliness:
    - 控制像素只占一条外边，不侵入主体区域。
    - 输出仍是整洁的单色矩形框与点图；新增变化只体现在 sibling 纵排与整体镜像。
  g_inferability:
    - 颜色新增仅一种控制色，且它总是边缘孤立标记，不参与对象排序。
    - 颜色规则由位置结构支撑：看它在左右边还是上下边，就能归纳出镜像轴；
      再看 outside blob 总从控制边的同侧翻到对侧，就能归纳出精确边向；看内部 sibling
      总被整理成竖列，就能归纳出规范化方式。
"""


_EDGE_TO_AXIS = {
    "left": "horizontal",
    "right": "horizontal",
    "top": "vertical",
    "bottom": "vertical",
}

_AXIS_TO_EDGES = {
    "horizontal": ("left", "right"),
    "vertical": ("top", "bottom"),
}


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.is_frame = False
        self.parent: _Node | None = None
        self.children: list[_Node] = []
        self.mask: np.ndarray | None = None
        self.anchor: tuple[int, int] | None = None
        self.axis_map: dict[str, list[tuple[float, int]]] = {"row": [], "col": []}


class _Obj:
    def __init__(self, kind: str, children=None, outside_side: str | None = None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side
        self.layout_mode: str | None = None


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _vertical_sort_key(node: _Node) -> tuple[int, int, float, float]:
    r0, c0, _r1, _c1 = node.bbox
    return (int(r0), int(c0), float(node.center[0]), float(node.center[1]))


def _legacy_axis_coordinates(nodes: list[_Node], axis: int) -> dict[_Node, int]:
    items = []
    for node in nodes:
        assert node.mask is not None and node.anchor is not None
        r0, c0, r1, c1 = node.bbox
        interval = (r0, r1) if axis == 0 else (c0, c1)
        mid = (interval[0] + interval[1]) / 2
        items.append((node, interval, mid, node.anchor[axis], node.mask.shape[axis]))

    bands: list[dict] = []
    for item in sorted(items, key=lambda x: x[2]):
        interval = item[1]
        for band in bands:
            if any(_intervals_overlap(interval, other[1]) for other in band["items"]):
                band["items"].append(item)
                band["interval"] = (
                    min(band["interval"][0], interval[0]),
                    max(band["interval"][1], interval[1]),
                )
                break
        else:
            bands.append({"items": [item], "interval": interval})

    coord_by_node: dict[_Node, int] = {}
    coord = 0
    for i, band in enumerate(bands):
        if i:
            prev = bands[i - 1]
            prev_down = max(item[4] - 1 - item[3] for item in prev["items"])
            curr_up = max(item[3] for item in band["items"])
            coord += prev_down + 2 + curr_up
        for item in band["items"]:
            coord_by_node[item[0]] = int(coord)
    return coord_by_node


def _build_layout(node: _Node, mode: str) -> None:
    if node.mask is not None:
        return

    if not node.is_frame:
        node.mask = np.ones((1, 1), dtype=bool)
        node.anchor = (0, 0)
        node.axis_map = {
            "row": [(float(node.center[0]), 0)],
            "col": [(float(node.center[1]), 0)],
        }
        return

    assert node.children
    child_mask, _placements, axis_map = _layout_group(node.children, mode)
    h, w = child_mask.shape
    out = np.zeros((h + 4, w + 4), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    out[2 : 2 + h, 2 : 2 + w] |= child_mask

    node.mask = out
    node.anchor = (out.shape[0] // 2, out.shape[1] // 2)
    node.axis_map = {
        "row": [(orig, pos + 2) for orig, pos in axis_map["row"]],
        "col": [(orig, pos + 2) for orig, pos in axis_map["col"]],
    }


def _layout_group(
    nodes: list[_Node],
    mode: str,
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node, mode)

    if mode == "vertical":
        ordered = sorted(nodes, key=_vertical_sort_key)
        row_coord: dict[_Node, int] = {}
        col_coord: dict[_Node, int] = {node: 0 for node in ordered}
        coord = 0
        for i, node in enumerate(ordered):
            assert node.mask is not None and node.anchor is not None
            if i:
                prev = ordered[i - 1]
                assert prev.mask is not None and prev.anchor is not None
                coord += prev.mask.shape[0] - 1 - prev.anchor[0] + 2 + node.anchor[0]
            row_coord[node] = int(coord)
    else:
        ordered = list(nodes)
        row_coord = _legacy_axis_coordinates(nodes, axis=0)
        col_coord = _legacy_axis_coordinates(nodes, axis=1)

    min_r = min(row_coord[n] - n.anchor[0] for n in ordered if n.anchor is not None)
    min_c = min(col_coord[n] - n.anchor[1] for n in ordered if n.anchor is not None)
    max_r = max(
        row_coord[n] - n.anchor[0] - min_r + n.mask.shape[0]
        for n in ordered
        if n.anchor is not None and n.mask is not None
    )
    max_c = max(
        col_coord[n] - n.anchor[1] - min_c + n.mask.shape[1]
        for n in ordered
        if n.anchor is not None and n.mask is not None
    )

    out = np.zeros((max_r, max_c), dtype=bool)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}
    for node in ordered:
        assert node.mask is not None and node.anchor is not None
        r = row_coord[node] - node.anchor[0] - min_r
        c = col_coord[node] - node.anchor[1] - min_c
        placements[node] = (int(r), int(c))
        out[r : r + node.mask.shape[0], c : c + node.mask.shape[1]] |= node.mask
        axis_map["row"].append((float(node.center[0]), int(r + node.anchor[0])))
        axis_map["col"].append((float(node.center[1]), int(c + node.anchor[1])))

    return out, placements, axis_map


def _outside_side(dot: _Node, frame: _Node) -> str:
    r0, c0, r1, c1 = dot.bbox
    fr0, fc0, fr1, fc1 = frame.bbox
    if c0 > fc1:
        return "right"
    if c1 < fc0:
        return "left"
    if r0 > fr1:
        return "bottom"
    if r1 < fr0:
        return "top"

    dr = float(dot.center[0] - (fr0 + fr1) / 2)
    dc = float(dot.center[1] - (fc0 + fc1) / 2)
    if abs(dc) >= abs(dr):
        return "right" if dc > 0 else "left"
    return "bottom" if dr > 0 else "top"


def _map_frame_axis(frame: _Node, axis: int, value: float) -> int:
    assert frame.anchor is not None
    key = "row" if axis == 0 else "col"
    candidates = frame.axis_map[key]
    if not candidates:
        return int(frame.anchor[axis])
    return int(min(candidates, key=lambda item: abs(item[0] - value))[1])


def _layout_root(nodes: list[_Node], mode: str) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node, mode)

    frames = [node for node in nodes if node.is_frame]
    dots = [node for node in nodes if not node.is_frame]
    if len(frames) == 1 and dots:
        frame = frames[0]
        assert frame.mask is not None
        placements: dict[_Node, tuple[int, int]] = {frame: (0, 0)}
        extents = [(0, 0, frame.mask.shape[0], frame.mask.shape[1])]

        for dot in dots:
            side = _outside_side(dot, frame)
            if side in {"right", "left"}:
                r = _map_frame_axis(frame, 0, float(dot.center[0]))
                c = frame.mask.shape[1] + 1 if side == "right" else -2
            else:
                r = frame.mask.shape[0] + 1 if side == "bottom" else -2
                c = _map_frame_axis(frame, 1, float(dot.center[1]))
            placements[dot] = (int(r), int(c))
            extents.append((r - 1, c - 1, r + 2, c + 2))

        min_r = min(e[0] for e in extents)
        min_c = min(e[1] for e in extents)
        max_r = max(e[2] for e in extents)
        max_c = max(e[3] for e in extents)
        out = np.zeros((max_r - min_r, max_c - min_c), dtype=bool)
        for node, (r, c) in placements.items():
            rr = r - min_r
            cc = c - min_c
            if node is frame:
                out[rr : rr + frame.mask.shape[0], cc : cc + frame.mask.shape[1]] |= frame.mask
            else:
                out[rr, cc] = True
        return out

    mask, _placements, _axis_map = _layout_group(nodes, mode)
    return mask


def _mirror_axis_from_edge(edge: str) -> str:
    return _EDGE_TO_AXIS[edge]


def _apply_variant_mirror(mask: np.ndarray, edge: str) -> np.ndarray:
    axis = _mirror_axis_from_edge(edge)
    return np.fliplr(mask) if axis == "horizontal" else np.flipud(mask)


def _control_edge(grid: np.ndarray, bg: int) -> tuple[int, str]:
    H, W = grid.shape
    border_colors = set(map(int, grid[0, :])) | set(map(int, grid[-1, :])) | set(map(int, grid[:, 0])) | set(map(int, grid[:, -1]))
    border_colors.discard(int(bg))
    assert len(border_colors) == 1
    ctrl = int(next(iter(border_colors)))

    pts = np.argwhere(grid == ctrl)
    assert 1 <= len(pts) <= 2
    edges = set()
    for r, c in pts:
        r = int(r)
        c = int(c)
        assert (r in {0, H - 1}) or (c in {0, W - 1})
        assert not ((r in {0, H - 1}) and (c in {0, W - 1}))
        if r == 0:
            edges.add("top")
        elif r == H - 1:
            edges.add("bottom")
        elif c == 0:
            edges.add("left")
        else:
            edges.add("right")
    assert len(edges) == 1
    return ctrl, next(iter(edges))


def _extract_nodes(grid: np.ndarray) -> tuple[int, int, list[_Node]]:
    bg = int(get_bg(grid))
    fg_colors = sorted(set(map(int, grid.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    assert comps

    nodes: list[_Node] = []
    holes: list[list[np.ndarray]] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        node = _Node(
            coords=coords,
            bbox=tuple(map(int, comp["bbox"])),
            center=coords.mean(axis=0),
        )
        node_holes = _component_holes(grid.shape, coords)
        node.is_frame = bool(node_holes)
        nodes.append(node)
        holes.append(node_holes)

    assert any(node.is_frame for node in nodes)
    for node_i, node in enumerate(nodes):
        containers = []
        for frame_i, frame in enumerate(nodes):
            if node_i == frame_i or not frame.is_frame:
                continue
            if any(np.all(hole[tuple(node.coords.T)]) for hole in holes[frame_i]):
                containers.append(frame)
        if containers:
            node.parent = min(containers, key=lambda item: int(item.coords.shape[0]))
            node.parent.children.append(node)
    return bg, fg, [node for node in nodes if node.parent is None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a control-marked rough nesting sketch, canonicalize it with forced vertical stacks, then mirror the finished diagram.

    Input structure:
    - The grid contains one background color, one foreground drawing color, and one control color.
    - Foreground 4-connected components are rough closed frames or small filled blobs.
    - One or two isolated control pixels sit on exactly one non-corner outer edge; left/right edges encode a left-right mirror, and top/bottom edges encode an up-down mirror.

    Rule:
    - Find the control edge first and conceptually erase those control pixels from the scene.
    - Rebuild the containment tree from the remaining foreground components, turning each blob into one pixel and each frame into a clean rectangular border with two cells of inner padding.
    - Inside every frame, ignore the original horizontal/diagonal sibling geometry and stack direct children from top to bottom by original row position with one blank row between them.
    - After the entire canonical tree has been built, mirror the whole canonical mask along the axis indicated by the control edge.

    Key insight:
    The edge marker does not act on the rough drawing directly; it only chooses the final mirror axis after the drawing has first been reduced to the variant's forced-vertical canonical form.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    ctrl, edge = _control_edge(g, bg)

    work = g.copy()
    work[work == ctrl] = bg
    bg, fg, roots = _extract_nodes(work)
    canonical_mask = _layout_root(roots, mode="vertical")
    final_mask = _apply_variant_mirror(canonical_mask, edge)
    out = np.full(final_mask.shape, bg, dtype=int)
    out[final_mask] = fg
    return out


# --------------- generator ---------------

# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6
MIN_SIDE_PAD = 1

# Frame geometry
FRAME_H_RANGE = (7, 20)
FRAME_W_RANGE = (7, 24)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

# Blob geometry
BLOB_SIZE_RANGE = (4, 11)
BLOB_BBOX_RANGE = (2, 5)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (2, 3)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 4)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 3
NON_VERTICAL_MODE_PROB = 0.88

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 5)

# Tree structure
DEPTH1_PROB = 0.45
DEPTH1_INSIDE_RANGE = (2, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (1, 1)

# Control pixels
CONTROL_COUNT_WEIGHTS = (
    (1, 2),
    (2, 3),
)
EDGE_CYCLE_ORDER = ("left", "right", "top", "bottom")
_EDGE_CYCLE_INDEX = 0

# Variant visibility thresholds
MIRROR_DIFF_MIN = 4
CONTROL_EFFECT_DIFF_MIN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 500


def _loop_ok(mask: np.ndarray) -> bool:
    _, n = label_mask(mask, connectivity=4)
    if n != 1:
        return False
    deg = np.zeros_like(mask, dtype=int)
    deg[1:, :] += mask[:-1, :]
    deg[:-1, :] += mask[1:, :]
    deg[:, 1:] += mask[:, :-1]
    deg[:, :-1] += mask[:, 1:]
    if not np.all(deg[mask] == 2):
        return False
    inv_lab, inv_n = label_mask(~mask, connectivity=4)
    boundary = set(inv_lab[0, :]) | set(inv_lab[-1, :]) | set(inv_lab[:, 0]) | set(inv_lab[:, -1])
    return sum(1 for k in range(1, inv_n + 1) if k not in boundary) == 1


def _step_mix(n_a: int, step_a: tuple[int, int], n_b: int, step_b: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    last = None
    while n_a or n_b:
        if n_a and n_b:
            if last is not None and random.random() < STEP_MIX_ALT_PROB:
                pick = "b" if last == "a" else "a"
            else:
                pick = "a" if random.random() < n_a / (n_a + n_b) else "b"
        elif n_a:
            pick = "a"
        else:
            pick = "b"
        if pick == "a":
            out.append(step_a)
            n_a -= 1
        else:
            out.append(step_b)
            n_b -= 1
        last = pick
    return out


def _add_line(path: list[tuple[int, int]], end: tuple[int, int]) -> None:
    r, c = path[-1]
    er, ec = end
    dr = 0 if er == r else (1 if er > r else -1)
    dc = 0 if ec == c else (1 if ec > c else -1)
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))


def _add_steps(path: list[tuple[int, int]], steps: list[tuple[int, int]]) -> None:
    r, c = path[-1]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))


def _turn_ratio(path: list[tuple[int, int]]) -> float:
    dirs = []
    cyc = path + [path[0]]
    for (r0, c0), (r1, c1) in zip(cyc, cyc[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = sum(1 for a, b in zip(dirs, dirs[1:] + [dirs[0]]) if a != b)
    return turns / len(path)


def _frame_mask(h: int, w: int) -> np.ndarray | None:
    max_dx = min(FRAME_CORNER_MAX, max(1, (w - 5) // 2))
    max_dy = min(FRAME_CORNER_MAX, max(1, (h - 5) // 2))
    lo, hi = FRAME_TURN_RATIO
    for _ in range(FRAME_RETRIES):
        tl_dx = random.randint(1, max_dx)
        tr_dx = random.randint(1, max_dx)
        br_dx = random.randint(1, max_dx)
        bl_dx = random.randint(1, max_dx)
        tl_dy = random.randint(1, max_dy)
        tr_dy = random.randint(1, max_dy)
        br_dy = random.randint(1, max_dy)
        bl_dy = random.randint(1, max_dy)

        p0 = (0, tl_dx)
        p1 = (0, w - 1 - tr_dx)
        p2 = (tr_dy, w - 1)
        p3 = (h - 1 - br_dy, w - 1)
        p4 = (h - 1, w - 1 - br_dx)
        p5 = (h - 1, bl_dx)
        p6 = (h - 1 - bl_dy, 0)
        p7 = (tl_dy, 0)
        if not (
            p1[1] - p0[1] >= FRAME_MIN_EDGE
            and p3[0] - p2[0] >= FRAME_MIN_EDGE
            and p4[1] - p5[1] >= FRAME_MIN_EDGE
            and p6[0] - p7[0] >= FRAME_MIN_EDGE
        ):
            continue

        path = [p0]
        _add_line(path, p1)
        _add_steps(path, _step_mix(tr_dy, (1, 0), tr_dx, (0, 1)))
        _add_line(path, p3)
        _add_steps(path, _step_mix(br_dy, (1, 0), br_dx, (0, -1)))
        _add_line(path, p5)
        _add_steps(path, _step_mix(bl_dy, (-1, 0), bl_dx, (0, -1)))
        _add_line(path, p7)
        _add_steps(path, _step_mix(tl_dy, (-1, 0), tl_dx, (0, 1)))
        if path[-1] == path[0]:
            path.pop()

        coords = np.array(path, dtype=int)
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        ratio = _turn_ratio(path)
        if lo <= ratio <= hi and _loop_ok(mask):
            return mask
    return None


def _blob_mask() -> np.ndarray:
    size = random.randint(*BLOB_SIZE_RANGE)
    feasible = [
        (h, w)
        for h in range(*BLOB_BBOX_RANGE)
        for w in range(*BLOB_BBOX_RANGE)
        if h * w >= size and h + w - 1 <= size
    ]
    h, w = random.choice(feasible)
    rr = random.randrange(h)
    cc = random.randrange(w)
    cells = {(r, cc) for r in range(h)} | {(rr, c) for c in range(w)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if 0 <= p[0] < h and 0 <= p[1] < w and p not in cells:
                    frontier.append(p)
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    pts = np.array(list(cells), dtype=int)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _arrange(children: list[_Obj], compact: bool = False) -> tuple[dict[_Obj, tuple[int, int]], int, int, str]:
    random.shuffle(children)
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1], "single"

    if random.random() < NON_VERTICAL_MODE_PROB:
        mode = random.choice(("horizontal", "diagonal"))
    else:
        mode = "vertical"

    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
    placements = {}

    if mode == "vertical":
        group_h = sum(h for h, _ in dims) + sum(gaps)
        group_w = max(w for _, w in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        r = 0
        for child, (h, w), gap in zip(children, dims, gaps + [0]):
            placements[child] = (r, random.randint(0, group_w - w))
            r += h + gap
        return placements, group_h, group_w, mode

    if mode == "horizontal":
        max_h = max(h for h, _ in dims)
        extra = max(random.randint(*ARRANGE_EXTRA_SPAN), k - 1)
        row_slots = sorted(random.sample(range(extra + 1), k))
        group_h = max_h + extra
        group_w = sum(w for _, w in dims) + sum(gaps)
        c = 0
        for child, (h, w), gap, slot in zip(children, dims, gaps + [0], row_slots):
            placements[child] = (slot, c)
            c += w + gap
        return placements, group_h, group_w, mode

    group_h = sum(h for h, _ in dims) + sum(gaps)
    group_w = sum(w for _, w in dims) + sum(gaps)
    r = c = 0
    for child, (h, w), gap in zip(children, dims, gaps + [0]):
        placements[child] = (r, c)
        r += h + gap
        c += w + gap
    return placements, group_h, group_w, mode


def _prepare(node: _Obj, compact: bool = False) -> bool:
    if node.kind == "blob":
        node.mask = _blob_mask()
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    placements, group_h, group_w, mode = _arrange(node.children, compact=tight)
    node.layout_mode = mode
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False
    h_hi = min(FRAME_H_RANGE[1], min_h + 2) if compact else FRAME_H_RANGE[1]
    w_hi = min(FRAME_W_RANGE[1], min_w + 3) if compact else FRAME_W_RANGE[1]
    h = random.randint(min_h, h_hi)
    w = random.randint(min_w, w_hi)
    top = margin + random.randint(0, h - 2 * margin - group_h)
    left = margin + random.randint(0, w - 2 * margin - group_w)
    for child, (r, c) in placements.items():
        child.rel = (top + r, left + c)
    node.mask = _frame_mask(h, w)
    return node.mask is not None


def _walk(node: _Obj, base=(0, 0)) -> list[_Obj]:
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _place_outside(root: _Obj, outside: _Obj) -> None:
    assert root.mask is not None and outside.mask is not None
    h, w = root.mask.shape
    bh, bw = outside.mask.shape
    gap = random.randint(*OUTSIDE_GAP_RANGE)
    side = outside.outside_side
    if side == "right":
        outside.rel = (random.randint(1, h - bh - 1), w + gap)
    elif side == "left":
        outside.rel = (random.randint(1, h - bh - 1), -gap - bw)
    elif side == "bottom":
        outside.rel = (h + gap, random.randint(1, w - bw - 1))
    else:
        outside.rel = (-gap - bh, random.randint(1, w - bw - 1))


def _node_from_obj(obj: _Obj) -> _Node:
    assert obj.mask is not None
    r0, c0 = obj.pos
    coords = np.argwhere(obj.mask) + np.array([r0, c0])
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    node = _Node(
        coords=coords.astype(int),
        bbox=(int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])),
        center=coords.mean(axis=0),
    )
    node.is_frame = obj.kind == "frame"
    return node


def _scene_to_roots(root: _Obj, outside: _Obj | None) -> list[_Node]:
    def add(obj: _Obj) -> _Node:
        node = _node_from_obj(obj)
        for child in obj.children:
            child_node = add(child)
            child_node.parent = node
            node.children.append(child_node)
        return node

    roots = [add(root)]
    if outside is not None:
        roots.append(add(outside))
    return roots


def _canonical_mask_from_scene(root: _Obj, outside: _Obj | None, mode: str) -> np.ndarray:
    roots = _scene_to_roots(root, outside)
    return _layout_root(roots, mode=mode)


def _expected_output(root: _Obj, outside: _Obj | None, bg: int, fg: int, edge: str) -> np.ndarray:
    canonical_mask = _canonical_mask_from_scene(root, outside, mode="vertical")
    final_mask = _apply_variant_mirror(canonical_mask, edge)
    out = np.full(final_mask.shape, bg, dtype=int)
    out[final_mask] = fg
    return out


def _weighted_choice(weighted_items):
    items = [item for item, _weight in weighted_items]
    weights = [weight for _item, weight in weighted_items]
    return random.choices(items, weights=weights, k=1)[0]


def _next_target_edge() -> str:
    global _EDGE_CYCLE_INDEX
    edge = EDGE_CYCLE_ORDER[_EDGE_CYCLE_INDEX % len(EDGE_CYCLE_ORDER)]
    _EDGE_CYCLE_INDEX += 1
    return edge


def _control_positions(shape: tuple[int, int], edge: str, count: int) -> list[tuple[int, int]]:
    H, W = shape
    if edge in {"top", "bottom"}:
        coords = [(0 if edge == "top" else H - 1, c) for c in range(1, W - 1)]
    else:
        coords = [(r, 0 if edge == "left" else W - 1) for r in range(1, H - 1)]
    assert len(coords) >= count
    if count == 1:
        return [random.choice(coords)]
    random.shuffle(coords)
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 2:
                return [a, b]
    return coords[:2]


def _add_controls(grid: np.ndarray, bg: int, fg: int, edge: str) -> np.ndarray:
    ctrl = random.choice([c for c in range(10) if c not in {bg, fg}])
    count = _weighted_choice(CONTROL_COUNT_WEIGHTS)
    pts = _control_positions(grid.shape, edge, count)
    out = grid.copy()
    for r, c in pts:
        out[r, c] = ctrl
    return out


def _draw(objects: list[_Obj], bg: int, fg: int) -> np.ndarray | None:
    min_r = min(obj.pos[0] for obj in objects)
    min_c = min(obj.pos[1] for obj in objects)
    max_r = max(obj.pos[0] + obj.mask.shape[0] for obj in objects)
    max_c = max(obj.pos[1] + obj.mask.shape[1] for obj in objects)
    draw_h = max_r - min_r
    draw_w = max_c - min_c
    if draw_h > GRID_MAX - 2 * MIN_SIDE_PAD or draw_w > GRID_MAX - 2 * MIN_SIDE_PAD:
        return None

    extra_h_min = max(2 * MIN_SIDE_PAD, GRID_MIN - draw_h)
    extra_w_min = max(2 * MIN_SIDE_PAD, GRID_MIN - draw_w)
    extra_h_max = min(MAX_EXTRA_PAD, GRID_MAX - draw_h)
    extra_w_max = min(MAX_EXTRA_PAD, GRID_MAX - draw_w)
    if extra_h_min > extra_h_max or extra_w_min > extra_w_max:
        return None

    extra_h = random.randint(extra_h_min, extra_h_max)
    extra_w = random.randint(extra_w_min, extra_w_max)
    top_lo = max(MIN_SIDE_PAD, extra_h - MAX_PAD_PER_SIDE)
    top_hi = min(MAX_PAD_PER_SIDE, extra_h - MIN_SIDE_PAD)
    left_lo = max(MIN_SIDE_PAD, extra_w - MAX_PAD_PER_SIDE)
    left_hi = min(MAX_PAD_PER_SIDE, extra_w - MIN_SIDE_PAD)
    if top_lo > top_hi or left_lo > left_hi:
        return None

    top_pad = random.randint(top_lo, top_hi)
    left_pad = random.randint(left_lo, left_hi)
    shift_r = top_pad - min_r
    shift_c = left_pad - min_c

    grid = np.full((draw_h + extra_h, draw_w + extra_w), bg, dtype=int)
    for obj in objects:
        assert obj.mask is not None
        r = obj.pos[0] + shift_r
        c = obj.pos[1] + shift_c
        obj.pos = (r, c)
        patch = np.full(obj.mask.shape, bg, dtype=int)
        patch[obj.mask] = fg
        paste(grid, patch, r, c, transparent=bg)

    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == fg):
        return None
    assert not np.any(grid[0, :] == fg)
    assert not np.any(grid[-1, :] == fg)
    assert not np.any(grid[:, 0] == fg)
    assert not np.any(grid[:, -1] == fg)
    return grid


def _make_tree(target_edge: str) -> tuple[_Obj, _Obj]:
    root = _Obj("frame")
    outside = _Obj("blob", outside_side=target_edge)

    if random.random() < DEPTH1_PROB:
        n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
        root.children = [_Obj("blob") for _ in range(n_inside)]
    else:
        inner = _Obj("frame")
        inner.children = [_Obj("blob") for _ in range(random.randint(*DEPTH2_INNER_BLOBS))]
        outer_blobs = [_Obj("blob") for _ in range(random.randint(*DEPTH2_OUTER_BLOBS))]
        root.children = [inner] + outer_blobs
    return root, outside


def _mirror_diff(mask: np.ndarray, edge: str) -> int:
    return int(np.count_nonzero(mask != _apply_variant_mirror(mask, edge)))


def _mask_difference_score(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        return int(a.size + b.size)
    return int(np.count_nonzero(a != b))


def _control_effect_diff(mask: np.ndarray, edge: str) -> int:
    axis = _mirror_axis_from_edge(edge)
    other_axis = "vertical" if axis == "horizontal" else "horizontal"
    other_edge = _AXIS_TO_EDGES[other_axis][0]
    return int(np.count_nonzero(_apply_variant_mirror(mask, edge) != _apply_variant_mirror(mask, other_edge)))


def _collect_layout_meta(node: _Obj):
    out = []
    if node.kind == "frame":
        out.append((len(node.children), node.layout_mode))
        for child in node.children:
            out.extend(_collect_layout_meta(child))
    return out


def _gen_attempt(target_edge: str | None = None, with_meta: bool = False):
    bg, fg = random.sample(range(10), 2)
    edge = _next_target_edge() if target_edge is None else target_edge
    target_axis = _mirror_axis_from_edge(edge)

    root, outside = _make_tree(edge)
    if not _prepare(root):
        return None
    if not _prepare(outside):
        return None
    _place_outside(root, outside)

    objects = _walk(root)
    outside.pos = outside.rel
    objects.append(outside)

    grid = _draw(objects, bg, fg)
    if grid is None:
        return None

    canonical_vertical = _canonical_mask_from_scene(root, outside, mode="vertical")
    canonical_legacy = _canonical_mask_from_scene(root, outside, mode="legacy")
    mirror_diff = _mirror_diff(canonical_vertical, edge)
    control_diff = _control_effect_diff(canonical_vertical, edge)
    verticalization_diff = _mask_difference_score(canonical_vertical, canonical_legacy)
    if verticalization_diff == 0:
        return None
    if mirror_diff < MIRROR_DIFF_MIN:
        return None
    if control_diff < CONTROL_EFFECT_DIFF_MIN:
        return None

    expected = _expected_output(root, outside, bg, fg, edge)
    grid = _add_controls(grid, bg, fg, edge)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None

    if with_meta:
        return grid, {
            "edge": edge,
            "axis": target_axis,
            "mirror_diff": mirror_diff,
            "control_diff": control_diff,
            "verticalization_diff": verticalization_diff,
            "layouts": _collect_layout_meta(root),
            "outside_side": outside.outside_side,
        }
    return grid


def generate_input():
    edge = _next_target_edge()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(target_edge=edge)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
