# Combo variant of 2d0172a1 (index 6)
# combo_spec: D7/2+D15/0
# Name: 行列顺序取反+边缘控制像素指示方向

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import find_components, get_bg, label_mask, paste


"""
design_analysis:
  concretization_and_trace:
    original_pipeline: >
      原题把唯一前景色的粗糙 frame/blob 连通块恢复成包含树，再按输入中的行带/列带顺序
      生成规范矩形框与像素点，最后直接输出该规范图。
    fused_pipeline: >
      融合后输入多一种控制色，并且这 1-2 个控制像素只出现在同一条外边上。solver 必须先
      读出控制边、擦除控制像素、恢复主体包含树，然后按“反向兄弟排序”生成基础规范图，
      最后根据控制边把整个规范图旋转到对应朝向。
    traced_example: >
      例：外框内有“上方一个 inner frame、下方右侧一个 blob”，外框右边还有一个 outside blob。
      原题输出会保持 inner frame 在上、blob 在下，outside blob 在右。D7 单变体会先把内层
      sibling band 顺序镜像，所以 blob 跑到上面、inner frame 到下面。再叠加 D15 控制后，
      若控制像素在左边，则这个“已镜像”的基础规范图再逆时针旋转 90 度，outside blob 也随之
      转到下侧。关键不是两条规则独立并列，而是必须先读控制、再用去掉控制后的主体做反向规范化、
      再整体定向。
  complexity_calibration:
    chosen_discreteness: >
      控制编码仍只用四个边位档位 top/left/bottom/right，避免引入颜色表或多步状态机。
      排序规则也只做单一的全局反向，不额外改变 band 分组、间距或 containment 规则。
    anti_simplification: >
      仅靠“多个 blob”时，反向排序常会因为对象同形而不显眼；仅靠旋转时，若基础图本身有旋转对称，
      控制边效果也会退化。generator 因此同时拒绝两类样本：
      1) reverse_bands=True 与 False 的基础规范图相同；
      2) 当前控制边旋转后的结果与 top 默认朝向相同。
      同时把树分布偏向“inner frame + blob siblings”的可见案例。
  implicit_assumption_audit:
    - >
      原 `main()` 假设网格里只有背景色和唯一前景色；融合后不成立，所以必须先从边界识别唯一控制色，
      再把该色视为背景抹掉，之后才能恢复“唯一主体前景色”的假设。
    - >
      原 `find_components()` 默认所有非背景像素都属于主体；融合后若不先剥离控制色，它会被误识别成
      blob 根节点，破坏 containment tree。
    - >
      原 `_axis_coordinates()` 默认 band 的分配顺序就是从小坐标到大坐标；融合后 band 检测逻辑仍成立，
      但 band 写入 canonical 坐标时必须整体反向，而且这个参数要递归传入所有 frame。
    - >
      原 `_layout_root()` 直接给出最终图；融合后它只负责生成“反向排序后的基础规范图”，最终输出还要
      经过 `_rotate_mask()`。这要求 `_map_frame_axis()` 继续读取已经反向后的 frame.axis_map。
    - >
      `_control_edge()` 隐含“主体前景不触碰外边界”，否则边界非背景色不再唯一归属于控制色。
      generator 需强制四侧至少留 1 格背景。
  placement_legality:
    - >
      主体绘制时保留 `MIN_SIDE_PAD=1`，因此 frame/blob 不会压到边界，也不会和控制像素重叠。
    - >
      控制像素只放在非角落边格；若有两个，它们位于同一条边且彼此不相邻，避免连通性和“编码边”
      解释歧义。
    - >
      保持原题的 output-first 验证：先生成输入，再调用新 solver 回算，只接受完全一致样本。
  visual_observability:
    - >
      人类会同时看到两个稳定线索：输入边缘多出 1-2 个孤立控制点；输出里的 sibling 相对上下/左右
      次序与原题风格相反。
    - >
      由于旋转作用在“已反向”的基础规范图上，少量样例中可以观察到：同样是粗糙包含树，控制点换边后
      最终外侧 blob 方位与内部 sibling 方向会一起变化。
  visual_cleanliness:
    - >
      额外标记只有边缘 1-2 像素，不和主体叠加；主体仍然是单色粗框与小块，布局层次清楚。
    - >
      非退化筛选减少了纯对称或纯同形案例，避免输出看起来“好像没变”。
  rule_inference:
    - >
      变体不依赖颜色编号大小、随机种子或不可见元数据。控制语义完全由“同色孤立边缘像素的位置”
      给出，排序反向完全由对象的可见相对位置给出，解题者可从 2-4 组样例归纳。
"""


_EDGE_TO_ROTATION = {
    "top": 0,
    "left": 1,
    "bottom": 2,
    "right": 3,
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


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _build_layout(node: _Node, reverse_bands: bool) -> None:
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
    child_mask, _placements, axis_map = _layout_group(node.children, reverse_bands=reverse_bands)
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


def _axis_coordinates(nodes: list[_Node], axis: int, reverse_bands: bool) -> dict[_Node, int]:
    items = []
    for node in nodes:
        assert node.mask is not None and node.anchor is not None
        r0, c0, r1, c1 = node.bbox
        interval = (r0, r1) if axis == 0 else (c0, c1)
        mid = (interval[0] + interval[1]) / 2
        items.append((node, interval, mid, node.anchor[axis], node.mask.shape[axis]))

    bands: list[dict[str, object]] = []
    for item in sorted(items, key=lambda x: x[2]):
        interval = item[1]
        for band in bands:
            band_items = band["items"]
            assert isinstance(band_items, list)
            if any(_intervals_overlap(interval, other[1]) for other in band_items):
                band_items.append(item)
                band_interval = band["interval"]
                assert isinstance(band_interval, tuple)
                band["interval"] = (
                    min(band_interval[0], interval[0]),
                    max(band_interval[1], interval[1]),
                )
                break
        else:
            bands.append({"items": [item], "interval": interval})

    ordered_bands = list(reversed(bands)) if reverse_bands else bands
    coord_by_node: dict[_Node, int] = {}
    coord = 0
    for i, band in enumerate(ordered_bands):
        band_items = band["items"]
        assert isinstance(band_items, list)
        if i:
            prev_items = ordered_bands[i - 1]["items"]
            assert isinstance(prev_items, list)
            prev_down = max(item[4] - 1 - item[3] for item in prev_items)
            curr_up = max(item[3] for item in band_items)
            coord += prev_down + 2 + curr_up
        for item in band_items:
            coord_by_node[item[0]] = int(coord)

    return coord_by_node


def _layout_group(
    nodes: list[_Node], reverse_bands: bool
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node, reverse_bands=reverse_bands)

    row_coord = _axis_coordinates(nodes, axis=0, reverse_bands=reverse_bands)
    col_coord = _axis_coordinates(nodes, axis=1, reverse_bands=reverse_bands)

    min_r = min(row_coord[n] - n.anchor[0] for n in nodes if n.anchor is not None)
    min_c = min(col_coord[n] - n.anchor[1] for n in nodes if n.anchor is not None)
    max_r = max(
        row_coord[n] - n.anchor[0] - min_r + n.mask.shape[0]
        for n in nodes
        if n.anchor is not None and n.mask is not None
    )
    max_c = max(
        col_coord[n] - n.anchor[1] - min_c + n.mask.shape[1]
        for n in nodes
        if n.anchor is not None and n.mask is not None
    )

    out = np.zeros((max_r, max_c), dtype=bool)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}
    for node in nodes:
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


def _layout_root(nodes: list[_Node], reverse_bands: bool) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node, reverse_bands=reverse_bands)

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

    mask, _placements, _axis_map = _layout_group(nodes, reverse_bands=reverse_bands)
    return mask


def _rotate_mask(mask: np.ndarray, edge: str) -> np.ndarray:
    return np.rot90(mask, k=_EDGE_TO_ROTATION[edge])


def _control_edge(grid: np.ndarray, bg: int) -> tuple[int, str]:
    h, w = grid.shape
    border_colors = set(map(int, grid[0, :])) | set(map(int, grid[-1, :]))
    border_colors |= set(map(int, grid[:, 0])) | set(map(int, grid[:, -1]))
    border_colors.discard(int(bg))
    assert len(border_colors) == 1
    ctrl = int(next(iter(border_colors)))

    pts = np.argwhere(grid == ctrl)
    assert 1 <= len(pts) <= 2
    edges = set()
    for r, c in pts:
        r = int(r)
        c = int(c)
        assert (r in {0, h - 1}) or (c in {0, w - 1})
        assert not ((r in {0, h - 1}) and (c in {0, w - 1}))
        if r == 0:
            edges.add("top")
        elif r == h - 1:
            edges.add("bottom")
        elif c == 0:
            edges.add("left")
        else:
            edges.add("right")
    assert len(edges) == 1
    return ctrl, next(iter(edges))


def _extract_nodes(grid: np.ndarray, bg: int, fg: int) -> list[_Node]:
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

    return [node for node in nodes if node.parent is None]


def _canonical_mask(grid: np.ndarray, bg: int, fg: int, reverse_bands: bool) -> np.ndarray:
    roots = _extract_nodes(grid, bg=bg, fg=fg)
    return _layout_root(roots, reverse_bands=reverse_bands)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a rough nested frame sketch into a clean frame-and-dot diagram whose sibling order is mirrored and whose final facing is set by border controls.

    Input structure:
    - The grid has one background color, one foreground drawing color, and one control color.
    - Foreground 4-connected components are rough closed frames with one hole or small solid blobs.
    - One or two isolated control pixels lie on the same outer edge and are not part of the drawing.

    Rule:
    - Read the control edge first, erase those control pixels conceptually, and recover the containment tree from the remaining foreground components.
    - Replace each blob by one foreground pixel and each frame by a 1-pixel rectangular border padded around its direct children.
    - Infer sibling row-bands and col-bands exactly as in the base task, but assign canonical band positions in the opposite direction: lower bands move upward and rightward bands move leftward.
    - Rotate that already-mirrored canonical result according to the control edge: top/left/bottom/right mean default/90CCW/180/90CW.

    Key insight:
    - The rough stroke geometry still only tells you the enclosure tree and sibling band structure; the edge marker chooses the final orientation, while the canonical sibling order itself is simultaneously reversed.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    ctrl, edge = _control_edge(g, bg)

    work = g.copy()
    work[work == ctrl] = bg
    fg_colors = sorted(set(map(int, work.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    mask = _rotate_mask(_canonical_mask(work, bg=bg, fg=fg, reverse_bands=True), edge)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
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

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 5)

# Tree structure probabilities
DEPTH1_PROB = 0.15
OUTSIDE_BLOB_PROB = 0.35
DEPTH1_INSIDE_RANGE = (2, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (1, 2)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Control pixels
CONTROL_EDGE_WEIGHTS = (
    ("top", 1),
    ("left", 1),
    ("bottom", 1),
    ("right", 1),
)
CONTROL_COUNT_WEIGHTS = (
    (1, 2),
    (2, 3),
)

# Generation retry budget
GEN_RETRIES = 500


class _Obj:
    def __init__(self, kind: str, children=None, outside_side: str | None = None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


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


def _arrange(children: list[_Obj], compact: bool = False) -> tuple[dict[_Obj, tuple[int, int]], int, int]:
    random.shuffle(children)
    dims = [child.mask.shape for child in children if child.mask is not None]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    mode = random.choice(("vertical", "horizontal", "diagonal"))
    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
    placements: dict[_Obj, tuple[int, int]] = {}

    if mode == "vertical":
        group_h = sum(h for h, _ in dims) + sum(gaps)
        group_w = max(w for _, w in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        r = 0
        for child, (h, w), gap in zip(children, dims, gaps + [0]):
            placements[child] = (r, random.randint(0, group_w - w))
            r += h + gap
        return placements, group_h, group_w

    if mode == "horizontal":
        group_h = max(h for h, _ in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        group_w = sum(w for _, w in dims) + sum(gaps)
        c = 0
        for child, (h, w), gap in zip(children, dims, gaps + [0]):
            placements[child] = (random.randint(0, group_h - h), c)
            c += w + gap
        return placements, group_h, group_w

    group_h = sum(h for h, _ in dims) + sum(gaps)
    group_w = sum(w for _, w in dims) + sum(gaps)
    r = 0
    c = 0
    for child, (h, w), gap in zip(children, dims, gaps + [0]):
        placements[child] = (r, c)
        r += h + gap
        c += w + gap
    return placements, group_h, group_w


def _prepare(node: _Obj, compact: bool = False) -> bool:
    if node.kind == "blob":
        node.mask = _blob_mask()
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    placements, group_h, group_w = _arrange(node.children, compact=tight)
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False
    h_hi = min(FRAME_H_RANGE[1], min_h + 2) if tight else FRAME_H_RANGE[1]
    w_hi = min(FRAME_W_RANGE[1], min_w + 3) if tight else FRAME_W_RANGE[1]
    h = random.randint(min_h, h_hi)
    w = random.randint(min_w, w_hi)
    top = margin + random.randint(0, h - 2 * margin - group_h)
    left = margin + random.randint(0, w - 2 * margin - group_w)
    for child, (r, c) in placements.items():
        child.rel = (top + r, left + c)
    node.mask = _frame_mask(h, w)
    return node.mask is not None


def _walk(node: _Obj, base: tuple[int, int] = (0, 0)) -> list[_Obj]:
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


def _expected_output(root: _Obj, outside: _Obj | None, bg: int, fg: int, edge: str, reverse_bands: bool) -> np.ndarray:
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
    mask = _rotate_mask(_layout_root(roots, reverse_bands=reverse_bands), edge)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _make_tree() -> tuple[_Obj, _Obj | None]:
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")
    if random.random() < DEPTH1_PROB:
        n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
        root.children = [_Obj("blob") for _ in range(n_inside)]
        if n_inside < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
            outside = _Obj("blob", outside_side=random.choice(sides))
    else:
        inner = _Obj("frame")
        inner.children = [_Obj("blob") for _ in range(random.randint(*DEPTH2_INNER_BLOBS))]
        outer_blobs = [_Obj("blob") for _ in range(random.randint(*DEPTH2_OUTER_BLOBS))]
        root.children = [inner] + outer_blobs
        if len(inner.children) + len(outer_blobs) < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
            outside = _Obj("blob", outside_side=random.choice(sides))
    return root, outside


def _draw(objects: list[_Obj], bg: int, fg: int) -> np.ndarray | None:
    min_r = min(obj.pos[0] for obj in objects)
    min_c = min(obj.pos[1] for obj in objects)
    max_r = max(obj.pos[0] + obj.mask.shape[0] for obj in objects if obj.mask is not None)
    max_c = max(obj.pos[1] + obj.mask.shape[1] for obj in objects if obj.mask is not None)
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


def _weighted_choice(weighted_items: tuple[tuple[object, int], ...] | list[tuple[object, int]]):
    items = [item for item, _weight in weighted_items]
    weights = [weight for _item, weight in weighted_items]
    return random.choices(items, weights=weights, k=1)[0]


def _weighted_choice_subset(weighted_items, allowed_items):
    filtered = [(item, weight) for item, weight in weighted_items if item in allowed_items]
    assert filtered
    return _weighted_choice(filtered)


def _control_positions(shape: tuple[int, int], edge: str, count: int) -> list[tuple[int, int]]:
    h, w = shape
    if edge in {"top", "bottom"}:
        coords = [(0 if edge == "top" else h - 1, c) for c in range(1, w - 1)]
    else:
        coords = [(r, 0 if edge == "left" else w - 1) for r in range(1, h - 1)]
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


def _gen_attempt() -> np.ndarray | None:
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree()
    if not _prepare(root):
        return None
    if outside is not None:
        if not _prepare(outside):
            return None
        _place_outside(root, outside)

    objects = _walk(root)
    if outside is not None:
        outside.pos = outside.rel
        objects.append(outside)

    grid = _draw(objects, bg, fg)
    if grid is None:
        return None

    reverse_top = _expected_output(root, outside, bg, fg, edge="top", reverse_bands=True)
    forward_top = _expected_output(root, outside, bg, fg, edge="top", reverse_bands=False)
    if np.array_equal(reverse_top, forward_top):
        return None

    expected_by_edge = {}
    candidate_edges = []
    for edge in _EDGE_TO_ROTATION:
        expected = _expected_output(root, outside, bg, fg, edge=edge, reverse_bands=True)
        expected_by_edge[edge] = expected
        if edge == "top" or not np.array_equal(expected, reverse_top):
            candidate_edges.append(edge)

    edge = _weighted_choice_subset(CONTROL_EDGE_WEIGHTS, candidate_edges)
    expected = expected_by_edge[edge]

    grid = _add_controls(grid, bg, fg, edge)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo D7/2+D15/0 input")
