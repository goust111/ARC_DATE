# Combo variant of 2d0172a1 (index 11)
# combo_spec: D8/2+D15/0
# Name: 外部色块被删除+边缘控制像素

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


"""
design_analysis:
  concrete_trace:
    - >
      例：输入里有一个粗糙外框，框内有两个 blob，框外右侧还有一个 blob；同时顶边或右边
      放 1-2 个控制像素。原题会把外部 blob 也规范化到输出外侧 3x3 槽位，且输出朝向固定。
    - >
      本融合版先读边缘控制像素，确定 top/left/bottom/right 对应
      默认/90CCW/180/90CW；随后把控制色从工作网格中擦除，再恢复框架-blob 包含树。
    - >
      在根层布局时，框外 blob 不再进入输出；也就是说“边上的非框内对象”里，控制像素直接忽略，
      外部 blob 虽然保留在输入里帮助制造干扰，但规范化时也被删除。最后只把框内结构旋转到控制边指定的朝向。
  complexity_calibration:
    - >
      控制指令只用四种边位档位，不再叠加颜色表、图例或计数编码，保证人类能在少量样例里稳定看出
      “哪条边决定朝向”。
    - >
      外部 blob 仍限制为 0-1 个，避免删除规则分支过多；但其出现概率提高，让“框外=删掉”在训练样例中可见。
    - >
      generator 会拒绝两类退化：一是旋转后与默认朝向完全相同，二是删除外部 blob 前后输出完全相同。
  implicit_assumption_audit:
    - >
      原 main() 假设全网格只有一个非背景色；加入控制像素后不成立，所以必须先从边界识别唯一控制色并擦除。
    - >
      原 main() 假设所有非背景连通块都属于 frame/blob；加入控制像素后不成立，否则控制点会被误当作 blob 根节点。
    - >
      原 _layout_root() 假设根层 dot 要么作为外部 blob 画出来，要么与 frame 一起布局；删除变体打破这个前提，
      因此根层规范化必须只保留 frame 根。
    - >
      控制像素与外部 blob 都处在边缘区域附近，solver 新增的关键区分前提是：
      控制像素必须恰好落在最外边界，而主体前景绝不能碰边；generator 会显式保证这一点。
  placement_legality:
    - >
      输入主体四侧至少保留 1 格背景边距，确保边界上的唯一非背景色来自控制像素，而不是 frame 或 blob。
    - >
      控制像素只放在非角落边格，若有两个则放在同一条边且中间至少隔 1 格背景，避免被看成小条形对象。
    - >
      外部 blob 仍按原题合法地放在最外框一侧，和框保持 2-5 格间距，不重叠、不越界；只是输出阶段删除。
  visual_observability:
    - >
      人类可见的两个稳定信号是：输入边缘出现同色孤立控制像素，且输出整体朝向随控制边改变；
      同时输入里偶尔出现框外 blob，但输出里消失。
    - >
      generator 轮转采样 top/left/bottom/right 四条控制边，并继续拒绝旋转不改变输出的对称样本，
      从而兼顾朝向线索可见性与分支覆盖率。
    - >
      为了让删除差异稳定可见，若生成了外部 blob，则会比较“删掉它”和“保留它”的输出，要求两者不同。
  visual_cleanliness:
    - >
      控制像素与主体分区明确：控制像素只在边界，主体总在内层；颜色数最多 3 个，图面不会因融合而拥挤混乱。
    - >
      删除规则不在输出中留下额外占位，因此输出依然是干净的矩形框+点阵结构。
  color_inference:
    - >
      颜色本身不编码方向，真正编码的是控制像素所在边；控制色只需作为“边界上那一类孤立像素”被识别。
      输出不保留控制色，也不依赖颜色 id 大小关系。
  new_main_assumptions_and_generator_support:
    - >
      _control_edge() 假设边界上恰有一种非背景色且只出现在一条边上；_draw() 和 _add_controls()
      共同保证这一点。
    - >
      场景解析假设擦除控制色后只剩一种前景色；generator 从 0-9 中为控制色单独采样，排除 bg/fg。
    - >
      删除逻辑假设至少存在一个 frame 根；树生成器始终从 root frame 开始，不会产生“只有 blob 没有 frame”的输入。
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


def _build_layout(node: _Node) -> None:
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
    child_mask, _placements, axis_map = _layout_group(node.children)
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


def _axis_coordinates(nodes: list[_Node], axis: int) -> dict[_Node, int]:
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


def _layout_group(nodes: list[_Node]) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node)

    row_coord = _axis_coordinates(nodes, axis=0)
    col_coord = _axis_coordinates(nodes, axis=1)

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


def _legacy_layout_root(nodes: list[_Node]) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node)

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

    mask, _placements, _axis_map = _layout_group(nodes)
    return mask


def _layout_root(nodes: list[_Node]) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node)

    kept = [node for node in nodes if node.is_frame]
    assert kept
    mask, _placements, _axis_map = _layout_group(kept)
    return mask


def _rotate_mask(mask: np.ndarray, edge: str) -> np.ndarray:
    return np.rot90(mask, k=_EDGE_TO_ROTATION[edge])


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


def _parse_scene(work: np.ndarray, bg: int, fg: int) -> list[_Node]:
    comps = find_components(work, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
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
        node_holes = _component_holes(work.shape, coords)
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
    return nodes


def _canonical_mask_from_grid(grid: np.ndarray, include_outside: bool) -> tuple[np.ndarray, int, int, str]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    ctrl, edge = _control_edge(g, bg)

    work = g.copy()
    work[work == ctrl] = bg
    fg_colors = sorted(set(map(int, work.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    nodes = _parse_scene(work, bg, fg)
    roots = [node for node in nodes if node.parent is None]
    base = _legacy_layout_root(roots) if include_outside else _layout_root(roots)
    return _rotate_mask(base, edge), bg, fg, edge


def _solve_output(grid: np.ndarray, include_outside: bool) -> np.ndarray:
    mask, bg, fg, _edge = _canonical_mask_from_grid(grid, include_outside=include_outside)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize a rough nested frame drawing, discard outer noise blobs, and orient the result by border markers.

    Input structure:
    - The grid contains one background color, one drawing color, and one control color.
    - Drawing-color components are rough closed frames or small filled blobs; one blob may also sit outside every frame.
    - One or two isolated control pixels lie on a single outer edge and are separate from the drawing.

    Rule:
    - Read the control edge first and erase those control pixels conceptually before parsing objects.
    - Recover the containment tree from frame holes, reduce every enclosed blob to one pixel, and turn every frame into a rectangular border padded around its direct children.
    - Ignore blobs that are outside all frames instead of allocating them an outer slot.
    - Rotate the cleaned canonical result according to the control edge: top/left/bottom/right map to default/90CCW/180/90CW.

    Key insight: Two different edge-region distractions must be filtered differently: boundary pixels are instructions that choose orientation, while non-enclosed blobs are scene noise that disappears from the canonical output.
    """
    return _solve_output(grid, include_outside=False)


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
DEPTH1_PROB = 0.35
OUTSIDE_BLOB_PROB = 0.72
DEPTH1_INSIDE_RANGE = (1, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (0, 2)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Control pixels
CONTROL_COUNT_WEIGHTS = (
    (1, 2),
    (2, 3),
)
KEEP_NO_OUTSIDE_PROB = 0.25
ASYMMETRIC_OUTSIDE_PROB = 0.7

_EDGE_ORDER = ("top", "left", "bottom", "right")
_EDGE_CYCLE_INDEX = 0

# Generation retry budget
GEN_RETRIES = 450


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


def _loop_ok(mask):
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


def _step_mix(n_a, step_a, n_b, step_b):
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


def _add_line(path, end):
    r, c = path[-1]
    er, ec = end
    dr = 0 if er == r else (1 if er > r else -1)
    dc = 0 if ec == c else (1 if ec > c else -1)
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))


def _add_steps(path, steps):
    r, c = path[-1]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))


def _turn_ratio(path):
    dirs = []
    cyc = path + [path[0]]
    for (r0, c0), (r1, c1) in zip(cyc, cyc[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = sum(1 for a, b in zip(dirs, dirs[1:] + [dirs[0]]) if a != b)
    return turns / len(path)


def _frame_mask(h, w):
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


def _blob_mask():
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


def _arrange(children, compact=False):
    random.shuffle(children)
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    mode = random.choice(("vertical", "horizontal", "diagonal"))
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
    r = c = 0
    for child, (h, w), gap in zip(children, dims, gaps + [0]):
        placements[child] = (r, c)
        r += h + gap
        c += w + gap
    return placements, group_h, group_w


def _prepare(node, compact=False):
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


def _walk(node, base=(0, 0)):
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _place_outside(root, outside):
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


def _node_from_obj(obj):
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


def _expected_output(root, outside, bg, fg, edge, include_outside):
    def add(obj):
        node = _node_from_obj(obj)
        for child in obj.children:
            child_node = add(child)
            child_node.parent = node
            node.children.append(child_node)
        return node

    roots = [add(root)]
    if include_outside and outside is not None:
        roots.append(add(outside))
    base = _legacy_layout_root(roots) if include_outside else _layout_root(roots)
    mask = _rotate_mask(base, edge)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _make_tree(target_edge="top"):
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")
    if target_edge != "top":
        inner = _Obj("frame")
        inner.children = [_Obj("blob")]
        root.children = [inner, _Obj("blob")]
        if random.random() < ASYMMETRIC_OUTSIDE_PROB:
            outside = _Obj("blob", outside_side=random.choice(sides))
        return root, outside

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


def _draw(objects, bg, fg):
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


def _weighted_choice(weighted_items):
    items = [item for item, _weight in weighted_items]
    weights = [weight for _item, weight in weighted_items]
    return random.choices(items, weights=weights, k=1)[0]


def _next_target_edge():
    return _EDGE_ORDER[_EDGE_CYCLE_INDEX]


def _control_positions(shape, edge, count):
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


def _add_controls(grid, bg, fg, edge):
    ctrl = random.choice([c for c in range(10) if c not in {bg, fg}])
    count = _weighted_choice(CONTROL_COUNT_WEIGHTS)
    pts = _control_positions(grid.shape, edge, count)
    out = grid.copy()
    for r, c in pts:
        out[r, c] = ctrl
    return out


def _direction_changes_output(base_out, rotated_out, edge):
    if edge == "top":
        return True
    return not np.array_equal(base_out, rotated_out)


def _gen_attempt(target_edge):
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree(target_edge=target_edge)
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
    if outside is None and random.random() > KEEP_NO_OUTSIDE_PROB:
        return None

    base_expected = _expected_output(root, outside, bg, fg, "top", include_outside=False)
    candidate_edges = []
    expected_by_edge = {}
    for edge in _EDGE_TO_ROTATION:
        expected = _expected_output(root, outside, bg, fg, edge, include_outside=False)
        if _direction_changes_output(base_expected, expected, edge):
            candidate_edges.append(edge)
            expected_by_edge[edge] = expected

    if target_edge not in candidate_edges:
        return None
    edge = target_edge
    expected = expected_by_edge[edge]

    if outside is not None:
        legacy = _expected_output(root, outside, bg, fg, edge, include_outside=True)
        if np.array_equal(expected, legacy):
            return None

    grid = _add_controls(grid, bg, fg, edge)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input():
    global _EDGE_CYCLE_INDEX

    target_edge = _next_target_edge()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(target_edge)
        if grid is not None:
            _EDGE_CYCLE_INDEX = (_EDGE_CYCLE_INDEX + 1) % len(_EDGE_ORDER)
            return grid
    raise RuntimeError(f"failed to generate valid 2d0172a1 combo input for edge {target_edge}")
