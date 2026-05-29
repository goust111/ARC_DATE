# Variant of 2d0172a1
# Dimension: D15 — 符号编码：控制像素指示变换参数 (symbol_instructions)
# Change: 无符号编码——变换规则固定，不依赖任何控制像素或图例... → 特殊颜色标记指示嵌套深度

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste

# Design analysis
# variant:
#   input:
#     - 保留原题的粗糙单色 frame/blob 结构色，所有几何对象仍靠 containment 决定语义。
#     - 新增每个 frame 一个 1 像素彩色 marker，放在该 frame 外侧 1 格背景中。
#     - marker 颜色固定编码深度：depth-1 外框=红(2)，depth-2 内框=蓝(1)。
#     - blob 不带 marker，仍只是结构色小团块；outside blob 仍允许存在。
#   output:
#     - 几何规范化规则不变：blob -> 1 像素，frame -> 矩形边框，兄弟顺序/分带规则沿用原题。
#     - 唯一变化是 frame 的规范化边框颜色不再统一，而是读取其 marker 颜色。
#   behavior_trace:
#     - 原题里 outer frame 与 inner frame 在输出里都会被刷成同一个 fg。
#     - 现在输入里 outer frame 旁多一个红点，inner frame 旁多一个蓝点。
#     - solver 先像原题一样恢复 frame/blob containment tree，再把每个 frame 匹配到最近 marker。
#     - 渲染规范化输出时，outer border 用红、inner border 用蓝，blob 仍用结构色。
#   complexity_calibration:
#     - 只用两档 marker 颜色，和原任务最大深度 2 对齐，避免无支撑的大量颜色映射。
#     - depth-2 样例提高出现率，让少量样例里较容易同时看到红外框与蓝内框。
#     - 结构色明确不同于 marker 色，避免“输出换色”退化为原题单色输出。
#   implicit_assumptions_checked:
#     - 原题 `main()` 假设非背景只有一个 fg；这里失效，所以改成“最大非背景色=结构色，其余非背景色=marker”。
#     - 原题布尔 mask 渲染默认所有对象同色；这里失效，所以新增按 node 颜色递归上色。
#     - 原题根层 outside blob 特判只摆几何，不关心颜色；这里仍可用，但需要对 frame 和 blob 分别涂色。
#     - 最近 marker 匹配依赖 marker 与目标 frame 的曼哈顿距离最小；generator 保证 marker 与所属 frame 距离 1，且离其他 frame 更远。
#   placement_legality:
#     - marker 只放在 frame bbox 外侧 1 格，不接触结构色对象，也不占用其他 marker。
#     - inner frame 的 marker 放在 parent hole 内的空白区；outer frame 的 marker 放在根 frame 外部空白区。
#     - `_draw()` 把 marker 也计入整体包围盒与 padding，避免越界。
#   visual_observability:
#     - 红/蓝 marker 是显眼孤立信号，且输出中只有 frame 边框会继承它们，blob 不会。
#     - 同一输入若有嵌套 frame，外框红、内框蓝的对比直接可见，不需要依赖颜色 id 排序等不可见机制。


DEPTH_MARKER_COLORS = {1: 2, 2: 1}
MARKER_GAP = 1


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
        self.child_placements: dict[_Node, tuple[int, int]] = {}
        self.out_color: int | None = None


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
        node.child_placements = {}
        return

    assert node.children
    child_mask, child_placements, axis_map = _layout_group(node.children)
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
    node.child_placements = {child: (r + 2, c + 2) for child, (r, c) in child_placements.items()}


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


def _layout_root_positions(nodes: list[_Node]) -> tuple[tuple[int, int], dict[_Node, tuple[int, int]]]:
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
        shifted = {node: (r - min_r, c - min_c) for node, (r, c) in placements.items()}
        return (max_r - min_r, max_c - min_c), shifted

    mask, placements, _axis_map = _layout_group(nodes)
    return mask.shape, placements


def _paint_node(node: _Node, blob_color: int) -> np.ndarray:
    if not node.is_frame:
        return np.array([[blob_color]], dtype=int)

    assert node.mask is not None and node.out_color is not None
    out = np.full(node.mask.shape, -1, dtype=int)
    out[0, :] = node.out_color
    out[-1, :] = node.out_color
    out[:, 0] = node.out_color
    out[:, -1] = node.out_color
    for child, (r, c) in node.child_placements.items():
        child_canvas = _paint_node(child, blob_color)
        _blit(out, child_canvas, r, c)
    return out


def _blit(dst: np.ndarray, src: np.ndarray, r: int, c: int) -> None:
    mask = src != -1
    view = dst[r : r + src.shape[0], c : c + src.shape[1]]
    view[mask] = src[mask]


def _render_root(nodes: list[_Node], bg: int, blob_color: int) -> np.ndarray:
    shape, placements = _layout_root_positions(nodes)
    out = np.full(shape, -1, dtype=int)
    for node, (r, c) in placements.items():
        canvas = _paint_node(node, blob_color)
        _blit(out, canvas, r, c)
    out[out == -1] = bg
    return out


def _assign_marker_colors(frames: list[_Node], markers: list[tuple[int, int, int]]) -> None:
    assert len(markers) == len(frames)
    pairs = []
    for fi, frame in enumerate(frames):
        for mi, (r, c, color) in enumerate(markers):
            dist = np.abs(frame.coords[:, 0] - r) + np.abs(frame.coords[:, 1] - c)
            pairs.append((int(dist.min()), fi, mi, color))
    pairs.sort()
    used_frames = set()
    used_markers = set()
    for dist, fi, mi, color in pairs:
        if fi in used_frames or mi in used_markers:
            continue
        if dist > 3:
            break
        frames[fi].out_color = int(color)
        used_frames.add(fi)
        used_markers.add(mi)
    assert len(used_frames) == len(frames)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough nested frame-and-blob sketch with color markers into a clean canonical diagram.

    Input structure:
    - One color is background; one dominant structural color draws all rough frames and blobs.
    - Each frame also has a nearby single-pixel marker in a special color, placed just outside that frame.
    - Marker color encodes frame depth: outer and inner frames use different marker colors, while blobs keep only the structural color.

    Rule:
    - Recover the structural containment tree from the rough monochrome geometry exactly as in the base task.
    - Match each detected frame to its nearest non-structural marker pixel and use that marker color as the frame's canonical border color.
    - Turn every blob into one structural-color pixel and every frame into a rectangular border padded two cells around its direct children.
    - Preserve sibling row/column band ordering and the special outside-blob placement from the original task.

    Key insight: The wobbly outline still matters only for enclosure and relative placement; the new information is carried by isolated colored marker pixels that tell you how each cleaned frame should be colored.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    non_bg = sorted(set(map(int, g.flatten())) - {bg})
    assert len(non_bg) >= 2

    structural_fg = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))
    marker_colors = [color for color in non_bg if color != structural_fg]

    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[structural_fg],
        return_format="dict",
    )
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
        node_holes = _component_holes(g.shape, coords)
        node.is_frame = bool(node_holes)
        if not node.is_frame:
            node.out_color = int(structural_fg)
        nodes.append(node)
        holes.append(node_holes)

    frames = [node for node in nodes if node.is_frame]
    assert frames

    markers = [(int(r), int(c), int(g[r, c])) for color in marker_colors for r, c in np.argwhere(g == color)]
    assert len(markers) == len(frames)
    _assign_marker_colors(frames, markers)

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

    roots = [node for node in nodes if node.parent is None]
    return _render_root(roots, bg=bg, blob_color=int(structural_fg))


# --------------- generator (o5x) ---------------

# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

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
DEPTH1_PROB = 0.2
OUTSIDE_BLOB_PROB = 0.35
DEPTH1_INSIDE_RANGE = (1, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (0, 2)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 300


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side
        self.depth = 0
        self.marker_color: int | None = None
        self.marker_pos: tuple[int, int] | None = None


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


def _annotate_depths(node: _Obj, depth: int = 1) -> None:
    if node.kind == "frame":
        node.depth = depth
        node.marker_color = DEPTH_MARKER_COLORS[depth]
        child_depth = depth + 1
    else:
        node.depth = depth
        child_depth = depth
    for child in node.children:
        _annotate_depths(child, child_depth)


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
    node.out_color = obj.marker_color if obj.kind == "frame" else None
    return node


def _expected_output(root, outside, bg, fg):
    def add(obj):
        node = _node_from_obj(obj)
        if not node.is_frame:
            node.out_color = fg
        for child in obj.children:
            child_node = add(child)
            child_node.parent = node
            node.children.append(child_node)
        return node

    roots = [add(root)]
    if outside is not None:
        roots.append(add(outside))
    return _render_root(roots, bg=bg, blob_color=fg)


def _make_tree():
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
    _annotate_depths(root)
    return root, outside


def _occupied_cells(objects: list[_Obj]) -> set[tuple[int, int]]:
    occupied = set()
    for obj in objects:
        assert obj.mask is not None
        rr, cc = obj.pos
        for r, c in np.argwhere(obj.mask):
            occupied.add((int(rr + r), int(cc + c)))
    return occupied


def _marker_candidates(frame: _Obj) -> list[tuple[int, int]]:
    assert frame.mask is not None
    r0, c0 = frame.pos
    h, w = frame.mask.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    row_samples = []
    for value in (r0 + h // 2, r0 + 1, r1 - 1):
        if r0 <= value <= r1 and value not in row_samples:
            row_samples.append(value)
    col_samples = []
    for value in (c0 + w // 2, c0 + 1, c1 - 1):
        if c0 <= value <= c1 and value not in col_samples:
            col_samples.append(value)

    candidates = []
    for c in col_samples:
        candidates.append((r0 - MARKER_GAP, c))
        candidates.append((r1 + MARKER_GAP, c))
    for r in row_samples:
        candidates.append((r, c0 - MARKER_GAP))
        candidates.append((r, c1 + MARKER_GAP))
    random.shuffle(candidates)
    return candidates


def _place_markers(objects: list[_Obj]) -> bool:
    occupied = _occupied_cells(objects)
    frames = [obj for obj in objects if obj.kind == "frame"]
    frames.sort(key=lambda obj: obj.depth, reverse=True)
    used = set()
    for frame in frames:
        frame.marker_pos = None
        for coord in _marker_candidates(frame):
            if coord in occupied or coord in used:
                continue
            frame.marker_pos = coord
            used.add(coord)
            break
        if frame.marker_pos is None:
            return False
    return True


def _draw(objects, bg, fg):
    marker_points = [(obj.marker_pos[0], obj.marker_pos[1], obj.marker_color) for obj in objects if obj.marker_pos is not None]
    min_r = min([obj.pos[0] for obj in objects] + [r for r, _c, _color in marker_points])
    min_c = min([obj.pos[1] for obj in objects] + [c for _r, c, _color in marker_points])
    max_r = max([obj.pos[0] + obj.mask.shape[0] for obj in objects] + [r + 1 for r, _c, _color in marker_points])
    max_c = max([obj.pos[1] + obj.mask.shape[1] for obj in objects] + [c + 1 for _r, c, _color in marker_points])
    draw_h = max_r - min_r
    draw_w = max_c - min_c
    if draw_h > GRID_MAX or draw_w > GRID_MAX:
        return None

    extra_h_min = max(0, GRID_MIN - draw_h)
    extra_w_min = max(0, GRID_MIN - draw_w)
    extra_h_max = min(MAX_EXTRA_PAD, GRID_MAX - draw_h)
    extra_w_max = min(MAX_EXTRA_PAD, GRID_MAX - draw_w)
    if extra_h_min > extra_h_max or extra_w_min > extra_w_max:
        return None
    extra_h = random.randint(extra_h_min, extra_h_max)
    extra_w = random.randint(extra_w_min, extra_w_max)
    top_pad = random.randint(max(0, extra_h - MAX_PAD_PER_SIDE), min(MAX_PAD_PER_SIDE, extra_h))
    left_pad = random.randint(max(0, extra_w - MAX_PAD_PER_SIDE), min(MAX_PAD_PER_SIDE, extra_w))
    shift_r = top_pad - min_r
    shift_c = left_pad - min_c

    grid = np.full((draw_h + extra_h, draw_w + extra_w), bg, dtype=int)
    for obj in objects:
        assert obj.mask is not None
        r = obj.pos[0] + shift_r
        c = obj.pos[1] + shift_c
        obj.pos = (r, c)
        if obj.marker_pos is not None:
            mr, mc = obj.marker_pos
            obj.marker_pos = (mr + shift_r, mc + shift_c)
        patch = np.full(obj.mask.shape, bg, dtype=int)
        patch[obj.mask] = fg
        paste(grid, patch, r, c, transparent=bg)

    for obj in objects:
        if obj.marker_pos is not None:
            r, c = obj.marker_pos
            grid[r, c] = obj.marker_color
    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid != bg):
        return None
    return grid


def _gen_attempt():
    palette = [0, 3, 4, 5, 6, 7, 8, 9]
    bg = random.choice(palette)
    fg = random.choice([color for color in palette if color != bg])
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
    if not _place_markers(objects):
        return None

    grid = _draw(objects, bg, fg)
    if grid is None:
        return None
    expected = _expected_output(root, outside, bg, fg)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    frame_colors = sorted(set(map(int, expected.flatten())) - {bg, fg})
    if frame_colors != sorted({obj.marker_color for obj in objects if obj.kind == "frame"}):
        return None
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 input")
