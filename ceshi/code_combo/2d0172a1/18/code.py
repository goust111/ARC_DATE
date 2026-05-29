# Combo variant of 2d0172a1 (index 18)
# combo_spec: D3/2+D6/0
# Name: 有缺口框架+外接矩形包含判定
#
# Design analysis
# - Concrete behavior:
#   原题把真正闭合的细框当作 frame，并用洞来决定谁在谁里面。这里每个 frame 都被故意打出一个 1-2 像素的边缘缺口，
#   于是 flood-fill 看不到洞；同时保留一个或多个“bbox-only”子 blob，它们位于 frame 的 bbox 内、但落在闭合版本轮廓的角落口袋里。
#   例子：外框 bbox 是 rows 2..14, cols 3..18，顶部中段缺了 2 个像素；左上角还有一个 3x3 blob 躲在原闭环之外的口袋里。
#   原逻辑会因为外框没有洞而根本不认它是 frame；就算先脑补补上缺口，角落 blob 也不在真实洞里。融合后则应把该对象识别为
#   “准闭合 frame”，并按 bbox 包含把角落 blob 收为子节点，所以输出仍是一个规范矩形框，blob 被放到框内。
# - Complexity calibration:
#   为了避免退化，generator 强制每个样本至少有一个 bbox-only blob，并且每个 frame 都恰好保留一个缺口。
#   这样样例同时体现两件事：一是不能依赖严格洞检测识别 frame，二是层级必须按 bbox 而非真实 enclosed area。
#   同时仍保留 1-2 层 frame、普通洞内子对象和可选外部 blob，避免所有样本都坍缩成单一模板。
# - Hidden assumptions audit:
#   原 main() 的 `_component_holes` 同时承担“识别 frame”和“判定包含”两件事；现在这两个前提都不成立。
#   新 solver 改成：frame 识别依赖“大 bbox + 细长路径 + 像素度数最多为 2”，包含关系只允许 frame 当父节点，且完全由 bbox 包含决定。
#   这又引入了新假设：blob 不会伪装成大号细路径，也不会出现 blob 包含 blob 的 bbox 歧义。generator 用尺寸范围、blob 生成方式和树结构
#   保证这些前提成立，并且所有 frame 只打一个缺口，继续保持单个 4 连通组件，满足 `find_components` 的对象模型。
# - Placement legality:
#   生成器先造闭合 frame，用闭合版本的 hole/pocket 区域搜索合法位置，再把渲染用的 mask 换成带缺口版本。
#   子对象放置时不仅禁止重叠，还禁止与 frame 或彼此 4 邻接接触，避免不同对象合并成同一前景连通块。
#   缺口只开在 bbox 外边缘的连续直段上，且不删掉整条边，因此 bbox 不变、frame 仍是单连通路径、洞数变为 0。
# - Visual observability:
#   人眼能直接看到两条线索：frame 明显有缺口，但输出仍把它规整成完整矩形；同时至少有一个 blob 位于 bbox 内的角落口袋，
#   看起来不在真实包围区里，却仍被收进输出框内。两种现象叠加后，少量样例就足以排除“严格闭环 + 真洞包含”的旧规则。
# - Cleanliness:
#   bbox-only blob 仅放在角口袋，普通子对象放在闭合版洞区，区域职责分明；对象间保留一格以上空隙，输出仍是整洁的矩形框加像素点。
# - Color inference:
#   颜色机制完全不变，仍然是单前景色几何规范化，不依赖任何颜色值排序或隐藏元数据。
# - New solver assumptions check:
#   `_is_quasi_frame` 假设 frame 的前景像素形成一条细长路径，且 bbox 至少有一边明显大于 blob 的 2..4 尺寸；generator 保证这一点。
#   bbox 选父节点时假设最小包含 frame 就是直接父节点；generator 保证 frame 只做真嵌套，不制造交叉或共享 bbox 的歧义结构。

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


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


def _component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    return mask


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = _component_mask(shape, coords)
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _degree_map(mask: np.ndarray) -> np.ndarray:
    deg = np.zeros_like(mask, dtype=int)
    deg[1:, :] += mask[:-1, :]
    deg[:-1, :] += mask[1:, :]
    deg[:, 1:] += mask[:, :-1]
    deg[:, :-1] += mask[:, 1:]
    return deg


def _is_quasi_frame(shape: tuple[int, int], coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if min(h, w) < 7:
        return False
    mask = _component_mask(shape, coords)
    deg = _degree_map(mask)[mask]
    if deg.size == 0 or np.any(deg == 0) or np.any(deg > 2):
        return False
    endpoints = int(np.count_nonzero(deg == 1))
    if endpoints not in (0, 2):
        return False
    return True


def _frame_regions(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab, n = label_mask(~mask, connectivity=4)
    boundary = (set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])) - {0}
    hole_labels = [k for k in range(1, n + 1) if k not in boundary]
    hole_mask = np.isin(lab, hole_labels)
    pocket_mask = np.isin(lab, list(boundary))
    return hole_mask, pocket_mask


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


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


def _layout_root(nodes: list[_Node]) -> np.ndarray:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough nested sketch with broken frames into a canonical border-and-dot diagram.

    Input structure:
    - The grid uses one background color and one foreground color.
    - Foreground components are either small filled blobs or large thin frame-like curves with a visible 1-2 pixel gap.
    - Objects may lie inside a frame's bounding rectangle even when the gap makes strict hole detection impossible.

    Rule:
    - Recognize frames from their geometry as large, thin quasi-loops rather than requiring a literal enclosed hole.
    - Build the hierarchy by bounding-box inclusion: a component belongs to the smallest frame whose bbox fully contains its bbox.
    - Replace each blob with a single foreground pixel and each frame with a clean rectangular border padded around its direct children.
    - Preserve sibling row/column ordering from the original arrangement when laying out the canonical children.

    Key insight:
    - The decisive information is bbox nesting plus relative sibling placement; the broken hand-drawn outline is only evidence that a large thin component should still count as a frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    assert comps

    nodes: list[_Node] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        bbox = tuple(map(int, comp["bbox"]))
        node = _Node(
            coords=coords,
            bbox=bbox,
            center=coords.mean(axis=0),
        )
        node.is_frame = _is_quasi_frame(g.shape, coords, bbox)
        nodes.append(node)

    assert any(node.is_frame for node in nodes)
    for node_i, node in enumerate(nodes):
        containers = []
        for frame_i, frame in enumerate(nodes):
            if node_i == frame_i or not frame.is_frame:
                continue
            if _bbox_contains(frame.bbox, node.bbox):
                containers.append(frame)
        if containers:
            node.parent = min(containers, key=lambda item: (_bbox_area(item.bbox), int(item.coords.shape[0])))
            node.parent.children.append(node)

    roots = [node for node in nodes if node.parent is None]
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


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
FRAME_GAP_WIDTH_RANGE = (1, 2)

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
DEPTH1_PROB = 0.4
OUTSIDE_BLOB_PROB = 0.2
DEPTH1_BBOX_BLOBS = (1, 2)
DEPTH1_HOLE_BLOBS = (0, 1)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_HOLE_BLOBS = (0, 1)
DEPTH2_OUTER_BBOX_BLOBS = (1, 2)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 300


class _Obj:
    def __init__(self, kind, children=None, outside_side=None, bbox_only=False, bbox_corner=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.closed_mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side
        self.bbox_only = bbox_only
        self.bbox_corner = bbox_corner


def _loop_ok(mask: np.ndarray) -> bool:
    _, n = label_mask(mask, connectivity=4)
    if n != 1:
        return False
    deg = _degree_map(mask)
    if not np.all(deg[mask] == 2):
        return False
    inv_lab, inv_n = label_mask(~mask, connectivity=4)
    boundary = set(inv_lab[0, :]) | set(inv_lab[-1, :]) | set(inv_lab[:, 0]) | set(inv_lab[:, -1])
    return sum(1 for k in range(1, inv_n + 1) if k not in boundary) == 1


def _open_curve_ok(mask: np.ndarray) -> bool:
    _, n = label_mask(mask, connectivity=4)
    if n != 1:
        return False
    deg = _degree_map(mask)[mask]
    if np.any(deg == 0) or np.any(deg > 2):
        return False
    if int(np.count_nonzero(deg == 1)) != 2:
        return False
    inv_lab, inv_n = label_mask(~mask, connectivity=4)
    boundary = set(inv_lab[0, :]) | set(inv_lab[-1, :]) | set(inv_lab[:, 0]) | set(inv_lab[:, -1])
    return sum(1 for k in range(1, inv_n + 1) if k not in boundary) == 0


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


def _closed_frame_mask(h, w, corner_mins=None):
    max_dx = min(FRAME_CORNER_MAX, max(1, (w - 5) // 2))
    max_dy = min(FRAME_CORNER_MAX, max(1, (h - 5) // 2))
    lo, hi = FRAME_TURN_RATIO
    corner_mins = corner_mins or {}
    min_tl_dy, min_tl_dx = corner_mins.get("tl", (1, 1))
    min_tr_dy, min_tr_dx = corner_mins.get("tr", (1, 1))
    min_br_dy, min_br_dx = corner_mins.get("br", (1, 1))
    min_bl_dy, min_bl_dx = corner_mins.get("bl", (1, 1))
    if max_dx < max(min_tl_dx, min_tr_dx, min_br_dx, min_bl_dx):
        return None
    if max_dy < max(min_tl_dy, min_tr_dy, min_br_dy, min_bl_dy):
        return None

    for _ in range(FRAME_RETRIES):
        tl_dx = random.randint(min_tl_dx, max_dx)
        tr_dx = random.randint(min_tr_dx, max_dx)
        br_dx = random.randint(min_br_dx, max_dx)
        bl_dx = random.randint(min_bl_dx, max_dx)
        tl_dy = random.randint(min_tl_dy, max_dy)
        tr_dy = random.randint(min_tr_dy, max_dy)
        br_dy = random.randint(min_br_dy, max_dy)
        bl_dy = random.randint(min_bl_dy, max_dy)

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


def _contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    runs = []
    start = int(values[0])
    prev = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value != prev + 1:
            runs.append((start, prev))
            start = value
        prev = value
    runs.append((start, prev))
    return runs


def _gap_candidates(mask: np.ndarray) -> list[tuple[str, int, int, int]]:
    h, w = mask.shape
    candidates = []
    top = np.flatnonzero(mask[0, :])
    bottom = np.flatnonzero(mask[h - 1, :])
    left = np.flatnonzero(mask[:, 0])
    right = np.flatnonzero(mask[:, w - 1])
    for start, end in _contiguous_runs(top):
        span = end - start + 1
        for width in range(FRAME_GAP_WIDTH_RANGE[0], min(FRAME_GAP_WIDTH_RANGE[1], span - 2) + 1):
            for offset in range(start + 1, end - width + 1):
                candidates.append(("top", 0, offset, width))
    for start, end in _contiguous_runs(bottom):
        span = end - start + 1
        for width in range(FRAME_GAP_WIDTH_RANGE[0], min(FRAME_GAP_WIDTH_RANGE[1], span - 2) + 1):
            for offset in range(start + 1, end - width + 1):
                candidates.append(("bottom", h - 1, offset, width))
    for start, end in _contiguous_runs(left):
        span = end - start + 1
        for width in range(FRAME_GAP_WIDTH_RANGE[0], min(FRAME_GAP_WIDTH_RANGE[1], span - 2) + 1):
            for offset in range(start + 1, end - width + 1):
                candidates.append(("left", 0, offset, width))
    for start, end in _contiguous_runs(right):
        span = end - start + 1
        for width in range(FRAME_GAP_WIDTH_RANGE[0], min(FRAME_GAP_WIDTH_RANGE[1], span - 2) + 1):
            for offset in range(start + 1, end - width + 1):
                candidates.append(("right", w - 1, offset, width))
    return candidates


def _apply_gap(mask: np.ndarray, candidate: tuple[str, int, int, int]) -> np.ndarray:
    side, fixed, start, width = candidate
    out = mask.copy()
    if side in {"top", "bottom"}:
        out[fixed, start : start + width] = False
    else:
        out[start : start + width, fixed] = False
    return out


def _gapped_frame_mask(closed_mask: np.ndarray) -> np.ndarray | None:
    candidates = _gap_candidates(closed_mask)
    random.shuffle(candidates)
    for candidate in candidates:
        gapped = _apply_gap(closed_mask, candidate)
        if _open_curve_ok(gapped):
            return gapped
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
    children = list(children)
    random.shuffle(children)
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 0:
        return {}, 0, 0
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


def _child_pixels(mask: np.ndarray, origin: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    return coords[:, 0] + origin[0], coords[:, 1] + origin[1]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _candidate_positions(region_mask: np.ndarray, child_mask: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    H, W = region_mask.shape
    h, w = child_mask.shape
    coords = np.argwhere(child_mask)
    out = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            rr = coords[:, 0] + r
            cc = coords[:, 1] + c
            if np.all(region_mask[rr, cc]) and not np.any(blocked[rr, cc]):
                out.append((r, c))
    return out


def _corner_metric(corner: str, pos: tuple[int, int], child_shape: tuple[int, int], frame_shape: tuple[int, int]) -> tuple[int, int, int]:
    r, c = pos
    h, w = child_shape
    H, W = frame_shape
    top_gap = r
    left_gap = c
    bottom_gap = H - (r + h)
    right_gap = W - (c + w)
    touches = int(top_gap == 0) + int(left_gap == 0) + int(bottom_gap == 0) + int(right_gap == 0)
    if corner == "tl":
        dist = top_gap + left_gap
        aux = top_gap + 2 * left_gap
    elif corner == "tr":
        dist = top_gap + right_gap
        aux = top_gap + 2 * right_gap
    elif corner == "br":
        dist = bottom_gap + right_gap
        aux = bottom_gap + 2 * right_gap
    else:
        dist = bottom_gap + left_gap
        aux = bottom_gap + 2 * left_gap
    return (-touches, dist, aux)


def _pick_position(candidates, *, frame_shape, child_shape, target=None, corner=None):
    if not candidates:
        return None
    if corner is not None:
        best_key = min(_corner_metric(corner, pos, child_shape, frame_shape) for pos in candidates)
        best = [pos for pos in candidates if _corner_metric(corner, pos, child_shape, frame_shape) == best_key]
        return random.choice(best)
    assert target is not None
    best_key = min((abs(pos[0] - target[0]) + abs(pos[1] - target[1]), pos[0], pos[1]) for pos in candidates)
    best = [pos for pos in candidates if (abs(pos[0] - target[0]) + abs(pos[1] - target[1]), pos[0], pos[1]) == best_key]
    return random.choice(best)


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, pos: tuple[int, int]) -> None:
    rr, cc = _child_pixels(mask, pos)
    placed = np.zeros_like(blocked, dtype=bool)
    placed[rr, cc] = True
    blocked |= _dilate4(placed)


def _prepare(node, compact=False):
    if node.kind == "blob":
        node.mask = _blob_mask()
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    regular_children = [child for child in node.children if not child.bbox_only]
    pocket_children = [child for child in node.children if child.bbox_only]
    has_frame_child = any(child.kind == "frame" for child in regular_children)
    tight = compact or has_frame_child

    placements, group_h, group_w = _arrange(regular_children, compact=tight)
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)

    corner_mins = {}
    for child in pocket_children:
        assert child.kind == "blob"
        h0, w0 = child.mask.shape
        dy_req = h0 + 1
        dx_req = w0 + 1
        old = corner_mins.get(child.bbox_corner, (1, 1))
        corner_mins[child.bbox_corner] = (max(old[0], dy_req), max(old[1], dx_req))
        min_h = max(min_h, 2 * dy_req + FRAME_MIN_EDGE + 1)
        min_w = max(min_w, 2 * dx_req + FRAME_MIN_EDGE + 1)

    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False
    h_hi = min(FRAME_H_RANGE[1], min_h + 2) if tight else FRAME_H_RANGE[1]
    w_hi = min(FRAME_W_RANGE[1], min_w + 3) if tight else FRAME_W_RANGE[1]
    h = random.randint(min_h, h_hi)
    w = random.randint(min_w, w_hi)

    node.closed_mask = _closed_frame_mask(h, w, corner_mins=corner_mins)
    if node.closed_mask is None:
        return False
    node.mask = _gapped_frame_mask(node.closed_mask)
    if node.mask is None:
        return False

    hole_mask, pocket_mask = _frame_regions(node.closed_mask)
    blocked = _dilate4(node.mask)
    targets = {}
    if regular_children:
        top = margin + random.randint(0, h - 2 * margin - group_h)
        left = margin + random.randint(0, w - 2 * margin - group_w)
        for child, (r, c) in placements.items():
            targets[child] = (top + r, left + c)

    for child in sorted(regular_children, key=lambda item: item.mask.shape[0] * item.mask.shape[1], reverse=True):
        candidates = _candidate_positions(hole_mask, child.mask, blocked)
        if not candidates:
            return False
        pos = _pick_position(candidates, frame_shape=node.mask.shape, child_shape=child.mask.shape, target=targets[child])
        if pos is None:
            return False
        child.rel = pos
        _mark_blocked(blocked, child.mask, pos)

    for child in sorted(pocket_children, key=lambda item: item.mask.shape[0] * item.mask.shape[1], reverse=True):
        candidates = _candidate_positions(pocket_mask, child.mask, blocked)
        if not candidates:
            return False
        pos = _pick_position(
            candidates,
            frame_shape=node.mask.shape,
            child_shape=child.mask.shape,
            corner=child.bbox_corner,
        )
        if pos is None:
            return False
        child.rel = pos
        _mark_blocked(blocked, child.mask, pos)
    return True


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


def _expected_output(root, outside, bg, fg):
    def add(obj):
        node = _node_from_obj(obj)
        for child in obj.children:
            child_node = add(child)
            child_node.parent = node
            node.children.append(child_node)
        return node

    roots = [add(root)]
    if outside is not None:
        roots.append(add(outside))
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _make_bbox_blob(corner):
    return _Obj("blob", bbox_only=True, bbox_corner=corner)


def _make_tree():
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")
    corners = ["tl", "tr", "br", "bl"]

    if random.random() < DEPTH1_PROB:
        random.shuffle(corners)
        n_bbox = random.randint(*DEPTH1_BBOX_BLOBS)
        n_hole = random.randint(*DEPTH1_HOLE_BLOBS)
        root.children = [_make_bbox_blob(corners[i]) for i in range(n_bbox)] + [_Obj("blob") for _ in range(n_hole)]
    else:
        inner = _Obj("frame")
        inner.children = [_Obj("blob") for _ in range(random.randint(*DEPTH2_INNER_BLOBS))]
        random.shuffle(corners)
        n_bbox = random.randint(*DEPTH2_OUTER_BBOX_BLOBS)
        n_hole = random.randint(*DEPTH2_OUTER_HOLE_BLOBS)
        root.children = [inner] + [_make_bbox_blob(corners[i]) for i in range(n_bbox)] + [_Obj("blob") for _ in range(n_hole)]

    total_children = len(root.children)
    if total_children < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
        outside = _Obj("blob", outside_side=random.choice(sides))
    return root, outside


def _draw(objects, bg, fg):
    min_r = min(obj.pos[0] for obj in objects)
    min_c = min(obj.pos[1] for obj in objects)
    max_r = max(obj.pos[0] + obj.mask.shape[0] for obj in objects)
    max_c = max(obj.pos[1] + obj.mask.shape[1] for obj in objects)
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
        patch = np.full(obj.mask.shape, bg, dtype=int)
        patch[obj.mask] = fg
        paste(grid, patch, r, c, transparent=bg)
    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == fg):
        return None
    return grid


def _count_bbox_only_children(root):
    total = 0
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            if child.bbox_only:
                total += 1
            stack.append(child)
    return total


def _gen_attempt():
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree()
    if _count_bbox_only_children(root) == 0:
        return None
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
    expected = _expected_output(root, outside, bg, fg)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
