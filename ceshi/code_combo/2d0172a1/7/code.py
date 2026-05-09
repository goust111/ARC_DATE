# Combo variant of 2d0172a1 (index 7)
# combo_spec: D5/2+D17/0
# Name: 色块保留原始尺寸+框架保留宽高比

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis
# combo_analysis:
#   concretization_and_behavior_trace:
#   - 原题把 blob 压成单点，把 frame 变成刚好包住 direct-children 的标准矩形边框，因此输入里的
#     度量信息几乎被全部丢掉。
#   - 本融合版保留两类度量：
#     1. blob -> 输出为与输入 blob 外接矩形同尺寸的实心矩形；
#     2. frame -> 输出仍是 1px 矩形边框，但其外框尺寸要保持输入 frame bbox 的宽高比，
#        只是在整数网格上按最小可容纳 direct-children 的倍数放大。
#   - 例子：若某 frame 里有两个 blob，输入 bbox 分别是 3x4 与 2x2，children 的 canonical
#     布局外包为 6x9，则原题输出 frame 固定为 10x13，blob 只会变成两个点。
#     本融合版中，两个 blob 变为 3x4 和 2x2 的实心块；若该 frame 输入 bbox 为 12x18，
#     最简比 2:3，则输出 frame 取满足 >=10x13 的最小 2:3 整数倍，即 10x15。
#   - 差异不是简单拼接：blob 变大后会反过来抬高 child_mask 最小尺寸，进而影响 frame 的最小可行
#     比例缩放倍数；因此 frame 的最终尺寸同时受 sibling 布局和输入 frame 比例约束。
#   complexity_calibration:
#   - blob 的输入 bbox 继续限制在 2..4 格量级，但形状保持不规则，这样人眼能看出“输出块大小来自
#     输入 bbox，而不是来自填充形状细节”。
#   - frame 的输入 bbox 不允许任意近似比例，而是从少量离散族 `(2:3, 3:2, 3:4, 4:3)` 中选，
#     再改变缩放倍数。这样 2-4 个样例里会重复出现“瘦高框输出仍瘦高、宽框输出仍宽”的证据。
#   - 不额外引入颜色或额外符号，避免同时学习过多自由度。
#   implicit_assumptions_review:
#   - 原 `_build_layout(blob)` 假设 blob 的 canonical mask 恒为 1x1；融合后必须改为 bbox 尺寸。
#   - 原 `_build_layout(frame)` 假设 frame 外框尺寸恒为 `child_mask + 4`；融合后改为先算最小容纳尺寸，
#     再按输入 bbox 最简比例做整数缩放。
#   - 原 `axis_map += 2` 依赖 children 固定 inset；融合后 frame 可能有额外留白，必须改成
#     `+= inset_r/inset_c`，否则 root 外侧 blob 的对齐行列会错位。
#   - 原 `_layout_root` 为 outside blob 预留固定 3x3 extents；融合后 outside blob 可能是 2x4
#     或 4x3，必须按真实 mask.shape 计算画布范围和放置坐标。
#   - `_axis_coordinates` 的 band/order 推理仍可沿用，因为它只依赖输入 bbox 区间和 canonical
#     mask 的尺寸/anchor；这些在融合后仍是明确定义的。
#   placement_legality:
#   - solver 侧：frame 外框先保证至少比 child 布局四周多两格，再居中嵌入 child_mask，不会压到边框。
#   - generator 侧：先排好 children 的粗糙输入布局，再按离散比例族选择能容纳它们的 frame bbox；
#     outside blob 仍与根 frame 保持至少 2 格输入间距，因此对象不重叠、不越界。
#   visual_observability:
#   - 看 input->output 时，人类能同时观察到：
#     1. 不规则 blob 在输出里变成同 bbox 的规整实心块；
#     2. 同样的 child 排列在不同 frame 纵横比下会落进明显偏宽或偏高的矩形边框。
#   cleanliness:
#   - 输出仍只有一个前景色，frame 是空心矩形边框，blob 是实心矩形块；对象之间保留背景缝隙，
#     不会出现多层标记叠在一起难以分辨的情况。
#   new_main_assumptions_and_generator_support:
#   - `main()` 新增假设：每个 frame 至少有一个 child，且输入 frame bbox 比例有信息量。
#     generator 继续只造非空 frame，并用离散比例族保证比例信息稳定可见。
#   - `main()` 新增假设：outside blob 的 bbox 尺寸需要完整保留并与 frame 保持 1 格输出间距。
#     generator 通过 `_place_outside` 和 `_layout_root` 的真实 extents 逻辑保证这点。


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.rect_shape = (int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1))
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


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _scaled_frame_shape(bbox: tuple[int, int, int, int], min_h: int, min_w: int) -> tuple[int, int]:
    src_h, src_w = _bbox_hw(bbox)
    g = math.gcd(src_h, src_w)
    unit_h = src_h // g
    unit_w = src_w // g
    k = max(_ceil_div(min_h, unit_h), _ceil_div(min_w, unit_w), 1)
    return unit_h * k, unit_w * k


def _build_layout(node: _Node) -> None:
    if node.mask is not None:
        return

    if not node.is_frame:
        h, w = node.rect_shape
        node.mask = np.ones((h, w), dtype=bool)
        node.anchor = (h // 2, w // 2)
        node.axis_map = {
            "row": [(float(node.center[0]), int(node.anchor[0]))],
            "col": [(float(node.center[1]), int(node.anchor[1]))],
        }
        return

    assert node.children
    child_mask, _placements, axis_map = _layout_group(node.children)
    child_h, child_w = child_mask.shape
    min_h = child_h + 4
    min_w = child_w + 4
    out_h, out_w = _scaled_frame_shape(node.bbox, min_h, min_w)
    inset_r = 2 + (out_h - min_h) // 2
    inset_c = 2 + (out_w - min_w) // 2

    out = np.zeros((out_h, out_w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    out[inset_r : inset_r + child_h, inset_c : inset_c + child_w] |= child_mask

    node.mask = out
    node.anchor = (out_h // 2, out_w // 2)
    node.axis_map = {
        "row": [(orig, pos + inset_r) for orig, pos in axis_map["row"]],
        "col": [(orig, pos + inset_c) for orig, pos in axis_map["col"]],
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


def _outside_side(blob: _Node, frame: _Node) -> str:
    r0, c0, r1, c1 = blob.bbox
    fr0, fc0, fr1, fc1 = frame.bbox
    if c0 > fc1:
        return "right"
    if c1 < fc0:
        return "left"
    if r0 > fr1:
        return "bottom"
    if r1 < fr0:
        return "top"

    dr = float(blob.center[0] - (fr0 + fr1) / 2)
    dc = float(blob.center[1] - (fc0 + fc1) / 2)
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
    blobs = [node for node in nodes if not node.is_frame]
    if len(frames) == 1 and blobs:
        frame = frames[0]
        assert frame.mask is not None and frame.anchor is not None
        placements: dict[_Node, tuple[int, int]] = {frame: (0, 0)}
        extents = [(0, 0, frame.mask.shape[0], frame.mask.shape[1])]

        for blob in blobs:
            assert blob.mask is not None and blob.anchor is not None
            blob_h, blob_w = blob.mask.shape
            side = _outside_side(blob, frame)
            if side in {"right", "left"}:
                r = _map_frame_axis(frame, 0, float(blob.center[0])) - blob.anchor[0]
                c = frame.mask.shape[1] + 1 if side == "right" else -blob_w - 1
            else:
                r = frame.mask.shape[0] + 1 if side == "bottom" else -blob_h - 1
                c = _map_frame_axis(frame, 1, float(blob.center[1])) - blob.anchor[1]
            placements[blob] = (int(r), int(c))
            extents.append((int(r), int(c), int(r + blob_h), int(c + blob_w)))

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
                out[rr : rr + node.mask.shape[0], cc : cc + node.mask.shape[1]] |= node.mask
        return out

    mask, _placements, _axis_map = _layout_group(nodes)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough nested frame-and-blob sketch into a clean rectangle-and-block diagram that preserves measured size cues.

    Input structure:
    - The grid uses one background color and one foreground color.
    - Foreground 4-connected components are either rough closed-loop frames or compact filled blobs.
    - Frames may contain other frames/blobs, and a blob may also sit outside the outermost frame.

    Rule:
    - Recover the containment tree by checking which components lie inside each frame hole.
    - Replace each blob with a solid rectangle whose height and width match that blob's input bbox.
    - Replace each frame with a 1-pixel rectangular border that keeps the input frame bbox aspect ratio, scaled to the smallest integer size that can still hold its direct children with the usual moat.
    - Preserve sibling row/column band ordering, keep one background cell between canonical objects, and attach any outside blob to the same side of the root frame.

    Key insight: The wobbly outlines are still normalized away, but now the useful information is not only enclosure and sibling order; blob bbox size and frame bbox aspect ratio also survive into the cleaned output.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
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
FRAME_H_RANGE = (7, 24)
FRAME_W_RANGE = (7, 24)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60
FRAME_RATIO_UNITS = ((2, 3), (3, 2), (3, 4), (4, 3))
FRAME_SLACK_CAP = 8
COMPACT_FRAME_SLACK_CAP = 4

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


def _arrange(children: list[_Obj], compact: bool = False, mode: str | None = None) -> tuple[dict[_Obj, tuple[int, int]], int, int]:
    random.shuffle(children)
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    mode = random.choice(("vertical", "horizontal", "diagonal")) if mode is None else mode
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


def _pick_frame_size(
    min_h: int,
    min_w: int,
    group_h: int,
    group_w: int,
    compact: bool,
    ratio_units: tuple[tuple[int, int], ...] | None = None,
) -> tuple[int, int] | None:
    child_aspect = group_w / group_h
    slack_cap = COMPACT_FRAME_SLACK_CAP if compact else FRAME_SLACK_CAP
    best_by_ratio: dict[tuple[int, int], tuple[float, int, int]] = {}
    unit_choices = FRAME_RATIO_UNITS if ratio_units is None else ratio_units
    for unit_h, unit_w in unit_choices:
        q_min = max(_ceil_div(min_h, unit_h), _ceil_div(min_w, unit_w))
        q_max = min(FRAME_H_RANGE[1] // unit_h, FRAME_W_RANGE[1] // unit_w)
        if q_min > q_max:
            continue
        for q in range(q_min, q_max + 1):
            h = unit_h * q
            w = unit_w * q
            slack_h = h - min_h
            slack_w = w - min_w
            if slack_h + slack_w > slack_cap:
                continue
            frame_aspect = w / h
            mismatch = abs(math.log(frame_aspect / child_aspect))
            one_axis_slack = int((slack_h == 0) != (slack_w == 0))
            score = mismatch + 0.12 * min(slack_h + slack_w, 6) + 0.2 * one_axis_slack
            score += random.random() * 0.03
            candidate = (score, h, w)
            best = best_by_ratio.get((unit_h, unit_w))
            if best is None or candidate < best:
                best_by_ratio[(unit_h, unit_w)] = candidate
    if not best_by_ratio:
        return None
    _, h, w = random.choice(list(best_by_ratio.values()))
    return h, w


def _mode_order_for_ratio(unit_h: int, unit_w: int) -> list[str]:
    if unit_h > unit_w:
        modes = ["vertical", "diagonal", "horizontal"]
    elif unit_w > unit_h:
        modes = ["horizontal", "diagonal", "vertical"]
    else:
        modes = ["diagonal", "horizontal", "vertical"]
    if random.random() < 0.35:
        modes[1], modes[2] = modes[2], modes[1]
    return modes


def _prepare(node: _Obj, compact: bool = False) -> bool:
    if node.kind == "blob":
        node.mask = _blob_mask()
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    choice = None
    for unit_h, unit_w in random.sample(FRAME_RATIO_UNITS, len(FRAME_RATIO_UNITS)):
        for mode in _mode_order_for_ratio(unit_h, unit_w):
            placements, group_h, group_w = _arrange(node.children, compact=tight, mode=mode)
            margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
            min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
            min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
            if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
                continue
            picked = _pick_frame_size(
                min_h,
                min_w,
                group_h,
                group_w,
                compact=tight,
                ratio_units=((unit_h, unit_w),),
            )
            if picked is not None:
                choice = (placements, group_h, group_w, margin, picked)
                break
        if choice is not None:
            break
    if choice is None:
        return False
    placements, group_h, group_w, margin, picked = choice
    h, w = picked
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


def _expected_output(root: _Obj, outside: _Obj | None, bg: int, fg: int) -> np.ndarray:
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
    mask = _layout_root(roots)
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
    expected = _expected_output(root, outside, bg, fg)
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
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
