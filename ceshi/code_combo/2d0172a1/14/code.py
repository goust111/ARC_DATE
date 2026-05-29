# Combo variant of 2d0172a1 (index 14)
# combo_spec: D7/0+D14/0
# Name: 按面积排序+强制垂直排列

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# concretization_and_trace:
# - 输入仍是单前景色的粗糙 frame/blob 图；frame 由一个 hole 表示，blob 是无 hole 的小连通块。
# - 原题对同一 frame 的 children 会先看 row/col band，再决定输出是 vertical、horizontal
#   还是 diagonal，并尽量保留输入中的先后关系。
# - D7/0 把“顺序来源”改成面积，但单独实现时 row/col 两轴都按同一排序推进，所以视觉上更像
#   左上到右下的 diagonal 排列。
# - D14/0 把“布局模式”改成固定 vertical，但单独实现时顺序仍来自空间位置。
# - 融合后两处同时改变：同一 sibling group 先按原始面积从大到小排序，再把排好序的对象统一
#   竖直堆叠。例子：三个 blob 在输入里从左到右摆成面积 4、10、7。原题输出仍近似横排/对角并
#   保持空间次序；融合后输出必为一列，顺序改成 10、7、4。
#
# complexity_calibration:
# - 如果 generator 继续大量生成已经竖直且恰好面积降序的 groups，这个融合会退化成原题或单
#   变体观感。因此这里强制多子节点 frame 常见，并偏向生成 horizontal/diagonal 输入。
# - 同组 blob 使用互不相同的面积档位，避免“面积排序”长期落入平手兜底，保证解题者真的需要
#   比较像素数。
# - 树深度、对象总量、颜色数仍保持原题量级：1-2 层 frame、2-6 个对象、仅 bg+fg 两色。
#
# implicit_assumption_audit:
# - 原 `_axis_coordinates()` 依赖 bbox band 能代表 sibling 输出相对位置；融合后这个前提失效，
#   因为输入空间位置对 frame 内 siblings 不再有决定性作用。
# - `_layout_group()` 也不能再分别计算 row/col 两轴的 band 坐标，而要改成单一 vertical stack。
# - `_build_layout()` 只要求 child mask 和 axis_map，接口仍可复用。
# - `_outside_side()` 与 `_map_frame_axis()` 仍然只服务于 root 外侧 blob 的侧别和对齐；两个单
#   变体都未改变这一分支，融合版继续保留，避免无关维度联动。
# - 新 solver 的隐式假设是：需要比较的 sibling 面积最好可区分。generator 用不同 blob 面积、
#   以及 frame 面积远大于 blob 的天然差异来保证这一点。
#
# placement_legality:
# - solver 的 vertical stack 在相邻 canonical 对象之间固定保留 1 格背景，因此不会重叠。
# - generator 仍先递归生成 child mask，再用 gap + margin 放进父 frame；所有布局都在 bbox
#   范围内取随机偏移，不会越界。
# - outside blob 仍只放在 outer frame 外侧，并保留足够 gap，不与 frame 接触。
#
# visual_observability_and_cleanliness:
# - 观察者可以先看输入里粗 blob 的像素多少，再看输出里谁在最上方；同时输入常被做成横排或斜排，
#   输出却固定变成竖排，因此两个维度的变化会在同一组 sibling 上同时出现。
# - 输出仍只包含矩形边框和单像素点，没有额外标记，层次清楚。
# - 规则完全依赖人类可见属性：包含关系、是否在 frame 外、以及各组件面积；不依赖颜色编号、
#   扫描顺序或随机种子。
#
# generator_guarantees_for_new_solver:
# - 内层 frame 只保留 1 个 blob，让嵌套关系存在但尺寸保持紧凑；可见的融合信号主要落在 root 的
#   “inner frame + 1~2 个 outer blob” sibling group 上，因为 frame 与 blob 的 canonical 形状
#   不同，面积排序才会在输出里真正可见，而且 3-sibling 样本能让排序信号更清楚。
# - 生成器在成功样本之间交替追求两种稳定可存活的信号模板：`horizontal + 3 siblings`
#   与 `diagonal + 2 siblings`。前者强化面积排序，后者覆盖另一种原始布局分支；同时至少一个带
#   frame child 的 sibling group 必须满足“输入不是 vertical”且“输入顺序不是面积降序”，否则
#   该样本会被丢弃，防止训练样本退化成仅剩 D14 或原题行为。


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.area = int(coords.shape[0])
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


def _node_sort_key(node: _Node) -> tuple[float, float, float, int, int]:
    return (-float(node.area), float(node.center[0]), float(node.center[1]), int(node.bbox[0]), int(node.bbox[1]))


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


def _layout_group(nodes: list[_Node]) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node)

    ordered = sorted(nodes, key=_node_sort_key)
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
    Convert a rough nested frame-and-blob sketch into a canonical border-and-dot diagram with vertically stacked, area-sorted siblings.

    Input structure:
    - The grid uses one background color and one foreground color.
    - Foreground components are rough closed frames or compact filled blobs.
    - Frames may contain other frames/blobs, and one extra blob may sit outside the outermost frame.

    Rule:
    - Recover the containment tree by checking which components lie inside holes of frame components.
    - Replace each blob with a single foreground pixel and each frame with a rectangular 1-pixel border padded two cells around its direct children.
    - For every sibling group inside a frame, ignore the original spatial arrangement, sort children by original component area from large to small, and stack the canonical children from top to bottom with one blank row between adjacent objects.
    - Keep an outside blob on the side of the outer frame where it originally appeared.

    Key insight: The hand-drawn geometry only matters for enclosure and for identifying an outside blob; once siblings share a parent frame, their clean output order comes from pixel counts, and their layout is always vertical.
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


# --------------- generator ---------------

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
BLOB_DISTINCT_SIZES = (4, 6, 8, 10, 11)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (1, 2)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 4)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 2
ANTI_AREA_ORDER_PROB = 0.88
NON_VERTICAL_MODE_PROB = 0.88

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 5)

# Tree structure probabilities
DEPTH1_PROB = 0.0
OUTSIDE_BLOB_PROB = 0.35
DEPTH1_INSIDE_RANGE = (2, 3)
DEPTH2_INNER_BLOBS = (1, 1)
DEPTH2_OUTER_BLOBS = (1, 2)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 450

# Successful examples alternate between two viable signal templates so both
# input-layout branches are covered while some samples still expose 3 siblings.
SIGNAL_SAMPLE_PLAN = (
    ("horizontal", 2),
    ("diagonal", 1),
)
SIGNAL_LAYOUT_MODES = tuple(mode for mode, _ in SIGNAL_SAMPLE_PLAN)
_signal_layout_cursor = 0
_preferred_signal_layout_mode, _preferred_outer_blob_count = SIGNAL_SAMPLE_PLAN[0]


class _Obj:
    def __init__(self, kind, children=None, outside_side=None, target_area=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side
        self.target_area = target_area
        self.layout_mode = "single"
        self.input_order: list[_Obj] = []


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


def _blob_mask(size=None):
    size = random.randint(*BLOB_SIZE_RANGE) if size is None else int(size)
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


def _obj_area(obj):
    assert obj.mask is not None
    return int(np.count_nonzero(obj.mask))


def _arrange(children, compact=False):
    ordered = list(children)
    if len(ordered) > 1:
        if random.random() < ANTI_AREA_ORDER_PROB:
            ordered.sort(key=lambda child: (_obj_area(child), child.mask.shape[0], child.mask.shape[1]))
        else:
            random.shuffle(ordered)

    dims = [child.mask.shape for child in ordered]
    k = len(ordered)
    if k == 1:
        return {ordered[0]: (0, 0)}, dims[0][0], dims[0][1], "single", ordered

    contains_frame = any(child.kind == "frame" for child in ordered)
    non_vertical_prob = NON_VERTICAL_MODE_PROB if contains_frame or not compact else 0.2
    if random.random() < non_vertical_prob:
        if contains_frame:
            mode = _preferred_signal_layout_mode
        else:
            mode = random.choice(SIGNAL_LAYOUT_MODES)
    else:
        mode = "vertical"

    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
    placements = {}

    if mode == "vertical":
        group_h = sum(h for h, _ in dims) + sum(gaps)
        group_w = max(w for _, w in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        r = 0
        for child, (h, w), gap in zip(ordered, dims, gaps + [0]):
            placements[child] = (r, random.randint(0, group_w - w))
            r += h + gap
        return placements, group_h, group_w, mode, ordered

    if mode == "horizontal":
        max_h = max(h for h, _ in dims)
        extra = max(random.randint(*ARRANGE_EXTRA_SPAN), k - 1)
        row_slots = sorted(random.sample(range(extra + 1), k))
        group_h = max_h + extra
        group_w = sum(w for _, w in dims) + sum(gaps)
        c = 0
        for child, (h, w), gap, slot in zip(ordered, dims, gaps + [0], row_slots):
            placements[child] = (slot, c)
            c += w + gap
        return placements, group_h, group_w, mode, ordered

    group_h = sum(h for h, _ in dims) + sum(gaps)
    group_w = sum(w for _, w in dims) + sum(gaps)
    r = c = 0
    for child, (h, w), gap in zip(ordered, dims, gaps + [0]):
        placements[child] = (r, c)
        r += h + gap
        c += w + gap
    return placements, group_h, group_w, mode, ordered


def _prepare(node, compact=False):
    if node.kind == "blob":
        node.mask = _blob_mask(node.target_area)
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    placements, group_h, group_w, mode, ordered = _arrange(node.children, compact=tight)
    node.layout_mode = mode
    node.input_order = list(ordered)
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


def _make_blob_group(n):
    sizes = random.sample(BLOB_DISTINCT_SIZES, n)
    return [_Obj("blob", target_area=size) for size in sizes]


def _make_tree():
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")
    if random.random() < DEPTH1_PROB:
        n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
        root.children = _make_blob_group(n_inside)
        if n_inside < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
            outside = _Obj("blob", outside_side=random.choice(sides), target_area=random.choice(BLOB_DISTINCT_SIZES))
    else:
        inner = _Obj("frame")
        inner.children = _make_blob_group(random.randint(*DEPTH2_INNER_BLOBS))
        outer_blob_count = _preferred_outer_blob_count
        outer_blobs = _make_blob_group(outer_blob_count)
        root.children = [inner] + outer_blobs
        if len(inner.children) + len(outer_blobs) < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
            outside = _Obj("blob", outside_side=random.choice(sides), target_area=random.choice(BLOB_DISTINCT_SIZES))
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


def _collect_group_meta(node):
    out = []
    if node.kind != "frame":
        return out
    if len(node.children) >= 2:
        input_areas = [_obj_area(child) for child in node.input_order]
        desc_areas = sorted(input_areas, reverse=True)
        out.append(
            {
                "layout_mode": node.layout_mode,
                "input_areas": input_areas,
                "output_areas": desc_areas,
                "anti_area": input_areas != desc_areas,
                "non_vertical": node.layout_mode in {"horizontal", "diagonal"},
                "has_frame_child": any(child.kind == "frame" for child in node.children),
            }
        )
    for child in node.children:
        out.extend(_collect_group_meta(child))
    return out


def _has_combo_signal(root):
    groups = _collect_group_meta(root)
    return any(group["has_frame_child"] and group["anti_area"] and group["non_vertical"] for group in groups)


def _gen_attempt(with_meta=False):
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree()
    if not _prepare(root):
        return None
    if outside is not None:
        if not _prepare(outside):
            return None
        _place_outside(root, outside)

    if not _has_combo_signal(root):
        return None

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
    if with_meta:
        return grid, {"groups": _collect_group_meta(root), "outside": outside is not None}
    return grid


def generate_input():
    global _signal_layout_cursor, _preferred_signal_layout_mode, _preferred_outer_blob_count
    _preferred_signal_layout_mode, _preferred_outer_blob_count = SIGNAL_SAMPLE_PLAN[
        _signal_layout_cursor % len(SIGNAL_SAMPLE_PLAN)
    ]
    _signal_layout_cursor += 1
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
