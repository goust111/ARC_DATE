# Combo variant of 2d0172a1 (index 17)
# combo_spec: D11/0+D17/2
# Name: 3层嵌套+框架宽高比编码子节点数
#
# Design analysis:
#   variant_spec:
#     core_change:
#       - Keep the original parsing problem: rough single-color closed loops are frames, solid clusters are blobs,
#         and containment is recovered from frame holes.
#       - Combine two changes at once:
#         1. The input tree is always a 3-frame chain: outer -> middle -> inner.
#         2. In the canonical output, each frame's outer rectangle uses a direct-child-count aspect code:
#            1 child -> 1:1, 2 children -> 2:1, 3 children -> 3:1.
#     concretization_and_trace:
#       - Concrete fused example:
#         outer.children = [middle, blob_A]
#         middle.children = [inner]
#         inner.children = [blob_B, blob_C, blob_D]
#       - Original task:
#         * generator could stop at depth 1 or 2;
#         * canonical frame size was always "tight child box + fixed 4-cell padding".
#       - Fused task:
#         * input always shows three rough nested loops;
#         * solver still rebuilds the same containment tree from holes;
#         * inner is canonicalized first as a 3:1 rectangle because it has 3 direct children;
#         * middle becomes a square because it has only 1 direct child;
#         * outer becomes a 2:1 rectangle because it directly contains [middle, blob_A].
#       - Difference:
#         the recursion depth is now always visible, and the rectangle proportions now carry local counting
#         information that was absent from the original output.
#     complexity_calibration:
#       - The count code uses only three discrete states {1,2,3}, which is small enough to infer from 2-4 ARC
#         pairs yet rich enough to be visually obvious.
#       - The generator uses a small library of blob distributions so samples repeatedly expose count=2 and
#         count=3 at different depths instead of collapsing to unary frames.
#       - Layout modes for multi-child frames are biased toward compact horizontal arrangements; this keeps
#         the 3-layer recursion legible and avoids giant tall boxes whose ratio code would dominate the canvas.
#     implicit_assumptions_review:
#       - `_component_holes` still assumes each frame is a single 4-connected loop with exactly one hole; the
#         generator preserves that invariant.
#       - Containment assignment via "choose the smallest containing frame" becomes more important at depth 3,
#         but it remains correct because inner lies inside both outer and middle while middle is the smaller box.
#       - `_build_layout` can no longer assume "frame = child box + fixed padding"; that would erase D17/2.
#         It now assumes every frame has 1-3 direct children so the ratio lookup is defined.
#       - `_layout_root` still relies on there being exactly one root frame plus optional outside blobs. The
#         fused generator maintains that structure.
#     placement_legality:
#       - Solver-side legality: choose the smallest exact-ratio outer rectangle whose interior still leaves at
#         least one background cell between border and canonical children.
#       - Generator-side legality: outer/middle/inner direct-child counts are kept in 1..3; child masks are
#         recursively arranged before enclosing frames are sampled; outside blobs are only placed when the open
#         side interval is large enough.
#       - Grid size is expanded to 25-30 so three rough loops fit without overlaps.
#     visual_observability:
#       - A human can count direct children in the input at each level and compare them with the output frame
#         shapes: unary frames are square, binary frames are 2:1, ternary frames are 3:1.
#       - Because every sample has 3 nested frames, the ratio rule is demonstrated repeatedly inside one grid
#         instead of appearing only once.
#       - No hidden signals are used: only visible nesting depth, visible child multiplicity, and visible geometry.
#     cleanliness:
#       - The palette stays at exactly two colors, so the new reasoning burden comes only from structure.
#       - Blob counts are capped and padding is tightened so nested regions remain visually separable.
#       - Outside blobs remain rare, preventing the perimeter of the outer frame from becoming cluttered.
#     new_solver_assumptions_to_match_in_generator:
#       - Every frame must have at least one direct child, and at most three.
#       - At least one frame in each generated sample should usually have 2 or 3 direct children; otherwise the
#         ratio rule would be weakly supported and the task could visually collapse toward the depth-only variant.
#       - Multi-child layouts should stay compact enough that their ratio-coded canonical rectangles remain
#         interpretable instead of exploding in height.

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


CHILD_RATIO = {1: 1, 2: 2, 3: 3}


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


def _ratio_frame_shape(child_h: int, child_w: int, n_children: int) -> tuple[int, int, int, int]:
    ratio = CHILD_RATIO[n_children]
    unit = max(child_h + 4, (child_w + 4 + ratio - 1) // ratio)
    out_h = int(unit)
    out_w = int(ratio * unit)
    top = 2 + (out_h - child_h - 4) // 2
    left = 2 + (out_w - child_w - 4) // 2
    return out_h, out_w, int(top), int(left)


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

    assert 1 <= len(node.children) <= 3
    child_mask, _placements, axis_map = _layout_group(node.children)
    child_h, child_w = child_mask.shape
    out_h, out_w, top, left = _ratio_frame_shape(child_h, child_w, len(node.children))

    out = np.zeros((out_h, out_w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    out[top : top + child_h, left : left + child_w] |= child_mask

    node.mask = out
    node.anchor = (out.shape[0] // 2, out.shape[1] // 2)
    node.axis_map = {
        "row": [(orig, pos + top) for orig, pos in axis_map["row"]],
        "col": [(orig, pos + left) for orig, pos in axis_map["col"]],
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
    Canonicalize a rough three-level nested frame sketch into clean rectangles and dots with count-coded frame proportions.

    Input structure:
    - The grid uses exactly one background color and one foreground color.
    - Foreground components are either small solid blobs or rough single-loop frames with one hole.
    - The containment tree is always a 3-frame chain (outer contains middle contains inner), with optional sibling blobs at any of those three levels and an optional blob outside the outer frame.

    Rule:
    - Recover the containment tree by checking which frame-hole contains each foreground component.
    - Convert every blob into one foreground pixel.
    - Convert every frame into a 1-pixel rectangular border whose outer width:height ratio is determined by its number of direct children: 1 -> 1:1, 2 -> 2:1, 3 -> 3:1.
    - Preserve sibling row/column ordering, keep one background cell between canonical siblings, and preserve the side relation of any outside blob.

    Key insight:
    - The rough loop geometry is noise; the durable information is the three-step enclosure chain plus each frame's direct child count, which is re-expressed in the output as square / 2:1 / 3:1 rectangles.
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


# --------------- generator (combo) ---------------

# Grid / canvas bounds
GRID_MIN = 25
GRID_MAX = 30
MAX_EXTRA_PAD = 8
MAX_PAD_PER_SIDE = 4

# Frame geometry
FRAME_H_RANGE = (7, 27)
FRAME_W_RANGE = (7, 30)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

# Blob geometry
BLOB_SIZE_RANGE = (4, 9)
BLOB_BBOX_RANGE = (2, 5)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (1, 2)
ARRANGE_GAP_NORMAL = (2, 3)
ARRANGE_EXTRA_SPAN = (0, 1)
FRAME_MARGIN_RANGE = (2, 4)
COMPACT_MARGIN = 2

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 3)

# Tree structure patterns:
# tuple = (outer_direct_children, middle_direct_children, inner_direct_children)
# Stay within the geometry combinations that generate reliably, but cycle them in an order that
# frontloads visible 3:1 cases at different depths instead of letting them appear too rarely.
DIRECT_COUNT_SCHEDULE = [
    (1, 3, 1),
    (3, 1, 1),
    (2, 2, 1),
    (1, 2, 1),
    (2, 1, 1),
    (1, 2, 2),
    (1, 3, 1),
    (3, 1, 1),
    (2, 2, 1),
    (1, 1, 3),
]
OUTSIDE_BLOB_PROB = 0.16

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 1000


_direct_count_index = 0


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


def _loop_ok(mask):
    """Check that mask is a single 4-connected simple closed curve with exactly one hole."""
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
    """Interleave n_a copies of step_a with n_b copies of step_b, biased toward alternation."""
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
    """Extend path with a straight line from its last point to end."""
    r, c = path[-1]
    er, ec = end
    dr = 0 if er == r else (1 if er > r else -1)
    dc = 0 if ec == c else (1 if ec > c else -1)
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))


def _add_steps(path, steps):
    """Extend path by applying a sequence of (dr, dc) deltas."""
    r, c = path[-1]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))


def _turn_ratio(path):
    """Fraction of direction changes along a cyclic path."""
    dirs = []
    cyc = path + [path[0]]
    for (r0, c0), (r1, c1) in zip(cyc, cyc[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = sum(1 for a, b in zip(dirs, dirs[1:] + [dirs[0]]) if a != b)
    return turns / len(path)


def _frame_mask(h, w):
    """Generate a wobbly simple closed curve mask of size (h, w)."""
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
    """Generate a small filled blob mask."""
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
    """Arrange children with a bias toward low-height layouts for multi-child frames."""
    children = list(children)
    random.shuffle(children)
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
    placements = {}

    if k == 2:
        mode = "horizontal" if compact or random.random() < 0.8 else "diagonal"
    else:
        mode = "horizontal" if compact or random.random() < 0.9 else "diagonal"

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
    """Recursively build masks and place children within frames."""
    if node.kind == "blob":
        node.mask = _blob_mask()
        return True

    assert 1 <= len(node.children) <= 3
    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child or len(node.children) >= 2
    placements, group_h, group_w = _arrange(node.children, compact=tight)
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False

    if tight:
        h_hi = min(FRAME_H_RANGE[1], min_h + 1)
        w_hi = min(FRAME_W_RANGE[1], min_w + 2)
    else:
        h_hi = FRAME_H_RANGE[1]
        w_hi = FRAME_W_RANGE[1]
    h = random.randint(min_h, h_hi)
    w = random.randint(min_w, w_hi)
    top = margin + random.randint(0, h - 2 * margin - group_h)
    left = margin + random.randint(0, w - 2 * margin - group_w)
    for child, (r, c) in placements.items():
        child.rel = (top + r, left + c)
    node.mask = _frame_mask(h, w)
    return node.mask is not None


def _walk(node, base=(0, 0)):
    """Flatten tree into list, computing absolute positions."""
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _place_outside(root, outside):
    """Position an outside blob adjacent to the root frame."""
    assert root.mask is not None and outside.mask is not None
    h, w = root.mask.shape
    bh, bw = outside.mask.shape
    gap = random.randint(*OUTSIDE_GAP_RANGE)
    side = outside.outside_side

    if side in {"right", "left"}:
        row_hi = h - bh - 1
        if row_hi < 1:
            return False
        row = random.randint(1, row_hi)
        col = w + gap if side == "right" else -gap - bw
    else:
        col_hi = w - bw - 1
        if col_hi < 1:
            return False
        row = h + gap if side == "bottom" else -gap - bh
        col = random.randint(1, col_hi)

    outside.rel = (row, col)
    return True


def _node_from_obj(obj):
    """Convert a generator _Obj (with absolute pos) to a solver _Node."""
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
    """Compute expected solver output for the generated tree."""

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


def _next_direct_counts():
    """Cycle through an observability-biased schedule of direct-child-count triples."""
    global _direct_count_index
    counts = DIRECT_COUNT_SCHEDULE[_direct_count_index]
    _direct_count_index = (_direct_count_index + 1) % len(DIRECT_COUNT_SCHEDULE)
    return counts


def _make_tree():
    """Build a fixed-depth-3 containment tree with count-bearing direct-child patterns."""
    outer = _Obj("frame")
    middle = _Obj("frame")
    inner = _Obj("frame")

    outer_count, middle_count, inner_count = _next_direct_counts()
    outer_extra = outer_count - 1
    middle_extra = middle_count - 1
    inner.children = [_Obj("blob") for _ in range(inner_count)]
    middle.children = [inner] + [_Obj("blob") for _ in range(middle_extra)]
    outer.children = [middle] + [_Obj("blob") for _ in range(outer_extra)]

    outside = None
    total_inside_blobs = outer_extra + middle_extra + inner_count
    if total_inside_blobs <= 4 and random.random() < OUTSIDE_BLOB_PROB:
        outside = _Obj("blob", outside_side=random.choice(("left", "right", "top", "bottom")))
    return outer, outside


def _draw(objects, bg, fg):
    """Render all objects onto a padded grid. Returns None if size constraints fail."""
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


def _gen_attempt():
    """Single generation attempt: build tree, render input, verify against solver."""
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree()
    if not _prepare(root):
        return None
    if outside is not None:
        if not _prepare(outside):
            return None
        if not _place_outside(root, outside):
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
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
