# Variant of 2d0172a1
# Dimension: D11 — 树结构：嵌套深度 (meta)
# Change: 嵌套深度1-2（1层框架或2层嵌套框架），总共1-2个框架，1-4个色块... → 多棵独立树（多个根框架）

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the D11 variant
# spec:
#   core_change:
#     - Replace the original single-root tree (+ optional outside blob) with a forest
#       of 2-3 independent root frames. Each root frame owns its own local subtree.
#   concrete_behavior:
#     - Input now shows 2-3 disjoint rough frames that do not contain one another.
#     - Each root frame has at least one child; a root may be depth-1 (blobs only)
#       or depth-2 (an inner frame, optionally plus sibling blobs).
#     - Output canonicalizes each tree exactly as before inside the tree, but the
#       forest level is laid out separately with a wider gap between root trees.
#   trace_example:
#     - Original critical path: detect components -> build one containment tree ->
#       canonicalize siblings with the same 1-cell gap everywhere.
#     - Variant critical path: detect components -> build a containment forest ->
#       canonicalize children inside each tree with the old gap, then lay out the
#       2-3 root trees with a larger forest gap. The changed step is root layout.
#   complexity_calibration:
#     - Root count is restricted to 2-3, not arbitrary many, so the global pattern
#       is visible and learnable from few samples.
#     - Each tree still uses only depth 1-2 and small blob/frame counts, so the new
#       difficulty is "multiple independent hierarchies" rather than unrelated noise.
#     - Root-level outside blobs are removed: they blur the forest boundary and are
#       not needed for the D11 change.
#   implicit_assumptions_checked:
#     - Foreground remains monochrome, so component extraction and hole detection are
#       unchanged.
#     - `_build_layout()` still assumes every frame has direct children; generator
#       guarantees this by construction.
#     - Parent assignment still relies on "smallest containing frame"; this remains
#       valid because root frames are explicitly generated as non-overlapping.
#     - The old `_layout_root()` special case for one frame plus root blobs is no
#       longer the relevant behavior. The new solver assumes all roots are frames and
#       there are 2-3 of them; generator enforces that.
#   placement_legality:
#     - Children are placed entirely inside their parent frame masks.
#     - Root trees are arranged by disjoint root-frame bounding boxes, so different
#       trees cannot overlap or accidentally become nested.
#     - Grid sizing and padding are checked after placement; failed attempts retry.
#   visual_observability:
#     - Humans can see several separate rough loops in the input and several separate
#       canonical rectangles in the output.
#     - The wider gap between independent root trees makes the forest structure
#       visually distinct from ordinary sibling spacing inside one frame.
#   color_reasoning:
#     - No color remapping is introduced; colors remain purely fg/background.


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


def _axis_coordinates(nodes: list[_Node], axis: int, band_gap: int = 2) -> dict[_Node, int]:
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
            coord += prev_down + band_gap + curr_up
        for item in band["items"]:
            coord_by_node[item[0]] = int(coord)

    return coord_by_node


def _layout_group(
    nodes: list[_Node], band_gap: int = 2
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node)

    row_coord = _axis_coordinates(nodes, axis=0, band_gap=band_gap)
    col_coord = _axis_coordinates(nodes, axis=1, band_gap=band_gap)

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


FOREST_BAND_GAP = 3


def _layout_root(nodes: list[_Node]) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node)

    assert all(node.is_frame for node in nodes)
    assert 2 <= len(nodes) <= 3
    mask, _placements, _axis_map = _layout_group(nodes, band_gap=FOREST_BAND_GAP)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn a rough forest of nested loops into a clean forest of rectangular frames and dots.

    Input structure:
    - The grid uses one background color and one foreground color.
    - Foreground 4-connected components are rough closed frames or small filled blobs.
    - There are 2-3 separate root frames; each root has its own contained children, and root frames do not contain one another.

    Rule:
    - Recover the containment forest by assigning every component to the smallest frame hole that fully contains it.
    - Convert each blob to one foreground pixel and each frame to a rectangular border with a one-cell moat around its direct children.
    - Canonicalize each root tree independently, then arrange the independent root frames by their original spatial row/column relations, using a wider gap between trees than inside a tree.

    Key insight: The hand-drawn geometry is disposable; what matters is the forest structure, local nesting inside each root frame, and the coarse global order of the separate root trees.
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
    assert all(node.is_frame for node in roots)
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


# --------------- generator (o5x) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────────────
# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

# Frame geometry
FRAME_H_RANGE = (6, 12)
FRAME_W_RANGE = (6, 12)
FRAME_CORNER_MAX = 4
FRAME_MIN_EDGE = 1
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

# Blob geometry
BLOB_SIZE_RANGE = (4, 7)
BLOB_BBOX_RANGE = (2, 4)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (1, 2)
ARRANGE_GAP_NORMAL = (1, 3)
ARRANGE_EXTRA_SPAN = (0, 2)
FRAME_MARGIN_RANGE = (2, 3)
COMPACT_MARGIN = 2
ROOT_ARRANGE_GAP = (2, 4)

# Forest structure
ROOT_COUNT_WEIGHTS = {2: 4, 3: 1}
ROOT_DEPTH2_WEIGHTS = {
    2: (2, 4, 1),  # 0,1,2 deep roots
    3: (3, 4, 0, 0),  # 0,1,2,3 deep roots
}
DEPTH1_INSIDE_RANGE = (1, 2)
DEPTH2_INNER_BLOBS = (1, 1)
DEPTH2_OUTER_BLOBS = (0, 1)

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 500
# ────────────────────────────────────────────────────────────────────────────


class _Obj:
    def __init__(self, kind, children=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)


def _hole_mask(mask):
    """Return the unique interior hole of a frame mask, or None if invalid."""
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    holes = [(lab == k) for k in range(1, n + 1) if k not in boundary]
    if len(holes) != 1:
        return None
    return holes[0]


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
    """Generate a small filled blob mask (size 4-11, bbox 2x2 to 4x4)."""
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
    """Arrange children in a random layout (vertical, horizontal, or diagonal)."""
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


def _arrange_roots(roots):
    """Arrange root frames as disjoint trees on the global canvas."""

    def extent(placements):
        max_r = max(r + root.mask.shape[0] for root, (r, c) in placements.items())
        max_c = max(c + root.mask.shape[1] for root, (r, c) in placements.items())
        return max_r, max_c

    def maybe_add(candidates, placements):
        h, w = extent(placements)
        if h <= GRID_MAX and w <= GRID_MAX:
            candidates.append(placements)

    ordered = list(roots)
    random.shuffle(ordered)
    dims = {root: root.mask.shape for root in ordered}
    candidates = []

    if len(ordered) == 2:
        a, b = ordered
        ha, wa = dims[a]
        hb, wb = dims[b]
        for _ in range(8):
            gap = random.randint(*ROOT_ARRANGE_GAP)
            top_h = max(ha, hb)
            maybe_add(
                candidates,
                {
                    a: (random.randint(0, top_h - ha), 0),
                    b: (random.randint(0, top_h - hb), wa + gap),
                },
            )
            left_w = max(wa, wb)
            maybe_add(
                candidates,
                {
                    a: (0, random.randint(0, left_w - wa)),
                    b: (ha + gap, random.randint(0, left_w - wb)),
                },
            )
            maybe_add(candidates, {a: (0, 0), b: (ha + gap, wa + gap)})
    else:
        a, b, c = ordered
        ha, wa = dims[a]
        hb, wb = dims[b]
        hc, wc = dims[c]
        for _ in range(12):
            gap_x = random.randint(*ROOT_ARRANGE_GAP)
            gap_y = random.randint(*ROOT_ARRANGE_GAP)

            top_h = max(ha, hb)
            top_w = wa + gap_x + wb
            row_offsets = {
                a: random.randint(0, top_h - ha),
                b: random.randint(0, top_h - hb),
            }
            bottom_c = random.randint(0, max(0, top_w - wc))
            maybe_add(
                candidates,
                {
                    a: (row_offsets[a], 0),
                    b: (row_offsets[b], wa + gap_x),
                    c: (top_h + gap_y, bottom_c),
                },
            )

            left_w = max(wa, wb)
            left_h = ha + gap_y + hb
            col_offsets = {
                a: random.randint(0, left_w - wa),
                b: random.randint(0, left_w - wb),
            }
            right_r = random.randint(0, max(0, left_h - hc))
            maybe_add(
                candidates,
                {
                    a: (0, col_offsets[a]),
                    b: (ha + gap_y, col_offsets[b]),
                    c: (right_r, left_w + gap_x),
                },
            )

            mid_c = random.randint(0, max(0, wa + gap_x))
            maybe_add(
                candidates,
                {
                    a: (0, 0),
                    b: (random.randint(0, max(0, ha - hb)), wa + gap_x),
                    c: (top_h + gap_y, mid_c),
                },
            )

    if not candidates:
        return None
    return random.choice(candidates)


def _prepare(node, compact=False):
    """Recursively build masks and place children within frames."""
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
    if node.mask is None:
        return False
    hole = _hole_mask(node.mask)
    if hole is None:
        return False
    occupancy = np.zeros_like(node.mask, dtype=bool)
    for child in node.children:
        rr, cc = child.rel
        ch, cw = child.mask.shape
        if rr < 0 or cc < 0 or rr + ch > hole.shape[0] or cc + cw > hole.shape[1]:
            return False
        if not np.all(hole[rr : rr + ch, cc : cc + cw][child.mask]):
            return False
        region = occupancy[rr : rr + ch, cc : cc + cw]
        if np.any(region & child.mask):
            return False
        region |= child.mask
    _, n = label_mask(occupancy, connectivity=4)
    if n != len(node.children):
        return False
    return True


def _walk(node, base=(0, 0)):
    """Flatten tree into list, computing absolute positions."""
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


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


def _expected_output(roots, bg, fg):
    """Compute expected solver output for the generated forest."""

    def add(obj):
        node = _node_from_obj(obj)
        for child in obj.children:
            child_node = add(child)
            child_node.parent = node
            node.children.append(child_node)
        return node

    forest = [add(root) for root in roots]
    mask = _layout_root(forest)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _make_tree(depth2):
    """Build one root tree: depth-1 blobs or depth-2 with one inner frame."""
    root = _Obj("frame")
    if not depth2:
        root.children = [_Obj("blob") for _ in range(random.randint(*DEPTH1_INSIDE_RANGE))]
        return root

    inner = _Obj("frame")
    inner.children = [_Obj("blob") for _ in range(random.randint(*DEPTH2_INNER_BLOBS))]
    outer_blobs = [_Obj("blob") for _ in range(random.randint(*DEPTH2_OUTER_BLOBS))]
    root.children = [inner] + outer_blobs
    return root


def _make_forest():
    """Build 2-3 independent root trees with mixed depths."""
    root_count = random.choices(list(ROOT_COUNT_WEIGHTS), weights=list(ROOT_COUNT_WEIGHTS.values()), k=1)[0]
    weights = ROOT_DEPTH2_WEIGHTS[root_count]
    deep_count = random.choices(range(root_count + 1), weights=weights, k=1)[0]
    depth_flags = [True] * deep_count + [False] * (root_count - deep_count)
    random.shuffle(depth_flags)
    return [_make_tree(flag) for flag in depth_flags]


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
    """Single generation attempt: build forest, render input, verify against solver."""
    bg, fg = random.sample(range(10), 2)
    roots = _make_forest()
    for root in roots:
        if not _prepare(root):
            return None

    placements = _arrange_roots(roots)
    if placements is None:
        return None
    for root, pos in placements.items():
        root.rel = pos

    objects = []
    for root in roots:
        objects.extend(_walk(root))

    grid = _draw(objects, bg, fg)
    if grid is None:
        return None
    expected = _expected_output(roots, bg, fg)
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
    raise RuntimeError("failed to generate valid 2d0172a1 D11 variant input")
