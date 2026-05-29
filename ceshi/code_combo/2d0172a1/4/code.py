# Combo variant of 2d0172a1 (index 4)
# combo_spec: D11/2+D14/1
# Name: 多棵独立树+网格化排列

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    Original task: one root tree, and every sibling set is canonicalized by row/col bands,
#    so four blobs could still become a 1x4 or diagonal chain.
#    Combo task: there are 2-3 independent root frames, and every root tree contains a
#    frame with exactly four direct children. Example:
#    - Root A: one frame containing four blobs drawn as a rough vertical line.
#    - Root B: another separate frame containing four blobs drawn as a rough horizontal line.
#    Original logic would output two independent trees but keep both four-child groups linear.
#    Combo logic first recovers the forest, then gridifies each four-child sibling set into a
#    2x2 near-square block while still arranging the separate root trees by coarse global position.
# b) Complexity calibration:
#    Root count is limited to 2-3, but the generator strongly biases toward 2 roots because
#    several four-child groups are already spatially demanding in a 25x25 canvas. Trees are kept
#    depth-1 on purpose: the fusion target here is "multiple independent roots + sibling tiling",
#    not extra nesting variety.
#    The grid trigger is fixed at four children. That keeps the forest/global change visible
#    without making the sample space too unconstrained.
#    Using repeated four-child groups gives the learner repeated evidence for the grid rule.
# c) Implicit-assumption audit:
#    - The old `_layout_group()` assumed every sibling set could be read from overlap bands.
#      That fails for this combo, so large sibling sets bypass band logic and use explicit tiling.
#    - The old root handling allowed root blobs around one frame; this combo removes them.
#      The solver now assumes all roots are frames, and the generator guarantees that.
#    - Parent assignment by smallest containing frame remains valid because generated root
#      frames are disjoint and no root contains another.
# d) Placement legality:
#    Child masks are checked against the actual hole mask of each wobbly frame, and occupancy
#    is validated for non-overlap. Root trees are placed as disjoint frame bounding boxes.
#    Failed placements are discarded and regenerated.
# e) Visual observability:
#    In a few examples, the input visibly contains several separate rough loops, and each tree
#    also visibly contains a four-object sibling set. The output then shows both changes at once:
#    multiple canonical root rectangles and 2x2 dot grids inside trees.
# f) Visual cleanliness:
#    Frame/blob sizes are tightened compared with the original task so that 2-3 trees plus
#    repeated four-child groups still fit within a 25x25 grid without cluttering.
# g) Color reasoning:
#    No color remapping is introduced; only one fg color and one bg color are used.
# New solver assumptions and generator guarantees:
#    - Any frame with >=4 direct children should have a stable reading order under sorting by
#      center row, then center col within row chunks. Generator enforces this by arranging such
#      groups as clean vertical or horizontal rough layouts.
#    - All roots are frames. The solver handles 2-3 roots; successful generated samples are
#      heavily biased toward 2 because that is the cleanest way to keep multiple 4-child groups
#      legible inside a 25x25 canvas.


GRID_LAYOUT_TRIGGER = 4
CANON_GAP = 1
FOREST_BAND_GAP = 3


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


def _near_square_shape(count: int) -> tuple[int, int]:
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    return rows, cols


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


def _layout_group_linear(
    nodes: list[_Node], band_gap: int = 2
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
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


def _layout_group_grid(
    nodes: list[_Node],
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    rows, cols = _near_square_shape(len(nodes))
    ordered = sorted(nodes, key=lambda node: (float(node.center[0]), float(node.center[1])))
    row_chunks = [ordered[i * cols : (i + 1) * cols] for i in range(rows)]
    row_chunks = [sorted(chunk, key=lambda node: float(node.center[1])) for chunk in row_chunks if chunk]

    row_heights = [max(node.mask.shape[0] for node in chunk) for chunk in row_chunks]
    col_widths = [0] * cols
    for chunk in row_chunks:
        for j, node in enumerate(chunk):
            col_widths[j] = max(col_widths[j], int(node.mask.shape[1]))

    total_h = sum(row_heights) + CANON_GAP * (len(row_chunks) - 1)
    total_w = sum(col_widths) + CANON_GAP * (cols - 1)
    out = np.zeros((total_h, total_w), dtype=bool)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}

    row_start = 0
    for i, chunk in enumerate(row_chunks):
        col_start = 0
        for j, node in enumerate(chunk):
            assert node.mask is not None and node.anchor is not None
            r = row_start + (row_heights[i] - node.mask.shape[0]) // 2
            c = col_start + (col_widths[j] - node.mask.shape[1]) // 2
            placements[node] = (int(r), int(c))
            out[r : r + node.mask.shape[0], c : c + node.mask.shape[1]] |= node.mask
            axis_map["row"].append((float(node.center[0]), int(r + node.anchor[0])))
            axis_map["col"].append((float(node.center[1]), int(c + node.anchor[1])))
            col_start += col_widths[j] + CANON_GAP
        row_start += row_heights[i] + CANON_GAP

    return out, placements, axis_map


def _layout_group(
    nodes: list[_Node], band_gap: int = 2
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node)

    if len(nodes) >= GRID_LAYOUT_TRIGGER:
        return _layout_group_grid(nodes)
    return _layout_group_linear(nodes, band_gap=band_gap)


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
    Turn a rough forest of enclosure sketches into a clean forest of rectangular frames and dots, with grid packing inside trees.

    Input structure:
    - The grid uses exactly one background color and one foreground color.
    - Foreground 4-connected components are rough closed frames or small filled blobs.
    - There are 2-3 separate root frames that do not contain one another; inside each root tree, at least one frame has several sibling children.

    Rule:
    - Recover the containment forest by assigning each component to the smallest frame hole that fully contains it.
    - Convert every blob to a single foreground pixel and every frame to a rectangular border with a one-cell moat around its direct children.
    - Keep small sibling groups on the original band-based layout logic, but whenever a frame has 4 or more direct children, reorder them by coarse reading order and pack them into the smallest near-square grid; then arrange the independent root trees by their original spatial row/column relations.

    Key insight: The exact wobbles are irrelevant; the output is determined by two structural signals only: which components belong to which tree, and which sibling groups are large enough to collapse into compact grids.
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


# --------------- generator (o5x, combo variant) ---------------

# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 10
MAX_PAD_PER_SIDE = 6

# Frame geometry
FRAME_H_RANGE = (6, 18)
FRAME_W_RANGE = (6, 18)
FRAME_CORNER_MAX = 4
FRAME_MIN_EDGE = 1
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

# Blob geometry
BLOB_SIZE_RANGE = (3, 4)
BLOB_BBOX_RANGE = (2, 3)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (1, 2)
ARRANGE_GAP_NORMAL = (1, 3)
ARRANGE_EXTRA_SPAN = (0, 2)
TRIGGER_LINE_GAP = (1, 2)
TRIGGER_LINE_EXTRA_SPAN = 1
FRAME_MARGIN_RANGE = (2, 2)
COMPACT_MARGIN = 2
ROOT_ARRANGE_GAP = (2, 4)

# Forest structure
ROOT_COUNT_WEIGHTS = {2: 1, 3: 1}
GRID_SIBLING_COUNT = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 600


class _Obj:
    def __init__(self, kind, children=None, layout_hint=None, compact=False):
        self.kind = kind
        self.children = list(children or [])
        self.layout_hint = layout_hint
        self.compact = compact
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


def _safe_hole_mask(hole: np.ndarray) -> np.ndarray:
    """4-neighborhood erosion of the hole so children stay separated from the frame."""
    up = np.zeros_like(hole, dtype=bool)
    down = np.zeros_like(hole, dtype=bool)
    left = np.zeros_like(hole, dtype=bool)
    right = np.zeros_like(hole, dtype=bool)
    up[1:, :] = hole[:-1, :]
    down[:-1, :] = hole[1:, :]
    left[:, 1:] = hole[:, :-1]
    right[:, :-1] = hole[:, 1:]
    return hole & up & down & left & right


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


def _blob_mask(compact=False):
    """Generate a small filled blob mask."""
    if compact:
        tiny_shapes = (
            np.array([[True]], dtype=bool),
            np.array([[True, True]], dtype=bool),
            np.array([[True], [True]], dtype=bool),
            np.array([[True, False], [True, True]], dtype=bool),
        )
        return random.choice(tiny_shapes).copy()

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


def _arrange(children, compact=False, forced_mode=None):
    """Arrange rough input children; large groups stay non-grid so output gridification is visible."""
    random.shuffle(children)
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    if k >= GRID_LAYOUT_TRIGGER:
        mode = forced_mode if forced_mode in {"vertical", "horizontal"} else random.choice(("vertical", "horizontal"))
        gaps = [random.randint(*TRIGGER_LINE_GAP) for _ in range(k - 1)]
        placements = {}
        if mode == "vertical":
            group_h = sum(h for h, _ in dims) + sum(gaps)
            group_w = max(w for _, w in dims) + TRIGGER_LINE_EXTRA_SPAN
            r = 0
            for child, (h, w), gap in zip(children, dims, gaps + [0]):
                placements[child] = (r, random.randint(0, group_w - w))
                r += h + gap
            return placements, group_h, group_w
        if mode == "horizontal":
            group_h = max(h for h, _ in dims) + TRIGGER_LINE_EXTRA_SPAN
            group_w = sum(w for _, w in dims) + sum(gaps)
            c = 0
            for child, (h, w), gap in zip(children, dims, gaps + [0]):
                placements[child] = (random.randint(0, group_h - h), c)
                c += w + gap
            return placements, group_h, group_w

    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    mode = random.choice(("vertical", "horizontal", "diagonal"))
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
    """Recursively build masks and place children within frames."""
    if node.kind == "blob":
        node.mask = _blob_mask(compact=node.compact)
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child or node.compact
    placements, group_h, group_w = _arrange(node.children, compact=tight, forced_mode=node.layout_hint)
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False
    h_hi = min(FRAME_H_RANGE[1], min_h + 2) if tight else FRAME_H_RANGE[1]
    w_hi = min(FRAME_W_RANGE[1], min_w + 2) if tight else FRAME_W_RANGE[1]
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
    safe_hole = _safe_hole_mask(hole)
    occupancy = np.zeros_like(node.mask, dtype=bool)
    for child in node.children:
        rr, cc = child.rel
        ch, cw = child.mask.shape
        if rr < 0 or cc < 0 or rr + ch > safe_hole.shape[0] or cc + cw > safe_hole.shape[1]:
            return False
        if not np.all(safe_hole[rr : rr + ch, cc : cc + cw][child.mask]):
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
        for _ in range(10):
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
        for _ in range(14):
            gap_x = random.randint(*ROOT_ARRANGE_GAP)
            gap_y = random.randint(*ROOT_ARRANGE_GAP)

            top_h = max(ha, hb)
            top_w = wa + gap_x + wb
            bottom_c = random.randint(0, max(0, top_w - wc))
            maybe_add(
                candidates,
                {
                    a: (random.randint(0, top_h - ha), 0),
                    b: (random.randint(0, top_h - hb), wa + gap_x),
                    c: (top_h + gap_y, bottom_c),
                },
            )

            left_w = max(wa, wb)
            left_h = ha + gap_y + hb
            right_r = random.randint(0, max(0, left_h - hc))
            maybe_add(
                candidates,
                {
                    a: (0, random.randint(0, left_w - wa)),
                    b: (ha + gap_y, random.randint(0, left_w - wb)),
                    c: (right_r, left_w + gap_x),
                },
            )

            maybe_add(
                candidates,
                {
                    a: (0, 0),
                    b: (random.randint(0, max(0, ha - hb)), wa + gap_x),
                    c: (top_h + gap_y, random.randint(0, max(0, wa + gap_x - wc))),
                },
            )

    if not candidates:
        return None
    return random.choice(candidates)


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


def _make_tree(layout_hint=None, child_count=GRID_SIBLING_COUNT, compact=False):
    """Build one root tree."""
    root = _Obj("frame", layout_hint=layout_hint, compact=compact)
    root.children = [_Obj("blob", compact=compact) for _ in range(child_count)]
    return root


def _make_forest():
    """Build 2-3 independent trees with at least one visible grid-triggering group."""
    root_count = random.choices(list(ROOT_COUNT_WEIGHTS), weights=list(ROOT_COUNT_WEIGHTS.values()), k=1)[0]
    if root_count == 3:
        roots = [
            _make_tree(),
            _make_tree(child_count=1, compact=True),
            _make_tree(child_count=1, compact=True),
        ]
        random.shuffle(roots)
        return roots
    if random.random() < 0.25:
        return [_make_tree() for _ in range(root_count)]
    roots = [
        _make_tree(),
        _make_tree(child_count=1, compact=True),
    ]
    random.shuffle(roots)
    return roots


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
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
