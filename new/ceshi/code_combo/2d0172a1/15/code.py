# Combo variant of 2d0172a1 (index 15)
# combo_spec: D12/0+D15/2
# Name: 多色层级+深度颜色标记

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
# - Base task: every foreground object shares one color, and every foreground
#   connected component is structural. A rough frame becomes a clean rectangle;
#   a rough blob becomes one pixel.
# - D12 adds depth-colored structure: outer objects use one color, children one
#   level deeper use another, and deepest blobs may use a third.
# - D15 adds one-pixel frame-side markers. In this combo the marker uses the
#   same depth color as its frame, so it acts as a visible legend swatch.
# - Concrete depth-2 example:
#   outer frame = red, its diagonal marker = red; inner frame and sibling blobs
#   = blue, inner frame marker = blue; deepest blobs = green.
#   Output removes the markers but keeps the red/blue/green canonical objects.
# - The important solver-side difference is that "all non-background components
#   are meaningful" is no longer true. Singleton components are now legend
#   pixels and must be excluded before blob/frame canonicalization.
#
# b) Complexity calibration
# - Keep the original object grammar: 1-2 frames, 1-4 blobs, depth 1-2.
# - Keep only 2 or 3 structural colors total, one per realized depth, so the
#   rule is visually obvious without exploding the color vocabulary.
# - Every sample has at least one frame marker and at least two structural
#   colors, so the fusion is always visible and never collapses back to the
#   original single-foreground task.
# - Depth-2 trees remain fairly common so training examples often show both
#   frame markers and three-level color propagation.
#
# c) Implicit-assumption audit for main()
# - `assert len(fg_colors) == 1` from the source solver is invalid now.
# - `find_components(... colors=[fg])` is invalid because multiple structural
#   colors exist.
# - "Every small component is a blob" is also invalid: single-pixel components
#   are markers, while real input blobs are still 4-11 cells by construction.
# - The old boolean-mask renderer is invalid because it collapses depth colors.
#   The solver therefore uses a transparent integer canvas and paints each
#   canonical object with its own stored color.
# - Containment detection still works unchanged on the non-singleton components,
#   because color does not change hole geometry.
#
# d) Placement legality
# - Markers must stay isolated even when they share a color with nearby objects.
#   The generator therefore places them in diagonally offset or gap-2 slots and
#   rejects any slot touching an occupied cell in 4-neighborhood.
# - Marker coordinates are included in the overall drawing bounds so padding can
#   shift the whole scene without clipping them.
#
# e) Visual observability
# - A solver can see two simultaneous regularities in a few examples:
#   structure is cleaned exactly as in the base task, and colors persist by
#   nesting depth while one-pixel swatches next to frames disappear.
# - No logic depends on numeric color IDs; colors are sampled randomly and only
#   their visible repetition pattern matters.
#
# f) Visual cleanliness
# - Markers are isolated singletons, never merged into frames or blobs, so they
#   read as annotations rather than extra geometry.
# - Canonical output remains uncluttered because those annotations are removed.
#
# g) Change inferability and new assumptions
# - The inferable rule is: "depth colors matter, but isolated single pixels near
#   frames are legend markers and should not survive the cleanup."
# - New solver assumption: every structural input object has area > 1, and every
#   marker is a singleton. The generator enforces this directly.


TRANSPARENT = -1


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray, color: int):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.color = int(color)
        self.is_frame = False
        self.parent: _Node | None = None
        self.children: list[_Node] = []
        self.mask: np.ndarray | None = None
        self.canvas: np.ndarray | None = None
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
    if node.mask is not None and node.canvas is not None:
        return

    if not node.is_frame:
        node.mask = np.ones((1, 1), dtype=bool)
        node.canvas = np.array([[node.color]], dtype=int)
        node.anchor = (0, 0)
        node.axis_map = {
            "row": [(float(node.center[0]), 0)],
            "col": [(float(node.center[1]), 0)],
        }
        return

    assert node.children
    child_canvas, _placements, axis_map = _layout_group(node.children)
    h, w = child_canvas.shape
    out = np.full((h + 4, w + 4), TRANSPARENT, dtype=int)
    out[0, :] = node.color
    out[-1, :] = node.color
    out[:, 0] = node.color
    out[:, -1] = node.color
    paste(out, child_canvas, 2, 2, transparent=TRANSPARENT)

    node.canvas = out
    node.mask = out != TRANSPARENT
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

    out = np.full((max_r, max_c), TRANSPARENT, dtype=int)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}
    for node in nodes:
        assert node.mask is not None and node.anchor is not None and node.canvas is not None
        r = row_coord[node] - node.anchor[0] - min_r
        c = col_coord[node] - node.anchor[1] - min_c
        placements[node] = (int(r), int(c))
        paste(out, node.canvas, r, c, transparent=TRANSPARENT)
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
        assert frame.mask is not None and frame.canvas is not None
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
        out = np.full((max_r - min_r, max_c - min_c), TRANSPARENT, dtype=int)
        for node, (r, c) in placements.items():
            rr = r - min_r
            cc = c - min_c
            assert node.canvas is not None
            paste(out, node.canvas, rr, cc, transparent=TRANSPARENT)
        return out

    canvas, _placements, _axis_map = _layout_group(nodes)
    return canvas


def _frame_marker_candidates_from_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mid_r = r0 + h // 2
    mid_c = c0 + w // 2
    return [
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
        (r0 - 2, mid_c),
        (r1 + 2, mid_c),
        (mid_r, c0 - 2),
        (mid_r, c1 + 2),
    ]


def _match_frame_marker(
    frame: _Node,
    markers: list[tuple[tuple[int, int], int]],
    used: set[int],
) -> int | None:
    wanted = {coord: idx for idx, coord in enumerate(_frame_marker_candidates_from_bbox(frame.bbox))}
    matches = []
    for marker_i, (coord, color) in enumerate(markers):
        if marker_i in used or coord not in wanted:
            continue
        matches.append((wanted[coord], marker_i, color))
    if not matches:
        return None
    _order, marker_i, color = min(matches)
    used.add(marker_i)
    return int(color)


def _apply_marker_colors(nodes: list[_Node], markers: list[tuple[tuple[int, int], int]]) -> None:
    used: set[int] = set()
    frames = [node for node in nodes if node.is_frame]
    frames.sort(key=lambda node: int(node.coords.shape[0]))
    for frame in frames:
        marker_color = _match_frame_marker(frame, markers, used)
        if marker_color is not None:
            frame.color = marker_color


def _parse_nodes(
    grid: np.ndarray,
    include_singletons: bool,
) -> tuple[int, list[_Node], list[list[np.ndarray]], list[tuple[tuple[int, int], int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert fg_colors

    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=fg_colors,
        return_format="dict",
    )
    assert comps

    nodes: list[_Node] = []
    holes: list[list[np.ndarray]] = []
    markers: list[tuple[tuple[int, int], int]] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        if coords.shape[0] == 1:
            markers.append((tuple(map(int, coords[0])), int(comp["color"])))
        if not include_singletons and coords.shape[0] == 1:
            continue
        node = _Node(
            coords=coords,
            bbox=tuple(map(int, comp["bbox"])),
            center=coords.mean(axis=0),
            color=int(comp["color"]),
        )
        node_holes = _component_holes(g.shape, coords)
        node.is_frame = bool(node_holes)
        nodes.append(node)
        holes.append(node_holes)

    assert nodes and any(node.is_frame for node in nodes)
    return bg, nodes, holes, markers


def _solve_nodes(bg: int, nodes: list[_Node], holes: list[list[np.ndarray]]) -> np.ndarray:
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
    canvas = _layout_root(roots)
    out = np.full(canvas.shape, bg, dtype=int)
    mask = canvas != TRANSPARENT
    out[mask] = canvas[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean a rough nested multi-color drawing with depth legend pixels into a canonical frame-and-dot diagram.

    Input structure:
    - The grid has one background color and 2-3 structural foreground colors, one per realized nesting depth.
    - Every non-singleton foreground component is a monochrome rough frame or a small filled blob.
    - Each frame also has an isolated one-pixel marker nearby in the same color; these pixels are legends, not structural blobs.

    Rule:
    - Use each frame-side singleton as a depth-color legend for that frame, then ignore those singletons as geometry.
    - Recover the containment tree from the remaining monochrome components using frame holes.
    - Replace each blob with one pixel of its depth color.
    - Replace each frame with a rectangular border of its depth color, padded two cells around its direct children.
    - Preserve the original sibling row/column ordering and keep the base task's outside-blob placement convention.

    Key insight: The rough geometry still only serves to reveal enclosure and relative ordering, but frame colors are confirmed by the nearby singleton legends, which must be read as annotations rather than objects.
    """
    bg, nodes, holes, markers = _parse_nodes(np.asarray(grid, dtype=int), include_singletons=False)
    _apply_marker_colors(nodes, markers)
    return _solve_nodes(bg, nodes, holes)


def _main_without_marker_filter(grid: np.ndarray) -> np.ndarray:
    bg, nodes, holes, _markers = _parse_nodes(np.asarray(grid, dtype=int), include_singletons=True)
    return _solve_nodes(bg, nodes, holes)


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

# Layout / arrangement
ARRANGE_GAP_COMPACT = (2, 3)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 4)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 3

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 5)

# Tree structure probabilities
DEPTH1_PROB = 0.25
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
        self.color: int | None = None
        self.depth: int | None = None
        self.marker_pos: tuple[int, int] | None = None


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


def _assign_depth_colors(root: _Obj, outside: _Obj | None, bg: int) -> None:
    def collect_max_depth(obj: _Obj, depth: int) -> int:
        best = depth
        for child in obj.children:
            best = max(best, collect_max_depth(child, depth + 1))
        return best

    max_depth = collect_max_depth(root, 0)
    palette = random.sample([c for c in range(10) if c != bg], max_depth + 1)

    def paint(obj: _Obj, depth: int) -> None:
        obj.depth = depth
        obj.color = palette[depth]
        for child in obj.children:
            paint(child, depth + 1)

    paint(root, 0)
    if outside is not None:
        outside.depth = 0
        outside.color = palette[0]


def _node_from_obj(obj: _Obj) -> _Node:
    assert obj.mask is not None and obj.color is not None
    r0, c0 = obj.pos
    coords = np.argwhere(obj.mask) + np.array([r0, c0])
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    node = _Node(
        coords=coords.astype(int),
        bbox=(int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])),
        center=coords.mean(axis=0),
        color=int(obj.color),
    )
    node.is_frame = obj.kind == "frame"
    return node


def _expected_output(root: _Obj, outside: _Obj | None, bg: int) -> np.ndarray:
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
    canvas = _layout_root(roots)
    out = np.full(canvas.shape, bg, dtype=int)
    mask = canvas != TRANSPARENT
    out[mask] = canvas[mask]
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


def _occupied_cells(objects: list[_Obj]) -> set[tuple[int, int]]:
    occupied = set()
    for obj in objects:
        assert obj.mask is not None
        rr, cc = obj.pos
        for r, c in np.argwhere(obj.mask):
            occupied.add((int(rr + r), int(cc + c)))
    return occupied


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _marker_candidates(frame: _Obj) -> list[tuple[int, int]]:
    assert frame.mask is not None
    r0, c0 = frame.pos
    h, w = frame.mask.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    mid_r = r0 + h // 2
    mid_c = c0 + w // 2
    candidates = [
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
        (r0 - 2, mid_c),
        (r1 + 2, mid_c),
        (mid_r, c0 - 2),
        (mid_r, c1 + 2),
    ]
    random.shuffle(candidates)
    return candidates


def _marker_slot_ok(cell: tuple[int, int], occupied: set[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
    if cell in occupied or cell in used:
        return False
    for nbr in _neighbors4(cell):
        if nbr in occupied or nbr in used:
            return False
    return True


def _place_markers(objects: list[_Obj]) -> bool:
    occupied = _occupied_cells(objects)
    used: set[tuple[int, int]] = set()
    frames = [obj for obj in objects if obj.kind == "frame"]
    frames.sort(key=lambda obj: int(obj.depth), reverse=True)
    for frame in frames:
        frame.marker_pos = None
        for coord in _marker_candidates(frame):
            if _marker_slot_ok(coord, occupied, used):
                frame.marker_pos = coord
                used.add(coord)
                break
        if frame.marker_pos is None:
            return False
    return True


def _draw(objects: list[_Obj], bg: int) -> np.ndarray | None:
    marker_points = [obj.marker_pos for obj in objects if obj.marker_pos is not None]
    min_r = min([obj.pos[0] for obj in objects] + [r for r, _ in marker_points])
    min_c = min([obj.pos[1] for obj in objects] + [c for _, c in marker_points])
    max_r = max([obj.pos[0] + obj.mask.shape[0] for obj in objects] + [r + 1 for r, _ in marker_points])
    max_c = max([obj.pos[1] + obj.mask.shape[1] for obj in objects] + [c + 1 for _, c in marker_points])
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
        assert obj.mask is not None and obj.color is not None
        r = obj.pos[0] + shift_r
        c = obj.pos[1] + shift_c
        obj.pos = (r, c)
        if obj.marker_pos is not None:
            mr, mc = obj.marker_pos
            obj.marker_pos = (mr + shift_r, mc + shift_c)
        patch = np.full(obj.mask.shape, bg, dtype=int)
        patch[obj.mask] = int(obj.color)
        paste(grid, patch, r, c, transparent=bg)

    for obj in objects:
        if obj.marker_pos is not None:
            assert obj.color is not None
            r, c = obj.marker_pos
            grid[r, c] = int(obj.color)

    fg_colors = sorted({int(obj.color) for obj in objects if obj.color is not None})
    bg_count = int(np.count_nonzero(grid == bg))
    if bg_count <= max(int(np.count_nonzero(grid == color)) for color in fg_colors):
        return None
    return grid


def _combo_signature(grid: np.ndarray) -> tuple[int, int, int]:
    bg = int(get_bg(grid))
    fg_colors = sorted(set(map(int, grid.flatten())) - {bg})
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=fg_colors,
        return_format="dict",
    )
    marker_count = sum(1 for comp in comps if len(comp["coords"]) == 1)
    frame_count = 0
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        if coords.shape[0] == 1:
            continue
        if _component_holes(grid.shape, coords):
            frame_count += 1
    return len(fg_colors), marker_count, frame_count


def _recolor_one_marker(grid: np.ndarray) -> np.ndarray | None:
    bg = int(get_bg(grid))
    fg_colors = sorted(set(map(int, grid.flatten())) - {bg})
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=fg_colors,
        return_format="dict",
    )
    markers = [comp for comp in comps if len(comp["coords"]) == 1]
    if not markers:
        return None

    replacement_colors = [color for color in range(10) if color not in fg_colors and color != bg]
    if not replacement_colors:
        return None

    mutated = np.array(grid, copy=True)
    coord = tuple(map(int, markers[0]["coords"][0]))
    mutated[coord] = replacement_colors[0]
    return mutated


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    root, outside = _make_tree()
    if not _prepare(root):
        return None
    if outside is not None:
        if not _prepare(outside):
            return None
        _place_outside(root, outside)
    _assign_depth_colors(root, outside, bg)

    objects = _walk(root)
    if outside is not None:
        outside.pos = outside.rel
        objects.append(outside)
    if not _place_markers(objects):
        return None

    grid = _draw(objects, bg)
    if grid is None:
        return None

    expected = _expected_output(root, outside, bg)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    if np.array_equal(_main_without_marker_filter(grid), expected):
        return None
    marker_mutation = _recolor_one_marker(grid)
    if marker_mutation is None:
        return None
    if np.array_equal(main(marker_mutation), got):
        return None

    n_colors, marker_count, frame_count = _combo_signature(grid)
    if n_colors < 2 or marker_count != frame_count or frame_count < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
