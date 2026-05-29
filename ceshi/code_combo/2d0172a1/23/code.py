# Combo variant of 2d0172a1 (index 23)
# combo_spec: D8/1+D11/2
# Name: 多棵独立树+多个外部色块
#
# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   input_objects:
#     - one foreground color on one background color
#     - 2-3 disjoint root frames, each owning its own local depth-1 or depth-2 subtree
#     - 0-3 blobs outside every frame hole, each placed adjacent to one chosen root frame
#     - an outside blob never shares the same side of the same root with another outside blob
#   output_objects:
#     - every frame becomes a clean rectangular border with a 1-cell moat to its direct children
#     - every blob becomes one foreground pixel
#     - the 2-3 root frames are laid out as an independent forest
#     - each outside blob is reattached to the corresponding side of its owner root and keeps a 3x3 slot
#   preserved_invariants:
#     - containment is still recovered only from holes of rough frame components
#     - local child ordering inside each frame still comes from row/column overlap bands
#     - colors are unchanged; the task stays purely geometric
#
# a) Concrete behavior and trace
#   - Base task: detect one containment tree, canonicalize children recursively, then maybe place one
#     root-level outside blob beside the single outer frame.
#   - D11 alone changed the root stage into a forest of 2-3 root frames, but removed outside blobs.
#   - D8 alone kept one root frame and allowed several outside blobs on different sides of that frame.
#   - Combo behavior: first canonicalize each root tree exactly as before, next arrange the 2-3 root
#     frames as a forest, then attach each root-level blob to the nearest and clearly facing root frame.
#     Example: if the input shows a left root frame with one outside blob above it and a right root frame
#     with another outside blob on its right, the output shows two canonical root rectangles in forest
#     order, with one top-side 3x3 slot above the left rectangle and one right-side 3x3 slot beside the
#     right rectangle.
#
# b) Complexity calibration
#   - Root count is capped at 2-3 and outside count at 0-3, so the fusion is visibly richer than the
#     base task but still learnable from a few ARC examples.
#   - Each root uses only shallow depth 1-2. The extra difficulty comes from combining local containment
#     with global frame ownership of outside blobs, not from deeper recursion.
#   - Generator allows some roots to own two outside blobs on different sides, so the D8 behavior is
#     visible even when several independent roots are present.
#
# c) Hidden assumptions review
#   - `find_components(...)` and hole detection still assume monochrome foreground; preserved.
#   - Parent assignment still assumes the correct parent is the smallest containing frame; generator keeps
#     root frames disjoint and non-nested, so this remains valid.
#   - `_build_layout()` still assumes every frame has at least one direct child; generator guarantees it.
#   - New root logic assumes each root-level blob has a unique owner root frame and a clear side relative
#     to that frame. Generator enforces positive side gaps, orthogonal overlap with the owner, and enough
#     inter-root spacing to avoid ownership ambiguity or slot collisions.
#   - New root logic also assumes at most one outside blob per `(root frame, side)`; generator enforces it.
#
# d) Placement legality
#   - Inside each frame, children are checked against the actual hole mask and against one another.
#   - Root frames are globally arranged as disjoint trees before any outside blobs are placed.
#   - Outside blobs are then placed with retries until they avoid all frame bboxes, avoid each other, and
#     are still scored as belonging to their intended owner root.
#
# e) Visual observability
#   - The input visibly contains several separate rough loops, not one global outer frame.
#   - Outside blobs remain visibly detached and side-specific relative to one root frame, and the output
#     mirrors that with explicit top/bottom/left/right 3x3 slots around individual root rectangles.
#   - No rule depends on hidden ids or numeric color values; only visible geometry matters.
#
# f) Visual neatness
#   - Root frames use a larger forest gap than ordinary in-frame sibling gaps.
#   - Outside blobs are limited to one per side per root and at most three total, preventing clutter.
#   - Root frame sizes are kept modest so the forest and side slots remain legible within 25x25.
#
# g) Color inference
#   - No color remapping is introduced. The variant remains strictly foreground/background.

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


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _hole_mask(mask: np.ndarray) -> np.ndarray | None:
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    holes = [(lab == k) for k in range(1, n + 1) if k not in boundary]
    if len(holes) != 1:
        return None
    return holes[0]


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _interval_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[1] < b[0]:
        return int(b[0] - a[1] - 1)
    if b[1] < a[0]:
        return int(a[0] - b[1] - 1)
    return 0


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

    bands: list[dict[str, object]] = []
    for item in sorted(items, key=lambda x: x[2]):
        interval = item[1]
        for band in bands:
            band_items = band["items"]
            assert isinstance(band_items, list)
            if any(_intervals_overlap(interval, other[1]) for other in band_items):
                band_items.append(item)
                band["interval"] = (
                    min(band["interval"][0], interval[0]),  # type: ignore[index]
                    max(band["interval"][1], interval[1]),  # type: ignore[index]
                )
                break
        else:
            bands.append({"items": [item], "interval": interval})

    coord_by_node: dict[_Node, int] = {}
    coord = 0
    for i, band in enumerate(bands):
        band_items = band["items"]
        assert isinstance(band_items, list)
        if i:
            prev_items = bands[i - 1]["items"]
            assert isinstance(prev_items, list)
            prev_down = max(item[4] - 1 - item[3] for item in prev_items)
            curr_up = max(item[3] for item in band_items)
            coord += prev_down + band_gap + curr_up
        for item in band_items:
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


def _outside_side(dot: _Node, frame: _Node) -> str:
    return _outside_side_geom(dot.bbox, dot.center, frame.bbox, frame.center)


def _outside_side_geom(
    dot_bbox: tuple[int, int, int, int],
    dot_center: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    frame_center: np.ndarray,
) -> str:
    r0, c0, r1, c1 = dot_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if c0 > fc1:
        return "right"
    if c1 < fc0:
        return "left"
    if r0 > fr1:
        return "bottom"
    if r1 < fr0:
        return "top"

    dr = float(dot_center[0] - frame_center[0])
    dc = float(dot_center[1] - frame_center[1])
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


def _separate_slot_centers(preferred: list[int], lower: int, upper: int, gap: int = 3) -> list[int]:
    if not preferred:
        return []

    centers = [int(v) for v in preferred]
    for i in range(1, len(centers)):
        centers[i] = max(centers[i], centers[i - 1] + gap)
    for i in range(len(centers) - 2, -1, -1):
        centers[i] = min(centers[i], centers[i + 1] - gap)

    if centers[0] < lower:
        shift = lower - centers[0]
        centers = [v + shift for v in centers]
    if centers[-1] > upper:
        shift = centers[-1] - upper
        centers = [v - shift for v in centers]
    return centers


def _owner_score(
    dot_bbox: tuple[int, int, int, int],
    dot_center: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    frame_center: np.ndarray,
    frame_area: int,
) -> tuple[int, int, float, int]:
    side = _outside_side_geom(dot_bbox, dot_center, frame_bbox, frame_center)
    r0, c0, r1, c1 = dot_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if side in {"left", "right"}:
        ortho_gap = _interval_gap((r0, r1), (fr0, fr1))
        side_gap = c0 - fc1 - 1 if side == "right" else fc0 - c1 - 1
    else:
        ortho_gap = _interval_gap((c0, c1), (fc0, fc1))
        side_gap = r0 - fr1 - 1 if side == "bottom" else fr0 - r1 - 1
    center_gap = abs(float(dot_center[0] - frame_center[0])) + abs(float(dot_center[1] - frame_center[1]))
    return max(0, ortho_gap), max(0, int(side_gap)), center_gap, frame_area


def _assign_outside_owner(dot: _Node, frames: list[_Node]) -> _Node:
    return min(
        frames,
        key=lambda frame: _owner_score(
            dot.bbox,
            dot.center,
            frame.bbox,
            frame.center,
            int(frame.coords.shape[0]),
        ),
    )


FOREST_BAND_GAP = 6


def _layout_root(nodes: list[_Node]) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node)

    frames = [node for node in nodes if node.is_frame]
    dots = [node for node in nodes if not node.is_frame]
    assert 2 <= len(frames) <= 3

    frame_mask, frame_places, _axis_map = _layout_group(frames, band_gap=FOREST_BAND_GAP)
    extents = [(0, 0, frame_mask.shape[0], frame_mask.shape[1])]
    placements: dict[_Node, tuple[int, int]] = {}
    for frame, (r, c) in frame_places.items():
        placements[frame] = (int(r), int(c))

    dots_by_owner: dict[_Node, dict[str, list[_Node]]] = {
        frame: {"left": [], "right": [], "top": [], "bottom": []} for frame in frames
    }
    for dot in dots:
        owner = _assign_outside_owner(dot, frames)
        dots_by_owner[owner][_outside_side(dot, owner)].append(dot)

    for frame in frames:
        for side in ("left", "right"):
            dots_by_owner[frame][side].sort(key=lambda node: float(node.center[0]))
        for side in ("top", "bottom"):
            dots_by_owner[frame][side].sort(key=lambda node: float(node.center[1]))

    for frame in frames:
        assert frame.mask is not None
        fr, fc = frame_places[frame]
        side_dots = dots_by_owner[frame]
        for side in ("left", "right"):
            dots_here = side_dots[side]
            preferred = [_map_frame_axis(frame, 0, float(dot.center[0])) for dot in dots_here]
            slots = _separate_slot_centers(preferred, lower=-1, upper=frame.mask.shape[0])
            for dot, slot in zip(dots_here, slots):
                r = fr + slot
                c = fc + (frame.mask.shape[1] + 1 if side == "right" else -2)
                placements[dot] = (int(r), int(c))
                extents.append((r - 1, c - 1, r + 2, c + 2))
        for side in ("top", "bottom"):
            dots_here = side_dots[side]
            preferred = [_map_frame_axis(frame, 1, float(dot.center[1])) for dot in dots_here]
            slots = _separate_slot_centers(preferred, lower=-1, upper=frame.mask.shape[1])
            for dot, slot in zip(dots_here, slots):
                r = fr + (frame.mask.shape[0] + 1 if side == "bottom" else -2)
                c = fc + slot
                placements[dot] = (int(r), int(c))
                extents.append((r - 1, c - 1, r + 2, c + 2))

    min_r = min(e[0] for e in extents)
    min_c = min(e[1] for e in extents)
    max_r = max(e[2] for e in extents)
    max_c = max(e[3] for e in extents)
    out = np.zeros((max_r - min_r, max_c - min_c), dtype=bool)

    for frame in frames:
        assert frame.mask is not None
        rr = placements[frame][0] - min_r
        cc = placements[frame][1] - min_c
        out[rr : rr + frame.mask.shape[0], cc : cc + frame.mask.shape[1]] |= frame.mask
    for dot in dots:
        rr = placements[dot][0] - min_r
        cc = placements[dot][1] - min_c
        out[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough forest of nested frame sketches into a clean forest of rectangles and side slots.

    Input structure:
    - The grid uses exactly one background color and one foreground color.
    - Foreground components are rough closed frames or small filled blobs.
    - There are 2-3 independent root frames, each with its own children, plus 0-3 blobs outside every frame hole and adjacent to one root frame side.

    Rule:
    - Detect which components are frames by checking for enclosed holes, then assign every component to the smallest frame that fully contains it.
    - Turn each blob into one foreground pixel and each frame into a rectangular border with a one-cell moat around its direct children.
    - Canonicalize every root tree locally, arrange the root frames as a forest, then attach each root-level outside blob to the side of the root frame it visibly belongs to; each such blob occupies the center of its own 3x3 slot.

    Key insight:
    - The exact hand-drawn contours are irrelevant. What survives canonicalization is the containment forest, the coarse spatial order of the independent roots, and the side-specific ownership of each outside blob.
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
    frame_roots = [node for node in roots if node.is_frame]
    dot_roots = [node for node in roots if not node.is_frame]
    assert 2 <= len(frame_roots) <= 3
    assert len(dot_roots) <= 3
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


# --------------- generator (o5x) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────────────
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 10
MAX_PAD_PER_SIDE = 6

FRAME_H_RANGE = (6, 12)
FRAME_W_RANGE = (6, 12)
FRAME_CORNER_MAX = 4
FRAME_MIN_EDGE = 1
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

BLOB_SIZE_RANGE = (4, 7)
BLOB_BBOX_RANGE = (2, 4)

ARRANGE_GAP_COMPACT = (1, 2)
ARRANGE_GAP_NORMAL = (1, 3)
ARRANGE_EXTRA_SPAN = (0, 2)
FRAME_MARGIN_RANGE = (2, 3)
COMPACT_MARGIN = 2

ROOT_ARRANGE_GAP = (4, 6)
OUTSIDE_GAP_RANGE = (2, 4)
OUTSIDE_PLACE_RETRIES = 60

ROOT_COUNT_WEIGHTS = {2: 1, 3: 1}
ROOT_DEPTH2_WEIGHTS = {
    2: (2, 4, 1),
    3: (3, 4, 0, 0),
}
DEPTH1_INSIDE_RANGE = (1, 2)
DEPTH2_INNER_BLOBS = (1, 1)
DEPTH2_OUTER_BLOBS = (0, 1)

OUTSIDE_COUNT_WEIGHTS = (1, 2, 5, 4)
MAX_OUTSIDE_BLOBS = 3
MAX_OUTSIDE_PER_ROOT = 2
MIN_OUTSIDE_BLOBS = 1

STEP_MIX_ALT_PROB = 0.7

GEN_RETRIES = 1000
# ────────────────────────────────────────────────────────────────────────────


class _Obj:
    def __init__(self, kind, children=None, outside_side=None, owner=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side
        self.owner: _Obj | None = owner
        self.frame_h_cap: int | None = None
        self.frame_w_cap: int | None = None


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
    h_cap = node.frame_h_cap if node.frame_h_cap is not None else FRAME_H_RANGE[1]
    w_cap = node.frame_w_cap if node.frame_w_cap is not None else FRAME_W_RANGE[1]
    if min_h > h_cap or min_w > w_cap:
        return False
    h_hi = min(h_cap, min_h + 2) if tight else h_cap
    w_hi = min(w_cap, min_w + 3) if tight else w_cap
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
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _bbox_from_pos_mask(pos, mask):
    r, c = pos
    h, w = mask.shape
    return int(r), int(c), int(r + h - 1), int(c + w - 1)


def _bbox_center_from_pos_mask(pos, mask):
    r0, c0, r1, c1 = _bbox_from_pos_mask(pos, mask)
    return np.array([(r0 + r1) / 2, (c0 + c1) / 2], dtype=float)


def _boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _owner_score_for_obj(dot_obj: _Obj, frame_obj: _Obj) -> tuple[int, int, float, int]:
    assert dot_obj.mask is not None and frame_obj.mask is not None
    dot_bbox = _bbox_from_pos_mask(dot_obj.pos, dot_obj.mask)
    frame_bbox = _bbox_from_pos_mask(frame_obj.pos, frame_obj.mask)
    dot_center = _bbox_center_from_pos_mask(dot_obj.pos, dot_obj.mask)
    frame_center = _bbox_center_from_pos_mask(frame_obj.pos, frame_obj.mask)
    return _owner_score(dot_bbox, dot_center, frame_bbox, frame_center, int(np.count_nonzero(frame_obj.mask)))


def _closest_root(dot_obj: _Obj, roots: list[_Obj]) -> _Obj:
    return min(roots, key=lambda root: _owner_score_for_obj(dot_obj, root))


def _root_has_inner_frame(root: _Obj) -> bool:
    return any(child.kind == "frame" for child in root.children)


def _apply_root_size_caps(root: _Obj, outside_sides: list[str]) -> None:
    h_cap = root.frame_h_cap if root.frame_h_cap is not None else FRAME_H_RANGE[1]
    w_cap = root.frame_w_cap if root.frame_w_cap is not None else FRAME_W_RANGE[1]
    has_inner_frame = _root_has_inner_frame(root)
    vertical = len(set(outside_sides) & {"top", "bottom"})
    horizontal = len(set(outside_sides) & {"left", "right"})
    if has_inner_frame:
        if vertical == 2:
            h_cap = min(h_cap, 11)
        if horizontal == 2:
            w_cap = min(w_cap, 11)
    else:
        if vertical == 1:
            h_cap = min(h_cap, 10)
        elif vertical == 2:
            h_cap = min(h_cap, 9)
        if horizontal == 1:
            w_cap = min(w_cap, 10)
        elif horizontal == 2:
            w_cap = min(w_cap, 9)
        if len(outside_sides) >= 2:
            h_cap = min(h_cap, 10)
            w_cap = min(w_cap, 10)
    root.frame_h_cap = h_cap
    root.frame_w_cap = w_cap


def _combo_is_salient(roots: list[_Obj], outside_blobs: list[_Obj]) -> bool:
    outside_count = len(outside_blobs)
    if outside_count < MIN_OUTSIDE_BLOBS:
        return False
    return True


def _arrange_roots(roots):
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
            row_offsets = {
                a: random.randint(0, top_h - ha),
                b: random.randint(0, top_h - hb),
            }
            maybe_add(
                candidates,
                {
                    a: (row_offsets[a], 0),
                    b: (row_offsets[b], wa + gap_x),
                    c: (top_h + gap_y, random.randint(0, max(0, top_w - wc))),
                },
            )

            left_w = max(wa, wb)
            left_h = ha + gap_y + hb
            col_offsets = {
                a: random.randint(0, left_w - wa),
                b: random.randint(0, left_w - wb),
            }
            maybe_add(
                candidates,
                {
                    a: (0, col_offsets[a]),
                    b: (ha + gap_y, col_offsets[b]),
                    c: (random.randint(0, max(0, left_h - hc)), left_w + gap_x),
                },
            )

            maybe_add(
                candidates,
                {
                    a: (0, 0),
                    b: (random.randint(0, max(0, ha - hb)), wa + gap_x),
                    c: (top_h + gap_y, random.randint(0, max(0, wa + gap_x - wc + wb))),
                },
            )

    if not candidates:
        return None
    return random.choice(candidates)


def _sample_outside_blobs(roots: list[_Obj]) -> list[_Obj]:
    max_outside = min(MAX_OUTSIDE_BLOBS, len(roots) * MAX_OUTSIDE_PER_ROOT)
    counts = list(range(max_outside + 1))
    weights = list(OUTSIDE_COUNT_WEIGHTS[: max_outside + 1])
    n_outside = random.choices(counts, weights=weights, k=1)[0]
    if n_outside == 0:
        return []

    chosen: list[tuple[_Obj, str]] = []
    side_counts = {root: 0 for root in roots}
    used_pairs: set[tuple[int, str]] = set()
    attempts = 0
    while len(chosen) < n_outside and attempts < 100:
        attempts += 1
        root_index = random.randrange(len(roots))
        root = roots[root_index]
        if side_counts[root] >= MAX_OUTSIDE_PER_ROOT:
            continue
        side = random.choice(("left", "right", "top", "bottom"))
        if (root_index, side) in used_pairs:
            continue
        used_pairs.add((root_index, side))
        side_counts[root] += 1
        chosen.append((root, side))

    blobs = [_Obj("blob", outside_side=side, owner=root) for root, side in chosen]
    by_root: dict[_Obj, list[str]] = {root: [] for root in roots}
    for blob in blobs:
        assert blob.owner is not None and blob.outside_side is not None
        by_root[blob.owner].append(blob.outside_side)
    for root, sides in by_root.items():
        _apply_root_size_caps(root, sides)
    return blobs


def _place_outside_blobs(roots: list[_Obj], outside_blobs: list[_Obj]) -> bool:
    root_boxes = []
    for root in roots:
        assert root.mask is not None
        root_boxes.append((root, _bbox_from_pos_mask(root.pos, root.mask)))

    placed_boxes: list[tuple[int, int, int, int]] = []
    for blob in outside_blobs:
        assert blob.owner is not None and blob.mask is not None and blob.outside_side is not None
        owner = blob.owner
        assert owner.mask is not None
        oh, ow = owner.mask.shape
        bh, bw = blob.mask.shape
        base_r, base_c = owner.pos
        ok = False
        for _ in range(OUTSIDE_PLACE_RETRIES):
            gap = random.randint(*OUTSIDE_GAP_RANGE)
            side = blob.outside_side
            if side == "right":
                r = base_r + random.randint(1, oh - bh - 1)
                c = base_c + ow + gap
            elif side == "left":
                r = base_r + random.randint(1, oh - bh - 1)
                c = base_c - gap - bw
            elif side == "bottom":
                r = base_r + oh + gap
                c = base_c + random.randint(1, ow - bw - 1)
            else:
                r = base_r - gap - bh
                c = base_c + random.randint(1, ow - bw - 1)
            blob.pos = (int(r), int(c))
            cand_box = _bbox_from_pos_mask(blob.pos, blob.mask)
            if any(_boxes_overlap(cand_box, box) for _root, box in root_boxes):
                continue
            if any(_boxes_overlap(cand_box, box) for box in placed_boxes):
                continue
            if _closest_root(blob, roots) is not owner:
                continue
            owner_box = _bbox_from_pos_mask(owner.pos, owner.mask)
            owner_center = _bbox_center_from_pos_mask(owner.pos, owner.mask)
            if _outside_side_geom(cand_box, _bbox_center_from_pos_mask(blob.pos, blob.mask), owner_box, owner_center) != side:
                continue
            placed_boxes.append(cand_box)
            ok = True
            break
        if not ok:
            return False
    return True


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


def _expected_output(roots, outside_blobs, bg, fg):
    def add(obj):
        node = _node_from_obj(obj)
        for child in obj.children:
            child_node = add(child)
            child_node.parent = node
            node.children.append(child_node)
        return node

    forest = [add(root) for root in roots]
    forest.extend(add(blob) for blob in outside_blobs)
    mask = _layout_root(forest)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _make_tree(depth2):
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
    root_count = random.choices(list(ROOT_COUNT_WEIGHTS), weights=list(ROOT_COUNT_WEIGHTS.values()), k=1)[0]
    weights = ROOT_DEPTH2_WEIGHTS[root_count]
    deep_count = random.choices(range(root_count + 1), weights=weights, k=1)[0]
    depth_flags = [True] * deep_count + [False] * (root_count - deep_count)
    random.shuffle(depth_flags)
    roots = [_make_tree(flag) for flag in depth_flags]
    outside_blobs = _sample_outside_blobs(roots)
    return roots, outside_blobs


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


def _gen_attempt():
    bg, fg = random.sample(range(10), 2)
    roots, outside_blobs = _make_forest()
    if not _combo_is_salient(roots, outside_blobs):
        return None
    for root in roots:
        if not _prepare(root):
            return None
    for blob in outside_blobs:
        if not _prepare(blob):
            return None

    placements = _arrange_roots(roots)
    if placements is None:
        return None
    for root, pos in placements.items():
        root.rel = pos

    objects = []
    for root in roots:
        objects.extend(_walk(root))
    if not _place_outside_blobs(roots, outside_blobs):
        return None
    objects.extend(outside_blobs)

    grid = _draw(objects, bg, fg)
    if grid is None:
        return None
    expected = _expected_output(roots, outside_blobs, bg, fg)
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
