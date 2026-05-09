from __future__ import annotations

# Combo variant of 2d0172a1 (index 30)
# combo_spec: D2/1+D19/1
# Name: 大色块+按填充率判定对象类型
#
# Design analysis:
# variant:
#   concretization:
#   - Foreground objects are still wobbly 1-pixel loops plus filled patches, but the filled patches
#     are enlarged to 12-25 pixels with 5x5/5x6/6x5/6x6 bounding boxes, so their footprint can
#     approach that of a small frame.
#   - The canonical output is unchanged: every patch becomes one foreground pixel and every frame
#     becomes a rectangular border with a two-cell inset around its direct children.
#   - Concrete behavior trace:
#     original: a 3x3 blob and a 9x9 loop were separated mainly by topology, because size already
#     hinted which object was which.
#     combo: a 5x5 15-pixel patch can sit beside a 7x9 frame and both look like substantial
#     foreground masses; the solver must first estimate pixel-count / bbox-area and use
#     `< 0.5 => frame`, `>= 0.5 => blob`, then recover containment and canonicalize.
#   complexity_calibration:
#   - The tree structure, outside-blob option, and sibling-order preservation remain intact, so the
#     task is still about nested diagram abstraction rather than isolated object classification.
#   - To keep the density cue visible but not trivial, the generator keeps frame fill ratios safely
#     below 0.5 and blob fill ratios safely above 0.5, while allowing their raw bbox areas to
#     overlap. This makes density, not scale, the robust cue.
#   - Layout freedom is reduced only enough to stay feasible on a 25x25 grid: large blobs are kept,
#     but deep trees and wide sibling spreads are slightly tightened.
#   implicit_assumptions_review:
#   - `node.is_frame` can no longer depend on enclosed holes; it now depends on bbox fill ratio,
#     which means the generator must guarantee a clean density gap across the 0.5 threshold.
#   - Holes are still computed afterward for containment only. That keeps the fusion faithful:
#     topology still tells which object contains which, but density decides what the object is.
#   - `_layout_group` assumes sibling ordering can be read from bbox bands. Larger blobs widen those
#     bands, so generation must keep siblings separated on at least one axis.
#   placement_legality:
#   - Every candidate child group is composed first and checked against the actual hole mask of the
#     sampled frame, so no blob or inner frame leaks through a wobbly border.
#   - Outside blobs stay 1-2 cells away from the root frame, preserving side information without
#     accidental touching or overlap.
#   visual_observability:
#   - Across examples, humans can see that some dense 5x5-ish patches and some sparse loop-like
#     objects have comparable extents, yet only the sparse ones expand into canonical rectangles.
#   - No hidden use of color ids or ordering is introduced; the only new cue is visible density.
#   cleanliness:
#   - There is still only one foreground color and no overlaid markers.
#   - Dense patches are connected and hole-free, while frames remain single loops, so the scene
#     stays readable even though the two object classes are more similar in size.

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


FILL_RATIO_THRESHOLD = 0.5


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


def _bbox_fill_ratio(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    r0, c0, r1, c1 = bbox
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return float(len(coords)) / float(area)


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
    Convert a rough nested sketch with bulky patches into a clean canonical frame-and-dot diagram.

    Input structure:
    - The grid uses one background color and one foreground color.
    - Foreground components are rough closed loops or large connected filled patches whose bounding
      boxes can be as large as 5x5 to 6x6, making them visually close to small frames.
    - Frames may contain other frames or patches, and one patch may also lie outside the outer frame.

    Rule:
    - Classify each component by its bbox fill ratio: `pixels / bbox_area < 0.5` means frame,
      otherwise it is a filled patch.
    - After classification, recover containment by checking which frame hole fully contains each
      other component.
    - Replace every filled patch with one foreground pixel and every frame with a rectangular border
      padded two cells around its direct children.
    - Preserve sibling row/column band order, and keep any outside patch beside the canonical outer frame.

    Key insight: topology still reveals enclosure relationships, but object type is no longer "has
    a hole or not" at first glance; the decisive cue is how densely a component fills its own
    bounding box.
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
        node.is_frame = _bbox_fill_ratio(coords, node.bbox) < FILL_RATIO_THRESHOLD
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

# ── CFG: tunable generation parameters ──────────────────────────────────────
# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

# Frame geometry: keep ratios safely below the 0.5 classification threshold.
FRAME_H_RANGE = (7, 24)
FRAME_W_RANGE = (7, 24)
FRAME_CORNER_MAX = 4
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 80
FRAME_FILL_RATIO_MAX = 0.499

# Blob geometry: large patches must stay above the 0.5 classification threshold.
BLOB_SIZE_RANGE = (12, 25)
BLOB_BBOX_CHOICES = ((5, 5), (5, 6), (6, 5), (6, 6))
BLOB_RETRIES = 120
BLOB_FILL_RATIO_MIN = 0.55

# Layout / arrangement
ARRANGE_GAP_COMPACT = (1, 2)
ARRANGE_GAP_NORMAL = (1, 2)
ARRANGE_EXTRA_SPAN = (0, 2)
ARRANGE_RETRIES = 40
FRAME_MARGIN_RANGE = (1, 2)
COMPACT_MARGIN = 1
FRAME_SIZE_RETRIES = 48
FRAME_POSITION_RETRIES = 36

# Outside blob placement
OUTSIDE_GAP_RANGE = (1, 2)

# Tree structure probabilities
DEPTH1_PROB = 0.4
OUTSIDE_BLOB_PROB = 0.25
DEPTH1_INSIDE_RANGE = (1, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (0, 1)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 500
# ────────────────────────────────────────────────────────────────────────────


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
        fill_ratio = float(np.count_nonzero(mask)) / float(h * w)
        if lo <= ratio <= hi and fill_ratio < FRAME_FILL_RATIO_MAX and _loop_ok(mask):
            return mask
    return None


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _blob_ok(mask: np.ndarray) -> bool:
    _, n = label_mask(mask, connectivity=4)
    if n != 1:
        return False
    return len(_component_holes(mask.shape, np.argwhere(mask))) == 0


def _blob_mask():
    """Generate a large irregular blob whose bbox fill ratio is safely above 0.5."""
    feasible = []
    for h, w in BLOB_BBOX_CHOICES:
        area = h * w
        size_lo = max(BLOB_SIZE_RANGE[0], int(np.ceil(area * BLOB_FILL_RATIO_MIN)))
        size_hi = min(BLOB_SIZE_RANGE[1], area)
        if size_lo <= size_hi:
            feasible.append((h, w, size_lo, size_hi))
    weights = [1.2 if max(h, w) == 6 else 1.0 for h, w, _, _ in feasible]

    for _ in range(BLOB_RETRIES):
        h, w, size_lo, size_hi = random.choices(feasible, weights=weights, k=1)[0]
        size = random.randint(size_lo, size_hi)
        cells = {(r, c) for r in range(h) for c in range(w)}
        while len(cells) > size:
            removable = list(cells)
            random.shuffle(removable)
            removed = False
            for cell in removable:
                trial = set(cells)
                trial.remove(cell)
                trial_mask = np.zeros((h, w), dtype=bool)
                pts = np.array(list(trial), dtype=int)
                trial_mask[pts[:, 0], pts[:, 1]] = True
                if _touches_all_sides(trial_mask) and _blob_ok(trial_mask):
                    cells = trial
                    removed = True
                    break
            if not removed:
                break
        mask = np.zeros((h, w), dtype=bool)
        pts = np.array(list(cells), dtype=int)
        mask[pts[:, 0], pts[:, 1]] = True
        if (
            len(cells) == size
            and _touches_all_sides(mask)
            and _blob_ok(mask)
            and float(np.count_nonzero(mask)) / float(h * w) >= BLOB_FILL_RATIO_MIN
        ):
            return mask
    return None


def _arrange(children, compact=False):
    """Arrange children in a random layout while keeping the minimal frame footprint feasible."""
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    modes = ("vertical", "horizontal") if k >= 3 else ("vertical", "horizontal", "diagonal")

    for _ in range(ARRANGE_RETRIES):
        order = children[:]
        random.shuffle(order)
        gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
        placements = {}
        mode = random.choice(modes)

        if mode == "vertical":
            group_h = sum(h for h, _ in dims) + sum(gaps)
            group_w = max(w for _, w in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
            r = 0
            for child, gap in zip(order, gaps + [0]):
                h, w = child.mask.shape
                placements[child] = (r, random.randint(0, group_w - w))
                r += h + gap
        elif mode == "horizontal":
            group_h = max(h for h, _ in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
            group_w = sum(w for _, w in dims) + sum(gaps)
            c = 0
            for child, gap in zip(order, gaps + [0]):
                h, w = child.mask.shape
                placements[child] = (random.randint(0, group_h - h), c)
                c += w + gap
        else:
            group_h = sum(h for h, _ in dims) + sum(gaps)
            group_w = sum(w for _, w in dims) + sum(gaps)
            r = c = 0
            for child, gap in zip(order, gaps + [0]):
                h, w = child.mask.shape
                placements[child] = (r, c)
                r += h + gap
                c += w + gap

        if group_h + 2 * COMPACT_MARGIN <= FRAME_H_RANGE[1] and group_w + 2 * COMPACT_MARGIN <= FRAME_W_RANGE[1]:
            return placements, group_h, group_w
    return None


def _compose_group(children, placements, group_h, group_w):
    mask = np.zeros((group_h, group_w), dtype=bool)
    for child in children:
        r, c = placements[child]
        h, w = child.mask.shape
        mask[r : r + h, c : c + w] |= child.mask
    return mask


def _single_hole_mask(mask: np.ndarray) -> np.ndarray | None:
    holes = _component_holes(mask.shape, np.argwhere(mask))
    if len(holes) != 1:
        return None
    return holes[0]


def _fits_in_hole(hole_mask: np.ndarray, group_mask: np.ndarray, top: int, left: int) -> bool:
    h, w = group_mask.shape
    patch = hole_mask[top : top + h, left : left + w]
    return patch.shape == group_mask.shape and bool(np.all(patch[group_mask]))


def _prepare(node, compact=False):
    """Recursively build masks and place children within frames."""
    if node.kind == "blob":
        node.mask = _blob_mask()
        return node.mask is not None

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    arranged = _arrange(node.children, compact=tight)
    if arranged is None:
        return False
    placements, group_h, group_w = arranged
    group_mask = _compose_group(node.children, placements, group_h, group_w)
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False

    h_hi = min(FRAME_H_RANGE[1], min_h + (1 if tight else 2))
    w_hi = min(FRAME_W_RANGE[1], min_w + (2 if tight else 3))
    for _ in range(FRAME_SIZE_RETRIES):
        h = random.randint(min_h, h_hi)
        w = random.randint(min_w, w_hi)
        frame_mask = _frame_mask(h, w)
        if frame_mask is None:
            continue
        hole_mask = _single_hole_mask(frame_mask)
        if hole_mask is None:
            continue
        top_lo = margin
        top_hi = h - margin - group_h
        left_lo = margin
        left_hi = w - margin - group_w
        for _ in range(FRAME_POSITION_RETRIES):
            top = random.randint(top_lo, top_hi)
            left = random.randint(left_lo, left_hi)
            if not _fits_in_hole(hole_mask, group_mask, top, left):
                continue
            for child, (r, c) in placements.items():
                child.rel = (top + r, left + c)
            node.mask = frame_mask
            return True
    return False


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
    max_r = max(1, h - bh - 1)
    max_c = max(1, w - bw - 1)
    if side == "right":
        outside.rel = (random.randint(1, max_r), w + gap)
    elif side == "left":
        outside.rel = (random.randint(1, max_r), -gap - bw)
    elif side == "bottom":
        outside.rel = (h + gap, random.randint(1, max_c))
    else:
        outside.rel = (-gap - bh, random.randint(1, max_c))


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


def _make_tree():
    """Build a random containment tree (1-2 frames, 1-4 blobs, optional outside blob)."""
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
        total_blobs = len(inner.children) + len(outer_blobs)
        if total_blobs < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
            outside = _Obj("blob", outside_side=random.choice(sides))
    return root, outside


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
    raise RuntimeError("failed to generate valid 2d0172a1 input")
