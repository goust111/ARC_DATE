# Variant of 88e364bc
# Dimension: D5 — 移动对象（源）类型 (src)
# Change: control_pixels——黄色单像素（颜色4），每个区域1-2个，作为被移动的对象... → 线段形源（1xN或Nx1）

"""
Design analysis for the D5 rigid-line variant.

No source_task was provided, so the design is based only on the original solver,
generator, spec, and evaluation examples.

variant_spec:
  preserved:
    - grid size 20x20, bg=0
    - border legends still map wall color -> movement direction
    - regions are still irregular enclosed passable blobs bounded by colored walls
    - legends may still encode cardinal or diagonal directions
  changed:
    - each yellow mover is now a 4-connected straight segment of length 2 or 3
    - a yellow component translates as a rigid body; it does not split into pixels
    - a rigid segment stops when any cell would leave the region, corner-cut diagonally,
      or hit an already stopped yellow segment

behavior_trace:
  - Original local behavior:
      region direction = right, yellow cells = {(5,4), (6,4), (7,4)}.
      Old solver slides each yellow independently, so in a jagged chamber they could end at
      {(5,7), (6,8), (7,8)} and the vertical segment visually breaks apart.
  - Variant local behavior:
      the same three cells are one rigid vertical segment.
      If its top cell would hit a notch at column 7 while the lower cells could still move,
      the whole segment stops together at {(5,6), (6,6), (7,6)}.
  - Difference:
      the original rule reasons per pixel; the variant reasons per connected yellow object.
      That single change affects solver motion, reverse construction, and degeneration checks.

complexity_calibration:
  - The visible novelty must come from object rigidity, not from changing legends/walls.
  - Only two segment lengths are used: 2 and 3. This keeps the rule learnable from 2-4 examples.
  - The generator uses one segment per region for clean visuals, but the solver supports multiple
    rigid yellow components inside the same region.
  - The generator rejects samples where every rigid segment would land in the same cells under
    the old point-wise solver; at least one region must exhibit genuinely new rigid behavior.

main_audit:
  - `_legend_directions(...)` is unchanged: it depends only on border rectangles and color counts.
  - Passable-region detection still works with line segments because yellow is still passable.
  - The original `_slide_points(...)` relied on "each yellow cell is an independent mover".
    That assumption is false now: sorting and stopping must operate on yellow components, not cells.
  - Clearing old movers is still safe, but repainting must use translated component cells, not
    per-pixel stop positions.
  - Border-color lookup by majority adjacent wall color still holds because regions themselves
    are unchanged.

placement_legality:
  - A candidate segment is only valid if all its cells lie inside one region interior and form a
    straight horizontal/vertical line of length 2 or 3.
  - Output-first generation places a segment at a resting position, then reverse-translates it.
    The reversed candidate is accepted only if rigid forward simulation returns exactly to the
    chosen resting segment.
  - Regions stay disjoint from legends and from each other exactly as in the original generator.

visual_observability:
  - Every mover is visibly longer than one pixel.
  - Many samples use a segment whose final stop is caused by a partial collision: some cells could
    continue under the old rule, but the rigid object cannot. This makes the D5 change visible.
  - The generator measures `legacy_diff_regions`, the number of regions whose rigid output differs
    from the old point-wise output, and rejects degenerate samples where that metric is zero.

new_solver_assumptions_and_generator_guarantees:
  - The solver treats each 4-connected yellow component as one rigid object.
  - The generator guarantees that every yellow component is a straight length-2/3 line.
  - The solver allows multiple yellow components per region; the generator chooses one per region
    to keep sample layouts legible and avoid order ambiguities overwhelming the core D5 change.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and map legend fill color -> movement direction."""
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        if len(others) < 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _point_sort_key(point: tuple[int, int], dr: int, dc: int) -> tuple[float, int, int]:
    return (dr * point[0] + dc * point[1], point[0], point[1])


def _shape_sort_key(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[float, float, int, int]:
    projs = [dr * r + dc * c for r, c in shape]
    anchor = min(shape)
    return (max(projs), sum(projs) / len(projs), anchor[0], anchor[1])


def _connected_point_components(points: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    pending = set(points)
    comps: list[tuple[tuple[int, int], ...]] = []
    while pending:
        start = pending.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in neighbors4(r, c):
                nxt = (int(nr), int(nc))
                if nxt in pending:
                    pending.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
        comps.append(tuple(sorted(comp)))
    return comps


def _single_cell_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _shape_can_step(
    region_set: set[tuple[int, int]],
    shape: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> bool:
    for r, c in shape:
        if not _single_cell_step_valid(region_set, r, c, dr, dc):
            return False
        if (r + dr, c + dc) in blocked:
            return False
    return True


def _shape_partial_block(
    region_set: set[tuple[int, int]],
    shape: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> bool:
    movable = []
    for r, c in shape:
        ok = _single_cell_step_valid(region_set, r, c, dr, dc) and ((r + dr, c + dc) not in blocked)
        movable.append(ok)
    return any(movable) and not all(movable)


def _shift_shape(
    shape: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    steps: int = 1,
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr * steps, c + dc * steps) for r, c in shape))


def _slide_shape(
    region_set: set[tuple[int, int]],
    shape: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    cur = shape
    while _shape_can_step(region_set, cur, dr, dc, blocked):
        cur = _shift_shape(cur, dr, dc, 1)
    return cur


def _slide_shapes(
    region_set: set[tuple[int, int]],
    shapes: list[tuple[tuple[int, int], ...]],
    dr: int,
    dc: int,
) -> list[tuple[tuple[int, int], ...]]:
    shapes_sorted = sorted(shapes, key=lambda shp: _shape_sort_key(shp, dr, dc), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_shapes = []
    for shape in shapes_sorted:
        moved = _slide_shape(region_set, shape, dr, dc, occupied)
        occupied.update(moved)
        out_shapes.append(moved)
    return out_shapes


def _legacy_slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: _point_sort_key(p, dr, dc), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow line segments inside walled regions in the direction dictated by border legends.

    The crucial difference from the original task is that each 4-connected yellow component is a
    rigid object. A rigid segment advances only if every cell can advance by the same legend vector.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        shapes = _connected_point_components(movers)

        for r, c in movers:
            out[r, c] = bg

        for moved_shape in _slide_shapes(pset, shapes, dr, dc):
            for r, c in moved_shape:
                out[r, c] = mover

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.75
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (18, 85),
    2: (18, 55),
    3: (15, 38),
    4: (15, 28),
}
BLOB_GROW_RETRIES = 25

SEGMENT_LENGTHS = [2, 3]
MAX_SLIDE_STEPS = 6

GEN_RETRIES = 400


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _grow_blob(
    canvas_mask: np.ndarray,
    seed: tuple[int, int],
    target_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wc in zip(corners, wall_colors):
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wc, dr, dc)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))
    return legend_occupied, legends


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1) for wr, wc in walls):
            continue
        return cand
    return None


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

    region_colors = []
    for color in all_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for rc in region_colors:
        dr, dc = color_dir[rc]
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            continue

        walls = _blob_wall_cells(blob, h, w)
        regions.append({"color": rc, "dir": (dr, dc), "interior": blob, "walls": walls})

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {reg["color"] for reg in regions} != set(all_colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _segment_orientation(shape: tuple[tuple[int, int], ...]) -> str:
    rows = {r for r, _ in shape}
    cols = {c for _, c in shape}
    if len(rows) == 1:
        return "h"
    if len(cols) == 1:
        return "v"
    return "other"


def _is_cardinal_orthogonal(orientation: str, dr: int, dc: int) -> bool:
    if dr == 0 and dc != 0:
        return orientation == "v"
    if dc == 0 and dr != 0:
        return orientation == "h"
    return False


def _enumerate_segments(blob: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    cands = set()
    for r, c in blob:
        for length in SEGMENT_LENGTHS:
            h_seg = tuple(sorted((r, c + k) for k in range(length)))
            if all(cell in blob for cell in h_seg):
                cands.add(h_seg)
            v_seg = tuple(sorted((r + k, c) for k in range(length)))
            if all(cell in blob for cell in v_seg):
                cands.add(v_seg)
    return sorted(cands)


def _find_reverse_segment(
    blob: set[tuple[int, int]],
    out_seg: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    rng: random.Random,
) -> tuple[tuple[int, int], ...] | None:
    steps_list = list(range(1, MAX_SLIDE_STEPS + 1))
    rng.shuffle(steps_list)
    steps_list.sort(reverse=True)
    for steps in steps_list:
        cand = _shift_shape(out_seg, -dr, -dc, steps)
        if any(cell not in blob for cell in cand):
            continue
        if _slide_shape(blob, cand, dr, dc, set()) == out_seg:
            return cand
    return None


def _choose_segment_for_region(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
    require_partial: bool,
) -> dict | None:
    cands = _enumerate_segments(blob)
    rng.shuffle(cands)
    infos = []
    for seg in cands:
        if _shape_can_step(blob, seg, dr, dc, set()):
            continue
        orient = _segment_orientation(seg)
        if orient == "other":
            continue
        start = _find_reverse_segment(blob, seg, dr, dc, rng)
        if start is None:
            continue
        partial = _shape_partial_block(blob, seg, dr, dc, set())
        orth = _is_cardinal_orthogonal(orient, dr, dc)
        infos.append(
            {
                "output": seg,
                "input": start,
                "length": len(seg),
                "orientation": orient,
                "partial_stop": partial,
                "orthogonal_to_motion": orth,
            }
        )
    if not infos:
        return None

    if require_partial:
        infos = [info for info in infos if info["partial_stop"]]
        if not infos:
            return None

    rng.shuffle(infos)
    infos.sort(
        key=lambda info: (
            info["partial_stop"],
            info["orthogonal_to_motion"],
            info["length"],
            _shape_sort_key(info["output"], dr, dc),
        ),
        reverse=True,
    )
    return infos[0]


def _place_segments(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    partial_seen = False
    for idx, reg in enumerate(regions):
        blob = reg["interior"]
        dr, dc = reg["dir"]
        require_partial = (idx == len(regions) - 1) and (not partial_seen)
        chosen = _choose_segment_for_region(rng, blob, dr, dc, require_partial=require_partial)
        if chosen is None:
            return False

        for r, c in chosen["input"]:
            grid[r, c] = 4
        reg["segment_input"] = chosen["input"]
        reg["segment_output"] = chosen["output"]
        reg["segment_length"] = chosen["length"]
        reg["segment_orientation"] = chosen["orientation"]
        reg["partial_stop"] = chosen["partial_stop"]
        reg["orthogonal_to_motion"] = chosen["orthogonal_to_motion"]
        reg["legacy_diff"] = set(chosen["output"]) != set(
            _legacy_slide_points(blob, list(chosen["input"]), dr, dc)
        )
        partial_seen = partial_seen or chosen["partial_stop"]
    return partial_seen


def _build_expected_output(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    out[out == 4] = 0
    for reg in regions:
        for r, c in reg["segment_output"]:
            out[r, c] = 4
    return out


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = _build_expected_output(input_grid, regions)
    if not np.array_equal(solver_out, expected):
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not np.all(regions_mask[solver_out == 4]):
        return False

    if sum(int(reg["legacy_diff"]) for reg in regions) == 0:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_segments(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
