# Combo variant of 88e364bc (index 38)
# combo_spec: D5/3+D11/0
# Name: 轨迹涂色+线段源

"""
Design analysis for the D5/3 + D11/0 combo variant.

combo_summary:
  - Preserve the original legend-to-wall-color mapping and enclosed-region topology.
  - Replace each movable yellow singleton with a rigid straight yellow segment of length 2 or 3.
  - When that segment slides, paint every intermediate occupied cell with a trail color, so the
    output shows a thick swept band rather than only the final resting segment.

behavior_trace:
  - Original local behavior:
      region direction = right, yellow cells = {(5,4), (6,4), (7,4)} inside a jagged chamber.
      The old task treats them as three unrelated pixels. They can end at
      {(5,7), (6,8), (7,8)}, and the input segment visually breaks apart.
      The output also erases the travel history.
  - Combo behavior:
      the same three cells are one rigid vertical segment.
      If the top cell would hit a notch one step earlier than the lower cells, the whole segment
      stops together, for example at {(5,6), (6,6), (7,6)}.
      During the slide, every occupied position of that whole segment is painted. If it started at
      column 4 and stopped at column 6, the output paints columns 4 and 5 as trail and column 6
      as the final yellow segment.
  - Difference:
      D5 changes the motion primitive from per-pixel to rigid-body translation.
      D11 changes the rendering primitive from "show only final cells" to "show the swept area".
      Together they create a wide trail band whose shape depends on the rigid segment orientation.

complexity_calibration:
  - Only straight 1x2, 1x3, 2x1, or 3x1 movers are used. This is enough to visibly differ from
    the source task without creating many unrelated shape classes.
  - The trail color is fixed globally to cyan(8). No extra per-region color mapping is added.
  - The generator strongly prefers cardinal motion with segment orientation orthogonal to motion,
    because that produces a visibly thick rectangular ribbon instead of a barely longer 1-cell line.
  - The generator still allows diagonal legends, but accepted samples must include at least one
    region where the combo is visibly non-degenerate.

main_audit:
  - `_legend_directions(...)` remains valid: legends themselves are unchanged.
  - Region detection via `passable = bg or yellow` still works because trails exist only in output.
  - The original `_slide_points(...)` assumption "each yellow cell moves independently" is broken.
    The combo solver must first group yellow cells into connected components and slide each
    component rigidly.
  - Returning only final cells is no longer enough. The solver must retain the full translation
    history per rigid shape so the swept band can be painted.
  - Clearing old yellow cells is still safe, but only if the swept band is repainted afterwards and
    final yellow cells are painted last.

placement_legality:
  - Every generated mover lies entirely inside one enclosed region and forms a straight segment of
    length 2 or 3.
  - Output-first generation chooses a resting segment, reverse-translates it by 1..6 steps, and
    accepts the input only if rigid forward simulation returns exactly to the chosen resting shape.
  - Walls, legends, and region interiors remain disjoint exactly as in the source task.

visual_observability:
  - Accepted samples must contain visible trail area, not just one-step nudges.
  - Accepted samples must also contain at least one region where rigid-body motion changes the final
    result relative to the legacy point-wise slide. Otherwise the D5 half of the combo would be
    visually absent.
  - A second metric enforces a "wide trail bonus": trail area must exceed the number of translation
    steps in at least one region, which can only happen when a multi-cell segment paints a band.

visual_cleanliness:
  - The input still contains only walls, legends, background, and yellow start segments.
  - The output adds only one new interior mark type: cyan trail.
  - Final yellows overwrite any trail on their endpoint cells, so each cell has a single readable
    role in the final image.

new_solver_assumptions_and_generator_guarantees:
  - Solver assumption: each interior yellow component should be interpreted as one rigid mover.
    Generator guarantee: every component is a straight length-2/3 segment.
  - Solver assumption: trail color does not appear inside regions in the input.
    Generator guarantee: cyan(8) is reserved for solver-painted trails and never appears in input.
  - Solver assumption: motion ordering is by farthest projection along the legend direction.
    Generator guarantee: it places one segment per region, so no ambiguous multi-segment collisions
    are needed to communicate the combo rule.
  - Solver assumption: diagonal steps cannot cut corners.
    Generator guarantee: reverse placement validates the same rigid motion predicate used by the
    solver.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER_COLOR = 4
TRAIL_COLOR = 8

# Legend
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 9]
CARDINAL_PROB = 0.85
LEGEND_BUFFER = 1

# Regions
REGIONS_PER_COLOR = (1, 2)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (20, 85),
    2: (18, 55),
    3: (16, 36),
    4: (14, 28),
}
BLOB_GROW_RETRIES = 25

# Segments
SEGMENT_LENGTHS = [2, 3]
MAX_SLIDE_STEPS = 6

# Visibility / anti-degeneracy constraints
MIN_NEW_TRAIL_CELLS = 6
MIN_LONGEST_STEPS = 2
MIN_WIDE_TRAIL_BONUS = 2

GEN_RETRIES = 600


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


def _point_sort_key(point: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int]:
    return (dr * point[0] + dc * point[1], point[0], point[1])


def _shape_sort_key(
    shape: tuple[tuple[int, int], ...], dr: int, dc: int
) -> tuple[float, float, int, int]:
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


def _slide_shape_with_history(
    region_set: set[tuple[int, int]],
    shape: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], list[tuple[tuple[int, int], ...]]]:
    cur = shape
    history = [shape]
    while _shape_can_step(region_set, cur, dr, dc, blocked):
        cur = _shift_shape(cur, dr, dc, 1)
        history.append(cur)
    return cur, history


def _slide_shapes_with_history(
    region_set: set[tuple[int, int]],
    shapes: list[tuple[tuple[int, int], ...]],
    dr: int,
    dc: int,
) -> list[tuple[tuple[tuple[int, int], ...], list[tuple[tuple[int, int], ...]]]]:
    shapes_sorted = sorted(shapes, key=lambda shp: _shape_sort_key(shp, dr, dc), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out = []
    for shape in shapes_sorted:
        final_shape, history = _slide_shape_with_history(region_set, shape, dr, dc, occupied)
        occupied.update(final_shape)
        out.append((final_shape, history))
    return out


def _trail_cells_from_history(history: list[tuple[tuple[int, int], ...]]) -> set[tuple[int, int]]:
    trail: set[tuple[int, int]] = set()
    for shape in history[:-1]:
        trail.update(shape)
    return trail


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


def _detect_regions(
    g: np.ndarray,
) -> list[tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]]:
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    passable = (g == bg) | (g == MOVER_COLOR)
    found = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER_COLOR]
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
                if v == bg or v == MOVER_COLOR:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        found.append((pset, movers, dirs[border_col]))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide rigid yellow line segments inside colored chambers and paint the swept band they cover.

    Input structure:
    - The background is color 0, while closed regions are enclosed by colored wall loops.
    - Border legends are filled rectangles on the outer edge; each legend's fill color matches the
      wall color of the regions it controls and its inner red marker encodes a direction.
    - Inside each enclosed region, yellow cells form one or more short straight segments rather than
      isolated pixels.

    Rule:
    - For each enclosed region, match its dominant wall color to the legend with the same fill color
      to obtain a direction vector.
    - Treat every 4-connected yellow component as one rigid segment and slide it maximally in that
      direction, stopping when any part would exit the region, cut a diagonal corner, or hit a
      segment that has already come to rest farther along the same direction.
    - Paint every non-final occupied cell along the segment's motion history cyan(8), then paint the
      final resting segment yellow(4).

    Key insight:
    - The original color-to-direction decoding still applies, but the mover is now an object, not a
      set of independent pixels. Because the whole segment sweeps through space, the output reveals a
      thick ribbon whose width comes from the segment's own shape.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    out = g.copy()
    for region_set, movers, (dr, dc) in _detect_regions(g):
        shapes = _connected_point_components(movers)
        for r, c in movers:
            out[r, c] = bg

        slides = _slide_shapes_with_history(region_set, shapes, dr, dc)
        for _final_shape, history in slides:
            for r, c in _trail_cells_from_history(history):
                out[r, c] = TRAIL_COLOR
        for final_shape, _history in slides:
            for r, c in final_shape:
                out[r, c] = MOVER_COLOR
    return out


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

    for region_color in region_colors:
        dr, dc = color_dir[region_color]
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
        regions.append({"color": region_color, "dir": (dr, dc), "interior": blob, "walls": walls})

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


def _reverse_segment_infos(
    blob: set[tuple[int, int]],
    out_seg: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
) -> list[dict]:
    infos = []
    for steps in range(MAX_SLIDE_STEPS, 0, -1):
        cand = _shift_shape(out_seg, -dr, -dc, steps)
        if any(cell not in blob for cell in cand):
            continue
        final_shape, history = _slide_shape_with_history(blob, cand, dr, dc, set())
        if final_shape != out_seg:
            continue
        trail_cells = _trail_cells_from_history(history)
        infos.append(
            {
                "input": cand,
                "steps": steps,
                "history": history,
                "trail_cell_count": len(trail_cells),
                "trail_cells": trail_cells,
                "wide_trail_bonus": len(trail_cells) - steps,
            }
        )
    return infos


def _choose_segment_for_region(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
    require_combo: bool,
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
        rev_infos = _reverse_segment_infos(blob, seg, dr, dc)
        for rev in rev_infos:
            partial = _shape_partial_block(blob, seg, dr, dc, set())
            orth = _is_cardinal_orthogonal(orient, dr, dc)
            legacy_final = set(_legacy_slide_points(blob, list(rev["input"]), dr, dc))
            legacy_diff = set(seg) != legacy_final
            wide_band = rev["wide_trail_bonus"] > 0
            infos.append(
                {
                    "output": seg,
                    "input": rev["input"],
                    "history": rev["history"],
                    "steps": rev["steps"],
                    "trail_cells": rev["trail_cells"],
                    "trail_cell_count": rev["trail_cell_count"],
                    "wide_trail_bonus": rev["wide_trail_bonus"],
                    "length": len(seg),
                    "orientation": orient,
                    "partial_stop": partial,
                    "orthogonal_to_motion": orth,
                    "legacy_diff": legacy_diff,
                    "wide_band": wide_band,
                    "combo_salient": legacy_diff and wide_band,
                }
            )
    if not infos:
        return None

    if require_combo:
        filtered = [info for info in infos if info["combo_salient"]]
        if not filtered:
            return None
        infos = filtered

    rng.shuffle(infos)
    infos.sort(
        key=lambda info: (
            info["combo_salient"],
            info["orthogonal_to_motion"],
            info["partial_stop"],
            info["wide_band"],
            info["legacy_diff"],
            info["length"],
            info["trail_cell_count"],
            info["steps"],
            _shape_sort_key(info["output"], dr, dc),
        ),
        reverse=True,
    )
    return infos[0]


def _place_segments(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    combo_seen = False
    for idx, reg in enumerate(regions):
        blob = reg["interior"]
        dr, dc = reg["dir"]
        require_combo = (idx == len(regions) - 1) and (not combo_seen)
        chosen = _choose_segment_for_region(rng, blob, dr, dc, require_combo=require_combo)
        if chosen is None:
            return False

        for r, c in chosen["input"]:
            grid[r, c] = MOVER_COLOR
        reg["segment_input"] = chosen["input"]
        reg["segment_output"] = chosen["output"]
        reg["history"] = chosen["history"]
        reg["steps"] = chosen["steps"]
        reg["trail_cells"] = chosen["trail_cells"]
        reg["trail_cell_count"] = chosen["trail_cell_count"]
        reg["wide_trail_bonus"] = chosen["wide_trail_bonus"]
        reg["segment_length"] = chosen["length"]
        reg["segment_orientation"] = chosen["orientation"]
        reg["partial_stop"] = chosen["partial_stop"]
        reg["orthogonal_to_motion"] = chosen["orthogonal_to_motion"]
        reg["legacy_diff"] = chosen["legacy_diff"]
        reg["wide_band"] = chosen["wide_band"]
        reg["combo_salient"] = chosen["combo_salient"]
        combo_seen = combo_seen or chosen["combo_salient"]
    return combo_seen


def _build_expected_output(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    out[out == MOVER_COLOR] = 0
    for reg in regions:
        for r, c in reg["trail_cells"]:
            out[r, c] = TRAIL_COLOR
    for reg in regions:
        for r, c in reg["segment_output"]:
            out[r, c] = MOVER_COLOR
    return out


def _behavior_metrics_from_regions(regions: list[dict]) -> dict[str, int]:
    return {
        "segment_count": len(regions),
        "new_trail_cells": int(sum(reg["trail_cell_count"] for reg in regions)),
        "longest_steps": int(max([0] + [reg["steps"] for reg in regions])),
        "wide_trail_bonus": int(sum(reg["wide_trail_bonus"] for reg in regions)),
        "legacy_diff_regions": int(sum(int(reg["legacy_diff"]) for reg in regions)),
        "combo_salient_regions": int(sum(int(reg["combo_salient"]) for reg in regions)),
    }


def measure_behavior(input_grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a sample exhibits the intended combo behavior."""
    g = np.asarray(input_grid, dtype=int)
    out = main(g)
    metrics = {
        "segment_count": 0,
        "new_trail_cells": int(((out == TRAIL_COLOR) & (g != TRAIL_COLOR)).sum()),
        "longest_steps": 0,
        "wide_trail_bonus": 0,
        "legacy_diff_regions": 0,
        "combo_salient_regions": 0,
    }
    for region_set, movers, (dr, dc) in _detect_regions(g):
        shapes = _connected_point_components(movers)
        slides = _slide_shapes_with_history(region_set, shapes, dr, dc)
        metrics["segment_count"] += len(shapes)
        for shape, (final_shape, history) in zip(shapes, slides):
            steps = len(history) - 1
            trail_count = len(_trail_cells_from_history(history))
            wide_bonus = trail_count - steps
            legacy_final = set(_legacy_slide_points(region_set, list(shape), dr, dc))
            legacy_diff = legacy_final != set(final_shape)
            combo_salient = legacy_diff and wide_bonus > 0
            metrics["longest_steps"] = max(metrics["longest_steps"], steps)
            metrics["wide_trail_bonus"] += wide_bonus
            metrics["legacy_diff_regions"] += int(legacy_diff)
            metrics["combo_salient_regions"] += int(combo_salient)
    return metrics


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = _build_expected_output(input_grid, regions)
    if not np.array_equal(solver_out, expected):
        return False

    region_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            region_mask[r, c] = True
    relevant_mask = (solver_out == MOVER_COLOR) | ((solver_out == TRAIL_COLOR) & (input_grid != TRAIL_COLOR))
    if not bool(np.all(region_mask[relevant_mask])):
        return False

    metrics = _behavior_metrics_from_regions(regions)
    if metrics["new_trail_cells"] < MIN_NEW_TRAIL_CELLS:
        return False
    if metrics["longest_steps"] < MIN_LONGEST_STEPS:
        return False
    if metrics["wide_trail_bonus"] < MIN_WIDE_TRAIL_BONUS:
        return False
    if metrics["legacy_diff_regions"] == 0:
        return False
    if metrics["combo_salient_regions"] == 0:
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
