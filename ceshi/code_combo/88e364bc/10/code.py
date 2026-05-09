# Combo variant of 88e364bc (index 10)
# combo_spec: D5/3+D9/0
# Name: 线段源+仅基数方向

"""
Design analysis for the D5/3 + D9/0 fused variant.

This module starts from task 88e364bc and fuses two changes:
  - D5/3: movers are rigid yellow line segments instead of independent yellow pixels
  - D9/0: legends encode only cardinal directions, so all motion is up/down/left/right

fusion_spec:
  preserved:
    - grid size stays 20x20 with background color 0
    - border legends still map wall color -> motion direction
    - enclosed irregular wall-bounded regions are still the active chambers
    - yellow remains the mover color (4), wall colors remain distinct from legend dot colors
  changed:
    - each yellow mover is a straight 4-connected segment of length 2 or 3
    - each segment translates as one rigid body; its cells never separate
    - only 5x5 cardinal legends are used; diagonal legends are removed
    - generated segments are always orthogonal to their motion direction:
      horizontal segments slide vertically, vertical segments slide horizontally

behavior_trace:
  - Original task:
      a right-pointing legend can act on three yellow cells at {(5,4), (6,4), (7,4)}.
      The solver sorts those cells individually and each one slides until blocked, so a jagged
      right wall might produce {(5,7), (6,8), (7,8)}.
  - Fused task:
      the same three cells form one vertical segment. With a right-pointing legend, the whole
      segment must move horizontally. If the top cell would hit a notch at column 7 first, the
      entire segment stops together at {(5,6), (6,6), (7,6)} even though lower cells still had
      room under the original per-pixel rule.
  - Difference:
      D9/0 removes diagonal ambiguity from the legends, while D5/3 makes the visible novelty come
      from rigid-body stopping inside irregular chambers. The human only needs to infer
      "legend color picks the chamber, legend arrow gives one of four directions, the yellow bar
      slides sideways/up-down as a single piece."

complexity_calibration:
  - Segment lengths are restricted to 2 or 3 so examples stay readable.
  - One segment is placed per region in the generator. This avoids clutter while still allowing
    the solver to support multiple yellow components if such an input appears.
  - All legends are 5x5 cardinal patterns, so the fusion does not introduce mixed legend formats.
  - The generator rejects degenerate samples unless at least one region exhibits a rigid-stop
    outcome that differs from the legacy point-wise slide behavior.

main_audit:
  - `_legend_directions(...)` remains valid because legend detection depends only on border
    rectangles and color counts. The fused change only filters outputs to cardinal vectors.
  - Region discovery via passable connected components still works because yellow segments are
    passable just like yellow pixels were.
  - The original `_slide_points(...)` assumed yellow cells are independent movers. That assumption
    is false after D5/3, so the fused solver replaces it with rigid component motion.
  - Clearing old yellow cells before repainting remains valid, but repainting must use translated
    segment cells rather than per-pixel destinations.
  - The border-color majority heuristic still holds because the wall layout itself is unchanged.

placement_legality:
  - A candidate mover is legal only if all of its cells lie inside one enclosed region, form a
    straight horizontal/vertical segment, and have length 2 or 3.
  - Output-first placement chooses a resting output segment first, then reverse-translates it by
    1..6 steps opposite the legend direction. The candidate input is accepted only if forward rigid
    simulation returns exactly to the chosen output segment.
  - Regions are generated disjoint from legends and from each other with a one-cell buffer around
    each wall ring, preserving enclosure clarity.

visual_observability:
  - Every mover is visibly a bar, not a single dot.
  - Because motion is cardinal-only, a vertical bar moving left/right or a horizontal bar moving
    up/down is easy to notice from 2-4 examples.
  - The generator biases toward "partial block" resting states where one end of the bar hits an
    irregular notch first. That makes the rigid-body rule visibly different from the original
    point-wise slide.

visual_cleanliness:
  - Each region contains at most one yellow segment, so there is no ambiguous overlap between
    multiple moving objects inside one chamber.
  - Legends stay at corners with padding, walls stay one cell thick, and yellow bars appear only in
    region interiors. This keeps color roles visually separated.

new_solver_assumptions_and_generator_guarantees:
  - The solver treats each 4-connected yellow component as one rigid object and expects only
    cardinal legend directions.
  - The generator guarantees every yellow component is a straight length-2/3 segment orthogonal to
    its motion direction.
  - The solver can move any rigid 4-connected yellow component, but the generator uses only the
    simpler straight-bar subset so the rule is inferable for humans.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
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

GEN_RETRIES = 450

LAST_GENERATION_INFO: dict[str, int] = {}


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and keep only cardinal center->indicator directions."""
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

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
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
        if abs(dr) + abs(dc) != 1:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _point_sort_key(point: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int]:
    return (dr * point[0] + dc * point[1], point[0], point[1])


def _shape_sort_key(
    shape: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
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


def _shape_orientation(shape: tuple[tuple[int, int], ...]) -> str:
    rows = {r for r, _ in shape}
    cols = {c for _, c in shape}
    if len(rows) == 1:
        return "h"
    if len(cols) == 1:
        return "v"
    return "other"


def _is_orthogonal_to_motion(orientation: str, dr: int, dc: int) -> bool:
    if dr == 0 and dc != 0:
        return orientation == "v"
    if dc == 0 and dr != 0:
        return orientation == "h"
    return False


def _single_cell_step_valid(
    region_set: set[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> bool:
    return (r + dr, c + dc) in region_set


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
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow rigid line segments inside closed wall regions using cardinal legend arrows.

    Input structure:
    - Background cells are color 0, and each active chamber is an enclosed passable region bounded
      by one dominant wall color.
    - Border legends are 5x5 filled rectangles placed on grid corners; their fill color matches the
      wall color they control, and their blue/red marker pattern encodes one of four cardinal
      directions.
    - Yellow movers are straight 2- or 3-cell horizontal/vertical segments inside the chambers.

    Rule:
    - For each enclosed region, find the matching legend by the region's surrounding wall color.
    - Read the legend's center-to-indicator offset to get a cardinal slide direction.
    - Treat each 4-connected yellow component as one rigid object and translate it repeatedly in
      that direction until any part would hit a wall or an already stopped yellow object.

    Key insight:
    - The legend color still selects which arrow applies to which chamber, but the yellow object is
      no longer a bag of independent pixels: a full bar moves together and can stop early when only
      one end reaches the far-side obstacle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one cardinal border legend"

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
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wc, dr, dc)
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
        regions.append({"color": rc, "dir": color_dir[rc], "interior": blob, "walls": walls})

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
    require_visible: bool,
) -> dict | None:
    cands = _enumerate_segments(blob)
    rng.shuffle(cands)
    infos = []
    for seg in cands:
        orient = _shape_orientation(seg)
        if orient == "other" or not _is_orthogonal_to_motion(orient, dr, dc):
            continue
        if _shape_can_step(blob, seg, dr, dc, set()):
            continue
        start = _find_reverse_segment(blob, seg, dr, dc, rng)
        if start is None:
            continue
        partial = _shape_partial_block(blob, seg, dr, dc, set())
        legacy_diff = set(seg) != set(_legacy_slide_points(blob, list(start), dr, dc))
        infos.append(
            {
                "output": seg,
                "input": start,
                "length": len(seg),
                "orientation": orient,
                "orthogonal_to_motion": True,
                "partial_stop": partial,
                "legacy_diff": legacy_diff,
            }
        )
    if not infos:
        return None

    visible_infos = [info for info in infos if info["partial_stop"] and info["legacy_diff"]]
    if require_visible:
        if not visible_infos:
            return None
        infos = visible_infos
    elif visible_infos:
        infos = visible_infos

    lengths = sorted({info["length"] for info in infos})
    chosen_length = rng.choice(lengths)
    same_length_infos = [info for info in infos if info["length"] == chosen_length]

    rng.shuffle(same_length_infos)
    same_length_infos.sort(
        key=lambda info: (
            info["legacy_diff"],
            info["partial_stop"],
            _shape_sort_key(info["output"], dr, dc),
        ),
        reverse=True,
    )
    return same_length_infos[0]


def _place_segments(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    visible_seen = False
    for idx, reg in enumerate(regions):
        blob = reg["interior"]
        dr, dc = reg["dir"]
        require_visible = (idx == len(regions) - 1) and (not visible_seen)
        chosen = _choose_segment_for_region(rng, blob, dr, dc, require_visible=require_visible)
        if chosen is None:
            return False

        for r, c in chosen["input"]:
            grid[r, c] = 4
        reg["segment_input"] = chosen["input"]
        reg["segment_output"] = chosen["output"]
        reg["segment_length"] = chosen["length"]
        reg["segment_orientation"] = chosen["orientation"]
        reg["orthogonal_to_motion"] = chosen["orthogonal_to_motion"]
        reg["partial_stop"] = chosen["partial_stop"]
        reg["legacy_diff"] = chosen["legacy_diff"]
        visible_seen = visible_seen or (chosen["partial_stop"] and chosen["legacy_diff"])
    return visible_seen


def _build_expected_output(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    out[out == 4] = 0
    for reg in regions:
        for r, c in reg["segment_output"]:
            out[r, c] = 4
    return out


def _metrics_from_regions(regions: list[dict]) -> dict[str, int]:
    return {
        "regions": len(regions),
        "segments": len(regions),
        "orthogonal_regions": sum(int(reg["orthogonal_to_motion"]) for reg in regions),
        "partial_stop_regions": sum(int(reg["partial_stop"]) for reg in regions),
        "legacy_diff_regions": sum(int(reg["legacy_diff"]) for reg in regions),
        "segment_len_2": sum(int(reg["segment_length"] == 2) for reg in regions),
        "segment_len_3": sum(int(reg["segment_length"] == 3) for reg in regions),
    }


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = _build_expected_output(input_grid, regions)
    if not np.array_equal(solver_out, expected):
        return False

    regions_mask = np.zeros(input_grid.shape, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    if not np.all(regions_mask[solver_out == 4]):
        return False

    metrics = _metrics_from_regions(regions)
    if metrics["orthogonal_regions"] != metrics["regions"]:
        return False
    if metrics["legacy_diff_regions"] == 0:
        return False
    return True


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, dict[str, int]] | None:
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

    metrics = _metrics_from_regions(regions)
    metrics["legend_count"] = len(legends)
    metrics["cardinal_legends"] = len(legends)
    return input_grid, metrics


def generate_input() -> np.ndarray:
    global LAST_GENERATION_INFO
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            result = _gen_attempt(rng)
        except Exception:
            continue
        if result is not None:
            grid, metrics = result
            LAST_GENERATION_INFO = metrics
            return grid
    raise RuntimeError("generate_input: exceeded retries")
