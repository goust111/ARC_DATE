# Combo variant of 88e364bc (index 17)
# combo_spec: D4/1+D14/0
# Name: 滑动变色+特定颜色停止

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste

"""
Design analysis for combo D4/1 + D14/0
======================================

a) Concrete behavior and trace
------------------------------
- Base task:
  legends on the outer border encode a direction for each wall color; yellow pixels slide inside the
  matching enclosed region until the next step would leave the region, then they stack at the far wall.
- Combo task:
  each yellow-bearing region also contains one interior stop marker (fixed color 1 / blue). The front
  yellow slides onto that marker and stops there even though more empty cells remain toward the real
  wall. After every yellow finishes sliding, it is recolored to that region's wall color.
- Small lane trace:
  `A . . M . . W`
  where `A` is an input yellow, `M` is the blue stop marker, and `W` is the wall ahead.
  Original output: `.` `.` `.` `M` `.` `Y` `W` if the yellow keeps going to the far end.
  Combo output:    `.` `.` `.` `C` `.` `.` `W` where `C` is the wall color, because the yellow stops
  on `M` and then changes color. If a second yellow exists, it queues immediately behind that `C`.
- Visible difference:
  the front terminal is strictly before the real wall, and the output contains no yellow at all; it
  shows new wall-colored cells inside the region instead.

b) Complexity calibration
-------------------------
- Marker color is fixed globally to blue (1). Reusing one marker color across all samples keeps the new
  stop mechanic visually stable and avoids introducing an additional color-mapping puzzle.
- There is exactly one marker per active region. More markers would create route-choice ambiguity that
  is not part of this combo.
- Wall colors stay in the original non-{0,1,2,4} palette, so the recolor target is always tied to an
  already visible structure: the enclosing wall. The generator ensures each sample repeats this relation
  across 1-4 regions rather than hiding it in a single special case.

c) Hidden-assumption audit against the original main()
------------------------------------------------------
1. `passable = (g == bg) | (g == mover)` no longer holds:
   the blue stop marker lies inside the traversable interior, so marker cells must belong to the same
   component as background and yellow cells.
2. Original slide termination only looked for boundary exit or prior occupancy:
   the combo adds a third stop cause, "current cell is a stop marker". This must happen after moving
   onto the marker, not before, otherwise the yellow would stop one cell too early.
3. Original output repaint assumed final cells remain yellow:
   the combo invalidates that and instead repaints every final cell to the region border color.
4. Original border-color counting ignored only bg/yellow:
   with interior markers present, the marker color must also be excluded so it is never mistaken for a
   wall color.
5. Original oracle relied on "all non-bg, non-yellow structure is unchanged":
   not true anymore, because used markers disappear and become wall-colored interior cells. The oracle
   must compare against an explicitly built expected output.

d) Placement legality
---------------------
- Legends are still buffered away from region-growth space.
- Regions are still generated as enclosed irregular blobs wrapped by one-cell-thick walls, separated by
  a one-cell reserve.
- A marker is placed only on cells with:
  - at least one valid reverse-slide source behind it
  - at least `MIN_MARKER_FORWARD_GAP` open steps remaining toward the true boundary
- This guarantees the stop marker is reachable, does not overlap walls or legends, and changes the
  physical destination relative to the original boundary-collision rule.

e) Visual observability
-----------------------
- Every generated region contains both a marker and at least one yellow, so the combo mechanic appears
  in every active object instead of only occasionally.
- The forward-gap constraint ensures the marker is not silently equivalent to the far wall.
- Because the output contains no yellow at all, the recolor effect is unavoidable and easy to compare
  with the input.

f) Visual cleanliness
---------------------
- The input uses only one extra interior symbol (the blue marker), and the output removes it by
  overwriting it with the wall color. That keeps the scene layered but not cluttered.
- When a region has two yellows, the second final cell is immediately behind the marker along the slide
  axis. This produces a compact short wall-colored queue instead of scattered same-color noise.

g) Change inferability
----------------------
- The recolor rule is not an arbitrary palette remap: the final color is always the color of the region
  boundary that already controls the direction legend. Human solvers can infer "same wall color" from
  the enclosing structure in a few examples.
- The generator therefore avoids regions with no yellow and avoids marker placements that would leave the
  combo visually identical to the source task.

New solver/generator contract introduced by the combo
-----------------------------------------------------
- Each yellow-bearing enclosed region contains exactly one blue stop marker.
- That marker lies in the same enclosed passable component as the yellows.
- The front yellow always lands on the marker, and the marker always sits at least two steps before the
  real boundary in the movement direction.
- `main()` assumes those properties; `generate_input()` enforces them explicitly.
"""


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (15, 85),
    2: (15, 55),
    3: (12, 35),
    4: (12, 25),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
MIN_MARKER_FORWARD_GAP = 2
MARKER_CANDIDATE_LIMIT = 20
GEN_RETRIES = 300

BG = 0
STOP_MARKER = 1
LEGEND_INDICATOR = 2
MOVER = 4


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legend rectangles and decode their directions."""
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
        others.sort()
        if len(others) != 2:
            continue

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
            center_col = others[1][1]

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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _slide_points(
    region_set: set[tuple[int, int]],
    stop_cells: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    """Slide movers until a wall, a prior mover, or a stop marker is reached."""
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        if (r, c) in stop_cells:
            occupied.add((r, c))
            out_pts.append((r, c))
            continue
        while True:
            if not _slide_step_valid(region_set, r, c, dr, dc):
                break
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
            if (r, c) in stop_cells:
                break
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _region_border_color(
    g: np.ndarray,
    pts: np.ndarray,
    passable_set: set[tuple[int, int]],
    bg: int,
) -> int | None:
    h, w = g.shape
    border_counts: dict[int, int] = {}
    for r, c in pts:
        for nr, nc in neighbors4(int(r), int(c)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in passable_set:
                continue
            v = int(g[nr, nc])
            if v in (bg, MOVER, STOP_MARKER):
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    if not border_counts:
        return None
    return max(border_counts, key=border_counts.get)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels through enclosed regions using border legends, but stop on interior markers and recolor the result.

    Input structure:
    - The background is the dominant color. Colored wall loops enclose passable regions made of background cells.
    - Small rectangular legends sit on the outer border; each legend's dominant fill color matches one wall color and its inner red indicator encodes a slide direction.
    - Active regions contain yellow movers plus one blue stop marker somewhere inside the same enclosed passable area.

    Rule:
    - For each enclosed region, infer its wall color from the surrounding boundary and use the matching legend to get the movement vector.
    - Slide all yellow cells in that direction using the original far-to-near stacking order, but stop immediately when a yellow lands on the blue marker; later yellows may stack behind the earlier ones.
    - Clear the original yellow cells, and paint every final resting cell with that region's wall color instead of yellow.

    Key insight:
    - The legend still tells you which way each region moves, but the true destination is no longer the far wall alone: the interior blue marker becomes the front stop, and the final color is recovered from the enclosing wall.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == MOVER) | (g == STOP_MARKER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        stop_cells = {(int(r), int(c)) for r, c in pts if int(g[r, c]) == STOP_MARKER}
        if not stop_cells:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        border_col = _region_border_color(g, pts, pset, bg)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg

        new_pts = _slide_points(pset, stop_cells, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = border_col

    return out


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = STOP_MARKER
    g[1, 2] = STOP_MARKER
    g[1, 3] = LEGEND_INDICATOR
    g[3, 1] = STOP_MARKER
    g[3, 2] = STOP_MARKER
    g[3, 3] = LEGEND_INDICATOR
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = STOP_MARKER
    g[2, 1] = STOP_MARKER
    g[2, 4] = STOP_MARKER
    g[3, 2] = STOP_MARKER
    g[3, 5] = LEGEND_INDICATOR
    g[4, 3] = LEGEND_INDICATOR
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


def _run_length(region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int) -> int:
    r, c = start
    steps = 0
    while _slide_step_valid(region_set, r, c, dr, dc):
        r += dr
        c += dc
        steps += 1
    return steps


def _forward_slide(
    region_set: set[tuple[int, int]],
    stop_cells: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    if (r, c) in stop_cells:
        return (r, c)
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
        if (r, c) in stop_cells:
            break
    return (r, c)


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
    for c in all_colors:
        region_colors.extend([c] * rng.randint(*REGIONS_PER_COLOR))
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
    labeled, n_comp = label_mask(grid == BG, connectivity=4)
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


def _marker_candidates(blob: set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    scored = []
    for cell in blob:
        forward_gap = _run_length(blob, cell, dr, dc)
        backward_gap = _run_length(blob, cell, -dr, -dc)
        if forward_gap < MIN_MARKER_FORWARD_GAP or backward_gap < 1:
            continue
        scored.append((forward_gap, backward_gap, cell))
    scored.sort(key=lambda t: (t[0], t[1], -(dr * t[2][0] + dc * t[2][1])), reverse=True)
    return [cell for _, _, cell in scored[:MARKER_CANDIDATE_LIMIT]]


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    stop_cells: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    candidates = []
    for steps in range(1, MAX_SLIDE_STEPS + 1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in blocked or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        start = (r, c)
        if _forward_slide(blob, stop_cells, start, dr, dc, blocked) == out_pos:
            candidates.append((steps, start))
    if not candidates:
        return None
    target = rng.randint(1, MAX_SLIDE_STEPS)
    candidates.sort(key=lambda t: (abs(t[0] - target), -t[0]))
    best_dist = abs(candidates[0][0] - target)
    tied = [start for steps, start in candidates if abs(steps - target) == best_dist]
    return tied[rng.randrange(len(tied))]


def _try_place_region_objects(rng: random.Random, reg: dict) -> bool:
    blob = reg["interior"]
    dr, dc = reg["dir"]
    marker_cells = _marker_candidates(blob, dr, dc)
    if not marker_cells:
        return False

    rng.shuffle(marker_cells)
    for marker in marker_cells:
        output_positions = [marker]
        pred = (marker[0] - dr, marker[1] - dc)
        if pred in blob and rng.choice(YELLOWS_PER_REGION) == 2:
            output_positions.append(pred)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        yellows_input = []
        success = True

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(
                blob,
                {marker},
                out_pos,
                dr,
                dc,
                already_stopped,
                yellows_input,
                rng,
            )
            if candidate is None:
                success = False
                break
            yellows_input.append(candidate)
            already_stopped.add(out_pos)
        if not success:
            continue

        reg["marker"] = marker
        reg["input_yellows"] = yellows_input
        reg["output_cells"] = out_sorted
        reg["marker_forward_gap"] = _run_length(blob, marker, dr, dc)
        return True
    return False


def _place_markers_and_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        if not _try_place_region_objects(rng, reg):
            return False
        mr, mc = reg["marker"]
        grid[mr, mc] = STOP_MARKER
        for r, c in reg["input_yellows"]:
            if (r, c) == (mr, mc):
                return False
            grid[r, c] = MOVER
    return True


def _expected_output(input_grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    expected = input_grid.copy()
    for reg in regions:
        for r, c in reg["input_yellows"]:
            expected[r, c] = BG
        for r, c in reg["output_cells"]:
            expected[r, c] = reg["color"]
    return expected


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    expected = _expected_output(input_grid, regions)
    return np.array_equal(solver_out, expected)


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
    if not _place_markers_and_yellows(rng, regions, input_grid):
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
