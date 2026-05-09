# Combo variant of 88e364bc (index 2)
# combo_spec: D1/2+D5/1
# Name: 颜色序列图例+多色移动像素
#
# Design analysis for this combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Backbone kept from the original task:
#      - The grid is still 20x20 with black background 0.
#      - Thin colored walls still create enclosed regions.
#      - Single-pixel movers still slide inside their own enclosed region until they hit
#        the far wall; diagonal motion still forbids corner cutting.
#    - D1/2 change applied concretely:
#      - Border legends are no longer filled rectangles.
#      - Each legend is now a 7-cell border-touching path:
#        straight 1x7 / 7x1 for cardinal directions, or a 4x4 L path for diagonal ones.
#      - The ordered color sequence is fixed as `m,m,m,1,1,2,2` from tail to head,
#        where `m` is the mover color controlled by that legend.
#      - Direction is `sign(path_end - path_start)`.
#    - D5/1 change applied concretely:
#      - Movers are not all yellow anymore. We use up to three ARC mover colors:
#        yellow=4, pink=6, orange=7.
#      - Each mover color has its own border legend, so direction lookup is by mover color
#        rather than by wall color.
#      - The same enclosed region can contain multiple mover colors; each color slides along
#        its own legend direction and stacks only against movers of the same color group.
#    - Example trace:
#      - Suppose yellow has a right-pointing strip legend, pink has an up-left L legend,
#        and orange has a downward strip legend.
#      - A region containing yellow+pink+orange movers will split them on output:
#        yellow piles against the right wall, pink against the upper-left reachable edge,
#        orange against the bottom wall.
#      - Original task behavior was "one region -> one direction". Here the region only
#        provides the walkable container; color decides which direction to use.
# b) Complexity calibration
#    - Only 2-3 mover colors are active per sample, so the rule is visible without becoming
#      a large hidden mapping problem.
#    - The legend encoding uses one fixed run pattern and only 8 possible directions.
#    - Each region gets 2-3 mover colors, making the fusion visible in every successful
#      sample instead of hiding it in edge cases.
# c) Hidden-assumption audit of the original main()
#    - Original assumption `mover = 4` is false now; solver must recognize all mover colors.
#    - Original assumption "border legend is a filled rectangle" is false; legend parsing
#      must reconstruct a 7-cell ordered path.
#    - Original assumption "region border color chooses the direction" is false here;
#      wall color only identifies a closed container.
#    - Original assumption `passable = bg or yellow` is false; all non-legend movers must be
#      treated as passable so they do not split a region.
#    - Original repaint logic "clear yellow then write back yellow" is false; solver must
#      preserve each mover's color.
#    - New interaction introduced by the fusion:
#      - Different mover colors in one region can have different directions.
#      - The solver therefore processes movers grouped by color, using one `_slide_points`
#        call per color.
# d) Placement legality
#    - Legends are pasted into distinct corners with a 1-cell buffer around their bbox.
#    - Walls are grown strictly away from border legends and away from the outer border.
#    - Only one mover per color is placed in a region, which avoids ambiguous same-color
#      collisions while still letting multiple colors coexist inside one region.
#    - Generator accepts a region only if chosen input cells reverse-slide back to the
#      intended output cells and all final cells are pairwise distinct.
# e) Visual observability
#    - A human can read the sequence legends directly:
#      - a straight colored strip means a cardinal direction;
#      - an L path reveals both row and column signs for a diagonal direction.
#    - Matching between legend and mover is visible because the legend tail uses the same
#      mover color that appears inside the regions.
#    - Each successful sample includes at least one region with at least two mover colors,
#      so "same region, different colors, different destinations" is observable from a
#      small number of examples.
# f) Visual cleanliness
#    - Legends occupy only 7 colored cells each and stay in corners.
#    - Regions contain at most one pixel of each mover color, so no color clutter or thick
#      stacks appear inside a single chamber.
# g) Change inferability
#    - No invisible color ranking is used. The legend tail color is the explicit key.
#    - Wall colors are not secretly remapped; they remain simple fences.
#    - Marker colors 1 and 2 are fixed across samples, which stabilizes the sequence reading.
# h) New solver assumptions introduced here
#    - Every active legend is a simple 7-cell 4-connected path that can be oriented to the
#      exact sequence `m,m,m,1,1,2,2`.
#    - Every active mover color has exactly one legend in the sample.
#    - Cross-color final destinations are distinct; otherwise repaint order would matter.
# i) Generator guarantees for those assumptions
#    - `_legend_patch()` creates only valid strip/L legends.
#    - `_place_legends()` assigns distinct corners and unique mover colors.
#    - `_place_movers()` enforces distinct outputs and reverse-slide validity.
#    - `_oracle_check()` runs `main()` and compares against an explicit expected grid.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

MOVER_COLORS = (4, 6, 7)
WALL_COLOR_CHOICES = (3, 5, 8, 9)
LEGEND_MID_COLOR = 1
LEGEND_HEAD_COLOR = 2

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]

LEGEND_RUNS = (3, 2, 2)
LEGEND_PATH_LEN = sum(LEGEND_RUNS)
LEGEND_BUFFER = 1
ACTIVE_LEGEND_COUNT = (2, 3)

MAX_TOTAL_REGIONS = 3
REGION_SIZE_BOUNDS = {
    1: (18, 85),
    2: (15, 52),
    3: (12, 30),
}
BLOB_GROW_RETRIES = 25
GEN_RETRIES = 300
MAX_SLIDE_STEPS = 6


def _sign(v: int) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _legend_path_coords(dr: int, dc: int) -> list[tuple[int, int]]:
    """Return the 7-cell legend path in tail->head order."""
    if (dr, dc) == (0, 1):
        return [(0, c) for c in range(LEGEND_PATH_LEN)]
    if (dr, dc) == (0, -1):
        return [(0, c) for c in range(LEGEND_PATH_LEN - 1, -1, -1)]
    if (dr, dc) == (1, 0):
        return [(r, 0) for r in range(LEGEND_PATH_LEN)]
    if (dr, dc) == (-1, 0):
        return [(r, 0) for r in range(LEGEND_PATH_LEN - 1, -1, -1)]
    if (dr, dc) == (1, 1):
        return [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
    if (dr, dc) == (1, -1):
        return [(0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0)]
    if (dr, dc) == (-1, 1):
        return [(3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3)]
    if (dr, dc) == (-1, -1):
        return [(3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0), (0, 0)]
    raise ValueError((dr, dc))


def _legend_patch(mover_color: int, dr: int, dc: int) -> np.ndarray:
    """Create a strip or L legend whose tail color identifies the mover color."""
    coords = _legend_path_coords(dr, dc)
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.zeros((h, w), dtype=int)
    colors = (
        [mover_color] * LEGEND_RUNS[0]
        + [LEGEND_MID_COLOR] * LEGEND_RUNS[1]
        + [LEGEND_HEAD_COLOR] * LEGEND_RUNS[2]
    )
    for (r, c), col in zip(coords, colors):
        patch[r, c] = col
    return patch


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


def _ordered_path(pset: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Return the unique endpoint-to-endpoint traversal of a simple 4-connected path."""
    degrees: dict[tuple[int, int], int] = {}
    for r, c in pset:
        deg = 0
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in pset:
                deg += 1
        degrees[(r, c)] = deg
    endpoints = [p for p, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return None
    if any(deg not in (1, 2) for deg in degrees.values()):
        return None

    start = endpoints[0]
    path = [start]
    prev = None
    cur = start
    while len(path) < len(pset):
        nxts = []
        for nr, nc in neighbors4(*cur):
            nxt = (nr, nc)
            if nxt in pset and nxt != prev:
                nxts.append(nxt)
        if len(nxts) != 1:
            return None
        prev, cur = cur, nxts[0]
        path.append(cur)
    return path


def _parse_legend_path(
    g: np.ndarray, path: list[tuple[int, int]]
) -> tuple[int, tuple[int, int]] | None:
    """Return (mover_color, direction) if the path matches `m,m,m,1,1,2,2`."""
    if len(path) != LEGEND_PATH_LEN:
        return None

    def parse(seq_path: list[tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
        seq = [int(g[r, c]) for r, c in seq_path]
        i0 = LEGEND_RUNS[0]
        i1 = LEGEND_RUNS[0] + LEGEND_RUNS[1]
        tail = seq[:i0]
        mid = seq[i0:i1]
        head = seq[i1:]
        if len(set(tail)) != 1 or mid != [LEGEND_MID_COLOR] * LEGEND_RUNS[1]:
            return None
        if head != [LEGEND_HEAD_COLOR] * LEGEND_RUNS[2]:
            return None
        mover_color = tail[0]
        if mover_color not in MOVER_COLORS:
            return None
        dr = _sign(seq_path[-1][0] - seq_path[0][0])
        dc = _sign(seq_path[-1][1] - seq_path[0][1])
        if dr == 0 and dc == 0:
            return None
        return mover_color, (dr, dc)

    parsed = parse(path)
    if parsed is not None:
        return parsed
    return parse(list(reversed(path)))


def _legend_direction_map(
    g: np.ndarray, bg: int
) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]]]:
    """Parse border legends and return mover-color -> direction plus occupied legend cells."""
    h, w = g.shape
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    legend_cells: set[tuple[int, int]] = set()
    for pts in components_mask(g != bg, connectivity=4):
        pts_list = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts_list):
            continue
        path = _ordered_path(set(pts_list))
        if path is None:
            continue
        parsed = _parse_legend_path(g, path)
        if parsed is None:
            continue
        mover_color, direction = parsed
        prev = best.get(mover_color)
        if prev is None or len(pts_list) > prev[0]:
            best[mover_color] = (len(pts_list), direction)
            legend_cells.update(pts_list)
    return {color: direction for color, (_, direction) in best.items()}, legend_cells


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(
        points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True
    )
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
    Slide multi-colored pixels inside enclosed wall regions using border color-sequence legends.

    Input structure:
    - Background is the dominant color 0; colored walls still carve out enclosed chambers.
    - Border legends are thin 7-cell strips or L paths placed at corners.
    - Each legend uses the fixed tail->head sequence `m,m,m,1,1,2,2`, where `m` is one of the
      mover colors appearing inside the chambers.
    - Chambers may contain several mover colors at once rather than only yellow.

    Rule:
    - Parse each border legend as an ordered path; its mover-color tail identifies which mover
      color it controls, and the ordered geometry gives the movement direction.
    - For each enclosed chamber, collect all movers inside it and group them by color.
    - Slide each color group in that chamber along its own legend direction, stacking that
      color against the far wall or against previously stopped movers of the same color group.

    Key insight:
    - Walls define where motion is allowed, but they no longer choose the direction.
    - The mover color itself selects the matching legend, and the legend's ordered color strip
      or L shape encodes the direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs, legend_cells = _legend_direction_map(g, bg)
    assert dirs, "legend parsing failed; generated inputs always include border legends"

    mover_mask = np.isin(g, MOVER_COLORS)
    legend_mask = np.zeros_like(g, dtype=bool)
    for r, c in legend_cells:
        legend_mask[r, c] = True
    passable = (g == bg) | (mover_mask & ~legend_mask)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        movers_by_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in pts:
            col = int(g[r, c])
            if col in MOVER_COLORS:
                movers_by_color.setdefault(col, []).append((int(r), int(c)))
        if not movers_by_color:
            continue

        for pts_color in movers_by_color.values():
            for r, c in pts_color:
                out[r, c] = bg

        placements: list[tuple[int, int, int]] = []
        for color, pts_color in movers_by_color.items():
            direction = dirs.get(color)
            if direction is None:
                for r, c in pts_color:
                    placements.append((r, c, color))
                continue
            dr, dc = direction
            for r, c in _slide_points(pset, pts_color, dr, dc):
                placements.append((r, c, color))

        for r, c, color in placements:
            out[r, c] = color

    return out


def _sample_legend_dirs(rng: random.Random, n_legends: int) -> list[tuple[int, int]]:
    """Pick distinct directions and force at least one diagonal legend for observability."""
    for _ in range(200):
        dirs = rng.sample(ALL_DIRS, n_legends)
        if any(d in DIAGONAL_DIRS for d in dirs):
            return dirs
    raise RuntimeError("_sample_legend_dirs: failed to sample distinct directions")


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    """Place 2-3 mover-color legends at distinct corners."""
    h, w = grid.shape
    n_legends = rng.randint(*ACTIVE_LEGEND_COUNT)
    mover_colors = rng.sample(list(MOVER_COLORS), n_legends)
    corners = rng.sample(CORNERS, n_legends)
    directions = _sample_legend_dirs(rng, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legend_map: dict[int, tuple[int, int]] = {}
    for mover_color, corner, direction in zip(mover_colors, corners, directions):
        dr, dc = direction
        patch = _legend_patch(mover_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0, transparent=0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legend_map[mover_color] = (dr, dc)
    return legend_occupied, legend_map


def _grow_blob(
    canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int, rng: random.Random
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


def _try_grow_valid_blob(
    rng: random.Random, canvas: np.ndarray, occupied: np.ndarray, target_size: int, h: int, w: int
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


def _grow_regions(rng: random.Random, legend_occupied: np.ndarray) -> list[dict[str, object]] | None:
    """Grow 1-3 enclosed regions; wall color is purely structural in this combo."""
    h, w = legend_occupied.shape
    n_regions = rng.randint(1, MAX_TOTAL_REGIONS)
    n_wall_colors = rng.randint(1, min(2, n_regions))
    wall_colors = rng.sample(list(WALL_COLOR_CHOICES), n_wall_colors)

    region_colors = wall_colors.copy()
    while len(region_colors) < n_regions:
        region_colors.append(rng.choice(wall_colors))
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[n_regions]
    occupied = legend_occupied.copy()
    regions: list[dict[str, object]] = []

    for wall_color in region_colors:
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
        regions.append({"wall_color": wall_color, "interior": blob, "walls": walls})

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if len(regions) != n_regions:
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [set(reg["interior"]) for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int) -> tuple[int, int]:
    r, c = start
    while _slide_step_valid(region_set, r, c, dr, dc):
        r, c = r + dr, c + dc
    return r, c


def _find_reverse_slide_single(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    used_inputs: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for steps in range(MAX_SLIDE_STEPS, 0, -1):
        r, c = out_pos
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in used_inputs:
                ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    ok = False
                    break
            r, c = pr, pc
        if not ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc) == out_pos:
            return r, c
    return None


def _choose_output_and_input(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
    used_outputs: set[tuple[int, int]],
    used_inputs: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    stop_cells = [p for p in blob if not _slide_step_valid(blob, p[0], p[1], dr, dc)]
    if not stop_cells:
        return None
    stop_cells.sort(key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    best_key = dr * stop_cells[0][0] + dc * stop_cells[0][1]
    shortlist = [p for p in stop_cells if dr * p[0] + dc * p[1] >= best_key - 2]
    rng.shuffle(shortlist)
    ordered = shortlist + [p for p in stop_cells if p not in shortlist]

    for out_pos in ordered:
        if out_pos in used_outputs:
            continue
        start = _find_reverse_slide_single(blob, out_pos, dr, dc, used_inputs)
        if start is not None:
            return out_pos, start
    return None


def _place_movers(
    rng: random.Random,
    regions: list[dict[str, object]],
    legend_map: dict[int, tuple[int, int]],
    grid: np.ndarray,
) -> bool:
    """Place 2-3 mover colors per region, with one mover per active color in that region."""
    legend_colors = list(legend_map)
    anchor_region_idx = rng.randrange(len(regions))
    used_global: set[int] = set()

    for idx, reg in enumerate(regions):
        blob = set(reg["interior"])
        if len(legend_colors) == 2:
            active_colors = legend_colors.copy()
        elif idx == anchor_region_idx:
            active_colors = legend_colors.copy()
        else:
            n_colors = rng.randint(2, len(legend_colors))
            active_colors = rng.sample(legend_colors, n_colors)

        used_outputs: set[tuple[int, int]] = set()
        used_inputs: set[tuple[int, int]] = set()
        expected_inputs: dict[int, tuple[int, int]] = {}
        expected_outputs: dict[int, tuple[int, int]] = {}

        for color in active_colors:
            dr, dc = legend_map[color]
            picked = _choose_output_and_input(rng, blob, dr, dc, used_outputs, used_inputs)
            if picked is None:
                return False
            out_pos, in_pos = picked
            expected_outputs[color] = out_pos
            expected_inputs[color] = in_pos
            used_outputs.add(out_pos)
            used_inputs.add(in_pos)
            used_global.add(color)

        for color, (r, c) in expected_inputs.items():
            grid[r, c] = color

        reg["active_colors"] = active_colors
        reg["expected_inputs"] = expected_inputs
        reg["expected_outputs"] = expected_outputs

    return used_global == set(legend_colors)


def _oracle_check(input_grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    expected = input_grid.copy()
    for reg in regions:
        for r, c in dict(reg["expected_inputs"]).values():
            expected[r, c] = 0
        for color, (r, c) in dict(reg["expected_outputs"]).items():
            expected[r, c] = int(color)
    return np.array_equal(solver_out, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legend_map = _place_legends(rng, grid)
    regions = _grow_regions(rng, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = int(reg["wall_color"])

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_movers(rng, regions, legend_map, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
