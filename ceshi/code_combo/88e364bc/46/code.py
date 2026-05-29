# Combo variant of 88e364bc (index 46)
# combo_spec: D1/2+D19/0
# Name: 颜色序列编码+单一图例

# Design analysis for this combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task:
#      - Border legends are filled rectangles.
#      - Each legend matches one wall color.
#      - Yellow pixels inside a region slide in the direction attached to that region's wall color.
#    - Fused task here:
#      - There is exactly one border legend.
#      - The legend is a thin 4-connected path whose ordered color sequence is
#        `8,8,8,1,1,2,2` from tail to head.
#      - Straight paths encode cardinal directions and L paths encode diagonal directions.
#      - All enclosed regions, regardless of wall color, use that single decoded direction.
#    - Concrete example trace:
#      - Suppose the corner legend is an L path with tail at upper-left and head at lower-right.
#        The ordered path therefore decodes to direction `(1, 1)`.
#      - A green-walled region and a gray-walled region both contain yellow pixels.
#      - Original logic would first identify each region's majority wall color and then look for
#        a matching legend. The fused logic skips that step entirely: both regions slide along
#        `(1, 1)` because there is only one legend.
#      - Difference: wall color still separates regions visually, but it no longer carries any
#        instruction semantics. The whole inference burden moves to the single sequence legend.
# b) Complexity calibration
#    - The direction space is still only 8-way, so the task stays tractable.
#    - Using one fixed legend palette and exactly one legend removes matching ambiguity.
#    - The generator always uses at least two distinct wall colors, so the "single legend controls
#      multiple wall colors" behavior is visible in each sample rather than hidden in edge cases.
# c) Audit of old implicit assumptions
#    - Old `_legend_directions()` assumed:
#      - legends are solid rectangles
#      - legend fill color equals the controlled wall color
#      - there may be multiple independent legends
#      - region processing must recover a wall-color key before motion lookup
#    - All of those assumptions are invalid here except "the legend touches the border".
#    - The new solver therefore:
#      - reconstructs an ordered border path instead of a filled box
#      - decodes exactly one global direction
#      - removes wall-color lookup from `main()` entirely
#    - The slide physics code itself remains valid because it only needs a direction vector.
# d) Placement legality
#    - The legend is pasted into one corner with a 1-cell buffer reservation.
#    - Regions are grown strictly inside the outer border and cannot overlap the legend buffer.
#    - Wall colors exclude legend colors `1,2,8`, so the legend cannot merge with walls.
#    - L-shaped legends leave their interior bbox mostly background, but because the path touches
#      the outer border, those zeros remain connected to outside background and do not form traps.
# e) Visual observability
#    - A human can read the unique sequence trail and see all yellow pixels across differently
#      colored regions move the same way.
#    - Repeating that same direction across multiple wall colors is the visual evidence that
#      wall color is no longer part of the instruction lookup.
# f) Visual cleanliness
#    - The legend uses only 7 cells and sits alone in a corner.
#    - Regions still use thin 1-cell walls with reserved spacing, so walls, legend, and movers
#      remain visually separable.
# g) Change inferability
#    - The legend palette is fixed rather than remapped, so no hidden color correspondence must
#      be learned.
#    - Wall colors vary only as region separators; because multiple wall colors coexist in one
#      grid yet share one motion, the intended abstraction is exposed directly.
# h) New solver assumptions introduced here
#    - Exactly one valid border legend exists.
#    - The legend is a simple 4-connected path with exactly 7 cells.
#    - One orientation of that path matches `8,8,8,1,1,2,2`, yielding a nonzero direction.
#    - All enclosed mover-containing regions should obey the same decoded direction.
# i) Generator guarantees for those assumptions
#    - `_place_legend()` creates exactly one valid path legend.
#    - Wall colors are sampled from `[3,5,6,7,9]`, so they never equal legend colors.
#    - `_grow_regions()` always places at least one region for each of two wall colors.
#    - `_place_yellows()` reverse-slides every yellow by at least one step, so the output differs
#      from the input and the global control effect is observable.

import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]

LEGEND_TAIL_COLOR = 8
LEGEND_MID_COLOR = 1
LEGEND_HEAD_COLOR = 2
LEGEND_RUNS = (3, 2, 2)
LEGEND_PATH_LEN = sum(LEGEND_RUNS)
LEGEND_BUFFER = 1
CARDINAL_PROB = 0.5

WALL_COLOR_CHOICES = [3, 5, 6, 7, 9]
N_WALL_COLORS = 2
REGIONS_PER_COLOR = (1, 2)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    2: (15, 55),
    3: (12, 35),
    4: (12, 25),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [1, 2]
MAX_SLIDE_STEPS = 5
GEN_RETRIES = 300


def _sign(v: int) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _legend_path_coords(dr: int, dc: int) -> list[tuple[int, int]]:
    """Legend path in tail->head order."""
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


def _legend_patch(dr: int, dc: int) -> np.ndarray:
    """Create the fixed-sequence strip or L legend."""
    coords = _legend_path_coords(dr, dc)
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.zeros((h, w), dtype=int)
    colors = (
        [LEGEND_TAIL_COLOR] * LEGEND_RUNS[0]
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
    """Return the unique endpoint-to-endpoint order of a simple 4-connected path."""
    degrees: dict[tuple[int, int], int] = {}
    for r, c in pset:
        deg = 0
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in pset:
                deg += 1
        degrees[(r, c)] = deg
    endpoints = [p for p, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2 or any(deg not in (1, 2) for deg in degrees.values()):
        return None

    path = [endpoints[0]]
    prev = None
    cur = endpoints[0]
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


def _parse_legend_direction(g: np.ndarray, path: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Decode direction if the path matches the fixed tail->head color sequence."""
    if len(path) != LEGEND_PATH_LEN:
        return None

    target = (
        [LEGEND_TAIL_COLOR] * LEGEND_RUNS[0]
        + [LEGEND_MID_COLOR] * LEGEND_RUNS[1]
        + [LEGEND_HEAD_COLOR] * LEGEND_RUNS[2]
    )

    def parse(seq_path: list[tuple[int, int]]) -> tuple[int, int] | None:
        seq = [int(g[r, c]) for r, c in seq_path]
        if seq != target:
            return None
        dr = _sign(seq_path[-1][0] - seq_path[0][0])
        dc = _sign(seq_path[-1][1] - seq_path[0][1])
        if dr == 0 and dc == 0:
            return None
        return dr, dc

    direction = parse(path)
    if direction is not None:
        return direction
    return parse(list(reversed(path)))


def _legend_direction(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Find the unique border legend and decode its global slide direction.

    The legend is the only border-touching non-background component that forms a simple 7-cell
    path whose ordered colors are `8,8,8,1,1,2,2`.
    """
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    for pts in components_mask(g != bg, connectivity=4):
        pts_list = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts_list):
            continue
        path = _ordered_path(set(pts_list))
        if path is None:
            continue
        direction = _parse_legend_direction(g, path)
        if direction is not None:
            candidates.append(direction)
    assert len(candidates) == 1, "expected exactly one valid color-sequence legend"
    return candidates[0]


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
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


def _region_border_color(
    pts: list[tuple[int, int]] | set[tuple[int, int]], g: np.ndarray, bg: int, mover: int
) -> int | None:
    """Majority wall color touching an enclosed passable region."""
    h, w = g.shape
    pset = {(int(r), int(c)) for r, c in pts}
    border_counts: dict[int, int] = {}
    for r, c in pset:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in pset:
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    if not border_counts:
        return None
    return max(border_counts, key=border_counts.get)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside enclosed wall regions using one shared border legend.

    Input structure:
    - Background cells are black, yellow cells sit inside enclosed regions bounded by colored walls.
    - Exactly one border-touching legend appears in a corner as a thin strip or L-shaped path.
    - The legend does not match wall colors; instead its tail->head color sequence is fixed as
      `8,8,8,1,1,2,2`, with straight paths for cardinal motion and L paths for diagonal motion.

    Rule:
    - Recover the ordered legend path and decode the direction from its tail toward its head.
    - Apply that one decoded direction to every enclosed passable region, ignoring wall color.
    - Within each region, yellow pixels slide until blocked by the far wall or by previously
      stopped yellow pixels, stacking from the destination side inward.

    Key insight:
    - The walls only carve the board into separate chambers. The only instruction-carrying object
      is the single color-sequence legend, so once its path orientation is understood, all regions
      move the same way.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4

    dr, dc = _legend_direction(g, bg)
    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        pts_list = [(int(r), int(c)) for r, c in pts]
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts_list):
            continue
        movers = [(r, c) for r, c in pts_list if int(g[r, c]) == mover]
        if not movers:
            continue
        pset = set(pts_list)
        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = mover
    return out


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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int, blocked: set[tuple[int, int]]
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _place_legend(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Place the unique corner legend and reserve its padded footprint."""
    h, w = grid.shape
    direction_pool = CARDINAL_DIRS if rng.random() < CARDINAL_PROB else DIAGONAL_DIRS
    dr, dc = rng.choice(direction_pool)
    patch = _legend_patch(dr, dc)
    ph, pw = patch.shape
    r0, c0 = _corner_origin(rng.choice(CORNERS), h, w, ph, pw)
    paste(grid, patch, r0, c0, transparent=0)

    occupied = np.zeros((h, w), dtype=bool)
    r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
    c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
    occupied[r_lo:r_hi, c_lo:c_hi] = True
    return occupied, (dr, dc)


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


def _grow_regions(
    rng: random.Random, direction: tuple[int, int], legend_occupied: np.ndarray
) -> list[dict[str, object]] | None:
    """Grow closed regions for two distinct wall colors; all share the same direction."""
    h, w = legend_occupied.shape
    wall_colors = rng.sample(WALL_COLOR_CHOICES, N_WALL_COLORS)

    region_colors: list[int] = []
    for color in wall_colors:
        region_colors.extend([color] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
        missing = [color for color in wall_colors if color not in region_colors]
        while missing:
            idx = rng.randrange(len(region_colors))
            region_colors[idx] = missing.pop()
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

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
        regions.append({"color": wall_color, "dir": direction, "interior": blob, "walls": walls})

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if len(regions) != len(region_colors):
        return None
    if {int(reg["color"]) for reg in regions} != set(wall_colors):
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
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in blob or (pr, pc + dc) not in blob:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return (r, c)
    return None


def _place_yellows(rng: random.Random, regions: list[dict[str, object]], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice(YELLOWS_PER_REGION)

        cells_sorted = sorted(blob, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
        tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]

        first = tier1[rng.randrange(len(tier1))]
        output_positions = [first]

        if n_yellows == 2:
            pr, pc = first[0] - dr, first[1] - dc
            cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
            if cands:
                second = cands[rng.randrange(len(cands))]
                if second != first:
                    output_positions.append(second)

        out_sorted = sorted(output_positions, key=lambda p: dr * p[0] + dc * p[1], reverse=True)
        already_stopped = set()
        yellow_inputs = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellow_inputs, rng)
            if candidate is None:
                return False
            yellow_inputs.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellow_inputs:
            grid[r, c] = 4
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False

    region_mask = np.zeros(input_grid.shape, dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            region_mask[r, c] = True
    if not np.all(region_mask[solver_out == 4]):
        return False

    diag = fusion_diagnostics(input_grid)
    if diag["legend_count"] != 1:
        return False
    if diag["distinct_active_wall_colors"] < 2:
        return False
    if diag["moved_yellow_count"] <= 0:
        return False
    if diag["legend_tail_in_active_walls"]:
        return False
    return True


def fusion_diagnostics(grid: np.ndarray) -> dict[str, object]:
    """
    Quantify whether the generated sample really exhibits the fused behavior.

    Metrics:
    - `legend_count`: number of valid sequence legends found on the border.
    - `distinct_active_wall_colors`: how many wall colors surround mover-containing regions.
    - `legend_tail_in_active_walls`: whether the legend tail color equals any active wall color.
    - `moved_yellow_count`: number of yellow pixels whose coordinates change after `main()`.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mover = 4
    out = main(g)

    legend_count = 0
    for pts in components_mask(g != bg, connectivity=4):
        pts_list = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in pts_list):
            continue
        path = _ordered_path(set(pts_list))
        if path is not None and _parse_legend_direction(g, path) is not None:
            legend_count += 1

    active_wall_colors = set()
    passable = (g == bg) | (g == mover)
    for pts in components_mask(passable, connectivity=4):
        pts_list = [(int(r), int(c)) for r, c in pts]
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in pts_list):
            continue
        if not any(int(g[r, c]) == mover for r, c in pts_list):
            continue
        border_color = _region_border_color(pts_list, g, bg, mover)
        if border_color is not None:
            active_wall_colors.add(border_color)

    moved_yellow_count = 0
    before = set(map(tuple, np.argwhere(g == mover)))
    after = set(map(tuple, np.argwhere(out == mover)))
    moved_yellow_count = len(before - after)

    return {
        "legend_count": legend_count,
        "legend_direction": _legend_direction(g, bg),
        "distinct_active_wall_colors": len(active_wall_colors),
        "active_wall_colors": tuple(sorted(active_wall_colors)),
        "legend_tail_in_active_walls": LEGEND_TAIL_COLOR in active_wall_colors,
        "moved_yellow_count": moved_yellow_count,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, direction = _place_legend(rng, grid)
    regions = _grow_regions(rng, direction, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
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
