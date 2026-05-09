# Variant of 88e364bc
# Dimension: D1 — 图例方向编码方式 (symbol_instructions)
# Change: 图例为5x5或6x7实心矩形，包含三种颜色：填充色C（最多）、中心色（4像素，靠近中心）、指示色/红色（2像素，更外围）... → 颜色序列编码方向

# Design analysis for this D1 variant
# -----------------------------------
# a) Concrete behavior and trace
#    - The walled regions and yellow sliding behavior stay unchanged.
#    - Only the border legend changes:
#      - Cardinal directions use a 1x7 or 7x1 strip.
#      - Diagonal directions use a 4x4 L-shaped path.
#      - In both cases the legend is a single 4-connected path whose color runs are
#        exactly `C,C,C,1,1,2,2` from tail to head.
#      - `C` is still the wall color controlled by that legend.
#    - Example trace:
#      - Original solver: detect a filled border rectangle, identify center vs indicator
#        by counts/centroids, then direction = sign(indicator centroid - center centroid).
#      - Variant solver: detect a border-touching path component, reconstruct its ordered
#        path, orient the path so the color run is `C,C,C,1,1,2,2`, then direction =
#        sign(path_end - path_start).
#      - Difference: direction is no longer read from a 2D centroid offset inside a solid
#        block; it is read from the ordered sequence and geometry of a skinny path / L path.
# b) Complexity calibration
#    - The legend keeps only 3 colors and only 8 possible directions, so inference stays
#      tractable.
#    - The new legend is visually distinct from the original: thin strips and L paths at
#      corners are easy to separate from the 1-cell wall boundaries.
#    - The fixed run lengths 3/2/2 remove ambiguity when orienting the path.
# c) Audit of old implicit assumptions
#    - Old `_legend_directions()` assumed:
#      - border-touching component is a filled rectangle
#      - the controlled wall color is the most common color of that rectangle
#      - the remaining two colors can be separated by frequency / centrality
#      - direction comes from their centroids
#    - In this variant, only "border-touching component" and "wall color is identifiable"
#      remain valid. Rectangle, centroid, and centrality assumptions are false, so the
#      legend parser must be replaced.
#    - The slide logic itself only needs a `(wall_color -> direction)` mapping and remains
#      valid unchanged.
# d) Placement legality
#    - Legends are still pasted into corners and surrounded by a 1-cell reserved buffer.
#    - Regions are still grown strictly inside the border and cannot overlap legends.
#    - The diagonal L legends are open to the outer border, so they do not create enclosed
#      zero pockets that would confuse region detection.
# e) Visual observability
#    - A human can now infer direction from the ordered trail:
#      - straight trail: the sequence points right/left/up/down
#      - L trail: the sequence turns, revealing both horizontal and vertical signs
#    - The legend shape itself now carries semantic information; this is directly visible
#      across 2-4 ARC training examples.
# f) Visual cleanliness
#    - Each legend uses only 7 colored cells, sits in a corner, and is buffered away from
#      walls. This keeps the grid readable and avoids clutter near the regions.
# g) Color-change inferability
#    - Colors are not remapped per example. The same three roles persist:
#      `C` = controlled wall color, `1` = middle trail, `2` = trail head.
#      The only novelty is the ordered path geometry.
# h) New solver assumptions introduced here
#    - Every legend is a simple 4-connected path with exactly 7 cells.
#    - The path can be oriented so its color sequence is exactly `C,C,C,1,1,2,2`.
#    - The path endpoints differ in position, so `sign(end-start)` is a nonzero direction.
# i) Generator guarantees for those assumptions
#    - `_legend_patch()` creates only those valid 7-cell path legends.
#    - Legends are unique by wall color and placed at corners with buffer.
#    - `_oracle_check()` still validates every generated input by running `main()`.

import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask, neighbors4, paste


# ── Shared configuration ──────────────────────────────────────────

GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]

WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
CARDINAL_PROB = 0.5

LEGEND_MID_COLOR = 1
LEGEND_HEAD_COLOR = 2
LEGEND_RUNS = (3, 2, 2)  # C,C,C,1,1,2,2
LEGEND_PATH_LEN = sum(LEGEND_RUNS)

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
GEN_RETRIES = 300


# ── Legend parsing / generation ──────────────────────────────────

def _sign(v: int) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _legend_path_coords(dr: int, dc: int) -> list[tuple[int, int]]:
    """Legend path in tail→head order."""
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


def _legend_patch(fill: int, dr: int, dc: int) -> np.ndarray:
    """Create a strip or L-shaped legend using the fixed sequence C,C,C,1,1,2,2."""
    coords = _legend_path_coords(dr, dc)
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.zeros((h, w), dtype=int)
    colors = (
        [fill] * LEGEND_RUNS[0]
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
    """Return the unique ordered endpoint-to-endpoint path of a simple 4-connected component."""
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


def _parse_legend_path(g: np.ndarray, path: list[tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
    """Return (wall_color, direction) if the path matches the fixed run-length encoding."""
    if len(path) != LEGEND_PATH_LEN:
        return None

    def parse(seq_path: list[tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
        seq = [int(g[r, c]) for r, c in seq_path]
        i0, i1 = LEGEND_RUNS[0], LEGEND_RUNS[0] + LEGEND_RUNS[1]
        tail = seq[:i0]
        mid = seq[i0:i1]
        head = seq[i1:]
        if not tail or not mid or not head:
            return None
        if len(set(tail)) != 1 or len(set(mid)) != 1 or len(set(head)) != 1:
            return None
        wall_col = tail[0]
        if len({wall_col, mid[0], head[0]}) != 3:
            return None
        dr = _sign(seq_path[-1][0] - seq_path[0][0])
        dc = _sign(seq_path[-1][1] - seq_path[0][1])
        if dr == 0 and dc == 0:
            return None
        return wall_col, (dr, dc)

    parsed = parse(path)
    if parsed is not None:
        return parsed
    return parse(list(reversed(path)))


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends whose direction is encoded by an ordered path.

    Each valid legend is a border-touching 4-connected component with exactly 7 cells and the
    tail→head color run `C,C,C,1,1,2,2`. The wall color is the first run color `C`, and the
    motion direction is `sign(path_end - path_start)`.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        pts_list = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts_list):
            continue
        pset = set(pts_list)
        path = _ordered_path(pset)
        if path is None:
            continue
        parsed = _parse_legend_path(g, path)
        if parsed is None:
            continue
        wall_col, direction = parsed
        prev = best.get(wall_col)
        if prev is None or len(pset) > prev[0]:
            best[wall_col] = (len(pset), direction)
    return {col: direction for col, (_, direction) in best.items()}


# ── Slide solver ─────────────────────────────────────────────────

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions in the direction dictated by border legends.

    Variant rule:
    - Legends are thin border paths rather than filled rectangles.
    - Each legend path has color order `C,C,C,1,1,2,2` from tail to head.
    - Straight paths encode cardinal directions; L paths encode diagonal directions.
    - The path's first run color `C` still identifies which wall color it controls.
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
            for nr, nc in neighbors4(int(r), int(c)):
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

        for r, c in movers:
            out[r, c] = bg

        new_pts = _slide_points(pset, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover

    return out


# ── Blob growing / topology ──────────────────────────────────────

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


# ── Slide helpers for generation ─────────────────────────────────

def _slide_step_valid(region_set: set, r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _forward_slide(
    region_set: set, start: tuple[int, int], dr: int, dc: int, blocked: set
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


# ── Generator phases ─────────────────────────────────────────────

def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    """Place 1-2 corner legends using strip/L path encoding.

    At least one legend is always diagonal/L-shaped. This is a deliberate observability guard:
    a sample made only of straight 1xN strips can still be partially explained by the original
    rectangle-centroid parser, while any sample containing an active diagonal L legend cannot.
    """
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []

    dir_pools: list[list[tuple[int, int]]] = []
    if n_legends == 1:
        dir_pools = [DIAGONAL_DIRS]
    else:
        diag_idx = rng.randrange(n_legends)
        for idx in range(n_legends):
            if idx == diag_idx:
                dir_pools.append(DIAGONAL_DIRS)
            else:
                dir_pools.append(CARDINAL_DIRS if rng.random() < CARDINAL_PROB else DIAGONAL_DIRS)

    for corner, wc, dirs_pool in zip(corners, wall_colors, dir_pools):
        dr, dc = rng.choice(dirs_pool)
        patch = _legend_patch(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0, transparent=0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))

    return legend_occupied, legends


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
    rng: random.Random, legends: list[tuple[int, tuple[int, int]]], legend_occupied: np.ndarray
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

    region_colors = []
    for col in all_colors:
        region_colors.extend([col] * rng.randint(*REGIONS_PER_COLOR))
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


def _find_reverse_slide(
    blob: set,
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set,
    existing_inputs: list,
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


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
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
        yellows_input = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = 4
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if (solver_out == 4).sum() != (input_grid == 4).sum():
        return False
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == 4]))


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
    if not _place_yellows(rng, regions, input_grid):
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
