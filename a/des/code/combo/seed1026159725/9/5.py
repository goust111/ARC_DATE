import numpy as np


YELLOW = 4
GRAY = 5
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_CHOICES = (0, 1, 2, 3, 6, 7, 8, 9)
GENERATE_RETRIES = 500
PLACEMENT_TRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _seed_data(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    seeds = []
    for idx, (r, c) in enumerate(np.argwhere((grid != bg) & (grid != GRAY))):
        seeds.append((idx, int(r), int(c), int(grid[r, c])))
    return seeds


def _trace_ray(
    grid: np.ndarray,
    seed_cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    bg: int,
) -> tuple[list[tuple[int, int]], str]:
    h, w = grid.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []

    while 0 <= r < h and 0 <= c < w:
        if (r, c) in seed_cells:
            return cells, "seed"
        value = int(grid[r, c])
        if value == GRAY:
            return cells, "obstacle"
        if value != bg:
            return cells, "seed"
        cells.append((r, c))
        r += dr
        c += dc
    return cells, "edge"


def _ray_summary(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int, int, int]], dict[tuple[int, int], set[int]], list[str]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    seeds = _seed_data(g, bg)
    seed_cells = {(r, c) for _idx, r, c, _color in seeds}
    hits: dict[tuple[int, int], set[int]] = {}
    stop_reasons: list[str] = []

    for idx, r, c, _color in seeds:
        for direction in DIRS:
            cells, reason = _trace_ray(g, seed_cells, (r, c), direction, bg)
            stop_reasons.append(reason)
            for cell in cells:
                if cell not in hits:
                    hits[cell] = set()
                hits[cell].add(idx)

    return bg, seeds, hits, stop_reasons


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast four-way rays from each colored seed until an obstacle, edge, or seed.

    Rays paint background cells in the seed color. Any background cell hit by
    rays from at least two different seeds becomes yellow. Seeds and gray
    obstacle pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg, seeds, hits, _stop_reasons = _ray_summary(g)
    color_by_seed = {idx: color for idx, _r, _c, color in seeds}

    for (r, c), seed_ids in hits.items():
        if len(seed_ids) >= 2:
            out[r, c] = YELLOW
        else:
            only_seed = next(iter(seed_ids))
            out[r, c] = color_by_seed[only_seed]

    return out


def _seed_palette(bg: int) -> list[int]:
    return [color for color in BG_CHOICES if color != bg]


def _is_seed_cell(grid: np.ndarray, pos: tuple[int, int], bg: int) -> bool:
    r, c = pos
    return int(grid[r, c]) != bg and int(grid[r, c]) != GRAY


def _can_use_seed_pos(
    pos: tuple[int, int],
    seed_positions: list[tuple[int, int]],
    allow_adjacent: set[frozenset[tuple[int, int]]],
    require_unique_axes: bool,
) -> bool:
    r, c = pos
    for other in seed_positions:
        orow, ocol = other
        pair = frozenset((pos, other))
        if max(abs(r - orow), abs(c - ocol)) <= 1 and pair not in allow_adjacent:
            return False
        if require_unique_axes and (r == orow or c == ocol):
            return False
    return True


def _random_seed_cell(
    rng: np.random.Generator,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
    allow_adjacent: set[frozenset[tuple[int, int]]],
    require_unique_axes: bool = True,
) -> tuple[int, int] | None:
    h, w = shape
    for _ in range(PLACEMENT_TRIES):
        pos = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        if pos in occupied or pos in reserved:
            continue
        if not _can_use_seed_pos(pos, seed_positions, allow_adjacent, require_unique_axes):
            continue
        return pos
    return None


def _place_seed(grid: np.ndarray, pos: tuple[int, int], color: int, occupied: set[tuple[int, int]]) -> None:
    grid[pos] = color
    occupied.add(pos)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        start, end = sorted((a[1], b[1]))
        return [(row, c) for c in range(start, end + 1)]
    col = a[1]
    start, end = sorted((a[0], b[0]))
    return [(r, col) for r in range(start, end + 1)]


def _force_cross(
    rng: np.random.Generator,
    grid: np.ndarray,
    seed_colors: list[int],
    occupied: set[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    if h < 5 or w < 5:
        return False

    for _ in range(PLACEMENT_TRIES):
        ir = int(rng.integers(2, h - 2))
        ic = int(rng.integers(2, w - 2))
        intersection = (ir, ic)
        if intersection in occupied:
            continue

        vertical_side = int(rng.choice((-1, 1)))
        horizontal_side = int(rng.choice((-1, 1)))
        max_v = ir if vertical_side == -1 else h - 1 - ir
        max_h = ic if horizontal_side == -1 else w - 1 - ic
        if max_v < 1 or max_h < 1:
            continue

        vdist = int(rng.integers(1, min(max_v, 4) + 1))
        hdist = int(rng.integers(1, min(max_h, 4) + 1))
        vseed = (ir + vertical_side * vdist, ic)
        hseed = (ir, ic + horizontal_side * hdist)
        if vseed == hseed:
            continue
        if vseed in occupied or hseed in occupied:
            continue
        if not _can_use_seed_pos(vseed, seed_positions, set(), False):
            continue
        if not _can_use_seed_pos(hseed, seed_positions + [vseed], set(), False):
            continue

        corridor = set(_segment_cells(vseed, intersection) + _segment_cells(hseed, intersection))
        if corridor & occupied:
            continue

        _place_seed(grid, vseed, seed_colors[0], occupied)
        _place_seed(grid, hseed, seed_colors[1], occupied)
        seed_positions.extend([vseed, hseed])
        reserved.update(corridor)
        return True

    return False


def _force_blocked_ray(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for seed in seed_positions:
        sr, sc = seed
        for dr, dc in DIRS:
            max_steps = 0
            rr, cc = sr + dr, sc + dc
            while 0 <= rr < h and 0 <= cc < w and (rr, cc) not in occupied and (rr, cc) not in reserved:
                max_steps += 1
                rr += dr
                cc += dc
            if max_steps >= 1:
                candidates.append((seed, (dr, dc), max_steps))

    if not candidates:
        return False

    seed, direction, max_steps = candidates[int(rng.integers(0, len(candidates)))]
    dr, dc = direction
    distance = int(rng.integers(1, min(max_steps, 4) + 1))
    obstacle = (seed[0] + dr * distance, seed[1] + dc * distance)
    grid[obstacle] = GRAY
    occupied.add(obstacle)
    reserved.update(_segment_cells(seed, obstacle))
    return True


def _force_seed_stop(
    rng: np.random.Generator,
    grid: np.ndarray,
    seed_colors: list[int],
    occupied: set[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_TRIES):
        vertical = bool(rng.integers(0, 2))
        if vertical:
            if h < 2:
                return False
            r = int(rng.integers(0, h - 1))
            c = int(rng.integers(0, w))
            a, b = (r, c), (r + 1, c)
        else:
            if w < 2:
                return False
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w - 1))
            a, b = (r, c), (r, c + 1)

        if a in occupied or b in occupied:
            continue
        if not _can_use_seed_pos(a, seed_positions, {frozenset((a, b))}, False):
            continue
        if not _can_use_seed_pos(b, seed_positions + [a], {frozenset((a, b))}, False):
            continue

        _place_seed(grid, a, seed_colors[0], occupied)
        _place_seed(grid, b, seed_colors[1], occupied)
        seed_positions.extend([a, b])
        reserved.update((a, b))
        return True

    return False


def _add_extra_seeds(
    rng: np.random.Generator,
    grid: np.ndarray,
    colors: list[int],
    occupied: set[tuple[int, int]],
    seed_positions: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
    allow_adjacent: set[frozenset[tuple[int, int]]],
    require_unique_axes: bool,
) -> bool:
    for color in colors:
        pos = _random_seed_cell(
            rng,
            grid.shape,
            occupied,
            seed_positions,
            reserved,
            allow_adjacent,
            require_unique_axes=require_unique_axes,
        )
        if pos is None:
            return False
        _place_seed(grid, pos, color, occupied)
        seed_positions.append(pos)
    return True


def _add_random_obstacles(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    minimum: int,
) -> None:
    h, w = grid.shape
    current = int(np.count_nonzero(grid == GRAY))
    target = max(minimum, current)
    target += int(rng.integers(0, max(2, (h * w) // 18)))

    for _ in range(PLACEMENT_TRIES * 2):
        if int(np.count_nonzero(grid == GRAY)) >= target:
            return
        pos = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        if pos in occupied or pos in reserved:
            continue
        grid[pos] = GRAY
        occupied.add(pos)


def _valid_generated(grid: np.ndarray, mode: str) -> bool:
    bg = _get_bg(grid)
    seeds = _seed_data(grid, bg)
    if not (2 <= len(seeds) <= 4):
        return False
    if len({color for _idx, _r, _c, color in seeds}) != len(seeds):
        return False
    if np.count_nonzero(grid == GRAY) < 1:
        return False
    if int(np.bincount(grid.ravel(), minlength=10)[bg]) <= grid.size // 2:
        return False

    out = main(grid)
    yellow_count = int(np.count_nonzero(out == YELLOW))
    _bg, _seeds, hits, stop_reasons = _ray_summary(grid)
    painted_cells = len(hits)
    if painted_cells < 4:
        return False

    if mode == "cross" and yellow_count < 1:
        return False
    if mode == "blocked" and ("obstacle" not in stop_reasons or yellow_count != 0):
        return False
    if mode == "seed_stop" and ("seed" not in stop_reasons or yellow_count != 0):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(9, 19))
    w = int(rng.integers(9, 19))
    bg = int(rng.choice(BG_CHOICES))
    grid = np.full((h, w), bg, dtype=int)

    mode = ("cross", "blocked", "seed_stop")[int(rng.integers(0, 3))]
    seed_count = int(rng.integers(2, 5))
    palette = _seed_palette(bg)
    seed_colors = list(rng.choice(palette, size=seed_count, replace=False))

    occupied: set[tuple[int, int]] = set()
    seed_positions: list[tuple[int, int]] = []
    reserved: set[tuple[int, int]] = set()

    if mode == "cross":
        if not _force_cross(rng, grid, seed_colors, occupied, seed_positions, reserved):
            return None
        extra_colors = seed_colors[2:]
        if not _add_extra_seeds(
            rng,
            grid,
            extra_colors,
            occupied,
            seed_positions,
            reserved,
            allow_adjacent=set(),
            require_unique_axes=True,
        ):
            return None
        if rng.random() < 0.7 and not _force_blocked_ray(rng, grid, occupied, seed_positions, reserved):
            return None
        min_obstacles = 1
    elif mode == "blocked":
        if not _add_extra_seeds(
            rng,
            grid,
            seed_colors,
            occupied,
            seed_positions,
            reserved,
            allow_adjacent=set(),
            require_unique_axes=True,
        ):
            return None
        if not _force_blocked_ray(rng, grid, occupied, seed_positions, reserved):
            return None
        min_obstacles = 2
    else:
        if not _force_seed_stop(rng, grid, seed_colors, occupied, seed_positions, reserved):
            return None
        extra_colors = seed_colors[2:]
        allowed_pair = {frozenset((seed_positions[0], seed_positions[1]))}
        if not _add_extra_seeds(
            rng,
            grid,
            extra_colors,
            occupied,
            seed_positions,
            reserved,
            allow_adjacent=allowed_pair,
            require_unique_axes=True,
        ):
            return None
        if rng.random() < 0.7 and not _force_blocked_ray(rng, grid, occupied, seed_positions, reserved):
            return None
        min_obstacles = 1

    _add_random_obstacles(rng, grid, occupied, reserved, minimum=min_obstacles)

    if not _valid_generated(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GENERATE_RETRIES} attempts")
