import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
RING_OFFSETS = [(0, 0), (0, 1), (1, 1), (1, 0)]
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

MIN_H = 10
MAX_H = 20
MIN_W = 10
MAX_W = 20
MAX_GENERATION_ATTEMPTS = 500
MAX_TOTAL_DRAWN = 70
MAX_FILL_RATIO = 0.35
MIN_TOTAL_DRAWN = 4
BOUNDARY_LAYOUT_ATTEMPTS = 120
GEN_RNG = random.Random()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _corner_origins(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]


def _legend_data(grid: np.ndarray, bg: int) -> tuple[list[int], dict[int, tuple[int, int]], dict[int, int], set[tuple[int, int]], tuple[int, int]]:
    candidates = []
    for r0, c0 in _corner_origins(grid.shape):
        patch = grid[r0 : r0 + 2, c0 : c0 + 2]
        flat = [int(v) for v in patch.reshape(-1)]
        if len(flat) != 4:
            continue
        if any(v == bg for v in flat):
            continue
        if len(set(flat)) != 4:
            continue
        cycle = [int(patch[0, 0]), int(patch[0, 1]), int(patch[1, 1]), int(patch[1, 0])]
        coords = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
        candidates.append((cycle, coords, (r0, c0)))

    assert len(candidates) == 1, "expected exactly one 2x2 corner legend"
    cycle, coords, origin = candidates[0]
    dir_by_color = {cycle[i]: DIRS[i] for i in range(4)}
    next_color = {cycle[i]: cycle[(i + 1) % 4] for i in range(4)}
    return cycle, dir_by_color, next_color, coords, origin


def _seed_list(grid: np.ndarray, bg: int, legend_coords: set[tuple[int, int]], palette: set[int]) -> list[tuple[int, int, int]]:
    seeds = []
    for r, c in np.argwhere(grid != bg):
        rr, cc = int(r), int(c)
        if (rr, cc) in legend_coords:
            continue
        color = int(grid[rr, cc])
        assert color in palette, "seed colors must come from the legend ring"
        seeds.append((rr, cc, color))
    seeds.sort()
    return seeds


def _trace_path(
    start: tuple[int, int],
    color: int,
    dir_by_color: dict[int, tuple[int, int]],
    next_color: dict[int, int],
    occupied: np.ndarray,
) -> tuple[list[tuple[int, int, int]], str, tuple[int, int] | None]:
    h, w = occupied.shape
    r, c = start
    segment_len = 1
    current = int(color)
    drawn: list[tuple[int, int, int]] = []

    while True:
        dr, dc = dir_by_color[current]
        for _ in range(segment_len):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return drawn, "boundary", None
            if bool(occupied[nr, nc]):
                return drawn, "occupied", (nr, nc)
            drawn.append((nr, nc, current))
            r, c = nr, nc
        current = next_color[current]
        segment_len += 1


def _grow_on_grid(
    out: np.ndarray,
    start: tuple[int, int],
    color: int,
    dir_by_color: dict[int, tuple[int, int]],
    next_color: dict[int, int],
    bg: int,
) -> None:
    r, c = start
    segment_len = 1
    current = int(color)
    h, w = out.shape

    while True:
        dr, dc = dir_by_color[current]
        for _ in range(segment_len):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return
            if int(out[nr, nc]) != bg:
                return
            out[nr, nc] = current
            r, c = nr, nc
        current = next_color[current]
        segment_len += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a 2x2 color-ring legend to encode both color cycling and direction mapping,
    then grow spiral arms with segment lengths 1, 2, 3, ... from each seed pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cycle, dir_by_color, next_color, legend_coords, _origin = _legend_data(g, bg)
    seeds = _seed_list(g, bg, legend_coords, set(cycle))

    out = g.copy()
    for r, c, color in seeds:
        _grow_on_grid(out, (r, c), color, dir_by_color, next_color, bg)
    return out


def _legend_patch(cycle: list[int]) -> np.ndarray:
    return np.array([[cycle[0], cycle[1]], [cycle[3], cycle[2]]], dtype=int)


def _legend_mask(shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0 = origin
    mask[r0 : r0 + 2, c0 : c0 + 2] = True
    return mask


def _moat_mask(shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0 = origin
    r1 = max(0, r0 - 1)
    c1 = max(0, c0 - 1)
    r2 = min(shape[0], r0 + 3)
    c2 = min(shape[1], c0 + 3)
    mask[r1:r2, c1:c2] = True
    return mask


def _render_input(
    shape: tuple[int, int],
    cycle: list[int],
    origin: tuple[int, int],
    seeds: Iterable[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    r0, c0 = origin
    grid[r0 : r0 + 2, c0 : c0 + 2] = _legend_patch(cycle)
    for r, c, color in seeds:
        grid[r, c] = int(color)
    return grid


def _build_occupied(shape: tuple[int, int], origin: tuple[int, int], seeds: list[tuple[int, int, int]]) -> np.ndarray:
    occupied = _legend_mask(shape, origin)
    for r, c, _color in seeds:
        occupied[r, c] = True
    return occupied


def _chebyshev_far_enough(a: tuple[int, int], b: tuple[int, int], gap: int = 1) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > gap


def _candidate_cells(
    shape: tuple[int, int],
    moat: np.ndarray,
    seeds: list[tuple[int, int, int]],
    gap: int = 1,
) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    existing = [(r, c) for r, c, _ in seeds]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if moat[r, c]:
                continue
            if any(not _chebyshev_far_enough((r, c), pt, gap=gap) for pt in existing):
                continue
            cells.append((r, c))
    return cells


def _analyze_layout(
    shape: tuple[int, int],
    cycle: list[int],
    origin: tuple[int, int],
    seeds: list[tuple[int, int, int]],
) -> dict[str, object] | None:
    dir_by_color = {cycle[i]: DIRS[i] for i in range(4)}
    next_color = {cycle[i]: cycle[(i + 1) % 4] for i in range(4)}
    occupied = _build_occupied(shape, origin, seeds)
    used: set[tuple[int, int]] = set()
    reasons: list[str] = []
    blocked: list[tuple[int, int] | None] = []
    path_sizes: list[int] = []

    for r, c, color in seeds:
        drawn, reason, hit = _trace_path((r, c), color, dir_by_color, next_color, occupied)
        coords = {(rr, cc) for rr, cc, _ in drawn}
        if used & coords:
            return None
        used |= coords
        reasons.append(reason)
        blocked.append(hit)
        path_sizes.append(len(drawn))

    return {
        "reasons": reasons,
        "blocked": blocked,
        "path_sizes": path_sizes,
        "total_drawn": len(used),
    }


def _weighted_seed_count(rng: random.Random, mode: str) -> int:
    if mode == "occupied":
        return rng.choices([2, 3, 4], weights=[2, 2, 1])[0]
    return rng.choices([1, 2, 3, 4], weights=[1, 3, 3, 1])[0]


def _add_random_seeds(
    rng: random.Random,
    shape: tuple[int, int],
    origin: tuple[int, int],
    seeds: list[tuple[int, int, int]],
    remaining_colors: list[int],
    gap: int = 1,
) -> bool:
    moat = _moat_mask(shape, origin)
    for color in remaining_colors:
        choices = _candidate_cells(shape, moat, seeds, gap=gap)
        if not choices:
            return False
        seeds.append((*rng.choice(choices), color))
    return True


def _boundary_seeds(
    rng: random.Random,
    shape: tuple[int, int],
    cycle: list[int],
    origin: tuple[int, int],
) -> list[tuple[int, int, int]] | None:
    n_seeds = _weighted_seed_count(rng, "boundary")
    colors = rng.sample(cycle, n_seeds)
    for _ in range(BOUNDARY_LAYOUT_ATTEMPTS):
        seeds: list[tuple[int, int, int]] = []
        if not _add_random_seeds(rng, shape, origin, seeds, colors, gap=2):
            return None
        analysis = _analyze_layout(shape, cycle, origin, seeds)
        if analysis is None:
            continue
        if any(reason != "boundary" for reason in analysis["reasons"]):
            continue
        return seeds
    return None


def _occupied_seeds(
    rng: random.Random,
    shape: tuple[int, int],
    cycle: list[int],
    origin: tuple[int, int],
) -> list[tuple[int, int, int]] | None:
    n_seeds = _weighted_seed_count(rng, "occupied")
    colors = rng.sample(cycle, n_seeds)
    dir_by_color = {cycle[i]: DIRS[i] for i in range(4)}
    next_color = {cycle[i]: cycle[(i + 1) % 4] for i in range(4)}
    moat = _moat_mask(shape, origin)

    first_color = colors[0]
    starters = _candidate_cells(shape, moat, [])
    rng.shuffle(starters)
    for start in starters:
        seeds = [(start[0], start[1], first_color)]
        occupied = _build_occupied(shape, origin, seeds)
        path, _reason, _hit = _trace_path(start, first_color, dir_by_color, next_color, occupied)
        blocker_candidates = [(r, c) for idx, (r, c, _col) in enumerate(path) if idx >= 2]
        rng.shuffle(blocker_candidates)

        for br, bc in blocker_candidates:
            if moat[br, bc]:
                continue
            if not _chebyshev_far_enough((br, bc), start):
                continue
            seeds2 = seeds + [(br, bc, colors[1])]
            extra_colors = colors[2:]
            if not _add_random_seeds(rng, shape, origin, seeds2, extra_colors):
                continue
            analysis = _analyze_layout(shape, cycle, origin, seeds2)
            if analysis is None:
                continue
            if "occupied" not in analysis["reasons"]:
                continue
            return seeds2
    return None


def _valid_example(
    inp: np.ndarray,
    cycle: list[int],
    origin: tuple[int, int],
    mode: str,
) -> bool:
    legend_coords = {(origin[0] + dr, origin[1] + dc) for dr in range(2) for dc in range(2)}
    analysis = _analyze_layout(inp.shape, cycle, origin, _seed_list(inp, get_bg(inp), legend_coords, set(cycle)))
    if analysis is None:
        return False

    total_drawn = int(analysis["total_drawn"])
    if total_drawn < MIN_TOTAL_DRAWN or total_drawn > MAX_TOTAL_DRAWN:
        return False

    out = main(inp)
    if np.array_equal(inp, out):
        return False
    if np.count_nonzero(out) > int(MAX_FILL_RATIO * inp.size):
        return False

    r0, c0 = origin
    if not np.array_equal(inp[r0 : r0 + 2, c0 : c0 + 2], out[r0 : r0 + 2, c0 : c0 + 2]):
        return False

    reasons = analysis["reasons"]
    if mode == "boundary" and any(reason != "boundary" for reason in reasons):
        return False
    if mode == "occupied" and "occupied" not in reasons:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_H, MAX_H)
    w = rng.randint(MIN_W, MAX_W)
    shape = (h, w)
    origin = rng.choice(_corner_origins(shape))
    cycle = rng.sample(PALETTE, 4)
    mode = rng.choices(["boundary", "occupied"], weights=[2, 1])[0]

    if mode == "boundary":
        seeds = _boundary_seeds(rng, shape, cycle, origin)
    else:
        seeds = _occupied_seeds(rng, shape, cycle, origin)
    if seeds is None:
        return None

    inp = _render_input(shape, cycle, origin, seeds)
    if not _valid_example(inp, cycle, origin, mode):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
