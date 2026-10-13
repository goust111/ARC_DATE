from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 800
MIN_RAY_LEN = 2


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _line_cells(coords: np.ndarray) -> list[Coord] | None:
    if len(coords) <= 2:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    raw = {(int(r), int(c)) for r, c in coords}
    if r0 == r1:
        cells = [(r0, c) for c in range(c0, c1 + 1)]
    elif c0 == c1:
        cells = [(r, c0) for r in range(r0, r1 + 1)]
    else:
        return None
    if len(cells) != len(coords):
        return None
    if any(cell not in raw for cell in cells):
        return None
    return cells


def _minimal_period(values: tuple[int, ...]) -> tuple[int, ...]:
    n = len(values)
    for p in range(1, n + 1):
        if n % p == 0 and all(values[i] == values[i % p] for i in range(n)):
            return tuple(int(x) for x in values[:p])
    return tuple(int(x) for x in values)


def _period_pair_index(period: tuple[int, ...]) -> dict[tuple[int, int], int]:
    n = len(period)
    assert n >= 5, "sample period must be long enough to encode phases"
    assert len(set(period)) < n, "sample period must contain repeated colors"
    pairs = [(int(period[i]), int(period[(i + 1) % n])) for i in range(n)]
    pair_set = set(pairs)
    assert len(pair_set) == n, "ordered adjacent pairs must be unique"
    for a, b in pairs:
        assert a != b, "adjacent period colors must be visibly different"
        assert (b, a) not in pair_set, "reverse ordered pairs must not be legal"
    return {pair: i for i, pair in enumerate(pairs)}


def _parse_input(
    grid: np.ndarray,
) -> tuple[int, tuple[int, ...], dict[tuple[int, int], int], list[np.ndarray]]:
    bg = _background(grid)
    comps = _components(grid != bg)
    assert comps, "expected one sample band and at least one seed"

    line_candidates: list[tuple[list[Coord], np.ndarray]] = []
    seed_candidates: list[np.ndarray] = []
    for comp in comps:
        cells = _line_cells(comp)
        if cells is not None:
            line_candidates.append((cells, comp))
        else:
            seed_candidates.append(comp)

    assert len(line_candidates) == 1, "expected exactly one straight sample band"
    band_cells, _band_comp = line_candidates[0]
    band_values = tuple(int(grid[r, c]) for r, c in band_cells)
    period = _minimal_period(band_values)
    assert len(band_values) >= 2 * len(period), "sample band must show at least two full cycles"
    pair_to_index = _period_pair_index(period)

    seeds: list[np.ndarray] = []
    for comp in seed_candidates:
        assert len(comp) == 2, "every non-band object must be a domino seed"
        a = tuple(map(int, comp[0]))
        b = tuple(map(int, comp[1]))
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "seed cells must be adjacent"
        ca, cb = int(grid[a]), int(grid[b])
        assert ((ca, cb) in pair_to_index) ^ ((cb, ca) in pair_to_index), (
            "seed colors must have exactly one legal reading direction"
        )
        seeds.append(comp)

    assert seeds, "expected at least one domino seed"
    return bg, period, pair_to_index, seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each two-cell seed as a phase-aligned ray that replays the sample-band period.

    Input structure:
    - The dominant background separates one straight multicolor sample band from several two-cell domino seeds.
    - The sample band is read left-to-right when horizontal and top-to-bottom when vertical, and it shows at least two full repeats of one cyclic color period.
    - The period contains repeated colors, but every ordered adjacent color pair in one period is unique and no reverse pair is also legal.
    - Each seed is two 4-adjacent cells whose colors are one adjacent ordered pair from that period, separated from the sample band and other seeds.

    Rule:
    - Recover the sample period and its unique ordered adjacent pairs from the sample band.
    - For each seed, choose the only reading direction whose two colors match a legal ordered pair.
    - Starting just beyond the second seed cell in that direction, write the following period colors cyclically until the grid edge.
    - Keep the sample band, every seed cell, and every other existing foreground pixel unchanged.

    Key insight:
    - A seed is a two-cell sample of the hidden periodic band: the ordered color pair simultaneously fixes both phase and physical ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    bg, period, pair_to_index, seeds = _parse_input(g)
    out = g.copy()

    for seed in seeds:
        a = tuple(map(int, seed[0]))
        b = tuple(map(int, seed[1]))
        ca, cb = int(g[a]), int(g[b])
        if (ca, cb) in pair_to_index:
            start, end = a, b
            phase = pair_to_index[(ca, cb)]
        else:
            start, end = b, a
            phase = pair_to_index[(cb, ca)]

        dr = int(end[0] - start[0])
        dc = int(end[1] - start[1])
        assert abs(dr) + abs(dc) == 1, "seed reading direction must be cardinal"

        r, c = end[0] + dr, end[1] + dc
        k = phase + 2
        while 0 <= r < h and 0 <= c < w:
            assert int(g[r, c]) == bg, "seed ray must pass only through background input cells"
            assert int(out[r, c]) == bg, "seed rays must not collide"
            out[r, c] = int(period[k % len(period)])
            r += dr
            c += dc
            k += 1

    return out


def _period_ok(seq: tuple[int, ...]) -> bool:
    if len(set(seq)) == len(seq):
        return False
    if _minimal_period(seq) != seq:
        return False
    try:
        _period_pair_index(seq)
    except AssertionError:
        return False
    return True


def _random_period() -> tuple[int, ...]:
    for _ in range(200):
        n = random.randint(5, 8)
        palette = random.sample(COLORS, random.randint(4, min(7, n - 1)))
        seq = [random.choice(palette)]
        used_edges: set[tuple[int, int]] = set()

        def backtrack(pos: int) -> bool:
            if pos == n:
                closing = (seq[-1], seq[0])
                if closing[0] == closing[1]:
                    return False
                if closing in used_edges or (closing[1], closing[0]) in used_edges:
                    return False
                return _period_ok(tuple(seq))

            choices = list(palette)
            random.shuffle(choices)
            for color in choices:
                if color == seq[-1]:
                    continue
                edge = (seq[-1], int(color))
                if edge in used_edges or (edge[1], edge[0]) in used_edges:
                    continue
                used_edges.add(edge)
                seq.append(int(color))
                if backtrack(pos + 1):
                    return True
                seq.pop()
                used_edges.remove(edge)
            return False

        if backtrack(1):
            return tuple(int(x) for x in seq)
    raise RuntimeError("failed to build a valid sample period")


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _ray_from(second: Coord, step: Coord, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    dr, dc = step
    r, c = second[0] + dr, second[1] + dc
    cells: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _touches_object(cells: list[Coord], objects: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in objects:
                    return True
    return False


def _choose_seed_dirs(n: int) -> list[Coord]:
    dirs = [random.choice(DIRS) for _ in range(n)]
    if len({0 if dr else 1 for dr, dc in dirs}) == 1:
        dirs[0] = random.choice(((-1, 0), (1, 0)))
        dirs[1] = random.choice(((0, -1), (0, 1)))
    if not any(step in ((-1, 0), (0, -1)) for step in dirs):
        dirs[0] = random.choice(((-1, 0), (0, -1)))
    if not any(step in ((1, 0), (0, 1)) for step in dirs):
        dirs[1] = random.choice(((1, 0), (0, 1)))
    random.shuffle(dirs)
    return dirs


def _place_seed(
    grid: np.ndarray,
    period: tuple[int, ...],
    phase: int,
    step: Coord,
    object_cells: set[Coord],
    output_occupied: set[Coord],
) -> tuple[list[Coord], list[Coord]] | None:
    shape = grid.shape
    h, w = shape
    dr, dc = step
    for _ in range(500):
        first = (random.randrange(h), random.randrange(w))
        second = (first[0] + dr, first[1] + dc)
        if not _in_bounds(shape, second):
            continue
        ray = _ray_from(second, step, shape)
        if len(ray) < MIN_RAY_LEN:
            continue
        seed = [first, second]
        footprint = seed + ray
        if any(cell in output_occupied for cell in footprint):
            continue
        if _touches_object(seed, object_cells, shape):
            continue

        grid[first] = int(period[phase])
        grid[second] = int(period[(phase + 1) % len(period)])
        object_cells.update(seed)
        output_occupied.update(footprint)
        return seed, ray
    return None


def _build_expected(
    inp: np.ndarray,
    period: tuple[int, ...],
    placements: list[tuple[int, list[Coord]]],
) -> np.ndarray:
    out = inp.copy()
    for phase, ray in placements:
        for offset, (r, c) in enumerate(ray):
            out[r, c] = int(period[(phase + 2 + offset) % len(period)])
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    period: tuple[int, ...],
    phases: list[int],
    dirs: list[Coord],
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not _period_ok(period):
        return False
    if len(set(phases)) < 2:
        return False
    repeated_colors = {color for color in set(period) if period.count(color) > 1}
    if not any(period[phase] in repeated_colors for phase in phases):
        return False
    if len(set(dirs)) < 2:
        return False
    if len({0 if dr else 1 for dr, dc in dirs}) < 2:
        return False
    if not any(step in ((-1, 0), (0, -1)) for step in dirs):
        return False
    if int(np.count_nonzero(expected != inp)) < MIN_RAY_LEN * len(phases):
        return False
    try:
        bg, parsed_period, _pairs, seeds = _parse_input(inp)
        solved = main(inp)
    except AssertionError:
        return False
    if bg != BG or parsed_period != period or len(seeds) != len(phases):
        return False
    if not np.array_equal(solved, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    period = _random_period()
    cycles = random.choice((2, 3))
    band_len = len(period) * cycles
    horizontal = random.random() < 0.5

    if horizontal:
        h = random.randint(16, 30)
        w = random.randint(max(16, band_len + 2), 30)
    else:
        h = random.randint(max(16, band_len + 2), 30)
        w = random.randint(16, 30)

    grid = np.full((h, w), BG, dtype=int)
    object_cells: set[Coord] = set()
    output_occupied: set[Coord] = set()

    if horizontal:
        row = random.randint(1, h - 2)
        c0 = random.randint(1, w - band_len - 1)
        band = [(row, c0 + i) for i in range(band_len)]
    else:
        r0 = random.randint(1, h - band_len - 1)
        col = random.randint(1, w - 2)
        band = [(r0 + i, col) for i in range(band_len)]

    for i, (r, c) in enumerate(band):
        grid[r, c] = int(period[i % len(period)])
    object_cells.update(band)
    output_occupied.update(band)

    n_seeds = random.randint(3, min(6, len(period)))
    repeated_phase_options = [i for i, color in enumerate(period) if period.count(color) > 1]
    phases = [random.choice(repeated_phase_options)]
    remaining = [i for i in range(len(period)) if i not in phases]
    phases.extend(random.sample(remaining, n_seeds - 1))
    random.shuffle(phases)
    dirs = _choose_seed_dirs(n_seeds)

    placements: list[tuple[int, list[Coord]]] = []
    for phase, step in zip(phases, dirs):
        placed = _place_seed(grid, period, int(phase), step, object_cells, output_occupied)
        if placed is None:
            return None
        _seed, ray = placed
        placements.append((int(phase), ray))

    expected = _build_expected(grid, period, placements)
    if not _semantic_checks(grid, expected, period, phases, dirs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/4/002/7 input")
