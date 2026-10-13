from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


ARC_COLORS = tuple(range(10))
PERIOD_H_RANGE = (2, 6)
PERIOD_W_RANGE = (2, 6)
REPEAT_MIN = 1
REPEAT_MAX = 5
GRID_SIDE_MIN = 6
PALETTE_SIZE_RANGE = (3, 5)
DEFECT_COUNT_MAX = 5
GENERATION_RETRIES = 500


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _best_color(counter: Counter[int]) -> int:
    best = max(counter.values())
    return min(color for color, count in counter.items() if count == best)


def _majority_tile(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    out = np.zeros((tile_h, tile_w), dtype=int)
    full_h, full_w = grid.shape
    for phase_r in range(tile_h):
        for phase_c in range(tile_w):
            counts = Counter(
                int(grid[r, c])
                for r in range(phase_r, full_h, tile_h)
                for c in range(phase_c, full_w, tile_w)
            )
            out[phase_r, phase_c] = _best_color(counts)
    return out


def _folded_defects(grid: np.ndarray, base_tile: np.ndarray) -> dict[tuple[int, int], Counter[int]]:
    tile_h, tile_w = base_tile.shape
    mismatches: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            phase = (r % tile_h, c % tile_w)
            color = int(grid[r, c])
            if color != int(base_tile[phase]):
                mismatches[phase][color] += 1
    return mismatches


def _period_cost(grid: np.ndarray, tile_h: int, tile_w: int) -> int:
    base_tile = _majority_tile(grid, tile_h, tile_w)
    cost = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != int(base_tile[r % tile_h, c % tile_w]):
                cost += 1
    return cost


def _infer_periods(grid: np.ndarray, *, min_copies: int = 2) -> tuple[int, int]:
    full_h, full_w = grid.shape
    candidates = [
        (tile_h, tile_w)
        for tile_h in _divisors(full_h)
        for tile_w in _divisors(full_w)
        if (full_h // tile_h) * (full_w // tile_w) >= min_copies
    ]
    if not candidates:
        candidates = [(full_h, full_w)]

    best_key = None
    best_periods = None
    for tile_h, tile_w in candidates:
        key = (_period_cost(grid, tile_h, tile_w), tile_h * tile_w, tile_h, tile_w)
        if best_key is None or key < best_key:
            best_key = key
            best_periods = (tile_h, tile_w)

    assert best_periods is not None
    return best_periods


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    out = np.empty(shape, dtype=int)
    tile_h, tile_w = tile.shape
    for r in range(shape[0]):
        for c in range(shape[1]):
            out[r, c] = int(tile[r % tile_h, c % tile_w])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the smallest repeated tile, then fold each off-pattern pixel back to its tile phase and overprint it.

    Input structure:
    - The grid is a rectangular periodic tiling of one base tile along both axes, except for a few recolored defect pixels.
    - The base tile height divides the grid height and the base tile width divides the grid width.
    - Any two defect pixels occupy different phase coordinates inside the repeated tile.

    Rule:
    - Infer the minimum vertical and horizontal periods of the repeated tiling.
    - For each tile phase, use the majority color among all of its copies as the correct base-tile color.
    - Output one base tile: first paint it with those majority colors, then overprint each defect pixel's observed color at the corresponding phase coordinate.

    Key insight:
    - The visible repetition defines a modulo coordinate system; the sparse anomalies carry the extra pattern, and that pattern only appears after folding every defect back into one period cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    tile_h, tile_w = _infer_periods(g)
    base_tile = _majority_tile(g, tile_h, tile_w)
    out = base_tile.copy()

    for phase, colors in _folded_defects(g, base_tile).items():
        out[phase] = _best_color(colors)
    return out


def _exact_periods(tile: np.ndarray) -> tuple[int, int]:
    return _infer_periods(tile, min_copies=1)


def _choose_dimensions() -> tuple[int, int, int, int] | None:
    tile_h = random.randint(*PERIOD_H_RANGE)
    tile_w = random.randint(*PERIOD_W_RANGE)

    max_rep_h = min(REPEAT_MAX, 30 // tile_h)
    max_rep_w = min(REPEAT_MAX, 30 // tile_w)
    if max_rep_h < REPEAT_MIN or max_rep_w < REPEAT_MIN:
        return None

    rep_h = random.randint(REPEAT_MIN, max_rep_h)
    rep_w = random.randint(REPEAT_MIN, max_rep_w)
    full_h = tile_h * rep_h
    full_w = tile_w * rep_w
    if full_h < GRID_SIDE_MIN or full_w < GRID_SIDE_MIN:
        return None
    if rep_h * rep_w < 3:
        return None
    return tile_h, tile_w, rep_h, rep_w


def _random_tile(tile_h: int, tile_w: int) -> np.ndarray | None:
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(ARC_COLORS, palette_size)
    tile = np.array(
        [[random.choice(palette) for _c in range(tile_w)] for _r in range(tile_h)],
        dtype=int,
    )
    if len(set(map(int, tile.ravel()))) < 2:
        return None
    if _exact_periods(tile) != (tile_h, tile_w):
        return None
    return tile


def _phase_neighbors(cell: tuple[int, int], tile_h: int, tile_w: int):
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < tile_h and 0 <= cc < tile_w:
            yield rr, cc


def _phase_cluster(tile_h: int, tile_w: int, count: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(tile_h), random.randrange(tile_w))
    chosen = {start}
    frontier = set(_phase_neighbors(start, tile_h, tile_w))
    while len(chosen) < count:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nb in _phase_neighbors(cell, tile_h, tile_w):
            if nb not in chosen:
                frontier.add(nb)
    return list(chosen)


def _choose_defect_phases(tile_h: int, tile_w: int) -> list[tuple[int, int]] | None:
    max_defects = min(DEFECT_COUNT_MAX, tile_h * tile_w // 2 or 1)
    if max_defects <= 0:
        return None
    if max_defects == 1 or random.random() < 0.4:
        defect_count = 1
    else:
        defect_count = random.randint(2, max_defects)
    return _phase_cluster(tile_h, tile_w, defect_count)


def _choose_defect_colors(tile: np.ndarray, phases: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    base_colors = set(map(int, tile.ravel()))
    preferred = [color for color in ARC_COLORS if color not in base_colors]
    if not preferred:
        preferred = list(ARC_COLORS)

    if len(phases) == 1 or random.random() < 0.55:
        palette = [random.choice(preferred)]
    else:
        count = min(2, len(preferred), len(phases))
        palette = random.sample(preferred, count)

    defects: dict[tuple[int, int], int] = {}
    for phase in phases:
        choices = [color for color in palette if color != int(tile[phase])]
        if not choices:
            choices = [color for color in ARC_COLORS if color != int(tile[phase])]
        defects[phase] = random.choice(choices)
    return defects


def _scatter_defects(
    grid: np.ndarray,
    tile_h: int,
    tile_w: int,
    rep_h: int,
    rep_w: int,
    defect_colors: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    copies = [(kr, kc) for kr in range(rep_h) for kc in range(rep_w)]
    assignments: list[tuple[int, int]]
    if len(defect_colors) <= len(copies):
        assignments = random.sample(copies, len(defect_colors))
    else:
        assignments = [random.choice(copies) for _ in range(len(defect_colors))]

    scattered = []
    phases = list(defect_colors.items())
    random.shuffle(phases)
    for (phase_r, phase_c), color in phases:
        copy_r, copy_c = assignments.pop() if assignments else random.choice(copies)
        abs_r = copy_r * tile_h + phase_r
        abs_c = copy_c * tile_w + phase_c
        grid[abs_r, abs_c] = int(color)
        scattered.append(((phase_r, phase_c), (copy_r, copy_c), int(color)))
    return scattered


def _validate_case(
    inp: np.ndarray,
    clean_tile: np.ndarray,
    defect_colors: dict[tuple[int, int], int],
    expected: np.ndarray,
) -> bool:
    tile_h, tile_w = clean_tile.shape
    full_h, full_w = inp.shape
    if full_h > 30 or full_w > 30:
        return False
    if full_h % tile_h or full_w % tile_w:
        return False
    if _exact_periods(clean_tile) != (tile_h, tile_w):
        return False
    if len(defect_colors) != len(set(defect_colors)):
        return False
    if (full_h // tile_h) * (full_w // tile_w) < 3:
        return False
    for phase, color in defect_colors.items():
        if int(clean_tile[phase]) == int(color):
            return False

    if _infer_periods(inp) != (tile_h, tile_w):
        return False
    if not np.array_equal(_majority_tile(inp, tile_h, tile_w), clean_tile):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    dims = _choose_dimensions()
    if dims is None:
        return None
    tile_h, tile_w, rep_h, rep_w = dims

    clean_tile = _random_tile(tile_h, tile_w)
    if clean_tile is None:
        return None

    phases = _choose_defect_phases(tile_h, tile_w)
    if not phases:
        return None
    defect_colors = _choose_defect_colors(clean_tile, phases)

    inp = _tile_fill((tile_h * rep_h, tile_w * rep_w), clean_tile)
    _scatter_defects(inp, tile_h, tile_w, rep_h, rep_w, defect_colors)

    expected = clean_tile.copy()
    for phase, color in defect_colors.items():
        expected[phase] = int(color)

    if not _validate_case(inp, clean_tile, defect_colors, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid folded-period ARC input")
