from __future__ import annotations

import numpy as np


BASE_COLORS = (1, 2, 3)
DISTRACTOR_COLORS = (4, 7)
PERIOD_CHOICES = ((2, 2), (3, 2))
GRID_HEIGHT_RANGE = (8, 13)
GRID_WIDTH_RANGE = (8, 13)
DISTRACTOR_COUNT_CHOICES = (2, 3, 4)
MAX_REPAIRS = 4
MAX_PERIOD_ROWS = 4
MAX_PERIOD_COLS = 4
MAX_ATTEMPTS = 600

RNG = np.random.default_rng()


def _period_key(period: tuple[int, int]) -> tuple[int, int, int]:
    pr, pc = map(int, period)
    return pr * pc, pr, pc


def _render_periodic(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    pr, pc = map(int, tile.shape)
    row_ids = np.arange(height) % pr
    col_ids = np.arange(width) % pc
    return tile[row_ids[:, None], col_ids[None, :]]


def _candidate_period(grid: np.ndarray, pr: int, pc: int) -> dict | None:
    g = np.asarray(grid, dtype=int)
    tile = np.zeros((pr, pc), dtype=int)
    repairs = 0

    for rr in range(pr):
        for cc in range(pc):
            values = g[rr::pr, cc::pc].ravel().astype(int)
            counts = np.bincount(values, minlength=10)
            best_count = int(counts.max())
            winners = np.flatnonzero(counts == best_count)
            if len(winners) != 1:
                return None
            tile[rr, cc] = int(winners[0])
            repairs += int(values.size - best_count)

    if len(np.unique(tile)) < 2:
        return None

    repaired = _render_periodic(tile, g.shape)
    return {
        "period": (int(pr), int(pc)),
        "tile": tile,
        "repaired": repaired,
        "loss": int(repairs),
    }


def _all_small_period_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    out: list[dict] = []
    for pr in range(1, min(height, MAX_PERIOD_ROWS) + 1):
        for pc in range(1, min(width, MAX_PERIOD_COLS) + 1):
            cand = _candidate_period(g, pr, pc)
            if cand is not None:
                out.append(cand)
    assert out
    return out


def _select_best_candidate(grid: np.ndarray) -> dict:
    candidates = _all_small_period_candidates(grid)
    near_exact = [cand for cand in candidates if cand["loss"] <= MAX_REPAIRS]
    if near_exact:
        return min(near_exact, key=lambda cand: _period_key(cand["period"]))
    return min(candidates, key=lambda cand: (cand["loss"],) + _period_key(cand["period"]))


def _exact_period(grid: np.ndarray, pr: int, pc: int) -> bool:
    g = np.asarray(grid, dtype=int)
    tile = g[:pr, :pc]
    return np.array_equal(g, _render_periodic(tile, g.shape))


def _minimal_exact_period(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    best: tuple[int, int] | None = None
    for pr in range(1, height + 1):
        for pc in range(1, width + 1):
            if not _exact_period(g, pr, pc):
                continue
            tile = g[:pr, :pc]
            if len(np.unique(tile)) < 2:
                continue
            period = (int(pr), int(pc))
            if best is None or _period_key(period) < _period_key(best):
                best = period
    assert best is not None
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the smallest repeating tile, erase the few periodicity-violating pixels, then mirror left-to-right.

    Input structure:
    - The whole grid is filled with nonzero colors, with no black background.
    - Most cells follow a small 2D periodic pattern built from blue, red, and green.
    - A few scattered pixels break that periodic pattern because they were replaced by distractor colors.

    Rule:
    - Search small row and column periods and group cells by `(row mod p_r, col mod p_c)`.
    - For the smallest period whose residue classes can be made consistent with only a few edits, assign each residue class its majority color.
    - Repair every inconsistent cell to that tile color, then horizontally mirror the repaired grid.

    Key insight:
    - The wrong pixels are defined by violating a hidden periodic template, so the period must be inferred before the mirror step can be applied correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    best = _select_best_candidate(g)
    return np.fliplr(best["repaired"])


def _sample_tile(period: tuple[int, int]) -> np.ndarray:
    pr, pc = map(int, period)
    while True:
        tile = RNG.choice(BASE_COLORS, size=(pr, pc), replace=True).astype(int)
        if len(np.unique(tile)) < 2:
            continue
        if _minimal_exact_period(_render_periodic(tile, (pr * 4, pc * 5))) == period:
            return tile


def _sample_shape(period: tuple[int, int]) -> tuple[int, int]:
    pr, pc = map(int, period)
    min_h = max(GRID_HEIGHT_RANGE[0], pr * 3)
    min_w = max(GRID_WIDTH_RANGE[0], pc * 4)
    while True:
        height = int(RNG.integers(min_h, GRID_HEIGHT_RANGE[1]))
        width = int(RNG.integers(min_w, GRID_WIDTH_RANGE[1]))
        if height <= 30 and width <= 30:
            return height, width


def _residue_buckets(shape: tuple[int, int], period: tuple[int, int]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    height, width = map(int, shape)
    pr, pc = map(int, period)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for rr in range(pr):
        for cc in range(pc):
            buckets[(rr, cc)] = []
    for r in range(height):
        for c in range(width):
            buckets[(r % pr, c % pc)].append((r, c))
    return buckets


def _inject_distractors(clean: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    g = clean.copy()
    buckets = _residue_buckets(g.shape, period)
    count = int(RNG.choice(DISTRACTOR_COUNT_CHOICES))
    residue_keys = list(buckets)
    chosen_residues = RNG.choice(len(residue_keys), size=count, replace=False)
    chosen_positions: list[tuple[int, int]] = []
    for idx in chosen_residues:
        bucket = buckets[residue_keys[int(idx)]]
        pos = bucket[int(RNG.integers(len(bucket)))]
        chosen_positions.append(pos)

    for r, c in chosen_positions:
        base = int(g[r, c])
        color_choices = [color for color in DISTRACTOR_COLORS if color != base]
        g[r, c] = int(RNG.choice(color_choices))
    return g


def _validate_case(inp: np.ndarray, clean: np.ndarray, period: tuple[int, int]) -> bool:
    if inp.ndim != 2 or clean.ndim != 2 or inp.shape != clean.shape:
        return False
    if max(inp.shape) > 30:
        return False
    if np.any(inp == 0) or np.any(clean == 0):
        return False

    diff_mask = inp != clean
    diff_coords = np.argwhere(diff_mask)
    noise_count = int(diff_mask.sum())
    if noise_count not in DISTRACTOR_COUNT_CHOICES:
        return False

    noise_residues = {(int(r) % period[0], int(c) % period[1]) for r, c in diff_coords}
    if len(noise_residues) != noise_count:
        return False

    if not set(np.unique(clean)).issubset(BASE_COLORS):
        return False
    if set(np.unique(inp[diff_mask])) - set(DISTRACTOR_COLORS):
        return False
    if _minimal_exact_period(clean) != period:
        return False
    if np.array_equal(clean, np.fliplr(clean)):
        return False

    candidates = _all_small_period_candidates(inp)
    best = _select_best_candidate(inp)
    if best["period"] != period:
        return False
    if best["loss"] != noise_count:
        return False

    for cand in candidates:
        if _period_key(cand["period"]) < _period_key(period) and cand["loss"] <= MAX_REPAIRS:
            return False

    out = main(inp)
    if not np.array_equal(out, np.fliplr(clean)):
        return False
    if np.array_equal(out, np.fliplr(inp)):
        return False
    if np.array_equal(out, clean):
        return False
    if int(np.count_nonzero(out != np.fliplr(inp))) != noise_count:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    period = PERIOD_CHOICES[int(RNG.integers(len(PERIOD_CHOICES)))]
    height, width = _sample_shape(period)
    tile = _sample_tile(period)
    clean = _render_periodic(tile, (height, width))
    inp = _inject_distractors(clean, period)

    if not _validate_case(inp, clean, period):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid periodic-noise ARC input")
