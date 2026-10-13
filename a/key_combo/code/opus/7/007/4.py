from __future__ import annotations

import random

import numpy as np


TOP_FG = 1
OUT_FG = 2
BOTTOM_FG = 3
BG = 0

HALF_HEIGHTS = (3, 4, 5)
PERIODS = (2, 3, 4)
GEN_ATTEMPTS = 1000
SEED_ATTEMPTS = 200
MASK_ATTEMPTS = 300


def _split_halves(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    assert h % 2 == 0
    mid = h // 2
    return g[:mid], g[mid:]


def _is_periodic_cols(grid: np.ndarray, period: int) -> bool:
    g = np.asarray(grid)
    _h, w = g.shape
    p = int(period)
    if p <= 0 or p > w:
        return False
    return bool(np.array_equal(g, g[:, np.arange(w) % p]))


def _minimal_col_period(grid: np.ndarray) -> int:
    g = np.asarray(grid)
    _h, w = g.shape
    for p in range(1, w + 1):
        if _is_periodic_cols(g, p):
            return p
    raise AssertionError("every finite-width grid must have a column period")


def _compress_masks(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    shared_bg = (top == BG) & (bottom == BG)
    return np.where(shared_bg, OUT_FG, BG).astype(int)


def _tile_unit(unit: np.ndarray, width: int) -> np.ndarray:
    h, p = unit.shape
    reps = (int(width) + p - 1) // p
    return np.tile(unit, (1, reps))[:, :width].reshape(h, width)


def main(grid: np.ndarray) -> np.ndarray:
    """Compress the shared background of two stacked masks, detect its shortest column period, and retile that unit.

    Input structure:
    - The grid has even height and is split by its horizontal midpoint into two equal subgrids.
    - The upper half uses black as background and blue as foreground; the lower half uses black as background and green as foreground.
    - The only structure relevant to the output is where both corresponding cells are simultaneously black.

    Rule:
    - Split the input into upper and lower halves of equal shape.
    - For each position, mark red in the compressed grid iff both halves contain black there; otherwise mark black.
    - Find the smallest positive column period of that compressed grid.
    - Take the first period-width block and repeat it horizontally until the full output width is filled.

    Key insight:
    - The usable periodic pattern is not visible in either input mask alone; it appears only after comparing the two masks cell by cell and isolating their shared background.
    """
    top, bottom = _split_halves(grid)
    compressed = _compress_masks(top, bottom)
    period = _minimal_col_period(compressed)
    unit = compressed[:, :period]
    return _tile_unit(unit, compressed.shape[1]).astype(int)


def _row_has_both_values(row: np.ndarray) -> bool:
    vals = set(map(int, row))
    return vals == {0, 1}


def _make_period_seed(height: int, period: int) -> np.ndarray:
    min_red = max(2, (height * period) // 4)
    max_red = max(min_red + 1, (3 * height * period) // 4)

    for _ in range(SEED_ATTEMPTS):
        seed = np.random.randint(0, 2, size=(height, period), dtype=int)
        if not all(_row_has_both_values(seed[r]) for r in range(height)):
            continue
        if _minimal_col_period(seed) != period:
            continue
        red = int(seed.sum())
        if not (min_red <= red <= max_red):
            continue
        row_signatures = {tuple(map(int, row)) for row in seed}
        if len(row_signatures) < 2:
            continue
        return seed.astype(bool)
    raise RuntimeError("failed to build a non-degenerate periodic seed")


def _has_short_input_period(mask: np.ndarray) -> bool:
    _h, w = mask.shape
    for p in range(1, min(4, w - 1) + 1):
        if _is_periodic_cols(mask, p):
            return True
    return False


def _balanced_half(mask: np.ndarray, fg: int) -> bool:
    bg_counts = np.sum(mask == BG, axis=1)
    fg_counts = np.sum(mask == fg, axis=1)
    return bool(np.all(bg_counts >= 2) and np.all(fg_counts >= 2))


def _build_input_from_compressed(compressed_red: np.ndarray) -> np.ndarray | None:
    h, w = compressed_red.shape
    top = np.full((h, w), TOP_FG, dtype=int)
    bottom = np.full((h, w), BOTTOM_FG, dtype=int)

    top[compressed_red] = BG
    bottom[compressed_red] = BG

    extra_top = 0
    extra_bottom = 0
    non_shared = [(int(r), int(c)) for r, c in np.argwhere(~compressed_red)]
    random.shuffle(non_shared)
    for r, c in non_shared:
        pick = random.choices((0, 1, 2), weights=(3, 3, 4))[0]
        if pick == 0:
            top[r, c] = BG
            extra_top += 1
        elif pick == 1:
            bottom[r, c] = BG
            extra_bottom += 1

    if extra_top < h or extra_bottom < h:
        return None
    if not _balanced_half(top, TOP_FG):
        return None
    if not _balanced_half(bottom, BOTTOM_FG):
        return None
    if _has_short_input_period(top):
        return None
    if _has_short_input_period(bottom):
        return None
    return np.vstack([top, bottom])


def _generate_candidate() -> np.ndarray:
    half_h = random.choice(HALF_HEIGHTS)
    period = random.choice(PERIODS)
    repeats = random.randint(3, 30 // period)
    width = period * repeats

    seed = _make_period_seed(half_h, period)
    compressed_red = _tile_unit(seed.astype(int), width).astype(bool)

    if _minimal_col_period(compressed_red.astype(int)) != period:
        raise AssertionError("tiling changed the intended minimal period")

    for _ in range(MASK_ATTEMPTS):
        grid = _build_input_from_compressed(compressed_red)
        if grid is not None:
            return grid
    raise RuntimeError("failed to hide the periodic pattern inside two non-periodic masks")


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 0 and h <= 30 and w <= 30
    assert set(map(int, np.unique(g))).issubset({BG, TOP_FG, BOTTOM_FG})

    top, bottom = _split_halves(g)
    assert _balanced_half(top, TOP_FG)
    assert _balanced_half(bottom, BOTTOM_FG)
    assert not _has_short_input_period(top)
    assert not _has_short_input_period(bottom)

    compressed = _compress_masks(top, bottom)
    red_mask = compressed == OUT_FG
    assert 0 < int(red_mask.sum()) < red_mask.size

    period = _minimal_col_period(compressed)
    assert 2 <= period <= 4
    assert w % period == 0
    assert not any(_is_periodic_cols(compressed, p) for p in range(1, period))

    shared_bg = (top == BG) & (bottom == BG)
    assert np.array_equal(shared_bg, red_mask)
    assert int(np.sum((top == BG) & ~shared_bg)) >= top.shape[0]
    assert int(np.sum((bottom == BG) & ~shared_bg)) >= bottom.shape[0]

    expected = _tile_unit(compressed[:, :period], w)
    out = main(g)
    assert out.shape == (h // 2, w)
    assert set(map(int, np.unique(out))).issubset({BG, OUT_FG})
    assert np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _generate_candidate()
        except RuntimeError:
            continue
        _validate_instance(grid)
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
