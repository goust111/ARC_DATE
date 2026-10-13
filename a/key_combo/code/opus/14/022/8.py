import random

import numpy as np


BACKGROUND = 0
PERIOD_CHOICES = (3, 4, 5)
HEIGHT_RANGE = (6, 9)
WIDTH_MAX = 28
MAX_GENERATION_ATTEMPTS = 500
def _rot180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(np.asarray(grid, dtype=int), 2).copy()


def _visible_width(grid: np.ndarray, bg: int) -> int:
    cols = np.where(np.any(grid != bg, axis=0))[0]
    assert cols.size > 0, "expected at least one visible foreground column"
    assert int(cols[0]) == 0, "expected the visible motif to start at the left edge"
    return int(cols[-1]) + 1


def _alternating_fill(tile: np.ndarray, width: int) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    height, period = tile.shape
    out = np.full((height, width), BACKGROUND, dtype=int)
    rotated = _rot180(tile)
    for start in range(0, width, period):
        source = tile if (start // period) % 2 == 0 else rotated
        seg_w = min(period, width - start)
        out[:, start : start + seg_w] = source[:, :seg_w]
    return out


def _plain_fill(tile: np.ndarray, width: int) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    height, period = tile.shape
    out = np.full((height, width), BACKGROUND, dtype=int)
    for start in range(0, width, period):
        seg_w = min(period, width - start)
        out[:, start : start + seg_w] = tile[:, :seg_w]
    return out


def _matches_period(grid: np.ndarray, bg: int, period: int) -> bool:
    if period <= 0:
        return False
    visible = _visible_width(grid, bg)
    if visible < 2 * period:
        return False

    tile = np.asarray(grid[:, :period], dtype=int)
    if not np.any(tile != bg):
        return False

    rotated = _rot180(tile)
    for start in range(0, visible, period):
        expected = tile if (start // period) % 2 == 0 else rotated
        seg_w = min(period, visible - start)
        if not np.array_equal(grid[:, start : start + seg_w], expected[:, :seg_w]):
            return False

    return bool(np.all(grid[:, visible:] == bg))


def _candidate_periods(grid: np.ndarray, bg: int) -> list[int]:
    visible = _visible_width(grid, bg)
    return [period for period in range(1, visible // 2 + 1) if _matches_period(grid, bg, period)]


def _infer_period(grid: np.ndarray, bg: int) -> int:
    candidates = _candidate_periods(grid, bg)
    assert candidates, "expected at least one alternating rotation period"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the left-edge motif by alternating a tile with its 180-degree rotation.

    Input structure:
    - The grid has black background and exactly one visible foreground color.
    - A motif occupies the left prefix: the first P columns form a tile T and the next P columns form rot180(T).
    - All columns to the right of that visible prefix are background.

    Rule:
    - Detect the smallest column width P for which the visible prefix is consistent with alternating copies of T and rot180(T).
    - Use the first P columns as T.
    - Fill the full grid width left to right with T, rot180(T), T, rot180(T), truncating the last copy if needed.

    Key insight:
    - The visible two-copy prefix is not a simple repeated block; it reveals both the tile width and the orientation parity of later copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BACKGROUND
    period = _infer_period(g, bg)
    tile = g[:, :period].copy()
    return _alternating_fill(tile, g.shape[1])


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _band_tile(height: int, period: int, color: int) -> np.ndarray:
    tile = np.zeros((height, period), dtype=int)
    top = random.randint(0, max(0, height - 3))
    bottom = random.randint(top + 1, height - 1)
    for col in range(period):
        if col > 0:
            top = _clamp(top + random.randint(-2, 2), 0, height - 2)
            bottom = _clamp(bottom + random.randint(-2, 2), top + 1, height - 1)
        tile[top : bottom + 1, col] = color
    return tile


def _split_tile(height: int, period: int, color: int) -> np.ndarray:
    tile = _band_tile(height, period, color)
    split_cols = random.sample(range(period), k=random.randint(1, period))
    for col in split_cols:
        rows = np.where(tile[:, col] == color)[0]
        if rows.size >= 4:
            hole_r = random.randint(int(rows[0]) + 1, int(rows[-1]) - 1)
            tile[hole_r, col] = BACKGROUND
    accent_count = random.randint(1, max(1, period - 1))
    for _ in range(accent_count):
        col = random.randrange(period)
        empty_rows = np.where(tile[:, col] == BACKGROUND)[0]
        if empty_rows.size:
            tile[int(random.choice(empty_rows)), col] = color
    return tile


def _stair_tile(height: int, period: int, color: int) -> np.ndarray:
    tile = np.zeros((height, period), dtype=int)
    current = random.randint(1, height - 1)
    for col in range(period):
        if col > 0:
            current = _clamp(current + random.randint(-2, 2), 1, height - 1)
        tile[height - current :, col] = color
        if random.random() < 0.35 and current < height:
            tile[random.randint(0, height - current - 1), col] = color
    return tile


def _sample_tile(height: int, period: int, color: int) -> np.ndarray:
    mode = random.choice(("band", "split", "stair"))
    if mode == "band":
        return _band_tile(height, period, color)
    if mode == "split":
        return _split_tile(height, period, color)
    return _stair_tile(height, period, color)


def _tile_is_clear(tile: np.ndarray, color: int) -> bool:
    if np.any(np.all(tile == BACKGROUND, axis=0)):
        return False
    fg = int(np.count_nonzero(tile == color))
    if fg < tile.shape[1] + 2:
        return False
    if fg > int(tile.size * 0.65):
        return False
    if len({tuple(tile[:, col]) for col in range(tile.shape[1])}) < 2:
        return False
    if len({tuple(tile[row, :]) for row in range(tile.shape[0])}) < 2:
        return False
    return True


def _sample_width(period: int) -> int:
    non_multiple = [w for w in range(2 * period + 1, WIDTH_MAX + 1) if w % period != 0]
    multiples = [w for w in range(3 * period, WIDTH_MAX + 1, period)]
    if random.random() < 0.5 and non_multiple:
        return int(random.choice(non_multiple))
    if multiples:
        return int(random.choice(multiples))
    return int(random.choice(non_multiple))


def _validate_example(inp: np.ndarray, out: np.ndarray, period: int, color: int) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.count_nonzero(inp == BACKGROUND) <= np.count_nonzero(inp != BACKGROUND):
        return False
    if period not in PERIOD_CHOICES:
        return False
    if not np.all(inp[:, 2 * period :] == BACKGROUND):
        return False
    if not np.any(inp[:, 0] == color):
        return False
    if not np.any(inp[:, 2 * period - 1] == color):
        return False

    tile = inp[:, :period].copy()
    if not _tile_is_clear(tile, color):
        return False
    if np.array_equal(tile, _rot180(tile)):
        return False

    candidates = _candidate_periods(inp, BACKGROUND)
    if candidates != [period]:
        return False
    if not np.array_equal(inp[:, period : 2 * period], _rot180(tile)):
        return False

    expected = _alternating_fill(tile, inp.shape[1])
    if not np.array_equal(out, expected):
        return False
    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(out, _plain_fill(tile, inp.shape[1])):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    period = random.choice(PERIOD_CHOICES)
    height = random.randint(*HEIGHT_RANGE)
    width = _sample_width(period)
    color = random.randint(1, 9)

    for _ in range(120):
        tile = _sample_tile(height, period, color)
        if not _tile_is_clear(tile, color):
            continue
        if np.array_equal(tile, _rot180(tile)):
            continue

        inp = np.zeros((height, width), dtype=int)
        inp[:, : 2 * period] = _alternating_fill(tile, 2 * period)
        out = _alternating_fill(tile, width)
        if _validate_example(inp, out, period, color):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alternating-rotation tiling input")
