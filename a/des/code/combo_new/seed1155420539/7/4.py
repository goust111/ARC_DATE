from __future__ import annotations

import random

import numpy as np


BORDER_COLOR = 8
NOISE_COLOR = 9

TILE_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7)
TILE_HEIGHT_RANGE = (2, 5)
TILE_WIDTH_RANGE = (2, 5)
GRID_TILE_COUNT_RANGE = (4, 7)
GENERATION_ATTEMPTS = 400


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == BORDER_COLOR
    assert np.any(mask), "expected an azure frame"

    rows, cols = np.where(mask)
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    expected = np.zeros_like(mask)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(mask, expected), "expected a single one-pixel rectangular frame"
    return r0, c0, r1, c1


def _candidate_tile(grid: np.ndarray, tile_h: int, tile_w: int, ignored: np.ndarray) -> np.ndarray | None:
    tile = np.full((tile_h, tile_w), -1, dtype=int)
    seen = np.zeros((tile_h, tile_w), dtype=bool)
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if bool(ignored[r, c]):
                continue
            rr = r % tile_h
            cc = c % tile_w
            value = int(grid[r, c])
            if bool(seen[rr, cc]):
                if int(tile[rr, cc]) != value:
                    return None
            else:
                tile[rr, cc] = value
                seen[rr, cc] = True

    if not bool(seen.all()):
        return None
    if len({int(x) for x in tile.flatten()}) < 2:
        return None
    return tile


def _infer_tile(grid: np.ndarray) -> np.ndarray:
    ignored = (grid == BORDER_COLOR) | (grid == NOISE_COLOR)
    h, w = grid.shape
    best_score = None
    best_tile = None

    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            tile = _candidate_tile(grid, tile_h, tile_w, ignored)
            if tile is None:
                continue
            score = (tile_h * tile_w, tile_h, tile_w)
            if best_score is None or score < best_score:
                best_score = score
                best_tile = tile

    assert best_tile is not None, "failed to infer a periodic tile"
    return best_tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the corrupted periodic tile wallpaper, repair the marked crop, and return only that repaired interior.

    Input structure:
    - The full grid is an occluded repetition of a small rectangular tile that uses two or three non-marker colors.
    - Some tile instances contain maroon noise pixels; maroon is not part of the true tile pattern.
    - One azure one-pixel rectangular frame marks a region whose interior spans whole tile copies.

    Rule:
    - Infer the smallest tile period that makes all non-maroon, non-azure pixels consistent at each modulo position.
    - Find the azure frame and take only its interior as the output area.
    - Replace every maroon pixel inside that interior with the inferred tile color at the same modulo position, leaving clean pixels unchanged.

    Key insight:
    The maroon pixels and the azure frame are both overlays on top of a hidden wallpaper: first recover the wallpaper period, then use the frame only as a crop-and-repair instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile = _infer_tile(g)
    tile_h, tile_w = map(int, tile.shape)
    r0, c0, r1, c1 = _frame_bbox(g)

    interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    assert interior.size > 0, "frame interior must be non-empty"
    assert interior.shape[0] % tile_h == 0 and interior.shape[1] % tile_w == 0, "marked area must cover whole tiles"

    for rr in range(interior.shape[0]):
        for cc in range(interior.shape[1]):
            if int(interior[rr, cc]) == NOISE_COLOR:
                interior[rr, cc] = tile[(r0 + 1 + rr) % tile_h, (c0 + 1 + cc) % tile_w]

    return interior


def _minimal_period_shape(pattern: np.ndarray) -> tuple[int, int]:
    empty = np.zeros_like(pattern, dtype=bool)
    best_score = None
    best_shape = None
    h, w = pattern.shape

    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            tile = _candidate_tile(pattern, tile_h, tile_w, empty)
            if tile is None:
                continue
            score = (tile_h * tile_w, tile_h, tile_w)
            if best_score is None or score < best_score:
                best_score = score
                best_shape = (tile_h, tile_w)

    assert best_shape is not None
    return best_shape


def _random_tile(rng: random.Random) -> np.ndarray | None:
    tile_h = rng.randint(*TILE_HEIGHT_RANGE)
    tile_w = rng.randint(*TILE_WIDTH_RANGE)
    n_colors = rng.choice((2, 3))
    colors = rng.sample(TILE_COLOR_POOL, n_colors)

    tile = np.array([[rng.choice(colors) for _ in range(tile_w)] for _ in range(tile_h)], dtype=int)
    if len({int(x) for x in tile.flatten()}) < 2:
        return None
    if _minimal_period_shape(tile) != (tile_h, tile_w):
        return None
    return tile


def _periodic_canvas(tile: np.ndarray, height: int, width: int, row_offset: int, col_offset: int) -> np.ndarray:
    tile_h, tile_w = map(int, tile.shape)
    rows = (np.arange(height) + row_offset) % tile_h
    cols = (np.arange(width) + col_offset) % tile_w
    return tile[rows[:, None], cols[None, :]].copy()


def _aligned_starts(limit: int, interior_size: int, period: int, offset: int) -> list[int]:
    starts = []
    for start in range(1, limit - interior_size):
        if (start + offset) % period == 0:
            starts.append(int(start))
    return starts


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = BORDER_COLOR
    grid[r1, c0 : c1 + 1] = BORDER_COLOR
    grid[r0 : r1 + 1, c0] = BORDER_COLOR
    grid[r0 : r1 + 1, c1] = BORDER_COLOR


def _place_noise(
    grid: np.ndarray,
    inner_bbox: tuple[int, int, int, int],
    tile_h: int,
    tile_w: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = inner_bbox
    border_mask = grid == BORDER_COLOR
    all_positions = [(int(r), int(c)) for r, c in zip(*np.where(~border_mask))]
    counts: dict[tuple[int, int], int] = {}
    for r, c in all_positions:
        key = (r % tile_h, c % tile_w)
        counts[key] = counts.get(key, 0) + 1

    inside_positions = [(r, c) for r, c in all_positions if r0 <= r <= r1 and c0 <= c <= c1]
    outside_positions = [(r, c) for r, c in all_positions if not (r0 <= r <= r1 and c0 <= c <= c1)]
    rng.shuffle(inside_positions)
    rng.shuffle(outside_positions)

    max_addable = sum(count - 1 for count in counts.values())
    if max_addable < 2:
        return []

    target_total = rng.randint(2, min(12, max_addable))
    noisy: list[tuple[int, int]] = []
    chosen: set[tuple[int, int]] = set()

    def try_add(position: tuple[int, int]) -> bool:
        if position in chosen:
            return False
        r, c = position
        key = (r % tile_h, c % tile_w)
        if counts[key] <= 1:
            return False
        counts[key] -= 1
        grid[r, c] = NOISE_COLOR
        chosen.add(position)
        noisy.append(position)
        return True

    inside_added = False
    for position in inside_positions:
        if try_add(position):
            inside_added = True
            break
    if not inside_added:
        return []

    outside_added = False
    for position in outside_positions:
        if try_add(position):
            outside_added = True
            break
    if not outside_added and outside_positions:
        return []

    remaining = inside_positions + outside_positions
    rng.shuffle(remaining)
    for position in remaining:
        if len(noisy) >= target_total:
            break
        try_add(position)

    return noisy


def _expected_crop(tile: np.ndarray, inner_bbox: tuple[int, int, int, int], row_offset: int, col_offset: int) -> np.ndarray:
    r0, c0, r1, c1 = inner_bbox
    rows = (np.arange(r0, r1 + 1) + row_offset) % tile.shape[0]
    cols = (np.arange(c0, c1 + 1) + col_offset) % tile.shape[1]
    return tile[rows[:, None], cols[None, :]].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tile = _random_tile(rng)
    if tile is None:
        return None
    tile_h, tile_w = map(int, tile.shape)

    max_tile_rows = min(GRID_TILE_COUNT_RANGE[1], 30 // tile_h)
    max_tile_cols = min(GRID_TILE_COUNT_RANGE[1], 30 // tile_w)
    if max_tile_rows < GRID_TILE_COUNT_RANGE[0] or max_tile_cols < GRID_TILE_COUNT_RANGE[0]:
        return None

    grid_tile_rows = rng.randint(GRID_TILE_COUNT_RANGE[0], max_tile_rows)
    grid_tile_cols = rng.randint(GRID_TILE_COUNT_RANGE[0], max_tile_cols)
    height = grid_tile_rows * tile_h
    width = grid_tile_cols * tile_w

    inner_tile_rows = rng.randint(1, grid_tile_rows - 2)
    inner_tile_cols = rng.randint(1, grid_tile_cols - 2)
    if inner_tile_rows * inner_tile_cols < 2:
        if inner_tile_rows < grid_tile_rows - 2:
            inner_tile_rows += 1
        else:
            inner_tile_cols += 1
    if inner_tile_rows >= grid_tile_rows - 1 or inner_tile_cols >= grid_tile_cols - 1:
        return None

    row_offset = rng.randrange(tile_h)
    col_offset = rng.randrange(tile_w)
    inner_height = inner_tile_rows * tile_h
    inner_width = inner_tile_cols * tile_w

    row_starts = _aligned_starts(height, inner_height, tile_h, row_offset)
    col_starts = _aligned_starts(width, inner_width, tile_w, col_offset)
    if not row_starts or not col_starts:
        return None

    inner_r0 = rng.choice(row_starts)
    inner_c0 = rng.choice(col_starts)
    inner_r1 = inner_r0 + inner_height - 1
    inner_c1 = inner_c0 + inner_width - 1

    base = _periodic_canvas(tile, height, width, row_offset, col_offset)
    grid = base.copy()
    _paint_frame(grid, inner_r0 - 1, inner_c0 - 1, inner_r1 + 1, inner_c1 + 1)

    noisy = _place_noise(grid, (inner_r0, inner_c0, inner_r1, inner_c1), tile_h, tile_w, rng)
    if len(noisy) < 2:
        return None
    if not any(inner_r0 <= r <= inner_r1 and inner_c0 <= c <= inner_c1 for r, c in noisy):
        return None

    expected = _expected_crop(tile, (inner_r0, inner_c0, inner_r1, inner_c1), row_offset, col_offset)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-tile repair input")
