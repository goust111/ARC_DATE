from __future__ import annotations

import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _crop_mask(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "mask must be non-empty"
    r0, c0, r1, c1 = box
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    submask = mask[r0 : r1 + 1, c0 : c1 + 1]
    subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]
    out[submask] = subgrid[submask]
    return out


def _repeat_row_score(grid: np.ndarray, period: int) -> int:
    h = grid.shape[0]
    return sum(int(np.array_equal(grid[r], grid[r + period])) for r in range(h - period))


def _repeat_col_score(grid: np.ndarray, period: int) -> int:
    w = grid.shape[1]
    return sum(int(np.array_equal(grid[:, c], grid[:, c + period])) for c in range(w - period))


def _majority_tile(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray:
    tile = np.empty((row_period, col_period), dtype=int)
    for rr in range(row_period):
        for cc in range(col_period):
            values = grid[rr::row_period, cc::col_period]
            counts = np.bincount(values.ravel(), minlength=10)
            tile[rr, cc] = int(np.argmax(counts))
    return tile


def _tile_grid(tile: np.ndarray, shape: tuple[int, int], *, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    h, w = shape
    row_period, col_period = tile.shape
    row_offset, col_offset = offset
    rows = (np.arange(h)[:, None] - int(row_offset)) % row_period
    cols = (np.arange(w)[None, :] - int(col_offset)) % col_period
    return tile[rows, cols]


def _primitive_periods(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    row_period = h
    for candidate in range(1, h + 1):
        ok = True
        for r in range(h):
            if not np.array_equal(tile[r], tile[r % candidate]):
                ok = False
                break
        if ok:
            row_period = candidate
            break

    col_period = w
    for candidate in range(1, w + 1):
        ok = True
        for c in range(w):
            if not np.array_equal(tile[:, c], tile[:, c % candidate]):
                ok = False
                break
        if ok:
            col_period = candidate
            break
    return row_period, col_period


def _infer_anomaly_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_scores = {period: _repeat_row_score(g, period) for period in range(1, h)}
    col_scores = {period: _repeat_col_score(g, period) for period in range(1, w)}

    def consider_pairs(
        row_periods: list[int],
        col_periods: list[int],
        best_key: tuple[int, int, int, int, int] | None,
        best_mask: np.ndarray | None,
    ) -> tuple[tuple[int, int, int, int, int] | None, np.ndarray | None]:
        for row_period in row_periods:
            for col_period in col_periods:
                tile = _majority_tile(g, row_period, col_period)
                if _primitive_periods(tile) != (row_period, col_period):
                    continue
                pred = _tile_grid(tile, g.shape)
                mismatch = pred != g
                mismatch_area = int(mismatch.sum())
                if mismatch_area == 0:
                    continue
                if _component_count(mismatch) != 1:
                    continue
                key = (
                    -(row_scores[row_period] + col_scores[col_period]),
                    row_period * col_period,
                    mismatch_area,
                    row_period,
                    col_period,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_mask = mismatch
        return best_key, best_mask

    best_key: tuple[int, int, int, int, int] | None = None
    best_mask: np.ndarray | None = None

    top_row_periods = sorted(row_scores, key=lambda period: (-row_scores[period], period))[:6]
    top_col_periods = sorted(col_scores, key=lambda period: (-col_scores[period], period))[:6]
    best_key, best_mask = consider_pairs(top_row_periods, top_col_periods, best_key, best_mask)

    if best_mask is None:
        all_row_periods = list(range(1, h))
        all_col_periods = list(range(1, w))
        best_key, best_mask = consider_pairs(all_row_periods, all_col_periods, best_key, best_mask)

    assert best_mask is not None, "expected one recoverable connected anomaly"
    return best_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden periodic wallpaper and crop the connected set of cells whose observed colors break that wallpaper.

    Input structure:
    - The full grid is a row/column-periodic tiling of one small non-black tile that uses two to four colors.
    - One much smaller 4-connected region has been recolored cell-by-cell using other colors from that same palette.
    - Outside that disguised region, the visible cells still obey the unique minimal wallpaper period exactly.

    Rule:
    - Infer the row period and column period from the repeated rows and columns that still match across the grid.
    - For each period residue, assign the majority visible color to reconstruct the underlying periodic tile, then repeat that tile over the whole canvas.
    - Mark every cell whose observed color disagrees with the reconstructed wallpaper, and output the tight bbox crop of that mismatch set with only mismatching cells preserved and every other cell black.

    Key insight:
    - The sprite is never drawn as a foreground object; it appears only as the single connected exception set to an otherwise rigid periodic background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    mask = _infer_anomaly_mask(g)
    return _crop_mask(g, mask)


_MAX_ATTEMPTS = 400
_PERIOD_CHOICES = (2, 3, 4, 5)
_PALETTE_SIZE_CHOICES = (2, 3, 4)
_REPEAT_CHOICES = (4, 5, 6)


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return None
    return cropped


def _gen_tile(row_period: int, col_period: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(200):
        tile = np.array(
            [[random.choice(palette) for _ in range(col_period)] for _ in range(row_period)],
            dtype=int,
        )
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _primitive_periods(tile) != (row_period, col_period):
            continue
        row_patterns = {tuple(map(int, row)) for row in tile}
        col_patterns = {tuple(map(int, tile[:, c])) for c in range(col_period)}
        if len(row_patterns) != row_period or len(col_patterns) != col_period:
            continue
        return tile
    return None


def _all_residues_visible(mask: np.ndarray, row_period: int, col_period: int) -> bool:
    rows = np.arange(mask.shape[0])[:, None] % row_period
    cols = np.arange(mask.shape[1])[None, :] % col_period
    for rr in range(row_period):
        for cc in range(col_period):
            visible = ((rows == rr) & (cols == cc) & (~mask)).any()
            if not bool(visible):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_period = random.choice(_PERIOD_CHOICES)
    col_period = random.choice(_PERIOD_CHOICES)
    repeat_rows = random.choice([n for n in _REPEAT_CHOICES if n * row_period <= 30])
    repeat_cols = random.choice([n for n in _REPEAT_CHOICES if n * col_period <= 30])
    height = row_period * repeat_rows
    width = col_period * repeat_cols

    palette_size = random.choice(_PALETTE_SIZE_CHOICES)
    palette = random.sample(range(1, 10), palette_size)
    tile = _gen_tile(row_period, col_period, palette)
    if tile is None:
        return None

    row_offset = random.randrange(row_period)
    col_offset = random.randrange(col_period)
    base = _tile_grid(tile, (height, width), offset=(row_offset, col_offset))

    max_h = min(7, height - 1)
    max_w = min(7, width - 1)
    if max_h < 2 or max_w < 2:
        return None

    shape_h = random.randint(2, max_h)
    shape_w = random.randint(2, max_w)
    max_area = min(shape_h * shape_w, max(4, (height * width) // 8), 12)
    min_area = min(3, max_area)
    if min_area > max_area:
        return None
    area = random.randint(min_area, max_area)
    local_mask = _random_connected_mask(shape_h, shape_w, area)
    if local_mask is None:
        return None

    shape_h, shape_w = local_mask.shape
    top = random.randint(0, height - shape_h)
    left = random.randint(0, width - shape_w)

    anomaly_mask = np.zeros((height, width), dtype=bool)
    anomaly_mask[top : top + shape_h, left : left + shape_w] = local_mask
    if not _all_residues_visible(anomaly_mask, row_period, col_period):
        return None

    grid = base.copy()
    for r, c in np.argwhere(anomaly_mask):
        src = int(grid[r, c])
        choices = [color for color in palette if color != src]
        grid[r, c] = random.choice(choices)

    expected = _crop_mask(grid, anomaly_mask)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_MAX_ATTEMPTS} attempts")
