from __future__ import annotations

import math

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
FG_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_TILE_HEIGHT = 2
MAX_TILE_HEIGHT = 5
MIN_SIDE_WIDTH = 4
MAX_SIDE_WIDTH = 10
MIN_REPEAT_COUNT = 3
MAX_REPEAT_COUNT = 6
MIN_ROW_FG = 3
MIN_MISSING_RATIO = 0.30
MAX_ATTEMPTS = 800

_SEEN_INPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return (int(arr.shape[0]), int(arr.shape[1])), arr.tobytes()


def _separator_col(grid: np.ndarray, bg: int = BG) -> int:
    h, w = map(int, grid.shape)
    candidates = []
    for c in range(w):
        color = int(grid[0, c])
        if color == bg or not np.all(grid[:, c] == color):
            continue
        if 2 * c + 1 != w:
            continue
        if np.any(grid[:, :c] == color) or np.any(grid[:, c + 1 :] == color):
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return int(candidates[0])


def _fold_overlay(grid: np.ndarray, bg: int = BG) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    sep = _separator_col(grid, bg)
    left = grid[:, :sep].copy()
    right = np.fliplr(grid[:, sep + 1 :]).copy()
    assert left.shape == right.shape, "separator must split equal-width halves"
    overlap = (left != bg) & (right != bg)
    assert np.all(left[overlap] == right[overlap]), "fold overlaps must agree in color"
    merged = left.copy()
    right_mask = right != bg
    merged[right_mask] = right[right_mask]
    return sep, left, right, merged


def _band_bounds(merged: np.ndarray, bg: int = BG) -> tuple[int, int]:
    rows = np.flatnonzero(np.any(merged != bg, axis=1))
    assert rows.size > 0, "folded image must contain a foreground band"
    r0 = int(rows[0])
    r1 = int(rows[-1])
    assert np.array_equal(rows, np.arange(r0, r1 + 1)), "foreground rows must form one contiguous band"
    assert np.all(merged[:r0] == bg) and np.all(merged[r1 + 1 :] == bg), "rows outside the band must be background"
    return r0, r1


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the complementary side fragments into a full row band, then repeat that recovered band vertically across the canvas.

    Input structure:
    - A black background grid contains one full-height non-black separator column that splits the canvas into equal-width left and right regions.
    - Outside one contiguous horizontal band, both side regions are entirely background.
    - Inside that band, the left region and the mirrored right region hold incomplete same-color fragments of one hidden multicolor tile.

    Rule:
    - Identify the unique separator column, remove it conceptually, and mirror the right region horizontally into the left-region coordinate system.
    - Transparently overlay the mirrored right region onto the left region, keeping whichever side provides a non-background pixel and requiring overlap colors to match.
    - The resulting merged foreground rows form the complete source band.
    - Repeat that full band upward and downward with the original band start row as the phase anchor until every output row is filled.

    Key insight:
    - The input does not directly show the tiling unit: the solver must first recover it by folding the two complementary side fragments together, and only then can the vertical repetition be determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    _sep, _left, _right, merged = _fold_overlay(g, BG)
    r0, r1 = _band_bounds(merged, BG)
    tile = merged[r0 : r1 + 1, :].copy()
    tile_h = int(tile.shape[0])
    row_indices = (np.arange(g.shape[0]) - r0) % tile_h
    return tile[row_indices].copy()


def _minimal_vertical_period(tile: np.ndarray) -> int:
    h = int(tile.shape[0])
    for period in range(1, h):
        if h % period != 0:
            continue
        if np.array_equal(tile, tile[np.arange(h) % period]):
            return period
    return h


def _rand_partition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = np.sort(np.random.choice(np.arange(1, total), size=parts - 1, replace=False))
    values = np.diff(np.concatenate(([0], cuts, [total])))
    return [int(v) for v in values]


def _build_periodic_row(width: int, palette: tuple[int, ...]) -> np.ndarray:
    max_fg = max(MIN_ROW_FG, min(width - 1, math.floor(0.78 * width)))
    min_fg = min(max_fg, max(MIN_ROW_FG, math.ceil(0.35 * width)))
    assert min_fg <= max_fg

    for _ in range(120):
        period = int(np.random.randint(2, min(6, width) + 1))
        motif = np.zeros(period, dtype=int)
        active = int(np.random.randint(max(1, math.ceil(0.35 * period)), period))
        indices = np.random.choice(period, size=active, replace=False)
        motif[indices] = np.random.choice(np.asarray(palette, dtype=int), size=active, replace=True)
        phase = int(np.random.randint(0, period))
        row = motif[(np.arange(width) + phase) % period].copy()
        fg = int(np.count_nonzero(row))
        if min_fg <= fg <= max_fg:
            return row
    raise RuntimeError("failed to build a periodic row")


def _build_scattered_row(width: int, palette: tuple[int, ...]) -> np.ndarray:
    max_fg = max(MIN_ROW_FG, min(width - 1, math.floor(0.78 * width)))
    min_fg = min(max_fg, max(MIN_ROW_FG, math.ceil(0.35 * width)))
    target = int(np.random.randint(min_fg, max_fg + 1))

    positions = {int(np.random.randint(0, width))}
    while len(positions) < target:
        if np.random.random() < 0.7:
            anchor = int(np.random.choice(tuple(positions)))
            step = int(np.random.choice((-2, -1, 1, 2)))
            candidate = anchor + step
            if 0 <= candidate < width:
                positions.add(candidate)
                continue
        positions.add(int(np.random.randint(0, width)))

    row = np.zeros(width, dtype=int)
    ordered = sorted(positions)
    color_count = min(len(palette), max(2, int(np.random.randint(2, min(5, len(ordered) + 1)))))
    colors = list(np.random.choice(np.asarray(palette, dtype=int), size=color_count, replace=False))
    np.random.shuffle(colors)

    if len(ordered) >= len(colors):
        run_sizes = _rand_partition(len(ordered), len(colors))
    else:
        run_sizes = [1] * len(ordered)
        colors = colors[: len(ordered)]

    idx = 0
    for run_size, color in zip(run_sizes, colors):
        for _ in range(run_size):
            row[ordered[idx]] = int(color)
            idx += 1
    return row


def _build_tile(tile_h: int, side_w: int, palette: tuple[int, ...]) -> np.ndarray | None:
    for _ in range(300):
        tile = np.zeros((tile_h, side_w), dtype=int)
        mode = int(np.random.randint(0, 3))
        for r in range(tile_h):
            if mode == 0:
                row = _build_periodic_row(side_w, palette)
            elif mode == 1:
                row = _build_scattered_row(side_w, palette)
            else:
                row = _build_periodic_row(side_w, palette) if r % 2 == 0 else _build_scattered_row(side_w, palette)
            tile[r] = row

        fg_mask = tile != BG
        fg_total = int(np.count_nonzero(fg_mask))
        if fg_total < tile_h * MIN_ROW_FG:
            continue
        if fg_total > int(0.78 * tile_h * side_w):
            continue
        if any(int(np.count_nonzero(tile[r])) < MIN_ROW_FG for r in range(tile_h)):
            continue
        if len({int(v) for v in tile[fg_mask]}) < 2:
            coords = np.argwhere(fg_mask)
            if coords.size == 0:
                continue
            rr, cc = map(int, coords[int(np.random.randint(len(coords)))])
            alt = [color for color in palette if color != int(tile[rr, cc])]
            if not alt:
                continue
            tile[rr, cc] = int(np.random.choice(alt))
        if _minimal_vertical_period(tile) != tile_h:
            continue
        return tile
    return None


def _split_tile(tile: np.ndarray, allow_overlap: bool) -> tuple[np.ndarray, np.ndarray] | None:
    fg_mask = tile != BG
    fg_total = int(np.count_nonzero(fg_mask))
    required_missing = max(1, math.ceil(MIN_MISSING_RATIO * fg_total))

    for _ in range(240):
        left = np.zeros_like(tile)
        right = np.zeros_like(tile)
        overlap_count = 0

        for r in range(tile.shape[0]):
            cols = np.flatnonzero(tile[r] != BG)
            if cols.size < MIN_ROW_FG:
                return None
            cols = cols.copy()
            np.random.shuffle(cols)

            left[r, cols[0]] = tile[r, cols[0]]
            right[r, cols[1]] = tile[r, cols[1]]
            for c in cols[2:]:
                color = int(tile[r, c])
                pick = float(np.random.random())
                if allow_overlap and pick < 0.22:
                    left[r, c] = color
                    right[r, c] = color
                    overlap_count += 1
                elif pick < 0.61:
                    left[r, c] = color
                else:
                    right[r, c] = color

        left_present = int(np.count_nonzero(left))
        right_present = int(np.count_nonzero(right))
        if fg_total - left_present < required_missing or fg_total - right_present < required_missing:
            continue
        if allow_overlap and overlap_count == 0:
            continue
        if not allow_overlap and overlap_count != 0:
            continue

        row_ok = True
        for r in range(tile.shape[0]):
            row_fg = tile[r] != BG
            if not np.any(left[r, row_fg] != BG) or not np.any(right[r, row_fg] != BG):
                row_ok = False
                break
            if not np.any(left[r, row_fg] == BG) or not np.any(right[r, row_fg] == BG):
                row_ok = False
                break
        if not row_ok:
            continue
        return left, right
    return None


def _repeat_tile(tile: np.ndarray, height: int, band_start: int) -> np.ndarray:
    row_indices = (np.arange(height) - band_start) % tile.shape[0]
    return tile[row_indices].copy()


def _semantic_ok(
    inp: np.ndarray,
    tile: np.ndarray,
    expected: np.ndarray,
    band_start: int,
    left_piece: np.ndarray,
    right_piece: np.ndarray,
    allow_overlap: bool,
) -> bool:
    try:
        sep, left, right, merged = _fold_overlay(inp, BG)
        r0, r1 = _band_bounds(merged, BG)
    except AssertionError:
        return False

    if sep != tile.shape[1]:
        return False
    if (r0, r1) != (band_start, band_start + tile.shape[0] - 1):
        return False
    if not np.array_equal(left[r0 : r1 + 1], left_piece):
        return False
    if not np.array_equal(right[r0 : r1 + 1], right_piece):
        return False
    if not np.array_equal(merged[r0 : r1 + 1], tile):
        return False
    if not np.all(merged[:r0] == BG) or not np.all(merged[r1 + 1 :] == BG):
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(solved, merged):
        return False
    if not np.any(solved[:r0] != BG) or not np.any(solved[r1 + 1 :] != BG):
        return False

    fg_total = int(np.count_nonzero(tile))
    required_missing = max(1, math.ceil(MIN_MISSING_RATIO * fg_total))
    if fg_total - int(np.count_nonzero(left_piece)) < required_missing:
        return False
    if fg_total - int(np.count_nonzero(right_piece)) < required_missing:
        return False

    overlap = (left_piece != BG) & (right_piece != BG)
    if allow_overlap != bool(np.any(overlap)):
        return False
    if np.any(overlap) and not np.array_equal(left_piece[overlap], right_piece[overlap]):
        return False

    for r in range(tile.shape[0]):
        row_fg = tile[r] != BG
        if not np.any((left_piece[r] == BG) & row_fg):
            return False
        if not np.any((right_piece[r] == BG) & row_fg):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    tile_h = int(np.random.randint(MIN_TILE_HEIGHT, MAX_TILE_HEIGHT + 1))
    max_repeats = min(MAX_REPEAT_COUNT, 30 // tile_h)
    if max_repeats < MIN_REPEAT_COUNT:
        return None
    repeat_count = int(np.random.randint(MIN_REPEAT_COUNT, max_repeats + 1))
    height = tile_h * repeat_count

    side_w = int(np.random.randint(MIN_SIDE_WIDTH, min(MAX_SIDE_WIDTH, 14) + 1))
    if 2 * side_w + 1 > 30:
        return None

    max_band_start = min(tile_h - 1, height - tile_h - 1)
    if max_band_start < 1:
        return None
    band_start = int(np.random.randint(1, max_band_start + 1))

    palette = tuple(int(v) for v in np.random.choice(np.asarray(FG_COLORS, dtype=int), size=int(np.random.randint(2, 5)), replace=False))
    tile = _build_tile(tile_h, side_w, palette)
    if tile is None:
        return None

    allow_overlap = bool(np.random.random() < 0.45)
    split = _split_tile(tile, allow_overlap)
    if split is None:
        return None
    left_piece, right_piece = split

    inp = np.full((height, 2 * side_w + 1), BG, dtype=int)
    inp[:, side_w] = SEPARATOR_COLOR
    inp[band_start : band_start + tile_h, :side_w] = left_piece
    inp[band_start : band_start + tile_h, side_w + 1 :] = np.fliplr(right_piece)

    expected = _repeat_tile(tile, height, band_start)
    if not _semantic_ok(inp, tile, expected, band_start, left_piece, right_piece, allow_overlap):
        return None

    inp_sig = _signature(inp)
    out_sig = _signature(expected)
    if inp_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_INPUT_SIGS.add(inp_sig)
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-and-tile ARC input")
