from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask.astype(bool).tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _left_shift(seq: list, amount: int) -> list:
    n = len(seq)
    k = int(amount) % n
    return [seq[(i + k) % n] for i in range(n)]


def _next_cycle_value(values: list[int]) -> int:
    assert values, "expected at least one observed shift"
    for period in range(1, len(values) + 1):
        if all(values[i] == values[i % period] for i in range(len(values))):
            return int(values[len(values) % period])
    raise AssertionError("unreachable cycle inference")


def _marker_count(block: np.ndarray, marker: int, bg: int) -> int:
    assert np.all((block == bg) | (block == marker)), "marker area may contain only marker and background"
    coords = np.argwhere(block == marker)
    if coords.size == 0:
        return 0
    rows = set(map(int, coords[:, 0]))
    cols = sorted(map(int, coords[:, 1]))
    assert len(rows) == 1, "each row marker must be one horizontal run"
    assert cols == list(range(len(cols))), "marker run must start at the left edge"
    return int(len(cols))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the next sprite row using marker-count-driven cyclic shifts and fixed column colors.

    Input structure:
    - The grid contains a 3 by 3 lattice of equal-size sprite cells separated by full background rows and columns.
    - A short marker run at the far left of each non-first sprite row counts how many sprite columns that row is
      shifted left relative to the previous row; the first sprite row has no marker run.
    - Ignoring color, each sprite-row mask sequence is the previous row's sequence cyclically shifted by its marker
      count, and each sprite column uses one fixed foreground color across all rows.

    Rule:
    - Read the two visible marker counts as the beginning of a short cyclic count sequence.
    - Take the next count in that sequence, cyclically left-shift the last sprite-row mask sequence by that count,
      and render the resulting row with the fixed color of each destination sprite column.
    - Return only this new horizontal sprite band, with no marker area and no separator columns.

    Key insight:
    - The shape motion is controlled by the row-start marker counts, and those counts themselves cycle; the masks
      move independently of color because color is determined by the destination column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    row_runs = _runs(np.any(g != bg, axis=1))
    assert len(row_runs) == 3, "expected exactly three sprite row bands"
    row_heights = [r1 - r0 + 1 for r0, r1 in row_runs]
    assert len(set(row_heights)) == 1, "sprite row bands must have equal height"
    sprite_h = row_heights[0]

    all_bg_cols = np.all(g == bg, axis=0)
    assert not bool(all_bg_cols[0]), "marker area must begin at the far left"
    marker_stop = next((c for c, is_bg in enumerate(all_bg_cols) if bool(is_bg)), None)
    assert marker_stop is not None and marker_stop > 0, "expected a background gap after marker area"

    marker_colors = set(map(int, np.unique(g[:, :marker_stop]))) - {bg}
    assert len(marker_colors) == 1, "expected one marker color in the far-left marker area"
    marker = marker_colors.pop()

    sprite_fg = (g != bg) & (g != marker)
    col_runs = [run for run in _runs(np.any(sprite_fg, axis=0)) if run[0] > marker_stop]
    assert len(col_runs) == 3, "expected exactly three sprite column bands"
    col_widths = [c1 - c0 + 1 for c0, c1 in col_runs]
    assert len(set(col_widths)) == 1, "sprite column bands must have equal width"
    sprite_w = col_widths[0]

    marker_counts = [
        _marker_count(g[r0 : r1 + 1, :marker_stop], marker, bg)
        for r0, r1 in row_runs
    ]
    assert marker_counts[0] == 0, "first sprite row should not have a marker count"
    shifts = marker_counts[1:]
    assert all(0 < shift <= len(col_runs) for shift in shifts), "marker counts must be valid cyclic shifts"

    column_colors: list[int] = []
    row_masks: list[list[np.ndarray]] = []
    for c0, c1 in col_runs:
        colors: set[int] = set()
        for r0, r1 in row_runs:
            colors.update(map(int, np.unique(g[r0 : r1 + 1, c0 : c1 + 1])))
        colors.discard(bg)
        assert len(colors) == 1, "each sprite column must have one fixed foreground color"
        column_colors.append(colors.pop())

    for r0, r1 in row_runs:
        masks = []
        for c0, c1 in col_runs:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            cell_colors = set(map(int, np.unique(cell))) - {bg}
            assert len(cell_colors) == 1, "each sprite cell must use its column color"
            masks.append(cell != bg)
        row_masks.append(masks)

    for row_i, shift in enumerate(shifts, start=1):
        expected = _left_shift(row_masks[row_i - 1], shift)
        assert all(
            np.array_equal(row_masks[row_i][col_i], expected[col_i])
            for col_i in range(len(col_runs))
        ), "sprite row masks must follow their marker-count shifts"

    next_shift = _next_cycle_value(shifts)
    next_masks = _left_shift(row_masks[-1], next_shift)

    out = np.full((sprite_h, len(col_runs) * sprite_w), bg, dtype=int)
    for col_i, mask in enumerate(next_masks):
        c0 = col_i * sprite_w
        cell = out[:, c0 : c0 + sprite_w]
        cell[mask] = column_colors[col_i]
    return out


# --------------- generator (o5x) ---------------

ARC_COLORS = list(range(10))
SPRITE_COUNT = 3
SPRITE_H_RANGE = (3, 5)
SPRITE_W_RANGE = (3, 5)
ROW_SEP_CHOICES = (1, 2)
COL_SEP_CHOICES = (1, 2)
MARKER_GAP_CHOICES = (1, 2)
TOP_PAD_CHOICES = (0, 1)
BOTTOM_PAD_CHOICES = (0, 1)
RIGHT_PAD_RANGE = (0, 2)
GENERATION_ATTEMPTS = 300


def _gen_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    lo = max(height, width, int(np.ceil(area * 0.35)))
    hi = min(area - 2, int(np.floor(area * 0.70)))
    if lo > hi:
        return None

    for _ in range(200):
        count = random.randint(lo, hi)
        flat = np.zeros(area, dtype=bool)
        flat[random.sample(range(area), count)] = True
        mask = flat.reshape((height, width))
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        return mask
    return None


def _gen_masks(height: int, width: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(SPRITE_COUNT):
        for _attempt in range(300):
            mask = _gen_mask(height, width)
            if mask is None:
                continue
            sig = mask.tobytes()
            if sig in seen:
                continue
            seen.add(sig)
            masks.append(mask)
            break
        else:
            return None
    return masks


def _render_compact_row(
    bg: int,
    masks: list[np.ndarray],
    column_colors: list[int],
) -> np.ndarray:
    height, width = masks[0].shape
    out = np.full((height, len(masks) * width), bg, dtype=int)
    for col_i, mask in enumerate(masks):
        c0 = col_i * width
        cell = out[:, c0 : c0 + width]
        cell[mask] = column_colors[col_i]
    return out


def _render_input(
    bg: int,
    marker: int,
    column_colors: list[int],
    rows: list[list[np.ndarray]],
    shifts: list[int],
    row_sep: int,
    col_sep: int,
    marker_gap: int,
    top_pad: int,
    bottom_pad: int,
    right_pad: int,
) -> np.ndarray:
    sprite_h, sprite_w = rows[0][0].shape
    marker_w = max(shifts)
    sprite_left = marker_w + marker_gap
    height = top_pad + SPRITE_COUNT * sprite_h + (SPRITE_COUNT - 1) * row_sep + bottom_pad
    width = sprite_left + SPRITE_COUNT * sprite_w + (SPRITE_COUNT - 1) * col_sep + right_pad
    grid = np.full((height, width), bg, dtype=int)

    for row_i, masks in enumerate(rows):
        r0 = top_pad + row_i * (sprite_h + row_sep)
        if row_i > 0:
            count = shifts[row_i - 1]
            grid[r0 + sprite_h // 2, :count] = marker
        for col_i, mask in enumerate(masks):
            c0 = sprite_left + col_i * (sprite_w + col_sep)
            cell = grid[r0 : r0 + sprite_h, c0 : c0 + sprite_w]
            cell[mask] = column_colors[col_i]
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*SPRITE_H_RANGE)
    width = random.randint(*SPRITE_W_RANGE)
    base = _gen_masks(height, width)
    if base is None:
        return None

    first_shift = random.choice([1, 2])
    shifts = [first_shift, SPRITE_COUNT]
    row0 = base
    row1 = _left_shift(row0, shifts[0])
    row2 = _left_shift(row1, shifts[1])
    rows = [row0, row1, row2]
    next_masks = _left_shift(row2, _next_cycle_value(shifts))

    bg, marker, *column_colors = random.sample(ARC_COLORS, 5)
    expected = _render_compact_row(bg, next_masks, column_colors)

    compact_rows = [_render_compact_row(bg, row, column_colors) for row in rows]
    if any(np.array_equal(expected, old) for old in compact_rows):
        return None

    grid = _render_input(
        bg=bg,
        marker=marker,
        column_colors=column_colors,
        rows=rows,
        shifts=shifts,
        row_sep=random.choice(ROW_SEP_CHOICES),
        col_sep=random.choice(COL_SEP_CHOICES),
        marker_gap=random.choice(MARKER_GAP_CHOICES),
        top_pad=random.choice(TOP_PAD_CHOICES),
        bottom_pad=random.choice(BOTTOM_PAD_CHOICES),
        right_pad=random.randint(*RIGHT_PAD_RANGE),
    )
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fable/3/049/8 input")
