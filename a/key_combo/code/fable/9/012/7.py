from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _count_background_bands(separator_mask: np.ndarray) -> int:
    bands = 0
    inside = False
    for is_separator in separator_mask:
        if is_separator:
            inside = False
        elif not inside:
            bands += 1
            inside = True
    return bands


def _qualifying_rows(grid: np.ndarray, fg: int) -> np.ndarray:
    rows = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] == fg)
        rows.append(len(cols) > 0 and int(cols[0]) == 0 and int(cols[-1]) == grid.shape[1] - 1)
    return np.array(rows, dtype=bool)


def _qualifying_cols(grid: np.ndarray, fg: int) -> np.ndarray:
    cols = []
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, c] == fg)
        cols.append(len(rows) > 0 and int(rows[0]) == 0 and int(rows[-1]) == grid.shape[0] - 1)
    return np.array(cols, dtype=bool)


def _true_run_count(mask: np.ndarray) -> int:
    runs = 0
    prev = False
    for value in mask:
        if value and not prev:
            runs += 1
        prev = bool(value)
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct dashed separator traces and summarize only the implied partition counts.

    Input structure
    - The grid has one dominant background color and exactly one foreground color.
    - Foreground cells form horizontal and vertical fragment traces on various rows
      and columns; some traces are dashed remnants of separators, others are noise.
    - A row trace qualifies as an implied horizontal separator only when the union
      of that row's foreground cells touches both the left and right borders; a
      column trace qualifies analogously only when it touches both the top and
      bottom borders.

    Rule
    - Identify every qualifying row and column trace and treat each one as a full
      one-cell-thick separator line, regardless of its internal gaps.
    - Ignore all non-qualifying fragment rows and columns.
    - Count the contiguous non-separator row bands and non-separator column bands
      induced by the qualifying lines.
    - Return a uniform background-colored grid whose shape is
      (row_band_count, column_band_count).

    Key insight
    - The visible marks are only incomplete residues: the task is to infer which
      rows and columns really span border-to-border, then keep only the partition
      structure they imply rather than the fragments themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    fg_colors = _colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    row_separators = _qualifying_rows(g, fg)
    col_separators = _qualifying_cols(g, fg)

    out_h = _count_background_bands(row_separators)
    out_w = _count_background_bands(col_separators)
    assert out_h > 0 and out_w > 0, "expected at least one region on each axis"
    return np.full((out_h, out_w), bg, dtype=int)


# --------------- generator (o5x) ---------------

_GEN_RETRIES = 600
_ROW_REGION_VALUES = tuple(range(2, 7))
_COL_REGION_VALUES = tuple(range(2, 7))
_GRID_MIN = 12
_GRID_MAX = 30
_BAND_SIZE_RANGE = (1, 5)
_NOISE_COUNT_RANGE = (2, 5)
_LINE_COLOR_POOL = tuple(range(10))
_USED_OUTPUT_SIGNATURES: set[tuple[int, int, int]] = set()
_GEN_CALL_COUNT = 0


def _positive_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _nonnegative_split(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _render_segment_pattern(length: int, touch_start: bool, touch_end: bool, segment_count: int) -> np.ndarray:
    interior_gap_count = max(0, segment_count - 1)
    part_count = segment_count + interior_gap_count + (0 if touch_start else 1) + (0 if touch_end else 1)
    min_total = part_count
    assert length >= min_total
    extras = _nonnegative_split(length - min_total, part_count)

    idx = 0
    start_gap = 0 if touch_start else 1 + extras[idx]
    if not touch_start:
        idx += 1

    segments = []
    for _ in range(segment_count):
        segments.append(1 + extras[idx])
        idx += 1

    gaps = []
    for _ in range(interior_gap_count):
        gaps.append(1 + extras[idx])
        idx += 1

    end_gap = 0 if touch_end else 1 + extras[idx]

    mask = np.zeros(length, dtype=bool)
    pos = start_gap
    for i, seg_len in enumerate(segments):
        mask[pos : pos + seg_len] = True
        pos += seg_len
        if i < len(gaps):
            pos += gaps[i]
    assert pos + end_gap == length
    return mask


def _sample_qualifying_pattern(length: int, dashed: bool) -> np.ndarray:
    min_segments = 2 if dashed else 1
    max_segments = min(4, max(2, length // 2))
    segment_count = random.randint(min_segments, max_segments)
    if dashed:
        segment_count = max(segment_count, 2)
    return _render_segment_pattern(length, touch_start=True, touch_end=True, segment_count=segment_count)


def _sample_noise_pattern(length: int, axis: str, force_border_touch: bool) -> tuple[np.ndarray, str]:
    assert axis in {"row", "col"}
    modes = ["none", "start", "end"]
    if force_border_touch:
        mode = random.choice(["start", "end"])
    else:
        mode = random.choice(modes)
    touch_start = mode == "start"
    touch_end = mode == "end"

    max_segments = min(3, max(1, length // 2))
    segment_count = random.randint(2 if force_border_touch else 1, max_segments)
    return _render_segment_pattern(length, touch_start, touch_end, segment_count), mode


def _sample_output_signature() -> tuple[int, int, int]:
    choices = [
        (rows, cols, bg)
        for rows in _ROW_REGION_VALUES
        for cols in _COL_REGION_VALUES
        for bg in _LINE_COLOR_POOL
        if (rows, cols, bg) not in _USED_OUTPUT_SIGNATURES
    ]
    if not choices:
        _USED_OUTPUT_SIGNATURES.clear()
        choices = [
            (rows, cols, bg)
            for rows in _ROW_REGION_VALUES
            for cols in _COL_REGION_VALUES
            for bg in _LINE_COLOR_POOL
        ]
    choice = random.choice(choices)
    _USED_OUTPUT_SIGNATURES.add(choice)
    return choice


def _sample_axis_plan(region_count: int, total_min: int, total_max: int, *, force_adjacent: bool, force_border: bool) -> np.ndarray | None:
    for _ in range(120):
        border_side = random.choice(["none", "start", "end"]) if not force_border else random.choice(["start", "end"])
        start_border = border_side == "start"
        end_border = border_side == "end"

        band_total_min = region_count * _BAND_SIZE_RANGE[0]
        band_total_max = region_count * _BAND_SIZE_RANGE[1]
        if band_total_min > total_max:
            return None

        band_total = random.randint(band_total_min, min(band_total_max, total_max))
        bands = _positive_composition(band_total, region_count)

        run_lengths = []
        forced_index = random.randrange(region_count - 1) if force_adjacent and region_count > 1 else None
        for i in range(region_count - 1):
            if forced_index == i:
                run_lengths.append(2)
            else:
                run_lengths.append(random.choice([1, 1, 2]))

        total = sum(bands) + sum(run_lengths) + int(start_border) + int(end_border)
        if total < total_min or total > total_max:
            continue

        mask = []
        if start_border:
            mask.append(True)
        for i, band in enumerate(bands):
            mask.extend([False] * band)
            if i < len(run_lengths):
                mask.extend([True] * run_lengths[i])
        if end_border:
            mask.append(True)
        if _count_background_bands(np.array(mask, dtype=bool)) != region_count:
            continue
        if force_adjacent and not any(run >= 2 for run in run_lengths):
            continue
        return np.array(mask, dtype=bool)
    return None


def _assemble_grid(
    height: int,
    width: int,
    bg: int,
    fg: int,
    sep_rows: np.ndarray,
    sep_cols: np.ndarray,
    row_patterns: dict[int, np.ndarray],
    col_patterns: dict[int, np.ndarray],
    noise_rows: dict[int, np.ndarray],
    noise_cols: dict[int, np.ndarray],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for r, pattern in row_patterns.items():
        grid[r, pattern] = fg
    for c, pattern in col_patterns.items():
        grid[pattern, c] = fg
    for r, pattern in noise_rows.items():
        grid[r, pattern] = fg
    for c, pattern in noise_cols.items():
        grid[pattern, c] = fg

    realized_rows = _qualifying_rows(grid, fg)
    realized_cols = _qualifying_cols(grid, fg)
    if not np.array_equal(realized_rows, sep_rows):
        raise AssertionError("row separator set drifted during rendering")
    if not np.array_equal(realized_cols, sep_cols):
        raise AssertionError("column separator set drifted during rendering")
    return grid


def _semantic_checks(
    grid: np.ndarray,
    bg: int,
    fg: int,
    expected_rows: int,
    expected_cols: int,
    *,
    require_adjacent_rows: bool,
    require_adjacent_cols: bool,
) -> bool:
    realized_rows = _qualifying_rows(grid, fg)
    realized_cols = _qualifying_cols(grid, fg)
    if _count_background_bands(realized_rows) != expected_rows:
        return False
    if _count_background_bands(realized_cols) != expected_cols:
        return False

    separator_rows = np.flatnonzero(realized_rows)
    separator_cols = np.flatnonzero(realized_cols)
    if len(separator_rows) == 0 or len(separator_cols) == 0:
        return False

    dashed_row_seen = any(np.any(grid[r] == bg) and _true_run_count(grid[r] == fg) >= 2 for r in separator_rows)
    dashed_col_seen = any(np.any(grid[:, c] == bg) and _true_run_count(grid[:, c] == fg) >= 2 for c in separator_cols)
    if not dashed_row_seen or not dashed_col_seen:
        return False

    noisy_rows = [r for r in range(grid.shape[0]) if not realized_rows[r] and np.any(grid[r] == fg)]
    noisy_cols = [c for c in range(grid.shape[1]) if not realized_cols[c] and np.any(grid[:, c] == fg)]
    if not noisy_rows or not noisy_cols:
        return False

    confounding_row_seen = any(
        _true_run_count(grid[r] == fg) >= 2 or bool(grid[r, 0] == fg) or bool(grid[r, -1] == fg)
        for r in noisy_rows
    )
    confounding_col_seen = any(
        _true_run_count(grid[:, c] == fg) >= 2 or bool(grid[0, c] == fg) or bool(grid[-1, c] == fg)
        for c in noisy_cols
    )
    if not confounding_row_seen or not confounding_col_seen:
        return False

    if require_adjacent_rows and not any(b - a == 1 for a, b in zip(separator_rows, separator_rows[1:])):
        return False
    if require_adjacent_cols and not any(b - a == 1 for a, b in zip(separator_cols, separator_cols[1:])):
        return False

    expected = np.full((expected_rows, expected_cols), bg, dtype=int)
    return np.array_equal(main(grid), expected)


def _gen_attempt(call_mode: int) -> np.ndarray | None:
    out_rows, out_cols, bg = _sample_output_signature()
    fg_choices = [color for color in _LINE_COLOR_POOL if color != bg]
    fg = random.choice(fg_choices)

    force_adj_rows = call_mode % 4 == 0
    force_adj_cols = call_mode % 4 == 1
    force_border_rows = call_mode % 4 == 2
    force_border_cols = call_mode % 4 == 3

    sep_rows = _sample_axis_plan(
        out_rows,
        _GRID_MIN,
        _GRID_MAX,
        force_adjacent=force_adj_rows,
        force_border=force_border_rows,
    )
    sep_cols = _sample_axis_plan(
        out_cols,
        _GRID_MIN,
        _GRID_MAX,
        force_adjacent=force_adj_cols,
        force_border=force_border_cols,
    )
    if sep_rows is None or sep_cols is None:
        return None

    height = int(len(sep_rows))
    width = int(len(sep_cols))
    row_indices = np.flatnonzero(sep_rows)
    col_indices = np.flatnonzero(sep_cols)

    row_patterns = {int(r): _sample_qualifying_pattern(width, dashed=True) for r in row_indices}
    col_patterns = {int(c): _sample_qualifying_pattern(height, dashed=True) for c in col_indices}

    free_rows = [r for r in range(height) if not sep_rows[r]]
    free_cols = [c for c in range(width) if not sep_cols[c]]
    if len(free_rows) < _NOISE_COUNT_RANGE[0] or len(free_cols) < _NOISE_COUNT_RANGE[0]:
        return None

    noise_rows: dict[int, np.ndarray] = {}
    noise_cols: dict[int, np.ndarray] = {}

    row_noise_count = random.randint(_NOISE_COUNT_RANGE[0], min(_NOISE_COUNT_RANGE[1], len(free_rows)))
    col_noise_count = random.randint(_NOISE_COUNT_RANGE[0], min(_NOISE_COUNT_RANGE[1], len(free_cols)))
    chosen_noise_rows = random.sample(free_rows, row_noise_count)
    chosen_noise_cols = random.sample(free_cols, col_noise_count)

    for i, r in enumerate(chosen_noise_rows):
        pattern, _ = _sample_noise_pattern(width, axis="row", force_border_touch=i == 0)
        noise_rows[int(r)] = pattern
    for i, c in enumerate(chosen_noise_cols):
        pattern, _ = _sample_noise_pattern(height, axis="col", force_border_touch=i == 0)
        noise_cols[int(c)] = pattern

    try:
        grid = _assemble_grid(
            height,
            width,
            bg,
            fg,
            sep_rows,
            sep_cols,
            row_patterns,
            col_patterns,
            noise_rows,
            noise_cols,
        )
    except AssertionError:
        return None

    if not _semantic_checks(
        grid,
        bg,
        fg,
        out_rows,
        out_cols,
        require_adjacent_rows=force_adj_rows,
        require_adjacent_cols=force_adj_cols,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_mode = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(call_mode)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid dashed-separator input")
