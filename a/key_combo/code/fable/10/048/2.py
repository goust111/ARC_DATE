from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(1, 10))
UNKNOWN = -1
MAX_RETRIES = 500


def _divider_intervals(dividers: list[int], limit: int) -> list[tuple[int, int]]:
    starts = [0] + [idx + 1 for idx in dividers]
    ends = [idx - 1 for idx in dividers] + [limit - 1]
    intervals = [(start, end) for start, end in zip(starts, ends) if start <= end]
    assert intervals, "expected at least one non-divider interval"
    return intervals


def _partition_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_dividers = [r for r in range(g.shape[0]) if np.all(g[r, :] == BLACK)]
    col_dividers = [c for c in range(g.shape[1]) if np.all(g[:, c] == BLACK)]
    row_spans = _divider_intervals(row_dividers, g.shape[0])
    col_spans = _divider_intervals(col_dividers, g.shape[1])
    return row_spans, col_spans


def _unique_majority_nonblack(region: np.ndarray) -> int | None:
    values = [int(v) for v in region.ravel() if int(v) != BLACK]
    if not values:
        return None
    counts = Counter(values).most_common()
    assert counts[0][1] > counts[1][1] if len(counts) > 1 else True, "expected a unique dominant color"
    return int(counts[0][0])


def _downsample_partial(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans, col_spans = _partition_spans(grid)
    coarse = np.full((len(row_spans), len(col_spans)), UNKNOWN, dtype=int)
    for r_idx, (r0, r1) in enumerate(row_spans):
        for c_idx, (c0, c1) in enumerate(col_spans):
            color = _unique_majority_nonblack(grid[r0 : r1 + 1, c0 : c1 + 1])
            if color is not None:
                coarse[r_idx, c_idx] = color
    return coarse, row_spans, col_spans


def _diag_cells(rows: int, cols: int, diag: int) -> list[tuple[int, int]]:
    return [(r, diag - r) for r in range(rows) if 0 <= diag - r < cols]


def _diag_colors_from_partial(coarse: np.ndarray) -> tuple[list[int | None], int]:
    rows, cols = coarse.shape
    diag_colors: list[int | None] = []
    missing = []
    for diag in range(rows + cols - 1):
        values = {int(coarse[r, c]) for r, c in _diag_cells(rows, cols, diag) if int(coarse[r, c]) != UNKNOWN}
        if values:
            assert len(values) == 1, f"conflicting visible colors on anti-diagonal {diag}: {values}"
            diag_colors.append(next(iter(values)))
        else:
            diag_colors.append(None)
            missing.append(diag)
    assert len(missing) == 1, "expected exactly one fully empty anti-diagonal"
    return diag_colors, int(missing[0])


def _period_predictions(diag_colors: list[int | None]) -> dict[int, int]:
    missing_idx = diag_colors.index(None)
    n = len(diag_colors)
    predictions: dict[int, int] = {}
    for period in range(1, n):
        residue_to_color: dict[int, int] = {}
        ok = True
        for idx, color in enumerate(diag_colors):
            if color is None:
                continue
            residue = idx % period
            prev = residue_to_color.get(residue)
            if prev is None:
                residue_to_color[residue] = int(color)
            elif prev != int(color):
                ok = False
                break
        if ok and (missing_idx % period) in residue_to_color:
            predictions[period] = residue_to_color[missing_idx % period]
    return predictions


def _infer_missing_diagonal_color(diag_colors: list[int | None]) -> int:
    predictions = _period_predictions(diag_colors)
    colors = {int(color) for color in predictions.values()}
    assert len(colors) == 1, f"periodic continuation is ambiguous: {predictions}"
    return next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downscale a black-line partition grid, then recover all missing coarse cells from anti-diagonal equality and a periodic diagonal sequence.

    Input structure:
    - Full black divider rows and columns split the grid into an R x C array of rectangular regions.
    - Every non-empty region contains scattered pixels of one dominant non-black color plus a few pixels of one secondary noise color.
    - Some regions are completely black, and the dominant-color coarse matrix is constant on each anti-diagonal.

    Rule:
    - Collapse each region to one coarse cell by taking the unique majority among its non-black pixels; fully black regions stay unknown.
    - If an unknown coarse cell lies on an anti-diagonal that already has visible cells, fill it with that anti-diagonal's unique visible color.
    - Exactly one whole anti-diagonal is fully unknown; recover its color by extending the periodic color sequence over anti-diagonal indices.
    - Return the completed R x C coarse grid.

    Key insight:
    - The real pattern lives at the partition level, not the pixel level: ignore local dot counts, read one dominant color per region, then use anti-diagonal constancy plus periodic order across diagonals.
    """
    coarse, _row_spans, _col_spans = _downsample_partial(np.asarray(grid, dtype=int))
    diag_colors, missing_diag = _diag_colors_from_partial(coarse)
    diag_colors[missing_diag] = _infer_missing_diagonal_color(diag_colors)

    out = coarse.copy()
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if int(out[r, c]) == UNKNOWN:
                out[r, c] = int(diag_colors[r + c])

    assert np.all(out != UNKNOWN), "completion left unknown coarse cells"
    return out


def _diag_len(rows: int, cols: int, diag: int) -> int:
    return len(_diag_cells(rows, cols, diag))


def _make_periodic_output(rows: int, cols: int) -> tuple[np.ndarray, list[int], int]:
    n_diag = rows + cols - 1
    max_period = min(4, n_diag - 2)
    assert max_period >= 2
    period = random.randint(2, max_period)
    palette = random.sample(list(ARC_COLORS), period)
    seq = [palette[idx % period] for idx in range(n_diag)]
    output = np.array([[seq[r + c] for c in range(cols)] for r in range(rows)], dtype=int)

    candidates = [
        diag
        for diag in range(1, n_diag - 1)
        if _diag_len(rows, cols, diag) >= 2 and (diag - period >= 0 or diag + period < n_diag)
    ]
    assert candidates, "failed to find a non-trivial missing anti-diagonal"
    missing_diag = random.choice(candidates)
    return output, seq, missing_diag


def _sample_visibility(rows: int, cols: int, missing_diag: int) -> tuple[np.ndarray, bool]:
    visible = np.ones((rows, cols), dtype=bool)
    for r, c in _diag_cells(rows, cols, missing_diag):
        visible[r, c] = False

    candidate_diags = [diag for diag in range(rows + cols - 1) if diag != missing_diag and _diag_len(rows, cols, diag) >= 2]
    random.shuffle(candidate_diags)
    partial_count = random.randint(1, min(3, len(candidate_diags)))
    used_partial = False
    for diag in candidate_diags[:partial_count]:
        cells = _diag_cells(rows, cols, diag)
        random.shuffle(cells)
        hole_count = random.randint(1, len(cells) - 1)
        for r, c in cells[:hole_count]:
            visible[r, c] = False
        used_partial = True

    if not np.all(np.any(visible, axis=0)) or not np.all(np.any(visible, axis=1)):
        raise ValueError("visibility erased an entire coarse row or column")
    return visible, used_partial


def _sample_spans(count: int) -> list[int]:
    return [random.randint(3, 5) for _ in range(count)]


def _spans_to_bounds(spans: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bounds: list[tuple[int, int]] = []
    dividers: list[int] = []
    cursor = 0
    for idx, size in enumerate(spans):
        bounds.append((cursor, cursor + size - 1))
        cursor += size
        if idx != len(spans) - 1:
            dividers.append(cursor)
            cursor += 1
    return bounds, dividers


def _paint_region(region: np.ndarray, main_color: int) -> int:
    height, width = region.shape
    area = height * width
    main_count = random.randint(max(4, area // 3), max(5, area // 2))
    noise_limit = min(3, main_count - 1, area - main_count)
    assert noise_limit >= 1
    noise_count = random.randint(1, noise_limit)
    noise_color = random.choice([color for color in ARC_COLORS if color != main_color])
    coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(coords)
    for r, c in coords[:main_count]:
        region[r, c] = main_color
    for r, c in coords[main_count : main_count + noise_count]:
        region[r, c] = noise_color
    return noise_color


def _ensure_nondivider_coverage(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    visible: np.ndarray,
    output: np.ndarray,
) -> None:
    regions = []
    for r_idx, (r0, r1) in enumerate(row_spans):
        for c_idx, (c0, c1) in enumerate(col_spans):
            if visible[r_idx, c_idx]:
                regions.append((r0, r1, c0, c1, int(output[r_idx, c_idx])))

    for r0, r1 in row_spans:
        for row in range(r0, r1 + 1):
            if np.any(grid[row, :] != BLACK):
                continue
            candidates = [item for item in regions if item[0] <= row <= item[1]]
            if not candidates:
                raise ValueError("a whole canvas row stayed empty without any visible region")
            _rr0, _rr1, c0, c1, color = random.choice(candidates)
            col = random.randint(c0, c1)
            grid[row, col] = color

    for c0, c1 in col_spans:
        for col in range(c0, c1 + 1):
            if np.any(grid[:, col] != BLACK):
                continue
            candidates = [item for item in regions if item[2] <= col <= item[3]]
            if not candidates:
                raise ValueError("a whole canvas column stayed empty without any visible region")
            r0, r1, _cc0, _cc1, color = random.choice(candidates)
            row = random.randint(r0, r1)
            grid[row, col] = color


def _build_input(output: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    rows, cols = output.shape
    row_sizes = _sample_spans(rows)
    col_sizes = _sample_spans(cols)
    row_spans, row_dividers = _spans_to_bounds(row_sizes)
    col_spans, col_dividers = _spans_to_bounds(col_sizes)
    height = sum(row_sizes) + len(row_dividers)
    width = sum(col_sizes) + len(col_dividers)
    assert height <= 30 and width <= 30

    grid = np.zeros((height, width), dtype=int)
    for divider in row_dividers:
        grid[divider, :] = BLACK
    for divider in col_dividers:
        grid[:, divider] = BLACK

    for r_idx, (r0, r1) in enumerate(row_spans):
        for c_idx, (c0, c1) in enumerate(col_spans):
            if not visible[r_idx, c_idx]:
                continue
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            _paint_region(region, int(output[r_idx, c_idx]))

    _ensure_nondivider_coverage(grid, row_spans, col_spans, visible, output)
    return grid, row_spans, col_spans


def _expected_partial(output: np.ndarray, visible: np.ndarray) -> np.ndarray:
    partial = np.full_like(output, UNKNOWN)
    partial[visible] = output[visible]
    return partial


def _validate_instance(
    grid: np.ndarray,
    output: np.ndarray,
    visible: np.ndarray,
    seq: list[int],
    missing_diag: int,
    expected_row_spans: list[tuple[int, int]],
    expected_col_spans: list[tuple[int, int]],
) -> bool:
    if grid.ndim != 2 or output.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(output.shape) > 30:
        return False
    if len(set(seq)) < 2:
        return False
    if not any(
        0 < sum(not bool(visible[r, c]) for r, c in _diag_cells(*output.shape, diag)) < _diag_len(*output.shape, diag)
        for diag in range(output.shape[0] + output.shape[1] - 1)
        if diag != missing_diag
    ):
        return False

    try:
        partial, row_spans, col_spans = _downsample_partial(grid)
    except AssertionError:
        return False
    if partial.shape != output.shape:
        return False
    if row_spans != expected_row_spans:
        return False
    if col_spans != expected_col_spans:
        return False
    if not np.array_equal(partial, _expected_partial(output, visible)):
        return False

    try:
        diag_colors, inferred_missing = _diag_colors_from_partial(partial)
        if inferred_missing != missing_diag:
            return False
        if _infer_missing_diagonal_color(diag_colors) != int(seq[missing_diag]):
            return False
        solved = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, output))


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    if rows + cols - 1 < 6:
        return None

    output, seq, missing_diag = _make_periodic_output(rows, cols)
    try:
        visible, used_partial = _sample_visibility(rows, cols, missing_diag)
    except ValueError:
        return None
    if not used_partial:
        return None

    try:
        grid, row_spans, col_spans = _build_input(output, visible)
    except (AssertionError, ValueError):
        return None
    if not _validate_instance(grid, output, visible, seq, missing_diag, row_spans, col_spans):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partition-downsampling puzzle")
