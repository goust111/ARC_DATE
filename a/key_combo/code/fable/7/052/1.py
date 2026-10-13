from __future__ import annotations

import random

import numpy as np


BG = 0
RULER = 5
CONTENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _prefix_run(vec: np.ndarray, color: int) -> int:
    k = 0
    for value in vec:
        if int(value) != int(color):
            break
        k += 1
    return k


def _extract_instance(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    content_mask = (g != bg) & (g != RULER)
    content_box = _bbox(content_mask)
    assert content_box is not None, "expected one non-grey content rectangle"
    r0, c0, r1, c1 = content_box

    content = _crop_bbox(g, content_box)
    assert np.all((content != bg) & (content != RULER)), "content bbox must be fully occupied"
    assert r0 > 0 and c0 > 0, "content must sit below top bars and right of left bars"

    stray_mask = np.zeros_like(content_mask, dtype=bool)
    stray_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert not np.any(content_mask & ~stray_mask), "non-grey foreground must belong to the content rectangle"

    row_scales = [_prefix_run(g[r0 + rr, :c0], RULER) for rr in range(content.shape[0])]
    col_scales = [_prefix_run(g[:r0, c0 + cc], RULER) for cc in range(content.shape[1])]

    for rr, scale in enumerate(row_scales):
        left = g[r0 + rr, :c0]
        assert np.all(left[:scale] == RULER), "row ruler must be a contiguous prefix"
        assert np.all(left[scale:] == bg), "row ruler tail must be background"

    for cc, scale in enumerate(col_scales):
        top = g[:r0, c0 + cc]
        assert np.all(top[:scale] == RULER), "column ruler must be a contiguous prefix"
        assert np.all(top[scale:] == bg), "column ruler tail must be background"

    if c0 > 0:
        assert np.all(g[:r0, :c0] == bg), "top-left corner must be empty"
    return content, row_scales, col_scales, content_box


def _stretch(content: np.ndarray, row_scales: list[int], col_scales: list[int]) -> np.ndarray:
    keep_rows = [idx for idx, scale in enumerate(row_scales) for _ in range(scale)]
    keep_cols = [idx for idx, scale in enumerate(col_scales) for _ in range(scale)]
    assert keep_rows and keep_cols, "at least one row and one column must survive"
    return np.asarray(content[np.ix_(keep_rows, keep_cols)], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the content rectangle by row and column factors encoded as grey histogram bars.

    Input structure:
    - The upper strip contains grey vertical bars descending from the top edge, aligned with content columns.
    - The left strip contains grey horizontal bars extending from the left edge, aligned with content rows.
    - The bottom-right region is one fully occupied multicolor content rectangle; the remaining cells are background.

    Rule:
    - Crop the content rectangle from the non-grey, non-background foreground.
    - For each content row, copy that row vertically as many times as the grey bar length on its left; if the bar is missing, delete the row.
    - For each content column, copy that column horizontally as many times as the grey bar height above it; if the bar is missing, delete the column.
    - Output only the stretched content, without the grey rulers or background margin.

    Key insight:
    - The grey edge bars form two independent histograms: left bars control per-row repetition, and top bars control per-column repetition, with missing bars meaning scale zero.
    """
    content, row_scales, col_scales, _ = _extract_instance(grid)
    return _stretch(content, row_scales, col_scales)


def _random_scales(length: int, limit: int, force_zero: bool) -> list[int] | None:
    assert length >= 3
    values = list(range(limit + 1))
    for _ in range(200):
        scales = [random.choice(values) for _ in range(length)]
        if sum(scale > 0 for scale in scales) < 2:
            continue
        positives = sorted({scale for scale in scales if scale > 0})
        if len(positives) < 2:
            continue
        if max(positives) < 2:
            continue
        if force_zero and 0 not in scales:
            continue
        if not force_zero and all(scale == 0 for scale in scales):
            continue
        if sum(scales) < 4:
            continue
        return scales
    return None


def _random_content(h: int, w: int) -> np.ndarray | None:
    palette = random.sample(CONTENT_COLORS, random.randint(3, min(5, len(CONTENT_COLORS))))
    for _ in range(300):
        content = np.array(
            [[random.choice(palette) for _ in range(w)] for _ in range(h)],
            dtype=int,
        )
        rows = [tuple(map(int, row)) for row in content]
        cols = [tuple(map(int, col)) for col in content.T]
        if len(set(rows)) != h:
            continue
        if len(set(cols)) != w:
            continue
        if len(set(map(int, content.ravel()))) < 3:
            continue
        if np.array_equal(content, np.flipud(content)):
            continue
        if np.array_equal(content, np.fliplr(content)):
            continue
        return content
    return None


def _render_input(content: np.ndarray, row_scales: list[int], col_scales: list[int]) -> np.ndarray:
    h, w = content.shape
    top_h = max(col_scales)
    left_w = max(row_scales)
    grid = np.full((top_h + h, left_w + w), BG, dtype=int)

    grid[top_h : top_h + h, left_w : left_w + w] = content
    for rr, scale in enumerate(row_scales):
        if scale > 0:
            grid[top_h + rr, :scale] = RULER
    for cc, scale in enumerate(col_scales):
        if scale > 0:
            grid[:scale, left_w + cc] = RULER
    return grid


def _check_shortcuts(content: np.ndarray, row_scales: list[int], col_scales: list[int], out: np.ndarray) -> bool:
    row_binary = [1 if scale > 0 else 0 for scale in row_scales]
    col_binary = [1 if scale > 0 else 0 for scale in col_scales]

    if np.array_equal(out, _stretch(content, row_binary, col_binary)):
        return False

    row_flat = [1 if scale > 0 else 0 for scale in row_scales]
    col_full = [scale for scale in col_scales]
    if np.array_equal(out, _stretch(content, row_flat, col_full)):
        return False

    row_full = [scale for scale in row_scales]
    col_flat = [1 if scale > 0 else 0 for scale in col_scales]
    if np.array_equal(out, _stretch(content, row_full, col_flat)):
        return False

    if 0 in row_scales:
        row_no_delete = [max(1, scale) for scale in row_scales]
        if np.array_equal(out, _stretch(content, row_no_delete, col_scales)):
            return False

    if 0 in col_scales:
        col_no_delete = [max(1, scale) for scale in col_scales]
        if np.array_equal(out, _stretch(content, row_scales, col_no_delete)):
            return False

    return True


def _validate_sample(grid: np.ndarray, content: np.ndarray, row_scales: list[int], col_scales: list[int]) -> bool:
    try:
        parsed_content, parsed_rows, parsed_cols, _ = _extract_instance(grid)
        out = main(grid)
    except Exception:
        return False

    if not np.array_equal(parsed_content, content):
        return False
    if parsed_rows != row_scales or parsed_cols != col_scales:
        return False

    expected = _stretch(content, row_scales, col_scales)
    if not np.array_equal(out, expected):
        return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    if out.shape[0] != sum(row_scales) or out.shape[1] != sum(col_scales):
        return False

    if len(set(tuple(map(int, row)) for row in content)) != content.shape[0]:
        return False
    if len(set(tuple(map(int, col)) for col in content.T)) != content.shape[1]:
        return False

    if not _check_shortcuts(content, row_scales, col_scales, out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(3, 7)
    w = random.randint(3, 7)

    row_zero_mode = random.random() < 0.65
    col_zero_mode = random.random() < 0.65

    row_scales = _random_scales(h, limit=4, force_zero=row_zero_mode)
    col_scales = _random_scales(w, limit=4, force_zero=col_zero_mode)
    if row_scales is None or col_scales is None:
        return None

    if max(row_scales) + w > 30 or max(col_scales) + h > 30:
        return None
    if sum(row_scales) > 30 or sum(col_scales) > 30:
        return None

    content = _random_content(h, w)
    if content is None:
        return None

    grid = _render_input(content, row_scales, col_scales)
    if not _validate_sample(grid, content, row_scales, col_scales):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid histogram-ruler stretch input")
