from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
BAR_COLOR = 2
MASK_COLOR = 5
SEGMENT_COLORS = (1, 3, 4, 6, 7)

GRID_HEIGHT_RANGE = (16, 20)
GRID_WIDTH_RANGE = (15, 19)
MASK_HEIGHTS = (2, 3)
SEGMENT_COUNT_RANGE = (3, 5)
SEGMENT_LENGTH_RANGE = (3, 6)
GEN_RETRY_LIMIT = 400


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _segment_runs(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    for color in SEGMENT_COLORS:
        for coords in _find_color_components(grid, color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0 = int(rows.min())
            r1 = int(rows.max())
            c0 = int(cols.min())
            c1 = int(cols.max())
            assert r0 == r1, "segments must be horizontal"
            assert int(coords.shape[0]) == c1 - c0 + 1, "segments must be solid one-cell bars"
            segments.append((r0, c0, c1, int(color)))
    assert segments, "expected colored segments"
    return segments


def _mask_columns(grid: np.ndarray) -> np.ndarray:
    mask = np.any(grid == int(MASK_COLOR), axis=0)
    assert bool(mask.any()) and not bool(mask.all()), "mask must contain both open and closed columns"
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each horizontal segment downward to the red edge bar, but only through columns opened by the grey downsampled mask.

    Input structure:
    - The background is black.
    - The bottom row is a solid red edge bar.
    - The upper area contains several one-cell-thick horizontal colored segments and one grey monochrome mask object.

    Rule:
    - Downsample the grey object into a column mask by marking each global column open iff that column contains at least one grey pixel.
    - Erase the grey object to background.
    - For every colored segment, fill the cells from the next row down to the row above the red bar in the same color, but only in open-mask columns.
    - When multiple projections overlap, the segment closer to the bottom bar overwrites the farther one; the original segment pixels and the red bar remain unchanged.

    Key insight:
    - The grey pattern is not output content; its occupied columns become a gate that selectively lets the vertical projections pass, while distance to the bottom wall resolves occlusion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 4
    assert bool(np.all(g[-1, :] == int(BAR_COLOR))), "expected a full red bottom bar"

    open_cols = _mask_columns(g)
    segments = _segment_runs(g)

    out = g.copy()
    out[out == int(MASK_COLOR)] = int(BACKGROUND)

    for row, c0, c1, color in sorted(segments, key=lambda item: item[0]):
        if row + 1 > h - 2:
            continue
        for c in range(c0, c1 + 1):
            if open_cols[c]:
                out[row + 1 : h - 1, c] = int(color)
    return out


def _mask_width_options(width: int, height: int) -> list[int]:
    lo = max(5, (15 + height - 1) // height)
    hi = min(width - 3, 25 // height)
    return list(range(lo, hi + 1))


def _choose_mask(window_width: int, side: str, total_width: int) -> tuple[int, int]:
    if side == "left":
        return 0, window_width - 1
    return total_width - window_width, total_width - 1


def _rand_length() -> int:
    return random.randint(*SEGMENT_LENGTH_RANGE)


def _anchor_span(mask_start: int, mask_end: int, width: int, side: str, overlap_col: int) -> tuple[int, int] | None:
    for _ in range(100):
        length = _rand_length()
        if side == "left":
            end_min = max(overlap_col, mask_end + 1)
            end_max = min(width - 1, mask_end + 3)
            if end_min > end_max:
                continue
            end = random.randint(end_min, end_max)
            start = end - length + 1
            if 0 <= start <= overlap_col:
                return start, end
        else:
            start_min = max(0, mask_start - 3)
            start_max = min(overlap_col, mask_start - 1)
            if start_min > start_max:
                continue
            start = random.randint(start_min, start_max)
            end = start + length - 1
            if overlap_col <= end < width:
                return start, end
    return None


def _random_span(mask_start: int, mask_end: int, width: int, side: str) -> tuple[int, int] | None:
    for _ in range(120):
        length = _rand_length()
        if side == "left":
            start = random.randint(0, min(mask_end, width - length))
            end = start + length - 1
            if end >= mask_start and start <= mask_end:
                return start, end
        else:
            start = random.randint(max(0, mask_start - length + 1), width - length)
            end = start + length - 1
            if end >= mask_start and start <= mask_end:
                return start, end
    return None


def _build_input(h: int, w: int, mask_h: int, mask_start: int, mask_end: int, segments: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[-1, :] = int(BAR_COLOR)
    grid[:mask_h, mask_start : mask_end + 1] = int(MASK_COLOR)
    for row, start, end, color in segments:
        grid[row, start : end + 1] = int(color)
    return grid


def _valid_rows(height: int, mask_h: int) -> list[int]:
    top = mask_h + 2
    bottom = min(height - 3, height // 2 + 2)
    return list(range(top, bottom + 1))


def _has_projection_overlap(mask_start: int, mask_end: int, segments: list[tuple[int, int, int, int]]) -> bool:
    for i in range(len(segments)):
        r1, s1, e1, _ = segments[i]
        for j in range(i + 1, len(segments)):
            r2, s2, e2, _ = segments[j]
            if r1 == r2:
                continue
            overlap_start = max(s1, s2, mask_start)
            overlap_end = min(e1, e2, mask_end)
            if overlap_start <= overlap_end:
                return True
    return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    mask_h = random.choice(MASK_HEIGHTS)
    mask_widths = _mask_width_options(w, mask_h)
    if not mask_widths:
        return None
    mask_w = random.choice(mask_widths)
    side = random.choice(("left", "right"))
    mask_start, mask_end = _choose_mask(mask_w, side, w)

    count = random.randint(*SEGMENT_COUNT_RANGE)
    rows = _valid_rows(h, mask_h)
    if len(rows) < count:
        return None
    chosen_rows = sorted(random.sample(rows, count))
    colors = random.sample(SEGMENT_COLORS, count)

    if side == "left":
        overlap_lo = max(mask_start, mask_end - 2)
        overlap_hi = mask_end
    else:
        overlap_lo = mask_start
        overlap_hi = min(mask_end, mask_start + 2)
    if overlap_lo > overlap_hi:
        return None
    overlap_col = random.randint(overlap_lo, overlap_hi)

    span_a = _anchor_span(mask_start, mask_end, w, side, overlap_col)
    span_b = _anchor_span(mask_start, mask_end, w, side, overlap_col)
    if span_a is None or span_b is None:
        return None

    spans = [span_a, span_b]
    for _ in range(count - 2):
        span = _random_span(mask_start, mask_end, w, side)
        if span is None:
            return None
        spans.append(span)

    random.shuffle(spans)
    segments = [
        (row, span[0], span[1], color)
        for row, span, color in zip(chosen_rows, spans, colors)
    ]

    if not _has_projection_overlap(mask_start, mask_end, segments):
        return None

    grid = _build_input(h, w, mask_h, mask_start, mask_end, segments)
    out = main(grid)

    if np.array_equal(out, grid):
        return None
    if np.any(out == int(MASK_COLOR)):
        return None
    if not np.any((out[:-1, :] != grid[:-1, :]) & (out[:-1, :] != int(BACKGROUND))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle input after {GEN_RETRY_LIMIT} attempts")
