from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BG = 0
COLOR_VALUES = tuple(range(10))
ACTIVE_MODES = ("row_only", "col_only", "both")
PATTERN_HEIGHT_RANGE = (5, 12)
PATTERN_WIDTH_RANGE = (5, 12)
PALETTE_SIZE_RANGE = (2, 4)
MAX_ARROW_LENGTH = 6
MIN_ARROW_SPACING = 3
LEFT_EXTRA_PAD_RANGE = (0, 2)
TOP_EXTRA_PAD_RANGE = (0, 2)
BOTTOM_PAD_RANGE = (1, 3)
RIGHT_PAD_RANGE = (1, 3)
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[list[tuple[int, int]], bool]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append((pts, touches_edge))
    return out


def _pattern_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = g != BG
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for pts, touches_edge in _components(mask):
        if touches_edge:
            continue
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        submask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if bool(np.all(submask)):
            area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            if area == len(pts):
                candidates.append((area, (r0, c0, r1, c1)))

    assert candidates, "expected a non-edge filled rectangle"
    best_area = max(area for area, _ in candidates)
    best = [bbox for area, bbox in candidates if area == best_area]
    assert len(best) == 1, "expected a unique central rectangle"
    return best[0]


def _apply_row_shifts(pattern: np.ndarray, row_shifts: np.ndarray) -> np.ndarray:
    out = np.array(pattern, copy=True)
    width = int(out.shape[1])
    for r, shift in enumerate(map(int, row_shifts)):
        if shift:
            out[r] = np.roll(out[r], shift % width)
    return out


def _apply_col_shifts(pattern: np.ndarray, col_shifts: np.ndarray) -> np.ndarray:
    out = np.array(pattern, copy=True)
    height = int(out.shape[0])
    for c, shift in enumerate(map(int, col_shifts)):
        if shift:
            out[:, c] = np.roll(out[:, c], shift % height)
    return out


def _apply_shifts(pattern: np.ndarray, row_shifts: np.ndarray, col_shifts: np.ndarray) -> np.ndarray:
    return _apply_col_shifts(_apply_row_shifts(pattern, row_shifts), col_shifts)


def _leading_run_length(values: np.ndarray) -> int:
    run = 0
    for value in map(int, values):
        if value == BG:
            break
        run += 1
    return run


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cyclically shift the rows and columns of the central rectangle according to edge arrows.

    Input structure:
    - The background is black.
    - One interior connected component is a filled rectangular pattern made of
      non-background colors.
    - The left edge may contain right-pointing arrows aligned with some pattern
      rows, and the top edge may contain downward-pointing arrows aligned with
      some pattern columns.

    Rule:
    - Find the unique non-edge component whose bounding box is fully filled; this
      box is the central pattern to keep.
    - For each pattern row, count the non-background shaft pixels to the left of
      the rectangle and cyclically shift that row right by that amount.
    - After all row shifts, for each pattern column, count the non-background
      shaft pixels above the rectangle and cyclically shift that column down by
      that amount.

    Key insight:
    - The arrows encode two pieces of information at once: their placement picks
      the affected row or column, and their shaft length gives the shift amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _pattern_bbox(g)
    pattern = g[r0 : r1 + 1, c0 : c1 + 1]
    height, width = map(int, pattern.shape)

    row_shifts = np.zeros(height, dtype=int)
    for i, r in enumerate(range(r0, r1 + 1)):
        row_shifts[i] = _leading_run_length(g[r, :c0])

    col_shifts = np.zeros(width, dtype=int)
    for j, c in enumerate(range(c0, c1 + 1)):
        col_shifts[j] = _leading_run_length(g[:r0, c])

    return _apply_shifts(pattern, row_shifts, col_shifts)


def _spaced_choices(n: int, min_gap: int) -> dict[int, list[tuple[int, ...]]]:
    choices: dict[int, list[tuple[int, ...]]] = {}
    for count in range(1, n + 1):
        valid = [
            combo
            for combo in combinations(range(n), count)
            if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1))
        ]
        if valid:
            choices[count] = valid
    return choices


def _pick_arrow_indices(n: int, active: bool) -> list[int]:
    if not active:
        return []
    options = _spaced_choices(n, MIN_ARROW_SPACING)
    assert options, "pattern must allow at least one arrow placement"
    count = random.choice(list(options))
    return list(random.choice(options[count]))


def _sample_pattern(
    height: int,
    width: int,
    palette: list[int],
    active_rows: list[int],
    active_cols: list[int],
    row_shifts: np.ndarray,
    col_shifts: np.ndarray,
) -> np.ndarray:
    for _ in range(200):
        values = random.choices(palette, k=height * width)
        pattern = np.array(values, dtype=int).reshape(height, width)
        if np.unique(pattern).size < 2:
            continue
        if any(np.unique(pattern[r]).size < 2 for r in active_rows):
            continue
        if any(np.unique(pattern[:, c]).size < 2 for c in active_cols):
            continue

        row_step = _apply_row_shifts(pattern, row_shifts)
        final = _apply_col_shifts(row_step, col_shifts)
        if active_rows and np.array_equal(row_step, pattern):
            continue
        if active_cols and np.array_equal(final, row_step):
            continue
        return pattern
    raise RuntimeError("failed to sample a visually informative pattern")


def _render_input(
    pattern: np.ndarray,
    row_shifts: np.ndarray,
    col_shifts: np.ndarray,
    marker: int,
) -> np.ndarray:
    height, width = map(int, pattern.shape)
    max_row_shift = int(row_shifts.max()) if row_shifts.size else 0
    max_col_shift = int(col_shifts.max()) if col_shifts.size else 0

    top_pad = max(1, max_col_shift + 2) + random.randint(*TOP_EXTRA_PAD_RANGE)
    left_pad = max(1, max_row_shift + 2) + random.randint(*LEFT_EXTRA_PAD_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    right_pad = random.randint(*RIGHT_PAD_RANGE)

    total_h = top_pad + height + bottom_pad
    total_w = left_pad + width + right_pad
    assert total_h <= 30 and total_w <= 30, "grid must stay within ARC bounds"

    g = np.full((total_h, total_w), BG, dtype=int)
    g[top_pad : top_pad + height, left_pad : left_pad + width] = pattern

    for row_idx, shift in enumerate(map(int, row_shifts)):
        if not shift:
            continue
        r = top_pad + row_idx
        g[r, :shift] = marker
        g[r - 1, shift] = marker
        g[r + 1, shift] = marker

    for col_idx, shift in enumerate(map(int, col_shifts)):
        if not shift:
            continue
        c = left_pad + col_idx
        g[:shift, c] = marker
        g[shift, c - 1] = marker
        g[shift, c + 1] = marker

    return g


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    mode = random.choice(ACTIVE_MODES)
    height = random.randint(*PATTERN_HEIGHT_RANGE)
    width = random.randint(*PATTERN_WIDTH_RANGE)

    row_active = mode in ("row_only", "both")
    col_active = mode in ("col_only", "both")
    active_rows = _pick_arrow_indices(height, row_active)
    active_cols = _pick_arrow_indices(width, col_active)

    row_limit = min(MAX_ARROW_LENGTH, width - 1)
    col_limit = min(MAX_ARROW_LENGTH, height - 1)
    assert row_limit >= 1 or not active_rows
    assert col_limit >= 1 or not active_cols

    row_shifts = np.zeros(height, dtype=int)
    for r in active_rows:
        row_shifts[r] = random.randint(1, row_limit)

    col_shifts = np.zeros(width, dtype=int)
    for c in active_cols:
        col_shifts[c] = random.randint(1, col_limit)

    marker = random.choice(COLOR_VALUES[1:])
    palette_pool = [color for color in COLOR_VALUES[1:] if color != marker]
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(palette_pool, palette_size)

    pattern = _sample_pattern(
        height,
        width,
        palette,
        active_rows,
        active_cols,
        row_shifts,
        col_shifts,
    )
    expected = _apply_shifts(pattern, row_shifts, col_shifts)
    return _render_input(pattern, row_shifts, col_shifts, marker), expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid, expected = _build_case()
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid shifting-arrows input")
