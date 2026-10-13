from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (9, 24)
GRID_WIDTH_RANGE = (9, 24)
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 13
GENERATION_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_hollow_rectangle(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return False

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    return np.array_equal(mask, expected)


def _extract_roles(grid: np.ndarray) -> tuple[int, int, tuple[int, int, int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    fg_colors = [int(v) for v in values if int(v) != bg]
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    frame_candidates = [color for color in fg_colors if _is_hollow_rectangle(grid == color)]
    assert len(frame_candidates) == 1, "expected exactly one hollow-rectangle frame color"
    frame_color = frame_candidates[0]
    marker_color = next(color for color in fg_colors if color != frame_color)

    frame_box = _bbox(grid == frame_color)
    assert frame_box is not None
    r0, c0, r1, c1 = frame_box
    assert 0 < r0 < r1 < h - 1 and 0 < c0 < c1 < w - 1, "frame must stay away from the border"

    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == marker_color)]
    assert len(marker_coords) == 3, "expected one interior marker and two edge markers"

    inside = [(r, c) for r, c in marker_coords if r0 < r < r1 and c0 < c < c1]
    row_sources = [(r, c) for r, c in marker_coords if r in (0, h - 1)]
    col_sources = [(r, c) for r, c in marker_coords if c in (0, w - 1)]

    assert len(inside) == 1, "expected exactly one marker inside the frame"
    assert len(row_sources) == 1, "expected exactly one top/bottom edge marker"
    assert len(col_sources) == 1, "expected exactly one left/right edge marker"

    inner_marker = inside[0]
    row_marker = row_sources[0]
    col_marker = col_sources[0]
    projected = (col_marker[0], row_marker[1])

    pr, pc = projected
    mr, mc = inner_marker
    assert r0 < pr < r1 and c0 < pc < c1, "projected corner must land inside the frame"
    assert pr != mr and pc != mc, "projected corner must differ from the interior marker in both axes"

    return frame_color, marker_color, frame_box, inner_marker, row_marker, col_marker


def main(grid: np.ndarray) -> np.ndarray:
    """Use one interior marker and two border markers to preserve a projected window inside a hollow frame.

    Input structure:
    - The grid has a dominant background, one one-cell-thick hollow rectangular frame away from the border,
      and three same-colored marker pixels.
    - One marker sits strictly inside the frame interior; one marker lies on the top or bottom border and
      contributes a column; one marker lies on the left or right border and contributes a row.

    Rule:
    - Combine the interior marker and the row/column intersection implied by the two edge markers to form the
      opposite corners of a protection window.
    - Recolor every background cell strictly inside the frame but outside that window to the frame color.
    - Keep the frame and the interior marker unchanged, and erase the two edge markers to background.

    Key insight:
    - The border markers are remote coordinates rather than drawable objects: together they hide the second
      window corner that determines which part of the frame interior stays empty.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color, _marker_color, frame_box, inner_marker, row_marker, col_marker = _extract_roles(g)
    r0, c0, r1, c1 = frame_box
    mr, mc = inner_marker
    projected = (col_marker[0], row_marker[1])
    pr, pc = projected

    out = g.copy()
    out[row_marker] = BG
    out[col_marker] = BG

    interior = np.zeros_like(g, dtype=bool)
    interior[r0 + 1 : r1, c0 + 1 : c1] = True

    window = np.zeros_like(g, dtype=bool)
    wr0, wr1 = sorted((mr, pr))
    wc0, wc1 = sorted((mc, pc))
    window[wr0 : wr1 + 1, wc0 : wc1 + 1] = True

    fill_mask = interior & ~window & (out == BG)
    out[fill_mask] = frame_color
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _expected_from_params(
    grid: np.ndarray,
    frame_color: int,
    frame_box: tuple[int, int, int, int],
    inner_marker: tuple[int, int],
    row_marker: tuple[int, int],
    col_marker: tuple[int, int],
) -> np.ndarray:
    out = grid.copy()
    out[row_marker] = BG
    out[col_marker] = BG

    r0, c0, r1, c1 = frame_box
    mr, mc = inner_marker
    pr, pc = col_marker[0], row_marker[1]
    wr0, wr1 = sorted((mr, pr))
    wc0, wc1 = sorted((mc, pc))

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if wr0 <= r <= wr1 and wc0 <= c <= wc1:
                continue
            if out[r, c] == BG:
                out[r, c] = frame_color
    return out


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    frame_color, marker_color = random.sample(PALETTE, 2)

    max_frame_h = min(FRAME_SIZE_MAX, h - 2)
    max_frame_w = min(FRAME_SIZE_MAX, w - 2)
    if max_frame_h < FRAME_SIZE_MIN or max_frame_w < FRAME_SIZE_MIN:
        return None

    frame_h = random.randint(FRAME_SIZE_MIN, max_frame_h)
    frame_w = random.randint(FRAME_SIZE_MIN, max_frame_w)
    r0 = random.randint(1, h - frame_h - 1)
    c0 = random.randint(1, w - frame_w - 1)
    frame_box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    interior_rows = list(range(r0 + 1, r0 + frame_h - 1))
    interior_cols = list(range(c0 + 1, c0 + frame_w - 1))
    if len(interior_rows) < 2 or len(interior_cols) < 2:
        return None

    projected_r = random.choice(interior_rows)
    projected_c = random.choice(interior_cols)
    marker_r = random.choice([r for r in interior_rows if r != projected_r])
    marker_c = random.choice([c for c in interior_cols if c != projected_c])

    win_h = abs(marker_r - projected_r) + 1
    win_w = abs(marker_c - projected_c) + 1
    interior_area = len(interior_rows) * len(interior_cols)
    window_area = win_h * win_w
    if window_area < 4 or interior_area - window_area < 2:
        return None

    row_marker = (random.choice((0, h - 1)), projected_c)
    col_marker = (projected_r, random.choice((0, w - 1)))
    inner_marker = (marker_r, marker_c)

    grid = np.full((h, w), BG, dtype=int)
    _draw_frame(grid, frame_box, frame_color)
    grid[inner_marker] = marker_color
    grid[row_marker] = marker_color
    grid[col_marker] = marker_color

    bg_count = int(np.count_nonzero(grid == BG))
    frame_count = int(np.count_nonzero(grid == frame_color))
    if bg_count <= max(frame_count, 3):
        return None

    expected = _expected_from_params(grid, frame_color, frame_box, inner_marker, row_marker, col_marker)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-source window instance")
