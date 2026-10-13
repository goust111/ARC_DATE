from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
LATTICE = 1
MARKER = 2
BLOCK = 2
MAX_ATTEMPTS = 500


def _rotate90(r: int, c: int, center_r: float, center_c: float, iters: int = 1) -> tuple[int, int]:
    rr = float(r)
    cc = float(c)
    for _ in range(iters % 4):
        rr, cc = center_r + (cc - center_c), center_c - (rr - center_r)
    return int(round(rr)), int(round(cc))


def _marker_info(grid: np.ndarray) -> tuple[int, int, int]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    for color in colors:
        coords = np.argwhere(grid == color)
        if len(coords) != 4:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != (2, 2):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            return int(color), int(r0), int(c0)
    raise AssertionError("expected a unique solid 2x2 marker block")


def _lattice_color(grid: np.ndarray, marker_color: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG and int(color) != marker_color]
    assert len(colors) == 1, "expected exactly one lattice color"
    return colors[0]


def _complete_lattice(grid: np.ndarray, lattice_color: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    height, width = out.shape
    center_r = (height - 1) / 2
    center_c = (width - 1) / 2

    for r, c in np.argwhere(grid == lattice_color):
        for iters in (1, 2, 3):
            rr, cc = _rotate90(int(r), int(c), center_r, center_c, iters)
            assert 0 <= rr < height and 0 <= cc < width, "rotation left the canvas"
            if out[rr, cc] == BG:
                out[rr, cc] = lattice_color
            else:
                assert out[rr, cc] != BG, "unexpected symmetry-completion state"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a damaged rotationally symmetric lattice, then recolor the marker's row-and-column bands.

    Input structure:
    - The grid has black background, one blue lattice color, and one distinct marker color.
    - The lattice is a regular pattern with some blue pixels erased in a single quadrant.
    - A solid 2x2 marker block sits on one lattice node inside the damaged region.

    Rule:
    - Find the 2x2 marker block and the lattice color.
    - Use the grid's 90-degree rotational symmetry to restore every missing blue pixel, but only fill black cells.
    - After completion, take the marker's two rows and two columns and recolor every blue lattice pixel on those bands to the marker color.

    Key insight:
    - The marker bands must be applied after symmetry completion, because some pixels on those bands exist only after the lattice is repaired.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_color, r0, c0 = _marker_info(g)
    lattice_color = _lattice_color(g, marker_color)

    completed = _complete_lattice(g, lattice_color)

    cross = np.zeros(g.shape, dtype=bool)
    cross[r0 : r0 + BLOCK, :] = True
    cross[:, c0 : c0 + BLOCK] = True

    out = completed.copy()
    out[cross & (completed == lattice_color)] = marker_color
    out[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = marker_color
    return out


def _valid_layouts() -> list[tuple[int, int, tuple[int, ...]]]:
    layouts: list[tuple[int, int, tuple[int, ...]]] = []
    for step in (3, 4, 5):
        for count in (4, 6):
            for offset in (0, 1, 2):
                size = 2 * offset + (count - 1) * step + BLOCK
                if 13 <= size <= 30:
                    starts = tuple(offset + i * step for i in range(count))
                    layouts.append((size, step, starts))
    return layouts


LAYOUTS = _valid_layouts()


def _build_complete_grid(size: int, starts: Iterable[int]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    starts = list(starts)
    for r0 in starts:
        for c0 in starts:
            grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = LATTICE
    return grid


def _quadrant_lists(size: int, starts: tuple[int, ...]) -> dict[str, tuple[list[int], list[int]]]:
    center = (size - 1) / 2
    top = [s for s in starts if s + 0.5 < center]
    bottom = [s for s in starts if s + 0.5 > center]
    left = [s for s in starts if s + 0.5 < center]
    right = [s for s in starts if s + 0.5 > center]
    return {
        "NW": (top, left),
        "NE": (top, right),
        "SW": (bottom, left),
        "SE": (bottom, right),
    }


def _expected_output(complete: np.ndarray, marker_r: int, marker_c: int) -> np.ndarray:
    out = complete.copy()
    cross = np.zeros(complete.shape, dtype=bool)
    cross[marker_r : marker_r + BLOCK, :] = True
    cross[:, marker_c : marker_c + BLOCK] = True
    out[cross & (out == LATTICE)] = MARKER
    out[marker_r : marker_r + BLOCK, marker_c : marker_c + BLOCK] = MARKER
    return out


def _cross_mask(shape: tuple[int, int], marker_r: int, marker_c: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[marker_r : marker_r + BLOCK, :] = True
    mask[:, marker_c : marker_c + BLOCK] = True
    return mask


def _visible_line_counts(grid: np.ndarray, marker_r: int, marker_c: int) -> bool:
    for row in (marker_r, marker_r + 1):
        if int(np.count_nonzero(grid[row, :] == LATTICE)) < 3:
            return False
    for col in (marker_c, marker_c + 1):
        if int(np.count_nonzero(grid[:, col] == LATTICE)) < 3:
            return False
    return True


def _candidate_rectangles(rows: list[int], cols: list[int]) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for r0_idx in range(len(rows)):
        for r1_idx in range(r0_idx, len(rows)):
            for c0_idx in range(len(cols)):
                for c1_idx in range(c0_idx, len(cols)):
                    rects.append((r0_idx, r1_idx, c0_idx, c1_idx))
    random.shuffle(rects)
    return rects


def _gen_attempt() -> np.ndarray | None:
    size, _step, starts = random.choice(LAYOUTS)
    complete = _build_complete_grid(size, starts)
    quadrants = _quadrant_lists(size, starts)
    quadrant = random.choice(tuple(quadrants))
    rows, cols = quadrants[quadrant]
    if len(rows) < 2 or len(cols) < 2:
        return None

    quadrant_pixels = len(rows) * len(cols) * (BLOCK * BLOCK)
    candidate_choices: list[tuple[int, int, int, int, int, int]] = []
    for r0_idx, r1_idx, c0_idx, c1_idx in _candidate_rectangles(rows, cols):
        block_count = (r1_idx - r0_idx + 1) * (c1_idx - c0_idx + 1)
        deleted_pixels = (block_count - 1) * (BLOCK * BLOCK)
        fraction = deleted_pixels / quadrant_pixels
        if not (0.30 <= fraction <= 0.60):
            continue
        for marker_row_idx in range(r0_idx, r1_idx + 1):
            for marker_col_idx in range(c0_idx, c1_idx + 1):
                if block_count == 1:
                    continue
                shares_row = block_count > 1 and (c1_idx > c0_idx)
                shares_col = block_count > 1 and (r1_idx > r0_idx)
                if not (shares_row or shares_col):
                    continue
                candidate_choices.append((r0_idx, r1_idx, c0_idx, c1_idx, marker_row_idx, marker_col_idx))

    if not candidate_choices:
        return None

    random.shuffle(candidate_choices)
    for r0_idx, r1_idx, c0_idx, c1_idx, marker_row_idx, marker_col_idx in candidate_choices:
        marker_r = rows[marker_row_idx]
        marker_c = cols[marker_col_idx]

        r_start = rows[r0_idx]
        r_end = rows[r1_idx] + BLOCK
        c_start = cols[c0_idx]
        c_end = cols[c1_idx] + BLOCK

        grid = complete.copy()
        defect_area = np.zeros_like(grid, dtype=bool)
        defect_area[r_start:r_end, c_start:c_end] = True
        marker_mask = np.zeros_like(grid, dtype=bool)
        marker_mask[marker_r : marker_r + BLOCK, marker_c : marker_c + BLOCK] = True
        grid[defect_area & (grid == LATTICE) & (~marker_mask)] = BG
        grid[marker_mask] = MARKER

        if not _visible_line_counts(grid, marker_r, marker_c):
            continue

        completed = _complete_lattice(grid, LATTICE)
        cross = _cross_mask(grid.shape, marker_r, marker_c)
        restored_cross = int(np.count_nonzero((grid == BG) & (completed == LATTICE) & cross))
        if restored_cross < 1:
            continue

        expected = _expected_output(complete, marker_r, marker_c)
        if not np.array_equal(main(grid), expected):
            continue

        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
