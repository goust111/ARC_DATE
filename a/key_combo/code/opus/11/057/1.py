from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
SEPARATOR = 8
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GEN_ATTEMPTS = 400
INNER_ATTEMPTS = 300
HEIGHT_RANGE = (8, 12)
WIDTH_RANGE = (5, 7)
COUNT_RANGE = (3, 6)
SHIFT_RANGE = (1, 3)


def _shift_down(mask: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    if shift < mask.shape[0]:
        out[shift:, :] = mask[: mask.shape[0] - shift, :]
    return out


def _union_with_halo(left_mask: np.ndarray, right_mask: np.ndarray, shift: int) -> np.ndarray:
    shifted_right = _shift_down(right_mask, int(shift))
    union = left_mask | shifted_right
    out = np.full(union.shape, BLACK, dtype=int)
    out[union] = BLUE

    halo = np.zeros_like(union, dtype=bool)
    height, width = union.shape
    for r, c in np.argwhere(union):
        r = int(r)
        c = int(c)
        for dr, dc in DIAGONALS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and not union[nr, nc]:
                halo[nr, nc] = True

    out[halo & ~union] = GREEN
    return out


def _raw_union(left_mask: np.ndarray, right_mask: np.ndarray, shift: int) -> np.ndarray:
    return _shift_down(right_mask, int(shift)) | left_mask


def _has_diagonal_blue_pair(union: np.ndarray) -> bool:
    for r, c in np.argwhere(union):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 1), (1, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < union.shape[0] and 0 <= nc < union.shape[1] and union[nr, nc]:
                return True
    return False


def _sample_unique_cells(height: int, width: int, count: int, preset: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set(preset)
    while len(cells) < count:
        cells.add((random.randrange(height), random.randrange(width)))
    return cells


def _build_masks(
    height: int,
    width: int,
    shift: int,
    left_count: int,
    right_count: int,
    overlap_count: int,
    require_edge: bool,
    require_conflict: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    right_only_count = right_count - overlap_count
    if right_only_count < 1 or left_count - overlap_count < 1:
        return None

    border_cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if r in (0, height - 1) or c in (0, width - 1)
    ]

    for _ in range(INNER_ATTEMPTS):
        left_seed: set[tuple[int, int]] = set()

        if require_conflict:
            diag_choices = []
            for r in range(height - 1):
                for c in range(width - 1):
                    diag_choices.append(((r, c), (r + 1, c + 1)))
                    diag_choices.append(((r + 1, c), (r, c + 1)))
            pair = random.choice(diag_choices)
            left_seed.update(pair)

        if require_edge:
            left_seed.add(random.choice(border_cells))

        left_cells = _sample_unique_cells(height, width, left_count, left_seed)
        overlap_candidates = [
            (r, c)
            for (r, c) in left_cells
            if r >= shift and (r - shift, c) not in left_cells
        ]
        if len(overlap_candidates) < overlap_count:
            continue

        overlap_cells = set(random.sample(overlap_candidates, overlap_count))
        used_origins = {(r - shift, c) for (r, c) in overlap_cells}

        right_shift_candidates = [
            (r, c)
            for r in range(shift, height)
            for c in range(width)
            if (r, c) not in left_cells
            and (r - shift, c) not in left_cells
            and (r - shift, c) not in used_origins
        ]
        if len(right_shift_candidates) < right_only_count:
            continue

        right_shifted = set(overlap_cells)
        right_shifted.update(random.sample(right_shift_candidates, right_only_count))
        right_cells = {(r - shift, c) for (r, c) in right_shifted}

        if len(right_cells) != right_count:
            continue
        if left_cells & right_cells:
            continue

        union = np.zeros((height, width), dtype=bool)
        for r, c in left_cells | right_shifted:
            union[r, c] = True

        if require_edge and not any(union[r, c] for r, c in border_cells):
            continue
        if require_conflict and not _has_diagonal_blue_pair(union):
            continue

        left_mask = np.zeros((height, width), dtype=bool)
        right_mask = np.zeros((height, width), dtype=bool)
        for r, c in left_cells:
            left_mask[r, c] = True
        for r, c in right_cells:
            right_mask[r, c] = True
        return left_mask, right_mask

    return None


def _render_input(left_mask: np.ndarray, right_mask: np.ndarray, shift: int) -> np.ndarray:
    height, width = left_mask.shape
    grid = np.full((height, 2 * width + 1), BLACK, dtype=int)
    grid[:, width] = SEPARATOR

    marker_rows = random.sample(range(height), int(shift))
    grid[marker_rows, width] = RED
    left_view = grid[:, :width]
    right_view = grid[:, width + 1 :]
    left_view[left_mask] = BLUE
    right_view[right_mask] = BLUE
    return grid


def _validate_generated(
    grid: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    shift: int,
    require_edge: bool,
    require_conflict: bool,
) -> None:
    height, width = left_mask.shape
    assert grid.shape == (height, 2 * width + 1)
    assert 1 <= shift <= 3
    assert COUNT_RANGE[0] <= int(left_mask.sum()) <= COUNT_RANGE[1]
    assert COUNT_RANGE[0] <= int(right_mask.sum()) <= COUNT_RANGE[1]
    assert not bool(np.any(left_mask & right_mask))

    separator = grid[:, width]
    assert int(np.count_nonzero(separator == RED)) == shift
    assert set(map(int, np.unique(separator))) <= {RED, SEPARATOR}
    assert np.all(np.isin(grid[:, :width], [BLACK, BLUE]))
    assert np.all(np.isin(grid[:, width + 1 :], [BLACK, BLUE]))

    shifted_right = _shift_down(right_mask, shift)
    overlap = left_mask & shifted_right
    assert bool(overlap.any())

    union = _raw_union(left_mask, right_mask, shift)
    expected = _union_with_halo(left_mask, right_mask, shift)
    raw_out = np.full((height, width), BLACK, dtype=int)
    raw_out[union] = BLUE

    assert bool(np.any(expected == GREEN))
    assert not np.array_equal(expected, raw_out)
    assert not np.array_equal(expected, _union_with_halo(left_mask, np.zeros_like(right_mask), shift))
    assert not np.array_equal(expected, _union_with_halo(np.zeros_like(left_mask), right_mask, shift))

    for alt_shift in range(height):
        alt = _union_with_halo(left_mask, right_mask, alt_shift)
        if alt_shift != shift:
            assert not np.array_equal(expected, alt)

    if require_edge:
        assert any(
            union[r, c]
            for r in range(height)
            for c in range(width)
            if r in (0, height - 1) or c in (0, width - 1)
        )
    if require_conflict:
        assert _has_diagonal_blue_pair(union)

    actual = main(grid)
    assert np.array_equal(actual, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """Count red markers in the separator, shift the right mask down by that amount, overlay both masks, and add green diagonal halo cells.

    Input structure:
    - The grid has odd width and a single middle separator column.
    - Separator cells are mostly cyan, with some red marker cells whose count encodes a vertical shift.
    - Left and right side panels have the same size and contain sparse blue foreground pixels on a black background.

    Rule:
    - Remove the separator column and count its red marker cells as K.
    - Shift the entire right panel downward by K rows, filling newly exposed top cells with black and cropping cells that fall below the panel.
    - Take the pixelwise union of the left panel and shifted right panel, coloring union cells blue.
    - For every blue union cell, paint each in-bounds diagonal black neighbor green; if a cell is both blue and a halo target, keep it blue.

    Key insight:
    - The separator markers are not decoration: their count determines the right-panel phase shift, and that shifted union is what controls the final diagonal halo pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, total_width = map(int, g.shape)
    assert total_width >= 3 and total_width % 2 == 1, "expected equal side panels split by one separator column"

    width = total_width // 2
    separator = g[:, width]
    assert set(map(int, np.unique(separator))) <= {RED, SEPARATOR}, "separator should contain only cyan cells and red markers"

    left = g[:, :width]
    right = g[:, width + 1 :]
    assert left.shape == right.shape
    assert np.all(np.isin(left, [BLACK, BLUE])), "left panel should use only black and blue"
    assert np.all(np.isin(right, [BLACK, BLUE])), "right panel should use only black and blue"

    shift = int(np.count_nonzero(separator == RED))
    assert shift >= 1, "expected at least one red separator marker"

    left_mask = left == BLUE
    right_mask = right == BLUE
    return _union_with_halo(left_mask, right_mask, shift)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(*HEIGHT_RANGE)
        width = random.randint(*WIDTH_RANGE)
        shift = random.randint(*SHIFT_RANGE)
        left_count = random.randint(*COUNT_RANGE)
        right_count = random.randint(*COUNT_RANGE)
        overlap_cap = min(2, left_count - 1, right_count - 1)
        if overlap_cap < 1:
            continue
        overlap_count = random.randint(1, overlap_cap)
        require_edge = random.random() < 0.5
        require_conflict = random.random() < 0.5

        masks = _build_masks(
            height=height,
            width=width,
            shift=shift,
            left_count=left_count,
            right_count=right_count,
            overlap_count=overlap_count,
            require_edge=require_edge,
            require_conflict=require_conflict,
        )
        if masks is None:
            continue

        left_mask, right_mask = masks
        grid = _render_input(left_mask, right_mask, shift)
        try:
            _validate_generated(grid, left_mask, right_mask, shift, require_edge, require_conflict)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid non-degenerate input")
