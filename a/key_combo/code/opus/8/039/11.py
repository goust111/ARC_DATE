from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, paste


BG = 0
FOLD_SEPARATOR = 5
SPRITE_SIZE = 3
STRIDE = SPRITE_SIZE + 1
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MASK_DENSITY_RANGE = (4, 6)
VISIBLE_RATIO_RANGE = (0.40, 0.60)
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 200
REFERENCE_ATTEMPTS = 120


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *separators, int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = {end - start for start, end in intervals}
    assert intervals and min(end - start for start, end in intervals) > 0
    assert len(sizes) == 1
    return intervals


def _extract_layout(top: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_separators = [r for r in range(top.shape[0]) if np.all(top[r, :] == bg)]
    col_separators = [c for c in range(top.shape[1]) if np.all(top[:, c] == bg)]
    row_intervals = _regular_intervals(row_separators, int(top.shape[0]))
    col_intervals = _regular_intervals(col_separators, int(top.shape[1]))
    sprite_size = row_intervals[0][1] - row_intervals[0][0]
    assert len(row_intervals) >= 2
    assert len(col_intervals) >= 3
    assert all(end - start == sprite_size for start, end in row_intervals)
    assert all(end - start == sprite_size for start, end in col_intervals)
    return row_intervals, col_intervals, int(sprite_size)


def _read_cyclic_masks(
    top: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    bg: int,
) -> tuple[list[list[np.ndarray]], list[int]]:
    masks: list[list[np.ndarray]] = []
    col_colors: list[int] = []

    for band_index, (r0, r1) in enumerate(row_intervals):
        row_masks: list[np.ndarray] = []
        for col_index, (c0, c1) in enumerate(col_intervals):
            block = top[r0:r1, c0:c1]
            colors = sorted(set(int(v) for v in block[block != bg]))
            assert len(colors) == 1
            color = colors[0]
            if band_index == 0:
                col_colors.append(color)
            else:
                assert color == col_colors[col_index]
            row_masks.append(block != bg)
        masks.append(row_masks)

    n_rows = len(row_intervals)
    n_cols = len(col_intervals)
    for band_index in range(n_rows):
        for col_index in range(n_cols):
            expected = masks[0][(col_index + band_index) % n_cols]
            assert np.array_equal(masks[band_index][col_index], expected)

    return masks, col_colors


def _render_band(
    masks: list[np.ndarray],
    col_colors: list[int],
    col_intervals: list[tuple[int, int]],
    width: int,
    bg: int,
) -> np.ndarray:
    sprite_size = masks[0].shape[0]
    band = np.full((sprite_size, width), int(bg), dtype=int)
    for color, mask, (c0, c1) in zip(col_colors, masks, col_intervals):
        patch = np.full((sprite_size, c1 - c0), int(bg), dtype=int)
        patch[mask] = int(color)
        paste(band, patch, 0, int(c0), transparent=int(bg))
    return band


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the next cyclic sprite row, then fold it onto the partial reference band below the grey separator.

    Input structure:
    - The upper region is a sprite lattice with black separator rows and columns between equal-sized sprite blocks.
    - Each sprite column keeps one fixed non-black color, while the binary sprite masks shift one column left from one lattice row to the next.
    - A full grey horizontal separator sits below the lattice, and under it is one sprite-height band that shows only a partial view of the next row.

    Rule:
    - Parse the lattice into sprite rows and columns, ignoring colors to recover the cyclic mask sequence.
    - Predict the next lattice row by shifting the last observed mask arrangement one more step left and recoloring each column with its fixed column color.
    - Remove the grey separator and overlay the partial reference band onto the predicted row, keeping the reference color wherever it is non-black.

    Key insight:
    - The top lattice determines which mask belongs in each next-row column, while the lower band is a partial but aligned copy of that next row, so the answer is the separator-free overlay of prediction and reference.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(get_bg(g))
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == FOLD_SEPARATOR)]
    assert len(sep_rows) == 1
    sep = sep_rows[0]
    assert not np.any(g[:sep, :] == FOLD_SEPARATOR)
    assert not np.any(g[sep + 1 :, :] == FOLD_SEPARATOR)

    top = g[:sep, :]
    reference = g[sep + 1 :, :]
    assert top.size > 0 and reference.size > 0

    row_intervals, col_intervals, sprite_size = _extract_layout(top, bg)
    assert reference.shape == (sprite_size, top.shape[1])

    masks, col_colors = _read_cyclic_masks(top, row_intervals, col_intervals, bg)
    n_rows = len(row_intervals)
    n_cols = len(col_intervals)
    predicted_masks = [masks[0][(col_index + n_rows) % n_cols] for col_index in range(n_cols)]
    predicted = _render_band(predicted_masks, col_colors, col_intervals, int(top.shape[1]), bg)

    ref_mask = reference != bg
    assert np.all(predicted[ref_mask] == reference[ref_mask])

    out = predicted.copy()
    out[ref_mask] = reference[ref_mask]
    return out


def _sample_mask() -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        density = random.randint(*MASK_DENSITY_RANGE)
        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        cells = random.sample(range(SPRITE_SIZE * SPRITE_SIZE), density)
        for index in cells:
            mask[index // SPRITE_SIZE, index % SPRITE_SIZE] = True
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        return mask
    return None


def _sample_base_masks(n_cols: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    while len(masks) < n_cols:
        mask = _sample_mask()
        if mask is None:
            return None
        key = mask.tobytes()
        if key in seen:
            continue
        masks.append(mask)
        seen.add(key)
    return masks


def _build_top_lattice(base_masks: list[np.ndarray], col_colors: list[int], n_rows: int) -> np.ndarray:
    n_cols = len(base_masks)
    height = n_rows * SPRITE_SIZE + (n_rows - 1)
    width = n_cols * SPRITE_SIZE + (n_cols - 1)
    top = np.full((height, width), BG, dtype=int)

    for band_index in range(n_rows):
        row_masks = [base_masks[(col_index + band_index) % n_cols] for col_index in range(n_cols)]
        band = np.full((SPRITE_SIZE, width), BG, dtype=int)
        for col_index, (color, mask) in enumerate(zip(col_colors, row_masks)):
            c0 = col_index * STRIDE
            patch = np.full((SPRITE_SIZE, SPRITE_SIZE), BG, dtype=int)
            patch[mask] = int(color)
            paste(band, patch, 0, c0, transparent=BG)
        r0 = band_index * STRIDE
        paste(top, band, r0, 0)

    return top


def _sample_reference_band(predicted: np.ndarray, n_cols: int) -> np.ndarray | None:
    total_foreground = int(np.count_nonzero(predicted))
    if total_foreground == 0:
        return None

    column_cells = []
    for col_index in range(n_cols):
        c0 = col_index * STRIDE
        coords = [tuple(map(int, rc)) for rc in np.argwhere(predicted[:, c0 : c0 + SPRITE_SIZE] != BG)]
        coords = [(r, c + c0) for r, c in coords]
        if len(coords) < 2:
            return None
        column_cells.append(coords)

    for _ in range(REFERENCE_ATTEMPTS):
        visible = np.zeros_like(predicted, dtype=bool)
        for coords in column_cells:
            keep = random.randint(1, len(coords) - 1)
            chosen = random.sample(coords, keep)
            for r, c in chosen:
                visible[r, c] = True

        ratio = float(visible.sum()) / float(total_foreground)
        if not (VISIBLE_RATIO_RANGE[0] <= ratio <= VISIBLE_RATIO_RANGE[1]):
            continue

        for r in range(predicted.shape[0]):
            row_fg = predicted[r] != BG
            if not np.any(row_fg):
                continue
            row_visible = visible[r] & row_fg
            if not row_visible.any() or np.all(row_visible == row_fg):
                break
        else:
            reference = np.full_like(predicted, BG)
            reference[visible] = predicted[visible]
            return reference

    return None


def _generate_attempt() -> np.ndarray | None:
    n_cols = random.randint(3, 4)
    n_rows = random.randint(3, 4)
    col_colors = random.sample(COLOR_POOL, n_cols)
    base_masks = _sample_base_masks(n_cols)
    if base_masks is None:
        return None

    top = _build_top_lattice(base_masks, col_colors, n_rows)
    next_masks = [base_masks[(col_index + n_rows) % n_cols] for col_index in range(n_cols)]
    col_intervals = [(col_index * STRIDE, col_index * STRIDE + SPRITE_SIZE) for col_index in range(n_cols)]
    predicted = _render_band(next_masks, col_colors, col_intervals, int(top.shape[1]), BG)
    reference = _sample_reference_band(predicted, n_cols)
    if reference is None:
        return None

    inp = np.full((top.shape[0] + 1 + SPRITE_SIZE, top.shape[1]), BG, dtype=int)
    paste(inp, top, 0, 0)
    inp[top.shape[0], :] = FOLD_SEPARATOR
    paste(inp, reference, top.shape[0] + 1, 0)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, predicted):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid cyclic-fold input")
