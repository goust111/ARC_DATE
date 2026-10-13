from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400
HEIGHT_RANGE = (3, 7)
WIDTH_RANGE = (3, 7)
FG_COUNT_MIN = 8
FG_COUNT_MAX = 28
MARGIN_MIN = 1
MARGIN_MAX = 6


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expand_by_counts(crop: np.ndarray) -> np.ndarray:
    fg = crop != BG
    row_heights = fg.sum(axis=1).astype(int)
    col_widths = fg.sum(axis=0).astype(int)
    assert np.all(row_heights > 0), "every bbox row must contain foreground"
    assert np.all(col_widths > 0), "every bbox column must contain foreground"
    return np.repeat(np.repeat(crop, row_heights, axis=0), col_widths, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Non-uniformly enlarge the foreground pattern using its own row and column foreground counts as the scale.

    Input structure:
    - A black background contains one multicolor foreground pattern.
    - The pattern's tight bounding box contains both foreground and background cells.
    - Every row and every column inside that bounding box contains at least one foreground cell.

    Rule:
    - Crop to the foreground bounding box.
    - For each cropped row, count its non-black cells and use that count as the row's output height.
    - For each cropped column, count its non-black cells and use that count as the column's output width.
    - Replace each original cell by a solid block of its own color with the corresponding row-height and column-width, preserving order.

    Key insight:
    - The pattern provides its own non-uniform ruler: row foreground counts set vertical scale and column foreground counts set horizontal scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg = g != BG
    box = _bbox(fg)
    r0, c0, r1, c1 = box
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.any(crop == BG), "expected at least one background cell inside the bbox"

    out = _expand_by_counts(crop)
    assert max(out.shape) <= 30, "output exceeds ARC size limit"
    return out


def _sample_mask(height: int, width: int) -> np.ndarray | None:
    cell_count = height * width
    max_fg = min(FG_COUNT_MAX, cell_count - 1)
    if max_fg < max(height, width):
        return None

    mask = np.zeros((height, width), dtype=bool)

    for r in range(height):
        mask[r, random.randrange(width)] = True
    for c in range(width):
        if not mask[:, c].any():
            mask[random.randrange(height), c] = True

    current = int(mask.sum())
    min_fg = max(FG_COUNT_MIN, current)
    if min_fg > max_fg:
        return None

    target_fg = random.randint(min_fg, max_fg)
    remaining = [(r, c) for r in range(height) for c in range(width) if not mask[r, c]]
    random.shuffle(remaining)
    for r, c in remaining[: target_fg - current]:
        mask[r, c] = True

    row_counts = mask.sum(axis=1).astype(int)
    col_counts = mask.sum(axis=0).astype(int)
    if not np.all(row_counts > 0) or not np.all(col_counts > 0):
        return None
    if int(mask.sum()) == cell_count:
        return None
    if len(set(map(int, row_counts))) == 1 and len(set(map(int, col_counts))) == 1:
        return None
    return mask


def _paint_pattern(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    fg_cells = int(mask.sum())
    palette_size = random.randint(2, min(4, fg_cells))
    colors = random.sample(PALETTE, palette_size)

    crop = np.zeros((height, width), dtype=int)
    used_colors = set()
    phase = random.randrange(palette_size)
    alt_phase = random.randrange(palette_size)

    for r, c in np.argwhere(mask):
        color = colors[(r + 2 * c + phase + ((r * c + alt_phase) % palette_size)) % palette_size]
        crop[int(r), int(c)] = color
        used_colors.add(int(color))

    if len(used_colors) == 1:
        cells = np.argwhere(mask)
        rr, cc = cells[random.randrange(len(cells))]
        crop[int(rr), int(cc)] = colors[1]
    return crop


def _place_crop(crop: np.ndarray) -> np.ndarray | None:
    height, width = crop.shape
    top = random.randint(MARGIN_MIN, MARGIN_MAX)
    left = random.randint(MARGIN_MIN, MARGIN_MAX)
    max_bottom = 30 - height - top
    max_right = 30 - width - left
    if max_bottom < MARGIN_MIN or max_right < MARGIN_MIN:
        return None

    bottom = random.randint(MARGIN_MIN, min(MARGIN_MAX, max_bottom))
    right = random.randint(MARGIN_MIN, min(MARGIN_MAX, max_right))
    grid = np.zeros((top + height + bottom, left + width + right), dtype=int)
    grid[top : top + height, left : left + width] = crop
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    mask = _sample_mask(height, width)
    if mask is None:
        return None

    crop = _paint_pattern(mask)
    grid = _place_crop(crop)
    if grid is None:
        return None

    fg = grid != BG
    r0, c0, r1, c1 = _bbox(fg)
    tight = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.array_equal(tight != BG, mask):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if max(out.shape) > 30:
        return None
    if out.shape[0] != int(mask.sum()) or out.shape[1] != int(mask.sum()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
