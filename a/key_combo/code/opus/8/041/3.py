from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7
MARKER_COLORS = (BLUE, GREEN)

MIN_HEIGHT = 8
MAX_HEIGHT = 13
MIN_HALF_WIDTH = 7
MAX_HALF_WIDTH = 12
MASK_DENSITY_RANGE = (0.16, 0.34)
UNION_DENSITY_RANGE = (0.22, 0.55)
ALT_DIFF_MIN = 3
GEN_RETRIES = 500


def _find_separator(grid: np.ndarray) -> int:
    sep_cols = np.flatnonzero(np.all(grid == RED, axis=0))
    assert len(sep_cols) == 1, "expected one full-height red separator column"
    return int(sep_cols[0])


def _extract_halves(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep = _find_separator(grid)
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    assert left.shape == right.shape and left.size > 0, "expected equal-sized half-grids"
    return left, right


def _half_mask(half: np.ndarray) -> np.ndarray:
    mask = np.asarray(half == ORANGE, dtype=bool)
    mask[0, 0] = False
    return mask


def _merge_masks(left_mask: np.ndarray, right_mask: np.ndarray, flip_right: bool) -> np.ndarray:
    right = np.fliplr(right_mask) if flip_right else right_mask
    out = np.full(left_mask.shape, BLACK, dtype=int)
    out[left_mask | right] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the corner marker colors to decide whether the right mask must be mirrored before OR-merging the two halves.

    Input structure:
    - The grid has a black background and exactly one full-height red separator column.
    - The separator splits the canvas into two equal half-grids of the same size.
    - Each half contains an orange mask plus a fixed top-left marker pixel colored blue or green.

    Rule:
    - Read the top-left marker color from each half-grid.
    - Ignore both marker pixels as mask content; only orange cells belong to the masks.
    - If the marker colors match, OR the two masks directly; otherwise first mirror the right mask horizontally, then OR them.
    - Output a grid the size of one half-grid, coloring OR-active cells blue and all other cells black.

    Key insight:
    - The two marker colors do not describe the masks themselves; they encode a boolean instruction that changes the right mask's alignment before the logical merge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    left, right = _extract_halves(g)
    allowed = {BLACK, BLUE, RED, GREEN, ORANGE}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors present"
    assert int(left[0, 0]) in MARKER_COLORS and int(right[0, 0]) in MARKER_COLORS, "invalid marker colors"

    left_mask = _half_mask(left)
    right_mask = _half_mask(right)
    flip_right = int(left[0, 0]) != int(right[0, 0])
    return _merge_masks(left_mask, right_mask, flip_right)


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) != (0, 0):
                cells.add((nr, nc))
    return list(cells)


def _random_seed_cell(h: int, w: int) -> tuple[int, int]:
    edge_bias = [
        (0, random.randrange(1, w)),
        (random.randrange(h), w - 1),
        (h - 1, random.randrange(w)),
        (random.randrange(1, h), 0),
    ]
    if random.random() < 0.45:
        return random.choice(edge_bias)
    while True:
        cell = (random.randrange(h), random.randrange(w))
        if cell != (0, 0):
            return cell


def _grow_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed_count = 1 if random.random() < 0.7 else 2
    seeds = [_random_seed_cell(h, w)]
    if seed_count == 2 and target >= 6:
        seeds.append(_random_seed_cell(h, w))

    for r, c in seeds:
        mask[r, c] = True

    while int(mask.sum()) < target:
        choices = _frontier(mask)
        if not choices:
            break
        if random.random() < 0.35:
            edge_cells = [cell for cell in choices if 0 in cell or cell[0] == h - 1 or cell[1] == w - 1]
            pick_pool = edge_cells or choices
        else:
            pick_pool = choices
        r, c = random.choice(pick_pool)
        mask[r, c] = True
    return mask


def _add_spur(mask: np.ndarray) -> None:
    choices = _frontier(mask)
    if not choices or random.random() < 0.45:
        return
    h, w = mask.shape
    r, c = random.choice(choices)
    length = random.randint(1, max(1, min(h, w) // 3))
    dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
    for _ in range(length):
        if not (0 <= r < h and 0 <= c < w) or (r, c) == (0, 0):
            break
        mask[r, c] = True
        r += dr
        c += dc


def _random_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    lo = max(4, int(np.ceil(MASK_DENSITY_RANGE[0] * area)))
    hi = max(lo, int(np.floor(MASK_DENSITY_RANGE[1] * area)))
    target = random.randint(lo, hi)

    for _ in range(80):
        mask = _grow_connected_mask(h, w, target)
        _add_spur(mask)
        mask[0, 0] = False
        count = int(mask.sum())
        if count < lo or count > hi:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if len(rows) < 2 or len(cols) < 2:
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate mask")


def _markers_for_branch(match_colors: bool) -> tuple[int, int]:
    left = random.choice(MARKER_COLORS)
    if match_colors:
        return left, left
    right = GREEN if left == BLUE else BLUE
    return left, right


def _valid_pair(left_mask: np.ndarray, right_mask: np.ndarray) -> bool:
    if np.array_equal(left_mask, right_mask):
        return False
    if np.array_equal(left_mask, np.fliplr(right_mask)):
        return False
    if np.array_equal(left_mask, ~right_mask):
        return False

    direct = left_mask | right_mask
    flipped = left_mask | np.fliplr(right_mask)
    diff = int(np.count_nonzero(direct != flipped))
    if diff < ALT_DIFF_MIN:
        return False

    area = left_mask.size
    union_count = int(direct.sum())
    union_lo = int(np.ceil(UNION_DENSITY_RANGE[0] * area))
    union_hi = int(np.floor(UNION_DENSITY_RANGE[1] * area))
    if not (union_lo <= union_count <= union_hi):
        return False

    left_only = int(np.count_nonzero(left_mask & ~right_mask))
    right_only = int(np.count_nonzero(~left_mask & right_mask))
    overlap = int(np.count_nonzero(left_mask & right_mask))
    if min(left_only, right_only, overlap) < 1:
        return False
    return True


def _render_half(mask: np.ndarray, marker_color: int) -> np.ndarray:
    half = np.full(mask.shape, BLACK, dtype=int)
    half[mask] = ORANGE
    half[0, 0] = marker_color
    return half


def _semantic_check(grid: np.ndarray) -> tuple[bool, bool]:
    left, right = _extract_halves(grid)
    assert int(left[0, 0]) in MARKER_COLORS
    assert int(right[0, 0]) in MARKER_COLORS

    left_mask = _half_mask(left)
    right_mask = _half_mask(right)
    assert _valid_pair(left_mask, right_mask)

    expected_flip = int(left[0, 0]) != int(right[0, 0])
    actual = main(grid)
    expected = _merge_masks(left_mask, right_mask, expected_flip)
    alt = _merge_masks(left_mask, right_mask, not expected_flip)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(expected, alt)
    return bool(expected_flip), bool(not expected_flip)


def _gen_attempt() -> np.ndarray:
    h = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    w = random.randint(MIN_HALF_WIDTH, MAX_HALF_WIDTH)
    same_markers = random.random() < 0.5
    left_marker, right_marker = _markers_for_branch(same_markers)

    for _ in range(120):
        left_mask = _random_mask(h, w)
        right_mask = _random_mask(h, w)
        if not _valid_pair(left_mask, right_mask):
            continue

        left_half = _render_half(left_mask, left_marker)
        right_half = _render_half(right_mask, right_marker)
        grid = np.concatenate(
            [left_half, np.full((h, 1), RED, dtype=int), right_half],
            axis=1,
        )
        _semantic_check(grid)
        return grid
    raise RuntimeError("failed to build a valid input grid")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt()
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid conditional-mirror merge input")
