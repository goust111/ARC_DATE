from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

MAX_GENERATION_ATTEMPTS = 400


def _marker_rows(region: np.ndarray) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    width: int | None = None
    for r in range(region.shape[0]):
        blue_cols = np.flatnonzero(region[r] == BLUE)
        teal_cols = np.flatnonzero(region[r] == TEAL)
        if len(blue_cols) == 0 and len(teal_cols) == 0:
            continue

        assert len(blue_cols) == 1 and len(teal_cols) == 1, "each marked row needs one blue and one teal marker"
        left = int(blue_cols[0])
        right = int(teal_cols[0])
        assert left < right - 1, "markers must bracket at least one cell"
        segment = region[r, left + 1 : right]
        assert not np.isin(segment, [BLUE, TEAL, GREY]).any(), "segments may only contain data colors"
        if width is None:
            width = int(segment.size)
        else:
            assert int(segment.size) == width, "all bracketed segments in one region must share a width"
        rows.append(segment.copy())

    assert rows, "expected at least one marked row in each region"
    return rows


def _extract_mask(region: np.ndarray) -> np.ndarray:
    return np.vstack(_marker_rows(region))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract two marker-bracketed row stacks from opposite sides of a grey divider and NOR them.

    Input structure:
    - The grid has a black background and one full-height grey vertical separator column.
    - On each side of the separator, some rows contain exactly one blue marker and one teal marker, with red/black cells between them.
    - Non-marker rows and marker-row cells outside the bracket span contain only red or black noise.

    Rule:
    - Split the grid at the grey separator.
    - In each half, scan top to bottom; every row containing one blue and one teal marker contributes the cells strictly between them.
    - Stack the extracted rows from the left half into one mask and those from the right half into another mask, then treat red as 1 and black as 0.
    - Output teal exactly where both stacked masks are 0; output black everywhere else.

    Key insight:
    - The NOR operands are not the visible half-grids themselves but the two compact masks created by independently extracting the bracketed row segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) == 1, "expected exactly one full-height grey separator"
    sep = sep_cols[0]
    assert 0 < sep < g.shape[1] - 1, "separator cannot be on the border"

    left_mask = _extract_mask(g[:, :sep])
    right_mask = _extract_mask(g[:, sep + 1 :])
    assert left_mask.shape == right_mask.shape, "the two extracted masks must align"

    out = np.zeros(left_mask.shape, dtype=int)
    inactive = (left_mask != RED) & (right_mask != RED)
    out[inactive] = TEAL
    return out


def _random_red_black(shape: tuple[int, int], red_prob: float) -> np.ndarray:
    return (np.random.random(shape) < red_prob).astype(int) * RED


def _choose_rows(height: int, count: int) -> tuple[list[int], list[int]]:
    all_rows = list(range(height))
    left_rows = sorted(random.sample(all_rows, count))

    min_overlap = max(0, 2 * count - height)
    max_overlap = count - 1
    branch = random.choice(["low", "medium", "high"])
    if branch == "low":
        target_lo, target_hi = min_overlap, min(min_overlap + 1, max_overlap)
    elif branch == "high":
        target_lo, target_hi = max(min_overlap, count - 2), max_overlap
    else:
        mid = count // 2
        target_lo = max(min_overlap, mid - 1)
        target_hi = min(max_overlap, mid + 1)
    overlap = random.randint(target_lo, target_hi)

    common = random.sample(left_rows, overlap)
    pool = [r for r in all_rows if r not in left_rows]
    right_unique = random.sample(pool, count - overlap)
    right_rows = sorted(common + right_unique)

    assert left_rows != right_rows
    return left_rows, right_rows


def _choose_blue_cols(region_width: int, seg_width: int, count: int) -> list[int]:
    choices = list(range(region_width - seg_width - 1))
    cols = [random.choice(choices) for _ in range(count)]
    if count > 1 and len(set(cols)) == 1 and len(choices) > 1:
        idx = random.randrange(count)
        alternatives = [c for c in choices if c != cols[idx]]
        cols[idx] = random.choice(alternatives)
    return cols


def _build_masks(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    area = rows * cols
    cats = np.random.choice(4, size=area, p=[0.30, 0.25, 0.25, 0.20])
    forced = [0, 1, 2, 3]
    forced_idx = np.random.choice(area, size=4, replace=False)
    for idx, cat in zip(forced_idx, forced):
        cats[idx] = cat

    left = np.zeros(area, dtype=int)
    right = np.zeros(area, dtype=int)
    left[np.isin(cats, [1, 3])] = RED
    right[np.isin(cats, [2, 3])] = RED
    return left.reshape(rows, cols), right.reshape(rows, cols)


def _same_shape_as_half_nor(grid: np.ndarray) -> bool:
    sep = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)][0]
    return grid[:, :sep].shape == grid[:, sep + 1 :].shape


def _global_row_pair_shortcut(grid: np.ndarray) -> np.ndarray | None:
    sep = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)][0]
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    rows = []
    width: int | None = None
    for r in range(grid.shape[0]):
        left_blue = np.flatnonzero(left[r] == BLUE)
        left_teal = np.flatnonzero(left[r] == TEAL)
        right_blue = np.flatnonzero(right[r] == BLUE)
        right_teal = np.flatnonzero(right[r] == TEAL)
        if len(left_blue) == len(left_teal) == len(right_blue) == len(right_teal) == 1:
            lseg = left[r, int(left_blue[0]) + 1 : int(left_teal[0])]
            rseg = right[r, int(right_blue[0]) + 1 : int(right_teal[0])]
            if lseg.shape != rseg.shape:
                return None
            if width is None:
                width = int(lseg.size)
            elif int(lseg.size) != width:
                return None
            row = np.zeros(lseg.shape, dtype=int)
            row[(lseg != RED) & (rseg != RED)] = TEAL
            rows.append(row)
    if not rows:
        return None
    return np.vstack(rows)


def _oracle_checks(
    grid: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_rows: list[int],
    right_rows: list[int],
) -> bool:
    try:
        solved = main(grid)
    except AssertionError:
        return False

    expected = np.zeros(left_mask.shape, dtype=int)
    expected[(left_mask != RED) & (right_mask != RED)] = TEAL
    if not np.array_equal(solved, expected):
        return False

    if solved.shape != left_mask.shape or not np.any(solved == TEAL) or not np.any(solved == BLACK):
        return False

    if left_rows == right_rows:
        return False
    if len(set(left_rows)) != len(left_rows) or len(set(right_rows)) != len(right_rows):
        return False

    both_zero = np.any((left_mask == BLACK) & (right_mask == BLACK))
    left_only = np.any((left_mask == RED) & (right_mask == BLACK))
    right_only = np.any((left_mask == BLACK) & (right_mask == RED))
    both_one = np.any((left_mask == RED) & (right_mask == RED))
    if not (both_zero and left_only and right_only and both_one):
        return False

    only_left = np.where(left_mask == BLACK, TEAL, BLACK)
    only_right = np.where(right_mask == BLACK, TEAL, BLACK)
    if np.array_equal(only_left, expected) or np.array_equal(only_right, expected):
        return False

    shared_row_shortcut = _global_row_pair_shortcut(grid)
    if shared_row_shortcut is not None and np.array_equal(shared_row_shortcut, expected):
        return False

    if _same_shape_as_half_nor(grid):
        sep = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)][0]
        direct = np.zeros((grid.shape[0], sep), dtype=int)
        left_half = grid[:, :sep]
        right_half = grid[:, sep + 1 :]
        direct[(left_half != RED) & (right_half != RED)] = TEAL
        if direct.shape == expected.shape and np.array_equal(direct, expected):
            return False

    if len({tuple(row.tolist()) for row in expected}) < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    marker_count = random.randint(4, 6)
    seg_width = random.randint(4, 8)
    region_width = random.randint(max(seg_width + 3, 9), min(seg_width + 6, 13))
    height = random.randint(marker_count + 3, min(marker_count + 8, 16))

    left_rows, right_rows = _choose_rows(height, marker_count)
    left_cols = _choose_blue_cols(region_width, seg_width, marker_count)
    right_cols = _choose_blue_cols(region_width, seg_width, marker_count)

    left_mask, right_mask = _build_masks(marker_count, seg_width)

    left_region = _random_red_black((height, region_width), red_prob=random.uniform(0.20, 0.55))
    right_region = _random_red_black((height, region_width), red_prob=random.uniform(0.20, 0.55))

    for out_r, grid_r in enumerate(left_rows):
        blue = left_cols[out_r]
        teal = blue + seg_width + 1
        row = _random_red_black((1, region_width), red_prob=random.uniform(0.15, 0.60))[0]
        row[blue] = BLUE
        row[blue + 1 : teal] = left_mask[out_r]
        row[teal] = TEAL
        left_region[grid_r] = row

    for out_r, grid_r in enumerate(right_rows):
        blue = right_cols[out_r]
        teal = blue + seg_width + 1
        row = _random_red_black((1, region_width), red_prob=random.uniform(0.15, 0.60))[0]
        row[blue] = BLUE
        row[blue + 1 : teal] = right_mask[out_r]
        row[teal] = TEAL
        right_region[grid_r] = row

    grid = np.concatenate(
        [left_region, np.full((height, 1), GREY, dtype=int), right_region],
        axis=1,
    )

    if not _oracle_checks(grid, left_mask, right_mask, left_rows, right_rows):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-extraction NOR puzzle")
