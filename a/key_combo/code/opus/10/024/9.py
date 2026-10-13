from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
RIGHT_PALETTE = (2, 3, 4, 6, 7, 8, 9)

HEIGHT_CHOICES = (6, 8, 10)
WIDTH_CHOICES = (4, 6, 8)
MAX_GENERATION_ATTEMPTS = 400


def _separator_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _split_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seps = _separator_columns(g)
    assert len(seps) == 2, "expected exactly two full-height grey separator columns"
    left = g[:, : seps[0]]
    middle = g[:, seps[0] + 1 : seps[1]]
    right = g[:, seps[1] + 1 :]
    assert left.shape == middle.shape == right.shape, "all three regions must have the same shape"
    assert left.size > 0, "regions must be non-empty"
    return left, middle, right


def _horizontal_axis_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)))


def _vertical_axis_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _unique_symmetry_axis(mask: np.ndarray) -> str | None:
    horizontal = _horizontal_axis_symmetric(mask)
    vertical = _vertical_axis_symmetric(mask)
    if horizontal == vertical:
        return None
    return "horizontal_axis" if horizontal else "vertical_axis"


def _vertical_duplication(region: np.ndarray) -> bool:
    height = region.shape[0]
    return height % 2 == 0 and np.array_equal(region[: height // 2, :], region[height // 2 :, :])


def _horizontal_duplication(region: np.ndarray) -> bool:
    width = region.shape[1]
    return width % 2 == 0 and np.array_equal(region[:, : width // 2], region[:, width // 2 :])


def _unique_duplication_axis(region: np.ndarray) -> str | None:
    vertical = _vertical_duplication(region)
    horizontal = _horizontal_duplication(region)
    if vertical == horizontal:
        return None
    return "vertical" if vertical else "horizontal"


def _symmetry_to_duplication(axis: str) -> str:
    if axis == "horizontal_axis":
        return "vertical"
    if axis == "vertical_axis":
        return "horizontal"
    raise AssertionError(f"unknown symmetry axis: {axis}")


def _first_half(region: np.ndarray, duplication_axis: str) -> np.ndarray:
    if duplication_axis == "vertical":
        assert _vertical_duplication(region), "right region must be two identical stacked halves"
        return region[: region.shape[0] // 2, :].copy()
    if duplication_axis == "horizontal":
        assert _horizontal_duplication(region), "right region must be two identical side-by-side halves"
        return region[:, : region.shape[1] // 2].copy()
    raise AssertionError(f"unknown duplication axis: {duplication_axis}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the overlap symmetry of two blue masks to choose which half of the colored tile to extract.

    Input structure:
    - Two full-height grey separator columns split the grid into equal-width left, middle, and right regions.
    - The left and middle regions are aligned blue/black bitmasks of the same size.
    - The right region is a colored pattern made from two identical halves, either stacked vertically or placed side-by-side.

    Rule:
    - Compute the pixelwise overlap of the left and middle masks: a cell is active only where both masks are blue.
    - If that overlap is symmetric about the horizontal midline, the right region must be a vertical duplication, so output its top half.
    - If that overlap is symmetric about the vertical midline, the right region must be a horizontal duplication, so output its left half.

    Key insight:
    - The overlap mask is not the answer; its unique symmetry axis is a control signal that selects how to cut the separate colored region.
    """
    left, middle, right = _split_regions(grid)
    overlap = (left == BLUE) & (middle == BLUE)
    symmetry_axis = _unique_symmetry_axis(overlap)
    assert symmetry_axis is not None, "overlap must have exactly one symmetry axis"

    duplication_axis = _symmetry_to_duplication(symmetry_axis)
    return _first_half(right, duplication_axis)


def _mask_from_region(region: np.ndarray) -> np.ndarray:
    return np.asarray(region == BLUE, dtype=bool)


def _make_symmetric_overlap(height: int, width: int, symmetry_axis: str) -> np.ndarray | None:
    min_cells = max(4, (height * width) // 6)
    max_cells = (height * width * 3) // 4

    for _ in range(200):
        if symmetry_axis == "horizontal_axis":
            top = np.random.rand(height // 2, width) < random.uniform(0.25, 0.6)
            mask = np.vstack([top, np.flipud(top)])
        else:
            left = np.random.rand(height, width // 2) < random.uniform(0.25, 0.6)
            mask = np.hstack([left, np.fliplr(left)])

        count = int(mask.sum())
        if count < min_cells or count > max_cells:
            continue
        if _unique_symmetry_axis(mask) != symmetry_axis:
            continue
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        if np.max(row_counts) == 0 or np.max(col_counts) == 0:
            continue
        return mask
    return None


def _make_mask_pair(overlap: np.ndarray, target_axis: str) -> tuple[np.ndarray, np.ndarray] | None:
    off_cells = np.argwhere(~overlap)
    overlap_count = int(overlap.sum())
    total = int(overlap.size)

    for _ in range(300):
        left = overlap.copy()
        middle = overlap.copy()
        assignments = np.random.choice(3, size=len(off_cells), p=(0.34, 0.33, 0.33))
        for idx, (r, c) in enumerate(off_cells):
            choice = int(assignments[idx])
            if choice == 1:
                left[r, c] = True
            elif choice == 2:
                middle[r, c] = True

        extra_left = int(left.sum()) - overlap_count
        extra_middle = int(middle.sum()) - overlap_count
        if extra_left < 2 or extra_middle < 2:
            continue
        if int(left.sum()) >= total - 1 or int(middle.sum()) >= total - 1:
            continue
        if _unique_symmetry_axis(left) is not None or _unique_symmetry_axis(middle) is not None:
            continue

        union_axis = _unique_symmetry_axis(left | middle)
        if union_axis == target_axis:
            continue

        if np.array_equal(left, middle):
            continue
        return left, middle
    return None


def _make_base_tile(height: int, width: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(height) for c in range(width)]

    for _ in range(200):
        color_count = random.randint(3, min(5, len(RIGHT_PALETTE)))
        colors = random.sample(RIGHT_PALETTE, color_count)
        tile = np.empty((height, width), dtype=int)

        random.shuffle(cells)
        for (r, c), color in zip(cells, colors):
            tile[r, c] = color
        for r, c in cells[len(colors) :]:
            tile[r, c] = random.choice(colors)

        if _vertical_duplication(tile) or _horizontal_duplication(tile):
            continue
        if len(np.unique(tile)) < 3:
            continue
        return tile
    return None


def _make_right_region(height: int, width: int, duplication_axis: str) -> np.ndarray | None:
    if duplication_axis == "vertical":
        tile = _make_base_tile(height // 2, width)
        if tile is None:
            return None
        right = np.vstack([tile, tile])
    else:
        tile = _make_base_tile(height, width // 2)
        if tile is None:
            return None
        right = np.hstack([tile, tile])

    if _unique_duplication_axis(right) != duplication_axis:
        return None
    return right


def _assemble_grid(left_mask: np.ndarray, middle_mask: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.where(left_mask, BLUE, BLACK).astype(int)
    middle = np.where(middle_mask, BLUE, BLACK).astype(int)
    sep = np.full((left.shape[0], 1), GREY, dtype=int)
    return np.concatenate([left, sep, middle, sep.copy(), right.astype(int)], axis=1)


def _semantic_validate(grid: np.ndarray) -> bool:
    try:
        left_region, middle_region, right_region = _split_regions(grid)
        left = _mask_from_region(left_region)
        middle = _mask_from_region(middle_region)
        overlap = left & middle

        symmetry_axis = _unique_symmetry_axis(overlap)
        if symmetry_axis is None:
            return False
        if _unique_symmetry_axis(left) is not None or _unique_symmetry_axis(middle) is not None:
            return False
        if _unique_symmetry_axis(left | middle) == symmetry_axis:
            return False
        if not np.any(left & ~overlap) or not np.any(middle & ~overlap):
            return False

        duplication_axis = _symmetry_to_duplication(symmetry_axis)
        if _unique_duplication_axis(right_region) != duplication_axis:
            return False

        expected = _first_half(right_region, duplication_axis)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        if max(grid.shape) > 30 or max(actual.shape) > 30:
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(HEIGHT_CHOICES)
    width = random.choice(WIDTH_CHOICES)
    symmetry_axis = random.choice(("horizontal_axis", "vertical_axis"))

    overlap = _make_symmetric_overlap(height, width, symmetry_axis)
    if overlap is None:
        return None

    mask_pair = _make_mask_pair(overlap, symmetry_axis)
    if mask_pair is None:
        return None
    left_mask, middle_mask = mask_pair

    duplication_axis = _symmetry_to_duplication(symmetry_axis)
    right = _make_right_region(height, width, duplication_axis)
    if right is None:
        return None

    grid = _assemble_grid(left_mask, middle_mask, right)
    return grid if _semantic_validate(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
