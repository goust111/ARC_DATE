from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_mirror_xor_diff",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
components_mask = _ARC_UTILS.components_mask
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location


GRID_H_RANGE = (10, 14)
HALF_W_RANGE = (6, 8)
FILL_RATIO_MIN = 0.15
FILL_RATIO_MAX = 0.40
TARGET_FILL_RATIO_RANGE = (0.18, 0.36)
DIFF_COUNT_RANGE = (2, 5)
GEN_RETRIES = 400
BRANCH_CONNECTED = "connected"
BRANCH_DISCONNECTED = "disconnected"


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Mirror-align the right half to the left half and reveal only their XOR differences.

    Input structure:
    - The grid is a black-background rectangle with an even number of columns.
    - Blue foreground pixels occupy the left half and right half of the grid.
    - The two halves are almost left-right mirror matches, except for a few mismatched pixels.

    Rule:
    - Split the grid at the vertical midpoint into left half L and right half R.
    - Flip R left-to-right so it is aligned in the same coordinate frame as L.
    - Compare the foreground masks of L and flipped R cell by cell.
    - Output a half-width grid where mismatched aligned cells become red and matched cells become black.

    Key insight:
    - The right half is not compared in its original orientation; it must be mirrored first, and only then does XOR isolate the symmetry defects.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    height, width = map(int, grid.shape)
    assert width % 2 == 0, "expected an even number of columns"

    fg_colors = sorted(set(map(int, np.unique(grid))) - {Color.BLACK})
    assert len(fg_colors) <= 1, "expected at most one foreground color"

    mid = width // 2
    left = grid[:, :mid] != Color.BLACK
    right = grid[:, mid:] != Color.BLACK
    diff = np.logical_xor(left, np.fliplr(right))
    return np.where(diff, Color.RED, Color.BLACK).astype(int)


def _fill_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def _bbox_spans_enough(mask: np.ndarray) -> bool:
    box = bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) >= 4 and (c1 - c0 + 1) >= 3


def _neighbors8(height: int, width: int, row: int, col: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = row + dr
            cc = col + dc
            if 0 <= rr < height and 0 <= cc < width:
                out.append((rr, cc))
    return out


def _grow_blob(height: int, width: int, target_cells: int) -> np.ndarray | None:
    if height <= 0 or width <= 0 or target_cells <= 0 or target_cells > height * width:
        return None

    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < target_cells:
        frontier = set()
        for row, col in np.argwhere(mask):
            for rr, cc in _neighbors8(height, width, int(row), int(col)):
                if not mask[rr, cc]:
                    frontier.add((rr, cc))
        if not frontier:
            return None

        row, col = random.choice(tuple(frontier))
        mask[row, col] = True

        if random.random() < 0.18:
            extras = [pos for pos in _neighbors8(height, width, row, col) if not mask[pos]]
            random.shuffle(extras)
            for rr, cc in extras[:1]:
                if int(mask.sum()) >= target_cells:
                    break
                mask[rr, cc] = True

    return mask


def _make_connected_left(height: int, half_width: int) -> np.ndarray | None:
    for _ in range(80):
        area = height * half_width
        target = random.randint(
            max(1, int(np.ceil(TARGET_FILL_RATIO_RANGE[0] * area))),
            max(1, int(np.floor(TARGET_FILL_RATIO_RANGE[1] * area))),
        )
        mask = _grow_blob(height, half_width, target)
        if mask is None:
            continue
        if not _bbox_spans_enough(mask):
            continue
        if not (FILL_RATIO_MIN <= _fill_ratio(mask) <= FILL_RATIO_MAX):
            continue
        if len(components_mask(mask, connectivity=8)) != 1:
            continue
        return mask
    return None


def _make_disconnected_left(height: int, half_width: int) -> np.ndarray | None:
    for _ in range(120):
        grid = np.full((height, half_width), Color.BLACK, dtype=int)
        num_parts = random.randint(2, 3)
        placed = 0

        for _part_idx in range(num_parts):
            max_part_h = max(3, height - 2)
            max_part_w = max(2, half_width - 1)
            if max_part_h < 3 or max_part_w < 2:
                break
            part_h = random.randint(3, max_part_h)
            part_w = random.randint(2, max_part_w)
            part_area = part_h * part_w
            part_target = random.randint(
                max(2, int(np.ceil(0.38 * part_area))),
                max(2, int(np.floor(0.68 * part_area))),
            )
            part_mask = _grow_blob(part_h, part_w, part_target)
            if part_mask is None:
                continue
            part = np.where(part_mask, Color.BLUE, Color.BLACK).astype(int)
            try:
                top, left = random_free_location(
                    grid,
                    part,
                    transparent=Color.BLACK,
                    border=0,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            paste(grid, part, top, left, transparent=Color.BLACK, clip=False)
            placed += 1

        if placed < 2:
            continue

        mask = grid == Color.BLUE
        comps = components_mask(mask, connectivity=8)
        if len(comps) < 2:
            continue
        if not _bbox_spans_enough(mask):
            continue
        if not (FILL_RATIO_MIN <= _fill_ratio(mask) <= FILL_RATIO_MAX):
            continue
        return mask
    return None


def _sample_diff_positions(height: int, half_width: int) -> list[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(height) for c in range(half_width)]
    for _ in range(80):
        diff_count = random.randint(*DIFF_COUNT_RANGE)
        positions = random.sample(all_cells, diff_count)
        rows = {r for r, _ in positions}
        cols = {c for _, c in positions}
        if len(rows) < 2 or len(cols) < 2:
            continue
        ok = True
        for i, (r1, c1) in enumerate(positions):
            for r2, c2 in positions[i + 1 :]:
                if abs(r1 - r2) + abs(c1 - c2) <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return positions
    return None


def _build_left_mask(height: int, half_width: int) -> tuple[np.ndarray, str] | None:
    branches = [BRANCH_CONNECTED, BRANCH_DISCONNECTED]
    random.shuffle(branches)
    for branch in branches:
        mask = _make_connected_left(height, half_width) if branch == BRANCH_CONNECTED else _make_disconnected_left(height, half_width)
        if mask is None:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask, branch
    return None


def _compose_grid(left_mask: np.ndarray, diff_positions: list[tuple[int, int]]) -> np.ndarray:
    aligned_right = left_mask.copy()
    for r, c in diff_positions:
        aligned_right[r, c] = ~aligned_right[r, c]

    right_half = np.fliplr(aligned_right)
    full = np.zeros((left_mask.shape[0], left_mask.shape[1] * 2), dtype=int)
    full[:, : left_mask.shape[1]] = np.where(left_mask, Color.BLUE, Color.BLACK)
    full[:, left_mask.shape[1] :] = np.where(right_half, Color.BLUE, Color.BLACK)
    return full


def _semantic_checks(grid: np.ndarray, branch: str) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    height, width = map(int, grid.shape)
    if height > 30 or width > 30 or width % 2 != 0:
        return False
    if not set(map(int, np.unique(grid))).issubset({Color.BLACK, Color.BLUE}):
        return False

    fill_ratio = float(np.count_nonzero(grid == Color.BLUE)) / float(grid.size)
    if not (FILL_RATIO_MIN <= fill_ratio <= FILL_RATIO_MAX):
        return False

    mid = width // 2
    left_mask = grid[:, :mid] == Color.BLUE
    right_mask = grid[:, mid:] == Color.BLUE
    left_components = len(components_mask(left_mask, connectivity=8))
    if branch == BRANCH_CONNECTED and left_components != 1:
        return False
    if branch == BRANCH_DISCONNECTED and left_components < 2:
        return False

    expected = np.logical_xor(left_mask, np.fliplr(right_mask))
    wrong_shortcut = np.logical_xor(left_mask, right_mask)
    if np.array_equal(expected, wrong_shortcut):
        return False

    diff_coords = np.argwhere(expected)
    if not (DIFF_COUNT_RANGE[0] <= len(diff_coords) <= DIFF_COUNT_RANGE[1]):
        return False
    if len({int(r) for r, _ in diff_coords}) < 2:
        return False
    if len({int(c) for _, c in diff_coords}) < 2:
        return False

    output = main(grid)
    if output.shape != (height, mid):
        return False
    if not np.array_equal(output == Color.RED, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    half_width = random.randint(*HALF_W_RANGE)

    built = _build_left_mask(height, half_width)
    if built is None:
        return None
    left_mask, branch = built

    diff_positions = _sample_diff_positions(height, half_width)
    if diff_positions is None:
        return None

    grid = _compose_grid(left_mask, diff_positions)
    if not _semantic_checks(grid, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-xor-difference input")
