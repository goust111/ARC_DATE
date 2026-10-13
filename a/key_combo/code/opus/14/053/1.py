from __future__ import annotations

import math
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    return int(cols[0])


def _pick_offset(total: int, size: int) -> int:
    if total == size:
        return 0
    if RNG.random() < 0.35:
        return int(RNG.choice([0, total - size]))
    return int(RNG.integers(0, total - size + 1))


def _choose_bbox_dims(left_h: int, left_w: int) -> tuple[int, int]:
    for _ in range(200):
        bbox_h = int(RNG.integers(4, min(left_h, 10) + 1))
        bbox_w = int(RNG.integers(4, min(left_w, 10) + 1))
        area = bbox_h * bbox_w
        interior = max(0, (bbox_h - 2) * (bbox_w - 2))
        min_missing = max(4, math.ceil(area * 0.20))
        max_missing = min(interior, math.floor(area * 0.40))
        if min_missing <= max_missing:
            return bbox_h, bbox_w
    raise RuntimeError("failed to choose a valid rectangle size")


def _interior_cells(bbox_h: int, bbox_w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, bbox_h - 1) for c in range(1, bbox_w - 1)]


def _sample_missing_mask(bbox_h: int, bbox_w: int, count: int) -> np.ndarray:
    interior = _interior_cells(bbox_h, bbox_w)
    interior_set = set(interior)
    mode = str(RNG.choice(["scatter", "cluster", "bands"]))

    if mode == "scatter":
        chosen = set(
            map(
                tuple,
                RNG.choice(np.array(interior, dtype=int), size=count, replace=False).tolist(),
            )
        )
    elif mode == "cluster":
        seed = interior[int(RNG.integers(0, len(interior)))]
        chosen = {seed}
        frontier: deque[tuple[int, int]] = deque([seed])
        while frontier and len(chosen) < count:
            r, c = frontier.popleft()
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            RNG.shuffle(neighbors)
            for nr, nc in neighbors:
                if (nr, nc) in interior_set and (nr, nc) not in chosen:
                    chosen.add((nr, nc))
                    frontier.append((nr, nc))
                    if len(chosen) == count:
                        break
            if not frontier and len(chosen) < count:
                frontier.append(tuple(RNG.choice(np.array(list(chosen)), axis=0)))
        if len(chosen) < count:
            remain = list(interior_set - chosen)
            extra = RNG.choice(np.array(remain, dtype=int), size=count - len(chosen), replace=False)
            chosen.update(map(tuple, extra.tolist()))
    else:
        chosen = set()
        rows = list(range(1, bbox_h - 1))
        cols = list(range(1, bbox_w - 1))
        if RNG.random() < 0.5:
            picked_rows = RNG.choice(rows, size=int(RNG.integers(1, len(rows) + 1)), replace=False)
            for r in picked_rows.tolist():
                for c in cols:
                    chosen.add((int(r), int(c)))
        else:
            picked_cols = RNG.choice(cols, size=int(RNG.integers(1, len(cols) + 1)), replace=False)
            for c in picked_cols.tolist():
                for r in rows:
                    chosen.add((int(r), int(c)))
        chosen = list(chosen)
        RNG.shuffle(chosen)
        chosen = set(chosen[:count])
        if len(chosen) < count:
            remain = list(interior_set - chosen)
            extra = RNG.choice(np.array(remain, dtype=int), size=count - len(chosen), replace=False)
            chosen.update(map(tuple, extra.tolist()))

    row_support = {r for r, _ in chosen}
    col_support = {c for _, c in chosen}
    if len(row_support) < 2 or len(col_support) < 2:
        return _sample_missing_mask(bbox_h, bbox_w, count)

    mask = np.zeros((bbox_h, bbox_w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _generate_right_mask(height: int, width: int) -> np.ndarray:
    mode = str(RNG.choice(["random", "row", "col", "blocks"]))
    if mode == "random":
        p = float(RNG.uniform(0.28, 0.72))
        mask = RNG.random((height, width)) < p
    elif mode == "row":
        period = int(RNG.integers(2, 5))
        phase = int(RNG.integers(0, period))
        mask = ((np.arange(height)[:, None] + phase) % period) == 0
        mask = np.broadcast_to(mask, (height, width)).copy()
        mask ^= RNG.random((height, width)) < 0.18
    elif mode == "col":
        period = int(RNG.integers(2, 5))
        phase = int(RNG.integers(0, period))
        mask = ((np.arange(width)[None, :] + phase) % period) == 0
        mask = np.broadcast_to(mask, (height, width)).copy()
        mask ^= RNG.random((height, width)) < 0.18
    else:
        mask = np.zeros((height, width), dtype=bool)
        block_count = int(RNG.integers(1, 4))
        for _ in range(block_count):
            block_h = int(RNG.integers(1, max(2, height // 2) + 1))
            block_w = int(RNG.integers(1, max(2, width // 2) + 1))
            r0 = int(RNG.integers(0, height - block_h + 1))
            c0 = int(RNG.integers(0, width - block_w + 1))
            mask[r0 : r0 + block_h, c0 : c0 + block_w] = True
        mask ^= RNG.random((height, width)) < 0.10
    return mask.astype(int)


def _build_example() -> np.ndarray | None:
    height = int(RNG.integers(9, 17))
    left_w = int(RNG.integers(7, 15))
    bbox_h, bbox_w = _choose_bbox_dims(height, left_w)
    area = bbox_h * bbox_w
    interior = (bbox_h - 2) * (bbox_w - 2)

    must_leave_teal_inside = interior >= 5 and RNG.random() < 0.75
    min_missing = max(4, math.ceil(area * 0.20))
    max_missing = min(interior, math.floor(area * 0.40))
    if must_leave_teal_inside:
        max_missing = min(max_missing, interior - 1)
    if min_missing > max_missing:
        return None
    missing_count = int(RNG.integers(min_missing, max_missing + 1))

    r0 = _pick_offset(height, bbox_h)
    c0 = _pick_offset(left_w, bbox_w)

    left = np.zeros((height, left_w), dtype=int)
    rect = np.full((bbox_h, bbox_w), TEAL, dtype=int)
    missing_mask = _sample_missing_mask(bbox_h, bbox_w, missing_count)
    rect[missing_mask] = BLACK
    left[r0 : r0 + bbox_h, c0 : c0 + bbox_w] = rect

    right = _generate_right_mask(height, left_w)
    missing_global = np.zeros_like(left, dtype=bool)
    missing_global[r0 : r0 + bbox_h, c0 : c0 + bbox_w] = missing_mask

    overlap = int(np.sum((right == BLUE) & missing_global))
    non_overlap = int(np.sum((right == BLACK) & missing_global))
    if overlap < 2 or non_overlap < 2:
        return None

    crop_right = right[r0 : r0 + bbox_h, c0 : c0 + bbox_w]
    crop_left = left[r0 : r0 + bbox_h, c0 : c0 + bbox_w]
    if np.sum((crop_left == TEAL) & (crop_right == BLUE)) < 1:
        return None
    if np.sum((crop_left == TEAL) & (crop_right == BLACK)) < 1:
        return None
    if np.sum(right == BLUE) < 4 or np.sum(right == BLACK) < 4:
        return None

    separator = np.full((height, 1), GREY, dtype=int)
    grid = np.concatenate([left, separator, right], axis=1)
    if grid.shape[1] > 30:
        return None

    expected = rect.copy()
    expected[missing_mask & (crop_right == BLUE)] = RED
    expected[missing_mask & (crop_right == BLACK)] = GREEN
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if not _semantic_check(grid):
        return None
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        sep = _find_separator(grid)
        left = grid[:, :sep]
        right = grid[:, sep + 1 :]
        if left.shape != right.shape:
            return False

        box = _bbox(left == TEAL)
        if box is None:
            return False
        r0, c0, r1, c1 = box
        crop_left = left[r0 : r1 + 1, c0 : c1 + 1]
        crop_right = right[r0 : r1 + 1, c0 : c1 + 1]
        missing = crop_left == BLACK

        if crop_left.shape[0] < 4 or crop_left.shape[1] < 4:
            return False
        ratio = float(missing.mean())
        if not (0.20 <= ratio <= 0.40):
            return False
        if int(missing.sum()) < 4:
            return False
        if int(np.sum(missing & (crop_right == BLUE))) < 2:
            return False
        if int(np.sum(missing & (crop_right == BLACK))) < 2:
            return False
        if not np.all(crop_left[0, :] == TEAL):
            return False
        if not np.all(crop_left[-1, :] == TEAL):
            return False
        if not np.all(crop_left[:, 0] == TEAL):
            return False
        if not np.all(crop_left[:, -1] == TEAL):
            return False

        teal_mask_values = crop_right[crop_left == TEAL]
        if teal_mask_values.size >= 2:
            if BLUE not in teal_mask_values or BLACK not in teal_mask_values:
                return False

        out = main(grid)
        if out.shape != crop_left.shape:
            return False
        if not np.all(out[crop_left == TEAL] == TEAL):
            return False
        if not np.all(out[missing & (crop_right == BLUE)] == RED):
            return False
        if not np.all(out[missing & (crop_right == BLACK)] == GREEN):
            return False
        if int(np.sum(out == RED)) < 2 or int(np.sum(out == GREEN)) < 2:
            return False
        return True
    except AssertionError:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the cropped teal rectangle by using the right-hand bitmask to classify each missing cell as red or green.

    Input structure:
    - The grid has a black background and one full-height grey separator column.
    - Left of the separator is an incomplete teal rectangle: its teal pixels define a minimum bounding box, and black cells inside that box are missing rectangle cells.
    - Right of the separator is an equally sized blue/black bitmask aligned cell-by-cell with the left half.

    Rule:
    - Split the grid at the grey separator and find the bounding box of all teal pixels in the left half.
    - Inside that bounding box, treat every black cell as a missing rectangle cell.
    - For each missing cell, inspect the corresponding aligned cell in the right-half bitmask: blue fills that left cell red, black fills that left cell green.
    - Return only the left-half bounding box crop with original teal cells preserved and missing cells filled.

    Key insight:
    - The teal pixels locate which cells belong to the incomplete rectangle, while the right-hand bitmask does not get overlaid directly; it acts as a per-cell binary classifier for the missing cells only.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "left and right panels must match"

    box = _bbox(left == TEAL)
    assert box is not None, "expected teal rectangle pixels in the left panel"
    r0, c0, r1, c1 = box

    rect = left[r0 : r1 + 1, c0 : c1 + 1].copy()
    aligned_mask = right[r0 : r1 + 1, c0 : c1 + 1]
    missing = rect == BLACK

    rect[missing & (aligned_mask == BLUE)] = RED
    rect[missing & (aligned_mask == BLACK)] = GREEN
    return rect


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bitmask-gated rectangle completion example")
