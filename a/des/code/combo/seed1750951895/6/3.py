from __future__ import annotations

import random

import numpy as np


AZURE = 8
BACKGROUND = 0
GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (10, 24)
FOREGROUND_CHOICES = [1, 2, 3, 4, 5, 6, 7, 9]
THICK_OBJECT_COUNT_RANGE = (1, 4)
THIN_OBJECT_COUNT_RANGE = (0, 4)
GENERATION_ATTEMPTS = 300


def _has_cross_center(mask: np.ndarray, r: int, c: int) -> bool:
    return bool(
        mask[r, c]
        and 0 < r < mask.shape[0] - 1
        and 0 < c < mask.shape[1] - 1
        and mask[r - 1, c]
        and mask[r + 1, c]
        and mask[r, c - 1]
        and mask[r, c + 1]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every foreground pixel whose 4-neighbors are all foreground, and
    also recolor its four orthogonal neighbors, to azure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg = g != BACKGROUND
    mark = np.zeros_like(fg, dtype=bool)
    H, W = g.shape

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _has_cross_center(fg, r, c):
                mark[r, c] = True
                mark[r - 1, c] = True
                mark[r + 1, c] = True
                mark[r, c - 1] = True
                mark[r, c + 1] = True

    out = g.copy()
    out[mark] = AZURE
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    H, W = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(H, H - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(W, W - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(H, H + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(W, W + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _can_place(shape: np.ndarray, occupied: np.ndarray) -> bool:
    if shape.shape != occupied.shape:
        return False
    return not bool((_dilate8(shape) & occupied).any())


def _random_rect_shape(H: int, W: int) -> np.ndarray | None:
    h = random.randint(3, min(8, H - 2))
    w = random.randint(3, min(8, W - 2))
    r0 = random.randint(1, H - h - 1)
    c0 = random.randint(1, W - w - 1)
    shape = np.zeros((H, W), dtype=bool)
    shape[r0 : r0 + h, c0 : c0 + w] = True
    return shape


def _random_orthogonal_union(H: int, W: int) -> np.ndarray | None:
    shape = np.zeros((H, W), dtype=bool)
    base_h = random.randint(3, min(7, H - 2))
    base_w = random.randint(3, min(7, W - 2))
    r0 = random.randint(1, H - base_h - 1)
    c0 = random.randint(1, W - base_w - 1)
    shape[r0 : r0 + base_h, c0 : c0 + base_w] = True

    arm_count = random.randint(1, 3)
    for _ in range(arm_count):
        vertical = random.random() < 0.5
        if vertical:
            arm_h = random.randint(3, min(8, H - 2))
            arm_w = random.randint(2, 4)
            anchor_r = random.randint(r0, r0 + base_h - 1)
            start_r_lo = max(1, anchor_r - arm_h + 1)
            start_r_hi = min(anchor_r, H - arm_h - 1)
            if start_r_lo > start_r_hi:
                continue
            arm_r0 = random.randint(start_r_lo, start_r_hi)
            overlap_c = random.randint(c0, c0 + base_w - 1)
            arm_c0_lo = max(1, overlap_c - arm_w + 1)
            arm_c0_hi = min(overlap_c, W - arm_w - 1)
            if arm_c0_lo > arm_c0_hi:
                continue
            arm_c0 = random.randint(arm_c0_lo, arm_c0_hi)
            shape[arm_r0 : arm_r0 + arm_h, arm_c0 : arm_c0 + arm_w] = True
        else:
            arm_h = random.randint(2, 4)
            arm_w = random.randint(3, min(8, W - 2))
            anchor_c = random.randint(c0, c0 + base_w - 1)
            start_c_lo = max(1, anchor_c - arm_w + 1)
            start_c_hi = min(anchor_c, W - arm_w - 1)
            if start_c_lo > start_c_hi:
                continue
            arm_c0 = random.randint(start_c_lo, start_c_hi)
            overlap_r = random.randint(r0, r0 + base_h - 1)
            arm_r0_lo = max(1, overlap_r - arm_h + 1)
            arm_r0_hi = min(overlap_r, H - arm_h - 1)
            if arm_r0_lo > arm_r0_hi:
                continue
            arm_r0 = random.randint(arm_r0_lo, arm_r0_hi)
            shape[arm_r0 : arm_r0 + arm_h, arm_c0 : arm_c0 + arm_w] = True
    return shape


def _random_blob_shape(H: int, W: int) -> np.ndarray | None:
    target = random.randint(12, min(48, (H - 2) * (W - 2) // 3))
    start = (random.randint(1, H - 2), random.randint(1, W - 2))
    cells = {start}
    frontier = set()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        frontier.add((start[0] + dr, start[1] + dc))

    while frontier and len(cells) < target:
        candidates = [p for p in frontier if 1 <= p[0] < H - 1 and 1 <= p[1] < W - 1]
        if not candidates:
            break
        weights = []
        for r, c in candidates:
            degree = sum((r + dr, c + dc) in cells for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
            weights.append((degree + 0.25) ** 3)
        r, c = random.choices(candidates, weights=weights, k=1)[0]
        frontier.discard((r, c))
        if (r, c) in cells:
            continue
        cells.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in cells:
                frontier.add((nr, nc))

    shape = np.zeros((H, W), dtype=bool)
    for r, c in cells:
        shape[r, c] = True

    if shape.sum() < 12:
        return None
    interior = 0
    for r, c in cells:
        if _has_cross_center(shape, r, c):
            interior += 1
    if interior == 0:
        return None
    return shape


def _sample_thick_shape(H: int, W: int) -> np.ndarray | None:
    builders = [_random_rect_shape, _random_orthogonal_union, _random_blob_shape]
    random.shuffle(builders)
    for builder in builders:
        shape = builder(H, W)
        if shape is None or shape.sum() == 0:
            continue
        if any(_has_cross_center(shape, r, c) for r, c in np.argwhere(shape)):
            return shape
    return None


def _random_segment_shape(H: int, W: int) -> np.ndarray | None:
    horizontal = random.random() < 0.5
    length = random.randint(2, min(7, (W - 2) if horizontal else (H - 2)))
    shape = np.zeros((H, W), dtype=bool)
    if horizontal:
        r = random.randint(1, H - 2)
        c0 = random.randint(1, W - length - 1)
        shape[r, c0 : c0 + length] = True
    else:
        c = random.randint(1, W - 2)
        r0 = random.randint(1, H - length - 1)
        shape[r0 : r0 + length, c] = True
    return shape


def _random_thin_l_shape(H: int, W: int) -> np.ndarray | None:
    h = random.randint(2, min(5, H - 2))
    w = random.randint(2, min(5, W - 2))
    r0 = random.randint(1, H - h - 1)
    c0 = random.randint(1, W - w - 1)
    shape = np.zeros((H, W), dtype=bool)
    if random.random() < 0.5:
        shape[r0 : r0 + h, c0] = True
        shape[r0 + h - 1, c0 : c0 + w] = True
    else:
        shape[r0, c0 : c0 + w] = True
        shape[r0 : r0 + h, c0 + w - 1] = True
    return shape


def _sample_thin_shape(H: int, W: int) -> np.ndarray | None:
    builders = [_random_segment_shape, _random_thin_l_shape]
    random.shuffle(builders)
    for builder in builders:
        shape = builder(H, W)
        if shape is None or shape.sum() == 0:
            continue
        if not any(_has_cross_center(shape, r, c) for r, c in np.argwhere(shape)):
            return shape
    return None


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    fg = random.choice(FOREGROUND_CHOICES)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    thick_target = random.randint(*THICK_OBJECT_COUNT_RANGE)
    thick_added = 0
    for _ in range(thick_target * 12):
        if thick_added >= thick_target:
            break
        shape = _sample_thick_shape(H, W)
        if shape is None or not _can_place(shape, occupied):
            continue
        grid[shape] = fg
        occupied |= shape
        thick_added += 1

    if thick_added == 0:
        return None

    thin_target = random.randint(*THIN_OBJECT_COUNT_RANGE)
    thin_added = 0
    for _ in range(max(1, thin_target) * 10):
        if thin_added >= thin_target:
            break
        shape = _sample_thin_shape(H, W)
        if shape is None or not _can_place(shape, occupied):
            continue
        grid[shape] = fg
        occupied |= shape
        thin_added += 1

    out = main(grid)
    changed = out != grid
    if not bool(changed.any()):
        return None

    # Ensure the example contains both transformed and untouched foreground often enough.
    fg_count = int((grid != BACKGROUND).sum())
    changed_count = int(changed.sum())
    if fg_count - changed_count <= 0 and thin_added == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
