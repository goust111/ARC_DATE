from __future__ import annotations

import random

import numpy as np


BG = 0
ORANGE = 7
AZURE = 8
MAROON = 9
FENCE_COLORS = (1, 2, 3, 4, 5, 6)

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FENCE_COUNT_RANGE = (2, 4)
FIT_SIZE_RANGE = (6, 12)
OVERFLOW_SIZE_RANGE = (4, 9)
PLACEMENT_PADDING = 2
GENERATION_ATTEMPTS = 400
PATCH_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200


def _count_gap_runs(line: np.ndarray) -> int:
    is_gap = np.asarray(line) == BG
    runs = 0
    prev_gap = False
    for value in is_gap:
        gap = bool(value)
        if gap and not prev_gap:
            runs += 1
        prev_gap = gap
    return runs


def _ring_capacity(height: int, width: int) -> int:
    top, bottom = 0, int(height) - 1
    left, right = 0, int(width) - 1
    layers = 0
    while top <= bottom and left <= right:
        layers += 1
        top += 1
        bottom -= 1
        left += 1
        right -= 1
    return layers


def _fence_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None

    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    rr = coords[:, 0]
    cc = coords[:, 1]
    on_perimeter = (rr == r0) | (rr == r1) | (cc == c0) | (cc == c1)
    if not bool(np.all(on_perimeter)):
        return None

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(grid[r, c]) != int(color) for r, c in corners):
        return None

    if r1 - r0 > 1 and c1 - c0 > 1 and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == int(color)):
        return None

    return r0, c0, r1, c1


def _gap_count(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    top = grid[r0, c0 + 1 : c1]
    bottom = grid[r1, c0 + 1 : c1]
    left = grid[r0 + 1 : r1, c0]
    right = grid[r0 + 1 : r1, c1]
    return (
        _count_gap_runs(top)
        + _count_gap_runs(bottom)
        + _count_gap_runs(left)
        + _count_gap_runs(right)
    )


def _paint_rings(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    fence_color: int,
    layers: int,
) -> None:
    r0, c0, r1, c1 = bbox
    top, bottom = r0 + 1, r1 - 1
    left, right = c0 + 1, c1 - 1
    palette = (int(fence_color), ORANGE, AZURE, MAROON)

    for layer in range(int(layers)):
        if top > bottom or left > right:
            break
        color = int(palette[layer % len(palette)])
        out[top, left : right + 1] = color
        out[bottom, left : right + 1] = color
        out[top : bottom + 1, left] = color
        out[top : bottom + 1, right] = color
        top += 1
        bottom -= 1
        left += 1
        right -= 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular fence with concentric rings whose depth equals
    the total number of single-pixel fence gaps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = sorted(int(color) for color in np.unique(g) if int(color) != BG)
    for color in colors:
        bbox = _fence_bbox(g, color)
        if bbox is None:
            continue
        gaps = _gap_count(g, bbox)
        if gaps <= 0:
            continue
        _paint_rings(out, bbox, color, gaps)

    return out


def _side_capacity(length: int) -> int:
    return (int(length) + 1) // 2


def _sample_nonadjacent_offsets(length: int, count: int, rng: random.Random) -> list[int]:
    if count == 0:
        return []
    slots = list(range(int(length)))
    for _ in range(PATCH_ATTEMPTS):
        chosen = sorted(rng.sample(slots, count))
        if all(b - a > 1 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return list(range(0, 2 * count, 2))


def _distribute_gaps(
    total_gaps: int,
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    capacities = [
        _side_capacity(inner_w),
        _side_capacity(inner_w),
        _side_capacity(inner_h),
        _side_capacity(inner_h),
    ]
    counts = [0, 0, 0, 0]
    for _ in range(int(total_gaps)):
        available = [i for i, cap in enumerate(capacities) if counts[i] < cap]
        assert available
        side = rng.choice(available)
        counts[side] += 1
    return tuple(counts)


def _build_patch(
    height: int,
    width: int,
    color: int,
    total_gaps: int,
    rng: random.Random,
) -> np.ndarray:
    patch = np.full((int(height), int(width)), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    inner_h = int(height) - 2
    inner_w = int(width) - 2
    top_n, bottom_n, left_n, right_n = _distribute_gaps(total_gaps, inner_h, inner_w, rng)

    for offset in _sample_nonadjacent_offsets(inner_w, top_n, rng):
        patch[0, 1 + offset] = BG
    for offset in _sample_nonadjacent_offsets(inner_w, bottom_n, rng):
        patch[-1, 1 + offset] = BG
    for offset in _sample_nonadjacent_offsets(inner_h, left_n, rng):
        patch[1 + offset, 0] = BG
    for offset in _sample_nonadjacent_offsets(inner_h, right_n, rng):
        patch[1 + offset, -1] = BG

    return patch


def _sample_patch(mode: str, color: int, rng: random.Random) -> np.ndarray | None:
    low, high = FIT_SIZE_RANGE if mode == "fit" else OVERFLOW_SIZE_RANGE
    for _ in range(PATCH_ATTEMPTS):
        height = rng.randint(low, high)
        width = rng.randint(low, high)
        inner_h = height - 2
        inner_w = width - 2
        ring_cap = _ring_capacity(inner_h, inner_w)
        max_gaps = 2 * _side_capacity(inner_h) + 2 * _side_capacity(inner_w)
        if ring_cap <= 0 or max_gaps <= 0:
            continue

        if mode == "overflow":
            if max_gaps <= ring_cap:
                continue
            total_gaps = rng.randint(ring_cap + 1, max_gaps)
        else:
            total_gaps = rng.randint(1, min(ring_cap, max_gaps))

        patch = _build_patch(height, width, color, total_gaps, rng)
        if _gap_count(patch, (0, 0, height - 1, width - 1)) != total_gaps:
            continue
        return patch
    return None


def _boxes_separated(
    bbox: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
) -> bool:
    r0, c0, r1, c1 = bbox
    for pr0, pc0, pr1, pc1 in placed:
        if not (
            r1 + PLACEMENT_PADDING < pr0
            or pr1 + PLACEMENT_PADDING < r0
            or c1 + PLACEMENT_PADDING < pc0
            or pc1 + PLACEMENT_PADDING < c0
        ):
            return False
    return True


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    placed: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> bool:
    grid_h, grid_w = map(int, grid.shape)
    patch_h, patch_w = map(int, patch.shape)
    if patch_h > grid_h or patch_w > grid_w:
        return False

    max_r = grid_h - patch_h
    max_c = grid_w - patch_w
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = rng.randint(0, max_r)
        c0 = rng.randint(0, max_c)
        bbox = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
        if not _boxes_separated(bbox, placed):
            continue
        mask = patch != BG
        grid[r0 : r0 + patch_h, c0 : c0 + patch_w][mask] = patch[mask]
        placed.append(bbox)
        return True
    return False


def _sample_modes(count: int, rng: random.Random) -> list[str]:
    modes = ["fit", "overflow"]
    while len(modes) < int(count):
        modes.append("fit" if rng.random() < 0.5 else "overflow")
    rng.shuffle(modes)
    return modes[:count]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    fence_count = rng.randint(*FENCE_COUNT_RANGE)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    colors = rng.sample(FENCE_COLORS, fence_count)
    modes = _sample_modes(fence_count, rng)

    patches: list[np.ndarray] = []
    for color, mode in zip(colors, modes):
        patch = _sample_patch(mode, int(color), rng)
        if patch is None:
            return None
        patches.append(patch)

    order = sorted(range(fence_count), key=lambda i: int(patches[i].size), reverse=True)
    placed: list[tuple[int, int, int, int]] = []
    for idx in order:
        if not _place_patch(grid, patches[idx], placed, rng):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
