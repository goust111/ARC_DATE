from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, find_components, in_bounds


BG = 0
SQUARE_COLORS = (1, 2, 3, 4)
GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (12, 18)
COLOR_COUNT_OPTIONS = (3, 4)
SQUARE_SIZE_OPTIONS = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 600
NOISE_COUNT_RANGE = (3, 7)

NOISE_SHAPES: tuple[np.ndarray, ...] = (
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
)


def _solid_square(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    side_r = int(r1 - r0 + 1)
    side_c = int(c1 - c0 + 1)
    return side_r == side_c and int(coords.shape[0]) == side_r * side_c


def _square_counts(grid: np.ndarray) -> dict[int, int]:
    counts: Counter[int] = Counter()
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True)
    for comp in comps:
        color = int(comp["color"])
        if color == BG:
            continue
        if _solid_square(comp["coords"]):
            counts[color] += 1
    return dict(counts)


def _appearance_order(grid: np.ndarray) -> dict[int, int]:
    order: dict[int, int] = {}
    idx = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color == BG or color in order:
                continue
            order[color] = idx
            idx += 1
    return order


def _run_sequence(color: int, count: int) -> list[int]:
    values: list[int] = []
    lengths = list(range(1, count + 1)) + list(range(count - 1, 0, -1))
    for idx, seg_len in enumerate(lengths):
        values.extend([color] * seg_len)
        if idx != len(lengths) - 1:
            values.append(BG)
    return values


def _ordered_counts(square_counts: dict[int, int]) -> list[tuple[int, int]]:
    nonzero = [(int(color), int(count)) for color, count in square_counts.items() if count > 0]
    assert len(nonzero) >= 3, "expected at least three square colors"
    count_values = [count for _color, count in nonzero]
    assert len(set(count_values)) == len(count_values), "square counts must be unique"
    return sorted(nonzero, key=lambda item: item[1], reverse=True)


def _build_output_from_counts(square_counts: dict[int, int], width: int) -> np.ndarray:
    ordered = _ordered_counts(square_counts)
    seq: list[int] = []
    for idx, (color, count) in enumerate(ordered):
        if idx:
            seq.extend([BG, BG])
        seq.extend(_run_sequence(color, count))

    rows = int(math.ceil(len(seq) / width))
    out = np.full((rows, width), BG, dtype=int)
    for idx, value in enumerate(seq):
        out[idx // width, idx % width] = value
    return out


def _all_components(grid: np.ndarray) -> list[dict]:
    return list(find_components(grid, bg=BG, connectivity=4, monochromatic=True))


def _component_counts(grid: np.ndarray) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for comp in _all_components(grid):
        color = int(comp["color"])
        if color != BG:
            counts[color] += 1
    return dict(counts)


def _pixel_counts(grid: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        counts[color] = int(np.count_nonzero(grid == color))
    return counts


def _deterministic_order(metric: dict[int, int], grid: np.ndarray) -> list[tuple[int, int]]:
    appearance = _appearance_order(grid)
    return sorted(
        [(int(color), int(value)) for color, value in metric.items() if value > 0],
        key=lambda item: (-item[1], appearance.get(item[0], 10**9)),
    )


def _output_from_metric(metric: dict[int, int], width: int, grid: np.ndarray) -> np.ndarray:
    ordered = _deterministic_order(metric, grid)
    seq: list[int] = []
    for idx, (color, value) in enumerate(ordered):
        if idx:
            seq.extend([BG, BG])
        seq.extend(_run_sequence(color, value))
    rows = int(math.ceil(len(seq) / width))
    out = np.full((rows, width), BG, dtype=int)
    for idx, cell in enumerate(seq):
        out[idx // width, idx % width] = cell
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count solid squares per color, turn the counts into descending rise-fall runs, then wrap the result by input width.

    Input structure:
    - The background is black.
    - Foreground colors contain several 4-connected monochromatic components, including solid single-color squares and non-square noise.
    - At least three colors have one or more solid squares, and the square counts across those colors are unique.

    Rule:
    - Find every 4-connected monochromatic component and keep only those whose bounding box is a filled k x k square.
    - Count how many such solid squares each color has, then sort the colors by that count from largest to smallest.
    - For each color with count n, generate a one-dimensional run sequence of lengths 1, 2, ..., n, ..., 2, 1 in that color, placing one black cell between consecutive segments.
    - Concatenate the color sequences in sorted order with two black cells between colors, then wrap the full sequence into rows whose width equals the input width, padding the final row with black if needed.

    Key insight:
    Square size is irrelevant: only the number of solid square components per color survives after filtering noise, and that same count controls both the output ordering and each color's rise-fall run length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    return _build_output_from_counts(_square_counts(g), int(g.shape[1]))


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = map(int, patch.shape)
    H, W = grid.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    for rr in range(top - 1, top + h + 1):
        for cc in range(left - 1, left + w + 1):
            if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BG:
                return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, color: int, top: int, left: int) -> None:
    rows, cols = np.nonzero(patch)
    for r_raw, c_raw in zip(rows.tolist(), cols.tolist(), strict=False):
        grid[top + int(r_raw), left + int(c_raw)] = int(color)


def _place_random_patch(grid: np.ndarray, patch: np.ndarray, color: int) -> bool:
    H, W = grid.shape
    h, w = map(int, patch.shape)
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for top, left in candidates:
        if _can_place(grid, patch, top, left):
            _place_patch(grid, patch, color, top, left)
            return True
    return False


def _square_patch(side: int) -> np.ndarray:
    return np.ones((side, side), dtype=int)


def _sequence_length(counts: dict[int, int]) -> int:
    total = 0
    ordered = _ordered_counts(counts)
    for idx, (_color, count) in enumerate(ordered):
        if idx:
            total += 2
        total += count * count + 2 * count - 2
    return total


def _sample_counts() -> dict[int, int]:
    color_count = random.choice(COLOR_COUNT_OPTIONS)
    colors = random.sample(list(SQUARE_COLORS), color_count)
    values = random.sample([1, 2, 3, 4], color_count)
    random.shuffle(values)
    return {int(color): int(value) for color, value in zip(colors, values, strict=False)}


def _choose_width(total_len: int) -> int:
    exact = [w for w in range(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1) if total_len > w and total_len % w == 0]
    padded = [w for w in range(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1) if total_len > w and total_len % w != 0]
    pool = padded if padded and random.random() < 0.65 else exact or padded
    if not pool:
        pool = [w for w in range(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1) if total_len > w]
    return int(random.choice(pool))


def _noise_allocation(square_counts: dict[int, int]) -> dict[int, int]:
    colors = list(square_counts)
    allocation = {color: 0 for color in colors}
    noisy_color = random.choice(colors)
    allocation[noisy_color] = 1
    extra = random.randint(*NOISE_COUNT_RANGE) - 1
    for _ in range(extra):
        allocation[random.choice(colors)] += 1
    return allocation


def _place_objects(square_counts: dict[int, int], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)

    square_specs: list[tuple[int, int]] = []
    for color, count in square_counts.items():
        for _ in range(count):
            square_specs.append((int(color), int(random.choice(SQUARE_SIZE_OPTIONS))))

    random.shuffle(square_specs)
    for color, side in square_specs:
        if not _place_random_patch(grid, _square_patch(side), color):
            return None

    noise_plan = _noise_allocation(square_counts)
    colors = list(square_counts)
    for color, amount in noise_plan.items():
        for _ in range(amount):
            shape = random.choice(NOISE_SHAPES)
            if not _place_random_patch(grid, shape, color):
                return None

    if sum(1 for amount in noise_plan.values() if amount > 0) < 1:
        return None
    if not any(amount > 0 and square_counts[color] > 0 for color, amount in noise_plan.items()):
        return None
    return grid


def _validate_input(grid: np.ndarray) -> bool:
    square_counts = _square_counts(grid)
    if len(square_counts) < 3:
        return False

    values = list(square_counts.values())
    if any(value < 1 or value > 4 for value in values):
        return False
    if len(set(values)) != len(values):
        return False

    components = _all_components(grid)
    if not any(not _solid_square(comp["coords"]) for comp in components):
        return False

    by_color_noise: Counter[int] = Counter()
    for comp in components:
        color = int(comp["color"])
        if color != BG and not _solid_square(comp["coords"]):
            by_color_noise[color] += 1
    if not any(by_color_noise.get(color, 0) > 0 for color in square_counts):
        return False

    true_order = [color for color, _count in _ordered_counts(square_counts)]
    component_order = [color for color, _count in _deterministic_order(_component_counts(grid), grid)]
    pixel_order = [color for color, _count in _deterministic_order(_pixel_counts(grid), grid)]
    if component_order == true_order:
        return False
    if pixel_order == true_order:
        return False

    width = int(grid.shape[1])
    true_out = main(grid)
    if true_out.shape[0] < 2:
        return False

    if np.array_equal(true_out, _output_from_metric(_component_counts(grid), width, grid)):
        return False
    if np.array_equal(true_out, _output_from_metric(_pixel_counts(grid), width, grid)):
        return False

    seq_len = _sequence_length(square_counts)
    if seq_len <= width:
        return False

    return True


def _generate_candidate() -> np.ndarray | None:
    square_counts = _sample_counts()
    total_len = _sequence_length(square_counts)
    width = _choose_width(total_len)
    height = random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    if width > height * 2 and random.random() < 0.5:
        height = max(height, 15)

    grid = _place_objects(square_counts, height, width)
    if grid is None:
        return None

    if _square_counts(grid) != square_counts:
        return None
    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
