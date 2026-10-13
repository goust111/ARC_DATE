from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, paste, random_free_location


BG = Color.BLACK
PALETTE = tuple(range(1, 10))

GEN_COLOR_COUNTS = (2, 3, 4)
GEN_COUNT_VALUES = tuple(range(1, 7))
GEN_TOTAL_COUNT_MIN = 4
GEN_TOTAL_COUNT_MAX = 12
GEN_MAX_OBJECT_AREA = 8
GEN_CANVAS_SIDE_RANGE = (14, 26)
GEN_OBJECT_PADDING = 1
GEN_PADDING_CONNECTIVITY = 8
GEN_MAX_ATTEMPTS = 400


def _colors_present(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BG]


def _count_objects_by_color(grid: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for color in _colors_present(grid):
        comps = components_mask(grid == color, connectivity=4)
        if comps:
            counts[color] = len(comps)
    assert counts, "expected at least one non-black color"
    return counts


def _count_sequence(counts: dict[int, int]) -> list[int]:
    ordered = sorted(counts.items(), key=lambda item: item[1])
    values = [count for _color, count in ordered]
    assert len(values) == len(set(values)), "object counts must be pairwise distinct"

    seq: list[int] = []
    for color, count in ordered:
        seq.extend([int(color)] * int(count))
    assert seq, "expected a non-empty encoded sequence"
    assert len(seq) <= 30, "output side length must not exceed 30"
    return seq


def _render_output(seq: list[int]) -> np.ndarray:
    side = len(seq)
    index_grid = np.maximum.outer(np.arange(side), np.arange(side))
    return np.asarray(seq, dtype=int)[index_grid]


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Count four-connected objects of each color, sort colors by that count, and encode the counts as concentric L-rings.

    Input structure:
    - A black background contains several colors of separated 4-connected objects.
    - Objects of the same color may vary in shape and size.
    - Each color appears in a unique number of objects, and pixel totals are not the cue.

    Rule:
    - Count the number of 4-connected objects for each non-black color.
    - Sort the colors by object count in ascending order.
    - Build a 1D color sequence by repeating each color exactly its object count times.
    - Return an N x N grid, where N is the total object count and each cell (r, c) takes the
      color at sequence index max(r, c).

    Key insight:
    - The output is the shadow of a 1D count-encoded sequence under the diagonal index map max(r, c),
      so the visible rings encode counts, not geometry or pixel area.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    counts = _count_objects_by_color(grid)
    seq = _count_sequence(counts)
    return _render_output(seq)


def _random_connected_mask(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    h = max(r for r, _c in norm) + 1
    w = max(c for _r, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _shape_patch(color: int, area: int) -> np.ndarray:
    mask = _random_connected_mask(area)
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _sample_counts(n_colors: int) -> list[int]:
    options = [
        combo
        for combo in combinations(GEN_COUNT_VALUES, n_colors)
        if GEN_TOTAL_COUNT_MIN <= sum(combo) <= GEN_TOTAL_COUNT_MAX
    ]
    counts = list(random.choice(options))
    random.shuffle(counts)
    return counts


def _sample_total_areas(counts: list[int]) -> list[int]:
    candidates = [
        list(range(count, min(GEN_MAX_OBJECT_AREA * count, 20) + 1))
        for count in counts
    ]
    count_order = sorted(range(len(counts)), key=lambda i: counts[i])

    for _ in range(1200):
        totals = [random.choice(options) for options in candidates]
        if len(set(totals)) != len(totals):
            continue
        total_order = sorted(range(len(totals)), key=lambda i: totals[i])
        if total_order != count_order:
            return totals

    for left_pos in range(len(count_order) - 1):
        for right_pos in range(left_pos + 1, len(count_order)):
            left = count_order[left_pos]
            right = count_order[right_pos]
            left_choices = [value for value in candidates[left] if value > min(candidates[right])]
            if not left_choices:
                continue

            for _ in range(200):
                totals = [None] * len(counts)
                totals[left] = random.choice(left_choices)
                right_choices = [value for value in candidates[right] if value < totals[left]]
                if not right_choices:
                    continue
                totals[right] = random.choice(right_choices)
                used = {totals[left], totals[right]}

                remaining = [idx for idx in range(len(counts)) if totals[idx] is None]
                random.shuffle(remaining)
                failed = False
                for idx in remaining:
                    choices = [value for value in candidates[idx] if value not in used]
                    if not choices:
                        failed = True
                        break
                    totals[idx] = random.choice(choices)
                    used.add(totals[idx])
                if failed:
                    continue

                total_order = sorted(range(len(totals)), key=lambda i: totals[i])
                if total_order != count_order:
                    return [int(value) for value in totals]
    raise RuntimeError("failed to sample misleading total areas")


def _split_total_area(count: int, total_area: int) -> list[int]:
    areas = [1] * count
    remaining = total_area - count
    while remaining > 0:
        candidates = [i for i, area in enumerate(areas) if area < GEN_MAX_OBJECT_AREA]
        areas[random.choice(candidates)] += 1
        remaining -= 1
    random.shuffle(areas)
    return areas


def _sample_canvas_side(total_objects: int, total_pixels: int) -> int:
    low = max(GEN_CANVAS_SIDE_RANGE[0], 2 * total_objects, int(np.ceil(np.sqrt(total_pixels * 6))))
    high = GEN_CANVAS_SIDE_RANGE[1]
    if low > high:
        raise RuntimeError("no feasible canvas side")
    return random.randint(low, high)


def _pixel_totals(grid: np.ndarray) -> dict[int, int]:
    return {color: int(np.count_nonzero(grid == color)) for color in _colors_present(grid)}


def _is_semantic_counterexample(grid: np.ndarray) -> bool:
    counts = _count_objects_by_color(grid)
    pixel_totals = _pixel_totals(grid)
    count_order = [color for color, _count in sorted(counts.items(), key=lambda item: item[1])]
    pixel_order = [color for color, _count in sorted(pixel_totals.items(), key=lambda item: item[1])]
    return pixel_order != count_order


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.choice(GEN_COLOR_COUNTS)
    colors = random.sample(PALETTE, n_colors)
    counts = _sample_counts(n_colors)
    total_areas = _sample_total_areas(counts)
    object_areas = {
        color: _split_total_area(count, total_area)
        for color, count, total_area in zip(colors, counts, total_areas)
    }

    total_objects = sum(counts)
    total_pixels = sum(total_areas)
    side = _sample_canvas_side(total_objects, total_pixels)
    grid = np.full((side, side), BG, dtype=int)

    specs = []
    for color in colors:
        for area in object_areas[color]:
            specs.append((color, area))
    random.shuffle(specs)

    border = random.choice((0, 1))
    try:
        for color, area in specs:
            patch = _shape_patch(color, area)
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=BG,
                border=border,
                padding=GEN_OBJECT_PADDING,
                padding_connectivity=GEN_PADDING_CONNECTIVITY,
            )
            paste(grid, patch, r0, c0, transparent=BG)
    except (RuntimeError, ValueError):
        return None

    if not _is_semantic_counterexample(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if out.shape != (total_objects, total_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")


def _validate_example(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30

    counts = _count_objects_by_color(inp)
    assert len(counts) >= 2
    assert len(set(counts.values())) == len(counts)

    seq = _count_sequence(counts)
    expected = _render_output(seq)
    assert np.array_equal(out, expected)
    assert out.shape == (len(seq), len(seq))

    diag_index = np.maximum.outer(np.arange(len(seq)), np.arange(len(seq)))
    for idx, color in enumerate(seq):
        assert np.all(out[diag_index == idx] == color)
