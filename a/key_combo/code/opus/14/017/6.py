from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
BACKGROUND = 0
INPUT_COLORS = (BLUE, RED, GREEN)

GEN_ATTEMPTS = 400
GEN_SIZE_RANGE = (12, 15)
GEN_COUNTS_DESC_OPTIONS: list[tuple[int, int, int]] = []
GEN_OUTPUT_COMBOS: list[tuple[int, int, int, int]] = []
GEN_OUTPUT_QUEUE: list[tuple[int, int, int, int]] = []


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _count_triplet_options() -> list[tuple[int, int, int]]:
    options = []
    for hi in range(5, 11):
        for mid in range(3, 8):
            if mid == hi:
                continue
            for lo in range(2, 10):
                if len({hi, mid, lo}) != 3:
                    continue
                if hi > mid > lo:
                    options.append((hi, mid, lo))
    return options


def _output_combos() -> list[tuple[int, int, int, int]]:
    combos = []
    for hi, mid, _lo in GEN_COUNTS_DESC_OPTIONS:
        for cmax, cmid, cmin in (
            (BLUE, RED, GREEN),
            (BLUE, GREEN, RED),
            (RED, BLUE, GREEN),
            (RED, GREEN, BLUE),
            (GREEN, BLUE, RED),
            (GREEN, RED, BLUE),
        ):
            combos.append((cmax, cmid, cmin, mid))
    dedup = sorted(set(combos))
    return dedup


def _reshuffle_output_queue() -> None:
    global GEN_OUTPUT_QUEUE
    GEN_OUTPUT_QUEUE = GEN_OUTPUT_COMBOS.copy()
    random.shuffle(GEN_OUTPUT_QUEUE)


def _counts_from_output_combo(combo: tuple[int, int, int, int]) -> dict[int, int]:
    cmax, cmid, cmin, mid = combo
    candidates = [triplet for triplet in GEN_COUNTS_DESC_OPTIONS if triplet[1] == mid]
    hi, _, lo = random.choice(candidates)
    return {cmax: hi, cmid: mid, cmin: lo}


def _partition_count(total: int) -> list[int]:
    feasible = []
    max_parts = min(3, total // 2)
    for parts in range(1, max_parts + 1):
        current: list[tuple[int, ...]] = []

        def rec(remaining: int, left: int, acc: list[int]) -> None:
            if left == 0:
                if remaining == 0:
                    current.append(tuple(acc))
                return
            low = 2
            high = min(8, remaining - 2 * (left - 1))
            for size in range(low, high + 1):
                if remaining - size < 0:
                    break
                acc.append(size)
                rec(remaining - size, left - 1, acc)
                acc.pop()

        rec(total, parts, [])
        feasible.extend(current)

    choice = list(random.choice(feasible))
    random.shuffle(choice)
    return choice


def _normalize_shape(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _transform_shape(shape: list[tuple[int, int]], rot: int, flip: bool) -> list[tuple[int, int]]:
    cells = shape
    for _ in range(rot):
        cells = [(-c, r) for r, c in cells]
    if flip:
        cells = [(r, -c) for r, c in cells]
    return _normalize_shape(set(cells))


def _grow_shape(size: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != size:
            continue
        norm = _normalize_shape(cells)
        rows = [r for r, _ in norm]
        cols = [c for _, c in norm]
        height = max(rows) + 1
        width = max(cols) + 1
        area = height * width
        fill = size / area
        if fill < 0.35:
            continue
        if size >= 4 and fill > 0.9:
            continue
        if size >= 4 and (height == 1 or width == 1):
            continue
        if size >= 5 and fill == 1.0:
            continue
        return _transform_shape(norm, random.randrange(4), bool(random.getrandbits(1)))
    return None


def _component_footprint(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    footprint = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    footprint.add((rr, cc))
    return footprint


def _place_component(
    shape: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    placements = []
    for r0 in range(h - shape_h + 1):
        for c0 in range(w - shape_w + 1):
            cells = {(r0 + r, c0 + c) for r, c in shape}
            if cells & occupied:
                continue
            if cells & blocked:
                continue
            placements.append(cells)
    if not placements:
        return None
    return random.choice(placements)


def _validate_input(grid: np.ndarray) -> tuple[dict[int, int], list[list[list[tuple[int, int]]]]]:
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert set(np.unique(grid)).issubset({BACKGROUND, *INPUT_COLORS})
    counts = {color: int(np.count_nonzero(grid == color)) for color in INPUT_COLORS}
    assert all(2 <= count <= 10 for count in counts.values())
    assert len(set(counts.values())) == 3
    mid_count = sorted(counts.values())[1]
    assert 3 <= mid_count <= 7

    per_color_components = []
    for color in INPUT_COLORS:
        comps = _components(grid == color)
        assert comps
        for comp in comps:
            assert 2 <= len(comp) <= 8
        per_color_components.append(comps)
    return counts, per_color_components


def _expected_output_from_counts(counts: dict[int, int]) -> np.ndarray:
    ranked = sorted(((count, color) for color, count in counts.items()), reverse=True)
    (max_count, cmax), (mid_count, cmid), (min_count, cmin) = ranked
    assert max_count > mid_count > min_count

    out_h = 2 * mid_count - 1
    out_w = mid_count
    mid_row = mid_count - 1
    out = np.zeros((out_h, out_w), dtype=int)

    for r in range(mid_row):
        distance = mid_row - r
        fill_width = min(out_w, mid_count + distance)
        out[r, :fill_width] = cmax

    out[mid_row, :] = cmid

    for r in range(mid_row + 1, out_h):
        distance = r - mid_row
        fill_width = mid_count - distance
        if fill_width <= 0:
            break
        out[r, :fill_width] = cmin

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count the three foreground colors, rank them by area, and turn that ranking into a colored wedge output.

    Input structure:
    - The grid has a black background and exactly three foreground colors: blue, red, and green.
    - Each foreground color occupies one or more 4-connected irregular regions, and only total pixel counts matter.
    - The three total color counts are pairwise distinct, with the median count between 3 and 7.

    Rule:
    - Count the total pixels of blue, red, and green, then sort the three colors by descending count as C_max, C_mid, C_min.
    - Create a new output grid with width = count(C_mid) and height = 2 * count(C_mid) - 1.
    - Fill the middle row completely with C_mid.
    - For rows above the middle, fill left-aligned cells with C_max using width = count(C_mid) + distance upward, clipped by the grid width.
    - For rows below the middle, fill left-aligned cells with C_min using width = count(C_mid) - distance downward, stopping when the width reaches 0.

    Key insight:
    - The median color is the shared pivot: its count determines the output size while its color also becomes the central line, and the larger and smaller counts only decide the upper and lower wedge colors.
    """
    g = np.asarray(grid, dtype=int)
    counts = {color: int(np.count_nonzero(g == color)) for color in INPUT_COLORS}
    assert all(count > 0 for count in counts.values())
    assert len(set(counts.values())) == 3
    mid_count = sorted(counts.values())[1]
    assert 3 <= mid_count <= 7
    return _expected_output_from_counts(counts)


def _build_attempt(target_combo: tuple[int, int, int, int]) -> np.ndarray | None:
    h = random.randint(*GEN_SIZE_RANGE)
    w = random.randint(*GEN_SIZE_RANGE)
    grid = np.zeros((h, w), dtype=int)

    counts = _counts_from_output_combo(target_combo)

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    color_order = list(INPUT_COLORS)
    random.shuffle(color_order)
    color_order.sort(key=lambda color: counts[color], reverse=True)

    for color in color_order:
        component_sizes = _partition_count(counts[color])
        component_cells: list[set[tuple[int, int]]] = []
        local_blocked = blocked.copy()
        for size in sorted(component_sizes, reverse=True):
            shape = _grow_shape(size)
            if shape is None:
                return None
            placed = _place_component(shape, occupied, local_blocked, h, w)
            if placed is None:
                return None
            component_cells.append(placed)
            local_blocked |= _component_footprint(placed, h, w)
            occupied |= placed
        for cells in component_cells:
            for r, c in cells:
                grid[r, c] = color
        blocked = local_blocked

    try:
        counts_checked, comps = _validate_input(grid)
    except AssertionError:
        return None

    for color, comp_list in zip(INPUT_COLORS, comps):
        if sum(len(comp) for comp in comp_list) != counts_checked[color]:
            return None

    out = _expected_output_from_counts(counts_checked)
    try:
        pred = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(pred, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    if not GEN_OUTPUT_QUEUE:
        _reshuffle_output_queue()
    for _ in range(GEN_ATTEMPTS):
        combo = GEN_OUTPUT_QUEUE[0]
        grid = _build_attempt(combo)
        if grid is not None:
            GEN_OUTPUT_QUEUE.pop(0)
            return grid
        GEN_OUTPUT_QUEUE.append(GEN_OUTPUT_QUEUE.pop(0))
    raise RuntimeError("failed to generate valid input for puzzle 6")


GEN_COUNTS_DESC_OPTIONS = _count_triplet_options()
GEN_OUTPUT_COMBOS = _output_combos()
