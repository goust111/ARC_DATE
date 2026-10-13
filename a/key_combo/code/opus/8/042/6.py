from __future__ import annotations

from itertools import permutations
import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 7)
OUTPUT_ORDERS = list(permutations(PALETTE, 4))
USED_OUTPUT_ORDERS: set[tuple[int, ...]] = set()

BLOCK_GRID_CHOICES = (2, 3)
BLOCK_SIZE_RANGE = (4, 6)
AREA_RANGE = tuple(range(2, 13))
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _intervals_from_separators(line_mask: np.ndarray) -> list[tuple[int, int]]:
    separators = [idx for idx, is_sep in enumerate(line_mask) if bool(is_sep)]
    assert separators, "expected separator rows/cols"
    bounds = [-1, *separators, int(line_mask.size)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separator lines must leave non-empty blocks"
    assert len(set(sizes)) == 1, "all blocks must have the same side length"
    return intervals


def _extract_block_records(grid: np.ndarray) -> list[tuple[int, int]]:
    row_mask = np.all(grid == SEPARATOR, axis=1)
    col_mask = np.all(grid == SEPARATOR, axis=0)
    row_intervals = _intervals_from_separators(row_mask)
    col_intervals = _intervals_from_separators(col_mask)

    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert row_sizes == col_sizes, "macro-cells must be square"
    assert len(row_intervals) == len(col_intervals), "block layout must be square"

    records: list[tuple[int, int]] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = grid[r0:r1, c0:c1]
            mask = (block != BACKGROUND) & (block != SEPARATOR)
            assert mask.any(), "each block must contain one foreground shape"
            colors = sorted(set(map(int, block[mask].tolist())))
            assert len(colors) == 1, "each block must contain a single-color shape"

            seen = np.zeros_like(mask, dtype=bool)
            start = tuple(map(int, np.argwhere(mask)[0]))
            stack = [start]
            seen[start] = True
            component = []
            while stack:
                rr, cc = stack.pop()
                component.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            assert len(component) == int(mask.sum()), "block foreground must be one 4-connected component"
            records.append((len(component), colors[0]))

    sizes = [area for area, _color in records]
    assert len(set(sizes)) == len(sizes), "all block areas must be distinct"
    return records


def _build_output(records: list[tuple[int, int]]) -> np.ndarray:
    ranked = sorted(records, reverse=True)
    n = len(ranked)
    side = 2 * n - 1
    out = np.zeros((side, side), dtype=int)
    for inset, (_area, color) in enumerate(ranked):
        lo = inset
        hi = side - inset - 1
        out[lo, lo : hi + 1] = color
        out[hi, lo : hi + 1] = color
        out[lo : hi + 1, lo] = color
        out[lo : hi + 1, hi] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure the shape size inside each block cell and reorder those colors into concentric square rings.

    Input structure:
    - The grid has black background with single-pixel grey separator rows and columns.
    - Those separators partition the canvas into equal square blocks arranged as a square block-grid.
    - Each block contains exactly one single-color 4-connected foreground shape; all other pixels inside the block are black.

    Rule:
    - Detect the separator rows and columns to recover the block-grid.
    - For each block, measure the pixel count of its unique foreground component and record that component's color.
    - Sort the blocks by component area from largest to smallest.
    - Create a square output of side `2N-1`, where `N` is the number of blocks, and draw centered concentric one-pixel-thick square rings in that order, outermost for the largest area and innermost for the smallest.

    Key insight:
    - The block-grid is only a measurement scaffold: spatial positions of the shapes are irrelevant after their areas are read, and only the size ranking controls the new concentric geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _build_output(_extract_block_records(g))


def _weighted_choice(items: list[tuple[tuple[int, int], int]]) -> tuple[int, int]:
    total = sum(weight for _cell, weight in items)
    pick = random.randint(1, total)
    running = 0
    for cell, weight in items:
        running += weight
        if pick <= running:
            return cell
    return items[-1][0]


def _generate_shape(block_size: int, area: int, mode: str) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        seed = (random.randrange(block_size), random.randrange(block_size))
        cells = {seed}
        trail = [seed]

        while len(cells) < area:
            frontier: list[tuple[tuple[int, int], int]] = []
            seen = set()
            bases = [trail[-1]] if mode == "snake" else trail
            for rr, cc in bases:
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < block_size and 0 <= nc < block_size):
                        continue
                    if (nr, nc) in cells or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    contacts = sum((ar, ac) in cells for ar, ac in _neighbors4(nr, nc))
                    if mode == "compact":
                        weight = 1 + 3 * contacts
                    elif mode == "snake":
                        dist = abs(nr - trail[-1][0]) + abs(nc - trail[-1][1])
                        weight = 4 if dist == 1 and contacts == 1 else 1
                    else:
                        weight = 1 + contacts
                    frontier.append(((nr, nc), weight))
            if not frontier:
                break
            nxt = _weighted_choice(frontier)
            cells.add(nxt)
            trail.append(nxt)

        if len(cells) != area:
            continue

        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if area >= 4 and (len(rows) == 1 or len(cols) == 1):
            continue

        mask = np.zeros((block_size, block_size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _sample_output_order() -> tuple[int, ...]:
    unused = [order for order in OUTPUT_ORDERS if order not in USED_OUTPUT_ORDERS]
    if unused:
        return random.choice(unused)
    return random.choice(OUTPUT_ORDERS)


def _render_input(
    grid_side: int,
    blocks_per_side: int,
    block_size: int,
    placement_records: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    grid = np.zeros((grid_side, grid_side), dtype=int)
    for sep in range(1, blocks_per_side):
        idx = sep * block_size + (sep - 1)
        grid[idx, :] = SEPARATOR
        grid[:, idx] = SEPARATOR

    for idx, (area, color, mask) in enumerate(placement_records):
        br = idx // blocks_per_side
        bc = idx % blocks_per_side
        r0 = br * (block_size + 1)
        c0 = bc * (block_size + 1)
        block = np.zeros((block_size, block_size), dtype=int)
        block[mask] = color
        grid[r0 : r0 + block_size, c0 : c0 + block_size] = block
    return grid


def _validate_generated(grid: np.ndarray, expected_order: tuple[int, ...], expected_areas: tuple[int, ...]) -> bool:
    try:
        parsed = _extract_block_records(grid)
    except AssertionError:
        return False

    areas = sorted((area for area, _color in parsed), reverse=True)
    if tuple(areas) != expected_areas:
        return False

    ranked_colors = tuple(color for _area, color in sorted(parsed, reverse=True))
    if ranked_colors != expected_order:
        return False

    expected = _build_output([(area, color) for area, color in zip(expected_areas, expected_order)])
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        blocks_per_side = random.choice(BLOCK_GRID_CHOICES)
        if blocks_per_side == 3:
            # The puzzle text allows 3x3 block grids, but the visible ARC palette cannot give
            # nine distinct ring colors. Keep solver support general and generate the non-degenerate branch.
            blocks_per_side = 2

        block_count = blocks_per_side * blocks_per_side
        block_size = random.randint(*BLOCK_SIZE_RANGE)
        grid_side = blocks_per_side * block_size + (blocks_per_side - 1)

        output_order = _sample_output_order()
        chosen_areas = tuple(sorted(random.sample(AREA_RANGE, block_count), reverse=True))
        records = []
        for area, color in zip(chosen_areas, output_order):
            mode = random.choice(("compact", "balanced", "snake"))
            mask = _generate_shape(block_size, area, mode)
            if mask is None:
                break
            records.append((area, color, mask))
        if len(records) != block_count:
            continue

        random.shuffle(records)
        grid = _render_input(grid_side, blocks_per_side, block_size, records)
        if _validate_generated(grid, output_order, chosen_areas):
            USED_OUTPUT_ORDERS.add(output_order)
            return grid

    raise RuntimeError("failed to generate a valid non-degenerate input")
