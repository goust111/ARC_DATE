from __future__ import annotations

import itertools
import random

import numpy as np


AZURE = 8
ARC_MAX = 30

# Generator settings.
COLOR_POOL = tuple(color for color in range(10) if color != AZURE)
RING_COUNT_RANGE = (2, 4)
RING_HEIGHT_RANGE = (4, 9)
RING_WIDTH_MIN = 4
GRID_HEIGHT_CAP = 24
GRID_WIDTH_CAP = 24
MIN_VISIBLE_CELLS = 6
MIN_VISIBLE_SPAN = 3
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _get_bg(grid: np.ndarray) -> int:
    if np.any(grid == AZURE):
        return AZURE
    return _most_common_color(grid)


def _ring_border_cells(
    r0: int,
    c0: int,
    height: int,
    width: int,
    grid_height: int,
    grid_width: int,
) -> list[tuple[int, int]]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    cells: list[tuple[int, int]] = []

    left = max(0, c0)
    right = min(grid_width - 1, c1)
    if left <= right:
        if 0 <= r0 < grid_height:
            for c in range(left, right + 1):
                cells.append((r0, c))
        if r1 != r0 and 0 <= r1 < grid_height:
            for c in range(left, right + 1):
                cells.append((r1, c))

    top = max(0, r0 + 1)
    bottom = min(grid_height - 1, r1 - 1)
    if top <= bottom:
        if 0 <= c0 < grid_width:
            for r in range(top, bottom + 1):
                cells.append((r, c0))
        if c1 != c0 and 0 <= c1 < grid_width:
            for r in range(top, bottom + 1):
                cells.append((r, c1))

    return cells


def _paint_ring(canvas: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 + 1 : r1, c0] = color
    canvas[r0 + 1 : r1, c1] = color


def _unique_topological_order(
    colors: list[int], parents_by_color: dict[int, frozenset[int]]
) -> list[int] | None:
    above_to_below = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for color in colors:
        parents = parents_by_color[color]
        indegree[color] = len(parents)
        for parent in parents:
            above_to_below[parent].add(color)

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        tops = [color for color in remaining if indegree[color] == 0]
        if len(tops) != 1:
            return None
        color = tops[0]
        order.append(color)
        remaining.remove(color)
        for below in above_to_below[color]:
            indegree[below] -= 1
    return order


def _candidate_parent_sets(
    grid: np.ndarray,
    bg: int,
    ring_colors: set[int],
    color: int,
    coords: list[tuple[int, int]],
    height: int,
    width: int,
) -> list[frozenset[int]]:
    if len(coords) > 2 * (height + width) - 4:
        return []

    grid_height, grid_width = grid.shape
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    if max_r - min_r + 1 > height or max_c - min_c + 1 > width:
        return []

    parent_sets: set[frozenset[int]] = set()
    for r0 in range(max_r - height + 1, min_r + 1):
        r1 = r0 + height - 1
        for c0 in range(max_c - width + 1, min_c + 1):
            c1 = c0 + width - 1

            valid = True
            for r, c in coords:
                if not (r0 <= r <= r1 and c0 <= c <= c1):
                    valid = False
                    break
                if r != r0 and r != r1 and c != c0 and c != c1:
                    valid = False
                    break
            if not valid:
                continue

            parents: set[int] = set()
            for r, c in _ring_border_cells(r0, c0, height, width, grid_height, grid_width):
                value = int(grid[r, c])
                if value == bg:
                    valid = False
                    break
                if value != color:
                    if value not in ring_colors:
                        valid = False
                        break
                    parents.add(value)
            if valid:
                parent_sets.add(frozenset(parents))

    return list(parent_sets)


def _solve_parameters(grid: np.ndarray) -> tuple[int, int, tuple[int, ...], int]:
    bg = _get_bg(grid)
    ring_colors = sorted(int(color) for color in np.unique(grid) if int(color) != bg)
    assert ring_colors, "expected at least one ring color"

    coords_by_color = {
        color: [tuple(map(int, cell)) for cell in np.argwhere(grid == color)]
        for color in ring_colors
    }
    min_height = max(3, max(max(r for r, _ in coords) - min(r for r, _ in coords) + 1 for coords in coords_by_color.values()))
    min_width = max(3, max(max(c for _, c in coords) - min(c for _, c in coords) + 1 for coords in coords_by_color.values()))
    max_width = min(ARC_MAX, (ARC_MAX - (len(ring_colors) - 1)) // len(ring_colors))
    assert min_width <= max_width, "output width exceeds ARC limit"

    outputs: set[tuple[int, int, tuple[int, ...]]] = set()
    ring_color_set = set(ring_colors)

    for height in range(min_height, ARC_MAX + 1):
        for width in range(min_width, max_width + 1):
            options: list[tuple[int, list[frozenset[int]]]] = []
            feasible = True
            for color in ring_colors:
                parent_sets = _candidate_parent_sets(
                    grid=grid,
                    bg=bg,
                    ring_colors=ring_color_set,
                    color=color,
                    coords=coords_by_color[color],
                    height=height,
                    width=width,
                )
                if not parent_sets:
                    feasible = False
                    break
                options.append((color, parent_sets))
            if not feasible:
                continue

            options.sort(key=lambda item: len(item[1]))
            for choice in itertools.product(*(parents for _, parents in options)):
                parents_by_color = {
                    color: parent_set for (color, _), parent_set in zip(options, choice)
                }
                order = _unique_topological_order(ring_colors, parents_by_color)
                if order is None:
                    continue
                outputs.add((height, width, tuple(order)))
                if len(outputs) > 1:
                    raise AssertionError("ambiguous ring reconstruction")

    assert len(outputs) == 1, "failed to infer a unique ring template and layer order"
    height, width, order = next(iter(outputs))
    return height, width, order, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapping rectangular ring sprites and lay complete copies out from left to right by layer order.

    Input structure:
    - The background is azure.
    - Each non-background color is one hollow axis-aligned rectangle with a
      one-pixel-thick border, and all rectangles share the same size.
    - The colored rings overlap, so upper rings replace lower-ring border pixels,
      and some rings may extend beyond the grid boundary and be clipped.

    Rule:
    - For every color, infer the full rectangle border whose visible pixels match
      that color and whose in-bounds border positions are all occupied by some ring.
    - Treat every foreign color found on a ring's inferred border as an upper
      layer covering that ring, then build the occlusion partial order.
    - Keep only explanations that yield one unique common ring size and one
      unique top-to-bottom layer sequence.
    - Output a new azure canvas containing one complete copy of each ring,
      ordered from top layer to bottom layer left-to-right, with one background
      column between neighbors.

    Key insight:
    - The visible pixels are incomplete ring fragments; the real task is to
      recover the shared rectangular template first, then read layer order from
      which colors occupy the hidden border cells of other rings.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    height, width, order, bg = _solve_parameters(grid)
    out = np.full((height, len(order) * width + max(0, len(order) - 1)), bg, dtype=int)
    for index, color in enumerate(order):
        _paint_ring(out, 0, index * (width + 1), height, width, color)
    return out


def _visible_fragment_ok(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    if len(cells) < MIN_VISIBLE_CELLS:
        return False
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    row_span = max(rows) - min(rows) + 1
    col_span = max(cols) - min(cols) + 1
    return row_span >= min(MIN_VISIBLE_SPAN, height) and col_span >= min(MIN_VISIBLE_SPAN, width)


def _full_ring_cells(
    offset: tuple[int, int], height: int, width: int, grid_height: int, grid_width: int
) -> set[tuple[int, int]]:
    return set(_ring_border_cells(offset[0], offset[1], height, width, grid_height, grid_width))


def _render_scene(
    grid_height: int,
    grid_width: int,
    height: int,
    width: int,
    colors: list[int],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((grid_height, grid_width), AZURE, dtype=int)
    for color, offset in reversed(list(zip(colors, offsets))):
        for r, c in _ring_border_cells(offset[0], offset[1], height, width, grid_height, grid_width):
            grid[r, c] = color
    return grid


def _expected_output(colors: list[int], height: int, width: int) -> np.ndarray:
    out = np.full((height, len(colors) * width + len(colors) - 1), AZURE, dtype=int)
    for index, color in enumerate(colors):
        _paint_ring(out, 0, index * (width + 1), height, width, color)
    return out


def _all_offsets(grid_height: int, grid_width: int, height: int, width: int) -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in range(-height + 1, grid_height)
        for c0 in range(-width + 1, grid_width)
    ]


def _is_clipped(offset: tuple[int, int], height: int, width: int, grid_height: int, grid_width: int) -> bool:
    r0, c0 = offset
    return not (0 <= r0 and r0 + height <= grid_height and 0 <= c0 and c0 + width <= grid_width)


def _initial_offset_candidates(
    grid_height: int, grid_width: int, height: int, width: int, require_clipped: bool
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for offset in _all_offsets(grid_height, grid_width, height, width):
        if _is_clipped(offset, height, width, grid_height, grid_width) != require_clipped:
            continue
        cells = _full_ring_cells(offset, height, width, grid_height, grid_width)
        if _visible_fragment_ok(cells, height, width):
            candidates.append(offset)
    return candidates


def _next_offset_candidates(
    grid_height: int,
    grid_width: int,
    height: int,
    width: int,
    offsets: list[tuple[int, int]],
    higher_cells: list[set[tuple[int, int]]],
) -> list[tuple[int, int]]:
    covered_by_higher = set().union(*higher_cells)
    covered_before_prev = set().union(*higher_cells[:-1]) if len(higher_cells) > 1 else set()
    visible_prev = higher_cells[-1] - covered_before_prev
    candidates: list[tuple[int, int]] = []

    for offset in _all_offsets(grid_height, grid_width, height, width):
        if offset in offsets:
            continue
        cells = _full_ring_cells(offset, height, width, grid_height, grid_width)
        visible = cells - covered_by_higher
        if not _visible_fragment_ok(visible, height, width):
            continue
        if not (cells & visible_prev):
            continue
        candidates.append(offset)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.randint(*RING_COUNT_RANGE)
    max_width = min(10, (ARC_MAX - (ring_count - 1)) // ring_count)
    if max_width < RING_WIDTH_MIN:
        return None

    height = random.randint(*RING_HEIGHT_RANGE)
    width = random.randint(RING_WIDTH_MIN, max_width)

    grid_height_min = max(12, height + 2)
    grid_width_min = max(12, width + 2)
    if grid_height_min > GRID_HEIGHT_CAP or grid_width_min > GRID_WIDTH_CAP:
        return None

    grid_height = random.randint(grid_height_min, GRID_HEIGHT_CAP)
    grid_width = random.randint(grid_width_min, GRID_WIDTH_CAP)
    colors = random.sample(list(COLOR_POOL), ring_count)
    expected = _expected_output(colors, height, width)

    require_clipped_top = random.random() < 0.5
    first_candidates = _initial_offset_candidates(
        grid_height, grid_width, height, width, require_clipped=require_clipped_top
    )
    if not first_candidates:
        return None

    offsets = [random.choice(first_candidates)]
    higher_cells = [_full_ring_cells(offsets[0], height, width, grid_height, grid_width)]

    for _ in range(1, ring_count):
        candidates = _next_offset_candidates(
            grid_height=grid_height,
            grid_width=grid_width,
            height=height,
            width=width,
            offsets=offsets,
            higher_cells=higher_cells,
        )
        if not candidates:
            return None
        offset = random.choice(candidates)
        offsets.append(offset)
        higher_cells.append(_full_ring_cells(offset, height, width, grid_height, grid_width))

    grid = _render_scene(grid_height, grid_width, height, width, colors, offsets)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangular-ring occlusion puzzle")
