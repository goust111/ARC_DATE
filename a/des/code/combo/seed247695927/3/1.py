from __future__ import annotations

import random

import numpy as np


RING = (
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 4),
    (2, 0),
    (2, 5),
    (3, 0),
    (3, 5),
    (4, 1),
    (4, 4),
    (5, 2),
    (5, 3),
)
RING_SIZE = 6


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ring_cells(offset: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0 = offset
    return {
        (r0 + dr, c0 + dc)
        for dr, dc in RING
        if 0 <= r0 + dr < h and 0 <= c0 + dc < w
    }


def _matching_offsets(coords: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    if not coords:
        return []

    h, w = shape
    candidates: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in RING:
            candidates.add((r - dr, c - dc))

    matches: list[tuple[int, int]] = []
    for offset in candidates:
        cells = _ring_cells(offset, (h, w))
        if coords.issubset(cells):
            matches.append(offset)
    return matches


def _infer_rings(
    grid: np.ndarray, bg: int
) -> dict[int, tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = grid.shape
    ring_colors = {int(color) for color in np.unique(grid) if int(color) != bg}
    assert ring_colors, "expected at least one ring color"

    rings: dict[int, tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]] = {}
    for color in ring_colors:
        visible = {tuple(map(int, cell)) for cell in np.argwhere(grid == color)}
        assert visible, "ring color must have visible pixels"

        candidates = []
        for offset in _matching_offsets(visible, (h, w)):
            full_cells = _ring_cells(offset, (h, w))
            hidden = full_cells - visible
            if all(int(grid[r, c]) in ring_colors - {color} for r, c in hidden):
                candidates.append((offset, full_cells))

        assert len(candidates) == 1, "each color must match one occluded ring placement"
        offset, full_cells = candidates[0]
        rings[color] = (offset, visible, full_cells)
    return rings


def _top_to_bottom_order(
    grid: np.ndarray,
    rings: dict[int, tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]],
) -> list[int]:
    colors = list(rings)
    above_to_below = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for color, (_offset, visible, full_cells) in rings.items():
        for r, c in full_cells - visible:
            cover = int(grid[r, c])
            assert cover != color, "hidden pixels cannot still show the same ring color"
            if color not in above_to_below[cover]:
                above_to_below[cover].add(color)
                indegree[color] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        available = [color for color in remaining if indegree[color] == 0]
        assert len(available) == 1, "occlusion graph must define a unique layer order"
        top = available[0]
        order.append(top)
        remaining.remove(top)
        for below in above_to_below[top]:
            indegree[below] -= 1
    return order


def _stack_full_rings(
    bg: int,
    order_top_to_bottom: list[int],
    rings: dict[int, tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]],
) -> np.ndarray:
    out = np.full((RING_SIZE * len(order_top_to_bottom), RING_SIZE), bg, dtype=int)
    for i, color in enumerate(order_top_to_bottom):
        base_r = i * RING_SIZE
        for dr, dc in RING:
            out[base_r + dr, dc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full overlapping 6x6 ring sprites and redraw the complete layer stack.

    Input structure:
    - The dominant color is the background.
    - Each non-background color is a visible fragment of one translated 6x6 ring outline.
    - Upper rings overwrite lower rings where they overlap, and a ring may be clipped by the grid boundary.

    Rule:
    - Infer the translated ring placement for every color.
    - For every missing in-bounds ring pixel, the color currently shown there belongs to a higher ring.
    - Topologically sort these occlusion constraints from top to bottom.
    - Output one complete 6x6 ring for each color, ordered from top layer to bottom layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    rings = _infer_rings(g, bg)
    order = _top_to_bottom_order(g, rings)
    return _stack_full_rings(bg, order, rings)


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 18)
RING_COUNT_RANGE = (2, 4)
VISIBLE_PIXELS_RANGE = (4, 10)
MIN_TOP_VISIBLE = 5
MAX_ATTEMPTS = 400
ALL_COLORS = tuple(range(10))


def _all_offsets(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [
        (r0, c0)
        for r0 in range(-RING_SIZE + 1, h)
        for c0 in range(-RING_SIZE + 1, w)
        if _ring_cells((r0, c0), shape)
    ]


def _unique_visible_offset(visible: set[tuple[int, int]], shape: tuple[int, int], offset: tuple[int, int]) -> bool:
    return _matching_offsets(visible, shape) == [offset]


def _build_offsets(shape: tuple[int, int], ring_count: int) -> list[tuple[int, int]] | None:
    all_offsets = _all_offsets(shape)

    top_candidates = []
    for offset in all_offsets:
        cells = _ring_cells(offset, shape)
        if len(cells) < MIN_TOP_VISIBLE:
            continue
        if _unique_visible_offset(cells, shape, offset):
            top_candidates.append(offset)
    if not top_candidates:
        return None

    offsets = [random.choice(top_candidates)]
    higher_cells = [_ring_cells(offsets[0], shape)]

    for _ in range(1, ring_count):
        covered = set().union(*higher_cells)
        covered_before_prev = set().union(*higher_cells[:-1]) if len(higher_cells) > 1 else set()
        prev_visible = higher_cells[-1] - covered_before_prev

        candidates = []
        for offset in all_offsets:
            if offset in offsets:
                continue
            cells = _ring_cells(offset, shape)
            visible = cells - covered
            if not VISIBLE_PIXELS_RANGE[0] <= len(visible) <= VISIBLE_PIXELS_RANGE[1]:
                continue
            if not (cells & prev_visible):
                continue
            if not _unique_visible_offset(visible, shape, offset):
                continue
            candidates.append(offset)

        if not candidates:
            return None

        offset = random.choice(candidates)
        offsets.append(offset)
        higher_cells.append(_ring_cells(offset, shape))

    return offsets


def _render_from_order(
    shape: tuple[int, int], bg: int, colors_top_to_bottom: list[int], offsets_top_to_bottom: list[tuple[int, int]]
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color, offset in reversed(list(zip(colors_top_to_bottom, offsets_top_to_bottom))):
        for r, c in _ring_cells(offset, shape):
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    shape = (h, w)
    ring_count = random.randint(*RING_COUNT_RANGE)

    bg = random.choice(ALL_COLORS)
    ring_colors = random.sample([color for color in ALL_COLORS if color != bg], ring_count)
    offsets = _build_offsets(shape, ring_count)
    if offsets is None:
        return None

    grid = _render_from_order(shape, bg, ring_colors, offsets)
    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return None

    expected = np.full((RING_SIZE * ring_count, RING_SIZE), bg, dtype=int)
    for i, color in enumerate(ring_colors):
        for dr, dc in RING:
            expected[i * RING_SIZE + dr, dc] = color
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
