from __future__ import annotations

import random

import numpy as np


_RING: tuple[tuple[int, int], ...] = (
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 1),
    (1, 5),
    (2, 0),
    (2, 6),
    (3, 0),
    (3, 6),
    (4, 0),
    (4, 6),
    (5, 1),
    (5, 5),
    (6, 2),
    (6, 3),
    (6, 4),
)
_RING_SIZE = 7


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _expected_cells(offset: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = offset
    return {(r0 + dr, c0 + dc) for dr, dc in _RING}


def _matching_offsets(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    offset_counts: dict[tuple[int, int], int] = {}
    for r, c in coords:
        for dr, dc in _RING:
            offset = (r - dr, c - dc)
            offset_counts[offset] = offset_counts.get(offset, 0) + 1
    return sorted(offset for offset, count in offset_counts.items() if count == len(coords))


def _find_fragments(
    grid: np.ndarray, bg: int
) -> dict[int, tuple[tuple[int, int], set[tuple[int, int]]]]:
    fragments: dict[int, tuple[tuple[int, int], set[tuple[int, int]]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == color)]
        assert coords, "every ring color must remain visible"
        offsets = _matching_offsets(coords)
        assert len(offsets) == 1, "each color must be one translated ring fragment"
        fragments[color] = (offsets[0], set(coords))
    assert len(fragments) >= 2, "expected multiple ring fragments"
    return fragments


def _layer_order_bottom_to_top(
    grid: np.ndarray,
    fragments: dict[int, tuple[tuple[int, int], set[tuple[int, int]]]],
) -> list[int]:
    h, w = grid.shape
    ring_colors = set(fragments)
    below_to_above = {color: set() for color in ring_colors}
    indegree = {color: 0 for color in ring_colors}

    for color, (offset, visible) in fragments.items():
        for r, c in _expected_cells(offset) - visible:
            if not (0 <= r < h and 0 <= c < w):
                continue
            cover_color = int(grid[r, c])
            assert cover_color in ring_colors, "hidden ring pixels must be covered by another ring"
            if cover_color not in below_to_above[color]:
                below_to_above[color].add(cover_color)
                indegree[cover_color] += 1

    order: list[int] = []
    remaining = set(ring_colors)
    while remaining:
        bottoms = [color for color in remaining if indegree[color] == 0]
        assert len(bottoms) == 1, "occlusion graph must define a unique bottom-to-top order"
        color = bottoms[0]
        order.append(color)
        remaining.remove(color)
        for above in below_to_above[color]:
            indegree[above] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially hidden ring and stack the completed copies from bottom layer to top layer.

    Input structure:
    - The dominant color is the background.
    - Every other color is the visible part of one translated copy of a fixed 7x7 ring template.
    - Higher rings overwrite lower rings, and some rings may be clipped by the grid boundary.

    Rule:
    - Recover the unique template offset for each color.
    - Whenever another ring color occupies an expected ring pixel, that other color is above it.
    - Topologically sort the occlusion graph from bottom to top.
    - Draw one complete ring per color on a fresh canvas, placing the bottom layer first.

    Key insight:
    - The output is a reconstructed layer stack of complete templates rather than a crop of the scene.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fragments = _find_fragments(g, bg)
    order = _layer_order_bottom_to_top(g, fragments)

    out = np.full((_RING_SIZE * len(order), _RING_SIZE), bg, dtype=int)
    for i, color in enumerate(order):
        base_r = i * _RING_SIZE
        for dr, dc in _RING:
            out[base_r + dr, dc] = color
    return out


# --------------- generator ---------------

_GEN_GRID_H_RANGE = (12, 18)
_GEN_GRID_W_RANGE = (12, 20)
_GEN_RING_COUNT_RANGE = (2, 4)
_GEN_VISIBLE_PIXELS_RANGE = (6, 13)
_GEN_COLOR_VALUES = tuple(range(10))
_GEN_MAX_ATTEMPTS = 300


def _ring_cells(offset: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r, c in _expected_cells(offset) if 0 <= r < h and 0 <= c < w}


def _is_clipped(offset: tuple[int, int], h: int, w: int) -> bool:
    return len(_ring_cells(offset, h, w)) < len(_RING)


def _has_unique_offset(cells: set[tuple[int, int]], offset: tuple[int, int]) -> bool:
    return _matching_offsets(sorted(cells)) == [offset]


def _offset_ranges(h: int, w: int, allow_clipped: bool) -> tuple[range, range]:
    if allow_clipped:
        return range(-(_RING_SIZE - 1), h), range(-(_RING_SIZE - 1), w)
    return range(0, h - _RING_SIZE + 1), range(0, w - _RING_SIZE + 1)


def _initial_offsets(h: int, w: int, allow_clipped: bool) -> list[tuple[int, int]]:
    rows, cols = _offset_ranges(h, w, allow_clipped)
    candidates: list[tuple[int, int]] = []
    for r0 in rows:
        for c0 in cols:
            offset = (r0, c0)
            cells = _ring_cells(offset, h, w)
            if not _GEN_VISIBLE_PIXELS_RANGE[0] <= len(cells) <= len(_RING):
                continue
            if not _has_unique_offset(cells, offset):
                continue
            candidates.append(offset)
    return candidates


def _candidate_offsets(
    h: int,
    w: int,
    allow_clipped: bool,
    offsets: list[tuple[int, int]],
    higher_cells: list[set[tuple[int, int]]],
) -> list[tuple[int, int]]:
    rows, cols = _offset_ranges(h, w, allow_clipped)
    covered_by_higher = set().union(*higher_cells)
    covered_before_prev = set().union(*higher_cells[:-1]) if len(higher_cells) > 1 else set()
    visible_prev = higher_cells[-1] - covered_before_prev
    candidates: list[tuple[int, int]] = []

    for r0 in rows:
        for c0 in cols:
            offset = (r0, c0)
            if offset in offsets:
                continue
            cells = _ring_cells(offset, h, w)
            visible = cells - covered_by_higher
            if not _GEN_VISIBLE_PIXELS_RANGE[0] <= len(visible) <= _GEN_VISIBLE_PIXELS_RANGE[1]:
                continue
            if not (cells & visible_prev):
                continue
            if not _has_unique_offset(visible, offset):
                continue
            candidates.append(offset)
    return candidates


def _build_layer_offsets(h: int, w: int, ring_count: int, want_clipped: bool) -> list[tuple[int, int]] | None:
    top_candidates = _initial_offsets(h, w, allow_clipped=want_clipped)
    if not top_candidates:
        return None

    top_offset = random.choice(top_candidates)
    offsets = [top_offset]
    higher_cells = [_ring_cells(top_offset, h, w)]

    for _ in range(1, ring_count):
        candidates = _candidate_offsets(h, w, want_clipped, offsets, higher_cells)
        if not candidates:
            return None
        offset = random.choice(candidates)
        offsets.append(offset)
        higher_cells.append(_ring_cells(offset, h, w))

    clipped_flags = [_is_clipped(offset, h, w) for offset in offsets]
    if want_clipped and not any(clipped_flags):
        return None
    if not want_clipped and any(clipped_flags):
        return None
    return offsets


def _render_scene(
    h: int,
    w: int,
    bg: int,
    colors_top_to_bottom: list[int],
    offsets_top_to_bottom: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for color, offset in reversed(list(zip(colors_top_to_bottom, offsets_top_to_bottom))):
        for r, c in _ring_cells(offset, h, w):
            grid[r, c] = color
    return grid


def _expected_output(bg: int, colors_top_to_bottom: list[int]) -> np.ndarray:
    colors_bottom_to_top = list(reversed(colors_top_to_bottom))
    out = np.full((_RING_SIZE * len(colors_bottom_to_top), _RING_SIZE), bg, dtype=int)
    for i, color in enumerate(colors_bottom_to_top):
        for dr, dc in _RING:
            out[i * _RING_SIZE + dr, dc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GEN_GRID_H_RANGE)
    w = random.randint(*_GEN_GRID_W_RANGE)
    ring_count = random.randint(*_GEN_RING_COUNT_RANGE)
    want_clipped = random.choice((False, True))
    bg = random.choice(_GEN_COLOR_VALUES)
    ring_colors = random.sample([color for color in _GEN_COLOR_VALUES if color != bg], ring_count)

    offsets = _build_layer_offsets(h, w, ring_count, want_clipped)
    if offsets is None:
        return None

    grid = _render_scene(h, w, bg, ring_colors, offsets)
    if _get_bg(grid) != bg:
        return None

    expected = _expected_output(bg, ring_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-occlusion input")
