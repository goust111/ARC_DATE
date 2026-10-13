from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


_RING: tuple[tuple[int, int], ...] = (
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
_RING_SIZE = 6
_MIN_VISIBLE_RING_PIXELS = 8


def _matching_ring_offsets(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    offset_counts: dict[tuple[int, int], int] = {}
    for r, c in coords:
        for dr, dc in _RING:
            offset = (r - dr, c - dc)
            offset_counts[offset] = offset_counts.get(offset, 0) + 1
    return [offset for offset, count in offset_counts.items() if count == len(coords)]


def _find_ring_fragments(g: np.ndarray, bg: int) -> dict[int, tuple[tuple[int, int], set[tuple[int, int]]]]:
    fragments: dict[int, tuple[tuple[int, int], set[tuple[int, int]]]] = {}
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        if len(coords) < _MIN_VISIBLE_RING_PIXELS:
            continue
        offsets = _matching_ring_offsets(coords)
        assert len(offsets) == 1, "each substantial foreground color must be one ring fragment"
        fragments[color] = (offsets[0], set(coords))
    assert fragments, "expected at least one ring fragment"
    return fragments


def _layer_order(g: np.ndarray, fragments: dict[int, tuple[tuple[int, int], set[tuple[int, int]]]]) -> list[int]:
    h, w = g.shape
    ring_colors = set(fragments)
    above_to_below = {color: set() for color in ring_colors}
    indegree = {color: 0 for color in ring_colors}

    for color, ((r0, c0), visible) in fragments.items():
        expected = {(r0 + dr, c0 + dc) for dr, dc in _RING}
        for r, c in expected - visible:
            if not (0 <= r < h and 0 <= c < w):
                continue
            cover_color = int(g[r, c])
            assert cover_color in ring_colors, "hidden ring pixels must be covered by another ring"
            if color not in above_to_below[cover_color]:
                above_to_below[cover_color].add(color)
                indegree[color] += 1

    order: list[int] = []
    remaining = set(ring_colors)
    while remaining:
        top = [color for color in remaining if indegree[color] == 0]
        assert len(top) == 1, "ring occlusion must determine a unique top-to-bottom order"
        color = top[0]
        order.append(color)
        remaining.remove(color)
        for below in above_to_below[color]:
            indegree[below] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially hidden 6x6 ring sprite and stack complete copies from top layer to bottom layer.

    Input structure:
    - The dominant color is the background.
    - Several non-background colors are visible fragments of the same 12-pixel
      outline ring in a 6x6 box; small off-template marker pixels are noise.
    - Ring fragments overlap, so upper rings replace some expected pixels of
      lower rings, and a ring may also be clipped by the input boundary.

    Rule:
    - Identify every substantial color whose pixels all lie on one translated
      copy of the ring template.
    - For each missing in-bounds template pixel, the color currently occupying
      that pixel is above the hidden ring.
    - Topologically sort these occlusion relations from top to bottom, then draw
      one complete 6x6 ring for each color in that order on a background canvas.

    Key insight:
    - The output is not a crop of the scene; it is the layer stack of complete
      ring sprites reconstructed from their visible fragments.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fragments = _find_ring_fragments(g, bg)
    order = _layer_order(g, fragments)

    out = np.full((_RING_SIZE * len(order), _RING_SIZE), bg, dtype=int)
    for i, color in enumerate(order):
        base_r = i * _RING_SIZE
        for dr, dc in _RING:
            out[base_r + dr, dc] = color
    return out

# --------------- generator (att2) ---------------

# Ring template shared by rendering, validation, and output stacking.
_GEN_RING: tuple[tuple[int, int], ...] = (
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
_GEN_RING_SIZE = 6

# Scene size and structural counts.
_GEN_GRID_SIZE_RANGE = (10, 20)
_GEN_RING_COUNT_RANGE = (2, 4)
_GEN_VISIBLE_PIXELS_RANGE = (8, 11)
_GEN_COLORS = tuple(range(10))

# Noise colors are isolated marker pixels on background cells.
_GEN_NOISE_COLOR_COUNT_RANGE = (1, 2)
_GEN_NOISE_PIXELS_PER_COLOR_RANGE = (2, 5)
_GEN_NOISE_CLEARANCE = 1

# Retry budget for rejection sampling a solver-consistent input.
_GEN_MAX_ATTEMPTS = 300


def _ring_cells(offset: tuple[int, int], n: int) -> set[tuple[int, int]]:
    r0, c0 = offset
    return {
        (r0 + dr, c0 + dc)
        for dr, dc in _GEN_RING
        if 0 <= r0 + dr < n and 0 <= c0 + dc < n
    }


def _gen_matching_ring_offsets(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    offset_counts: dict[tuple[int, int], int] = {}
    for r, c in coords:
        for dr, dc in _GEN_RING:
            offset = (r - dr, c - dc)
            offset_counts[offset] = offset_counts.get(offset, 0) + 1
    return [offset for offset, count in offset_counts.items() if count == len(coords)]


def _has_unique_offset(coords: set[tuple[int, int]], offset: tuple[int, int]) -> bool:
    return _gen_matching_ring_offsets(coords) == [offset]


def _stack_output(bg: int, colors: list[int]) -> np.ndarray:
    out = np.full((_GEN_RING_SIZE * len(colors), _GEN_RING_SIZE), bg, dtype=int)
    for i, color in enumerate(colors):
        for dr, dc in _GEN_RING:
            out[i * _GEN_RING_SIZE + dr, dc] = color
    return out


def _render_scene(n: int, bg: int, colors: list[int], offsets: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((n, n), bg, dtype=int)
    for color, offset in reversed(list(zip(colors, offsets))):
        for r, c in _ring_cells(offset, n):
            grid[r, c] = color
    return grid


def _choose_noise_cells(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    for cell in candidates:
        r, c = cell
        if all(max(abs(r - pr), abs(c - pc)) > _GEN_NOISE_CLEARANCE for pr, pc in picked):
            picked.append(cell)
            if len(picked) == count:
                return picked
    return None


def _add_noise(grid: np.ndarray, bg: int, ring_colors: list[int], offsets: list[tuple[int, int]]) -> bool:
    n = grid.shape[0]
    occupied_ring_cells: set[tuple[int, int]] = set()
    for offset in offsets:
        occupied_ring_cells |= _ring_cells(offset, n)

    available = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if grid[r, c] == bg and (r, c) not in occupied_ring_cells
    ]
    noise_colors = random.sample(
        [color for color in _GEN_COLORS if color != bg and color not in ring_colors],
        random.randint(*_GEN_NOISE_COLOR_COUNT_RANGE),
    )

    for color in noise_colors:
        count = random.randint(*_GEN_NOISE_PIXELS_PER_COLOR_RANGE)
        cells = _choose_noise_cells(available[:], count)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
        chosen = set(cells)
        available = [
            (r, c)
            for r, c in available
            if (r, c) not in chosen
            and all(max(abs(r - pr), abs(c - pc)) > _GEN_NOISE_CLEARANCE for pr, pc in chosen)
        ]
    return True


def _candidate_offsets(
    n: int,
    max_offset: int,
    offsets: list[tuple[int, int]],
    higher_cells: list[set[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Return valid next-ring offsets against the already placed higher layers."""
    covered_by_higher = set().union(*higher_cells)
    covered_before_prev = set().union(*higher_cells[:-1]) if len(higher_cells) > 1 else set()
    visible_prev = higher_cells[-1] - covered_before_prev
    candidates: list[tuple[int, int]] = []

    for r0 in range(max_offset + 1):
        for c0 in range(max_offset + 1):
            offset = (r0, c0)
            if offset in offsets:
                continue
            cells = _ring_cells(offset, n)
            visible = cells - covered_by_higher
            if not _GEN_VISIBLE_PIXELS_RANGE[0] <= len(visible) <= _GEN_VISIBLE_PIXELS_RANGE[1]:
                continue
            if not cells & visible_prev:
                continue
            if not _has_unique_offset(visible, offset):
                continue
            candidates.append(offset)
    return candidates


def _build_layer_offsets(n: int, ring_count: int) -> list[tuple[int, int]] | None:
    """Place top-to-bottom rings so each lower ring overlaps the visible chain."""
    max_offset = n - _GEN_RING_SIZE
    offsets = [(random.randint(0, max_offset), random.randint(0, max_offset))]
    higher_cells = [_ring_cells(offsets[0], n)]

    for _ in range(1, ring_count):
        candidates = _candidate_offsets(n, max_offset, offsets, higher_cells)
        if not candidates:
            return None
        offset = random.choice(candidates)
        offsets.append(offset)
        higher_cells.append(_ring_cells(offset, n))
    return offsets


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.randint(*_GEN_RING_COUNT_RANGE)
    bg = random.randrange(len(_GEN_COLORS))
    ring_colors = random.sample([color for color in _GEN_COLORS if color != bg], ring_count)
    n = random.randint(*_GEN_GRID_SIZE_RANGE)
    expected_output = _stack_output(bg, ring_colors)

    offsets = _build_layer_offsets(n, ring_count)
    if offsets is None:
        return None

    grid = _render_scene(n, bg, ring_colors, offsets)
    if not _add_noise(grid, bg, ring_colors, offsets):
        return None

    try:
        if not np.array_equal(main(grid), expected_output):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 652646ff input")
