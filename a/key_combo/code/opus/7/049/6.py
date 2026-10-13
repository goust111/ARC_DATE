from __future__ import annotations

import random
from collections import Counter

import numpy as np


RING_TEMPLATE: tuple[tuple[int, int], ...] = (
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
MIN_VISIBLE_PIXELS = 3
BACKGROUND = 0

GRID_SIZE_RANGE = (12, 16)
RING_COUNT_RANGE = (3, 5)
NOISE_COUNT_RANGE = (1, 2)
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _ring_cells(offset: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0 = map(int, offset)
    return {
        (r0 + dr, c0 + dc)
        for dr, dc in RING_TEMPLATE
        if 0 <= r0 + dr < h and 0 <= c0 + dc < w
    }


def _offset_supports(coords: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for r, c in coords:
        for dr, dc in RING_TEMPLATE:
            offset = (r - dr, c - dc)
            counts[offset] = counts.get(offset, 0) + 1
    return counts


def _recover_rings(grid: np.ndarray, bg: int) -> dict[int, dict[str, object]]:
    rings: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(coords) < MIN_VISIBLE_PIXELS:
            continue

        supports = _offset_supports(coords)
        viable = {offset: score for offset, score in supports.items() if score >= MIN_VISIBLE_PIXELS}
        if not viable:
            continue

        best_score = max(viable.values())
        best_offsets = [offset for offset, score in viable.items() if score == best_score]
        if len(best_offsets) != 1:
            raise AssertionError(f"ambiguous ring placement for color {color}")

        offset = best_offsets[0]
        template = _ring_cells(offset, grid.shape)
        visible = {coord for coord in coords if coord in template}
        rings[color] = {
            "offset": offset,
            "visible": visible,
            "template": template,
        }
    assert rings, "expected at least one recoverable ring"
    return rings


def _layer_order(grid: np.ndarray, rings: dict[int, dict[str, object]], bg: int) -> list[int]:
    ring_colors = set(rings)
    above: dict[int, set[int]] = {color: set() for color in ring_colors}
    indegree = {color: 0 for color in ring_colors}

    for color, info in rings.items():
        for r, c in info["template"]:
            seen = int(grid[r, c])
            if seen == color or seen == bg or seen not in ring_colors:
                continue
            if color not in above[seen]:
                above[seen].add(color)
                indegree[color] += 1

    order: list[int] = []
    remaining = set(ring_colors)
    while remaining:
        current = [color for color in remaining if indegree[color] == 0]
        assert len(current) == 1, "occlusion relations must define a unique total order"
        top = current[0]
        order.append(top)
        remaining.remove(top)
        for below in above[top]:
            indegree[below] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Compress overlapping ring layers into a one-row top-to-bottom color sequence.

    Input structure:
    - The background is black and the foreground contains 3 to 5 colors that belong
      to translated copies of the same 6x6 hollow ring template.
    - Rings may overlap each other, be partly clipped by the canvas boundary, and be
      accompanied by 1 to 2 isolated noise pixels that do not belong to any ring layer.
    - A noise pixel may even share a ring color, so each color must be matched to its
      best-fitting ring fragment rather than treated as a perfect clean component.

    Rule:
    - For each non-background color, find the unique translated 6x6 ring template that
      explains the largest subset of that color's pixels; subsets smaller than 3 pixels
      are treated as noise-only colors and ignored.
    - For every recovered ring, inspect each in-bounds template cell: if that cell shows
      another recovered ring color instead of the current ring color, the displayed ring
      must lie above the hidden ring.
    - Topologically sort these occlusion relations from top layer to bottom layer and
      output a 1xK row whose cells are exactly those layer colors in order.

    Key insight:
    - The compression axis is not spatial; the output groups pixels by inferred depth
      layer, so the task is to reconstruct each ring's hidden template and order the
      rings by who visibly covers whose expected border cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    rings = _recover_rings(g, bg)
    order = _layer_order(g, rings, bg)
    return np.array([order], dtype=int)


def _all_offsets(size: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for r0 in range(-RING_SIZE + 1, size):
        for c0 in range(-RING_SIZE + 1, size):
            if len(_ring_cells((r0, c0), (size, size))) >= MIN_VISIBLE_PIXELS:
                offsets.append((r0, c0))
    return offsets


def _build_offsets(size: int, ring_count: int, require_clipped: bool) -> list[tuple[int, int]] | None:
    all_offsets = _all_offsets(size)
    clipped_offsets = [offset for offset in all_offsets if len(_ring_cells(offset, (size, size))) < len(RING_TEMPLATE)]
    unclipped_offsets = [offset for offset in all_offsets if len(_ring_cells(offset, (size, size))) == len(RING_TEMPLATE)]
    if require_clipped and not clipped_offsets:
        return None

    first_pool = clipped_offsets if require_clipped and random.random() < 0.7 else unclipped_offsets or all_offsets
    if not first_pool:
        first_pool = all_offsets

    offsets = [random.choice(first_pool)]
    cells_by_layer = [_ring_cells(offsets[0], (size, size))]
    visible_by_layer = [set(cells_by_layer[0])]

    for _ in range(1, ring_count):
        covered_by_higher = set().union(*cells_by_layer)
        prev_visible = visible_by_layer[-1]
        candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

        for offset in all_offsets:
            if offset in offsets:
                continue
            cells = _ring_cells(offset, (size, size))
            visible = cells - covered_by_higher
            if len(visible) < MIN_VISIBLE_PIXELS:
                continue
            if not (cells & prev_visible):
                continue
            candidates.append((offset, visible))

        if require_clipped and not any(len(cells) < len(RING_TEMPLATE) for cells in cells_by_layer):
            clipped_candidates = [
                (offset, visible)
                for offset, visible in candidates
                if len(_ring_cells(offset, (size, size))) < len(RING_TEMPLATE)
            ]
            if clipped_candidates:
                candidates = clipped_candidates

        if not candidates:
            return None

        offset, visible = random.choice(candidates)
        offsets.append(offset)
        cells_by_layer.append(_ring_cells(offset, (size, size)))
        visible_by_layer.append(visible)

    if require_clipped and not any(len(cells) < len(RING_TEMPLATE) for cells in cells_by_layer):
        return None
    return offsets


def _render_scene(size: int, colors: list[int], offsets: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((size, size), BACKGROUND, dtype=int)
    for color, offset in reversed(list(zip(colors, offsets))):
        for r, c in _ring_cells(offset, grid.shape):
            grid[r, c] = int(color)
    return grid


def _neighbors8(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                out.append((nr, nc))
    return out


def _add_noise(
    grid: np.ndarray,
    ring_colors: list[int],
    offsets: list[tuple[int, int]],
) -> bool:
    size = grid.shape[0]
    ring_cells = set().union(*(_ring_cells(offset, grid.shape) for offset in offsets))
    noise_count = random.randint(*NOISE_COUNT_RANGE)
    chosen_positions: list[tuple[int, int]] = []

    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if int(grid[r, c]) == BACKGROUND
        and (r, c) not in ring_cells
        and all(int(grid[nr, nc]) == BACKGROUND for nr, nc in _neighbors8(r, c, size))
    ]
    random.shuffle(candidates)

    for r, c in candidates:
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen_positions):
            continue
        chosen_positions.append((r, c))
        if len(chosen_positions) == noise_count:
            break
    if len(chosen_positions) != noise_count:
        return False

    fresh_colors = [color for color in COLOR_POOL if color not in ring_colors]
    for r, c in chosen_positions:
        if fresh_colors and random.random() < 0.55:
            color = random.choice(fresh_colors)
        else:
            color = random.choice(ring_colors)
        grid[r, c] = int(color)
    return True


def _attempt_generate() -> np.ndarray | None:
    size = random.randint(*GRID_SIZE_RANGE)
    ring_count = random.randint(*RING_COUNT_RANGE)
    colors = random.sample(list(COLOR_POOL), ring_count)
    require_clipped = random.random() < 0.6

    offsets = _build_offsets(size, ring_count, require_clipped)
    if offsets is None:
        return None

    grid = _render_scene(size, colors, offsets)
    if not _add_noise(grid, colors, offsets):
        return None

    expected = np.array([colors], dtype=int)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
