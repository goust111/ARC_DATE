from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
FILL_SEQUENCE = (4, 7, 6, 8)
RAY_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SOURCE_COLOR_SETS = {
    2: ((1, 2), (2, 3), (3, 8), (1, 9)),
    3: ((1, 2, 3), (3, 1, 2), (2, 8, 1), (9, 3, 1)),
}

GEN_MIN_SIDE = 11
GEN_MAX_SIDE = 17
GEN_MARGIN = 2
GEN_MAX_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_sources(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != BLACK)
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _pairwise_source_constraints(sources: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(sources):
        for r2, c2 in sources[i + 1 :]:
            if r1 == r2 or c1 == c2:
                return False
            if r1 - c1 == r2 - c2 or r1 + c1 == r2 + c2:
                return False
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _layout_spread_ok(sources: list[tuple[int, int]]) -> bool:
    if len(sources) < 2:
        return True
    rows = [r for r, _ in sources]
    cols = [c for _, c in sources]
    return max(rows) - min(rows) >= 4 and max(cols) - min(cols) >= 4


def _draw_rays(
    shape: tuple[int, int], sources: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    walls = np.zeros(shape, dtype=int)
    hit_counts = np.zeros(shape, dtype=int)
    for r0, c0 in sources:
        for dr, dc in RAY_DIRS:
            r, c = r0, c0
            while _in_bounds(shape, r, c):
                walls[r, c] = GREY
                hit_counts[r, c] += 1
                r += dr
                c += dc
    return walls, hit_counts


def _components_of_black(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(cells)
    return components


def _top_left(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return min(cells)


def _solve_from_positions(
    shape: tuple[int, int], sources: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, tuple[int, int], list[tuple[int, int]]]]]:
    walls, hit_counts = _draw_rays(shape, sources)
    components = _components_of_black(walls == BLACK)
    ranked = sorted(
        ((len(cells), _top_left(cells), cells) for cells in components),
        key=lambda item: (item[0], item[1][0], item[1][1]),
    )

    out = walls.copy()
    for idx, (_, _, cells) in enumerate(ranked):
        fill_color = FILL_SEQUENCE[idx % len(FILL_SEQUENCE)]
        for r, c in cells:
            out[r, c] = fill_color
    return out, hit_counts, ranked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from isolated source pixels, partition the canvas, and color the resulting regions by area rank.

    Input structure:
    - The grid is mostly black.
    - There are 2 or 3 isolated non-black single-pixel sources with distinct colors.
    - No two sources share a row, column, or diagonal; their positions define the ray geometry.

    Rule:
    - Replace every source pixel by a grey pixel and extend grey rays from it in all four diagonal directions until the grid boundary.
    - Treat the grey ray network together with the grid boundary as walls, then find every 4-connected black region.
    - Sort regions by area from small to large, breaking equal areas by the row-major top-left cell, and fill them with the repeating sequence yellow, orange, magenta, teal.
    - Leave every grey ray pixel unchanged.

    Key insight:
    - The input colors only mark ray origins; the output depends on the topology created by the diagonal wall network and on the area ordering of the enclosed black regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    sources_with_colors = _extract_sources(g)
    assert 2 <= len(sources_with_colors) <= 3, "expected 2 or 3 source pixels"

    source_positions = [(r, c) for r, c, _ in sources_with_colors]
    assert _pairwise_source_constraints(source_positions), "invalid source geometry"

    out, _hit_counts, _ranked = _solve_from_positions(g.shape, source_positions)
    return out


def _count_internal_crossings(
    hit_counts: np.ndarray, sources: list[tuple[int, int]]
) -> int:
    source_set = set(sources)
    h, w = hit_counts.shape
    crossings = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in source_set:
                continue
            if hit_counts[r, c] >= 2:
                crossings += 1
    return crossings


def _choose_shape() -> tuple[int, int]:
    if random.random() < 0.5:
        side = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)
        return side, side
    return (
        random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE),
        random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE),
    )


def _candidate_cells(shape: tuple[int, int], parity: int) -> list[tuple[int, int]]:
    h, w = shape
    cells = [
        (r, c)
        for r in range(GEN_MARGIN, h - GEN_MARGIN)
        for c in range(GEN_MARGIN, w - GEN_MARGIN)
        if (r + c) % 2 == parity
    ]
    random.shuffle(cells)
    return cells


def _pick_sources(shape: tuple[int, int], n_sources: int) -> list[tuple[int, int]] | None:
    anchor_parity = random.randint(0, 1)
    parity_plan = [anchor_parity, anchor_parity] if n_sources == 2 else [
        anchor_parity,
        anchor_parity,
        1 - anchor_parity,
    ]

    chosen: list[tuple[int, int]] = []
    for parity in parity_plan:
        found = None
        for cell in _candidate_cells(shape, parity):
            tentative = chosen + [cell]
            if _pairwise_source_constraints(tentative) and _layout_spread_ok(tentative):
                found = cell
                break
        if found is None:
            return None
        chosen.append(found)

    if not _layout_spread_ok(chosen):
        return None
    return chosen


def _source_colors(shape: tuple[int, int], sources: list[tuple[int, int]]) -> tuple[int, ...]:
    palette_options = SOURCE_COLOR_SETS[len(sources)]
    signature = shape[0] * 31 + shape[1] * 17 + sum(r * 7 + c for r, c in sources)
    return palette_options[signature % len(palette_options)]


def _is_valid_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    sources_with_colors = _extract_sources(grid)
    if len(sources_with_colors) not in (2, 3):
        return False
    if len({color for _, _, color in sources_with_colors}) != len(sources_with_colors):
        return False
    if GREY in {color for _, _, color in sources_with_colors}:
        return False

    sources = [(r, c) for r, c, _ in sources_with_colors]
    if not _pairwise_source_constraints(sources):
        return False
    if not _layout_spread_ok(sources):
        return False

    out, hit_counts, ranked = _solve_from_positions(grid.shape, sources)
    if _count_internal_crossings(hit_counts, sources) < 1:
        return False
    if len(ranked) < 4:
        return False

    areas = [area for area, _, _ in ranked]
    if len(set(areas)) != len(areas):
        return False

    expected_colors = {GREY}
    for idx in range(len(ranked)):
        expected_colors.add(FILL_SEQUENCE[idx % len(FILL_SEQUENCE)])
    if set(map(int, np.unique(out))) != expected_colors:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        shape = _choose_shape()
        n_sources = random.choice((2, 3))
        sources = _pick_sources(shape, n_sources)
        if sources is None:
            continue

        grid = np.zeros(shape, dtype=int)
        for color, (r, c) in zip(_source_colors(shape, sources), sorted(sources)):
            grid[r, c] = color

        if _is_valid_generated_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid ray-partition puzzle input")
