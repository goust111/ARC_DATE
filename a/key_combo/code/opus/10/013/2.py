from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
ORANGE = 7
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SOURCE_PALETTE = (1, 2, 3, 4, 6, 8, 9)

# --------------- generator bounds ---------------
GRID_MIN = 13
GRID_MAX = 17
SOURCE_COUNT_CHOICES = (2, 3)
SOURCE_MARGIN = 2
MIN_CHEBYSHEV_GAP = 3
MIN_MANHATTAN_GAP = 5
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_sources(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != bg) & (grid != BLACK))
    sources = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    colors = [color for _, _, color in sources]
    assert 2 <= len(sources) <= 3, "expected 2 or 3 colored source pixels"
    assert len(colors) == len(set(colors)), "source colors must be distinct"
    return sources


def _ray_owner_masks(shape: tuple[int, int], sources: list[tuple[int, int, int]]) -> np.ndarray:
    owner_masks = np.zeros(shape, dtype=np.uint8)
    for idx, (r, c, _color) in enumerate(sources):
        bit = 1 << idx
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            while _in_bounds(shape, nr, nc):
                owner_masks[nr, nc] |= bit
                nr += dr
                nc += dc
    return owner_masks


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            queue = deque([(r, c)])
            part: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                part.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(part)

    return parts


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    sources = _find_sources(g, bg)
    owner_masks = _ray_owner_masks(g.shape, sources)
    ray_mask = owner_masks != 0

    source_mask_map = np.zeros(g.shape, dtype=np.uint8)
    source_colors: dict[int, int] = {}
    for idx, (r, c, color) in enumerate(sources):
        bit = 1 << idx
        source_mask_map[r, c] = bit
        source_colors[bit] = color

    out = g.copy()
    out[ray_mask] = BLACK

    source_regions: list[int] = []
    empty_regions: list[int] = []
    multi_source_regions: list[int] = []

    for part in _components(~ray_mask):
        source_bits = 0
        for r, c in part:
            source_bits |= int(source_mask_map[r, c])

        source_count = int(source_bits).bit_count()
        area = len(part)
        if source_count == 1:
            fill_color = source_colors[source_bits]
            source_regions.append(area)
            for r, c in part:
                out[r, c] = fill_color
        elif source_count == 0:
            empty_regions.append(area)
        else:
            multi_source_regions.append(area)

    filled_background_cells = int(np.sum((g == bg) & (out != bg) & (out != BLACK)))
    crossings = 0
    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(owner_masks[r, c]).bit_count() >= 2:
                crossings += 1

    return {
        "output": out,
        "bg": bg,
        "sources": sources,
        "source_regions": source_regions,
        "empty_regions": empty_regions,
        "multi_source_regions": multi_source_regions,
        "filled_background_cells": filled_background_cells,
        "crossings": crossings,
        "ray_density": float(np.mean(ray_mask)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal black rays from the isolated colored pixels and fill the regions that trap a single source.

    Input structure:
    - The grid is mostly a uniform orange background.
    - There are 2 or 3 isolated non-orange source pixels, each with a distinct color.
    - No source pixels share a row, column, or diagonal, so their diagonal rays stay distinguishable.

    Rule:
    - Keep the source pixels and draw black rays from each source in the four diagonal directions until the grid edge.
    - Treat the black rays as barriers that partition the remaining non-black cells into 4-connected regions.
    - If one such region contains exactly one original source pixel, recolor that whole region with that source's color; regions containing no source or multiple sources stay unchanged apart from the added black rays.

    Key insight:
    - The output is decided globally after all rays are drawn: region ownership comes from which source is isolated inside a partitioned area, not from local distance to a source.
    """
    return np.asarray(_analyze(grid)["output"], dtype=int)


def _is_valid_source(candidate: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = candidate
    for pr, pc in placed:
        dr = abs(r - pr)
        dc = abs(c - pc)
        if r == pr or c == pc or dr == dc:
            return False
        if max(dr, dc) < MIN_CHEBYSHEV_GAP:
            return False
        if dr + dc < MIN_MANHATTAN_GAP:
            return False
    return True


def _sample_sources(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(SOURCE_MARGIN, height - SOURCE_MARGIN)
        for c in range(SOURCE_MARGIN, width - SOURCE_MARGIN)
    ]
    random.shuffle(candidates)

    placed: list[tuple[int, int]] = []
    for cell in candidates:
        if _is_valid_source(cell, placed):
            placed.append(cell)
            if len(placed) == count:
                return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    source_count = random.choice(SOURCE_COUNT_CHOICES)
    positions = _sample_sources(height, width, source_count)
    if positions is None:
        return None

    grid = np.full((height, width), ORANGE, dtype=int)
    for (r, c), color in zip(positions, random.sample(SOURCE_PALETTE, source_count)):
        grid[r, c] = color

    analysis = _analyze(grid)
    source_regions = analysis["source_regions"]
    empty_regions = analysis["empty_regions"]
    multi_source_regions = analysis["multi_source_regions"]

    if len(analysis["sources"]) != source_count:
        return None
    if analysis["crossings"] < 2:
        return None
    if multi_source_regions:
        return None
    if len(source_regions) != source_count:
        return None
    if min(source_regions) < 5:
        return None
    if sum(source_regions) < 18:
        return None
    if not empty_regions or max(empty_regions) < 6:
        return None
    if analysis["filled_background_cells"] < 10:
        return None
    if not (0.12 <= analysis["ray_density"] <= 0.35):
        return None

    if not np.array_equal(main(grid), analysis["output"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
