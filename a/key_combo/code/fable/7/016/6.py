"""ARC task: recolor a rectangular mosaic by following every color pointer twice."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        cells: list[Coord] = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc),
                           (rr, cc - 1), (rr, cc + 1)):
                if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                        and not seen[nr, nc]):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> BBox:
    """Inclusive bbox using the common ARC (row, column) convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _read_regions(grid: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Recover arbitrary region masks and their embedded color pointers."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    main_parts: dict[int, np.ndarray] = {}
    for color in map(int, np.unique(g)):
        comps = _components(g == color)
        assert comps, "every visible color must have a component"
        areas = [len(comp) for comp in comps]
        largest = max(areas)
        assert areas.count(largest) == 1, "each color needs one unique main region"
        assert largest > 1, "a region color must occupy more than its pointer copy"
        main_parts[color] = comps[areas.index(largest)]

    owner = np.full(g.shape, -1, dtype=int)
    for color, coords in main_parts.items():
        assert np.all(owner[coords[:, 0], coords[:, 1]] == -1)
        owner[coords[:, 0], coords[:, 1]] = color

    marker_of: dict[int, Coord] = {}
    pointers: dict[int, int] = {}
    for r, c in np.argwhere(owner == -1):
        r, c = int(r), int(c)
        assert 0 < r < g.shape[0] - 1 and 0 < c < g.shape[1] - 1, (
            "every pointer must be an interior pixel"
        )
        neighbors = {int(owner[nr, nc]) for nr, nc in
                     ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))}
        assert len(neighbors) == 1 and -1 not in neighbors, (
            "an isolated pointer must be surrounded by one region"
        )
        region_color = neighbors.pop()
        assert region_color not in marker_of, "each region must have exactly one pointer"
        pointer = int(g[r, c])
        assert pointer in main_parts and pointer != region_color, (
            "pointer must name another region"
        )
        marker_of[region_color] = (r, c)
        pointers[region_color] = pointer
        owner[r, c] = region_color

    assert set(marker_of) == set(main_parts), "each region must contain one pointer"
    assert set(pointers.values()) == set(main_parts), "pointers must form disjoint cycles"

    cycle_lengths: list[int] = []
    unseen = set(main_parts)
    while unseen:
        start = next(iter(unseen))
        cur = start
        length = 0
        while cur in unseen:
            unseen.remove(cur)
            cur = pointers[cur]
            length += 1
        assert cur == start, "pointers must close into disjoint cycles"
        cycle_lengths.append(length)
    assert 2 in cycle_lengths and any(length >= 3 for length in cycle_lengths), (
        "the puzzle requires both a 2-cycle and a longer cycle"
    )
    masks = {color: owner == color for color in main_parts}
    return masks, pointers


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every region with the color reached by following its embedded pointer twice.

    Input structure:
    - The grid is tiled by monochrome regions with distinct region colors.
    - Every region contains exactly one isolated interior pixel of another region's
      color; these color pointers form disjoint cycles, including 2-cycles and
      longer cycles.

    Rule:
    - For region A, read its isolated pixel to find region B, then read B's
      isolated pixel to obtain color C. Fill all of A, including its pointer
      position, with C. Apply this simultaneously to every region.

    Key insight:
    - The isolated colors are lookup keys, not replacement colors: the required
      color is the reference of the referenced region. Thus 2-cycles return to
      their original colors while longer cycles advance by two links.
    """
    g = np.asarray(grid, dtype=int)
    masks, pointers = _read_regions(g)
    out = np.empty_like(g)
    for color, mask in masks.items():
        out[mask] = pointers[pointers[color]]
    return out


def _split_mosaic(h: int, w: int, count: int) -> list[BBox] | None:
    """Create a random guillotine partition with all rectangle sides at least 3."""
    rects: list[BBox] = [(0, 0, h - 1, w - 1)]
    while len(rects) < count:
        choices: list[tuple[int, str]] = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            if r1 - r0 + 1 >= 6:
                choices.extend([(i, "h")] * (r1 - r0 - 4))
            if c1 - c0 + 1 >= 6:
                choices.extend([(i, "v")] * (c1 - c0 - 4))
        if not choices:
            return None
        i, axis = random.choice(choices)
        r0, c0, r1, c1 = rects.pop(i)
        if axis == "h":
            cut = random.randint(r0 + 3, r1 - 2)
            children = [(r0, c0, cut - 1, c1), (cut, c0, r1, c1)]
        else:
            cut = random.randint(c0 + 3, c1 - 2)
            children = [(r0, c0, r1, cut - 1), (r0, cut, r1, c1)]
        rects[i:i] = children
    random.shuffle(rects)
    return rects


def _expected(boxes: list[BBox], colors: list[int], nxt: list[int],
              shape: tuple[int, int]) -> np.ndarray:
    """Independent generator oracle using region indices, not input parsing."""
    out = np.empty(shape, dtype=int)
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        out[r0:r1 + 1, c0:c1 + 1] = colors[nxt[nxt[i]]]
    return out


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(5, 8)
    h = random.randint(15, 30)
    w = random.randint(15, 30)
    boxes = _split_mosaic(h, w, region_count)
    if boxes is None:
        return None

    colors = random.sample(range(10), region_count)
    order = list(range(region_count))
    random.shuffle(order)
    # One 2-cycle plus one longer cycle makes both semantic branches visible.
    cycles = [order[:2], order[2:]]
    nxt = [-1] * region_count
    for cycle in cycles:
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            nxt[a] = b

    grid = np.empty((h, w), dtype=int)
    marker_positions: list[Coord] = []
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        grid[r0:r1 + 1, c0:c1 + 1] = colors[i]
        marker_positions.append((random.randint(r0 + 1, r1 - 1),
                                 random.randint(c0 + 1, c1 - 1)))
    for i, (r, c) in enumerate(marker_positions):
        grid[r, c] = colors[nxt[i]]

    expected = _expected(boxes, colors, nxt, grid.shape)
    one_hop = np.empty_like(grid)
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        one_hop[r0:r1 + 1, c0:c1 + 1] = colors[nxt[i]]

    try:
        solved = main(grid)
    except (AssertionError, ValueError):
        return None
    # Semantic checks: two-hop oracle, erased markers, and rejection of one-hop.
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, one_hop) or np.array_equal(solved, grid):
        return None
    if any(int(solved[r, c]) != colors[nxt[nxt[i]]]
           for i, (r, c) in enumerate(marker_positions)):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate valid two-hop pointer mosaic."""
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid two-hop pointer mosaic")
