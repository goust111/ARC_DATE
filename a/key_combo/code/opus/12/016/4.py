import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 7)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

TEMPLATES_BY_COUNT = {
    4: [
        ((0, 0), (0, 2), (0, 4), (2, 0)),
        ((0, 0), (0, 2), (0, 4), (2, 3)),
    ],
    5: [
        ((0, 0), (0, 2), (0, 4), (2, 0), (3, 3)),
        ((0, 0), (0, 2), (0, 4), (2, 0), (2, 3)),
        ((0, 0), (0, 2), (0, 4), (2, 0), (2, 2)),
    ],
    6: [
        ((0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (3, 4)),
        ((0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (4, 0)),
        ((0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (4, 2)),
    ],
}


def _neighbors4(r, c, shape):
    height, width = shape
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _neighbors8(r, c, shape):
    height, width = shape
    for dr, dc in NEIGHBORS8:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _find_components(grid):
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, g.shape):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": _bbox(cells),
                }
            )
    return comps


def _dilation_region(grid, cells):
    g = np.asarray(grid, dtype=int)
    region = set()
    for r, c in cells:
        for nr, nc in _neighbors8(r, c, g.shape):
            if int(g[nr, nc]) == BLACK:
                region.add((nr, nc))
    return region


def _component_graph(grid):
    comps = _find_components(grid)
    regions = [_dilation_region(grid, comp["cells"]) for comp in comps]
    degrees = [0] * len(comps)
    adjacency = [set() for _ in comps]
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if regions[i] & regions[j]:
                degrees[i] += 1
                degrees[j] += 1
                adjacency[i].add(j)
                adjacency[j].add(i)
    return comps, degrees, adjacency


def _encode_output(comps, degrees):
    order = sorted(
        range(len(comps)),
        key=lambda idx: (-degrees[idx], comps[idx]["bbox"][0], comps[idx]["bbox"][1]),
    )
    width = max(1, max(degrees, default=0))
    out = np.zeros((len(comps), width), dtype=int)
    for row, idx in enumerate(order):
        degree = degrees[idx]
        if degree > 0:
            out[row, :degree] = comps[idx]["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode each component by the number of other components whose 8-neighborhood dilations overlap with it.
    
    Input structure:
    - The grid is black in the background and contains 4-6 isolated 4-connected colored foreground components.
    - Each foreground component is monochromatic, irregular, and uses a distinct visible color.
    - Components do not touch directly; their proximity is expressed only through small blank gaps between them.
    
    Rule:
    - For every component, build its 1-cell 8-neighborhood dilation region on blank cells only.
    - Two components are adjacent exactly when their dilation regions share at least one blank cell.
    - Count the adjacency degree of each component, sort components by descending degree with ties broken by bbox top-left position, and output one row per component using that component's color repeated horizontally for exactly its degree.
    
    Key insight:
    - The answer is not based on the original shapes or colors alone; the essential hidden structure is the overlap graph created by one-step dilation, and the output keeps only each component's degree in that graph.
    """
    comps, degrees, _ = _component_graph(grid)
    return _encode_output(comps, degrees)


def _transform_points(points, mode):
    pts = []
    for r, c in points:
        if mode == 0:
            nr, nc = r, c
        elif mode == 1:
            nr, nc = r, -c
        elif mode == 2:
            nr, nc = -r, c
        elif mode == 3:
            nr, nc = -r, -c
        elif mode == 4:
            nr, nc = c, r
        elif mode == 5:
            nr, nc = c, -r
        elif mode == 6:
            nr, nc = -c, r
        else:
            nr, nc = -c, -r
        pts.append((nr, nc))
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _anchor_edges(anchors):
    edges = set()
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if _chebyshev(anchors[i], anchors[j]) == 2:
                edges.add((i, j))
    return edges


def _candidate_ok(cell, idx, occupied, anchors, edge_set):
    for j, other in enumerate(occupied):
        if j == idx:
            continue
        threshold = 2 if (min(idx, j), max(idx, j)) in edge_set else 3
        for other_cell in other:
            if _chebyshev(cell, other_cell) < threshold:
                return False
    for j, other_anchor in enumerate(anchors):
        if j == idx:
            continue
        threshold = 2 if (min(idx, j), max(idx, j)) in edge_set else 3
        if _chebyshev(cell, other_anchor) < threshold:
            return False
    return True


def _irregular(cells):
    if len(cells) < 3:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    return height >= 2 and width >= 2 and len(cells) < height * width


def _grow_component(idx, anchor, occupied, anchors, edge_set, shape, center, target_area):
    for _ in range(40):
        cells = {anchor}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, shape):
                    if (nr, nc) in cells:
                        continue
                    if max(abs(nr - anchor[0]), abs(nc - anchor[1])) > 3:
                        continue
                    if not _candidate_ok((nr, nc), idx, occupied, anchors, edge_set):
                        continue
                    frontier.add((nr, nc))
            if not frontier:
                break

            ranked = []
            for nr, nc in frontier:
                outward = (nr - center[0]) * (anchor[0] - center[0]) + (nc - center[1]) * (anchor[1] - center[1])
                room = min(_chebyshev((nr, nc), other) for k, other in enumerate(anchors) if k != idx)
                bend_bonus = 0
                if len(cells) >= 2:
                    neighbor_count = sum((rr, cc) in cells for rr, cc in _neighbors4(nr, nc, shape))
                    bend_bonus = 2 - neighbor_count
                ranked.append((room, outward, bend_bonus, random.random(), (nr, nc)))
            ranked.sort(reverse=True)
            pick_pool = [item[-1] for item in ranked[: min(6, len(ranked))]]
            cells.add(random.choice(pick_pool))

        if len(cells) == target_area and _irregular(cells):
            return cells
    return None


def _min_component_chebyshev(cells_a, cells_b):
    return min(_chebyshev(a, b) for a in cells_a for b in cells_b)


def _pair_gap_counts(components):
    gaps = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            gaps.append(_min_component_chebyshev(components[i], components[j]) - 1)
    return gaps


def _validate_example(grid):
    comps, degrees, adjacency = _component_graph(grid)
    if not (4 <= len(comps) <= 6):
        return False
    if len({comp["color"] for comp in comps}) != len(comps):
        return False
    if any(not (3 <= len(comp["cells"]) <= 8) for comp in comps):
        return False
    if any(not _irregular(comp["cells"]) for comp in comps):
        return False

    gaps = _pair_gap_counts([set(comp["cells"]) for comp in comps])
    if not gaps or min(gaps) < 1 or max(gaps) > 3:
        return False
    if sum(gap == 1 for gap in gaps) < 2:
        return False
    if not any(gap == 3 for gap in gaps):
        return False
    if len(set(degrees)) == 1:
        return False

    edge_pairs = sum(len(nbrs) for nbrs in adjacency) // 2
    if edge_pairs < 2:
        return False

    expected = _encode_output(comps, degrees)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False

    return True


def _place_attempt():
    height = random.randint(14, 18)
    width = random.randint(14, 18)
    target_count = random.choices([4, 5, 6], weights=[0.34, 0.36, 0.30], k=1)[0]
    base_template = random.choice(TEMPLATES_BY_COUNT[target_count])
    anchors0 = _transform_points(base_template, random.randrange(8))
    max_r = max(r for r, _ in anchors0)
    max_c = max(c for _, c in anchors0)

    row_offset = random.randint(3, height - max_r - 4)
    col_offset = random.randint(3, width - max_c - 4)
    anchors = [(r + row_offset, c + col_offset) for r, c in anchors0]

    colors = random.sample(PALETTE, len(anchors))
    edge_set = _anchor_edges(anchors)
    center = (
        sum(r for r, _ in anchors) / len(anchors),
        sum(c for _, c in anchors) / len(anchors),
    )

    occupied = [{anchor} for anchor in anchors]
    grow_order = sorted(range(len(anchors)), key=lambda idx: _chebyshev(anchors[idx], (center[0], center[1])), reverse=True)
    size_choices = [3, 4, 5, 6, 7, 8]
    targets = [random.choice(size_choices) for _ in anchors]

    for idx in grow_order:
        grown = _grow_component(
            idx=idx,
            anchor=anchors[idx],
            occupied=occupied,
            anchors=anchors,
            edge_set=edge_set,
            shape=(height, width),
            center=center,
            target_area=targets[idx],
        )
        if grown is None:
            return None
        occupied[idx] = grown

    grid = np.zeros((height, width), dtype=int)
    for color, cells in zip(colors, occupied):
        for r, c in cells:
            grid[r, c] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _place_attempt()
        if grid is None:
            continue
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid dilation-degree ARC input")
