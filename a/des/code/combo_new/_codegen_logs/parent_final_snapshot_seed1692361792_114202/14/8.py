from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BACKGROUND = 0
RELATIONS = ("same", "horizontal", "vertical", "rotation")
RELATION_TO_LEGEND_OFFSET = {
    "same": (0, 0),
    "horizontal": (0, 1),
    "vertical": (1, 0),
    "rotation": (1, 1),
}
ORIENTATIONS = ((0, 0), (0, 1), (1, 0), (1, 1))
TRIANGLE_OFFSETS = {
    (0, 0): ((0, 0), (-1, 0), (0, -1)),
    (0, 1): ((0, 0), (-1, 0), (0, 1)),
    (1, 0): ((0, 0), (1, 0), (0, -1)),
    (1, 1): ((0, 0), (1, 0), (0, 1)),
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver helpers ---------------


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat, minlength=10).argmax())


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[str, int], np.ndarray]:
    h, w = grid.shape
    assert h >= 2 and w >= 2, "grid must be at least 2x2"

    candidates: list[tuple[int, int, np.ndarray]] = []
    for r0, c0 in ((0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)):
        block = grid[r0 : r0 + 2, c0 : c0 + 2]
        if block.shape != (2, 2):
            continue
        if np.all(block != bg) and len({int(v) for v in block.flat}) == 4:
            candidates.append((r0, c0, block.copy()))

    assert len(candidates) == 1, "expected a unique 2x2 corner legend"
    r0, c0, block = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + 2, c0 : c0 + 2] = True
    legend = {
        "same": int(block[0, 0]),
        "horizontal": int(block[0, 1]),
        "vertical": int(block[1, 0]),
        "rotation": int(block[1, 1]),
    }
    return legend, mask


def _monochrome_components(grid: np.ndarray, bg: int, blocked: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = blocked.copy()
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))

            components.append((color, coords))

    return components


def _triangle_info(coords: list[tuple[int, int]]) -> dict[str, object]:
    assert len(coords) == 3, "triangle must have area 3"
    pts = np.asarray(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "triangle must occupy a 2x2 box"

    occupied = {(int(r - r0), int(c - c0)) for r, c in coords}
    all_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = all_cells - occupied
    assert len(missing) == 1, "triangle must be a 2x2 block with one missing corner"
    opening = next(iter(missing))
    vertex = (int(r0 + 1 - opening[0]), int(c0 + 1 - opening[1]))
    return {
        "opening": opening,
        "vertex": vertex,
        "coords": {tuple(map(int, cell)) for cell in coords},
    }


def _relation(opening_a: tuple[int, int], opening_b: tuple[int, int]) -> str:
    if opening_a == opening_b:
        return "same"
    if opening_a[0] == opening_b[0]:
        return "horizontal"
    if opening_a[1] == opening_b[1]:
        return "vertical"
    return "rotation"


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _path_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells = [(int(r0), int(c)) for c in _inclusive_range(int(c0), int(c1))]
    cells.extend((int(r), int(c1)) for r in list(_inclusive_range(int(r0), int(r1)))[1:])
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color triangle pair with a legend-colored horizontal-then-vertical line.

    Input structure:
    - The grid has a black background, a single 2x2 legend block in one corner, and several monochrome 3-pixel triangles.
    - Each triangle is an L-shaped 4-connected component that fills three cells of a 2x2 box; the missing corner gives its opening direction.
    - Every non-legend triangle color appears exactly twice, so the two components of that color form one pair.

    Rule:
    - Ignore the legend while finding connected components, then pair triangles by color.
    - For each pair, compare the two opening directions and classify the relation as same, horizontal mirror, vertical mirror, or 180-degree rotation.
    - Read the corresponding line color from the legend's top-left, top-right, bottom-left, or bottom-right cell.
    - Draw an L path from the earlier right-angle vertex to the later one, moving horizontally first and then vertically, while restoring the original triangles and legend afterward.

    Key insight:
    - Triangle color only tells you which two objects belong together; the visible 2x2 legend encodes a second layer that maps orientation relations to line colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    legend, legend_mask = _find_legend(g, bg)
    components = _monochrome_components(g, bg, legend_mask)
    assert components, "expected at least one triangle pair"

    triangles_by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for color, coords in components:
        triangles_by_color[int(color)].append(_triangle_info(coords))

    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for color, triangles in triangles_by_color.items():
        assert len(triangles) == 2, f"color {color} must appear as exactly two triangles"
        first, second = sorted(triangles, key=lambda tri: tri["vertex"])
        pairs.append((first, second))

    pairs.sort(key=lambda pair: pair[0]["vertex"])

    out = g.copy()
    for first, second in pairs:
        relation = _relation(first["opening"], second["opening"])
        line_color = legend[relation]
        for r, c in _path_cells(first["vertex"], second["vertex"]):
            out[r, c] = int(line_color)

    out[g != bg] = g[g != bg]
    return out


# --------------- generator helpers ---------------


GEN_ATTEMPTS = 300
GRID_HEIGHT_RANGE = (23, 30)
GRID_WIDTH_RANGE = (18, 30)
PAIR_COUNT_RANGE = (3, 4)
BAND_COUNT = 4
BAND_GAP = 1
MIN_BAND_HEIGHT = 4
MIN_HORIZONTAL_SPAN = 4


def _transform_opening(opening: tuple[int, int], relation: str) -> tuple[int, int]:
    r, c = opening
    if relation == "same":
        return opening
    if relation == "horizontal":
        return (r, 1 - c)
    if relation == "vertical":
        return (1 - r, c)
    assert relation == "rotation"
    return (1 - r, 1 - c)


def _triangle_cells(vertex: tuple[int, int], opening: tuple[int, int]) -> list[tuple[int, int]]:
    vr, vc = vertex
    return [(int(vr + dr), int(vc + dc)) for dr, dc in TRIANGLE_OFFSETS[opening]]


def _fits(cells: list[tuple[int, int]], top: int, bottom: int, width: int) -> bool:
    return all(top <= r <= bottom and 0 <= c < width for r, c in cells)


def _band_spans(start_row: int, end_row: int, rng: random.Random) -> list[tuple[int, int]]:
    usable = end_row - start_row + 1
    total_gap = BAND_GAP * (BAND_COUNT - 1)
    available = usable - total_gap
    assert available >= BAND_COUNT * MIN_BAND_HEIGHT

    heights = [MIN_BAND_HEIGHT] * BAND_COUNT
    extra = available - BAND_COUNT * MIN_BAND_HEIGHT
    for _ in range(extra):
        heights[rng.randrange(BAND_COUNT)] += 1

    spans = []
    row = start_row
    for height in heights:
        spans.append((row, row + height - 1))
        row += height + BAND_GAP
    return spans


def _valid_vertices(top: int, bottom: int, width: int, opening: tuple[int, int]) -> list[tuple[int, int]]:
    vertices = []
    for r in range(top, bottom + 1):
        for c in range(width):
            cells = _triangle_cells((r, c), opening)
            if _fits(cells, top, bottom, width):
                vertices.append((r, c))
    return vertices


def _choose_pair_for_band(
    top: int,
    bottom: int,
    width: int,
    relation: str,
    rng: random.Random,
) -> tuple[dict[str, object], dict[str, object]] | None:
    split_left_max = max(2, width // 2 - 2)
    split_right_min = min(width - 3, width // 2 + 1)
    valid = {opening: _valid_vertices(top, bottom, width, opening) for opening in ORIENTATIONS}
    candidates: list[tuple[dict[str, object], dict[str, object]]] = []

    for left_open in ORIENTATIONS:
        right_open = _transform_opening(left_open, relation)
        left_vertices = [v for v in valid[left_open] if v[1] <= split_left_max]
        right_vertices = [v for v in valid[right_open] if v[1] >= split_right_min]

        for left_vertex in left_vertices:
            left_cells = set(_triangle_cells(left_vertex, left_open))
            for right_vertex in right_vertices:
                if left_vertex[0] > right_vertex[0]:
                    continue
                if right_vertex[1] - left_vertex[1] < MIN_HORIZONTAL_SPAN:
                    continue
                right_cells = set(_triangle_cells(right_vertex, right_open))
                if left_cells & right_cells:
                    continue
                candidates.append(
                    (
                        {"vertex": left_vertex, "opening": left_open, "cells": left_cells},
                        {"vertex": right_vertex, "opening": right_open, "cells": right_cells},
                    )
                )

    if not candidates:
        return None
    return rng.choice(candidates)


def _legend_origin(height: int, width: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - 2
    if corner == "bl":
        return height - 2, 0
    return height - 2, width - 2


def _sample_relations(rng: random.Random) -> list[str]:
    count = rng.randint(*PAIR_COUNT_RANGE)
    chosen = list(RELATIONS) if count == 4 else rng.sample(list(RELATIONS), count)
    rng.shuffle(chosen)
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    corner = rng.choice(("tl", "tr", "bl", "br"))

    if corner in ("tl", "tr"):
        usable_top, usable_bottom = 2, height - 1
    else:
        usable_top, usable_bottom = 0, height - 3

    spans = _band_spans(usable_top, usable_bottom, rng)
    relations = _sample_relations(rng)
    active_bands = rng.sample(range(BAND_COUNT), len(relations))

    palette = list(range(1, 10))
    rng.shuffle(palette)
    pair_colors = palette[: len(relations)]
    line_colors = palette[len(relations) : len(relations) + 4]
    legend = {relation: color for relation, color in zip(RELATIONS, line_colors)}

    placements: list[tuple[int, str, dict[str, object], dict[str, object]]] = []
    for band_index, relation, pair_color in zip(active_bands, relations, pair_colors):
        top, bottom = spans[band_index]
        pair = _choose_pair_for_band(top, bottom, width, relation, rng)
        if pair is None:
            return None
        first, second = pair
        placements.append((pair_color, relation, first, second))

    grid = np.full((height, width), BACKGROUND, dtype=int)
    lr, lc = _legend_origin(height, width, corner)
    grid[lr + 0, lc + 0] = legend["same"]
    grid[lr + 0, lc + 1] = legend["horizontal"]
    grid[lr + 1, lc + 0] = legend["vertical"]
    grid[lr + 1, lc + 1] = legend["rotation"]

    for pair_color, _relation, first, second in placements:
        for r, c in first["cells"]:
            grid[r, c] = int(pair_color)
        for r, c in second["cells"]:
            grid[r, c] = int(pair_color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid triangle-pair input")
