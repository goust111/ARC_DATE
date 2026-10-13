from __future__ import annotations

from collections import deque
import random

import numpy as np


# --------------- helpers ---------------

MIN_GRID_SIZE = 8
MAX_GRID_SIZE = 18
MIN_RECT_SIDE = 2
MIN_RECT_COUNT = 6
MAX_RECT_COUNT = 14
GENERATION_RETRIES = 400
PALETTE = tuple(range(1, 10))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_rectangles(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    rects: list[tuple[int, int, int, int, int]] = []
    rect_ids = np.full((height, width), -1, dtype=int)

    for start_r in range(height):
        for start_c in range(width):
            if seen[start_r, start_c]:
                continue
            color = int(g[start_r, start_c])
            queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            r0 = r1 = start_r
            c0 = c1 = start_c

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                r0 = min(r0, r)
                r1 = max(r1, r)
                c0 = min(c0, c)
                c1 = max(c1, c)
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(cells) == expected_area, "each color region must be a filled rectangle"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be solid"

            rect_id = len(rects)
            rects.append((color, r0, c0, r1, c1))
            rect_ids[r0 : r1 + 1, c0 : c1 + 1] = rect_id

    assert np.all(rect_ids >= 0), "rectangles must cover the whole grid"
    return rects, rect_ids


def _build_adjacency(rect_ids: np.ndarray, n_rects: int) -> list[set[int]]:
    height, width = map(int, rect_ids.shape)
    neighbors = [set() for _ in range(n_rects)]

    for r in range(height - 1):
        for c in range(width):
            a = int(rect_ids[r, c])
            b = int(rect_ids[r + 1, c])
            if a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)

    for r in range(height):
        for c in range(width - 1):
            a = int(rect_ids[r, c])
            b = int(rect_ids[r, c + 1])
            if a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)

    return neighbors


def _split_rectangles(
    height: int,
    width: int,
    target_count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = [(0, 0, height, width)]

    while len(rects) < target_count:
        splittable = [
            i
            for i, (_r0, _c0, h, w) in enumerate(rects)
            if h >= 2 * MIN_RECT_SIDE or w >= 2 * MIN_RECT_SIDE
        ]
        if not splittable:
            break

        weights = [rects[i][2] * rects[i][3] for i in splittable]
        rect_index = rng.choices(splittable, weights=weights, k=1)[0]
        r0, c0, h, w = rects.pop(rect_index)

        orientations: list[str] = []
        if h >= 2 * MIN_RECT_SIDE:
            orientations.append("horizontal")
        if w >= 2 * MIN_RECT_SIDE:
            orientations.append("vertical")
        assert orientations

        if len(orientations) == 2:
            horiz_weight = float(h)
            vert_weight = float(w)
            orientation = rng.choices(orientations, weights=[horiz_weight, vert_weight], k=1)[0]
        else:
            orientation = orientations[0]

        if orientation == "horizontal":
            cut = rng.randint(MIN_RECT_SIDE, h - MIN_RECT_SIDE)
            rects.append((r0, c0, cut, w))
            rects.append((r0 + cut, c0, h - cut, w))
        else:
            cut = rng.randint(MIN_RECT_SIDE, w - MIN_RECT_SIDE)
            rects.append((r0, c0, h, cut))
            rects.append((r0, c0 + cut, h, w - cut))

    if len(rects) != target_count:
        return None
    return rects


def _rectangles_to_ids(
    rects: list[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> np.ndarray:
    rect_ids = np.full((height, width), -1, dtype=int)
    for rect_id, (r0, c0, h, w) in enumerate(rects):
        rect_ids[r0 : r0 + h, c0 : c0 + w] = rect_id
    assert np.all(rect_ids >= 0), "partition must cover the full canvas"
    return rect_ids


def _color_partition(neighbors: list[set[int]], rng: random.Random) -> dict[int, int] | None:
    n_rects = len(neighbors)
    max_palette_size = min(7, len(PALETTE), n_rects)
    min_palette_size = min(4, max_palette_size)
    palette_size = rng.randint(min_palette_size, max_palette_size)
    palette = rng.sample(PALETTE, palette_size)
    order = sorted(range(n_rects), key=lambda idx: (len(neighbors[idx]), idx), reverse=True)
    assignment: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == n_rects:
            return True
        rect_id = order[pos]
        used = {assignment[nbr] for nbr in neighbors[rect_id] if nbr in assignment}
        options = [color for color in palette if color not in used]
        rng.shuffle(options)
        for color in options:
            assignment[rect_id] = color
            if backtrack(pos + 1):
                return True
            del assignment[rect_id]
        return False

    if not backtrack(0):
        return None
    return assignment


def _paint_partition(
    rects: list[tuple[int, int, int, int]],
    assignment: dict[int, int],
    height: int,
    width: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for rect_id, (r0, c0, h, w) in enumerate(rects):
        grid[r0 : r0 + h, c0 : c0 + w] = assignment[rect_id]
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    max_rects = min(MAX_RECT_COUNT, (height * width) // (MIN_RECT_SIDE * MIN_RECT_SIDE))
    if max_rects < MIN_RECT_COUNT:
        return None

    target_count = rng.randint(MIN_RECT_COUNT, max_rects)
    rects = _split_rectangles(height, width, target_count, rng)
    if rects is None:
        return None

    rect_ids = _rectangles_to_ids(rects, height, width)
    neighbors = _build_adjacency(rect_ids, len(rects))
    degrees = [len(adj) for adj in neighbors]
    odd_count = sum(degree % 2 == 1 for degree in degrees)
    even_count = len(rects) - odd_count

    if odd_count == 0 or even_count == 0:
        return None
    if min(odd_count, even_count) < 2:
        return None
    if max(degrees) < 3:
        return None

    assignment = _color_partition(neighbors, rng)
    if assignment is None:
        return None

    grid = _paint_partition(rects, assignment, height, width)
    extracted, _extracted_ids = _extract_rectangles(grid)
    if len(extracted) != len(rects):
        return None
    generated_boxes = sorted((r0, c0, r0 + h - 1, c0 + w - 1) for r0, c0, h, w in rects)
    extracted_boxes = sorted((r0, c0, r1, c1) for _color, r0, c0, r1, c1 in extracted)
    if extracted_boxes != generated_boxes:
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each rectangular tile black exactly when it has an odd number of edge-sharing neighbors.

    Input structure:
    - The grid is fully tiled by solid-color axis-aligned rectangles with no gaps.
    - Tiles meet only by shared sides or corners, and the outer border acts like a wall.
    - Black is the background/death color, while input tiles may use any visible colors.

    Rule:
    - Identify every 4-connected monochrome region and verify that each region is a filled rectangle.
    - For each rectangle, count how many other rectangles share at least one side-adjacent cell with it.
    - If that neighbor count is even, keep the rectangle unchanged; if it is odd, recolor the whole rectangle black.

    Key insight:
    - The decisive property is not color or size but the parity of each rectangle's adjacency degree in the mosaic.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects, rect_ids = _extract_rectangles(g)
    neighbors = _build_adjacency(rect_ids, len(rects))

    out = g.copy()
    for rect_id, (_color, r0, c0, r1, c1) in enumerate(rects):
        if len(neighbors[rect_id]) % 2 == 1:
            out[r0 : r1 + 1, c0 : c1 + 1] = 0

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangular-mosaic input")
