from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
GREY = 5
MARKER_COLORS = (2, 3, 4)
DISTRACTOR_COLORS = (1, 6, 7, 8, 9)
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_ATTEMPTS = 400


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _corner_bracket_match(coords: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    pts = np.array(sorted(coords), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3 or len(coords) != height + width - 1:
        return None

    candidates = [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]
    for expected in candidates:
        if coords == expected:
            return int(r0), int(c0), int(r1), int(c1)
    return None


def _find_bracket_and_marker(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int, int, int], tuple[int, int], int]:
    grey_components = _components_from_mask(grid == GREY)
    matches = []
    for comp in grey_components:
        coords = set(comp)
        bbox = _corner_bracket_match(coords)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        markers = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) in coords:
                    continue
                color = int(grid[r, c])
                if color not in (BG, GREY):
                    markers.append((r, c, color))
        if len(markers) == 1:
            mr, mc, color = markers[0]
            matches.append((coords, bbox, (mr, mc), color))

    assert len(matches) == 1, "expected exactly one grey corner bracket with one internal marker"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the marker color inside the grey corner bracket and split same-color components by size.

    Input structure:
    - The grid has black background and exactly one grey 4-connected corner bracket shaped like an L.
    - Inside the bracket's bounding box, but not on bracket cells, there is exactly one marker pixel with color C.
    - Outside that bounding box, several other pixels of color C form 4-connected components of mixed sizes.

    Rule:
    - Find the grey corner bracket and read the marker color C from the unique non-black, non-grey pixel inside its bounding box.
    - Ignore everything inside the bracket bounding box, then find every 4-connected component of color C elsewhere in the grid.
    - Recolor each size-1 component to blue, erase each larger component to black, and keep the bracket, marker, and all non-C pixels unchanged.

    Key insight:
    - The bracket first selects which color matters, and only then component size decides whether that selected color is repainted or erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bracket_coords, bbox, _marker_pos, marker_color = _find_bracket_and_marker(g)
    r0, c0, r1, c1 = bbox

    target_mask = g == marker_color
    target_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    components = _components_from_mask(target_mask)

    out = g.copy()
    for comp in components:
        if len(comp) == 1:
            rr, cc = comp[0]
            out[rr, cc] = BLUE
        else:
            for rr, cc in comp:
                out[rr, cc] = BG
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _transform_shape(shape: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    cells = set(shape)
    for _ in range(rng.randrange(4)):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < 0.5:
        cells = {(r, -c) for r, c in cells}
    if rng.random() < 0.5:
        cells = {(-r, c) for r, c in cells}
    return _normalize_shape(cells)


def _random_polyomino(size: int, rng: random.Random) -> list[tuple[int, int]]:
    assert size >= 1
    if size == 1:
        return [(0, 0)]
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for dr, dc in N4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(rng.choice(frontier))
        shape = _transform_shape(_normalize_shape(cells), rng)
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        if max(rows) <= 3 and max(cols) <= 3:
            return shape
    return [(i, 0) for i in range(size)]


def _bracket_cells(top: int, left: int, height: int, width: int, orientation: str) -> set[tuple[int, int]]:
    bottom = top + height - 1
    right = left + width - 1
    if orientation == "tl":
        return {(top, c) for c in range(left, right + 1)} | {(r, left) for r in range(top, bottom + 1)}
    if orientation == "tr":
        return {(top, c) for c in range(left, right + 1)} | {(r, right) for r in range(top, bottom + 1)}
    if orientation == "bl":
        return {(bottom, c) for c in range(left, right + 1)} | {(r, left) for r in range(top, bottom + 1)}
    if orientation == "br":
        return {(bottom, c) for c in range(left, right + 1)} | {(r, right) for r in range(top, bottom + 1)}
    raise ValueError(f"unknown orientation: {orientation}")


def _place_shape(
    grid: np.ndarray,
    shape: list[tuple[int, int]],
    color: int,
    rng: random.Random,
    forbidden: set[tuple[int, int]],
    border: int = 1,
) -> bool:
    h, w = grid.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    top_min = border
    left_min = border
    top_max = h - border - shape_h
    left_max = w - border - shape_w
    if top_max < top_min or left_max < left_min:
        return False

    for _ in range(300):
        top = rng.randint(top_min, top_max)
        left = rng.randint(left_min, left_max)
        coords = {(top + r, left + c) for r, c in shape}
        if any(coord in forbidden for coord in coords):
            continue
        ok = True
        for rr, cc in coords:
            if int(grid[rr, cc]) != BG:
                ok = False
                break
            for dr, dc in N8:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords and int(grid[nr, nc]) != BG:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for rr, cc in coords:
            grid[rr, cc] = color
        return True
    return False


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        bracket_coords, bbox, marker_pos, marker_color = _find_bracket_and_marker(grid)
    except AssertionError:
        return False

    if marker_color not in MARKER_COLORS:
        return False

    r0, c0, r1, c1 = bbox
    target_mask = grid == marker_color
    target_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components_from_mask(target_mask)
    singletons = sum(len(comp) == 1 for comp in comps)
    multis = [comp for comp in comps if len(comp) >= 3]
    if singletons < 2 or not multis:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if int(out[marker_pos]) != marker_color:
        return False
    for rr, cc in bracket_coords:
        if int(out[rr, cc]) != GREY:
            return False

    changed_to_blue = sum(int(out[rr, cc]) == BLUE for comp in comps if len(comp) == 1 for rr, cc in comp)
    changed_to_bg = sum(int(out[rr, cc]) == BG for comp in multis for rr, cc in comp)
    if changed_to_blue != singletons or changed_to_bg == 0:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(10, 16)
    width = rng.randint(10, 16)
    grid = np.zeros((height, width), dtype=int)

    bracket_h, bracket_w = rng.choice([(3, 3), (3, 4), (4, 3), (4, 4)])
    bracket_top = rng.randint(1, height - bracket_h - 2)
    bracket_left = rng.randint(1, width - bracket_w - 2)
    bracket_bbox = (bracket_top, bracket_left, bracket_top + bracket_h - 1, bracket_left + bracket_w - 1)
    bracket = _bracket_cells(bracket_top, bracket_left, bracket_h, bracket_w, rng.choice(("tl", "tr", "bl", "br")))
    for rr, cc in bracket:
        grid[rr, cc] = GREY

    interior = [
        (r, c)
        for r in range(bracket_bbox[0], bracket_bbox[2] + 1)
        for c in range(bracket_bbox[1], bracket_bbox[3] + 1)
        if (r, c) not in bracket
    ]
    marker_color = rng.choice(MARKER_COLORS)
    marker_pos = rng.choice(interior)
    grid[marker_pos] = marker_color

    forbidden = {
        (r, c)
        for r in range(bracket_bbox[0], bracket_bbox[2] + 1)
        for c in range(bracket_bbox[1], bracket_bbox[3] + 1)
    }

    multi_shapes = [_random_polyomino(rng.randint(3, 6), rng) for _ in range(rng.randint(1, 2))]
    singletons = [[(0, 0)] for _ in range(rng.randint(2, 3))]
    target_shapes = multi_shapes + singletons
    rng.shuffle(target_shapes)
    for shape in sorted(target_shapes, key=len, reverse=True):
        if not _place_shape(grid, shape, marker_color, rng, forbidden):
            return None

    distractor_count = rng.randint(2, 5)
    for _ in range(distractor_count):
        color = rng.choice([c for c in DISTRACTOR_COLORS if c != marker_color])
        size = rng.randint(1, 4)
        shape = _random_polyomino(size, rng)
        if not _place_shape(grid, shape, color, rng, forbidden):
            return None

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
