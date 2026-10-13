from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
PairSpec = tuple[int, str, Coord, Coord]


RELATIONS = ("horizontal", "vertical", "diagonal", "lshape")

# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ordered_pair(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    return (a, b) if a <= b else (b, a)


def _pair_relation(a: Coord, b: Coord) -> str:
    (r0, c0), (r1, c1) = _ordered_pair(a, b)
    if r0 == r1:
        return "horizontal"
    if c0 == c1:
        return "vertical"
    if abs(r1 - r0) == abs(c1 - c0):
        return "diagonal"
    return "lshape"


def _path_cells(a: Coord, b: Coord) -> list[Coord]:
    (r0, c0), (r1, c1) = _ordered_pair(a, b)

    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]

    if c0 == c1:
        return [(r, c0) for r in range(r0, r1 + 1)]

    if abs(r1 - r0) == abs(c1 - c0):
        dc = 1 if c1 > c0 else -1
        return [(r0 + i, c0 + i * dc) for i in range(r1 - r0 + 1)]

    cells = []
    step_c = 1 if c1 > c0 else -1
    for c in range(c0, c1 + step_c, step_c):
        cells.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cells.append((r, c1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair using a line style determined by their spatial relation.

    Input structure:
    - The background is the dominant color.
    - Several foreground colors appear exactly twice each, creating same-color pairs.
    - No other foreground pixels lie on the intended connector path for any pair.

    Rule:
    - Same row: draw a solid horizontal connector between the markers.
    - Same column: draw a solid vertical connector between the markers.
    - Same diagonal: draw a diagonal connector between the markers.
    - Otherwise: draw an L-shaped connector by moving horizontally first, then vertically.
    - Only background cells are painted; original markers stay unchanged.

    Key insight:
    - The relative alignment of each pair determines which connector geometry to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground pair"

    out = g.copy()
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))

        for r, c in _path_cells(a, b):
            if int(g[r, c]) == bg:
                out[r, c] = color

    return out


# --------------- generator ---------------

# Canvas and palette.
GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (14, 20)
BG_COLORS = tuple(range(10))

# Pair counts and placement.
BASE_RELATIONS = list(RELATIONS)
MIN_PAIR_COUNT = 4
MAX_PAIR_COUNT = 6
MIN_SPAN = 3
MAX_SPAN = 8
CLEARANCE_RADIUS = 1
PLACEMENT_TRIES_PER_PAIR = 250
GENERATION_RETRIES = 400


def _expanded_cells(cells: list[Coord], shape: tuple[int, int], radius: int = CLEARANCE_RADIUS) -> set[Coord]:
    height, width = shape
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    expanded.add((nr, nc))
    return expanded


def _sample_horizontal(height: int, width: int) -> tuple[Coord, Coord] | None:
    if width < MIN_SPAN:
        return None
    span = random.randint(MIN_SPAN, min(MAX_SPAN, width))
    r = random.randrange(height)
    c0 = random.randint(0, width - span)
    return (r, c0), (r, c0 + span - 1)


def _sample_vertical(height: int, width: int) -> tuple[Coord, Coord] | None:
    if height < MIN_SPAN:
        return None
    span = random.randint(MIN_SPAN, min(MAX_SPAN, height))
    r0 = random.randint(0, height - span)
    c = random.randrange(width)
    return (r0, c), (r0 + span - 1, c)


def _sample_diagonal(height: int, width: int) -> tuple[Coord, Coord] | None:
    if min(height, width) < MIN_SPAN:
        return None
    span = random.randint(MIN_SPAN, min(MAX_SPAN, height, width))
    r0 = random.randint(0, height - span)
    if random.random() < 0.5:
        c0 = random.randint(0, width - span)
        return (r0, c0), (r0 + span - 1, c0 + span - 1)
    c0 = random.randint(span - 1, width - 1)
    return (r0, c0), (r0 + span - 1, c0 - span + 1)


def _sample_lshape(height: int, width: int) -> tuple[Coord, Coord] | None:
    if height < MIN_SPAN or width < MIN_SPAN:
        return None

    max_dr = min(MAX_SPAN, height - 1)
    max_dc = min(MAX_SPAN, width - 1)
    if max_dr < 2 or max_dc < 2:
        return None

    for _ in range(50):
        dr = random.randint(2, max_dr)
        dc = random.randint(2, max_dc)
        if dr == dc:
            continue
        r0 = random.randint(0, height - dr - 1)
        if random.random() < 0.5:
            c0 = random.randint(0, width - dc - 1)
            c1 = c0 + dc
        else:
            c0 = random.randint(dc, width - 1)
            c1 = c0 - dc
        return (r0, c0), (r0 + dr, c1)

    return None


def _sample_pair(relation: str, height: int, width: int) -> tuple[Coord, Coord] | None:
    if relation == "horizontal":
        return _sample_horizontal(height, width)
    if relation == "vertical":
        return _sample_vertical(height, width)
    if relation == "diagonal":
        return _sample_diagonal(height, width)
    if relation == "lshape":
        return _sample_lshape(height, width)
    raise ValueError(f"unknown relation: {relation}")


def _place_pairs(height: int, width: int, colors: list[int], relations: list[str]) -> list[PairSpec] | None:
    blocked: set[Coord] = set()
    specs: list[PairSpec] = []

    for color, relation in zip(colors, relations):
        placed = False
        for _ in range(PLACEMENT_TRIES_PER_PAIR):
            endpoints = _sample_pair(relation, height, width)
            if endpoints is None:
                return None
            a, b = endpoints
            path = _path_cells(a, b)

            if _pair_relation(a, b) != relation:
                continue
            if any(cell in blocked for cell in path):
                continue

            blocked.update(_expanded_cells(path, (height, width)))
            specs.append((color, relation, a, b))
            placed = True
            break

        if not placed:
            return None

    return specs


def _render_input(height: int, width: int, bg: int, specs: list[PairSpec]) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for color, _relation, a, b in specs:
        grid[a] = color
        grid[b] = color
    return grid


def _validate_generated(grid: np.ndarray, specs: list[PairSpec]) -> bool:
    bg = _background_color(grid)
    if int(np.count_nonzero(grid != bg)) != 2 * len(specs):
        return False

    for color, relation, a, b in specs:
        coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
        if coords != {a, b}:
            return False
        if _pair_relation(a, b) != relation:
            return False
        for cell in _path_cells(a, b):
            if cell not in {a, b} and int(grid[cell]) != bg:
                return False

    out = main(grid)
    for color, _relation, a, b in specs:
        for cell in _path_cells(a, b):
            if int(out[cell]) != color:
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        max_pairs = min(MAX_PAIR_COUNT, 9)
        pair_count = random.randint(MIN_PAIR_COUNT, max_pairs)

        bg = random.choice(BG_COLORS)
        colors = random.sample([color for color in range(10) if color != bg], k=pair_count)

        relations = BASE_RELATIONS.copy()
        relations.extend(random.choice(RELATIONS) for _ in range(pair_count - len(BASE_RELATIONS)))
        random.shuffle(relations)

        specs = _place_pairs(height, width, colors, relations)
        if specs is None:
            continue

        grid = _render_input(height, width, bg, specs)
        if _validate_generated(grid, specs):
            return grid

    raise RuntimeError("failed to generate a valid connector-pair grid")
