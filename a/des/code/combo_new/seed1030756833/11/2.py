import random
from collections import defaultdict

import numpy as np


AZURE = 8
RED = 2
GREEN = 3
YELLOW = 4

DIR_ORDER = ("h", "v", "d")
DIR_TO_COLOR = {"h": RED, "v": GREEN, "d": YELLOW}
LEGEND_COLORS = {RED, GREEN, YELLOW}
PAIR_COLORS = [0, 1, 5, 6, 7, 9]


def _centered_span(center: int, width: int, limit: int) -> range:
    start = center - width // 2
    stop = start + width
    assert 0 <= start < stop <= limit, "thick bridge must fit inside the grid"
    return range(start, stop)


def _legend_colors_for_dirs(directions: set[str]) -> list[int]:
    return [DIR_TO_COLOR[d] for d in DIR_ORDER if d in directions]


def _edge_strip_candidates(g: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    height, width = g.shape
    candidates = []

    for r in (0, height - 1):
        c = 0
        while c < width:
            if int(g[r, c]) not in LEGEND_COLORS:
                c += 1
                continue
            start = c
            while c < width and int(g[r, c]) in LEGEND_COLORS:
                c += 1
            coords = tuple((r, cc) for cc in range(start, c))
            colors = [int(g[pos]) for pos in coords]
            if 1 <= len(coords) <= 3 and len(set(colors)) == len(colors):
                candidates.append(coords)

    for c in (0, width - 1):
        r = 0
        while r < height:
            if int(g[r, c]) not in LEGEND_COLORS:
                r += 1
                continue
            start = r
            while r < height and int(g[r, c]) in LEGEND_COLORS:
                r += 1
            coords = tuple((rr, c) for rr in range(start, r))
            colors = [int(g[pos]) for pos in coords]
            if 1 <= len(coords) <= 3 and len(set(colors)) == len(colors):
                candidates.append(coords)

    deduped = []
    seen = set()
    for coords in candidates:
        key = frozenset(coords)
        if key not in seen:
            deduped.append(coords)
            seen.add(key)
    return deduped


def _pair_direction(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        return "h"
    if c1 == c2:
        return "v"
    if abs(r1 - r2) == abs(c1 - c2):
        return "d"
    return None


def _classify_pairs(
    g: np.ndarray, strip: tuple[tuple[int, int], ...], bg: int
) -> dict[str, list[tuple[int, tuple[int, int], tuple[int, int]]]] | None:
    strip_set = set(strip)
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(g != bg):
        pos = (int(r), int(c))
        if pos in strip_set:
            continue
        by_color[int(g[pos])].append(pos)

    if not by_color:
        return None

    pairs: dict[str, list[tuple[int, tuple[int, int], tuple[int, int]]]] = {
        "h": [],
        "v": [],
        "d": [],
    }
    for color, coords in by_color.items():
        if len(coords) != 2:
            return None
        a, b = coords
        direction = _pair_direction(a, b)
        if direction is None:
            return None
        pairs[direction].append((color, a, b))

    directions = {d for d, items in pairs.items() if items}
    expected_legend = _legend_colors_for_dirs(directions)
    actual_legend = [int(g[pos]) for pos in strip]
    if actual_legend != expected_legend:
        return None
    return pairs


def _paint_if_empty(
    out: np.ndarray,
    original: np.ndarray,
    r: int,
    c: int,
    color: int,
    bg: int,
) -> None:
    if int(original[r, c]) == bg and int(out[r, c]) == bg:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color pixel pairs with direction-colored bridges whose thickness is set by pair counts.

    Input structure:
    - The background is azure.
    - Away from the edge legend, every non-background color appears as exactly
      two pixels, forming one horizontal, vertical, or diagonal pair.
    - A contiguous red/green/yellow color strip lies on one grid edge; it lists
      the bridge colors for the direction classes that actually appear.

    Rule:
    - Classify the same-color pairs as horizontal, vertical, or diagonal and
      count how many pairs fall in each class.
    - Draw red horizontal bridges between horizontal endpoints with thickness
      equal to the number of horizontal pairs.
    - Draw green vertical bridges between vertical endpoints with thickness
      equal to the number of vertical pairs.
    - Draw one-cell-thick yellow diagonal bridges between diagonal endpoints.
    - Preserve all original colored pixels, including the endpoints and legend.

    Key insight:
    - The scattered pairs must first be grouped by direction and counted; those
      counts then become the geometric thickness of the corresponding bridges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height <= 30 and width <= 30, "ARC grids must be at most 30x30"
    bg = AZURE
    assert np.any(g == bg), "expected azure background"

    valid = []
    for strip in _edge_strip_candidates(g):
        pairs = _classify_pairs(g, strip, bg)
        if pairs is not None:
            valid.append((strip, pairs))
    assert len(valid) == 1, "expected one unambiguous edge color strip"

    _strip, pairs = valid[0]
    out = g.copy()

    n_h = len(pairs["h"])
    for _color, a, b in pairs["h"]:
        r = a[0]
        c1, c2 = sorted((a[1], b[1]))
        for rr in _centered_span(r, n_h, height):
            for cc in range(c1 + 1, c2):
                _paint_if_empty(out, g, rr, cc, RED, bg)

    n_v = len(pairs["v"])
    for _color, a, b in pairs["v"]:
        c = a[1]
        r1, r2 = sorted((a[0], b[0]))
        for cc in _centered_span(c, n_v, width):
            for rr in range(r1 + 1, r2):
                _paint_if_empty(out, g, rr, cc, GREEN, bg)

    for _color, a, b in pairs["d"]:
        r1, c1 = a
        r2, c2 = b
        dr = 1 if r2 > r1 else -1
        dc = 1 if c2 > c1 else -1
        steps = abs(r2 - r1)
        for k in range(1, steps):
            _paint_if_empty(out, g, r1 + dr * k, c1 + dc * k, YELLOW, bg)

    return out


def _random_direction_set() -> tuple[str, ...]:
    subsets = [
        ("h",),
        ("v",),
        ("d",),
        ("h", "v"),
        ("h", "d"),
        ("v", "d"),
        ("h", "v", "d"),
    ]
    return random.choice(subsets)


def _random_counts(directions: tuple[str, ...]) -> dict[str, int]:
    counts = {d: 1 for d in directions}
    max_total = min(len(PAIR_COLORS), 3 * len(directions))
    extras = random.randint(0, max_total - len(directions))
    for _ in range(extras):
        choices = [d for d in directions if counts[d] < 3]
        if not choices:
            break
        counts[random.choice(choices)] += 1
    return counts


def _place_legend(grid: np.ndarray, directions: tuple[str, ...]) -> set[tuple[int, int]]:
    height, width = grid.shape
    colors = _legend_colors_for_dirs(set(directions))
    horizontal = random.choice([True, False])
    coords = set()
    if horizontal:
        r = random.choice([0, height - 1])
        start = random.randint(1, width - len(colors) - 1)
        for i, color in enumerate(colors):
            grid[r, start + i] = color
            coords.add((r, start + i))
    else:
        c = random.choice([0, width - 1])
        start = random.randint(1, height - len(colors) - 1)
        for i, color in enumerate(colors):
            grid[start + i, c] = color
            coords.add((start + i, c))
    return coords


def _bridge_cells_for_horizontal(
    r: int, c1: int, c2: int, width: int, shape: tuple[int, int]
) -> set[tuple[int, int]] | None:
    height, grid_width = shape
    try:
        rows = _centered_span(r, width, height)
    except AssertionError:
        return None
    if min(rows) < 1 or max(rows) > height - 2:
        return None
    if not (1 <= c1 < c2 <= grid_width - 2):
        return None
    return {(rr, cc) for rr in rows for cc in range(c1 + 1, c2)}


def _bridge_cells_for_vertical(
    r1: int, r2: int, c: int, width: int, shape: tuple[int, int]
) -> set[tuple[int, int]] | None:
    height, grid_width = shape
    try:
        cols = _centered_span(c, width, grid_width)
    except AssertionError:
        return None
    if min(cols) < 1 or max(cols) > grid_width - 2:
        return None
    if not (1 <= r1 < r2 <= height - 2):
        return None
    return {(rr, cc) for rr in range(r1 + 1, r2) for cc in cols}


def _bridge_cells_for_diagonal(
    a: tuple[int, int], b: tuple[int, int], shape: tuple[int, int]
) -> set[tuple[int, int]] | None:
    height, width = shape
    r1, c1 = a
    r2, c2 = b
    if not (1 <= r1 <= height - 2 and 1 <= r2 <= height - 2):
        return None
    if not (1 <= c1 <= width - 2 and 1 <= c2 <= width - 2):
        return None
    if abs(r1 - r2) != abs(c1 - c2) or abs(r1 - r2) < 3:
        return None
    dr = 1 if r2 > r1 else -1
    dc = 1 if c2 > c1 else -1
    return {(r1 + dr * k, c1 + dc * k) for k in range(1, abs(r2 - r1))}


def _try_place_pair(
    grid: np.ndarray,
    direction: str,
    color: int,
    thickness: int,
    reserved: set[tuple[int, int]],
) -> bool:
    height, width = grid.shape

    for _ in range(250):
        if direction == "h":
            r = random.randint(2, height - 3)
            c1 = random.randint(1, width - 5)
            c2 = random.randint(c1 + 3, width - 2)
            endpoints = {(r, c1), (r, c2)}
            bridge = _bridge_cells_for_horizontal(r, c1, c2, thickness, grid.shape)
        elif direction == "v":
            c = random.randint(2, width - 3)
            r1 = random.randint(1, height - 5)
            r2 = random.randint(r1 + 3, height - 2)
            endpoints = {(r1, c), (r2, c)}
            bridge = _bridge_cells_for_vertical(r1, r2, c, thickness, grid.shape)
        else:
            distance = random.randint(3, min(height, width) - 3)
            if random.choice([True, False]):
                r1 = random.randint(1, height - distance - 2)
                c1 = random.randint(1, width - distance - 2)
                r2, c2 = r1 + distance, c1 + distance
            else:
                r1 = random.randint(1, height - distance - 2)
                c1 = random.randint(distance + 1, width - 2)
                r2, c2 = r1 + distance, c1 - distance
            endpoints = {(r1, c1), (r2, c2)}
            bridge = _bridge_cells_for_diagonal((r1, c1), (r2, c2), grid.shape)

        if bridge is None:
            continue
        if endpoints & reserved or bridge & reserved:
            continue
        for r, c in endpoints:
            grid[r, c] = color
        reserved.update(endpoints)
        reserved.update(bridge)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 24)
    width = random.randint(12, 24)
    grid = np.full((height, width), AZURE, dtype=int)

    directions = _random_direction_set()
    counts = _random_counts(directions)
    reserved = _place_legend(grid, directions)

    total_pairs = sum(counts.values())
    colors = random.sample(PAIR_COLORS, total_pairs)
    tasks = []
    for direction in directions:
        tasks.extend([direction] * counts[direction])
    random.shuffle(tasks)
    tasks.sort(key=lambda d: counts[d] if d in ("h", "v") else 1, reverse=True)

    for color, direction in zip(colors, tasks):
        if not _try_place_pair(grid, direction, color, counts[direction], reserved):
            return None

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-pair input")
