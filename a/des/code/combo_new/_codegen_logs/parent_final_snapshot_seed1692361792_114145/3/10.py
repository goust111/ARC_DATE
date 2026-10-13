from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 1000

Coord = tuple[int, int]


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _midpoint_distance2(point: Coord, a: Coord, b: Coord) -> int:
    return abs(2 * point[0] - a[0] - b[0]) + abs(2 * point[1] - a[1] - b[1])


def _segment_cells(start: Coord, end: Coord) -> list[Coord]:
    if start[0] == end[0]:
        step = 1 if end[1] >= start[1] else -1
        return [(start[0], c) for c in range(start[1], end[1] + step, step)]
    if start[1] == end[1]:
        step = 1 if end[0] >= start[0] else -1
        return [(r, start[1]) for r in range(start[0], end[0] + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _path_cells(a: Coord, bend: Coord, b: Coord) -> list[Coord]:
    first = _segment_cells(a, bend)
    second = _segment_cells(bend, b)
    return first + second[1:]


def _source_colors(grid: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)]


def _pair_descriptors(grid: np.ndarray) -> list[dict[str, object]]:
    greys = [tuple(map(int, p)) for p in np.argwhere(grid == GREY)]
    assert greys, "expected at least one grey indicator"

    descriptors: list[dict[str, object]] = []
    for color in _source_colors(grid):
        coords = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        assert len(coords) == 2, f"color {color} must appear exactly twice"
        a, b = coords
        assert a[0] != b[0] and a[1] != b[1], "paired pixels must differ in both row and column"

        midpoint_distances = [(_midpoint_distance2(grey, a, b), grey) for grey in greys]
        best_mid = min(dist for dist, _ in midpoint_distances)
        nearest = [grey for dist, grey in midpoint_distances if dist == best_mid]
        assert len(nearest) == 1, f"color {color} needs a unique nearest grey indicator"
        indicator = nearest[0]

        p1 = (a[0], b[1])
        p2 = (b[0], a[1])
        d1 = _manhattan(indicator, p1)
        d2 = _manhattan(indicator, p2)
        assert d1 != d2, f"grey indicator for color {color} must prefer one bend"
        bend = p1 if d1 < d2 else p2

        descriptors.append(
            {
                "color": color,
                "a": a,
                "b": b,
                "indicator": indicator,
                "bend": bend,
            }
        )
    return descriptors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored pixel pair with an L-shaped path chosen by the nearest grey indicator.

    Input structure:
    - The background is black.
    - Every non-grey foreground color appears exactly twice, forming a pair of singleton pixels.
    - The two pixels of a pair lie in different rows and different columns, and grey singleton pixels act as turn indicators.

    Rule:
    - For each color pair A and B, form the two candidate bend points (rA, cB) and (rB, cA).
    - Find the unique grey pixel nearest to the pair midpoint by Manhattan distance, allowing half-cell midpoints.
    - Compare that grey pixel's Manhattan distance to the two candidate bends, choose the nearer bend, and paint the full one-cell-wide L path from A to B through it.
    - Keep the original grey indicator pixels unchanged in the output.

    Key insight:
    - The task is a three-step inference: identify color pairs, realize they connect by one-bend orthogonal paths, and use grey proximity to choose which of the two possible bends is correct.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    out = g.copy()
    for desc in _pair_descriptors(g):
        color = int(desc["color"])
        path = _path_cells(desc["a"], desc["bend"], desc["b"])
        for r, c in path:
            if int(g[r, c]) == GREY:
                continue
            current = int(out[r, c])
            assert current in (BG, color), "paths must not collide with another color"
            out[r, c] = color
    return out


def _layout_options(pair_count: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for rows in (1, 2):
        for cols in (1, 2, 3):
            capacity = rows * cols
            if capacity < pair_count or capacity - pair_count > 1:
                continue
            max_zone_h = (30 - (rows - 1)) // rows
            max_zone_w = (30 - (cols - 1)) // cols
            if max_zone_h >= 7 and max_zone_w >= 7:
                options.append((rows, cols))
    return options


def _balanced_branches(pair_count: int, rng: random.Random) -> list[int]:
    branches = [i % 2 for i in range(pair_count)]
    if pair_count % 2 == 1 and rng.randrange(2):
        branches = [1 - v for v in branches]
    rng.shuffle(branches)
    return branches


def _corner_positions(top: int, left: int, r0: int, r1: int, c0: int, c1: int) -> dict[str, Coord]:
    return {
        "tl": (top + r0, left + c0),
        "tr": (top + r0, left + c1),
        "bl": (top + r1, left + c0),
        "br": (top + r1, left + c1),
    }


def _indicator_options(corner_name: str, corners: dict[str, Coord]) -> list[Coord]:
    r, c = corners[corner_name]
    if corner_name == "tl":
        return [(r - 1, c), (r, c - 1), (r - 1, c - 1)]
    if corner_name == "tr":
        return [(r - 1, c), (r, c + 1), (r - 1, c + 1)]
    if corner_name == "bl":
        return [(r + 1, c), (r, c - 1), (r + 1, c - 1)]
    if corner_name == "br":
        return [(r + 1, c), (r, c + 1), (r + 1, c + 1)]
    raise AssertionError(f"unknown corner {corner_name}")


def _place_pair(
    grid: np.ndarray,
    top: int,
    left: int,
    zone_h: int,
    zone_w: int,
    color: int,
    diag_type: int,
    branch: int,
    rng: random.Random,
) -> dict[str, Coord]:
    row_pairs = [(r0, r1) for r0 in range(1, zone_h - 2) for r1 in range(r0 + 2, zone_h - 1)]
    col_pairs = [(c0, c1) for c0 in range(1, zone_w - 2) for c1 in range(c0 + 2, zone_w - 1)]
    r0, r1 = rng.choice(row_pairs)
    c0, c1 = rng.choice(col_pairs)
    corners = _corner_positions(top, left, r0, r1, c0, c1)

    if diag_type == 0:
        endpoints = (corners["tl"], corners["br"])
        bend_names = ("tr", "bl")
    else:
        endpoints = (corners["tr"], corners["bl"])
        bend_names = ("tl", "br")

    bend_name = bend_names[branch]
    bend = corners[bend_name]
    indicator = rng.choice(_indicator_options(bend_name, corners))

    for r, c in endpoints:
        assert int(grid[r, c]) == BG
        grid[r, c] = color
    assert int(grid[indicator]) == BG
    grid[indicator] = GREY

    return {"indicator": indicator, "bend": bend}


def _validate_generated_grid(grid: np.ndarray, expected: dict[int, dict[str, Coord]]) -> bool:
    try:
        descriptors = _pair_descriptors(grid)
    except AssertionError:
        return False

    actual = {int(desc["color"]): desc for desc in descriptors}
    if set(actual) != set(expected):
        return False

    for color, meta in expected.items():
        desc = actual[color]
        if desc["indicator"] != meta["indicator"] or desc["bend"] != meta["bend"]:
            return False
        for r, c in _path_cells(desc["a"], desc["bend"], desc["b"]):
            if (r, c) == meta["indicator"]:
                return False
            value = int(grid[r, c])
            if value not in (BG, color):
                return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(2, 4)
    rows, cols = rng.choice(_layout_options(pair_count))

    max_zone_h = min(11, (30 - (rows - 1)) // rows)
    max_zone_w = min(11, (30 - (cols - 1)) // cols)
    if cols == 3:
        max_zone_w = min(max_zone_w, 9)
    zone_h = rng.randint(7, max_zone_h)
    zone_w = rng.randint(7, max_zone_w)

    grid_h = rows * zone_h + (rows - 1)
    grid_w = cols * zone_w + (cols - 1)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    all_cells = list(range(rows * cols))
    rng.shuffle(all_cells)
    used_cells = all_cells[:pair_count]

    colors = rng.sample(PALETTE, pair_count)
    branches = _balanced_branches(pair_count, rng)
    diag_types = [rng.randrange(2) for _ in range(pair_count)]

    expected: dict[int, dict[str, Coord]] = {}
    for cell_id, color, branch, diag_type in zip(used_cells, colors, branches, diag_types):
        zone_r = cell_id // cols
        zone_c = cell_id % cols
        top = zone_r * (zone_h + 1)
        left = zone_c * (zone_w + 1)
        expected[color] = _place_pair(
            grid=grid,
            top=top,
            left=left,
            zone_h=zone_h,
            zone_w=zone_w,
            color=color,
            diag_type=diag_type,
            branch=branch,
            rng=rng,
        )

    if not _validate_generated_grid(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
