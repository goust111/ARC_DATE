from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
MAGENTA = 6
ORANGE = 7

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], p: Coord) -> bool:
    r, c = p
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _aligned(a: Coord, b: Coord) -> bool:
    return a[0] == b[0] or a[1] == b[1]


def _rotate_ccw(point: Coord, pivot: Coord) -> Coord:
    dr = point[0] - pivot[0]
    dc = point[1] - pivot[1]
    return pivot[0] - dc, pivot[1] + dr


def _trace(shape: tuple[int, int], source: Coord, pivots: tuple[Coord, ...]) -> tuple[list[Coord], list[Coord]]:
    used: set[Coord] = set()
    pivot_steps: list[Coord] = []
    landings: list[Coord] = []
    current = source

    while True:
        candidates = [p for p in pivots if p not in used and _aligned(current, p)]
        if not candidates:
            break
        assert len(candidates) == 1, "current point must align with at most one unused pivot"

        pivot = candidates[0]
        assert pivot != current, "source/landing cannot occupy a pivot"
        landing = _rotate_ccw(current, pivot)
        assert _in_bounds(shape, landing), "rotation landing must stay inside the grid"
        assert landing not in pivots, "rotation landing cannot overwrite a blue pivot"

        used.add(pivot)
        pivot_steps.append(pivot)
        landings.append(landing)
        current = landing

    return pivot_steps, landings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay-rotate the magenta source around aligned blue pivots, marking the relay landings.

    Input structure:
    - The grid has a black background, several blue single-pixel pivots, and exactly one magenta source pixel.
    - The source aligns by row or column with exactly one blue pivot; later landings align with at most one unused pivot.
    - Used pivots remain visible but cannot trigger another rotation.

    Rule:
    - Rotate the current magenta/relay point 90 degrees counterclockwise around its aligned unused blue pivot.
    - If the landing aligns with a different unused pivot, repeat the same rotation from that landing.
    - Clear the original magenta source, color every non-final landing yellow, color the final landing orange, and leave all blue pivots fixed.

    Key insight:
    - The first rotation is only the entry step; the answer is determined by the chain of newly aligned unused pivots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, MAGENTA}), "unexpected input colors"

    source_cells = np.argwhere(g == MAGENTA)
    assert len(source_cells) == 1, "expected exactly one magenta source pixel"
    source = tuple(map(int, source_cells[0]))

    pivots = tuple(sorted(tuple(map(int, p)) for p in np.argwhere(g == BLUE)))
    assert pivots, "expected at least one blue pivot"

    first = [p for p in pivots if _aligned(source, p)]
    assert len(first) == 1, "source must align with exactly one pivot"

    _pivot_steps, landings = _trace(g.shape, source, pivots)
    assert landings, "expected at least one rotation"

    out = g.copy()
    out[source] = BLACK
    for p in landings[:-1]:
        out[p] = YELLOW
    out[landings[-1]] = ORANGE
    return out


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _separated(point: Coord, points: set[Coord], gap: int = 2) -> bool:
    return all(_chebyshev(point, other) >= gap for other in points)


def _step_options(shape: tuple[int, int], current: Coord, occupied: set[Coord], previous_points: list[Coord]) -> list[tuple[Coord, Coord]]:
    h, w = shape
    options: list[tuple[Coord, Coord]] = []
    max_d = min(7, max(h, w) - 1)

    for dist in range(2, max_d + 1):
        for sign in (-1, 1):
            # Horizontal offset from pivot to current.
            pivot = (current[0], current[1] - sign * dist)
            landing = (current[0] - sign * dist, current[1] - sign * dist)
            if _gen_step_ok(shape, pivot, landing, occupied, previous_points):
                options.append((pivot, landing))

            # Vertical offset from pivot to current.
            pivot = (current[0] - sign * dist, current[1])
            landing = (current[0] - sign * dist, current[1] + sign * dist)
            if _gen_step_ok(shape, pivot, landing, occupied, previous_points):
                options.append((pivot, landing))

    random.shuffle(options)
    return options


def _gen_step_ok(
    shape: tuple[int, int],
    pivot: Coord,
    landing: Coord,
    occupied: set[Coord],
    previous_points: list[Coord],
) -> bool:
    if not (_in_bounds(shape, pivot) and _in_bounds(shape, landing)):
        return False
    if pivot in occupied or landing in occupied:
        return False
    if not _separated(pivot, occupied) or not _separated(landing, occupied):
        return False
    # A future pivot must not also be visible as an unused aligned pivot from an earlier relay point.
    if any(_aligned(pivot, p) for p in previous_points[:-1]):
        return False
    return True


def _try_build_chain(shape: tuple[int, int], length: int) -> tuple[Coord, list[Coord], list[Coord]] | None:
    h, w = shape
    margin = 2
    source = (random.randint(margin, h - 1 - margin), random.randint(margin, w - 1 - margin))
    trajectory = [source]
    pivots: list[Coord] = []
    landings: list[Coord] = []
    occupied: set[Coord] = {source}
    current = source

    for _ in range(length):
        options = _step_options(shape, current, occupied, trajectory)
        if not options:
            return None
        pivot, landing = random.choice(options[: min(12, len(options))])
        pivots.append(pivot)
        landings.append(landing)
        occupied.add(pivot)
        occupied.add(landing)
        trajectory.append(landing)
        current = landing

    try:
        traced_pivots, traced_landings = _trace(shape, source, tuple(pivots))
    except AssertionError:
        return None
    if traced_pivots != pivots or traced_landings != landings:
        return None
    if len(set(landings)) != len(landings):
        return None
    return source, pivots, landings


def _add_distractors(
    grid: np.ndarray,
    occupied: set[Coord],
    trajectory: list[Coord],
    count: int,
) -> list[Coord] | None:
    h, w = grid.shape
    candidates: list[Coord] = []
    for r in range(h):
        for c in range(w):
            p = (r, c)
            if p in occupied:
                continue
            if not _separated(p, occupied):
                continue
            if any(_aligned(p, q) for q in trajectory):
                continue
            candidates.append(p)

    if len(candidates) < count:
        return None
    random.shuffle(candidates)
    distractors = candidates[:count]
    for p in distractors:
        grid[p] = BLUE
        occupied.add(p)
    return distractors


def _expected_output(inp: np.ndarray, source: Coord, landings: list[Coord]) -> np.ndarray:
    out = inp.copy()
    out[source] = BLACK
    for p in landings[:-1]:
        out[p] = YELLOW
    out[landings[-1]] = ORANGE
    return out


def _validate_generated(inp: np.ndarray, source: Coord, pivots: list[Coord], landings: list[Coord]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(landings) < 2:
        return False
    if len(set(landings)) != len(landings):
        return False

    all_pivots = tuple(sorted(tuple(map(int, p)) for p in np.argwhere(inp == BLUE)))
    try:
        traced_pivots, traced_landings = _trace(inp.shape, source, all_pivots)
    except AssertionError:
        return False

    if traced_pivots != pivots:
        return False
    if traced_landings != landings:
        return False

    expected = _expected_output(inp, source, landings)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if int(np.count_nonzero(expected == YELLOW)) < 1:
        return False
    if int(np.count_nonzero(expected == ORANGE)) != 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(800):
        h = random.randint(10, 28)
        w = random.randint(10, 28)
        length = random.choices([2, 3, 4, 5, 6], weights=[2, 4, 4, 3, 1], k=1)[0]

        built = _try_build_chain((h, w), length)
        if built is None:
            continue
        source, pivots, landings = built

        inp = np.full((h, w), BLACK, dtype=int)
        for p in pivots:
            inp[p] = BLUE
        inp[source] = MAGENTA

        occupied = {source, *pivots, *landings}
        trajectory = [source, *landings]
        distractor_count = random.randint(0, min(4, max(0, (h * w) // 80)))
        if _add_distractors(inp, occupied, trajectory, distractor_count) is None:
            continue

        if _validate_generated(inp, source, pivots, landings):
            return inp

    raise RuntimeError("failed to generate a valid relay-rotation input")
