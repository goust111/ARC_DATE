from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
SEED_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_HEIGHT_RANGE = (8, 19)
GRID_WIDTH_RANGE = (8, 19)
TOTAL_ABSORBERS_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400

STOP_STEP_BOUNDS = {
    "direct": (4, 10),
    "reflect": (6, 16),
    "cross": (8, 24),
    "corner_cross": (8, 24),
}
SCENARIOS = ("direct", "reflect", "cross", "corner_cross")
SCENARIO_WEIGHTS = (1, 1, 6, 4)


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return True iff (r, c) is inside a grid of shape (H, W)."""
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _bounce_step(
    shape: tuple[int, int], r: int, c: int, dr: int, dc: int
) -> tuple[int, int, int, int, bool, bool]:
    """
    Advance one diagonal billiard step.

    If the next row or column would leave the grid, flip that direction first,
    then move into the reflected in-bounds cell.
    """
    bounced_row = False
    bounced_col = False
    if not _in_bounds(shape, r + dr, c):
        dr = -dr
        bounced_row = True
    if not _in_bounds(shape, r, c + dc):
        dc = -dc
        bounced_col = True
    nr = int(r + dr)
    nc = int(c + dc)
    assert _in_bounds(shape, nr, nc), "reflected step must land inside the grid"
    return nr, nc, int(dr), int(dc), bounced_row, bounced_col


def _parse_input(grid: np.ndarray) -> tuple[Coord, Coord, Coord, set[Coord], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    non_bg_non_grey = np.argwhere((g != BLACK) & (g != GREY))
    assert non_bg_non_grey.shape == (1, 2), "expected exactly one colored seed pixel"
    seed = tuple(map(int, non_bg_non_grey[0]))
    seed_color = int(g[seed])
    assert seed_color in SEED_COLORS, "seed color must stay distinct from black, blue, and grey"

    marker_candidates: list[Coord] = []
    for dr, dc in DIAGONALS:
        mr = seed[0] + dr
        mc = seed[1] + dc
        if _in_bounds(g.shape, mr, mc) and int(g[mr, mc]) == GREY:
            marker_candidates.append((mr, mc))

    assert len(marker_candidates) == 1, "expected exactly one grey diagonal direction marker"
    marker = marker_candidates[0]
    direction = (int(seed[0] - marker[0]), int(seed[1] - marker[1]))
    assert direction in DIAGONALS, "marker must be diagonally adjacent to the seed"

    grey_cells = {tuple(map(int, coord)) for coord in np.argwhere(g == GREY)}
    absorbers = grey_cells - {marker}
    assert absorbers, "expected at least one grey absorber"
    return seed, marker, direction, absorbers, seed_color


def _simulate_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict], Coord, Coord, int]:
    """
    Simulate the reflected diagonal ray.

    The direction marker is a protected grey pass-through cell; all other grey
    cells are absorbers that stop the ray before they are painted.
    """
    g = np.asarray(grid, dtype=int)
    seed, marker, direction, absorbers, seed_color = _parse_input(g)
    out = g.copy()

    visits: dict[Coord, int] = {seed: 1}
    steps: list[dict] = []

    r, c = seed
    dr, dc = direction
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "ray must reach an absorber before cycling"
        seen_states.add(state)

        nr, nc, dr, dc, bounced_row, bounced_col = _bounce_step(g.shape, r, c, dr, dc)
        pos = (nr, nc)
        hit_marker = pos == marker
        hit_absorber = pos in absorbers

        steps.append(
            {
                "pos": pos,
                "row_bounce": bounced_row,
                "col_bounce": bounced_col,
                "hit_marker": hit_marker,
                "hit_absorber": hit_absorber,
            }
        )

        if hit_absorber:
            break

        if not hit_marker:
            prev = visits.get(pos, 0)
            visits[pos] = prev + 1
            if prev == 0:
                assert int(out[pos]) == BLACK, "new path cells must start as background"
                out[pos] = seed_color
            else:
                out[pos] = BLUE

        r, c = pos

    grey_mask = g == GREY
    assert np.array_equal(out[grey_mask], g[grey_mask]), "grey marker and absorbers must stay grey"
    return out, steps, seed, marker, seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect a diagonal seed ray off the grid boundary until it is absorbed, painting self-crossings blue.

    Input structure:
    - The background is black.
    - Exactly one non-grey colored seed pixel appears, and exactly one grey pixel is diagonally adjacent to it as the direction marker.
    - Additional isolated grey pixels are absorber cells elsewhere in the grid.

    Rule:
    - Start from the seed and move in the diagonal direction pointing away from the grey marker.
    - Whenever the next step would leave the grid, reflect on that boundary by flipping the row direction, the column direction, or both at a corner.
    - Paint each first-visited non-grey path cell with the seed color, turn any second-or-later visit to that cell blue, and stop immediately when the ray enters an absorber cell.
    - Keep the direction marker and all absorber cells grey.

    Key insight:
    - The foreground line is a billiard trajectory on cell centers: the only fixed cue is the grey diagonal marker, while the grey absorber pixels determine where the reflected walk terminates and repeated path cells expose self-intersections in blue.
    """
    out, _steps, _seed, _marker, _seed_color = _simulate_grid(grid)
    return out


def _trace_on_empty(
    shape: tuple[int, int], seed: Coord, marker: Coord, direction: Coord
) -> list[dict]:
    """
    Simulate the unabsorbed path on an otherwise empty board.

    The returned steps describe the next occupied cell after each reflected move.
    """
    r, c = seed
    dr, dc = direction
    seen_states: set[tuple[int, int, int, int]] = set()
    visits: dict[Coord, int] = {seed: 1}
    steps: list[dict] = []

    while True:
        state = (r, c, dr, dc)
        if state in seen_states:
            return steps
        seen_states.add(state)

        nr, nc, dr, dc, bounced_row, bounced_col = _bounce_step(shape, r, c, dr, dc)
        pos = (nr, nc)
        hit_marker = pos == marker
        repeat_paintable = (not hit_marker) and (visits.get(pos, 0) >= 1)

        steps.append(
            {
                "pos": pos,
                "row_bounce": bounced_row,
                "col_bounce": bounced_col,
                "corner_bounce": bounced_row and bounced_col,
                "marker": hit_marker,
                "repeat_paintable": repeat_paintable,
            }
        )

        if not hit_marker:
            visits[pos] = visits.get(pos, 0) + 1

        r, c = pos


def _valid_starts(shape: tuple[int, int]) -> list[tuple[Coord, Coord, Coord]]:
    h, w = map(int, shape)
    starts: list[tuple[Coord, Coord, Coord]] = []
    for r in range(h):
        for c in range(w):
            for dr, dc in DIAGONALS:
                marker = (r - dr, c - dc)
                if _in_bounds(shape, marker[0], marker[1]):
                    starts.append(((r, c), marker, (dr, dc)))
    return starts


def _corner_start(shape: tuple[int, int]) -> tuple[Coord, Coord, Coord]:
    h, w = map(int, shape)
    options = [
        ((0, 0), (1, 1), (-1, -1)),
        ((0, w - 1), (1, w - 2), (-1, 1)),
        ((h - 1, 0), (h - 2, 1), (1, -1)),
        ((h - 1, w - 1), (h - 2, w - 2), (1, 1)),
    ]
    seed, marker, direction = random.choice(options)
    return seed, marker, direction


def _scenario_ok(
    scenario: str,
    prefix_bounces: int,
    prefix_corners: int,
    blue_any: bool,
    blue_nonseed: bool,
) -> bool:
    if scenario == "direct":
        return prefix_bounces == 0 and not blue_any
    if scenario == "reflect":
        return prefix_bounces >= 1 and prefix_corners == 0 and not blue_any
    if scenario == "cross":
        return prefix_bounces >= 1 and blue_nonseed
    assert scenario == "corner_cross"
    return prefix_corners >= 1 and blue_nonseed


def _candidate_stop_indices(
    steps: list[dict], seed: Coord, marker: Coord, scenario: str
) -> list[int]:
    min_k, max_k = STOP_STEP_BOUNDS[scenario]
    candidates: list[int] = []

    prefix_bounces = 0
    prefix_corners = 0
    marker_hits = 0
    visits: dict[Coord, int] = {seed: 1}
    blue_any = False
    blue_nonseed = False

    for k, info in enumerate(steps):
        pos = info["pos"]
        if (
            min_k <= k <= max_k
            and not info["marker"]
            and not info["repeat_paintable"]
            and _chebyshev(pos, seed) > 1
            and _chebyshev(pos, marker) > 1
            and _scenario_ok(scenario, prefix_bounces, prefix_corners, blue_any, blue_nonseed)
        ):
            marker_ok = marker_hits == 0 if scenario != "corner_cross" else marker_hits >= 1
            if marker_ok and (scenario != "direct" or not (info["row_bounce"] or info["col_bounce"])):
                candidates.append(k)

        if info["row_bounce"] or info["col_bounce"]:
            prefix_bounces += 1
        if info["corner_bounce"]:
            prefix_corners += 1
        if info["marker"]:
            marker_hits += 1
            continue

        visits[pos] = visits.get(pos, 0) + 1
        if visits[pos] >= 2:
            blue_any = True
            if pos != seed:
                blue_nonseed = True

    return candidates


def _expected_from_prefix(
    grid: np.ndarray, seed: Coord, marker: Coord, stop_index: int, steps: list[dict], seed_color: int
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    visits: dict[Coord, int] = {seed: 1}
    for info in steps[:stop_index]:
        pos = info["pos"]
        if pos == marker:
            continue
        visits[pos] = visits.get(pos, 0) + 1
        out[pos] = seed_color if visits[pos] == 1 else BLUE
    return out


def _scatter_absorbers(
    shape: tuple[int, int], blocked: set[Coord], anchors: list[Coord], count: int
) -> list[Coord] | None:
    h, w = map(int, shape)
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)

    chosen: list[Coord] = []
    for cell in cells:
        if cell in blocked:
            continue
        if any(_chebyshev(cell, anchor) <= 1 for anchor in anchors):
            continue
        if any(_chebyshev(cell, other) <= 1 for other in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _build_grid(
    shape: tuple[int, int],
    seed: Coord,
    marker: Coord,
    seed_color: int,
    stop_pos: Coord,
    extra_absorbers: list[Coord],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[seed] = seed_color
    grid[marker] = GREY
    grid[stop_pos] = GREY
    for cell in extra_absorbers:
        grid[cell] = GREY
    return grid


def _pick_start(shape: tuple[int, int], scenario: str) -> tuple[Coord, Coord, Coord]:
    starts = _valid_starts(shape)
    edge_starts = [
        item
        for item in starts
        if 0 in item[0] or item[0][0] == shape[0] - 1 or item[0][1] == shape[1] - 1
    ]

    if scenario.startswith("corner"):
        search_pool = edge_starts if edge_starts else starts
        search_pool = random.sample(search_pool, min(len(search_pool), 80))
        for item in search_pool:
            seed, marker, direction = item
            steps = _trace_on_empty(shape, seed, marker, direction)
            for info in steps:
                if info["marker"]:
                    break
                if info["corner_bounce"]:
                    return item
        return random.choice(edge_starts if edge_starts else starts)

    if scenario in {"reflect", "cross"}:
        if edge_starts and random.random() < 0.7:
            return random.choice(edge_starts)
    return random.choice(starts)


def _variant_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    scenario: str,
    seed: Coord,
    marker: Coord,
    stop_pos: Coord,
    steps: list[dict],
    stop_index: int,
    seed_color: int,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    grey_coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == GREY)]
    if len(grey_coords) < 3:
        return False

    blue_count = int(np.count_nonzero(expected == BLUE))
    trail_count = int(np.count_nonzero((expected == BLUE) | (expected == seed_color)))
    if trail_count < 5 or trail_count > min(grid.size // 3 + 4, 26):
        return False
    if blue_count > 12:
        return False

    prefix_cells = [info["pos"] for info in steps[:stop_index]]
    if prefix_cells.count(stop_pos) != 0:
        return False

    marker_in_prefix = any(cell == marker for cell in prefix_cells)
    if scenario in {"direct", "reflect"} and blue_count != 0:
        return False
    if scenario in {"cross", "corner_cross"} and blue_count == 0:
        return False
    if scenario == "corner_cross" and not marker_in_prefix:
        return False
    if scenario != "corner_cross" and marker_in_prefix:
        return False

    if _chebyshev(seed, marker) != 1:
        return False
    if any(_chebyshev(seed, grey) <= 1 and grey != marker for grey in grey_coords):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]
    shape = (
        random.randrange(*GRID_HEIGHT_RANGE),
        random.randrange(*GRID_WIDTH_RANGE),
    )
    seed, marker, direction = _pick_start(shape, scenario)
    steps = _trace_on_empty(shape, seed, marker, direction)
    candidates = _candidate_stop_indices(steps, seed, marker, scenario)
    if not candidates:
        return None

    random.shuffle(candidates)
    for stop_index in candidates:
        stop_pos = steps[stop_index]["pos"]
        seed_color = random.choice(SEED_COLORS)

        total_absorbers = random.randint(*TOTAL_ABSORBERS_RANGE)
        blocked = {seed, marker, stop_pos}
        blocked |= {info["pos"] for info in steps}
        extras = _scatter_absorbers(shape, blocked, [seed, marker, stop_pos], total_absorbers - 1)
        if extras is None:
            continue

        grid = _build_grid(shape, seed, marker, seed_color, stop_pos, extras)
        expected = _expected_from_prefix(grid, seed, marker, stop_index, steps, seed_color)
        if _variant_ok(grid, expected, scenario, seed, marker, stop_pos, steps, stop_index, seed_color):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid billiard-seed input")
