from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
MAROON = 9

MOTION_COLORS = (BLUE, RED, ORANGE, MAROON)
DIR_BY_COLOR = {
    ORANGE: (-1, 0),
    RED: (0, -1),
    MAROON: (1, 0),
    BLUE: (0, 1),
}
COLOR_BY_DIR = {delta: color for color, delta in DIR_BY_COLOR.items()}
DIRS4 = tuple(COLOR_BY_DIR)

GRID_MIN = 8
GRID_MAX = 18
CHAIN_LENGTHS = (1, 2, 3, 4, 5)
CHAIN_WEIGHTS = (1, 3, 4, 3, 2)
DECOY_RANGE = (1, 5)
MAX_GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_isolated(coord: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = coord
    return all(nbr not in others for nbr in _neighbors4(r, c))


def _step_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    out = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return out
        out.append((r, c))


def _trace_run(
    grid: np.ndarray,
    *,
    include_trace: bool = False,
) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_coords = np.argwhere(g == GREY)
    assert marker_coords.shape[0] == 1, "expected exactly one grey marker"
    marker = tuple(map(int, marker_coords[0]))

    occupied: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(g != BG):
        rc = (int(r), int(c))
        color = int(g[rc])
        if color == GREY:
            continue
        assert color in MOTION_COLORS, "only motion colors and one grey marker are allowed"
        occupied[rc] = color

    assert occupied, "expected at least one motion pixel"
    pixel_set = set(occupied)
    assert all(_is_isolated(rc, pixel_set - {rc}) for rc in pixel_set), "motion pixels must be isolated"

    starters = [rc for rc in occupied if marker in _neighbors4(*rc)]
    assert len(starters) == 1, "grey marker must identify exactly one starter"
    active = starters[0]

    trace: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[tuple[int, int, int], ...]]] = set()
    while True:
        state = (active, tuple(sorted((r, c, occupied[(r, c)]) for r, c in occupied)))
        assert state not in seen, "relay must not loop forever"
        seen.add(state)

        color = occupied[active]
        dr, dc = DIR_BY_COLOR[color]
        r, c = active
        path_cells = []
        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds((h, w), nr, nc):
                del occupied[active]
                trace.append(
                    {
                        "type": "exit",
                        "start": active,
                        "color": color,
                        "direction": (dr, dc),
                        "path": tuple(path_cells),
                    }
                )
                out = np.full_like(g, BG)
                for (rr, cc), cell_color in occupied.items():
                    out[rr, cc] = int(cell_color)
                return (out, trace) if include_trace else out

            next_rc = (nr, nc)
            if next_rc in occupied and next_rc != active:
                final_rc = (r, c)
                if final_rc != active:
                    moved_color = occupied.pop(active)
                    assert final_rc not in occupied, "collision stop cell must be empty"
                    occupied[final_rc] = moved_color
                trace.append(
                    {
                        "type": "hit",
                        "start": active,
                        "stop": final_rc,
                        "hit": next_rc,
                        "color": color,
                        "direction": (dr, dc),
                        "path": tuple(path_cells),
                    }
                )
                active = next_rc
                break

            path_cells.append(next_rc)
            r, c = next_rc


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Simulate the serial collision relay triggered by the grey-marked motion pixel.

    Input structure:
    - A black background contains isolated single motion pixels: orange means up,
      red means left, maroon means down, and blue means right.
    - Exactly one grey marker sits orthogonally adjacent to exactly one motion
      pixel, identifying the unique starter.

    Rule:
    - The starter slides straight along its own color direction.
    - If it reaches another motion pixel, it stops in the last empty cell before
      impact and the hit pixel immediately becomes the next slider, using its own
      color-coded direction.
    - If a slider finds no blocker, it leaves the grid and disappears; erase the
      grey marker and keep every never-hit pixel where it started.

    Key insight:
    - Colors store dormant movement instructions, but only the grey-marked pixel
      starts; the whole puzzle is one causal relay, not a simultaneous shift.
    """
    return _trace_run(input_grid)


def _line_target_options(
    current: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    blocked_now: set[tuple[int, int]],
    initial_pixels: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = current
    out = []
    distance = 0
    while True:
        distance += 1
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return out
        rc = (r, c)
        if rc in blocked_now:
            return out
        if rc in initial_pixels:
            return out
        if distance >= 2 and _is_isolated(rc, initial_pixels):
            out.append(rc)


def _clear_to_edge(
    current: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    blocked_now: set[tuple[int, int]],
) -> bool:
    return all(rc not in blocked_now for rc in _step_cells(current, direction, shape))


def _build_chain(
    shape: tuple[int, int],
    chain_length: int,
) -> tuple[list[dict], list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    start = (random.randrange(h), random.randrange(w))
    chain_positions = [start]
    initial_pixels = {start}
    blocked_stops: set[tuple[int, int]] = set()
    steps: list[dict] = []
    current = start

    for index in range(chain_length):
        directions = random.sample(DIRS4, k=len(DIRS4))
        picked = None
        if index == chain_length - 1:
            for direction in directions:
                if _clear_to_edge(current, direction, shape, blocked_stops):
                    picked = {"start": current, "direction": direction, "target": None}
                    break
        else:
            for direction in directions:
                options = _line_target_options(current, direction, shape, blocked_stops, initial_pixels)
                if not options:
                    continue
                target = random.choice(options)
                picked = {"start": current, "direction": direction, "target": target}
                break

        if picked is None:
            return None

        steps.append(picked)
        if picked["target"] is None:
            break

        tr, tc = picked["target"]
        dr, dc = picked["direction"]
        stop = (tr - dr, tc - dc)
        blocked_stops.add(stop)
        current = picked["target"]
        chain_positions.append(current)
        initial_pixels.add(current)

    if len(steps) != chain_length:
        return None

    path_forbidden: set[tuple[int, int]] = set()
    for step in steps:
        start = step["start"]
        direction = step["direction"]
        cells = _step_cells(start, direction, shape)
        if step["target"] is None:
            path_forbidden.update(cells)
        else:
            target = step["target"]
            for rc in cells:
                path_forbidden.add(rc)
                if rc == target:
                    break

    return steps, chain_positions, blocked_stops, path_forbidden


def _pick_marker(
    starter: tuple[int, int],
    starter_dir: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    disallowed = {starter_dir}
    candidates = []
    sr, sc = starter
    for dr, dc in DIRS4:
        if (dr, dc) in disallowed:
            continue
        rc = (sr + dr, sc + dc)
        if not _in_bounds(shape, *rc):
            continue
        if rc in occupied:
            continue
        if any(nbr in occupied and nbr != starter for nbr in _neighbors4(*rc)):
            continue
        candidates.append(rc)
    if not candidates:
        return None
    return random.choice(candidates)


def _candidate_decoys(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    isolated_from: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r in range(h):
        for c in range(w):
            rc = (r, c)
            if rc in forbidden:
                continue
            if not _is_isolated(rc, isolated_from):
                continue
            out.append(rc)
    random.shuffle(out)
    return out


def _alt_simulate(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker = tuple(map(int, np.argwhere(g == GREY)[0]))
    occupied = {tuple(map(int, rc)): int(g[tuple(rc)]) for rc in np.argwhere(np.isin(g, MOTION_COLORS))}
    active = next(rc for rc in occupied if marker in _neighbors4(*rc))

    h, w = g.shape
    while True:
        color = occupied[active]
        dr, dc = DIR_BY_COLOR[color]
        ray = []
        r, c = active
        while True:
            r += dr
            c += dc
            if not _in_bounds((h, w), r, c):
                del occupied[active]
                out = np.zeros_like(g)
                for (rr, cc), cell_color in occupied.items():
                    out[rr, cc] = int(cell_color)
                return out
            rc = (r, c)
            if rc in occupied and rc != active:
                stop = ray[-1] if ray else active
                if stop != active:
                    moved = occupied.pop(active)
                    occupied[stop] = moved
                active = rc
                break
            ray.append(rc)


def _is_valid_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    marker_coords = np.argwhere(g == GREY)
    if marker_coords.shape[0] != 1:
        return False
    pixels = np.argwhere(np.isin(g, MOTION_COLORS))
    if pixels.shape[0] < 1:
        return False

    pixel_set = {tuple(map(int, rc)) for rc in pixels}
    if any(not _is_isolated(rc, pixel_set - {rc}) for rc in pixel_set):
        return False

    marker = tuple(map(int, marker_coords[0]))
    starters = [rc for rc in pixel_set if marker in _neighbors4(*rc)]
    if len(starters) != 1:
        return False

    try:
        out1, trace = _trace_run(g, include_trace=True)
        out2 = _alt_simulate(g)
    except Exception:
        return False

    if not isinstance(out1, np.ndarray) or out1.ndim != 2 or max(out1.shape) > 30:
        return False
    if not np.array_equal(out1, out2):
        return False

    hit_steps = sum(step["type"] == "hit" for step in trace)
    if hit_steps == 0 and pixels.shape[0] == 1:
        return True
    return len(trace) >= 1 and any(step["type"] == "exit" for step in trace)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        shape = (
            random.randint(GRID_MIN, GRID_MAX),
            random.randint(GRID_MIN, GRID_MAX),
        )
        chain_length = random.choices(CHAIN_LENGTHS, weights=CHAIN_WEIGHTS, k=1)[0]
        built = _build_chain(shape, chain_length)
        if built is None:
            continue

        steps, chain_positions, blocked_stops, path_forbidden = built
        occupied = set(chain_positions)
        starter = chain_positions[0]
        starter_dir = steps[0]["direction"]
        marker = _pick_marker(starter, starter_dir, shape, occupied)
        if marker is None:
            continue

        forbidden = set(occupied) | blocked_stops | path_forbidden | {marker}
        decoys: list[tuple[tuple[int, int], int]] = []
        isolated_from = set(occupied) | {marker}
        for rc in _candidate_decoys(shape, forbidden, isolated_from):
            decoys.append((rc, random.choice(MOTION_COLORS)))
            isolated_from.add(rc)
            if len(decoys) >= random.randint(*DECOY_RANGE):
                break

        grid = np.zeros(shape, dtype=int)
        for step in steps:
            sr, sc = step["start"]
            grid[sr, sc] = COLOR_BY_DIR[step["direction"]]
        for (r, c), color in decoys:
            grid[r, c] = int(color)
        grid[marker] = GREY

        if _is_valid_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid collision-relay input")
