from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ROOT_SCRIPT_ORDER = {"yellow": 0, "green": 1, "blue": 2, "boundary": 3}
GRID_MIN = 18
GRID_MAX = 27
SOURCE_MARGIN = 6
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _add(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return int(cell[0] + direction[0] * steps), int(cell[1] + direction[1] * steps)


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _shuffled(items, rng: random.Random):
    items = list(items)
    rng.shuffle(items)
    return items


def _find_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(sorted(comp))
    comps.sort()
    return comps


def _decode_gate(coords: list[tuple[int, int]]) -> dict[str, object]:
    assert len(coords) == 3, "gate must have exactly three pixels"
    coord_set = set(coords)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in coords:
        degrees[(r, c)] = sum((r + dr, c + dc) in coord_set for dr, dc in DIR4)
    assert sorted(degrees.values()) == [1, 1, 2], "gate must be an L shape"
    pivot = next(cell for cell, degree in degrees.items() if degree == 2)
    arms = [cell for cell, degree in degrees.items() if degree == 1]
    arm_dirs = tuple((int(r - pivot[0]), int(c - pivot[1])) for r, c in arms)
    return {"pivot": pivot, "arm_dirs": arm_dirs}


def _parse_sources(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    roots: list[tuple[int, int, int, int]] = []
    red_components = _find_color_components(grid, RED)
    assert red_components, "expected at least one red source marker"
    for comp in red_components:
        assert len(comp) == 2, "each red source component must be a two-pixel domino"
        (r0, c0), (r1, c1) = comp
        if r0 == r1 and abs(c0 - c1) == 1:
            left = (r0, min(c0, c1))
            right = (r0, max(c0, c1))
            roots.append((left[0], left[1], 0, 1))
            roots.append((right[0], right[1], 0, -1))
        elif c0 == c1 and abs(r0 - r1) == 1:
            top = (min(r0, r1), c0)
            bottom = (max(r0, r1), c0)
            roots.append((top[0], top[1], 1, 0))
            roots.append((bottom[0], bottom[1], -1, 0))
        else:
            raise AssertionError("red source component must be horizontal or vertical")
    roots.sort()
    return roots


def _parse_gates(grid: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    gates: dict[tuple[int, int], dict[str, object]] = {}
    for color in (GREEN, YELLOW, BLUE):
        for comp in _find_color_components(grid, color):
            gate = _decode_gate(comp)
            pivot = gate["pivot"]
            assert pivot not in gates, "gate pivots must be unique"
            gate["color"] = int(color)
            gates[pivot] = gate
    return gates


def _turn_direction(arm_dirs: tuple[tuple[int, int], tuple[int, int]], direction: tuple[int, int]) -> tuple[int, int] | None:
    for idx, arm in enumerate(arm_dirs):
        if direction == (-arm[0], -arm[1]):
            return arm_dirs[1 - idx]
    return None


def _trace_endpoints(
    shape: tuple[int, int],
    roots: list[tuple[int, int, int, int]],
    gates: dict[tuple[int, int], dict[str, object]],
) -> set[tuple[int, int]]:
    h, w = shape
    endpoints: set[tuple[int, int]] = set()
    stack = list(roots)
    seen: set[tuple[int, int, int, int]] = set()
    while stack:
        r, c, dr, dc = stack.pop()
        state = (int(r), int(c), int(dr), int(dc))
        if state in seen:
            continue
        seen.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            endpoints.add((int(r), int(c)))
            continue

        gate = gates.get((nr, nc))
        if gate is None:
            stack.append((nr, nc, dr, dc))
            continue

        turn = _turn_direction(gate["arm_dirs"], (dr, dc))
        assert turn is not None, "signals must approach a gate pivot along one of its arms"
        color = int(gate["color"])
        if color == BLUE:
            endpoints.add((nr, nc))
        elif color == GREEN:
            stack.append((nr, nc, turn[0], turn[1]))
        elif color == YELLOW:
            stack.append((nr, nc, dr, dc))
            stack.append((nr, nc, turn[0], turn[1]))
        else:
            raise AssertionError("unexpected gate color")
    return endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace signals launched by red domino endpoints through colored L-shaped gates.

    Each red pixel is treated as a source whose initial direction points toward
    its unique red neighbor. A gate pivot only reacts when the signal enters
    from one of the gate's two arm sides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    allowed = {BLACK, RED, GREEN, YELLOW, BLUE}
    assert colors <= allowed, "unexpected input colors"

    roots = _parse_sources(g)
    gates = _parse_gates(g)
    endpoints = _trace_endpoints(g.shape, roots, gates)

    out = g.copy()
    for r, c in endpoints:
        out[r, c] = ORANGE
    return out


def _component_placeable(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in protected:
            return False
        if int(grid[r, c]) != BLACK:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) not in cells and int(grid[nr, nc]) != BLACK:
                return False
    return True


def _place_component(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _segment_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
) -> list[tuple[int, int]]:
    return [_add(start, direction, step) for step in range(1, steps + 1)]


def _steps_to_boundary(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    steps = 0
    cell = start
    while True:
        nxt = _add(cell, direction)
        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
            return steps
        steps += 1
        cell = nxt


def _cells_to_boundary(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    steps = _steps_to_boundary(start, direction, shape)
    return _segment_cells(start, direction, steps)


def _path_clear(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    return True


def _blue_outcomes(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[np.ndarray, set[tuple[int, int]]]]:
    max_steps = _steps_to_boundary(start, direction, grid.shape)
    if max_steps < 3:
        return []
    outcomes: list[tuple[np.ndarray, set[tuple[int, int]]]] = []
    for lead in _shuffled(range(3, max_steps + 1), rng):
        path_to_pivot = _segment_cells(start, direction, lead)
        pre_cells = path_to_pivot[:-1]
        pivot = path_to_pivot[-1]
        if not _path_clear(grid, pre_cells):
            continue
        for extra_dir in _shuffled(_perpendiculars(direction), rng):
            arm_in = _add(pivot, (-direction[0], -direction[1]))
            arm_extra = _add(pivot, extra_dir)
            gate_cells = {pivot, arm_in, arm_extra}
            if not _component_placeable(grid, gate_cells, protected):
                continue
            new_grid = grid.copy()
            _place_component(new_grid, gate_cells, BLUE)
            outcomes.append((new_grid, set(path_to_pivot)))
            if len(outcomes) >= 8:
                return outcomes
    return outcomes


def _boundary_outcomes(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[np.ndarray, set[tuple[int, int]]]]:
    path = _cells_to_boundary(start, direction, grid.shape)
    if not _path_clear(grid, path):
        return []
    if path:
        return [(grid.copy(), set(path))]
    return [(grid.copy(), {start})]


def _terminal_outcomes(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[np.ndarray, set[tuple[int, int]]]]:
    tagged: list[tuple[str, np.ndarray, set[tuple[int, int]]]] = []
    for new_grid, added in _boundary_outcomes(grid, start, direction):
        tagged.append(("boundary", new_grid, added))
    for new_grid, added in _blue_outcomes(grid, start, direction, protected, rng):
        tagged.append(("blue", new_grid, added))
    rng.shuffle(tagged)
    return [(grid_item, added) for _kind, grid_item, added in tagged]


def _green_outcomes(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[np.ndarray, set[tuple[int, int]]]]:
    max_steps = _steps_to_boundary(start, direction, grid.shape)
    if max_steps < 3:
        return []
    outcomes: list[tuple[np.ndarray, set[tuple[int, int]]]] = []
    for lead in _shuffled(range(3, max_steps + 1), rng):
        path_to_pivot = _segment_cells(start, direction, lead)
        pre_cells = path_to_pivot[:-1]
        pivot = path_to_pivot[-1]
        if not _path_clear(grid, pre_cells):
            continue
        for turn_dir in _shuffled(_perpendiculars(direction), rng):
            arm_in = _add(pivot, (-direction[0], -direction[1]))
            arm_out = _add(pivot, turn_dir)
            gate_cells = {pivot, arm_in, arm_out}
            if not _component_placeable(grid, gate_cells, protected):
                continue
            base_grid = grid.copy()
            _place_component(base_grid, gate_cells, GREEN)
            base_added = set(path_to_pivot) | {arm_out}
            next_protected = protected | base_added
            for tail_grid, tail_added in _terminal_outcomes(base_grid, arm_out, turn_dir, next_protected, rng):
                outcomes.append((tail_grid, base_added | tail_added))
                if len(outcomes) >= 12:
                    return outcomes
    return outcomes


def _yellow_outcomes(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[np.ndarray, set[tuple[int, int]]]]:
    max_steps = _steps_to_boundary(start, direction, grid.shape)
    if max_steps < 3:
        return []
    outcomes: list[tuple[np.ndarray, set[tuple[int, int]]]] = []
    for lead in _shuffled(range(3, max_steps + 1), rng):
        path_to_pivot = _segment_cells(start, direction, lead)
        pre_cells = path_to_pivot[:-1]
        pivot = path_to_pivot[-1]
        if not _path_clear(grid, pre_cells):
            continue
        for turn_dir in _shuffled(_perpendiculars(direction), rng):
            arm_in = _add(pivot, (-direction[0], -direction[1]))
            arm_out = _add(pivot, turn_dir)
            gate_cells = {pivot, arm_in, arm_out}
            if not _component_placeable(grid, gate_cells, protected):
                continue
            base_grid = grid.copy()
            _place_component(base_grid, gate_cells, YELLOW)
            base_added = set(path_to_pivot) | {arm_out}
            base_protected = protected | base_added

            branch_specs = [
                ("straight", pivot, direction),
                ("turn", arm_out, turn_dir),
            ]
            for order in _shuffled([(0, 1), (1, 0)], rng):
                first_idx, second_idx = order
                first_name, first_start, first_dir = branch_specs[first_idx]
                _ = first_name
                for first_grid, first_added in _terminal_outcomes(
                    base_grid,
                    first_start,
                    first_dir,
                    base_protected,
                    rng,
                ):
                    after_first_protected = base_protected | first_added
                    second_name, second_start, second_dir = branch_specs[second_idx]
                    _ = second_name
                    for second_grid, second_added in _terminal_outcomes(
                        first_grid,
                        second_start,
                        second_dir,
                        after_first_protected,
                        rng,
                    ):
                        outcomes.append((second_grid, base_added | first_added | second_added))
                        if len(outcomes) >= 16:
                            return outcomes
    return outcomes


def _route_root(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    script: str,
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    if script == "boundary":
        outcomes = _boundary_outcomes(grid, start, direction)
    elif script == "blue":
        outcomes = _blue_outcomes(grid, start, direction, protected, rng)
    elif script == "green":
        outcomes = _green_outcomes(grid, start, direction, protected, rng)
    elif script == "yellow":
        outcomes = _yellow_outcomes(grid, start, direction, protected, rng)
    else:
        raise AssertionError(f"unknown script {script}")
    return outcomes[0] if outcomes else None


def _choose_sources(grid: np.ndarray, rng: random.Random) -> list[dict[str, object]] | None:
    h, w = grid.shape
    if h < GRID_MIN or w < GRID_MIN:
        return None

    for _ in range(60):
        horizontal_row = rng.randint(SOURCE_MARGIN, h - SOURCE_MARGIN - 1)
        horizontal_col = rng.randint(SOURCE_MARGIN, w - SOURCE_MARGIN - 2)
        vertical_row = rng.randint(SOURCE_MARGIN, h - SOURCE_MARGIN - 2)
        vertical_col = rng.randint(SOURCE_MARGIN, w - SOURCE_MARGIN - 1)

        if abs(horizontal_row - vertical_row) < 4 or abs(horizontal_row - (vertical_row + 1)) < 4:
            continue
        if abs(horizontal_col - vertical_col) < 3 or abs((horizontal_col + 1) - vertical_col) < 3:
            continue

        horizontal_cells = {(horizontal_row, horizontal_col), (horizontal_row, horizontal_col + 1)}
        vertical_cells = {(vertical_row, vertical_col), (vertical_row + 1, vertical_col)}
        if not _component_placeable(grid, horizontal_cells, set()):
            continue
        _place_component(grid, horizontal_cells, RED)
        if not _component_placeable(grid, vertical_cells, set()):
            for r, c in horizontal_cells:
                grid[r, c] = BLACK
            continue
        _place_component(grid, vertical_cells, RED)
        return [
            {"start": (horizontal_row, horizontal_col + 1), "dir": (0, 1)},
            {"start": (horizontal_row, horizontal_col), "dir": (0, -1)},
            {"start": (vertical_row, vertical_col), "dir": (-1, 0)},
            {"start": (vertical_row + 1, vertical_col), "dir": (1, 0)},
        ]
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)

    roots = _choose_sources(grid, rng)
    if roots is None:
        return None

    scripts = ["boundary", "blue", "green", "yellow"]
    rng.shuffle(scripts)
    assignments = []
    for root, script in zip(roots, scripts):
        assignments.append(
            {
                "start": root["start"],
                "dir": root["dir"],
                "script": script,
            }
        )
    assignments.sort(key=lambda item: ROOT_SCRIPT_ORDER[item["script"]])

    protected: set[tuple[int, int]] = set()
    current = grid
    for spec in assignments:
        routed = _route_root(
            current,
            spec["start"],
            spec["dir"],
            spec["script"],
            protected,
            rng,
        )
        if routed is None:
            return None
        current, added = routed
        protected |= added

    out = main(current)
    orange_count = int(np.count_nonzero(out == ORANGE))
    if orange_count != 5:
        return None
    if int(np.count_nonzero(current == GREEN)) != 3:
        return None
    if int(np.count_nonzero(current == YELLOW)) != 3:
        return None
    if int(np.count_nonzero(current == BLUE)) < 3:
        return None
    if len(_find_color_components(current, RED)) != 2:
        return None
    return current


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
