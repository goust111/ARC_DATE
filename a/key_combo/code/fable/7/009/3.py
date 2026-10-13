from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UTIL_ROOT = Path("/home/lixinyu/snap/new")
if str(UTIL_ROOT) not in sys.path:
    sys.path.append(str(UTIL_ROOT))

from solvers.arc_utils import Color, in_bounds, neighbors4


BG = Color.BLACK
PATH = Color.GREEN
GATE_END = Color.RED
GRAY = Color.GREY
ORANGE = Color.ORANGE
MARKER_COLORS = {GRAY, ORANGE}
PATHLIKE_COLORS = {PATH, GRAY, ORANGE}

MAX_ATTEMPTS = 400
GRID_H_RANGE = (11, 23)
GRID_W_RANGE = (15, 29)


@dataclass(frozen=True)
class Gate:
    index: int
    ends: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class Case:
    shape: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    gates: tuple[Gate, ...]
    markers: tuple[tuple[int, int], ...]
    scenario: str


def _path_neighbors(path_mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (nr, nc)
        for nr, nc in neighbors4(r, c)
        if in_bounds(path_mask.shape, nr, nc) and bool(path_mask[nr, nc])
    ]


def _trace_path(path_mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    order = [start]
    prev = None
    current = start
    while True:
        nxt = [cell for cell in _path_neighbors(path_mask, current) if cell != prev]
        if not nxt:
            return order
        assert len(nxt) == 1, "path must be a simple chain"
        prev, current = current, nxt[0]
        order.append(current)


def _gate_ends_for_index(
    path: list[tuple[int, int]], idx: int, shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if not (1 <= idx < len(path) - 1):
        return None
    r, c = path[idx]
    pr, pc = path[idx - 1]
    nr, nc = path[idx + 1]

    if pr == r == nr and pc == c - 1 and nc == c + 1:
        ends = ((r - 1, c), (r + 1, c))
    elif pc == c == nc and pr == r - 1 and nr == r + 1:
        ends = ((r, c - 1), (r, c + 1))
    else:
        return None

    return ends if all(in_bounds(shape, rr, cc) for rr, cc in ends) else None


def _ordered_state(
    grid: np.ndarray, path: list[tuple[int, int]]
) -> tuple[list[Gate], list[tuple[int, int]]]:
    gates: list[Gate] = []
    markers: list[tuple[int, int]] = []
    for idx, cell in enumerate(path):
        value = int(grid[cell])
        if value in MARKER_COLORS:
            markers.append((idx, value))
        if value != PATH:
            continue
        ends = _gate_ends_for_index(path, idx, grid.shape)
        if ends is None:
            continue
        if all(int(grid[rr, cc]) == GATE_END for rr, cc in ends):
            gates.append(Gate(idx, ends))
    return gates, markers


def _extract_case_state(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[Gate], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    path_mask = np.isin(g, tuple(PATHLIKE_COLORS))
    coords = np.argwhere(path_mask)
    assert coords.size > 0, "missing path"

    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _path_neighbors(path_mask, (r, c)):
            if (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    assert len(seen) == int(path_mask.sum()), "path cells must form one component"

    endpoints = []
    for r, c in seen:
        deg = len(_path_neighbors(path_mask, (r, c)))
        assert deg in (1, 2), "path must be branch-free"
        if deg == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    candidates = []
    for endpoint in endpoints:
        path = _trace_path(path_mask, endpoint)
        assert len(path) == int(path_mask.sum())
        gates, markers = _ordered_state(g, path)
        if gates and markers and max(idx for idx, _ in markers) < min(gate.index for gate in gates):
            candidates.append((path, gates, markers))

    if len(candidates) != 1:
        traced = []
        for endpoint in endpoints:
            path = _trace_path(path_mask, endpoint)
            gates, markers = _ordered_state(g, path)
            score = (
                max((idx for idx, _ in markers), default=len(path)),
                min((gate.index for gate in gates), default=-1),
            )
            traced.append((score, path, gates, markers))
        traced.sort(key=lambda item: item[0])
        _, path, gates, markers = traced[0]
        return path, gates, markers

    return candidates[0]


def _simulate_positions(
    path_len: int, gate_indices: list[int], marker_colors: list[int]
) -> tuple[list[int], list[tuple[int, int]]]:
    remaining_gates = list(gate_indices)
    queue = list(marker_colors)

    while queue and remaining_gates:
        if queue[-1] == GRAY:
            queue.pop()
            remaining_gates.pop(0)
            continue

        stop = remaining_gates[0] - 1
        start = stop - len(queue) + 1
        assert start >= 0, "frozen queue must fit before the gate"
        return remaining_gates, list(zip(range(start, stop + 1), queue))

    if not queue:
        return remaining_gates, []

    stop = path_len - 1
    start = stop - len(queue) + 1
    return remaining_gates, list(zip(range(start, stop + 1), queue))


def _render_case(
    shape: tuple[int, int],
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    gates: list[Gate] | tuple[Gate, ...],
    markers: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in path:
        grid[r, c] = PATH
    gate_by_index = {gate.index: gate for gate in gates}
    for gate in gate_by_index.values():
        for rr, cc in gate.ends:
            grid[rr, cc] = GATE_END
    for idx, color in markers:
        r, c = path[idx]
        grid[r, c] = int(color)
    return grid


def _expected_output(case: Case) -> np.ndarray:
    gate_indices = [gate.index for gate in case.gates]
    marker_colors = [color for _idx, color in case.markers]
    surviving_indices, final_markers = _simulate_positions(len(case.path), gate_indices, marker_colors)
    surviving_gates = [gate for gate in case.gates if gate.index in set(surviving_indices)]
    return _render_case(case.shape, case.path, surviving_gates, final_markers)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Slide the queued markers along the green path so gray markers annihilate gates and orange markers freeze the remaining queue.

    Input structure:
    - The background is black, and one one-cell-wide green path runs from one grid edge to another.
    - Several marker pixels, colored gray or orange, sit on an initial stretch of that path near its start.
    - One or more gates lie later on the path: each gate keeps its green center on the path and has two red endpoints on the perpendicular side cells.

    Rule:
    - Read the path from the endpoint whose early cells contain the markers and whose later cells contain the gates.
    - Process markers from front to back along that order: each front gray marker moves to the nearest surviving gate, then both the marker and that gate's red endpoints disappear.
    - When a front orange marker reaches the cell immediately before the nearest surviving gate, it stops there and every marker behind it compacts tightly behind it; if no gates remain, all remaining markers compact tightly at the path end.

    Key insight:
    - The gates are consumable only for gray markers, so the final marker positions depend on the whole front-to-back history of which earlier markers removed which gates.
    """
    g = np.asarray(input_grid, dtype=int)
    path, gates, markers = _extract_case_state(g)
    surviving_indices, final_markers = _simulate_positions(
        len(path),
        [gate.index for gate in gates],
        [color for _idx, color in markers],
    )
    surviving = [gate for gate in gates if gate.index in set(surviving_indices)]
    return _render_case(g.shape, path, surviving, final_markers)


def _random_composition(total: int, parts: int, min_each: int) -> list[int]:
    extra = total - parts * min_each
    cuts = sorted(random.sample(range(extra + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [extra + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return [min_each + value for value in values]


def _build_monotone_path() -> list[tuple[int, int]]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    horizontal = random.choice([True, False])
    if not horizontal:
        h, w = w, h

    turn_count = random.randint(2, 4)
    min_run = 3
    if w - 1 < (turn_count + 1) * min_run:
        raise ValueError("canvas too narrow")

    runs = _random_composition(w - 1, turn_count + 1, min_run)
    rows = [random.randint(1, h - 2)]
    for _ in range(turn_count):
        choices = [row for row in range(1, h - 1) if row != rows[-1]]
        if not choices:
            raise ValueError("no row change available")
        rows.append(random.choice(choices))

    path = [(rows[0], 0)]
    c = 0
    for seg, run in enumerate(runs):
        for _ in range(run):
            c += 1
            path.append((rows[seg], c))
        if seg == turn_count:
            continue
        step = 1 if rows[seg + 1] > rows[seg] else -1
        for r in range(rows[seg] + step, rows[seg + 1] + step, step):
            path.append((r, c))

    if not horizontal:
        path = [(c0, r0) for r0, c0 in path]
    return path


def _gate_candidates(path: list[tuple[int, int]], shape: tuple[int, int]) -> list[Gate]:
    path_set = set(path)
    candidates = []
    for idx in range(1, len(path) - 1):
        ends = _gate_ends_for_index(path, idx, shape)
        if ends is None:
            continue
        if any(end in path_set for end in ends):
            continue
        candidates.append(Gate(idx, ends))
    return candidates


def _pick_spaced_gates(candidates: list[Gate], marker_count: int, gate_count: int) -> list[Gate] | None:
    start_min = marker_count + 2
    usable = [gate for gate in candidates if gate.index >= start_min]
    if not usable:
        return None

    chosen = [random.choice(usable)]
    while len(chosen) < gate_count:
        later = [gate for gate in usable if gate.index >= chosen[-1].index + 4]
        if not later:
            return None
        chosen.append(random.choice(later))
        usable = later
    return chosen


def _choose_marker_colors(gate_count: int, marker_count: int) -> tuple[str, list[int]] | None:
    scenarios = ["freeze_first", "all_gray"]
    if gate_count >= 2 and marker_count >= 2:
        scenarios.append("freeze_after_gray")
    if marker_count > gate_count:
        scenarios.append("clear_all_gates")

    scenario = random.choice(scenarios)
    if scenario == "freeze_first":
        colors = [random.choice((GRAY, ORANGE)) for _ in range(marker_count - 1)] + [ORANGE]
        return scenario, colors

    if scenario == "freeze_after_gray":
        gray_front = random.randint(1, min(gate_count - 1, marker_count - 1))
        colors = [random.choice((GRAY, ORANGE)) for _ in range(marker_count - gray_front - 1)]
        colors.append(ORANGE)
        colors.extend([GRAY] * gray_front)
        return scenario, colors

    if scenario == "clear_all_gates":
        tail = [random.choice((GRAY, ORANGE)) for _ in range(marker_count - gate_count)]
        colors = tail + [GRAY] * gate_count
        return scenario, colors

    return scenario, [GRAY] * marker_count


def _choose_marker_positions(first_gate_index: int, marker_count: int) -> list[int] | None:
    limit = min(first_gate_index - 1, marker_count + random.randint(2, 5))
    if limit < marker_count:
        return None
    positions = sorted(random.sample(range(limit), marker_count))
    if positions[-1] >= first_gate_index - 1:
        return None
    return positions


def _make_case() -> Case:
    path = _build_monotone_path()
    shape = (max(r for r, _ in path) + 1, max(c for _, c in path) + 1)
    candidates = _gate_candidates(path, shape)
    if len(candidates) < 2:
        raise ValueError("too few gate candidates")

    marker_cap = min(6, max(2, len(path) // 6))
    marker_count = random.randint(2, marker_cap)
    gate_count = random.randint(1, min(4, len(candidates)))
    chosen_gates = _pick_spaced_gates(candidates, marker_count, gate_count)
    if chosen_gates is None:
        raise ValueError("unable to place gates")

    scenario, marker_colors = _choose_marker_colors(gate_count, marker_count)
    first_gate_index = chosen_gates[0].index
    marker_positions = _choose_marker_positions(first_gate_index, marker_count)
    if marker_positions is None:
        raise ValueError("unable to place markers")

    freeze_gray = 0
    if scenario == "freeze_after_gray":
        for color in reversed(marker_colors):
            if color == GRAY:
                freeze_gray += 1
            else:
                break
    if scenario in {"freeze_first", "freeze_after_gray"}:
        freeze_gate = chosen_gates[freeze_gray].index
        if freeze_gate < marker_count - freeze_gray:
            raise ValueError("frozen queue would not fit")

    case = Case(
        shape=shape,
        path=tuple(path),
        gates=tuple(chosen_gates),
        markers=tuple(zip(marker_positions, marker_colors)),
        scenario=scenario,
    )
    expected = _expected_output(case)
    inp = _render_case(case.shape, case.path, case.gates, case.markers)
    if np.array_equal(inp, expected):
        raise ValueError("degenerate case")
    return case


def _validate_case(case: Case) -> None:
    inp = _render_case(case.shape, case.path, case.gates, case.markers)
    out = _expected_output(case)

    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert np.array_equal(main(inp), out)

    path_mask = np.isin(inp, tuple(PATHLIKE_COLORS))
    endpoints = [
        (r, c)
        for r, c in case.path
        if len(_path_neighbors(path_mask, (r, c))) == 1
    ]
    assert len(endpoints) == 2

    valid_orders = 0
    for endpoint in endpoints:
        traced = _trace_path(path_mask, endpoint)
        gates, markers = _ordered_state(inp, traced)
        if gates and markers and max(idx for idx, _ in markers) < min(gate.index for gate in gates):
            valid_orders += 1
    assert valid_orders == 1, "input must determine a unique path direction"


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            case = _make_case()
            _validate_case(case)
        except Exception:
            continue
        return _render_case(case.shape, case.path, case.gates, case.markers)
    raise RuntimeError("failed to generate a valid annihilation-queue input")


def _generate_case_retry() -> Case:
    for _ in range(MAX_ATTEMPTS):
        try:
            case = _make_case()
            _validate_case(case)
            return case
        except Exception:
            continue
    raise RuntimeError("failed to build a validated case")


if __name__ == "__main__":
    seen = set()
    scenarios = set()
    for _ in range(80):
        case = _generate_case_retry()
        grid = _render_case(case.shape, case.path, case.gates, case.markers)
        seen.add((grid.shape, grid.tobytes()))
        scenarios.add(case.scenario)
    assert len(seen) >= 20
    assert len(scenarios) >= 3
