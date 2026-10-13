from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import get_bg, in_bounds, neighbors4
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(values[np.argmax(counts)])

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])

    def neighbors4(r: int, c: int):
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1


Coord = tuple[int, int]

BLACK = 0
PATH = 3
START = 5
NON_SPECIAL_COLORS = [1, 2, 4, 6, 7, 8, 9]

GRID_MIN = 16
GRID_MAX = 26
LANE_COUNT_MIN = 4
LANE_COUNT_MAX = 6
LANE_GAPS = (2, 3)
MIN_LANE_SPAN = 8
MAX_RETRIES = 500


def _add(a: Coord, b: Coord) -> Coord:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _sub(a: Coord, b: Coord) -> Coord:
    return int(a[0] - b[0]), int(a[1] - b[1])


def _left_of(direction: Coord) -> Coord:
    dr, dc = map(int, direction)
    return -dc, dr


def _right_of(direction: Coord) -> Coord:
    dr, dc = map(int, direction)
    return dc, -dr


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _ordered_path(grid: np.ndarray) -> list[Coord]:
    g = np.asarray(grid, dtype=int)
    start_coords = np.argwhere(g == START)
    assert start_coords.shape[0] == 1, "expected one gray start marker"
    start = tuple(map(int, start_coords[0]))

    path_set = {tuple(map(int, coord)) for coord in np.argwhere((g == PATH) | (g == START))}
    assert start in path_set and len(path_set) >= 2

    adjacency: dict[Coord, list[Coord]] = {}
    for cell in path_set:
        adjacency[cell] = [
            (int(nr), int(nc))
            for nr, nc in neighbors4(*cell)
            if (int(nr), int(nc)) in path_set
        ]

    assert len(adjacency[start]) == 1, "start marker must sit on one path endpoint"
    degrees = [len(neis) for neis in adjacency.values()]
    assert all(deg in (1, 2) for deg in degrees), "path must be unbranched"
    assert sum(deg == 1 for deg in degrees) == 2, "path must have two endpoints"

    ordered = [start]
    prev: Coord | None = None
    cur = start
    while True:
        nxts = [nxt for nxt in adjacency[cur] if nxt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must be uniquely traversable"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt

    assert len(ordered) == len(path_set), "path traversal must cover the whole path"
    return ordered


def _direction_at(path: list[Coord], idx: int) -> Coord | None:
    if idx <= 0 or idx >= len(path) - 1:
        return None
    prev_cell = path[idx - 1]
    cur = path[idx]
    next_cell = path[idx + 1]
    d_in = _sub(cur, prev_cell)
    d_out = _sub(next_cell, cur)
    if d_in != d_out:
        return None
    return d_out


def _gate_at(grid: np.ndarray, path: list[Coord], idx: int, bg: int) -> dict[str, object] | None:
    direction = _direction_at(path, idx)
    if direction is None:
        return None

    center = path[idx]
    left = _add(center, _left_of(direction))
    right = _add(center, _right_of(direction))
    if not (in_bounds(grid.shape, *left) and in_bounds(grid.shape, *right)):
        return None

    left_color = int(grid[left])
    right_color = int(grid[right])
    if left_color in {bg, PATH, START} or right_color in {bg, PATH, START}:
        return None

    return {
        "index": int(idx),
        "center": center,
        "direction": direction,
        "left": left,
        "right": right,
        "left_color": left_color,
        "right_color": right_color,
    }


def _all_path_gates(grid: np.ndarray) -> tuple[list[Coord], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    path = _ordered_path(g)
    gates: list[dict[str, object]] = []
    for idx in range(1, len(path) - 1):
        gate = _gate_at(g, path, idx, bg)
        if gate is not None:
            gates.append(gate)
    return path, gates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the green path from the gray start and write each crossed gate's left-hand and right-hand pillar colors until the black stop gate.

    Input structure:
    - A uniform background contains one 4-connected, one-cell-wide, unbranched green path.
    - One endpoint of that path is replaced by a single gray start marker.
    - Some straight path cells have a pair of adjacent side pillars that form real gates, while other off-path pillar pairs are pseudo gates.

    Rule:
    - Traverse the path from the gray endpoint to the opposite endpoint in connectivity order.
    - Whenever a straight path cell has two non-background side pillars perpendicular to the travel direction, record one output row of length two.
    - Put the left-hand pillar color in the left output column and the right-hand pillar color in the right output column.
    - If both pillars are black, stop before recording that gate and ignore every later gate on the path.

    Key insight:
    - Gate order comes from walking the path, and pillar order comes from the current heading's left and right hands rather than any fixed absolute top/bottom or left/right orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    path = _ordered_path(g)

    rows: list[tuple[int, int]] = []
    for idx in range(1, len(path) - 1):
        gate = _gate_at(g, path, idx, bg)
        if gate is None:
            continue
        if int(gate["left_color"]) == BLACK and int(gate["right_color"]) == BLACK:
            break
        rows.append((int(gate["left_color"]), int(gate["right_color"])))

    if not rows:
        return np.zeros((0, 2), dtype=int)
    return np.array(rows, dtype=int)


def _build_snake_path(
    height: int,
    width: int,
    lane_count: int,
    lane_gap: int,
    start_left: bool,
) -> list[Coord] | None:
    if height < 3 + (lane_count - 1) * lane_gap or width < MIN_LANE_SPAN + 2:
        return None

    top_limit = height - 2 - (lane_count - 1) * lane_gap
    if top_limit < 1:
        return None
    top_row = random.randint(1, top_limit)
    rows = [top_row + lane_gap * i for i in range(lane_count)]

    max_left = width - 2 - MIN_LANE_SPAN
    if max_left < 1:
        return None
    left_col = random.randint(1, max_left)
    right_col = random.randint(left_col + MIN_LANE_SPAN - 1, width - 2)

    row = rows[0]
    col = left_col if start_left else right_col
    path: list[Coord] = [(row, col)]

    for lane_idx, lane_row in enumerate(rows):
        target_col = right_col if (start_left == (lane_idx % 2 == 0)) else left_col
        while col != target_col:
            col += 1 if target_col > col else -1
            path.append((row, col))
        if lane_idx == len(rows) - 1:
            continue
        next_row = rows[lane_idx + 1]
        while row != next_row:
            row += 1
            path.append((row, col))

    if len(path) != len(set(path)):
        return None
    return path


def _horizontal_gate_candidates(path: list[Coord], shape: tuple[int, int]) -> list[dict[str, object]]:
    path_set = set(path)
    candidates: list[dict[str, object]] = []
    for idx in range(1, len(path) - 1):
        direction = _direction_at(path, idx)
        if direction not in {(0, 1), (0, -1)}:
            continue
        center = path[idx]
        left = _add(center, _left_of(direction))
        right = _add(center, _right_of(direction))
        if not (in_bounds(shape, *left) and in_bounds(shape, *right)):
            continue
        if left in path_set or right in path_set:
            continue
        candidates.append(
            {
                "index": int(idx),
                "center": center,
                "direction": direction,
                "left": left,
                "right": right,
            }
        )
    return candidates


def _gate_conflict(a: dict[str, object], b: dict[str, object]) -> bool:
    a_cells = [a["left"], a["right"]]
    b_cells = [b["left"], b["right"]]
    for cell_a in a_cells:
        for cell_b in b_cells:
            if _manhattan(cell_a, cell_b) <= 1:
                return True
    return False


def _choose_real_gates(path: list[Coord], shape: tuple[int, int]) -> tuple[list[dict[str, object]], int] | None:
    candidates = _horizontal_gate_candidates(path, shape)
    east_candidates = [cand for cand in candidates if cand["direction"] == (0, 1)]
    west_candidates = [cand for cand in candidates if cand["direction"] == (0, -1)]
    west_rows: dict[int, list[dict[str, object]]] = {}
    for cand in west_candidates:
        west_rows.setdefault(int(cand["center"][0]), []).append(cand)

    valid_west_rows = [row for row, group in west_rows.items() if len(group) >= 2]
    if not east_candidates or not valid_west_rows:
        return None

    post_count = random.choices([0, 1, 2], weights=[1, 4, 3])[0]
    record_count = random.choice([3, 4])

    for _ in range(300):
        west_row = random.choice(valid_west_rows)
        west_pair = sorted(random.sample(west_rows[west_row], 2), key=lambda gate: gate["index"])
        east_gate = random.choice(east_candidates)
        recorded = sorted(west_pair + [east_gate], key=lambda gate: gate["index"])

        remaining = [cand for cand in candidates if cand not in recorded]
        if record_count == 4:
            before_extra = [cand for cand in remaining if cand["index"] < max(g["index"] for g in remaining + recorded)]
            if not before_extra:
                continue
            extra = random.choice(before_extra)
            recorded = sorted(recorded + [extra], key=lambda gate: gate["index"])

        if any(_gate_conflict(recorded[i], recorded[j]) for i in range(len(recorded)) for j in range(i + 1, len(recorded))):
            continue

        later_pool = [cand for cand in candidates if cand["index"] > recorded[-1]["index"] and cand not in recorded]
        if len(later_pool) < 1 + post_count:
            continue

        stop_choices = [
            cand for cand in later_pool
            if len([other for other in later_pool if other["index"] > cand["index"]]) >= post_count
        ]
        if not stop_choices:
            continue
        stop_gate = random.choice(stop_choices)
        after_pool = [cand for cand in later_pool if cand["index"] > stop_gate["index"]]
        if len(after_pool) < post_count:
            continue
        after_stop = sorted(random.sample(after_pool, post_count), key=lambda gate: gate["index"])

        selected = recorded + [stop_gate] + after_stop
        if any(_gate_conflict(selected[i], selected[j]) for i in range(len(selected)) for j in range(i + 1, len(selected))):
            continue

        west_recorded_rows = {}
        for gate in recorded:
            if gate["direction"] == (0, -1):
                west_recorded_rows.setdefault(int(gate["center"][0]), 0)
                west_recorded_rows[int(gate["center"][0])] += 1
        if not any(count >= 2 for count in west_recorded_rows.values()):
            continue

        return selected, record_count
    return None


def _assign_gate_colors(
    selected: list[dict[str, object]],
    record_count: int,
    bg: int,
) -> list[dict[str, object]] | None:
    palette = [color for color in NON_SPECIAL_COLORS if color != bg]
    want_visible_same = record_count == 4 and random.random() < 0.45

    for _ in range(200):
        colored: list[dict[str, object]] = []
        visible_rows: set[tuple[int, int]] = set()
        for idx, gate in enumerate(selected):
            gate_info = dict(gate)
            if idx == record_count:
                pair = (BLACK, BLACK)
            else:
                if idx < record_count:
                    make_same = want_visible_same and idx == record_count - 1
                    if make_same:
                        color = random.choice(palette)
                        pair = (color, color)
                    else:
                        pair = tuple(random.sample(palette, 2))
                else:
                    if random.random() < 0.35:
                        color = random.choice(palette)
                        pair = (color, color)
                    else:
                        pair = tuple(random.sample(palette, 2))

            gate_info["left_color"] = int(pair[0])
            gate_info["right_color"] = int(pair[1])
            if idx < record_count:
                row = (int(pair[0]), int(pair[1]))
                if row in visible_rows:
                    break
                visible_rows.add(row)
            colored.append(gate_info)
        else:
            return colored
    return None


def _distance_to_occupied(cells: list[Coord], occupied: set[Coord]) -> bool:
    for cell in cells:
        for other in occupied:
            if _manhattan(cell, other) <= 1:
                return False
    return True


def _place_pseudo_gates(
    shape: tuple[int, int],
    occupied: set[Coord],
    bg: int,
) -> list[dict[str, object]] | None:
    height, width = map(int, shape)
    target = random.randint(2, 4)
    palette = [color for color in NON_SPECIAL_COLORS if color != bg]
    placed: list[dict[str, object]] = []
    blocked = set(occupied)

    for _ in range(target):
        candidates: list[tuple[Coord, Coord, Coord]] = []
        for r in range(height):
            for c in range(width):
                center = (r, c)
                for dr, dc in ((1, 0), (0, 1)):
                    a = (r - dr, c - dc)
                    b = (r + dr, c + dc)
                    cells = [center, a, b]
                    if not all(in_bounds(shape, *cell) for cell in cells):
                        continue
                    if any(cell in blocked for cell in cells):
                        continue
                    if not _distance_to_occupied(cells, blocked):
                        continue
                    candidates.append((center, a, b))
        if not candidates:
            return None
        center, a, b = random.choice(candidates)
        if random.random() < 0.4:
            color = random.choice(palette)
            colors = (color, color)
        else:
            colors = tuple(random.sample(palette, 2))
        placed.append(
            {
                "center": center,
                "a": a,
                "b": b,
                "a_color": int(colors[0]),
                "b_color": int(colors[1]),
            }
        )
        blocked.update([center, a, b])

    return placed


def _render_input(
    shape: tuple[int, int],
    bg: int,
    path: list[Coord],
    gates: list[dict[str, object]],
    pseudo_gates: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for coord in path[1:]:
        grid[coord] = PATH
    grid[path[0]] = START

    for gate in gates:
        grid[gate["left"]] = int(gate["left_color"])
        grid[gate["right"]] = int(gate["right_color"])

    for gate in pseudo_gates:
        grid[gate["a"]] = int(gate["a_color"])
        grid[gate["b"]] = int(gate["b_color"])
    return grid


def _expected_output(gates: list[dict[str, object]], record_count: int) -> np.ndarray:
    rows = [[int(gate["left_color"]), int(gate["right_color"])] for gate in gates[:record_count]]
    return np.array(rows, dtype=int)


def _absolute_rows(gates: list[dict[str, object]], forward: bool) -> list[list[int]]:
    rows: list[list[int]] = []
    for gate in gates:
        pair = [
            (tuple(map(int, gate["left"])), int(gate["left_color"])),
            (tuple(map(int, gate["right"])), int(gate["right_color"])),
        ]
        pair.sort(key=lambda item: (item[0][0], item[0][1]))
        if forward:
            rows.append([pair[0][1], pair[1][1]])
        else:
            rows.append([pair[1][1], pair[0][1]])
    return rows


def _validate_scene(
    grid: np.ndarray,
    planned_gates: list[dict[str, object]],
    record_count: int,
    expected: np.ndarray,
) -> bool:
    path = _ordered_path(grid)
    if len(path) != len(set(path)):
        return False
    if sum(int(len([n for n in neighbors4(*cell) if tuple(map(int, n)) in set(path)]) == 1) for cell in path) != 2:
        return False

    _, actual_gates = _all_path_gates(grid)
    if len(actual_gates) != len(planned_gates):
        return False

    for actual, planned in zip(actual_gates, planned_gates):
        for key in ("index", "center", "direction", "left", "right", "left_color", "right_color"):
            if actual[key] != planned[key]:
                return False

    if sum(int(gate["left_color"] == BLACK and gate["right_color"] == BLACK) for gate in actual_gates) != 1:
        return False
    if not (actual_gates[record_count]["left_color"] == BLACK and actual_gates[record_count]["right_color"] == BLACK):
        return False

    visible = actual_gates[:record_count]
    if (0, 1) not in {tuple(gate["direction"]) for gate in visible}:
        return False
    west_rows: dict[int, int] = {}
    for gate in visible:
        if gate["direction"] == (0, -1):
            west_rows.setdefault(int(gate["center"][0]), 0)
            west_rows[int(gate["center"][0])] += 1
    if not any(count >= 2 for count in west_rows.values()):
        return False

    centers_path_order = [tuple(map(int, gate["center"])) for gate in visible]
    centers_scan_order = [tuple(map(int, gate["center"])) for gate in sorted(visible, key=lambda gate: gate["center"])]
    if centers_path_order == centers_scan_order:
        return False

    expected_list = expected.tolist()
    if len(expected_list) != len({tuple(row) for row in expected_list}):
        return False
    if expected_list == _absolute_rows(visible, True) or expected_list == _absolute_rows(visible, False):
        return False

    actual_out = main(grid)
    if not np.array_equal(actual_out, expected):
        return False

    if len(actual_gates) > record_count + 1:
        no_stop_rows = np.array(
            [[int(gate["left_color"]), int(gate["right_color"])] for gate in actual_gates if gate["left_color"] != BLACK or gate["right_color"] != BLACK],
            dtype=int,
        )
        if np.array_equal(no_stop_rows, expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    lane_gap = random.choice(LANE_GAPS)
    lane_options = [
        n for n in range(LANE_COUNT_MIN, LANE_COUNT_MAX + 1)
        if height >= 3 + (n - 1) * lane_gap
    ]
    if not lane_options:
        return None

    lane_count = random.choice(lane_options)
    start_left = random.choice([False, True])
    path = _build_snake_path(height, width, lane_count, lane_gap, start_left)
    if path is None:
        return None

    bg = random.choice([color for color in NON_SPECIAL_COLORS if color not in {PATH, START}])
    choice = _choose_real_gates(path, (height, width))
    if choice is None:
        return None
    selected_gates, record_count = choice

    colored_gates = _assign_gate_colors(selected_gates, record_count, bg)
    if colored_gates is None:
        return None

    occupied = set(path)
    for gate in colored_gates:
        occupied.add(tuple(map(int, gate["left"])))
        occupied.add(tuple(map(int, gate["right"])))

    pseudo_gates = _place_pseudo_gates((height, width), occupied, bg)
    if pseudo_gates is None:
        return None

    grid = _render_input((height, width), bg, path, colored_gates, pseudo_gates)
    expected = _expected_output(colored_gates, record_count)
    if expected.shape[0] < 3 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if int(get_bg(grid)) != bg:
        return None
    if not _validate_scene(grid, colored_gates, record_count, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-ledger input")
