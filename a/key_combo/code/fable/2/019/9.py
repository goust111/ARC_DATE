from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _sorted_points(coords: np.ndarray) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords.tolist())


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected colored endpoint pixels"

    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []
    anchor_coord: tuple[int, int] | None = None
    anchor_top: str | None = None

    for color in colors:
        pts = _sorted_points(np.argwhere(g == int(color)))
        if len(pts) == 2:
            rows = {r for r, _c in pts}
            cols = {c for _r, c in pts}
            if len(rows) == 1:
                row = pts[0][0]
                cols_sorted = sorted(c for _r, c in pts)
                horizontals.append({"color": int(color), "r": int(row), "c0": int(cols_sorted[0]), "c1": int(cols_sorted[1])})
            else:
                assert len(cols) == 1, "each endpoint pair must share a row or a column"
                col = pts[0][1]
                rows_sorted = sorted(r for r, _c in pts)
                verticals.append({"color": int(color), "c": int(col), "r0": int(rows_sorted[0]), "r1": int(rows_sorted[1])})
        elif len(pts) == 3:
            assert anchor_coord is None, "expected exactly one anchored color"
            rows = {r for r, _c in pts}
            cols = {c for _r, c in pts}
            if len(rows) == 1:
                row = pts[0][0]
                ordered = sorted(pts, key=lambda p: p[1])
                anchor_coord = (int(ordered[1][0]), int(ordered[1][1]))
                anchor_top = "h"
                horizontals.append({"color": int(color), "r": int(row), "c0": int(ordered[0][1]), "c1": int(ordered[2][1])})
            else:
                assert len(cols) == 1, "the anchored color must still define a single horizontal or vertical line"
                col = pts[0][1]
                ordered = sorted(pts, key=lambda p: p[0])
                anchor_coord = (int(ordered[1][0]), int(ordered[1][1]))
                anchor_top = "v"
                verticals.append({"color": int(color), "c": int(col), "r0": int(ordered[0][0]), "r1": int(ordered[2][0])})
        else:
            raise AssertionError("every non-background color must appear exactly twice, except the anchored line color which appears three times")

    assert anchor_coord is not None and anchor_top is not None, "expected one extra anchor pixel"
    assert horizontals and verticals, "expected both horizontal and vertical generated lines"
    return g, horizontals, verticals, anchor_coord, anchor_top


def _build_crossings(horizontals: list[dict[str, int]], verticals: list[dict[str, int]]):
    crossings: list[dict[str, int | tuple[int, int]]] = []
    h_incident: list[list[int]] = [[] for _ in horizontals]
    v_incident: list[list[int]] = [[] for _ in verticals]

    for hid, hline in enumerate(horizontals):
        for vid, vline in enumerate(verticals):
            if int(hline["c0"]) <= int(vline["c"]) <= int(hline["c1"]) and int(vline["r0"]) <= int(hline["r"]) <= int(vline["r1"]):
                cid = len(crossings)
                crossings.append({"hid": int(hid), "vid": int(vid), "coord": (int(hline["r"]), int(vline["c"]))})
                h_incident[hid].append(cid)
                v_incident[vid].append(cid)

    for hid, ids in enumerate(h_incident):
        ids.sort(key=lambda cid: int(crossings[cid]["coord"][1]))
        for order, cid in enumerate(ids):
            crossings[cid]["h_order"] = int(order)
            crossings[cid]["h_degree"] = int(len(ids))
    for vid, ids in enumerate(v_incident):
        ids.sort(key=lambda cid: int(crossings[cid]["coord"][0]))
        for order, cid in enumerate(ids):
            crossings[cid]["v_order"] = int(order)
            crossings[cid]["v_degree"] = int(len(ids))

    return crossings, h_incident, v_incident


def _solve_winners_by_line_phase(
    horizontals: list[dict[str, int]],
    verticals: list[dict[str, int]],
    crossings: list[dict[str, int | tuple[int, int]]],
    h_incident: list[list[int]],
    v_incident: list[list[int]],
    anchor_coord: tuple[int, int],
    anchor_top: str,
) -> list[int]:
    assert crossings, "expected at least one line crossing"

    anchor_ids = [cid for cid, cross in enumerate(crossings) if cross["coord"] == anchor_coord]
    assert len(anchor_ids) == 1, "anchor pixel must sit on exactly one generated crossing"
    anchor_cid = int(anchor_ids[0])
    anchor_cross = crossings[anchor_cid]
    anchor_state = 1 if anchor_top == "h" else 0

    h_phase: list[int | None] = [None] * len(horizontals)
    v_phase: list[int | None] = [None] * len(verticals)
    ah = int(anchor_cross["hid"])
    av = int(anchor_cross["vid"])
    h_phase[ah] = anchor_state ^ (int(anchor_cross["h_order"]) & 1)
    v_phase[av] = anchor_state ^ (int(anchor_cross["v_order"]) & 1)

    queue: deque[tuple[str, int]] = deque([("h", ah), ("v", av)])
    while queue:
        kind, idx = queue.popleft()
        if kind == "h":
            phase = h_phase[idx]
            assert phase is not None
            for cid in h_incident[idx]:
                cross = crossings[cid]
                state = int(phase) ^ (int(cross["h_order"]) & 1)
                vid = int(cross["vid"])
                expected = int(state) ^ (int(cross["v_order"]) & 1)
                if v_phase[vid] is None:
                    v_phase[vid] = int(expected)
                    queue.append(("v", vid))
                else:
                    assert int(v_phase[vid]) == int(expected), "alternation constraints are inconsistent"
        else:
            phase = v_phase[idx]
            assert phase is not None
            for cid in v_incident[idx]:
                cross = crossings[cid]
                state = int(phase) ^ (int(cross["v_order"]) & 1)
                hid = int(cross["hid"])
                expected = int(state) ^ (int(cross["h_order"]) & 1)
                if h_phase[hid] is None:
                    h_phase[hid] = int(expected)
                    queue.append(("h", hid))
                else:
                    assert int(h_phase[hid]) == int(expected), "alternation constraints are inconsistent"

    winners: list[int] = []
    for cross in crossings:
        hid = int(cross["hid"])
        vid = int(cross["vid"])
        assert h_phase[hid] is not None and v_phase[vid] is not None, "all crossings must be connected to the anchor"
        from_h = int(h_phase[hid]) ^ (int(cross["h_order"]) & 1)
        from_v = int(v_phase[vid]) ^ (int(cross["v_order"]) & 1)
        assert from_h == from_v, "horizontal and vertical phase propagation disagree"
        winners.append(int(from_h))
    return winners


def _solve_winners_by_crossing_bfs(
    crossings: list[dict[str, int | tuple[int, int]]],
    h_incident: list[list[int]],
    v_incident: list[list[int]],
    anchor_cid: int,
    anchor_state: int,
) -> list[int]:
    states: list[int | None] = [None] * len(crossings)
    states[anchor_cid] = int(anchor_state)
    queue: deque[int] = deque([int(anchor_cid)])

    while queue:
        cid = queue.popleft()
        state = states[cid]
        assert state is not None
        cross = crossings[cid]

        hid = int(cross["hid"])
        for other in h_incident[hid]:
            expected = int(state) ^ ((int(crossings[other]["h_order"]) - int(cross["h_order"])) & 1)
            if states[other] is None:
                states[other] = int(expected)
                queue.append(int(other))
            else:
                assert int(states[other]) == int(expected), "row-wise alternation produced a contradiction"

        vid = int(cross["vid"])
        for other in v_incident[vid]:
            expected = int(state) ^ ((int(crossings[other]["v_order"]) - int(cross["v_order"])) & 1)
            if states[other] is None:
                states[other] = int(expected)
                queue.append(int(other))
            else:
                assert int(states[other]) == int(expected), "column-wise alternation produced a contradiction"

    assert all(state is not None for state in states), "crossings must be anchor-connected"
    return [int(state) for state in states]


def _render_output(
    shape: tuple[int, int],
    horizontals: list[dict[str, int]],
    verticals: list[dict[str, int]],
    crossings: list[dict[str, int | tuple[int, int]]],
    winners: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for hline in horizontals:
        out[int(hline["r"]), int(hline["c0"]) : int(hline["c1"]) + 1] = int(hline["color"])
    for vline in verticals:
        out[int(vline["r0"]) : int(vline["r1"]) + 1, int(vline["c"])] = int(vline["color"])
    for cid, cross in enumerate(crossings):
        r, c = cross["coord"]
        if int(winners[cid]) == 1:
            out[int(r), int(c)] = int(horizontals[int(cross["hid"])]["color"])
        else:
            out[int(r), int(c)] = int(verticals[int(cross["vid"])]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden horizontal and vertical lines from same-color endpoints, then resolve each crossing with anchored weave alternation.

    Input structure:
    - The background is black, and every non-black color marks a generated line rather than a filled segment.
    - Most colors appear as exactly two same-color single-pixel endpoints lying in one row or one column.
    - Exactly one line color appears a third time at a crossing cell, and that extra pixel is the weave anchor telling which line is on top there.

    Rule:
    - Turn each same-color endpoint pair into its full horizontal or vertical segment, including the endpoints.
    - For every horizontal/vertical crossing, the winner along each individual line must alternate from one crossing to the next in that line's own crossing order.
    - Use the anchor pixel to fix the phase at one crossing, propagate the alternation through the connected crossing network, then color each crossing with the top line's color while keeping the rest of each segment in its own color.

    Key insight:
    - The visible input contains only endpoints plus one anchored crossing; the weave pattern is determined by crossing order along each generated line, not by raw checkerboard coordinate parity.
    """
    g, horizontals, verticals, anchor_coord, anchor_top = _parse_input(grid)
    crossings, h_incident, v_incident = _build_crossings(horizontals, verticals)
    winners = _solve_winners_by_line_phase(horizontals, verticals, crossings, h_incident, v_incident, anchor_coord, anchor_top)
    return _render_output(g.shape, horizontals, verticals, crossings, winners)


def _sample_axis_positions(count: int) -> tuple[np.ndarray, int]:
    start = int(RNG.integers(2, 5))
    positions = [start]
    for _ in range(count - 1):
        positions.append(int(positions[-1] + RNG.integers(3, 6)))
    size = int(positions[-1] + RNG.integers(3, 6))
    return np.array(positions, dtype=int), size


def _generate_interval_matrix(nh: int, nv: int) -> np.ndarray:
    if bool(RNG.integers(0, 4) == 0):
        return np.ones((nh, nv), dtype=bool)

    left = [0]
    right = [int(RNG.integers(0, nv))]
    for _ in range(1, nh):
        new_left = int(RNG.integers(left[-1], right[-1] + 1))
        new_right = int(RNG.integers(max(new_left, right[-1]), nv))
        left.append(new_left)
        right.append(new_right)
    right[-1] = nv - 1

    matrix = np.zeros((nh, nv), dtype=bool)
    for ridx in range(nh):
        matrix[ridx, left[ridx] : right[ridx] + 1] = True

    if bool(RNG.integers(0, 2)):
        matrix = matrix[::-1, :]
    if bool(RNG.integers(0, 2)):
        matrix = matrix[:, ::-1]
    return matrix


def _build_lines_from_matrix(matrix: np.ndarray):
    nh, nv = matrix.shape
    row_positions, height = _sample_axis_positions(nh)
    col_positions, width = _sample_axis_positions(nv)
    assert height <= 30 and width <= 30

    colors = RNG.choice(COLOR_POOL, size=nh + nv, replace=False).astype(int)
    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []

    for ridx in range(nh):
        cols = np.flatnonzero(matrix[ridx])
        prev_col = -1 if int(cols[0]) == 0 else int(col_positions[int(cols[0]) - 1])
        next_col = width if int(cols[-1]) == nv - 1 else int(col_positions[int(cols[-1]) + 1])
        left_choices = np.arange(prev_col + 1, int(col_positions[int(cols[0])]), dtype=int)
        right_choices = np.arange(int(col_positions[int(cols[-1])]) + 1, next_col, dtype=int)
        if left_choices.size == 0 or right_choices.size == 0:
            return None
        c0 = int(RNG.choice(left_choices))
        c1 = int(RNG.choice(right_choices))
        horizontals.append({"color": int(colors[ridx]), "r": int(row_positions[ridx]), "c0": int(c0), "c1": int(c1)})

    for cidx in range(nv):
        rows = np.flatnonzero(matrix[:, cidx])
        prev_row = -1 if int(rows[0]) == 0 else int(row_positions[int(rows[0]) - 1])
        next_row = height if int(rows[-1]) == nh - 1 else int(row_positions[int(rows[-1]) + 1])
        top_choices = np.arange(prev_row + 1, int(row_positions[int(rows[0])]), dtype=int)
        bottom_choices = np.arange(int(row_positions[int(rows[-1])]) + 1, next_row, dtype=int)
        if top_choices.size == 0 or bottom_choices.size == 0:
            return None
        r0 = int(RNG.choice(top_choices))
        r1 = int(RNG.choice(bottom_choices))
        verticals.append({"color": int(colors[nh + cidx]), "c": int(col_positions[cidx]), "r0": int(r0), "r1": int(r1)})

    crossings, h_incident, v_incident = _build_crossings(horizontals, verticals)
    expected_edges = {(int(ridx), int(cidx)) for ridx in range(nh) for cidx in range(nv) if bool(matrix[ridx, cidx])}
    actual_edges = {(int(cross["hid"]), int(cross["vid"])) for cross in crossings}
    if actual_edges != expected_edges:
        return None

    return (height, width), horizontals, verticals, crossings, h_incident, v_incident


def _connected_crossing_graph(crossings: list[dict[str, int | tuple[int, int]]], h_incident: list[list[int]], v_incident: list[list[int]]) -> bool:
    if not crossings:
        return False
    seen = {0}
    queue: deque[int] = deque([0])
    while queue:
        cid = queue.popleft()
        cross = crossings[cid]
        for other in h_incident[int(cross["hid"])] + v_incident[int(cross["vid"])]:
            if int(other) not in seen:
                seen.add(int(other))
                queue.append(int(other))
    return len(seen) == len(crossings)


def _make_input_grid(
    shape: tuple[int, int],
    horizontals: list[dict[str, int]],
    verticals: list[dict[str, int]],
    anchor_cid: int,
    anchor_top: str,
    crossings: list[dict[str, int | tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for hline in horizontals:
        grid[int(hline["r"]), int(hline["c0"])] = int(hline["color"])
        grid[int(hline["r"]), int(hline["c1"])] = int(hline["color"])
    for vline in verticals:
        grid[int(vline["r0"]), int(vline["c"])] = int(vline["color"])
        grid[int(vline["r1"]), int(vline["c"])] = int(vline["color"])

    anchor_cross = crossings[int(anchor_cid)]
    ar, ac = anchor_cross["coord"]
    if anchor_top == "h":
        grid[int(ar), int(ac)] = int(horizontals[int(anchor_cross["hid"])]["color"])
    else:
        grid[int(ar), int(ac)] = int(verticals[int(anchor_cross["vid"])]["color"])
    return grid


def _semantic_checks(
    input_grid: np.ndarray,
    expected_output: np.ndarray,
    matrix: np.ndarray,
    horizontals: list[dict[str, int]],
    verticals: list[dict[str, int]],
    crossings: list[dict[str, int | tuple[int, int]]],
    h_incident: list[list[int]],
    v_incident: list[list[int]],
    winners: list[int],
    anchor_cid: int,
    anchor_top: str,
) -> None:
    assert input_grid.ndim == 2 and expected_output.ndim == 2
    assert max(input_grid.shape) <= 30 and max(expected_output.shape) <= 30
    assert _connected_crossing_graph(crossings, h_incident, v_incident)
    assert int(matrix.sum()) == len(crossings)
    assert np.max(matrix.sum(axis=0)) >= 2 and np.max(matrix.sum(axis=1)) >= 2
    assert len(crossings) >= 4, "samples should expose alternation instead of only a trivial single crossing"

    counts = Counter(int(color) for color in input_grid.ravel() if int(color) != BG)
    count_values = sorted(counts.values())
    assert count_values.count(3) == 1 and all(value in {2, 3} for value in count_values)

    anchor_cross = crossings[int(anchor_cid)]
    anchor_color = int(horizontals[int(anchor_cross["hid"])]["color"] if anchor_top == "h" else verticals[int(anchor_cross["vid"])]["color"])
    ar, ac = anchor_cross["coord"]
    assert int(input_grid[int(ar), int(ac)]) == anchor_color
    assert set(winners) == {0, 1}, "samples should include both over and under states"

    solved = main(input_grid)
    assert np.array_equal(solved, expected_output), "generator and solver disagree"


def _generate_instance():
    nh = int(RNG.integers(2, 5))
    nv = int(RNG.integers(2, 5))
    matrix = _generate_interval_matrix(nh, nv)
    if np.max(matrix.sum(axis=0)) < 2 or np.max(matrix.sum(axis=1)) < 2:
        return None

    built = _build_lines_from_matrix(matrix)
    if built is None:
        return None
    shape, horizontals, verticals, crossings, h_incident, v_incident = built
    if len(crossings) < 4 or not _connected_crossing_graph(crossings, h_incident, v_incident):
        return None

    anchor_cid = int(RNG.integers(0, len(crossings)))
    anchor_top = "h" if bool(RNG.integers(0, 2)) else "v"
    anchor_state = 1 if anchor_top == "h" else 0
    winners = _solve_winners_by_crossing_bfs(crossings, h_incident, v_incident, anchor_cid, anchor_state)
    if set(winners) != {0, 1}:
        return None

    input_grid = _make_input_grid(shape, horizontals, verticals, anchor_cid, anchor_top, crossings)
    expected_output = _render_output(shape, horizontals, verticals, crossings, winners)
    _semantic_checks(input_grid, expected_output, matrix, horizontals, verticals, crossings, h_incident, v_incident, winners, anchor_cid, anchor_top)
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-weave ARC input")
