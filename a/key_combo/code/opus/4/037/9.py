import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
YELLOW = 4
OBSTACLE = 5
MARKER_COLORS = (2, 3, 7)

Coord = Tuple[int, int]


def _marker_cells(grid: np.ndarray) -> List[Tuple[int, int, int]]:
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(np.isin(grid, MARKER_COLORS))]
    markers.sort()
    assert 2 <= len(markers) <= 3, "expected 2 or 3 markers"
    assert len({color for _, _, color in markers}) == len(markers), "marker colors must be distinct"
    rows = [r for r, _, _ in markers]
    assert len(set(rows)) == len(rows), "marker rows must be distinct"
    return markers


def _path_end_right(grid: np.ndarray, row: int, col: int) -> int:
    width = grid.shape[1]
    c = int(col)
    while c + 1 < width and int(grid[row, c + 1]) != OBSTACLE:
        c += 1
    return c


def _path_end_down(grid: np.ndarray, row: int, col: int, next_marker_row: Optional[int]) -> int:
    height = grid.shape[0]
    limit = height - 1 if next_marker_row is None else min(height - 1, int(next_marker_row) - 1)
    r = int(row)
    while r + 1 <= limit and int(grid[r + 1, col]) != OBSTACLE:
        r += 1
    return r


def _analyze(grid: np.ndarray) -> Dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BG, YELLOW, OBSTACLE, *MARKER_COLORS}
    assert YELLOW not in set(map(int, np.unique(g))), "input must not contain yellow"

    markers = _marker_cells(g)
    height, width = g.shape

    segments: List[Dict[str, object]] = []
    column_actions: List[Dict[str, int]] = []
    row_actions: List[Dict[str, int]] = []
    base = g.copy()

    for idx, (r, c, color) in enumerate(markers):
        next_row = markers[idx + 1][0] if idx + 1 < len(markers) else None
        h_col = _path_end_right(g, r, c)
        base[r, c : h_col + 1] = color

        v_row = _path_end_down(g, r, h_col, next_row)
        base[r : v_row + 1, h_col] = color

        horiz_obstacle = h_col < width - 1 and int(g[r, h_col + 1]) == OBSTACLE
        down_limit = height - 1 if next_row is None else min(height - 1, int(next_row) - 1)
        vert_obstacle = v_row < down_limit and int(g[v_row + 1, h_col]) == OBSTACLE

        info = {
            "marker": (r, c),
            "color": color,
            "h_end": (r, h_col),
            "v_end": (v_row, h_col),
            "horizontal_obstacle": bool(horiz_obstacle),
            "vertical_obstacle": bool(vert_obstacle),
            "next_marker_row": next_row,
        }
        segments.append(info)
        if horiz_obstacle:
            column_actions.append({"color": color, "col": h_col, "marker_row": r})
        if vert_obstacle:
            row_actions.append({"color": color, "row": v_row, "marker_row": r})

    return {
        "grid": g,
        "markers": markers,
        "segments": segments,
        "base": base,
        "column_actions": column_actions,
        "row_actions": row_actions,
    }


def _apply_expansion(
    out: np.ndarray,
    owner: List[List[Optional[Tuple[str, int]]]],
    orientation: str,
    fixed_index: int,
    color: int,
) -> None:
    height, width = out.shape
    coords = (
        [(fixed_index, c) for c in range(width)]
        if orientation == "row"
        else [(r, fixed_index) for r in range(height)]
    )
    for r, c in coords:
        current = int(out[r, c])
        if current == BG:
            out[r, c] = int(color)
            owner[r][c] = (orientation, int(color))
            continue
        prev = owner[r][c]
        if prev is None or prev[0] == orientation or current == YELLOW:
            continue
        if prev[1] != int(color):
            out[r, c] = YELLOW
            owner[r][c] = ("cross", YELLOW)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Draw right-then-down marker paths, let obstacle-truncated endpoints emit full-row or full-column expansions, and mark orthogonal different-color expansion intersections yellow.

    Input structure:
    - A black background contains 2 or 3 single-pixel markers colored red, green, or orange, with at most one marker per row.
    - Several gray obstacle pixels are scattered away from the markers.
    - The markers are read from top to bottom, and each marker may have an obstacle on its rightward leg, its downward leg, or both.

    Rule:
    - For each marker in row order, extend its color rightward from the marker until the next cell would be a gray obstacle or would leave the grid, and record the last painted cell H.
    - From H, extend downward in the same color until the next cell would be a gray obstacle, until the row just above the next marker, or until the bottom edge, and record the last painted cell V.
    - If the horizontal leg stopped because of an obstacle, expand that marker's color through the whole column of H, painting only black cells.
    - If the vertical leg stopped because of an obstacle, expand that marker's color through the whole row of V, painting only black cells.
    - When a row expansion and a column expansion of different colors meet on a cell reached only by expansions, recolor that cell yellow; keep markers and gray obstacles unchanged.

    Key insight:
    - The turning path itself is only the first stage: obstacle-caused early stops become signals that launch row or column fills, and the final yellow cells appear only where differently colored orthogonal expansions intersect on background.
    """
    analysis = _analyze(input_grid)
    out = np.array(analysis["base"], copy=True)
    owner: List[List[Optional[Tuple[str, int]]]] = [[None for _ in range(out.shape[1])] for _ in range(out.shape[0])]

    segments = analysis["segments"]
    for idx, segment in enumerate(segments):
        if segment["horizontal_obstacle"]:
            _apply_expansion(out, owner, "col", int(segment["h_end"][1]), int(segment["color"]))
        if segment["vertical_obstacle"]:
            _apply_expansion(out, owner, "row", int(segment["v_end"][0]), int(segment["color"]))
    return out


def _empty_grid(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=int)


def _place(grid: np.ndarray, coord: Coord, color: int) -> None:
    r, c = coord
    assert int(grid[r, c]) == BG
    grid[r, c] = int(color)


def _add_obstacles(grid: np.ndarray, coords: Sequence[Coord]) -> None:
    for r, c in coords:
        assert int(grid[r, c]) == BG
        grid[r, c] = OBSTACLE


def _count_yellow_crossings(inp: np.ndarray, out: np.ndarray) -> int:
    analysis = _analyze(inp)
    row_actions = analysis["row_actions"]
    col_actions = analysis["column_actions"]
    base = np.asarray(analysis["base"], dtype=int)
    count = 0
    for row_action in row_actions:
        for col_action in col_actions:
            if int(row_action["color"]) == int(col_action["color"]):
                continue
            r = int(row_action["row"])
            c = int(col_action["col"])
            if int(base[r, c]) == BG and int(out[r, c]) == YELLOW:
                count += 1
    return count


def _validate_semantics(inp: np.ndarray) -> bool:
    analysis = _analyze(inp)
    out = main(inp)
    markers = analysis["markers"]
    segments = analysis["segments"]
    row_actions = analysis["row_actions"]
    col_actions = analysis["column_actions"]
    base = np.asarray(analysis["base"], dtype=int)

    assert len(markers) in (2, 3)
    assert 6 <= int(np.count_nonzero(inp == OBSTACLE)) <= 10
    assert len(row_actions) >= 1, "need at least one row expansion"
    assert len(col_actions) >= 1, "need at least one column expansion"
    assert _count_yellow_crossings(inp, out) >= 1, "need at least one visible yellow cross"

    for segment in segments:
        assert segment["horizontal_obstacle"] or segment["vertical_obstacle"], "each marker needs a real obstacle cue"
        mr, mc = segment["marker"]
        hr, hc = segment["h_end"]
        vr, vc = segment["v_end"]
        assert hr == mr and vc == hc
        if segment["horizontal_obstacle"]:
            assert hc + 1 < inp.shape[1] and int(inp[mr, hc + 1]) == OBSTACLE
        if segment["vertical_obstacle"]:
            assert vr + 1 < inp.shape[0] and int(inp[vr + 1, vc]) == OBSTACLE

    for row_action in row_actions:
        for col_action in col_actions:
            if int(row_action["color"]) == int(col_action["color"]):
                continue
            r = int(row_action["row"])
            c = int(col_action["col"])
            if int(base[r, c]) == BG:
                assert int(out[r, c]) in {int(col_action["color"]), int(row_action["color"]), YELLOW}

    assert np.array_equal(inp[inp == OBSTACLE], out[inp == OBSTACLE])
    for r, c, color in markers:
        assert int(out[r, c]) == color
    return True


def _noise_candidates(
    height: int,
    width: int,
    markers: Sequence[Coord],
    reserved: Sequence[Coord],
) -> List[Coord]:
    reserved_set = {tuple(cell) for cell in reserved}
    marker_set = {tuple(cell) for cell in markers}
    cells: List[Coord] = []
    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if cell in reserved_set or cell in marker_set:
                continue
            if any(abs(r - mr) + abs(c - mc) <= 1 for mr, mc in marker_set):
                continue
            cells.append(cell)
    random.shuffle(cells)
    return cells


def _gen_two_marker_case() -> np.ndarray:
    height = random.randint(13, 20)
    width = random.randint(13, 20)
    grid = _empty_grid(height, width)

    colors = random.sample(list(MARKER_COLORS), 2)
    r1 = random.randint(1, max(1, height // 3))
    r2 = random.randint(r1 + 4, height - 4)
    c1 = random.randint(1, max(1, width // 3))
    k1 = random.randint(c1 + 2, width - 4)
    c2 = random.randint(max(1, width // 3), width - 6)
    k2_lo = max(c2 + 2, k1 + 2)
    if k2_lo > width - 3:
        return _gen_two_marker_case()
    k2 = random.randint(k2_lo, width - 3)
    v1 = random.randint(r1 + 1, r2 - 2)
    v2 = random.randint(r2 + 1, height - 2)

    m1 = (r1, c1)
    m2 = (r2, c2)
    _place(grid, m1, colors[0])
    _place(grid, m2, colors[1])

    core_obstacles = [
        (r1, k1 + 1),
        (v1 + 1, k1),
        (v2 + 1, k2),
    ]
    if random.random() < 0.7:
        core_obstacles.append((r2, k2 + 1))

    reserved = set(core_obstacles)
    reserved.update([m1, m2])
    reserved.update((r1, c) for c in range(c1, k1 + 1))
    reserved.add((r1, k1))
    reserved.update((r, k1) for r in range(r1, v1 + 1))
    reserved.update((r2, c) for c in range(c2, k2 + 1))
    reserved.update((r, k2) for r in range(r2, v2 + 1))
    reserved.add((v2, k1))

    _add_obstacles(grid, core_obstacles)
    fillers = _noise_candidates(height, width, [m1, m2], list(reserved))
    needed = random.randint(6, 10) - len(core_obstacles)
    _add_obstacles(grid, fillers[:needed])

    return grid


def _gen_three_marker_case() -> np.ndarray:
    height = random.randint(15, 22)
    width = random.randint(15, 22)
    grid = _empty_grid(height, width)

    colors = random.sample(list(MARKER_COLORS), 3)
    r1 = random.randint(1, max(1, height // 4))
    r2 = random.randint(r1 + 3, max(r1 + 3, height // 2))
    r3 = random.randint(r2 + 4, height - 4)

    c1 = random.randint(1, max(1, width // 4))
    c2 = random.randint(c1 + 1, max(c1 + 1, width // 3))
    c3 = random.randint(1, max(1, width // 4))
    k1 = random.randint(c1 + 2, width - 6)
    k2_lo = max(c2 + 2, k1 + 2)
    if k2_lo > width - 4:
        return _gen_three_marker_case()
    k2 = random.randint(k2_lo, width - 4)
    k3_lo = max(c3 + 2, max(2, width // 2))
    k3_choices = [k for k in range(k3_lo, width - 2) if k not in {k1, k2}]
    if not k3_choices:
        return _gen_three_marker_case()
    k3 = random.choice(k3_choices)

    v1 = random.randint(r1 + 1, r2 - 2)
    v2 = random.randint(r2 + 1, r3 - 2)
    v3 = random.randint(r3 + 1, height - 2)

    markers = [(r1, c1), (r2, c2), (r3, c3)]
    for marker, color in zip(markers, colors):
        _place(grid, marker, color)

    core_obstacles = [
        (r1, k1 + 1),
        (v1 + 1, k1),
        (r2, k2 + 1),
        (v2 + 1, k2),
        (v3 + 1, k3),
    ]
    if random.random() < 0.6:
        core_obstacles.append((r3, k3 + 1))

    reserved = set(core_obstacles)
    reserved.update(markers)
    reserved.update((r1, c) for c in range(c1, k1 + 1))
    reserved.update((r, k1) for r in range(r1, v1 + 1))
    reserved.update((r2, c) for c in range(c2, k2 + 1))
    reserved.update((r, k2) for r in range(r2, v2 + 1))
    reserved.update((r3, c) for c in range(c3, k3 + 1))
    reserved.update((r, k3) for r in range(r3, v3 + 1))
    reserved.update([(v3, k1), (v3, k2)])

    _add_obstacles(grid, core_obstacles)
    fillers = _noise_candidates(height, width, markers, list(reserved))
    needed = random.randint(6, 10) - len(core_obstacles)
    _add_obstacles(grid, fillers[:needed])

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_two_marker_case() if random.random() < 0.45 else _gen_three_marker_case()
        try:
            if _validate_semantics(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
