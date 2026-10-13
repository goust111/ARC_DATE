from __future__ import annotations

import random
from collections import Counter

import numpy as np


Room = tuple[int, int]
Edge = tuple[Room, Room]

BG = 0
WALL = 1
FLOOR_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 300


def _edge_key(a: Room, b: Room) -> Edge:
    return (a, b) if a <= b else (b, a)


def _neighbors(room: Room, nrows: int, ncols: int) -> list[tuple[str, Room]]:
    r, c = room
    out: list[tuple[str, Room]] = []
    if r > 0:
        out.append(("up", (r - 1, c)))
    if r + 1 < nrows:
        out.append(("down", (r + 1, c)))
    if c > 0:
        out.append(("left", (r, c - 1)))
    if c + 1 < ncols:
        out.append(("right", (r, c + 1)))
    return out


def _all_edges(nrows: int, ncols: int) -> list[Edge]:
    edges: list[Edge] = []
    for r in range(nrows):
        for c in range(ncols):
            room = (r, c)
            if r + 1 < nrows:
                edges.append(_edge_key(room, (r + 1, c)))
            if c + 1 < ncols:
                edges.append(_edge_key(room, (r, c + 1)))
    return edges


def _wall_rows_cols(grid: np.ndarray, wall_color: int) -> tuple[list[int], list[int]]:
    h, w = map(int, grid.shape)
    row_hits = [r for r in range(h) if int(np.count_nonzero(grid[r] == wall_color)) >= (w + 1) // 2]
    col_hits = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == wall_color)) >= (h + 1) // 2]
    return row_hits, col_hits


def _find_wall_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    colors = [int(v) for v in np.unique(grid)]
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in colors:
        rows, cols = _wall_rows_cols(grid, color)
        if len(rows) >= 2 and len(cols) >= 2:
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1, "expected one wall color that forms full grid lines"
    return candidates[0]


def _regular_step(lines: list[int]) -> int:
    assert len(lines) >= 2, "expected at least two wall lines"
    diffs = [int(b - a) for a, b in zip(lines, lines[1:])]
    assert diffs and len(set(diffs)) == 1, "wall lines must be evenly spaced"
    return diffs[0]


def _find_background(grid: np.ndarray, wall_color: int, wall_rows: list[int], wall_cols: list[int]) -> int:
    gap_colors: set[int] = set()
    for r in wall_rows:
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != wall_color:
                gap_colors.add(int(grid[r, c]))
    for c in wall_cols:
        for r in range(grid.shape[0]):
            if int(grid[r, c]) != wall_color:
                gap_colors.add(int(grid[r, c]))
    if gap_colors:
        assert len(gap_colors) == 1, "all wall gaps must share the background color"
        return next(iter(gap_colors))

    counts = Counter(int(v) for v in grid.ravel() if int(v) != wall_color)
    assert counts, "grid must contain background or room colors"
    return counts.most_common(1)[0][0]


def _room_bounds(wall_rows: list[int], wall_cols: list[int], room: Room) -> tuple[int, int, int, int]:
    r, c = room
    return wall_rows[r], wall_rows[r + 1], wall_cols[c], wall_cols[c + 1]


def _room_fill_color(grid: np.ndarray, bounds: tuple[int, int, int, int], bg: int, wall_color: int) -> int:
    top, bottom, left, right = bounds
    interior = grid[top + 1 : bottom, left + 1 : right]
    colors = {int(v) for v in np.unique(interior)}
    assert wall_color not in colors, "room interior must not contain wall pixels"
    if colors == {bg}:
        return bg
    assert len(colors) == 1 and bg not in colors, "colored rooms must have a uniform floor color"
    return next(iter(colors))


def _gap_directions(
    grid: np.ndarray,
    room: Room,
    wall_rows: list[int],
    wall_cols: list[int],
    wall_color: int,
) -> dict[str, bool]:
    top, bottom, left, right = _room_bounds(wall_rows, wall_cols, room)
    out = {"up": False, "down": False, "left": False, "right": False}

    if room[0] > 0:
        gap_cols = [c for c in range(left + 1, right) if int(grid[top, c]) != wall_color]
        assert len(gap_cols) <= 1, "a wall segment may contain at most one gap"
        out["up"] = bool(gap_cols)
    if room[0] + 1 < len(wall_rows) - 1:
        gap_cols = [c for c in range(left + 1, right) if int(grid[bottom, c]) != wall_color]
        assert len(gap_cols) <= 1, "a wall segment may contain at most one gap"
        out["down"] = bool(gap_cols)
    if room[1] > 0:
        gap_rows = [r for r in range(top + 1, bottom) if int(grid[r, left]) != wall_color]
        assert len(gap_rows) <= 1, "a wall segment may contain at most one gap"
        out["left"] = bool(gap_rows)
    if room[1] + 1 < len(wall_cols) - 1:
        gap_rows = [r for r in range(top + 1, bottom) if int(grid[r, right]) != wall_color]
        assert len(gap_rows) <= 1, "a wall segment may contain at most one gap"
        out["right"] = bool(gap_rows)

    return out


def _neighbor(room: Room, direction: str) -> Room:
    r, c = room
    if direction == "up":
        return (r - 1, c)
    if direction == "down":
        return (r + 1, c)
    if direction == "left":
        return (r, c - 1)
    if direction == "right":
        return (r, c + 1)
    raise AssertionError("unknown direction")


def _arm_cells(bounds: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    top, bottom, left, right = bounds
    center_r = (top + bottom) // 2
    center_c = (left + right) // 2
    if direction == "up":
        return [(center_r - 1, center_c), (center_r - 2, center_c)]
    if direction == "down":
        return [(center_r + 1, center_c), (center_r + 2, center_c)]
    if direction == "left":
        return [(center_r, center_c - 1), (center_r, center_c - 2)]
    if direction == "right":
        return [(center_r, center_c + 1), (center_r, center_c + 2)]
    raise AssertionError("unknown direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw colored mini direction arms inside blank rooms according to wall gaps and adjacent room colors.

    Input structure:
    - The background is black.
    - One non-background color forms a one-pixel-thick orthogonal wall grid that
      partitions the canvas into equal rectangular rooms of at least 5x5 interior size.
    - Some room floors are fully filled with a uniform non-background, non-wall color,
      while other room floors remain black.
    - Some shared room walls contain a single one-pixel gap that opens a passage to the
      adjacent room.

    Rule:
    - Keep every colored room unchanged.
    - For each blank room, inspect its four wall segments. A gap on a side activates that
      direction only if the adjacent room across the gap is a colored room.
    - For every activated direction, draw a two-pixel arm in the room's central area,
      pointing toward that side. The arm uses the adjacent room's floor color.
    - Leave blank-to-blank gaps silent, and keep the wall grid and the gap cells unchanged.

    Key insight:
    - The task is solved by reconstructing the regular room lattice, detecting which wall
      segments have openings, and using the neighboring room color as the label for a
      small direction marker rather than moving or copying the whole room color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_color, wall_rows, wall_cols = _find_wall_color(g)
    step_h = _regular_step(wall_rows)
    step_w = _regular_step(wall_cols)
    assert step_h >= 6 and step_w >= 6, "room interiors must be at least 5x5"

    bg = _find_background(g, wall_color, wall_rows, wall_cols)
    nrows = len(wall_rows) - 1
    ncols = len(wall_cols) - 1

    room_colors: dict[Room, int] = {}
    for r in range(nrows):
        for c in range(ncols):
            room = (r, c)
            bounds = _room_bounds(wall_rows, wall_cols, room)
            room_colors[room] = _room_fill_color(g, bounds, bg, wall_color)

    out = g.copy()
    for r in range(nrows):
        for c in range(ncols):
            room = (r, c)
            if room_colors[room] != bg:
                continue
            bounds = _room_bounds(wall_rows, wall_cols, room)
            for direction, has_gap in _gap_directions(g, room, wall_rows, wall_cols, wall_color).items():
                if not has_gap:
                    continue
                neighbor_room = _neighbor(room, direction)
                fill_color = room_colors[neighbor_room]
                if fill_color == bg:
                    continue
                for rr, cc in _arm_cells(bounds, direction):
                    out[rr, cc] = fill_color

    return out


def _sample_layout() -> tuple[int, int, int, int]:
    choices: list[tuple[int, int, int, int]] = []
    for nrows in range(2, 5):
        for ncols in range(2, 5):
            for room_h in range(5, 14, 2):
                for room_w in range(5, 14, 2):
                    height = nrows * (room_h + 1) + 1
                    width = ncols * (room_w + 1) + 1
                    if height <= 30 and width <= 30:
                        choices.append((nrows, ncols, room_h, room_w))
    assert choices
    return random.choice(choices)


def _wall_lines(nrooms: int, room_size: int) -> list[int]:
    step = room_size + 1
    return [i * step for i in range(nrooms + 1)]


def _choose_color(exclude: set[int]) -> int:
    options = [color for color in FLOOR_COLORS if color not in exclude]
    assert options
    return random.choice(options)


def _build_plan() -> dict | None:
    nrows, ncols, room_h, room_w = _sample_layout()
    rooms = [(r, c) for r in range(nrows) for c in range(ncols)]
    edges = _all_edges(nrows, ncols)

    need_multi = random.choice([False, True])
    need_silent = random.choice([False, True])

    for _ in range(80):
        min_degree = 2 if need_multi else 1
        hub_choices = [room for room in rooms if len(_neighbors(room, nrows, ncols)) >= min_degree]
        if not hub_choices:
            return None
        hub = random.choice(hub_choices)
        hub_neighbors = [nbr for _dir, nbr in _neighbors(hub, nrows, ncols)]
        if need_multi:
            k_action = random.randint(2, min(4, len(hub_neighbors)))
        else:
            k_action = 1
        action_neighbors = random.sample(hub_neighbors, k_action)
        action_edges = {_edge_key(hub, nbr) for nbr in action_neighbors}
        action_rooms = set(action_neighbors)

        silent_edge: Edge | None = None
        if need_silent:
            silent_candidates = [
                edge
                for edge in edges
                if edge not in action_edges and edge[0] not in action_rooms and edge[1] not in action_rooms
            ]
            if not silent_candidates:
                continue
            silent_edge = random.choice(silent_candidates)

        room_colors = {room: BG for room in rooms}
        used_action_colors: set[int] = set()
        for nbr in action_neighbors:
            color = _choose_color(used_action_colors)
            room_colors[nbr] = color
            used_action_colors.add(color)

        locked_blank = {hub}
        if silent_edge is not None:
            locked_blank.update(silent_edge)

        extra_candidates = [room for room in rooms if room not in locked_blank and room not in action_rooms]
        max_extra_colored = max(0, len(extra_candidates) // 2)
        n_extra_colored = random.randint(0, max_extra_colored)
        for room in random.sample(extra_candidates, n_extra_colored):
            room_colors[room] = _choose_color(set())

        gap_edges = set(action_edges)
        if silent_edge is not None:
            gap_edges.add(silent_edge)

        extra_gap_prob = random.uniform(0.08, 0.28)
        for edge in edges:
            if edge in gap_edges:
                continue
            if random.random() < extra_gap_prob:
                gap_edges.add(edge)

        gap_positions: dict[Edge, int] = {}
        wall_rows = _wall_lines(nrows, room_h)
        wall_cols = _wall_lines(ncols, room_w)
        for a, b in gap_edges:
            if a[0] == b[0]:
                ri = a[0]
                gap_positions[(a, b)] = random.randint(wall_rows[ri] + 1, wall_rows[ri + 1] - 1)
            else:
                cj = a[1]
                gap_positions[(a, b)] = random.randint(wall_cols[cj] + 1, wall_cols[cj + 1] - 1)

        plan = {
            "nrows": nrows,
            "ncols": ncols,
            "room_h": room_h,
            "room_w": room_w,
            "room_colors": room_colors,
            "gap_edges": gap_edges,
            "gap_positions": gap_positions,
        }
        if _plan_has_visible_change(plan):
            return plan
    return None


def _render_input(plan: dict) -> np.ndarray:
    nrows = int(plan["nrows"])
    ncols = int(plan["ncols"])
    room_h = int(plan["room_h"])
    room_w = int(plan["room_w"])
    wall_rows = _wall_lines(nrows, room_h)
    wall_cols = _wall_lines(ncols, room_w)
    height = wall_rows[-1] + 1
    width = wall_cols[-1] + 1

    grid = np.full((height, width), BG, dtype=int)
    for r in wall_rows:
        grid[r, :] = WALL
    for c in wall_cols:
        grid[:, c] = WALL

    room_colors: dict[Room, int] = plan["room_colors"]
    for room, color in room_colors.items():
        if color == BG:
            continue
        top, bottom, left, right = _room_bounds(wall_rows, wall_cols, room)
        grid[top + 1 : bottom, left + 1 : right] = int(color)

    for edge, pos in plan["gap_positions"].items():
        a, b = edge
        if a[0] == b[0]:
            wall_col = wall_cols[max(a[1], b[1])]
            grid[int(pos), wall_col] = BG
        else:
            wall_row = wall_rows[max(a[0], b[0])]
            grid[wall_row, int(pos)] = BG

    return grid


def _expected_output(plan: dict) -> np.ndarray:
    inp = _render_input(plan)
    out = inp.copy()

    nrows = int(plan["nrows"])
    ncols = int(plan["ncols"])
    room_h = int(plan["room_h"])
    room_w = int(plan["room_w"])
    wall_rows = _wall_lines(nrows, room_h)
    wall_cols = _wall_lines(ncols, room_w)
    room_colors: dict[Room, int] = plan["room_colors"]
    gap_edges: set[Edge] = plan["gap_edges"]

    for r in range(nrows):
        for c in range(ncols):
            room = (r, c)
            if room_colors[room] != BG:
                continue
            bounds = _room_bounds(wall_rows, wall_cols, room)
            for direction, neighbor_room in _neighbors(room, nrows, ncols):
                edge = _edge_key(room, neighbor_room)
                if edge not in gap_edges:
                    continue
                color = room_colors[neighbor_room]
                if color == BG:
                    continue
                for rr, cc in _arm_cells(bounds, direction):
                    out[rr, cc] = color
    return out


def _plan_has_visible_change(plan: dict) -> bool:
    inp = _render_input(plan)
    out = _expected_output(plan)
    return not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        plan = _build_plan()
        if plan is None:
            continue
        inp = _render_input(plan)
        expected = _expected_output(plan)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid room-gap puzzle")
