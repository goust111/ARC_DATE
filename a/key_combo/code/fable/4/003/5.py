from __future__ import annotations

import random

import numpy as np


BG = 0
SCENE_TOP = 2

# Visible ARC palette direction code: red=up, blue=down, green=left, yellow=right.
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
DIRECTIONS = {
    RED: (-1, 0),
    BLUE: (1, 0),
    GREEN: (0, -1),
    YELLOW: (0, 1),
}
DIRECTION_COLORS = tuple(DIRECTIONS)
MARKER_AND_BLOCKER_COLORS = (5, 6, 7, 8, 9)

GEN_ATTEMPTS = 700


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _inside_scene(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return SCENE_TOP <= int(r) < h and 0 <= int(c) < w


def _parse_grid(
    grid: np.ndarray,
) -> tuple[int, list[int], int, tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert h >= 6 and w >= 6 and max(g.shape) <= 30, "invalid ARC canvas size"

    bg = _background(g)
    row0 = g[0]
    tape_cols = np.flatnonzero(np.isin(row0, DIRECTION_COLORS))
    assert tape_cols.size >= 4, "expected a top instruction tape"
    c0 = int(tape_cols[0])
    c1 = int(tape_cols[-1])
    assert np.array_equal(tape_cols, np.arange(c0, c1 + 1)), "instruction tape must be contiguous"
    assert np.all(np.isin(row0[c0 : c1 + 1], DIRECTION_COLORS)), "tape must use direction colors"
    assert np.all(row0[:c0] == bg) and np.all(row0[c1 + 1 :] == bg), "only the tape may appear on the top row"
    assert np.all(g[1, :] == bg), "one background separator row must split tape and scene"
    tape = [int(x) for x in row0[c0 : c1 + 1]]

    scene = g[SCENE_TOP:, :]
    assert not np.any(np.isin(scene, DIRECTION_COLORS)), "direction colors belong only to the tape in the input"
    scene_colors = [int(c) for c in np.unique(scene) if int(c) != bg]
    assert len(scene_colors) >= 2, "expected one start marker color and one blocker color"

    counts = {color: int(np.count_nonzero(scene == color)) for color in scene_colors}
    singleton_colors = [color for color, count in counts.items() if count == 1]
    assert len(singleton_colors) == 1, "the start marker must be the only singleton scene color"
    start_color = singleton_colors[0]
    start_rcs = np.argwhere(g == start_color)
    assert start_rcs.shape[0] == 1, "expected exactly one start marker"
    start = tuple(map(int, start_rcs[0]))
    assert _inside_scene(g.shape, *start), "start marker must lie in the scene"

    obstacle_mask = np.zeros(g.shape, dtype=bool)
    for color in scene_colors:
        if color == start_color:
            continue
        obstacle_mask |= g == color
    assert int(obstacle_mask.sum()) > 0, "expected at least one obstacle pixel"
    return bg, tape, start_color, start, obstacle_mask


def _solve_parts(
    grid: np.ndarray,
    bg: int,
    tape: list[int],
    start_color: int,
    start: tuple[int, int],
    obstacle_mask: np.ndarray,
    *,
    collect_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    r, c = map(int, start)
    out[r, c] = bg
    trace: list[dict] = []

    for color in tape:
        dr, dc = DIRECTIONS[int(color)]
        steps = 0
        painted = 0
        path: list[tuple[int, int]] = []
        reason = "edge"

        while True:
            nr = r + dr
            nc = c + dc
            if not _inside_scene(g.shape, nr, nc):
                reason = "edge"
                break
            if bool(obstacle_mask[nr, nc]):
                reason = "obstacle"
                break

            r, c = nr, nc
            steps += 1
            path.append((r, c))
            if int(out[r, c]) == bg:
                out[r, c] = int(color)
                painted += 1

        if collect_trace:
            trace.append(
                {
                    "color": int(color),
                    "steps": steps,
                    "painted": painted,
                    "reason": reason,
                    "end": (r, c),
                    "path": tuple(path),
                }
            )

    out[r, c] = int(start_color)
    if collect_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute a top instruction tape as a path-drawing program through a blocker terrain.

    Input structure:
    - The top row contains one contiguous one-cell-high instruction tape, separated from the scene by one all-background row.
    - Tape colors use a fixed visible direction code: red means up, blue means down, green means left, and yellow means right.
    - The scene below the separator contains one singleton start-marker color and repeated obstacle pixels on the background.

    Rule:
    - Treat the start marker as the moving cursor: clear its original cell, then execute the tape left-to-right.
    - For each instruction, move in its coded direction inside the scene until the next scene cell is an obstacle or outside the scene; paint only newly walked background cells with that instruction color.
    - After all instructions, paint the final cursor cell with the start-marker color, while preserving the tape and all obstacles.

    Key insight:
    - The tape supplies only the ordered turns; the terrain supplies every segment length, so the final drawing is determined by executing the program against the blockers and scene boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg, tape, start_color, start, obstacle_mask = _parse_grid(g)
    return _solve_parts(g, bg, tape, start_color, start, obstacle_mask)  # type: ignore[return-value]


# --------------- generator ---------------


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    nr, nc = r + dr, c + dc
    while _inside_scene(shape, nr, nc):
        cells.append((nr, nc))
        nr += dr
        nc += dc
    return cells


def _sample_tape(width: int) -> tuple[int, list[int]]:
    length = random.randint(8, min(13, width - 2))
    tape = list(DIRECTION_COLORS)
    while len(tape) < length:
        tape.append(random.choice(DIRECTION_COLORS))
    random.shuffle(tape)
    c0 = random.randint(0, width - length)
    return c0, tape


def _candidate_actions(
    shape: tuple[int, int],
    pos: tuple[int, int],
    color: int,
    obstacles: set[tuple[int, int]],
    used_path: set[tuple[int, int]],
    start: tuple[int, int],
) -> list[dict]:
    ray = _ray_cells(shape, pos, DIRECTIONS[int(color)])
    first_obstacle: int | None = None
    for idx, cell in enumerate(ray):
        if cell in obstacles:
            first_obstacle = idx
            break

    actions: list[dict] = []
    max_blocker_index = first_obstacle if first_obstacle is not None else len(ray) - 1
    if max_blocker_index >= 0:
        for blocker_index in range(max_blocker_index + 1):
            path = tuple(ray[:blocker_index])
            blocker = ray[blocker_index]
            if any(cell in used_path for cell in path):
                continue
            if blocker not in obstacles and (blocker in used_path or blocker == start):
                continue
            actions.append(
                {
                    "reason": "obstacle",
                    "steps": blocker_index,
                    "path": path,
                    "blocker": blocker if blocker not in obstacles else None,
                }
            )

    if first_obstacle is None:
        path = tuple(ray)
        if path and len(path) <= 8 and not any(cell in used_path for cell in path):
            actions.append({"reason": "edge", "steps": len(path), "path": path, "blocker": None})
        elif not path:
            actions.append({"reason": "edge", "steps": 0, "path": path, "blocker": None})

    return actions


def _choose_action(actions: list[dict], need_zero: bool, need_edge: bool, need_color_progress: bool) -> dict | None:
    if not actions:
        return None

    positive = [a for a in actions if int(a["steps"]) > 0]
    zero_obstacle = [a for a in actions if int(a["steps"]) == 0 and a["reason"] == "obstacle"]
    edge = [a for a in actions if a["reason"] == "edge" and int(a["steps"]) > 0]
    positive_obstacle = [a for a in actions if a["reason"] == "obstacle" and int(a["steps"]) > 0]

    if need_color_progress and positive:
        pool = positive
    elif need_zero and zero_obstacle and random.random() < 0.55:
        pool = zero_obstacle
    elif need_edge and edge and random.random() < 0.50:
        pool = edge
    else:
        weighted = positive_obstacle * 5 + edge * 3 + zero_obstacle * 2
        pool = weighted if weighted else actions
    return random.choice(pool)


def _plan_scene(
    shape: tuple[int, int],
    tape: list[int],
    start: tuple[int, int],
) -> tuple[set[tuple[int, int]], list[dict]] | None:
    obstacles: set[tuple[int, int]] = set()
    used_path: set[tuple[int, int]] = set()
    trace: list[dict] = []
    pos = start
    progressed_colors: set[int] = set()

    for i, color in enumerate(tape):
        remaining_by_color = {c: tape[i:].count(c) for c in DIRECTION_COLORS}
        need_color_progress = int(color) not in progressed_colors and remaining_by_color[int(color)] <= 2
        need_zero = not any(t["reason"] == "obstacle" and t["steps"] == 0 for t in trace)
        need_edge = not any(t["reason"] == "edge" and t["steps"] > 0 for t in trace)

        actions = _candidate_actions(shape, pos, int(color), obstacles, used_path, start)
        action = _choose_action(actions, need_zero, need_edge, need_color_progress)
        if action is None:
            return None

        blocker = action["blocker"]
        if blocker is not None:
            obstacles.add(blocker)
        for cell in action["path"]:
            used_path.add(cell)
        if action["path"]:
            pos = action["path"][-1]
            progressed_colors.add(int(color))

        trace.append(
            {
                "color": int(color),
                "steps": int(action["steps"]),
                "reason": str(action["reason"]),
                "end": pos,
                "path": tuple(action["path"]),
            }
        )

    return obstacles, trace


def _add_harmless_obstacles(
    grid: np.ndarray,
    obstacle_color: int,
    occupied: set[tuple[int, int]],
    count: int,
) -> None:
    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(SCENE_TOP, h)
        for c in range(w)
        if (r, c) not in occupied and int(grid[r, c]) == BG
    ]
    random.shuffle(candidates)
    for r, c in candidates[:count]:
        grid[r, c] = int(obstacle_color)
        occupied.add((r, c))


def _passes_semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg, tape, start_color, start, obstacle_mask = _parse_grid(grid)
        out, trace = _solve_parts(
            grid,
            bg,
            tape,
            start_color,
            start,
            obstacle_mask,
            collect_trace=True,
        )
    except AssertionError:
        return False

    painted_by_color = {color: 0 for color in DIRECTION_COLORS}
    for item in trace:
        painted_by_color[int(item["color"])] += int(item["painted"])
    if not all(painted_by_color[color] > 0 for color in DIRECTION_COLORS):
        return False
    if not any(item["reason"] == "obstacle" and int(item["steps"]) == 0 for item in trace):
        return False
    if not any(item["reason"] == "obstacle" and int(item["steps"]) > 0 for item in trace):
        return False
    if not any(item["reason"] == "edge" and int(item["steps"]) > 0 for item in trace):
        return False
    if sum(painted_by_color.values()) < 8:
        return False

    reverse_out = _solve_parts(grid, bg, list(reversed(tape)), start_color, start, obstacle_mask)
    if np.array_equal(out, reverse_out):
        return False

    no_obstacle_grid = grid.copy()
    no_obstacle_grid[obstacle_mask] = bg
    no_obstacle_out, no_obstacle_trace = _solve_parts(
        no_obstacle_grid,
        bg,
        tape,
        start_color,
        start,
        np.zeros_like(obstacle_mask),
        collect_trace=True,
    )
    original_paths = tuple(item["path"] for item in trace)
    terrainless_paths = tuple(item["path"] for item in no_obstacle_trace)
    if original_paths == terrainless_paths or np.array_equal(out[~obstacle_mask], no_obstacle_out[~obstacle_mask]):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 24)
    width = random.randint(13, 26)
    grid = np.full((height, width), BG, dtype=int)

    tape_c0, tape = _sample_tape(width)
    grid[0, tape_c0 : tape_c0 + len(tape)] = np.array(tape, dtype=int)

    marker_color, obstacle_color = random.sample(MARKER_AND_BLOCKER_COLORS, 2)
    start = (random.randint(SCENE_TOP + 2, height - 3), random.randint(2, width - 3))

    planned = _plan_scene(grid.shape, tape, start)
    if planned is None:
        return None
    obstacles, trace = planned
    if len(obstacles) < 3:
        return None

    occupied = {start} | set(obstacles)
    for r, c in obstacles:
        grid[r, c] = int(obstacle_color)
    grid[start] = int(marker_color)

    used_path = {cell for item in trace for cell in item["path"]}
    _add_harmless_obstacles(
        grid,
        obstacle_color,
        occupied | used_path,
        random.randint(0, 4),
    )

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if not _passes_semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instruction-tape terrain puzzle")
