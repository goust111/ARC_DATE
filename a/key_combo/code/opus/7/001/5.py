from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color
in_bounds = _ARC_UTILS.in_bounds

BG = Color.BLACK
OBSTACLE = Color.TEAL
START = Color.GREEN
END = Color.RED

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400


def _copy_positions(h: int, w: int, pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [
        (r, c),
        (r, 2 * w - 1 - c),
        (2 * h - 1 - r, c),
        (2 * h - 1 - r, 2 * w - 1 - c),
    ]


def _tile_reflect(base: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [base, np.fliplr(base)],
            [np.flipud(base), np.flipud(np.fliplr(base))],
        ]
    )


def _build_tiled_grid(
    base_obstacles: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    convert_start_copies: bool = True,
    convert_end_copies: bool = True,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    h, w = base_obstacles.shape
    grid = _tile_reflect(base_obstacles)
    copy_obstacles: set[tuple[int, int]] = set()

    for idx, pos in enumerate(_copy_positions(h, w, start)):
        if idx > 0 and convert_start_copies:
            grid[pos] = OBSTACLE
            copy_obstacles.add(pos)
    for idx, pos in enumerate(_copy_positions(h, w, end)):
        if idx > 0 and convert_end_copies:
            grid[pos] = OBSTACLE
            copy_obstacles.add(pos)

    grid[start] = START
    grid[end] = END
    return grid, copy_obstacles


def _turn_left(direction_idx: int) -> int:
    return (direction_idx - 1) % 4


def _turn_right(direction_idx: int) -> int:
    return (direction_idx + 1) % 4


def _initial_direction(grid: np.ndarray, start: tuple[int, int]) -> int:
    blockers = []
    for idx, (dr, dc) in enumerate(DIRS):
        nr, nc = start[0] + dr, start[1] + dc
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == OBSTACLE:
            blockers.append(idx)
    assert len(blockers) == 1, "start must have exactly one adjacent obstacle"
    return (blockers[0] + 2) % 4


def _simulate_path(
    grid: np.ndarray,
    *,
    require_end: bool,
    copy_obstacles: set[tuple[int, int]] | None = None,
) -> dict | None:
    start_cells = np.argwhere(grid == START)
    if len(start_cells) != 1:
        return None
    end_cells = np.argwhere(grid == END)
    if require_end and len(end_cells) != 1:
        return None

    start = tuple(map(int, start_cells[0]))
    end = tuple(map(int, end_cells[0])) if len(end_cells) == 1 else None
    direction = _initial_direction(grid, start)
    pos = start
    seen_states = {(pos, direction)}
    path: list[tuple[int, int]] = []
    turns: list[dict] = []
    max_steps = grid.shape[0] * grid.shape[1] * 8
    copy_obstacles = copy_obstacles or set()

    for _ in range(max_steps):
        dr, dc = DIRS[direction]
        nr, nc = pos[0] + dr, pos[1] + dc
        if end is not None and (nr, nc) == end:
            return {"ok": True, "path": path, "turns": turns}

        blocked = not in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == OBSTACLE
        if blocked:
            left_idx = _turn_left(direction)
            right_idx = _turn_right(direction)
            options = []
            for next_idx in (left_idx, right_idx):
                tr, tc = pos[0] + DIRS[next_idx][0], pos[1] + DIRS[next_idx][1]
                can_use = in_bounds(grid.shape, tr, tc) and int(grid[tr, tc]) != OBSTACLE
                options.append(can_use)
            if not any(options):
                return {"ok": False, "path": path, "turns": turns, "stop": "dead_end"}

            direction = left_idx if options[0] else right_idx
            turns.append(
                {
                    "cause": "boundary" if not in_bounds(grid.shape, nr, nc) else "obstacle",
                    "both_open": bool(options[0] and options[1]),
                    "pos": pos,
                    "blocked_cell": (nr, nc),
                    "blocked_is_copy": (nr, nc) in copy_obstacles,
                }
            )
            dr, dc = DIRS[direction]
            nr, nc = pos[0] + dr, pos[1] + dc
            if end is not None and (nr, nc) == end:
                return {"ok": True, "path": path, "turns": turns}

        if not in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == OBSTACLE:
            return {"ok": False, "path": path, "turns": turns, "stop": "invalid_step"}

        pos = (nr, nc)
        path.append(pos)
        state = (pos, direction)
        if state in seen_states:
            return {"ok": False, "path": path, "turns": turns, "stop": "loop"}
        seen_states.add(state)

    return {"ok": False, "path": path, "turns": turns, "stop": "step_cap"}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect-tile the small obstacle pattern into a four-quadrant maze and draw the deterministic turn-on-block path.

    Input structure:
    - A 5x5 to 7x7 black canvas contains teal obstacle cells plus exactly one green start and one red end.
    - The start and end lie strictly inside the small grid.
    - The start has exactly one 4-neighbor teal obstacle; its opposite direction is the initial heading.

    Rule:
    - Build a 2x2 reflected tiling: original in the top-left, left-right mirror in the top-right, up-down mirror in the bottom-left, and both mirrors in the bottom-right.
    - Turn every reflected copy of the green start and red end into teal obstacles, keeping only the original top-left start and end markers.
    - Starting at the green cell, move straight; when the next cell is teal or outside the tiled grid, turn 90 degrees with left priority, otherwise right. If both turns are blocked, stop.
    - If the current forward cell is the red end, stop before entering it, and color every visited path cell except the start and end green.

    Key insight:
    - The path is defined only after reflection tiling and copy-to-obstacle conversion, because mirrored quadrants create the usable corridors and extra blockers that force the turn sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 5 <= h <= 7 and 5 <= w <= 7, "expected a small ARC base grid"

    start_cells = np.argwhere(g == START)
    end_cells = np.argwhere(g == END)
    assert len(start_cells) == 1 and len(end_cells) == 1, "expected one start and one end"
    start = tuple(map(int, start_cells[0]))
    end = tuple(map(int, end_cells[0]))
    assert 0 < start[0] < h - 1 and 0 < start[1] < w - 1, "start may not lie on the border"
    assert 0 < end[0] < h - 1 and 0 < end[1] < w - 1, "end may not lie on the border"

    base_obstacles = np.where(g == OBSTACLE, OBSTACLE, BG)
    tiled, copy_obstacles = _build_tiled_grid(base_obstacles, start, end)
    sim = _simulate_path(tiled, require_end=True, copy_obstacles=copy_obstacles)
    assert sim is not None and sim["ok"], "generated maze must admit a valid path"

    out = tiled.copy()
    for r, c in sim["path"]:
        out[r, c] = START
    out[start] = START
    out[end] = END
    return out


def _base_has_valid_start_neighborhood(base: np.ndarray, start: tuple[int, int]) -> bool:
    obstacle_neighbors = 0
    for dr, dc in DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if int(base[nr, nc]) == OBSTACLE:
            obstacle_neighbors += 1
    return obstacle_neighbors == 1


def _path_signature(sim: dict | None) -> tuple:
    if sim is None:
        return ("missing",)
    return (
        bool(sim.get("ok")),
        tuple(sim.get("path", [])),
        tuple(
            (
                turn["cause"],
                turn["both_open"],
                turn["pos"],
                turn["blocked_cell"],
                turn["blocked_is_copy"],
            )
            for turn in sim.get("turns", [])
        ),
    )


def _small_grid_simulation(base_obstacles: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> dict | None:
    grid = base_obstacles.copy()
    grid[start] = START
    grid[end] = END
    return _simulate_path(grid, require_end=True)


def _generate_base_obstacles() -> tuple[np.ndarray, tuple[int, int]]:
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    density = random.uniform(0.12, 0.28)
    base = np.where(np.random.random((h, w)) < density, OBSTACLE, BG).astype(int)
    start = (random.randint(1, h - 2), random.randint(1, w - 2))
    blocked_side = random.randrange(4)

    base[start] = BG
    for idx, (dr, dc) in enumerate(DIRS):
        nr, nc = start[0] + dr, start[1] + dc
        base[nr, nc] = OBSTACLE if idx == blocked_side else BG
    return base, start


def _choose_end_and_validate(
    base_obstacles: np.ndarray,
    start: tuple[int, int],
    *,
    require_boundary_turn: bool,
) -> np.ndarray | None:
    h, w = base_obstacles.shape
    tiled_without_end, start_copy_obstacles = _build_tiled_grid(
        base_obstacles,
        start,
        start,
        convert_end_copies=False,
    )
    tiled_without_end[start] = START
    no_end_sim = _simulate_path(
        tiled_without_end,
        require_end=False,
        copy_obstacles=start_copy_obstacles,
    )
    if no_end_sim is None or len(no_end_sim["path"]) < 10:
        return None

    candidates = []
    seen = set()
    for pos in no_end_sim["path"][4:]:
        if pos in seen or pos == start:
            continue
        seen.add(pos)
        if pos[0] >= h or pos[1] >= w:
            continue
        if not (0 < pos[0] < h - 1 and 0 < pos[1] < w - 1):
            continue
        candidates.append(pos)
    random.shuffle(candidates)

    for end in candidates[:24]:
        tiled, copy_obstacles = _build_tiled_grid(base_obstacles, start, end)
        final_sim = _simulate_path(tiled, require_end=True, copy_obstacles=copy_obstacles)
        if final_sim is None or not final_sim["ok"]:
            continue

        turns = final_sim["turns"]
        boundary_present = any(turn["cause"] == "boundary" for turn in turns)
        if boundary_present != require_boundary_turn:
            continue
        if len(final_sim["path"]) < 6 or len(final_sim["path"]) > (tiled.size // 2):
            continue
        if len(turns) < 2:
            continue
        if not any(turn["both_open"] for turn in turns):
            continue
        if not any(turn["blocked_is_copy"] for turn in turns):
            continue
        if not any(r >= h or c >= w for r, c in final_sim["path"]):
            continue

        small_sim = _small_grid_simulation(base_obstacles, start, end)
        if small_sim is not None and small_sim.get("ok"):
            continue

        no_copy_grid, _ = _build_tiled_grid(
            base_obstacles,
            start,
            end,
            convert_start_copies=False,
            convert_end_copies=False,
        )
        no_copy_sim = _simulate_path(no_copy_grid, require_end=True)
        if _path_signature(no_copy_sim) == _path_signature(final_sim):
            continue

        input_grid = base_obstacles.copy()
        input_grid[start] = START
        input_grid[end] = END
        if not _base_has_valid_start_neighborhood(base_obstacles, start):
            continue
        return input_grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        require_boundary_turn = random.random() < 0.5
        base_obstacles, start = _generate_base_obstacles()
        grid = _choose_end_and_validate(
            base_obstacles,
            start,
            require_boundary_turn=require_boundary_turn,
        )
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-maze ARC input")
