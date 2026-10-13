import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = (BLUE, GREEN, YELLOW, GREY)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
GEN_ATTEMPTS = 400


SPRITE_PATTERNS = {
    0: {
        "cells": {(0, 1), (1, 0), (1, 1), (1, 2)},
        "center": (1, 1),
    },
    1: {
        "cells": {(0, 0), (1, 0), (1, 1), (2, 0)},
        "center": (1, 0),
    },
    2: {
        "cells": {(0, 0), (0, 1), (0, 2), (1, 1)},
        "center": (0, 1),
    },
    3: {
        "cells": {(0, 1), (1, 0), (1, 1), (2, 1)},
        "center": (1, 1),
    },
}


BASE_PLANS = {
    3: {
        3: [
            {"start": 0, "init_dir": 0, "marker_cells": [1, 4, 7], "dirs": [1, 2, 2]},
        ],
        4: [
            {"start": 0, "init_dir": 0, "marker_cells": [2, 5, 4, 1], "dirs": [1, 2, 3, 0]},
        ],
        5: [
            {"start": 0, "init_dir": 0, "marker_cells": [1, 7, 8, 5, 4], "dirs": [1, 2, 1, 0, 3]},
        ],
    },
    4: {
        3: [
            {"start": 0, "init_dir": 0, "marker_cells": [1, 5, 9], "dirs": [1, 2, 2]},
        ],
        4: [
            {"start": 0, "init_dir": 0, "marker_cells": [2, 6, 5, 1], "dirs": [1, 2, 3, 0]},
        ],
        5: [
            {"start": 0, "init_dir": 0, "marker_cells": [3, 7, 6, 2, 1], "dirs": [1, 2, 3, 0, 3]},
            {"start": 0, "init_dir": 0, "marker_cells": [1, 9, 10, 6, 5], "dirs": [1, 2, 1, 0, 3]},
        ],
    },
}


def _reverse_dir(dir_idx: int) -> int:
    return (int(dir_idx) + 2) % 4


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    ar, ac = a
    br, bc = b
    if ac == bc:
        if br < ar:
            return 0
        if br > ar:
            return 2
    if ar == br:
        if bc > ac:
            return 1
        if bc < ac:
            return 3
    return None


def _build_layout(cols: int, cell_h: int, cell_w: int) -> tuple[np.ndarray, list[dict], np.ndarray]:
    rows = 3
    divider_rows = [0]
    for _ in range(rows):
        divider_rows.append(divider_rows[-1] + int(cell_h) + 1)
    divider_cols = [0]
    for _ in range(cols):
        divider_cols.append(divider_cols[-1] + int(cell_w) + 1)

    height = divider_rows[-1] + 1
    width = divider_cols[-1] + 1
    grid = np.full((height, width), BLACK, dtype=int)
    for rr in divider_rows:
        grid[int(rr), :] = ORANGE
    for cc in divider_cols:
        grid[:, int(cc)] = ORANGE

    cells: list[dict] = []
    cell_lookup = np.full((height, width), -1, dtype=int)
    for r in range(rows):
        for c in range(cols):
            r0 = int(divider_rows[r] + 1)
            r1 = int(divider_rows[r + 1] - 1)
            c0 = int(divider_cols[c] + 1)
            c1 = int(divider_cols[c + 1] - 1)
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            cid = len(cells)
            cells.append(
                {
                    "id": cid,
                    "rc": (r, c),
                    "span": (r0, r1, c0, c1),
                    "center": center,
                }
            )
            cell_lookup[r0 : r1 + 1, c0 : c1 + 1] = cid
    return grid, cells, cell_lookup


def _detect_layout(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == ORANGE)]
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == ORANGE)]
    assert len(divider_rows) >= 4, "expected 3 cell rows bounded by orange dividers"
    assert len(divider_cols) >= 4, "expected 3 or 4 cell columns bounded by orange dividers"

    cells: list[dict] = []
    cell_lookup = np.full(grid.shape, -1, dtype=int)
    for r in range(len(divider_rows) - 1):
        r0 = divider_rows[r] + 1
        r1 = divider_rows[r + 1] - 1
        if r0 > r1:
            continue
        for c in range(len(divider_cols) - 1):
            c0 = divider_cols[c] + 1
            c1 = divider_cols[c + 1] - 1
            if c0 > c1:
                continue
            cid = len(cells)
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            cells.append(
                {
                    "id": cid,
                    "rc": (r, c),
                    "span": (r0, r1, c0, c1),
                    "center": center,
                }
            )
            cell_lookup[r0 : r1 + 1, c0 : c1 + 1] = cid
    assert len({rc[0] for rc in (cell["rc"] for cell in cells)}) == 3, "expected exactly 3 cell rows"
    assert len({rc[1] for rc in (cell["rc"] for cell in cells)}) in {3, 4}, "expected 3 or 4 cell columns"
    return cells, cell_lookup


def _sprite_cells(center: tuple[int, int], dir_idx: int) -> list[tuple[int, int]]:
    center_r, center_c = center
    pat = SPRITE_PATTERNS[int(dir_idx)]
    off_r, off_c = pat["center"]
    return [
        (center_r + rr - off_r, center_c + cc - off_c)
        for rr, cc in sorted(pat["cells"])
    ]


def _find_sprite(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    coords = np.argwhere(grid == RED)
    assert coords.shape[0] == 4, "expected one 4-pixel red sprite"
    r0, c0 = coords.min(axis=0)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    for dir_idx, info in SPRITE_PATTERNS.items():
        if rel == info["cells"]:
            center = (int(r0 + info["center"][0]), int(c0 + info["center"][1]))
            return center, int(dir_idx)
    raise AssertionError("red component is not a recognized arrow sprite")


def _marker_map(grid: np.ndarray) -> dict[tuple[int, int], int]:
    markers: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(np.isin(grid, MARKER_COLORS)):
        markers[(int(r), int(c))] = int(grid[r, c])
    return markers


def _trace_cells(
    start: tuple[int, int], end: tuple[int, int], cell_lookup: np.ndarray
) -> list[int]:
    dir_idx = _direction(start, end)
    assert dir_idx is not None, "movement must be axis-aligned"
    dr, dc = DIRS4[dir_idx]
    r, c = start
    out: list[int] = []
    steps = abs(int(end[0]) - int(start[0])) + abs(int(end[1]) - int(start[1]))
    for _ in range(steps):
        r += dr
        c += dc
        cid = int(cell_lookup[r, c])
        if cid >= 0 and (not out or out[-1] != cid):
            out.append(cid)
    return out


def _choose_target(
    current: tuple[int, int],
    facing: int,
    markers: dict[tuple[int, int], int],
) -> tuple[int, tuple[int, int]] | None:
    candidates: list[tuple[int, int, tuple[int, int]]] = []
    for pos in markers:
        dir_idx = _direction(current, pos)
        if dir_idx is None or dir_idx == _reverse_dir(facing):
            continue
        dist = abs(pos[0] - current[0]) + abs(pos[1] - current[1])
        candidates.append((int(dir_idx), int(dist), pos))
    if not candidates:
        return None
    dirs_present = {dir_idx for dir_idx, _dist, _pos in candidates}
    assert len(dirs_present) == 1, "expected a unique non-reverse search ray"
    dir_idx = next(iter(dirs_present))
    reachable = [(dist, pos) for dd, dist, pos in candidates if dd == dir_idx]
    reachable.sort()
    return int(dir_idx), reachable[0][1]


def _simulate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    cells, cell_lookup = _detect_layout(g)
    current, facing = _find_sprite(g)
    markers = _marker_map(g)
    start_cell = int(cell_lookup[current])

    visited = {start_cell}
    entered_order: list[int] = []
    cell_trace_steps: list[list[int]] = []
    colors: list[int] = []
    marker_positions: list[tuple[int, int]] = []
    marker_cells: list[int] = []
    dirs: list[int] = []

    while markers:
        choice = _choose_target(current, facing, markers)
        assert choice is not None, "generator should leave no stranded markers"
        dir_idx, target = choice
        traversed = _trace_cells(current, target, cell_lookup)
        newly_entered: list[int] = []
        for cid in traversed:
            if cid not in visited:
                visited.add(cid)
                entered_order.append(cid)
                newly_entered.append(cid)
        marker_positions.append(target)
        marker_cells.append(int(cell_lookup[target]))
        colors.append(int(markers.pop(target)))
        cell_trace_steps.append(newly_entered)
        dirs.append(int(dir_idx))
        current = target
        facing = dir_idx

    return {
        "cells": cells,
        "cell_lookup": cell_lookup,
        "start_cell": start_cell,
        "entered_order": entered_order,
        "step_new_cells": cell_trace_steps,
        "colors": colors,
        "marker_positions": marker_positions,
        "marker_cells": marker_cells,
        "dirs": dirs,
    }


def _render_output(shape: tuple[int, int], cells: list[dict], entered_order: list[int], colors: list[int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    interior = np.zeros(shape, dtype=bool)
    for cell in cells:
        r0, r1, c0, c1 = cell["span"]
        interior[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    out[~interior] = ORANGE
    for cid, color in zip(entered_order, colors):
        r0, r1, c0, c1 = cells[int(cid)]["span"]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill divider-grid rooms in the order first entered by a turning triangle, using the consumed marker colors as the paint sequence.

    Input structure:
    - A black grid is partitioned by full orange horizontal and vertical divider lines into a 3x3 or 4x3 room layout.
    - One room contains a red triangular arrow sprite; several other rooms contain isolated one-pixel markers in blue, green, yellow, or grey.
    - Divider lines remain solid and define the only room boundaries; the sprite and markers occupy ordinary room pixels.

    Rule:
    - Read the sprite's pointing direction from its 4-pixel arrow shape.
    - From the sprite center, consider all remaining markers on cardinal half-lines except the exact reverse of the current facing; exactly one such direction is valid, and the nearest marker on that ray is consumed.
    - After each consumption, move the sprite center to that marker and re-face in the movement direction.
    - Track the first time each room is entered along the whole trajectory, excluding the starting room.
    - Paint the first-entered rooms in that order with the consumed marker colors; erase the sprite and markers, and keep orange dividers unchanged.

    Key insight:
    - The output colors come from marker-consumption order, but the places that receive those colors are the rooms first entered along the path, so motion order and marker-room locations must both be reconstructed.
    """
    info = _simulate(grid)
    return _render_output(grid.shape, info["cells"], info["entered_order"], info["colors"])


def _cid_to_rc(cid: int, cols: int) -> tuple[int, int]:
    return divmod(int(cid), int(cols))


def _rc_to_cid(r: int, c: int, cols: int) -> int:
    return int(r) * int(cols) + int(c)


def _transform_dir(dir_idx: int, flip_h: bool, flip_v: bool) -> int:
    d = int(dir_idx)
    if flip_h:
        if d == 1:
            d = 3
        elif d == 3:
            d = 1
    if flip_v:
        if d == 0:
            d = 2
        elif d == 2:
            d = 0
    return d


def _transform_plan(plan: dict, cols: int, flip_h: bool, flip_v: bool) -> dict:
    def map_cell(cid: int) -> int:
        r, c = _cid_to_rc(cid, cols)
        if flip_v:
            r = 2 - r
        if flip_h:
            c = cols - 1 - c
        return _rc_to_cid(r, c, cols)

    return {
        "start": map_cell(int(plan["start"])),
        "init_dir": _transform_dir(int(plan["init_dir"]), flip_h, flip_v),
        "marker_cells": [map_cell(int(cid)) for cid in plan["marker_cells"]],
        "dirs": [_transform_dir(int(d), flip_h, flip_v) for d in plan["dirs"]],
    }


def _cell_positions(cell: dict) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = cell["span"]
    return [(int(r), int(c)) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)]


def _compatible_with_history(
    history: list[tuple[tuple[int, int], int, int, int]], pos: tuple[int, int]
) -> bool:
    for prev_pos, prev_facing, chosen_dir, chosen_dist in history:
        dir_idx = _direction(prev_pos, pos)
        if dir_idx is None or dir_idx == _reverse_dir(prev_facing):
            continue
        dist = abs(int(pos[0]) - int(prev_pos[0])) + abs(int(pos[1]) - int(prev_pos[1]))
        if dir_idx != int(chosen_dir) or dist <= int(chosen_dist):
            return False
    return True


def _assign_marker_positions(
    plan: dict, cells: list[dict]
) -> list[tuple[int, int]] | None:
    start_cell = cells[int(plan["start"])]
    start_pos = tuple(start_cell["center"])
    marker_cells = [int(cid) for cid in plan["marker_cells"]]
    dirs = [int(d) for d in plan["dirs"]]
    pos_choices = {cid: _cell_positions(cells[int(cid)]) for cid in set(marker_cells)}

    def rec(
        idx: int,
        current: tuple[int, int],
        facing: int,
        used: set[tuple[int, int]],
        history: list[tuple[tuple[int, int], int, int, int]],
        acc: list[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        if idx == len(marker_cells):
            return list(acc)
        target_cell = int(marker_cells[idx])
        step_dir = int(dirs[idx])
        candidates = [
            pos
            for pos in pos_choices[target_cell]
            if pos not in used
            and _direction(current, pos) == step_dir
            and _compatible_with_history(history, pos)
        ]
        random.shuffle(candidates)
        candidates.sort(
            key=lambda pos: abs(int(pos[0]) - int(current[0])) + abs(int(pos[1]) - int(current[1])),
            reverse=True,
        )
        for pos in candidates:
            dist = abs(int(pos[0]) - int(current[0])) + abs(int(pos[1]) - int(current[1]))
            res = rec(
                idx + 1,
                pos,
                step_dir,
                used | {pos},
                history + [(current, facing, step_dir, dist)],
                acc + [pos],
            )
            if res is not None:
                return res
        return None

    return rec(0, start_pos, int(plan["init_dir"]), {start_pos}, [], [])


def _trace_cell_path_by_ids(start_cid: int, end_cid: int, cols: int) -> list[int]:
    sr, sc = _cid_to_rc(start_cid, cols)
    er, ec = _cid_to_rc(end_cid, cols)
    if sr == er:
        step = 1 if ec > sc else -1
        return [_rc_to_cid(sr, c, cols) for c in range(sc + step, ec + step, step)]
    step = 1 if er > sr else -1
    return [_rc_to_cid(r, sc, cols) for r in range(sr + step, er + step, step)]


def _first_entered_order(plan: dict, cols: int) -> tuple[list[int], list[list[int]]]:
    current = int(plan["start"])
    visited = {current}
    entered: list[int] = []
    per_step: list[list[int]] = []
    for target in [int(cid) for cid in plan["marker_cells"]]:
        path = _trace_cell_path_by_ids(current, target, cols)
        new_cells = [cid for cid in path if cid not in visited]
        per_step.append(new_cells)
        for cid in new_cells:
            visited.add(cid)
            entered.append(cid)
        current = target
    return entered, per_step


def _sample_colors(n: int, marker_cells: list[int], entered_order: list[int]) -> list[int]:
    while True:
        colors = [int(random.choice(MARKER_COLORS)) for _ in range(n)]
        if len(set(colors)) < 2:
            continue
        if n >= 4 and len(set(colors)) < 3:
            continue
        same_cell_order = []
        cell_to_idx = {int(cid): i for i, cid in enumerate(marker_cells)}
        for cid in entered_order:
            same_cell_order.append(int(colors[cell_to_idx[int(cid)]]))
        if n >= 4 and same_cell_order == colors:
            continue
        return colors


def _plan_instance(cols: int, n_markers: int) -> dict:
    base = random.choice(BASE_PLANS[int(cols)][int(n_markers)])
    transformed = _transform_plan(
        base,
        int(cols),
        flip_h=bool(random.getrandbits(1)),
        flip_v=bool(random.getrandbits(1)),
    )
    entered_order, per_step = _first_entered_order(transformed, int(cols))
    transformed["entered_order"] = entered_order
    transformed["per_step"] = per_step
    return transformed


def _compose_input(cols: int, cell_h: int, cell_w: int, plan: dict, marker_positions: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    grid, cells, _lookup = _build_layout(int(cols), int(cell_h), int(cell_w))
    sprite_center = tuple(cells[int(plan["start"])]["center"])
    for r, c in _sprite_cells(sprite_center, int(plan["init_dir"])):
        grid[int(r), int(c)] = RED
    for pos, color in zip(marker_positions, colors):
        grid[int(pos[0]), int(pos[1])] = int(color)
    return grid


def _same_cell_heuristic_output(grid: np.ndarray) -> np.ndarray:
    info = _simulate(grid)
    marker_by_cell = {int(cid): int(color) for cid, color in zip(info["marker_cells"], info["colors"])}
    colors = [marker_by_cell[int(cid)] for cid in info["entered_order"]]
    return _render_output(grid.shape, info["cells"], info["entered_order"], colors)


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        info = _simulate(grid)
        out = main(grid)
    except AssertionError:
        return False

    n = len(info["colors"])
    if n not in {3, 4, 5}:
        return False
    if len(info["entered_order"]) != n:
        return False
    if len(set(info["entered_order"])) != n:
        return False
    if info["start_cell"] in info["marker_cells"]:
        return False
    if len(set(info["marker_cells"])) != n:
        return False
    if not np.array_equal(out, _render_output(grid.shape, info["cells"], info["entered_order"], info["colors"])):
        return False
    if n >= 4:
        step_counts = [len(step) for step in info["step_new_cells"]]
        if max(step_counts) < 2 or min(step_counts) > 0:
            return False
        if _same_cell_heuristic_output(grid).tolist() == out.tolist():
            return False
    if sum(info["dirs"][i] != info["dirs"][i - 1] for i in range(1, len(info["dirs"]))) < 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    cols = random.choice([3, 4])
    n_markers = random.choices([3, 4, 5], weights=[3, 4, 4], k=1)[0]

    cell_h = random.choice([3, 5])
    cell_w = random.choice([3, 5])
    plan = _plan_instance(int(cols), int(n_markers))

    _base_grid, cells, _lookup = _build_layout(int(cols), int(cell_h), int(cell_w))
    marker_positions = _assign_marker_positions(plan, cells)
    if marker_positions is None:
        return None

    colors = _sample_colors(int(n_markers), [int(cid) for cid in plan["marker_cells"]], [int(cid) for cid in plan["entered_order"]])
    grid = _compose_input(int(cols), int(cell_h), int(cell_w), plan, marker_positions, colors)
    if _semantic_checks(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
