from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
MAGENTA = 6

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

TOWER_COLORS = (1, 2, 3, 4, 7, 8, 9)
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
TOWER_COUNT_RANGE = (2, 4)
MIRROR_COUNT_CHOICES = (0, 1, 2)
MIRROR_COUNT_WEIGHTS = (3, 4, 1)
GENERATE_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (int(a[0] + b[0]), int(a[1] + b[1]))


def _reflect(direction: tuple[int, int]) -> tuple[int, int]:
    return DOWN if direction[0] == 0 else RIGHT


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _tower_cells(body_top_left: tuple[int, int], direction: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r0, c0 = body_top_left
    body = [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]
    if direction == UP:
        arrow = (r0 - 1, c0 + 1)
    elif direction == DOWN:
        arrow = (r0 + 3, c0 + 1)
    elif direction == LEFT:
        arrow = (r0 + 1, c0 - 1)
    else:
        assert direction == RIGHT
        arrow = (r0 + 1, c0 + 3)
    return body, arrow


def _tower_from_component(cells: list[tuple[int, int]], color: int) -> dict | None:
    if len(cells) != 10:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    rel = {(r - r0, c - c0) for r, c in cells}

    patterns = {
        UP: ({(rr, cc) for rr in range(1, 4) for cc in range(3)} | {(0, 1)}, (r0 + 1, c0)),
        DOWN: ({(rr, cc) for rr in range(3) for cc in range(3)} | {(3, 1)}, (r0, c0)),
        LEFT: ({(rr, cc) for rr in range(3) for cc in range(1, 4)} | {(1, 0)}, (r0, c0 + 1)),
        RIGHT: ({(rr, cc) for rr in range(3) for cc in range(3)} | {(1, 3)}, (r0, c0)),
    }

    for direction, (pattern, body_top_left) in patterns.items():
        if rel != pattern:
            continue
        body, arrow = _tower_cells(body_top_left, direction)
        return {
            "color": int(color),
            "direction": direction,
            "body_top_left": body_top_left,
            "body": tuple(body),
            "arrow": arrow,
            "emit_start": _add(arrow, direction),
        }
    return None


def _find_towers(grid: np.ndarray, bg: int) -> list[dict]:
    towers: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in {bg, GREY, MAGENTA}:
            continue
        for comp in _components_of_color(grid, color):
            tower = _tower_from_component(comp, color)
            assert tower is not None, "non-background non-special colors must be tower shapes"
            towers.append(tower)
    towers.sort(key=lambda t: (t["body_top_left"][0], t["body_top_left"][1], t["color"]))
    assert towers, "expected at least one tower"
    return towers


def _receiver_from_component(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...] | None:
    if len(cells) != 3:
        return None
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if (h, w) not in {(1, 3), (3, 1)}:
        return None
    expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    if set(cells) != expected:
        return None
    return tuple(sorted(cells))


def _find_receivers(grid: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    receivers: list[tuple[tuple[int, int], ...]] = []
    for comp in _components_of_color(grid, GREY):
        receiver = _receiver_from_component(comp)
        assert receiver is not None, "grey components must be 1x3 or 3x1 receivers"
        receivers.append(receiver)
    return receivers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a ray from each 3x3-plus-arrow tower, turning on magenta mirrors and
    recoloring grey receivers when hit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    towers = _find_towers(g, bg)
    receivers = _find_receivers(g)
    receiver_by_cell: dict[tuple[int, int], int] = {}
    for idx, receiver in enumerate(receivers):
        for cell in receiver:
            receiver_by_cell[cell] = int(idx)

    out = g.copy()
    h, w = g.shape
    for tower in towers:
        color = int(tower["color"])
        dr, dc = tower["direction"]
        r, c = tower["emit_start"]
        while 0 <= r < h and 0 <= c < w:
            cell = (int(r), int(c))
            value = int(g[r, c])
            if value == MAGENTA:
                out[r, c] = color
                dr, dc = _reflect((dr, dc))
            elif cell in receiver_by_cell:
                for rr, cc in receivers[receiver_by_cell[cell]]:
                    out[rr, cc] = color
                break
            elif value == bg:
                out[r, c] = color
            r += dr
            c += dc
    return out


def _all_in_bounds(shape: tuple[int, int], cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in cells)


def _can_place_object(
    shape: tuple[int, int],
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    blocked_exact: set[tuple[int, int]],
    existing_objects: set[tuple[int, int]],
) -> bool:
    if not _all_in_bounds(shape, cells):
        return False
    for r, c in cells:
        if (r, c) in blocked_exact:
            return False
        for er, ec in existing_objects:
            if max(abs(er - r), abs(ec - c)) <= 1:
                return False
    return True


def _line_free_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    blocked_exact: set[tuple[int, int]],
    local_used: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], bool]:
    h, w = shape
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and (r, c) not in blocked_exact and (r, c) not in local_used:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells, not (0 <= r < h and 0 <= c < w)


def _receiver_options(hit: tuple[int, int], direction: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    r, c = hit
    options: list[tuple[tuple[int, int], ...]] = []
    if direction in {LEFT, RIGHT}:
        options.append(((r - 1, c), (r, c), (r + 1, c)))
        if direction == RIGHT:
            options.append(((r, c), (r, c + 1), (r, c + 2)))
        else:
            options.append(((r, c), (r, c - 1), (r, c - 2)))
    else:
        options.append(((r, c - 1), (r, c), (r, c + 1)))
        if direction == DOWN:
            options.append(((r, c), (r + 1, c), (r + 2, c)))
        else:
            options.append(((r, c), (r - 1, c), (r - 2, c)))
    random.shuffle(options)
    return options


def _merge_plan(prefix_path: list[tuple[int, int]], mirror: tuple[int, int], suffix: dict) -> dict:
    return {
        "path_cells": prefix_path + suffix["path_cells"],
        "mirror_cells": [mirror] + suffix["mirror_cells"],
        "receiver_cells": suffix["receiver_cells"],
    }


def _plan_ray(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    mirrors_left: int,
    terminal: str,
    blocked_exact: set[tuple[int, int]],
    existing_objects: set[tuple[int, int]],
) -> dict | None:
    def dfs(
        current: tuple[int, int],
        current_dir: tuple[int, int],
        remaining: int,
        local_used: set[tuple[int, int]],
        local_objects: set[tuple[int, int]],
    ) -> dict | None:
        line, exits_grid = _line_free_cells(shape, current, current_dir, blocked_exact, local_used)
        if not line:
            return None

        if remaining == 0:
            if terminal == "boundary" and exits_grid and len(line) >= 2:
                return {"path_cells": list(line), "mirror_cells": [], "receiver_cells": None}

            if terminal != "receiver":
                return None

            hit_indices = list(range(1, len(line)))
            random.shuffle(hit_indices)
            for idx in hit_indices:
                hit = line[idx]
                bg_prefix = list(line[:idx])
                for receiver_cells in _receiver_options(hit, current_dir):
                    if any(cell in local_used for cell in receiver_cells):
                        continue
                    if not _can_place_object(
                        shape,
                        receiver_cells,
                        blocked_exact | local_used,
                        existing_objects | local_objects,
                    ):
                        continue
                    return {
                        "path_cells": bg_prefix,
                        "mirror_cells": [],
                        "receiver_cells": receiver_cells,
                    }
            return None

        mirror_indices = list(range(1, len(line)))
        random.shuffle(mirror_indices)
        for idx in mirror_indices:
            mirror = line[idx]
            next_dir = _reflect(current_dir)
            next_start = _add(mirror, next_dir)
            if not _in_bounds(shape, next_start[0], next_start[1]):
                continue
            if next_start in blocked_exact or next_start in local_used:
                continue
            if not _can_place_object(shape, [mirror], blocked_exact | local_used, existing_objects | local_objects):
                continue

            suffix = dfs(
                next_start,
                next_dir,
                remaining - 1,
                local_used | set(line[:idx]) | {mirror},
                local_objects | {mirror},
            )
            if suffix is not None:
                return _merge_plan(list(line[:idx]), mirror, suffix)
        return None

    return dfs(start, direction, mirrors_left, set(), set())


def _tower_body_ranges(shape: tuple[int, int], direction: tuple[int, int]) -> tuple[range, range]:
    h, w = shape
    if direction == UP:
        return range(2, h - 2), range(0, w - 2)
    if direction == DOWN:
        return range(0, h - 4), range(0, w - 2)
    if direction == LEFT:
        return range(0, h - 2), range(2, w - 2)
    assert direction == RIGHT
    return range(0, h - 2), range(0, w - 4)


def _scenario_list(n_towers: int) -> list[tuple[int, str, tuple[int, int]]]:
    scenarios: list[tuple[int, str, tuple[int, int]]] = []
    for _ in range(n_towers):
        mirror_count = random.choices(MIRROR_COUNT_CHOICES, weights=MIRROR_COUNT_WEIGHTS, k=1)[0]
        terminal = "receiver" if random.random() < 0.7 else "boundary"
        direction = random.choice(DIRS)
        scenarios.append((int(mirror_count), terminal, direction))

    if all(terminal != "receiver" for _mirrors, terminal, _direction in scenarios):
        idx = random.randrange(n_towers)
        scenarios[idx] = (scenarios[idx][0], "receiver", scenarios[idx][2])
    if n_towers >= 2 and all(terminal == "receiver" for _mirrors, terminal, _direction in scenarios) and random.random() < 0.6:
        idx = random.randrange(n_towers)
        scenarios[idx] = (scenarios[idx][0], "boundary", scenarios[idx][2])
    if all(mirror_count == 0 for mirror_count, _terminal, _direction in scenarios) and random.random() < 0.75:
        idx = random.randrange(n_towers)
        scenarios[idx] = (random.choice((1, 2)), scenarios[idx][1], scenarios[idx][2])

    random.shuffle(scenarios)
    return scenarios


def _render_input(shape: tuple[int, int], bg: int, plans: list[dict]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for plan in plans:
        for r, c in plan["body"]:
            grid[r, c] = int(plan["color"])
        ar, ac = plan["arrow"]
        grid[ar, ac] = int(plan["color"])
    for plan in plans:
        for r, c in plan["mirror_cells"]:
            grid[r, c] = MAGENTA
        receiver_cells = plan["receiver_cells"]
        if receiver_cells is not None:
            for r, c in receiver_cells:
                grid[r, c] = GREY
    return grid


def _render_expected_output(inp: np.ndarray, plans: list[dict]) -> np.ndarray:
    out = inp.copy()
    for plan in plans:
        color = int(plan["color"])
        for r, c in plan["path_cells"]:
            out[r, c] = color
        for r, c in plan["mirror_cells"]:
            out[r, c] = color
        receiver_cells = plan["receiver_cells"]
        if receiver_cells is not None:
            for r, c in receiver_cells:
                out[r, c] = color
    return out


def _attempt_generate() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    bg = 0
    n_towers = random.randint(*TOWER_COUNT_RANGE)
    colors = random.sample([c for c in TOWER_COLORS if c != bg], n_towers)
    scenarios = _scenario_list(n_towers)

    blocked_exact: set[tuple[int, int]] = set()
    object_cells: set[tuple[int, int]] = set()
    plans: list[dict] = []

    for color, (mirror_count, terminal, direction) in zip(colors, scenarios):
        row_range, col_range = _tower_body_ranges(shape, direction)
        candidates = [(r0, c0) for r0 in row_range for c0 in col_range]
        random.shuffle(candidates)
        placed = False

        for r0, c0 in candidates:
            body, arrow = _tower_cells((r0, c0), direction)
            tower_cells = body + [arrow]
            if not _can_place_object(shape, tower_cells, blocked_exact, object_cells):
                continue

            emit_start = _add(arrow, direction)
            if not _in_bounds(shape, emit_start[0], emit_start[1]):
                continue

            ray = _plan_ray(
                shape=shape,
                start=emit_start,
                direction=direction,
                mirrors_left=mirror_count,
                terminal=terminal,
                blocked_exact=blocked_exact | set(tower_cells),
                existing_objects=object_cells | set(tower_cells),
            )
            if ray is None:
                continue

            plan = {
                "color": int(color),
                "direction": direction,
                "body_top_left": (int(r0), int(c0)),
                "body": tuple(body),
                "arrow": arrow,
                "emit_start": emit_start,
                "path_cells": ray["path_cells"],
                "mirror_cells": ray["mirror_cells"],
                "receiver_cells": ray["receiver_cells"],
            }
            plans.append(plan)

            blocked_exact |= set(tower_cells)
            blocked_exact |= set(ray["path_cells"])
            blocked_exact |= set(ray["mirror_cells"])
            if ray["receiver_cells"] is not None:
                blocked_exact |= set(ray["receiver_cells"])
            object_cells |= set(tower_cells)
            object_cells |= set(ray["mirror_cells"])
            if ray["receiver_cells"] is not None:
                object_cells |= set(ray["receiver_cells"])
            placed = True
            break

        if not placed:
            return None

    inp = _render_input(shape, bg, plans)
    expected = _render_expected_output(inp, plans)

    if _most_common_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        inp = _attempt_generate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
