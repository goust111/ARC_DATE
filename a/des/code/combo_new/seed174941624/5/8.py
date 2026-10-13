from __future__ import annotations

from collections import Counter
import random

import numpy as np


UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
PERPENDICULARS = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}
BACKSLASH_REFLECT = {
    RIGHT: DOWN,
    DOWN: RIGHT,
    LEFT: UP,
    UP: LEFT,
}
SLASH_REFLECT = {
    RIGHT: UP,
    UP: RIGHT,
    LEFT: DOWN,
    DOWN: LEFT,
}

MIRROR_COLOR = 5
ALL_COLORS = tuple(range(10))
NON_MIRROR_COLORS = tuple(color for color in ALL_COLORS if color != MIRROR_COLOR)
TOWER_COLORS = tuple(color for color in ALL_COLORS if color != MIRROR_COLOR)

GRID_MIN = 12
GRID_MAX = 24
GENERATE_RETRIES = 400
TOWER_RETRIES = 160
MAX_TOWERS = 4


def _add(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return (cell[0] + direction[0] * steps, cell[1] + direction[1] * steps)


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cell = stack.pop()
                coords.append(cell)
                for nr, nc in _neighbors4(cell):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _tower_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(coords) != 4:
        return None
    coord_set = set(coords)
    degrees = {
        cell: sum(neighbor in coord_set for neighbor in _neighbors4(cell))
        for cell in coords
    }
    centers = [cell for cell, degree in degrees.items() if degree == 3]
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    if len(centers) != 1 or len(endpoints) != 3:
        return None
    center = centers[0]
    endpoint_vectors = [(cell[0] - center[0], cell[1] - center[1]) for cell in endpoints]
    tip_dirs = [vec for vec in endpoint_vectors if (-vec[0], -vec[1]) not in endpoint_vectors]
    if len(tip_dirs) != 1 or tip_dirs[0] not in DIRECTIONS:
        return None
    direction = tip_dirs[0]
    tip = (center[0] + direction[0], center[1] + direction[1])
    if tip not in coord_set:
        return None
    return (tip, direction)


def _reflect(direction: tuple[int, int], mirror_type: str) -> tuple[int, int]:
    if mirror_type == "\\":
        return BACKSLASH_REFLECT[direction]
    return SLASH_REFLECT[direction]


def _infer_towers_and_mirrors(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], dict[tuple[int, int], str]]:
    towers: list[dict[str, object]] = []
    tower_cells: set[tuple[int, int]] = set()
    mirror_cells: set[tuple[int, int]] = set()
    for comp in _monochrome_components(grid, bg):
        coords = [tuple(map(int, cell)) for cell in comp["coords"]]
        signature = _tower_signature(coords)
        if signature is not None:
            tip, direction = signature
            towers.append(
                {
                    "color": int(comp["color"]),
                    "coords": coords,
                    "tip": tip,
                    "direction": direction,
                }
            )
            tower_cells.update(coords)
        else:
            mirror_cells.update(coords)

    mirror_type_by_cell: dict[tuple[int, int], str] = {}
    for cell in mirror_cells:
        options: list[str] = []
        if (cell[0] - 1, cell[1] - 1) in mirror_cells or (cell[0] + 1, cell[1] + 1) in mirror_cells:
            options.append("\\")
        if (cell[0] - 1, cell[1] + 1) in mirror_cells or (cell[0] + 1, cell[1] - 1) in mirror_cells:
            options.append("/")
        assert len(options) == 1, "each mirror cell must belong to exactly one diagonal mirror pair"
        mirror_type_by_cell[cell] = options[0]

    mirror_only_cells = set(mirror_type_by_cell)
    assert not (tower_cells & mirror_only_cells), "tower cells and mirror cells must be disjoint"
    return towers, mirror_type_by_cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch a ray from each T-shaped signal tower, reflecting off diagonal mirror pairs and painting the traversed background.

    Input structure:
    - The background is the dominant color.
    - Each signal tower is a monochrome 4-cell T shape whose stem direction
      indicates the launch direction.
    - The remaining non-background cells are diagonal two-pixel mirror pairs;
      one diagonal encodes "\" mirrors and the other encodes "/" mirrors.

    Rule:
    - For every tower, start from the stem tip and repeatedly inspect the next
      cell in the launch direction.
    - If the next cell is background, move into it and recolor it with that
      tower's color.
    - If the next cell is a mirror cell, stay in place and rotate the ray
      according to that mirror type; mirror cells themselves never change.
    - If the next step would leave the grid or enter any other non-background
      cell, stop that ray, and keep towers unchanged.

    Key insight:
    - The task combines three structural cues: the unique degree-3 center of a
      T shape reveals the launch tip, diagonal mirror pairs reveal the
      reflection rule, and the final output is the full reflected ray trace.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    towers, mirrors = _infer_towers_and_mirrors(g, bg)
    assert towers, "expected at least one T-shaped signal tower"

    out = g.copy()
    towers.sort(key=lambda tower: (tower["tip"][0], tower["tip"][1], tower["color"]))
    for tower in towers:
        color = int(tower["color"])
        cur = tuple(tower["tip"])
        direction = tuple(tower["direction"])
        seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        while True:
            state = (cur, direction)
            if state in seen_states:
                break
            seen_states.add(state)

            nxt = _add(cur, direction)
            if not _in_bounds(g.shape, nxt):
                break
            if nxt in mirrors:
                direction = _reflect(direction, mirrors[nxt])
                continue
            if int(out[nxt]) != bg:
                break
            out[nxt] = color
            cur = nxt

    return out


def _tower_cells_from_tip(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    center = _add(tip, (-direction[0], -direction[1]))
    arm_a, arm_b = PERPENDICULARS[direction]
    return {
        tip,
        center,
        _add(center, arm_a),
        _add(center, arm_b),
    }


def _cells_fit(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, cell) for cell in cells)


def _has_static_clearance(cells: set[tuple[int, int]], occupied_static: set[tuple[int, int]]) -> bool:
    if cells & occupied_static:
        return False
    for cell in cells:
        for neighbor in _neighbors4(cell):
            if neighbor in occupied_static and neighbor not in cells:
                return False
    return True


def _has_mirror_clearance(cells: set[tuple[int, int]], occupied_mirrors: set[tuple[int, int]]) -> bool:
    for cell in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = (cell[0] + dr, cell[1] + dc)
                if neighbor in occupied_mirrors and neighbor not in cells:
                    return False
    return True


def _free_path_cell(
    shape: tuple[int, int],
    cell: tuple[int, int],
    occupied_static: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    local_paths: set[tuple[int, int]],
) -> bool:
    return (
        _in_bounds(shape, cell)
        and cell not in occupied_static
        and cell not in reserved_paths
        and cell not in local_paths
    )


def _mirror_second_candidates(
    turn_cell: tuple[int, int],
    incoming: tuple[int, int],
    outgoing: tuple[int, int],
) -> list[tuple[int, int]]:
    head = _add(turn_cell, incoming)
    diag_a = (head[0] + incoming[0] + outgoing[0], head[1] + incoming[1] + outgoing[1])
    diag_b = (head[0] - incoming[0] - outgoing[0], head[1] - incoming[1] - outgoing[1])
    return [diag_a, diag_b]


def _finish_segment(
    shape: tuple[int, int],
    cur: tuple[int, int],
    direction: tuple[int, int],
    occupied_static: set[tuple[int, int]],
    blocker_towers: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    local_paths: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str] | None:
    cells: list[tuple[int, int]] = []
    step = 1
    while True:
        nxt = _add(cur, direction, step)
        if not _in_bounds(shape, nxt):
            return (cells, "boundary") if cells else None
        if nxt in reserved_paths or nxt in local_paths:
            return None
        if nxt in occupied_static:
            return (cells, "tower") if cells and nxt in blocker_towers else None
        cells.append(nxt)
        step += 1


def _plan_path_from(
    shape: tuple[int, int],
    cur: tuple[int, int],
    direction: tuple[int, int],
    turns_left: int,
    occupied_static_base: set[tuple[int, int]],
    occupied_mirrors_base: set[tuple[int, int]],
    blocker_towers: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    local_paths: set[tuple[int, int]],
    local_mirrors: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], str] | None:
    occupied_static = occupied_static_base | local_mirrors
    if turns_left == 0:
        finished = _finish_segment(
            shape,
            cur,
            direction,
            occupied_static,
            blocker_towers,
            reserved_paths,
            local_paths,
        )
        if finished is None:
            return None
        tail_cells, reason = finished
        return (local_paths | set(tail_cells), local_mirrors, reason)

    prefix: list[tuple[int, int]] = []
    candidates: list[tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    max_probe = max(shape)
    for steps in range(1, max_probe + 1):
        turn_cell = _add(cur, direction, steps)
        mirror_head = _add(cur, direction, steps + 1)
        if not _free_path_cell(shape, turn_cell, occupied_static, reserved_paths, local_paths):
            break
        prefix.append(turn_cell)
        if not _free_path_cell(shape, mirror_head, occupied_static, reserved_paths, local_paths):
            break
        for outgoing in PERPENDICULARS[direction]:
            first_after = _add(turn_cell, outgoing)
            if not _free_path_cell(shape, first_after, occupied_static, reserved_paths, set(prefix) | local_paths):
                continue
            for second_mirror in _mirror_second_candidates(turn_cell, direction, outgoing):
                candidate_mirrors = {mirror_head, second_mirror}
                if not _cells_fit(shape, candidate_mirrors):
                    continue
                if not _has_static_clearance(candidate_mirrors, occupied_static):
                    continue
                if not _has_mirror_clearance(candidate_mirrors, occupied_mirrors_base | local_mirrors):
                    continue
                if second_mirror in reserved_paths or second_mirror in local_paths or second_mirror in prefix:
                    continue
                candidates.append((prefix.copy(), turn_cell, outgoing, mirror_head, second_mirror))

    random.shuffle(candidates)
    for prefix_cells, turn_cell, outgoing, mirror_head, second_mirror in candidates:
        next_paths = local_paths | set(prefix_cells)
        next_mirrors = local_mirrors | {mirror_head, second_mirror}
        result = _plan_path_from(
            shape,
            turn_cell,
            outgoing,
            turns_left - 1,
            occupied_static_base,
            occupied_mirrors_base,
            blocker_towers,
            reserved_paths,
            next_paths,
            next_mirrors,
        )
        if result is not None:
            return result

    return None


def _place_one_tower(
    shape: tuple[int, int],
    direction: tuple[int, int],
    turns: int,
    color: int,
    occupied_towers: set[tuple[int, int]],
    occupied_mirrors: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> dict[str, object] | None:
    occupied_static = occupied_towers | occupied_mirrors
    all_tips = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    random.shuffle(all_tips)

    for tip in all_tips[: TOWER_RETRIES]:
        tower_cells = _tower_cells_from_tip(tip, direction)
        if not _cells_fit(shape, tower_cells):
            continue
        if tower_cells & reserved_paths:
            continue
        if not _has_static_clearance(tower_cells, occupied_static):
            continue

        result = _plan_path_from(
            shape=shape,
            cur=tip,
            direction=direction,
            turns_left=turns,
            occupied_static_base=occupied_static | tower_cells,
            occupied_mirrors_base=occupied_mirrors,
            blocker_towers=occupied_towers,
            reserved_paths=reserved_paths,
            local_paths=set(),
            local_mirrors=set(),
        )
        if result is None:
            continue

        path_cells, mirror_cells, stop_reason = result
        if stop_reason == "tower" and not occupied_towers:
            continue
        if len(path_cells) < 2:
            continue
        return {
            "color": color,
            "tip": tip,
            "direction": direction,
            "tower_cells": tower_cells,
            "path_cells": path_cells,
            "mirror_cells": mirror_cells,
        }

    return None


def _tower_colors(bg: int, count: int) -> list[int]:
    palette = [color for color in TOWER_COLORS if color != bg]
    return random.sample(palette, count)


def _normalize_direction_order(count: int) -> list[tuple[int, int]]:
    if count <= len(DIRECTIONS):
        dirs = list(DIRECTIONS)
        random.shuffle(dirs)
        return dirs[:count]
    dirs = list(DIRECTIONS)
    while len(dirs) < count:
        dirs.append(random.choice(DIRECTIONS))
    random.shuffle(dirs)
    return dirs


def _turn_count_sequence(count: int, max_turns: int) -> list[int]:
    choices = list(range(max_turns + 1))
    turns = [random.choice(choices) for _ in range(count)]
    if max_turns >= 1 and all(turn == 0 for turn in turns):
        turns[random.randrange(count)] = 1
    return turns


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(NON_MIRROR_COLORS)

    max_towers = min(MAX_TOWERS, max(2, (h * w) // 120 + 1))
    tower_count = random.randint(2, max_towers) if max_towers >= 2 else 1
    colors = _tower_colors(bg, tower_count)
    directions = _normalize_direction_order(tower_count)
    max_turns = 3 if min(h, w) >= 16 else 2
    turn_counts = _turn_count_sequence(tower_count, max_turns)

    specs = [
        {"color": color, "direction": direction, "turns": turns}
        for color, direction, turns in zip(colors, directions, turn_counts)
    ]
    random.shuffle(specs)
    specs.sort(key=lambda spec: (-spec["turns"], random.random()))

    occupied_towers: set[tuple[int, int]] = set()
    occupied_mirrors: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    plans: list[dict[str, object]] = []

    for spec in specs:
        plan = _place_one_tower(
            shape=(h, w),
            direction=spec["direction"],
            turns=int(spec["turns"]),
            color=int(spec["color"]),
            occupied_towers=occupied_towers,
            occupied_mirrors=occupied_mirrors,
            reserved_paths=reserved_paths,
        )
        if plan is None:
            return None
        occupied_towers |= set(plan["tower_cells"])
        occupied_mirrors |= set(plan["mirror_cells"])
        reserved_paths |= set(plan["path_cells"])
        plans.append(plan)

    total_turns = sum(len(plan["mirror_cells"]) // 2 for plan in plans)
    if total_turns == 0:
        return None

    grid = np.full((h, w), bg, dtype=int)
    expected = grid.copy()
    for plan in plans:
        for cell in plan["tower_cells"]:
            grid[cell] = int(plan["color"])
            expected[cell] = int(plan["color"])
        for cell in plan["mirror_cells"]:
            grid[cell] = MIRROR_COLOR
            expected[cell] = MIRROR_COLOR
        for cell in plan["path_cells"]:
            expected[cell] = int(plan["color"])

    bg_count = int(np.sum(grid == bg))
    if bg_count <= len(occupied_towers) + len(occupied_mirrors):
        return None

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-signal input")
