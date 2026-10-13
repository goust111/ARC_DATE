from __future__ import annotations

import random

import numpy as np


BG_ATTEMPTS = 500
COLOR_POOL = tuple(range(10))
DIR_UP = (-1, 0)
DIR_RIGHT = (0, 1)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
ALL_DIRS = (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT)
TOWER_COUNT_RANGE = (2, 5)
GRID_H_RANGE = (12, 26)
GRID_W_RANGE = (12, 26)
PRISM_STOP_PROB = 0.35


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _rot_cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _rot_ccw_local(local: tuple[int, int]) -> tuple[int, int]:
    r, c = local
    return 1 - c, r


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((cr, cc))
                for dr, dc in ALL_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _tower_from_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    if len(comp) != 2:
        return None
    (r1, c1), (r2, c2) = comp
    if abs(r1 - r2) + abs(c1 - c2) != 1:
        return None
    color = int(grid[r1, c1])
    if int(grid[r2, c2]) != color:
        return None

    if r1 == r2:
        row = int(r1)
        c0, c1 = sorted((int(c1), int(c2)))
        cells = {(row, c0), (row, c1)}
        return {
            "color": color,
            "orientation": "H",
            "cells": cells,
            "sort_key": (row, c0),
            "emitters": [
                {"start": (row, c0 - 1), "direction": DIR_LEFT},
                {"start": (row, c1 + 1), "direction": DIR_RIGHT},
            ],
        }

    col = int(c1)
    r0, r1 = sorted((int(r1), int(r2)))
    cells = {(r0, col), (r1, col)}
    return {
        "color": color,
        "orientation": "V",
        "cells": cells,
        "sort_key": (r0, col),
        "emitters": [
            {"start": (r0 - 1, col), "direction": DIR_UP},
            {"start": (r1 + 1, col), "direction": DIR_DOWN},
        ],
    }


def _prism_from_component(grid: np.ndarray, bg: int, comp: list[tuple[int, int]]) -> dict | None:
    if len(comp) != 4:
        return None
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 != r0 + 1 or c1 != c0 + 1:
        return None
    cells = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
    if set(comp) != cells:
        return None

    patch = np.asarray(grid[r0 : r0 + 2, c0 : c0 + 2], dtype=int)
    if np.any(patch == bg):
        return None
    if not (int(patch[0, 0]) == int(patch[1, 1]) and int(patch[0, 1]) == int(patch[1, 0])):
        return None
    if int(patch[0, 0]) == int(patch[0, 1]):
        return None
    return {
        "top_left": (int(r0), int(c0)),
        "cells": cells,
        "patch": patch.copy(),
    }


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int], dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)

    towers: list[dict] = []
    tower_cells: set[tuple[int, int]] = set()
    prism_by_cell: dict[tuple[int, int], dict] = {}

    for comp in _components(g != bg):
        tower = _tower_from_component(g, comp)
        if tower is not None:
            towers.append(tower)
            tower_cells.update(tower["cells"])
            continue

        prism = _prism_from_component(g, bg, comp)
        assert prism is not None, "every non-background object must be a tower or a 2x2 prism"
        for cell in prism["cells"]:
            prism_by_cell[cell] = prism

    assert towers, "expected at least one signal tower"
    assert prism_by_cell, "expected at least one prism"
    towers.sort(key=lambda tower: tower["sort_key"])
    return bg, towers, prism_by_cell, tower_cells


def _deflect_from_prism(
    grid: np.ndarray,
    prism: dict,
    hit_cell: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], int]:
    r0, c0 = prism["top_left"]
    lr = hit_cell[0] - r0
    lc = hit_cell[1] - c0
    assert (lr, lc) in {(0, 0), (0, 1), (1, 0), (1, 1)}

    dr, dc = direction
    if dc == 1:
        assert lc == 0
    elif dc == -1:
        assert lc == 1
    elif dr == 1:
        assert lr == 0
    else:
        assert dr == -1 and lr == 1

    er, ec = _rot_ccw_local((lr, lc))
    exit_cell = (r0 + er, c0 + ec)
    next_direction = _rot_cw(direction)
    next_start = (exit_cell[0] + next_direction[0], exit_cell[1] + next_direction[1])
    next_color = int(grid[exit_cell])
    return next_start, next_direction, next_color


def _cast_ray(
    grid: np.ndarray,
    out: np.ndarray,
    bg: int,
    tower_cells: set[tuple[int, int]],
    prism_by_cell: dict[tuple[int, int], dict],
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    allow_turn: bool,
) -> None:
    h, w = map(int, grid.shape)
    dr, dc = direction
    r, c = start

    while 0 <= r < h and 0 <= c < w:
        cell = (r, c)
        if int(grid[cell]) == bg:
            out[cell] = int(color)
            r += dr
            c += dc
            continue

        if cell in tower_cells:
            return

        prism = prism_by_cell.get(cell)
        assert prism is not None, "unexpected foreground object on ray path"
        if not allow_turn:
            return
        if int(grid[cell]) != int(color):
            return

        next_start, next_direction, next_color = _deflect_from_prism(grid, prism, cell, direction)
        _cast_ray(
            grid,
            out,
            bg,
            tower_cells,
            prism_by_cell,
            next_start,
            next_direction,
            next_color,
            False,
        )
        return


def main(grid: np.ndarray) -> np.ndarray:
    """Emit colored rays from domino towers and bend matching rays once through 2x2 diagonal prisms.

    Input structure:
    - The background is the most common color.
    - Some foreground objects are monochrome size-2 domino towers, either horizontal or vertical.
    - The remaining foreground objects are detached solid 2x2 prisms whose two diagonals use two different colors.

    Rule:
    - Process the towers in scan order by their top-left cell. A horizontal tower emits leftward from its left cell and rightward from its right cell; a vertical tower emits upward from its top cell and downward from its bottom cell.
    - Each ray paints every background cell on its row or column until it reaches the grid edge or a foreground object. If the first object hit is a prism and the contacted prism cell matches the ray color, do not paint the prism: rotate the travel direction 90 degrees clockwise, switch to the prism's other diagonal color, and continue from the corresponding perpendicular side.
    - The turned ray travels straight only once more and then stops at the boundary, at any prism, or at any tower. Original tower and prism pixels are never changed, and later towers overwrite earlier ray colors on shared background cells.

    Key insight:
    - A tower's orientation determines its emission axis, while a prism's 2x2 diagonal layout fixes both which incoming color can enter and which new color appears after the clockwise bend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, towers, prism_by_cell, tower_cells = _analyze_scene(g)
    out = g.copy()
    for tower in towers:
        color = int(tower["color"])
        for emitter in tower["emitters"]:
            _cast_ray(
                g,
                out,
                bg,
                tower_cells,
                prism_by_cell,
                emitter["start"],
                emitter["direction"],
                color,
                True,
            )
    return out


def _can_place_object(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied and (rr, cc) not in cells:
                    return False
    return True


def _make_tower(r: int, c: int, orientation: str, color: int) -> dict:
    if orientation == "H":
        cells = {(r, c), (r, c + 1)}
    else:
        cells = {(r, c), (r + 1, c)}
    return {
        "color": int(color),
        "orientation": orientation,
        "cells": cells,
    }


def _tower_emissions(tower: dict) -> list[dict]:
    color = int(tower["color"])
    cells = sorted(tower["cells"])
    if tower["orientation"] == "H":
        row = cells[0][0]
        c0 = min(c for _r, c in cells)
        c1 = max(c for _r, c in cells)
        return [
            {"tower_color": color, "start": (row, c0 - 1), "direction": DIR_LEFT},
            {"tower_color": color, "start": (row, c1 + 1), "direction": DIR_RIGHT},
        ]

    col = cells[0][1]
    r0 = min(r for r, _c in cells)
    r1 = max(r for r, _c in cells)
    return [
        {"tower_color": color, "start": (r0 - 1, col), "direction": DIR_UP},
        {"tower_color": color, "start": (r1 + 1, col), "direction": DIR_DOWN},
    ]


def _trace_free_line(
    start: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    h, w = shape
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and (r, c) not in occupied:
        cells.append((r, c))
        r += dr
        c += dc
    blocker = (r, c) if 0 <= r < h and 0 <= c < w and (r, c) in occupied else None
    return cells, blocker


def _entry_locals(direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == DIR_RIGHT:
        return [(0, 0), (1, 0)]
    if direction == DIR_LEFT:
        return [(0, 1), (1, 1)]
    if direction == DIR_DOWN:
        return [(0, 0), (0, 1)]
    return [(1, 0), (1, 1)]


def _prism_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _paint_prism(
    grid: np.ndarray,
    r0: int,
    c0: int,
    entry_local: tuple[int, int],
    incident_color: int,
    exit_color: int,
) -> None:
    grid[r0 : r0 + 2, c0 : c0 + 2] = int(exit_color)
    er, ec = entry_local
    grid[r0 + er, c0 + ec] = int(incident_color)
    grid[r0 + (1 - er), c0 + (1 - ec)] = int(incident_color)


def _other_path_union(emissions: list[dict], keep_index: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for idx, emission in enumerate(emissions):
        if idx == keep_index:
            continue
        cells.update(emission["base_path"])
    return cells


def _place_blocker_prism(
    grid: np.ndarray,
    bg: int,
    turn_path: list[tuple[int, int]],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if len(turn_path) < 2:
        return None

    options = list(range(1, len(turn_path)))
    random.shuffle(options)
    shape = grid.shape
    for idx in options:
        entry = turn_path[idx]
        locals_ = _entry_locals(direction)
        random.shuffle(locals_)
        for local in locals_:
            r0 = entry[0] - local[0]
            c0 = entry[1] - local[1]
            cells = _prism_cells(r0, c0)
            if cells & forbidden:
                continue
            if not _can_place_object(cells, occupied, shape):
                continue
            if any(cell in forbidden for cell in cells):
                continue
            colors = [color for color in COLOR_POOL if color != bg]
            color_a, color_b = random.sample(colors, 2)
            _paint_prism(grid, r0, c0, local, color_a, color_b)
            occupied.update(cells)
            return cells
    return None


def _place_turning_prism(
    grid: np.ndarray,
    bg: int,
    emission: dict,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    path = emission["base_path"]
    if len(path) < 3:
        return False

    indices = list(range(1, len(path)))
    random.shuffle(indices)
    shape = grid.shape
    direction = emission["direction"]
    incident_color = int(emission["tower_color"])

    for path_index in indices:
        entry = path[path_index]
        locals_ = _entry_locals(direction)
        random.shuffle(locals_)
        for local in locals_:
            r0 = entry[0] - local[0]
            c0 = entry[1] - local[1]
            cells = _prism_cells(r0, c0)
            if cells & forbidden:
                continue
            if not _can_place_object(cells, occupied, shape):
                continue

            exit_local = _rot_ccw_local(local)
            exit_cell = (r0 + exit_local[0], c0 + exit_local[1])
            next_direction = _rot_cw(direction)
            next_start = (exit_cell[0] + next_direction[0], exit_cell[1] + next_direction[1])

            occupied_after = occupied | cells
            turn_path, _turn_blocker = _trace_free_line(next_start, next_direction, occupied_after, shape)
            if len(turn_path) < 1:
                continue

            exit_choices = [color for color in COLOR_POOL if color not in (bg, incident_color)]
            exit_color = random.choice(exit_choices)
            _paint_prism(grid, r0, c0, local, incident_color, exit_color)
            occupied.update(cells)

            if random.random() < PRISM_STOP_PROB:
                _place_blocker_prism(grid, bg, turn_path, next_direction, occupied, forbidden)
            return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice(COLOR_POOL)
    colors = [color for color in COLOR_POOL if color != bg]

    tower_count = random.randint(*TOWER_COUNT_RANGE)
    if tower_count > len(colors):
        return None
    tower_colors = random.sample(colors, tower_count)

    orientations = []
    if tower_count >= 2:
        orientations.extend(["H", "V"])
    while len(orientations) < tower_count:
        orientations.append(random.choice(("H", "V")))
    random.shuffle(orientations)

    towers: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    shape = (h, w)

    for color, orientation in zip(tower_colors, orientations):
        placed = False
        for _ in range(120):
            if orientation == "H":
                r = random.randint(2, h - 3)
                c = random.randint(2, w - 4)
            else:
                r = random.randint(2, h - 4)
                c = random.randint(2, w - 3)
            tower = _make_tower(r, c, orientation, color)
            if not _can_place_object(tower["cells"], occupied, shape):
                continue
            towers.append(tower)
            occupied.update(tower["cells"])
            placed = True
            break
        if not placed:
            return None

    emissions: list[dict] = []
    for tower in towers:
        for emission in _tower_emissions(tower):
            base_path, blocker = _trace_free_line(emission["start"], emission["direction"], occupied, shape)
            emission["base_path"] = base_path
            emission["base_blocker"] = blocker
            emissions.append(emission)

    candidates = [idx for idx, emission in enumerate(emissions) if len(emission["base_path"]) >= 3]
    if not candidates:
        return None

    max_prisms = min(len(candidates), max(1, len(emissions) - 1), 4)
    target_prisms = random.randint(1, max_prisms)
    selected = candidates[:]
    random.shuffle(selected)

    grid = np.full((h, w), bg, dtype=int)
    for tower in towers:
        for r, c in tower["cells"]:
            grid[r, c] = int(tower["color"])

    placed_prisms = 0
    for emission_index in selected:
        if placed_prisms >= target_prisms:
            break
        forbidden = _other_path_union(emissions, emission_index)
        if _place_turning_prism(grid, bg, emissions[emission_index], occupied, forbidden):
            placed_prisms += 1

    if placed_prisms != target_prisms:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(BG_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prism-tower ARC input")
