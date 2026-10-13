from __future__ import annotations

import random

import numpy as np


UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
SCENARIOS = ("clear", "pass", "multi", "block")
GEN_RETRIES = 500


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _tower_from_component(coords: np.ndarray) -> dict | None:
    if int(coords.shape[0]) != 11:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    if h == 5 and w == 3:
        row_counts = mask.sum(axis=1).astype(int).tolist()
        if row_counts == [1, 1, 3, 3, 3] and bool(mask[0, 1]) and bool(mask[1, 1]):
            return {"direction": UP, "tip": (r0, c0 + 1)}
        if row_counts == [3, 3, 3, 1, 1] and bool(mask[3, 1]) and bool(mask[4, 1]):
            return {"direction": DOWN, "tip": (r1, c0 + 1)}

    if h == 3 and w == 5:
        col_counts = mask.sum(axis=0).astype(int).tolist()
        if col_counts == [1, 1, 3, 3, 3] and bool(mask[1, 0]) and bool(mask[1, 1]):
            return {"direction": LEFT, "tip": (r0 + 1, c0)}
        if col_counts == [3, 3, 3, 1, 1] and bool(mask[1, 3]) and bool(mask[1, 4]):
            return {"direction": RIGHT, "tip": (r0 + 1, c1)}

    return None


def _lens_from_component(coords: np.ndarray, grid: np.ndarray) -> dict | None:
    if int(coords.shape[0]) != 3:
        return None

    pts = {tuple(map(int, coord)) for coord in coords}
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    if r0 == r1 and c1 - c0 == 2:
        ordered = [(r0, c0), (r0, c0 + 1), (r0, c0 + 2)]
        if set(ordered) != pts:
            return None
        a0 = int(grid[ordered[0]])
        b = int(grid[ordered[1]])
        a1 = int(grid[ordered[2]])
        if a0 == a1 and a0 != b:
            return {
                "orientation": "h",
                "cells": ordered,
                "center": ordered[1],
                "middle_color": b,
            }

    if c0 == c1 and r1 - r0 == 2:
        ordered = [(r0, c0), (r0 + 1, c0), (r0 + 2, c0)]
        if set(ordered) != pts:
            return None
        a0 = int(grid[ordered[0]])
        b = int(grid[ordered[1]])
        a1 = int(grid[ordered[2]])
        if a0 == a1 and a0 != b:
            return {
                "orientation": "v",
                "cells": ordered,
                "center": ordered[1],
                "middle_color": b,
            }

    return None


def _direction_is_vertical(direction: tuple[int, int]) -> bool:
    return bool(direction[0] != 0)


def _perpendicular_lens(lens: dict, direction: tuple[int, int]) -> bool:
    return (_direction_is_vertical(direction) and lens["orientation"] == "h") or (
        not _direction_is_vertical(direction) and lens["orientation"] == "v"
    )


def _find_towers_and_lenses(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    towers: list[dict] = []
    tower_mask = np.zeros(grid.shape, dtype=bool)

    for color in sorted(int(x) for x in np.unique(grid) if int(x) != int(bg)):
        for comp in _components(grid == int(color)):
            tower = _tower_from_component(comp)
            if tower is None:
                continue
            tower["color"] = int(color)
            tower["cells"] = {tuple(map(int, coord)) for coord in comp}
            towers.append(tower)
            tower_mask[comp[:, 0], comp[:, 1]] = True

    lenses: list[dict] = []
    for comp in _components((grid != int(bg)) & (~tower_mask)):
        lens = _lens_from_component(comp, grid)
        if lens is not None:
            lenses.append(lens)

    return towers, lenses


def _cast_rays(grid: np.ndarray, towers: list[dict], lenses: list[dict], bg: int) -> np.ndarray:
    h, w = grid.shape
    out = np.array(grid, dtype=int, copy=True)
    cell_to_lens: dict[tuple[int, int], dict] = {}
    for lens in lenses:
        for cell in lens["cells"]:
            cell_to_lens[cell] = lens

    for tower in sorted(towers, key=lambda item: (item["tip"][0], item["tip"][1], item["color"])):
        color = int(tower["color"])
        dr, dc = tower["direction"]
        r, c = map(int, tower["tip"])
        while True:
            r += int(dr)
            c += int(dc)
            if not (0 <= r < h and 0 <= c < w):
                break
            lens = cell_to_lens.get((r, c))
            if lens is not None:
                if (r, c) == lens["center"] and _perpendicular_lens(lens, tower["direction"]):
                    color = int(lens["middle_color"])
                    continue
                break
            if int(grid[r, c]) != int(bg):
                break
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find signal towers, cast rays from their antenna tips, recolor rays when they pass through
    perpendicular A-B-A lenses, and stop them before parallel lenses.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    towers, lenses = _find_towers_and_lenses(g, bg)
    return _cast_rays(g, towers, lenses, bg)


def _tower_cells(tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, tip)
    if direction == UP:
        return [(r, c), (r + 1, c)] + [(rr, cc) for rr in range(r + 2, r + 5) for cc in range(c - 1, c + 2)]
    if direction == DOWN:
        return [(r, c), (r - 1, c)] + [(rr, cc) for rr in range(r - 4, r - 1) for cc in range(c - 1, c + 2)]
    if direction == LEFT:
        return [(r, c), (r, c + 1)] + [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c + 2, c + 5)]
    return [(r, c), (r, c - 1)] + [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 4, c - 1)]


def _ray_path(tip: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    dr, dc = direction
    r, c = map(int, tip)
    path: list[tuple[int, int]] = []
    while True:
        r += int(dr)
        c += int(dc)
        if not (0 <= r < h and 0 <= c < w):
            break
        path.append((r, c))
    return path


def _sample_spaced(values: list[int], count: int, min_gap: int, retries: int = 300) -> list[int] | None:
    if len(values) < count:
        return None
    for _ in range(retries):
        chosen = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _choose_lens_colors(palette: list[int], current_color: int) -> tuple[int, int]:
    middle_choices = [color for color in palette if int(color) != int(current_color)]
    middle = int(random.choice(middle_choices))
    end_choices = [color for color in palette if int(color) != int(middle)]
    end = int(random.choice(end_choices))
    return end, middle


def _make_pass_lens(
    center: tuple[int, int], direction: tuple[int, int], palette: list[int], current_color: int
) -> dict:
    r, c = map(int, center)
    end_color, middle_color = _choose_lens_colors(palette, current_color)
    if _direction_is_vertical(direction):
        cells = [(r, c - 1), (r, c), (r, c + 1)]
        orientation = "h"
    else:
        cells = [(r - 1, c), (r, c), (r + 1, c)]
        orientation = "v"
    return {
        "orientation": orientation,
        "cells": cells,
        "center": center,
        "colors": (end_color, middle_color, end_color),
        "middle_color": middle_color,
    }


def _make_block_lens(
    segment: list[tuple[int, int]], direction: tuple[int, int], palette: list[int], current_color: int
) -> dict:
    end_color, middle_color = _choose_lens_colors(palette, current_color)
    orientation = "v" if _direction_is_vertical(direction) else "h"
    return {
        "orientation": orientation,
        "cells": list(segment),
        "center": segment[1],
        "colors": (end_color, middle_color, end_color),
        "middle_color": middle_color,
    }


def _draw_lens(grid: np.ndarray, lens: dict) -> None:
    for (r, c), color in zip(lens["cells"], lens["colors"]):
        grid[int(r), int(c)] = int(color)


def _scenario_min_path(scenario: str) -> int:
    return {"clear": 4, "pass": 5, "multi": 7, "block": 5}[scenario]


def _choose_tip(direction: tuple[int, int], lane: int, scenario: str, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = map(int, shape)
    min_path = _scenario_min_path(scenario)
    if direction == UP:
        lo, hi = min_path, h - 5
        return None if lo > hi else (random.randint(lo, hi), lane)
    if direction == DOWN:
        lo, hi = 4, h - 1 - min_path
        return None if lo > hi else (random.randint(lo, hi), lane)
    if direction == LEFT:
        lo, hi = min_path, w - 5
        return None if lo > hi else (lane, random.randint(lo, hi))
    lo, hi = 4, w - 1 - min_path
    return None if lo > hi else (lane, random.randint(lo, hi))


def _plan_lenses_for_tower(path: list[tuple[int, int]], direction: tuple[int, int], palette: list[int], tower_color: int, scenario: str) -> list[dict] | None:
    if scenario == "clear":
        return []

    current_color = int(tower_color)
    if scenario == "pass":
        idx = random.choice(list(range(1, len(path) - 1)))
        return [_make_pass_lens(path[idx], direction, palette, current_color)]

    if scenario == "multi":
        indices = _sample_spaced(list(range(1, len(path) - 1)), 2, 3)
        if indices is None:
            return None
        lenses: list[dict] = []
        for idx in indices:
            lens = _make_pass_lens(path[idx], direction, palette, current_color)
            lenses.append(lens)
            current_color = int(lens["middle_color"])
        return lenses

    if len(path) < 5:
        return None
    start = random.choice(list(range(1, len(path) - 3)))
    return [_make_block_lens(path[start : start + 3], direction, palette, current_color)]


def _build_expected(grid: np.ndarray, towers: list[dict], lenses: list[dict], bg: int) -> np.ndarray:
    return _cast_rays(grid, towers, lenses, bg)


def _generate_attempt() -> np.ndarray | None:
    n_towers = random.choices([2, 3, 4], weights=[1, 2, 2])[0]
    axis = random.choice(["vertical", "horizontal"])
    scenarios = (
        random.sample(["pass", "multi"], 1) + random.sample(["clear", "block"], 1)
        if n_towers == 2
        else random.sample(list(SCENARIOS), n_towers)
    )
    random.shuffle(scenarios)

    if axis == "vertical":
        h = random.randint(18, 26)
        w = random.randint(max(14, 5 * n_towers), 24)
        lanes = _sample_spaced(list(range(2, w - 2)), n_towers, 5)
        directions = [random.choice([UP, DOWN]) for _ in range(n_towers)]
    else:
        h = random.randint(max(14, 5 * n_towers), 24)
        w = random.randint(18, 26)
        lanes = _sample_spaced(list(range(2, h - 2)), n_towers, 5)
        directions = [random.choice([LEFT, RIGHT]) for _ in range(n_towers)]

    if lanes is None:
        return None

    bg = int(random.randrange(10))
    color_pool = [color for color in range(10) if int(color) != int(bg)]
    if len(color_pool) < n_towers + 2:
        return None
    tower_colors = random.sample(color_pool, n_towers)
    lens_palette = [color for color in color_pool if int(color) not in tower_colors]
    if len(lens_palette) < 2:
        return None

    grid = np.full((h, w), bg, dtype=int)
    towers: list[dict] = []
    lenses: list[dict] = []

    for lane, direction, color, scenario in zip(lanes, directions, tower_colors, scenarios):
        tip = _choose_tip(direction, int(lane), scenario, grid.shape)
        if tip is None:
            return None

        tower_cells = _tower_cells(tip, direction)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in tower_cells):
            return None
        if any(int(grid[r, c]) != int(bg) for r, c in tower_cells):
            return None
        for r, c in tower_cells:
            grid[r, c] = int(color)

        path = _ray_path(tip, direction, grid.shape)
        if len(path) < _scenario_min_path(scenario):
            return None

        planned = _plan_lenses_for_tower(path, direction, lens_palette, int(color), scenario)
        if planned is None:
            return None
        for lens in planned:
            if any(not (0 <= r < h and 0 <= c < w) for r, c in lens["cells"]):
                return None
            if any(int(grid[r, c]) != int(bg) for r, c in lens["cells"]):
                return None
            _draw_lens(grid, lens)
            lenses.append(lens)

        towers.append({"tip": tip, "direction": direction, "color": int(color), "cells": set(tower_cells)})

    if int(np.count_nonzero(grid != int(bg))) == 0:
        return None
    if _background_color(grid) != int(bg):
        return None

    expected = _build_expected(grid, towers, lenses, bg)
    if np.array_equal(expected, grid):
        return None

    detected_towers, detected_lenses = _find_towers_and_lenses(grid, bg)
    if len(detected_towers) != len(towers) or len(detected_lenses) != len(lenses):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
