from __future__ import annotations

import random
from collections import Counter

import numpy as np


YELLOW = 4
GREY = 5
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [0, 1, 2, 3, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_arrow(coords: list[tuple[int, int]], color: int) -> dict | None:
    if len(coords) != 5:
        return None
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    pts = set((int(r), int(c)) for r, c in coords)
    scan_key = min(pts)

    if h == 3 and w == 2:
        row_counts = [sum(1 for rr, _ in pts if rr == row) for row in range(r0, r1 + 1)]
        if row_counts == [1, 2, 2]:
            tip = next((int(r), int(c)) for r, c in pts if int(r) == int(r0))
            return {"color": int(color), "tip": tip, "dir": DIRS["up"], "scan_key": scan_key}
        if row_counts == [2, 2, 1]:
            tip = next((int(r), int(c)) for r, c in pts if int(r) == int(r1))
            return {"color": int(color), "tip": tip, "dir": DIRS["down"], "scan_key": scan_key}
        return None

    if h == 2 and w == 3:
        col_counts = [sum(1 for _, cc in pts if cc == col) for col in range(c0, c1 + 1)]
        if col_counts == [1, 2, 2]:
            tip = next((int(r), int(c)) for r, c in pts if int(c) == int(c0))
            return {"color": int(color), "tip": tip, "dir": DIRS["left"], "scan_key": scan_key}
        if col_counts == [2, 2, 1]:
            tip = next((int(r), int(c)) for r, c in pts if int(c) == int(c1))
            return {"color": int(color), "tip": tip, "dir": DIRS["right"], "scan_key": scan_key}
        return None

    return None


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    colors = sorted(set(map(int, grid.reshape(-1))) - {int(bg), YELLOW, GREY})
    arrows: list[dict] = []
    for color in colors:
        for comp in _components(grid == int(color)):
            arrow = _parse_arrow(comp, int(color))
            assert arrow is not None, f"unexpected non-arrow component for color {color}: {comp}"
            arrows.append(arrow)
    arrows.sort(key=lambda arrow: arrow["scan_key"])
    assert arrows, "expected at least one arrow"
    return arrows


def _cycle_colors(arrows: list[dict]) -> list[int]:
    cycle: list[int] = []
    seen: set[int] = set()
    for arrow in arrows:
        color = int(arrow["color"])
        if color not in seen:
            seen.add(color)
            cycle.append(color)
    assert cycle
    return cycle


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire rays from 5-pixel arrow sprites; yellow lenses cycle the ray color
    through the arrows' reading-order color list before the ray stops at a wall
    or the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = Counter(map(int, g.reshape(-1)))
    bg = int(max(counts.items(), key=lambda item: item[1])[0])
    arrows = _find_arrows(g, bg)
    cycle = _cycle_colors(arrows)
    nxt = {int(color): int(cycle[(idx + 1) % len(cycle)]) for idx, color in enumerate(cycle)}

    out = g.copy()
    h, w = g.shape

    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = map(int, arrow["dir"])
        r, c = map(int, arrow["tip"])
        while True:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                break
            cell = int(g[nr, nc])
            if cell == GREY:
                out[nr, nc] = color
                break
            if cell == YELLOW:
                color = int(nxt[color])
                r, c = nr, nc
                continue
            if cell == bg:
                out[nr, nc] = color
                r, c = nr, nc
                continue
            break

    return out


def _in_bounds(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = map(int, cell)
    return 0 <= r < h and 0 <= c < w


def _touches_any(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = map(int, cell)
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (int(rr), int(cc)) in occupied:
                return True
    return False


def _arrow_cells_from_tip(
    tip: tuple[int, int], direction: str, shift: int
) -> list[tuple[int, int]] | None:
    r, c = map(int, tip)
    if direction == "right":
        rows = [int(r + shift), int(r + shift + 1)]
        cols = [int(c - 2), int(c - 1)]
    elif direction == "left":
        rows = [int(r + shift), int(r + shift + 1)]
        cols = [int(c + 1), int(c + 2)]
    elif direction == "up":
        rows = [int(r + 1), int(r + 2)]
        cols = [int(c + shift), int(c + shift + 1)]
    elif direction == "down":
        rows = [int(r - 2), int(r - 1)]
        cols = [int(c + shift), int(c + shift + 1)]
    else:
        raise ValueError(direction)

    cells = [(int(r), int(c))]
    for rr in rows:
        for cc in cols:
            cells.append((int(rr), int(cc)))
    if len(set(cells)) != 5:
        return None
    return cells


def _max_steps_to_edge(tip: tuple[int, int], direction: str, h: int, w: int) -> int:
    r, c = map(int, tip)
    if direction == "right":
        return int(w - 1 - c)
    if direction == "left":
        return int(c)
    if direction == "down":
        return int(h - 1 - r)
    if direction == "up":
        return int(r)
    raise ValueError(direction)


def _ray_cells(tip: tuple[int, int], direction: str, steps: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = map(int, tip)
    return [(int(r + dr * step), int(c + dc * step)) for step in range(1, int(steps) + 1)]


def _wall_cells(
    hit: tuple[int, int], direction: str, length: int, offset: int
) -> list[tuple[int, int]]:
    r, c = map(int, hit)
    cells: list[tuple[int, int]] = []
    if direction in ("left", "right"):
        top = int(r - offset)
        for dr in range(length):
            cells.append((int(top + dr), int(c)))
    else:
        left = int(c - offset)
        for dc in range(length):
            cells.append((int(r), int(left + dc)))
    return cells


def _choose_lens_steps(limit: int, lens_count: int, terminal_kind: str) -> list[int] | None:
    if lens_count == 0:
        return []
    if terminal_kind == "wall":
        candidates = list(range(2, int(limit - 1)))
    else:
        candidates = list(range(2, int(limit)))
    if len(candidates) < lens_count:
        return None
    for _ in range(60):
        steps = sorted(random.sample(candidates, lens_count))
        if all(int(b - a) >= 2 for a, b in zip(steps, steps[1:])):
            return [int(step) for step in steps]
    return None


def _valid_plan(
    h: int,
    w: int,
    arrow_cells: list[tuple[int, int]],
    wall_cells: list[tuple[int, int]],
    lens_cells: list[tuple[int, int]],
    path_cells: list[tuple[int, int]],
    solid_cells: set[tuple[int, int]],
    path_reserved: set[tuple[int, int]],
) -> bool:
    candidate_structure = set((int(r), int(c)) for r, c in arrow_cells + wall_cells)
    candidate_lenses = set((int(r), int(c)) for r, c in lens_cells)
    candidate_solid = candidate_structure | candidate_lenses
    candidate_path = set((int(r), int(c)) for r, c in path_cells)

    if not all(_in_bounds(cell, h, w) for cell in candidate_solid):
        return False
    if len(candidate_solid) != len(arrow_cells) + len(set(wall_cells)) + len(set(lens_cells)):
        return False
    if candidate_solid & solid_cells:
        return False
    if candidate_solid & path_reserved:
        return False
    if candidate_path & solid_cells:
        return False
    if candidate_path & path_reserved:
        return False
    for cell in candidate_structure:
        if _touches_any(cell, solid_cells):
            return False
    seen_lenses: set[tuple[int, int]] = set()
    for cell in candidate_lenses:
        if _touches_any(cell, solid_cells):
            return False
        if _touches_any(cell, candidate_structure):
            return False
        if _touches_any(cell, seen_lenses):
            return False
        seen_lenses.add(cell)
    return True


def _sample_ray_plan(
    h: int,
    w: int,
    solid_cells: set[tuple[int, int]],
    path_reserved: set[tuple[int, int]],
    color: int,
    lens_count: int,
    terminal_kind: str,
) -> dict | None:
    for _ in range(240):
        direction = random.choice(tuple(DIRS))
        shift = random.choice((0, -1))
        tip = (random.randrange(h), random.randrange(w))
        arrow_cells = _arrow_cells_from_tip(tip, direction, shift)
        if arrow_cells is None or not all(_in_bounds(cell, h, w) for cell in arrow_cells):
            continue

        max_steps = _max_steps_to_edge(tip, direction, h, w)
        if terminal_kind == "boundary":
            if max_steps < max(3, 2 * lens_count + 1):
                continue
            steps = max_steps
            wall = []
        else:
            wall_len = random.randint(3, 5)
            wall_offset = random.randrange(wall_len)
            min_steps = max(3, 2 * lens_count + 2)
            if max_steps < min_steps:
                continue
            steps = random.randint(min_steps, max_steps)
            hit = _ray_cells(tip, direction, steps)[-1]
            wall = _wall_cells(hit, direction, wall_len, wall_offset)
            if not all(_in_bounds(cell, h, w) for cell in wall):
                continue

        lens_steps = _choose_lens_steps(steps, lens_count, terminal_kind)
        if lens_steps is None:
            continue
        path_cells = _ray_cells(tip, direction, steps)
        lens_cells = [path_cells[int(step - 1)] for step in lens_steps]

        if not _valid_plan(h, w, arrow_cells, wall, lens_cells, path_cells, solid_cells, path_reserved):
            continue

        return {
            "color": int(color),
            "direction": direction,
            "tip": (int(tip[0]), int(tip[1])),
            "arrow_cells": [(int(r), int(c)) for r, c in arrow_cells],
            "wall_cells": [(int(r), int(c)) for r, c in wall],
            "lens_cells": [(int(r), int(c)) for r, c in lens_cells],
            "path_cells": [(int(r), int(c)) for r, c in path_cells],
        }
    return None


def _place_extra_wall(
    h: int,
    w: int,
    solid_cells: set[tuple[int, int]],
    path_reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    for _ in range(200):
        length = random.randint(3, 5)
        if random.random() < 0.5:
            r = random.randrange(h)
            c0 = random.randint(0, w - length)
            cells = [(int(r), int(c0 + dc)) for dc in range(length)]
        else:
            c = random.randrange(w)
            r0 = random.randint(0, h - length)
            cells = [(int(r0 + dr), int(c)) for dr in range(length)]
        cell_set = set(cells)
        if cell_set & solid_cells:
            continue
        if cell_set & path_reserved:
            continue
        if any(_touches_any(cell, solid_cells) for cell in cells):
            continue
        return cells
    return None


def _place_extra_lens(
    h: int,
    w: int,
    solid_cells: set[tuple[int, int]],
    path_reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for _ in range(200):
        cell = (random.randrange(h), random.randrange(w))
        if cell in solid_cells or cell in path_reserved:
            continue
        if _touches_any(cell, solid_cells):
            continue
        return (int(cell[0]), int(cell[1]))
    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(15, 24)
    w = random.randint(15, 24)
    bg = random.choice(PALETTE)

    arrow_count = random.randint(2, 4)
    arrow_colors = random.sample([color for color in PALETTE if int(color) != int(bg)], arrow_count)
    lens_counts = [random.randint(0, 2) for _ in range(arrow_count)]
    if sum(lens_counts) == 0:
        lens_counts[random.randrange(arrow_count)] = random.randint(1, 2)
    terminal_kinds = [random.choice(("wall", "boundary")) for _ in range(arrow_count)]

    solid_cells: set[tuple[int, int]] = set()
    path_reserved: set[tuple[int, int]] = set()
    plans: list[dict] = []

    for color, lens_count, terminal_kind in zip(arrow_colors, lens_counts, terminal_kinds):
        plan = _sample_ray_plan(h, w, solid_cells, path_reserved, int(color), int(lens_count), terminal_kind)
        if plan is None:
            return None
        plans.append(plan)
        for cell in plan["arrow_cells"] + plan["wall_cells"] + plan["lens_cells"]:
            solid_cells.add((int(cell[0]), int(cell[1])))
        for cell in plan["path_cells"]:
            path_reserved.add((int(cell[0]), int(cell[1])))

    wall_segments = sum(1 for plan in plans if plan["wall_cells"])
    extra_wall_target = random.randint(0 if wall_segments else 1, 2 if wall_segments else 3)
    extra_walls: list[list[tuple[int, int]]] = []
    for _ in range(extra_wall_target):
        cells = _place_extra_wall(h, w, solid_cells, path_reserved)
        if cells is None:
            break
        extra_walls.append(cells)
        for cell in cells:
            solid_cells.add((int(cell[0]), int(cell[1])))

    extra_lens_target = random.randint(0, 2)
    extra_lenses: list[tuple[int, int]] = []
    for _ in range(extra_lens_target):
        cell = _place_extra_lens(h, w, solid_cells, path_reserved)
        if cell is None:
            break
        extra_lenses.append(cell)
        solid_cells.add((int(cell[0]), int(cell[1])))

    grid = np.full((h, w), int(bg), dtype=int)
    for plan in plans:
        for r, c in plan["arrow_cells"]:
            grid[int(r), int(c)] = int(plan["color"])
        for r, c in plan["wall_cells"]:
            grid[int(r), int(c)] = GREY
        for r, c in plan["lens_cells"]:
            grid[int(r), int(c)] = YELLOW
    for wall in extra_walls:
        for r, c in wall:
            grid[int(r), int(c)] = GREY
    for r, c in extra_lenses:
        grid[int(r), int(c)] = YELLOW

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if Counter(map(int, grid.reshape(-1))).most_common(1)[0][0] != int(bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
