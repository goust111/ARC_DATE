from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for dr, dc in DIRS4)


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for dr, dc in DIRS8)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            coords_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "area": int(coords_arr.shape[0]),
                }
            )

    return components


def _parse_arrow(grid: np.ndarray, component: dict) -> dict:
    shape = grid.shape
    color = int(component["color"])
    cells = {tuple(map(int, rc)) for rc in component["coords"]}
    assert len(cells) >= 4, "arrow must contain a 3-pixel head plus a tail"

    deg4 = {cell: sum((nbr in cells) for nbr in _neighbors4(*cell)) for cell in cells}
    deg8 = {cell: sum((nbr in cells) for nbr in _neighbors8(*cell)) for cell in cells}

    tip_candidates = [cell for cell in cells if deg4[cell] == 1 and deg8[cell] == 3]
    assert len(tip_candidates) == 1, "expected one unique arrow tip"
    tip = tip_candidates[0]

    tip_neighbors = [nbr for nbr in _neighbors4(*tip) if nbr in cells]
    assert len(tip_neighbors) == 1, "arrow tip must touch the body at one cell"
    tail_base = tip_neighbors[0]

    direction = (tip[0] - tail_base[0], tip[1] - tail_base[1])
    assert abs(direction[0]) + abs(direction[1]) == 1, "arrow direction must be cardinal"
    back = (-direction[0], -direction[1])
    perp = (direction[1], -direction[0])

    leg_a = (tip[0] + back[0] + perp[0], tip[1] + back[1] + perp[1])
    leg_b = (tip[0] + back[0] - perp[0], tip[1] + back[1] - perp[1])
    assert leg_a in cells and leg_b in cells, "arrow head must be a 3-pixel triangle"

    tail_len = 0
    while True:
        k = tail_len + 1
        tail_cell = (tip[0] + back[0] * k, tip[1] + back[1] * k)
        if tail_cell not in cells:
            break
        tail_len += 1
    assert tail_len >= 1, "arrow tail must have positive length"

    expected = {tip, leg_a, leg_b}
    for k in range(1, tail_len + 1):
        expected.add((tip[0] + back[0] * k, tip[1] + back[1] * k))
    assert expected == cells, "component must match the arrow geometry exactly"

    marker = (tip[0] + back[0] * (tail_len + 1), tip[1] + back[1] * (tail_len + 1))
    assert _in_bounds(shape, *marker), "marker must stay in bounds"
    marker_color = int(grid[marker])
    assert marker_color not in (BG, color), "marker must be a different non-background color"
    assert all(
        not _in_bounds(shape, nr, nc) or int(grid[nr, nc]) != marker_color
        for nr, nc in _neighbors4(*marker)
    ), "marker must be a single-pixel component"

    return {
        "tip": tip,
        "direction": direction,
        "size": int(component["area"]),
        "marker_color": marker_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    arrows = []
    for component in _monochrome_components(g):
        if int(component["area"]) == 1:
            continue
        arrows.append(_parse_arrow(g, component))

    assert arrows, "expected at least one arrow"

    arrows.sort(key=lambda arrow: (arrow["tip"][0], arrow["tip"][1]))
    for arrow in arrows:
        tip_r, tip_c = map(int, arrow["tip"])
        dr, dc = map(int, arrow["direction"])
        length = int(arrow["size"])
        color = int(arrow["marker_color"])

        for step in range(1, length + 1):
            rr = tip_r + dr * step
            cc = tip_c + dc * step
            assert _in_bounds(g.shape, rr, cc), "projected band must stay in bounds"
            if int(g[rr, cc]) != BG:
                continue
            assert int(out[rr, cc]) in (BG, color), "bands must not conflict"
            out[rr, cc] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 25)
GRID_W_RANGE = (14, 25)
ARROW_COUNT_CHOICES = (1, 2, 3, 4)
ARROW_COUNT_WEIGHTS = (1, 3, 3, 2)
TAIL_LEN_RANGE = (1, 5)
COLORS = tuple(range(1, 10))
PLACEMENT_TRIES = 200
GEN_ATTEMPTS = 400


def _halo8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS8:
            rr = r + dr
            cc = c + dc
            if _in_bounds(shape, rr, cc):
                out.add((int(rr), int(cc)))
        out.add((int(r), int(c)))
    return out


def _arrow_group(
    shape: tuple[int, int],
    tip: tuple[int, int],
    direction: tuple[int, int],
    tail_len: int,
) -> dict | None:
    tip_r, tip_c = map(int, tip)
    dr, dc = map(int, direction)
    back = (-dr, -dc)
    perp = (dc, -dr)

    head = {
        (tip_r, tip_c),
        (tip_r + back[0] + perp[0], tip_c + back[1] + perp[1]),
        (tip_r + back[0] - perp[0], tip_c + back[1] - perp[1]),
    }
    tail = {
        (tip_r + back[0] * k, tip_c + back[1] * k)
        for k in range(1, int(tail_len) + 1)
    }
    marker = (tip_r + back[0] * (tail_len + 1), tip_c + back[1] * (tail_len + 1))

    arrow_cells = head | tail
    size = len(arrow_cells)
    ray = {
        (tip_r + dr * step, tip_c + dc * step)
        for step in range(1, size + 1)
    }
    all_cells = arrow_cells | {marker} | ray

    if not all(_in_bounds(shape, r, c) for r, c in all_cells):
        return None

    return {
        "arrow_cells": arrow_cells,
        "marker": marker,
        "ray_cells": ray,
        "full_output": arrow_cells | {marker} | ray,
    }


def _paint_instance(
    shape: tuple[int, int],
    placed: list[dict],
    arrow_colors: list[int],
    marker_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for group, body_color, marker_color in zip(placed, arrow_colors, marker_colors):
        for r, c in group["arrow_cells"]:
            grid[r, c] = int(body_color)
        mr, mc = group["marker"]
        grid[mr, mc] = int(marker_color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    count = random.choices(ARROW_COUNT_CHOICES, weights=ARROW_COUNT_WEIGHTS, k=1)[0]
    if 2 * count > len(COLORS):
        return None

    palette = random.sample(list(COLORS), 2 * count)
    arrow_colors = palette[:count]
    marker_colors = palette[count:]

    directions = list(DIRS4)
    random.shuffle(directions)
    directions = directions[:count]

    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []
    for direction in directions:
        success = False
        for _ in range(PLACEMENT_TRIES):
            tail_len = random.randint(*TAIL_LEN_RANGE)
            tip = (random.randrange(h), random.randrange(w))
            group = _arrow_group(shape, tip, direction, tail_len)
            if group is None:
                continue
            if group["full_output"] & blocked:
                continue

            placed.append(group)
            blocked |= _halo8(group["full_output"], shape)
            success = True
            break
        if not success:
            return None

    random.shuffle(arrow_colors)
    random.shuffle(marker_colors)
    return _paint_instance(shape, placed, arrow_colors, marker_colors)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
            if grid is None:
                continue
            out = main(grid)
            if out.shape != grid.shape:
                continue
            if not np.any(out != grid):
                continue
            return grid
        except Exception:
            continue
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
