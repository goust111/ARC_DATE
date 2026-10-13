from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400

DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_prisms(grid: np.ndarray) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    prisms: dict[tuple[int, int], tuple[int, int]] = {}
    prism_cells: dict[tuple[int, int], tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    for r in range(h - 1):
        for c in range(w - 1):
            cells = ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
            if any(cell in used for cell in cells):
                continue
            a = int(grid[r, c])
            b = int(grid[r, c + 1])
            d = int(grid[r + 1, c + 1])
            e = int(grid[r + 1, c])
            if a == BG or b == BG or d == BG or e == BG:
                continue
            if a != d or b != e or a == b:
                continue
            prisms[(r, c)] = (a, b)
            for cell in cells:
                used.add(cell)
                prism_cells[cell] = (r, c)
    return prisms, prism_cells


def _find_towers(grid: np.ndarray, prism_cells: dict[tuple[int, int], tuple[int, int]]) -> list[dict[str, object]]:
    h, w = grid.shape
    towers: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        mask = (grid == color).copy()
        for r, c in prism_cells:
            mask[r, c] = False
        for comp in _components(mask):
            if len(comp) != 2:
                continue
            (r0, c0), (r1, c1) = sorted(comp)
            start: tuple[int, int] | None = None
            direction: tuple[int, int] | None = None
            if r0 == r1 and abs(c1 - c0) == 1:
                left, right = sorted((c0, c1))
                if left == 0 and right == 1:
                    start = (r0, 2)
                    direction = DIR_RIGHT
                elif left == w - 2 and right == w - 1:
                    start = (r0, w - 3)
                    direction = DIR_LEFT
            elif c0 == c1 and abs(r1 - r0) == 1:
                top, bottom = sorted((r0, r1))
                if top == 0 and bottom == 1:
                    start = (2, c0)
                    direction = DIR_DOWN
                elif top == h - 2 and bottom == h - 1:
                    start = (h - 3, c0)
                    direction = DIR_UP
            if start is None or direction is None:
                continue
            towers.append(
                {
                    "color": color,
                    "cells": tuple(comp),
                    "start": start,
                    "dir": direction,
                }
            )
    return towers


def _find_bulbs(
    grid: np.ndarray,
    prism_cells: dict[tuple[int, int], tuple[int, int]],
    tower_cells: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    bulbs: dict[tuple[int, int], int] = {}
    blocked = set(prism_cells) | tower_cells
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        mask = (grid == color).copy()
        for r, c in blocked:
            mask[r, c] = False
        for comp in _components(mask):
            assert len(comp) == 1, "non-prism, non-tower objects must be singleton bulbs"
            bulbs[comp[0]] = color
    return bulbs


def _split_from_prism(
    grid: np.ndarray,
    anchor: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int, int, int, int]]:
    r, c = anchor
    a = int(grid[r, c])
    b = int(grid[r, c + 1])
    if direction == DIR_DOWN:
        return [
            (r, c - 1, 0, -1, a),
            (r, c + 2, 0, 1, b),
        ]
    if direction == DIR_UP:
        return [
            (r + 1, c - 1, 0, -1, b),
            (r + 1, c + 2, 0, 1, a),
        ]
    if direction == DIR_RIGHT:
        return [
            (r - 1, c, -1, 0, a),
            (r + 2, c, 1, 0, b),
        ]
    assert direction == DIR_LEFT
    return [
        (r - 1, c + 1, -1, 0, b),
        (r + 2, c + 1, 1, 0, a),
    ]


def _simulate_scene(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    prisms, prism_cells = _find_prisms(g)
    towers = _find_towers(g, prism_cells)
    tower_cells = {cell for tower in towers for cell in tower["cells"]}  # type: ignore[index]
    bulbs = _find_bulbs(g, prism_cells, tower_cells)

    lit: set[tuple[int, int]] = set()
    seen_states: set[tuple[int, int, int, int, int]] = set()
    queue = deque(
        (
            int(tower["start"][0]),
            int(tower["start"][1]),
            int(tower["dir"][0]),
            int(tower["dir"][1]),
            int(tower["color"]),
        )
        for tower in towers
    )

    while queue:
        r, c, dr, dc, color = queue.popleft()
        while 0 <= r < h and 0 <= c < w:
            state = (r, c, dr, dc, color)
            if state in seen_states:
                break
            seen_states.add(state)

            if (r, c) in prism_cells:
                for nr, nc, ndr, ndc, ncolor in _split_from_prism(g, prism_cells[(r, c)], (dr, dc)):
                    if 0 <= nr < h and 0 <= nc < w:
                        queue.append((nr, nc, ndr, ndc, ncolor))
                break

            if (r, c) in tower_cells:
                break

            bulb_color = bulbs.get((r, c))
            if bulb_color is not None:
                if bulb_color == color:
                    lit.add((r, c))
                    break
                r += dr
                c += dc
                continue

            if int(g[r, c]) == BG:
                out[r, c] = color
                r += dr
                c += dc
                continue

            break

    for cell in bulbs:
        if cell not in lit:
            out[cell] = BG

    if not return_meta:
        return out
    return out, {
        "prisms": prisms,
        "towers": towers,
        "bulbs": bulbs,
        "lit": lit,
    }


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate_scene(grid)


def _place_prism(grid: np.ndarray, row: int, col: int, color_a: int, color_b: int) -> None:
    assert np.all(grid[row : row + 2, col : col + 2] == BG)
    grid[row, col] = color_a
    grid[row, col + 1] = color_b
    grid[row + 1, col] = color_b
    grid[row + 1, col + 1] = color_a


def _place_top_tower(grid: np.ndarray, col: int, color: int) -> None:
    assert grid[0, col] == BG and grid[1, col] == BG
    grid[0, col] = color
    grid[1, col] = color


def _place_bottom_tower(grid: np.ndarray, col: int, color: int) -> None:
    assert grid[-2, col] == BG and grid[-1, col] == BG
    grid[-2, col] = color
    grid[-1, col] = color


def _place_bulb(grid: np.ndarray, row: int, col: int, color: int) -> None:
    assert grid[row, col] == BG
    grid[row, col] = color


def _build_single_prism() -> tuple[np.ndarray, dict[str, int]] | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    grid = np.zeros((h, w), dtype=int)

    add_second_tower = random.random() < 0.55
    chosen = random.sample(ALL_COLORS, 4 if add_second_tower else 3)
    tower_color, left_color, right_color = chosen[:3]
    second_tower_color = chosen[3] if add_second_tower else None
    dead_colors = [color for color in ALL_COLORS if color not in chosen]
    random.shuffle(dead_colors)

    prism_row = random.randint(4, h - 6)
    prism_col = random.randint(4, w - 6)
    tower_col = prism_col + random.choice((0, 1))

    _place_top_tower(grid, tower_col, tower_color)
    _place_prism(grid, prism_row, prism_col, left_color, right_color)

    bulbs = 0
    matching_side = random.choice(("left", "right"))
    add_pass_through = bool(dead_colors) and random.random() < 0.5

    if matching_side == "left":
        match_col = random.randint(1, prism_col - 2)
        _place_bulb(grid, prism_row, match_col, left_color)
        bulbs += 1
        if add_pass_through and match_col + 1 <= prism_col - 2:
            dead_col = random.randint(match_col + 1, prism_col - 2)
            _place_bulb(grid, prism_row, dead_col, dead_colors.pop())
            bulbs += 1
        dead_col = random.randint(prism_col + 3, w - 2)
        _place_bulb(grid, prism_row, dead_col, dead_colors.pop())
        bulbs += 1
    else:
        match_col = random.randint(prism_col + 3, w - 2)
        _place_bulb(grid, prism_row, match_col, right_color)
        bulbs += 1
        if add_pass_through and prism_col + 3 <= match_col - 1:
            dead_col = random.randint(prism_col + 3, match_col - 1)
            _place_bulb(grid, prism_row, dead_col, dead_colors.pop())
            bulbs += 1
        dead_col = random.randint(1, prism_col - 2)
        _place_bulb(grid, prism_row, dead_col, dead_colors.pop())
        bulbs += 1

    towers = 1
    if second_tower_color is not None and prism_row + 3 <= h - 4:
        free_cols = [c for c in range(2, w - 2) if c not in {tower_col, prism_col, prism_col + 1}]
        if free_cols:
            second_col = random.choice(free_cols)
            bulb_row = random.randint(prism_row + 3, h - 4)
            _place_bottom_tower(grid, second_col, second_tower_color)
            _place_bulb(grid, bulb_row, second_col, second_tower_color)
            towers += 1
            bulbs += 1

    return grid, {"prisms": 1, "towers": towers, "bulbs": bulbs}


def _build_double_prism() -> tuple[np.ndarray, dict[str, int]] | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    grid = np.zeros((h, w), dtype=int)

    add_second_tower = random.random() < 0.5
    chosen = random.sample(ALL_COLORS, 6 if add_second_tower else 5)
    tower_color, p1_left, p1_right, p2_left, p2_right = chosen[:5]
    second_tower_color = chosen[5] if add_second_tower else None
    dead_colors = [color for color in ALL_COLORS if color not in chosen]
    random.shuffle(dead_colors)

    prism_row = random.randint(5, h - 7)
    prism1_col = random.randint(max(8, w // 2), w - 6)
    prism2_col = random.randint(3, prism1_col - 5)
    tower_col = prism1_col + random.choice((0, 1))

    _place_top_tower(grid, tower_col, tower_color)
    _place_prism(grid, prism_row, prism1_col, p1_left, p1_right)
    _place_prism(grid, prism_row, prism2_col, p2_left, p2_right)

    bulbs = 0
    if dead_colors and prism2_col + 2 <= prism1_col - 2 and random.random() < 0.5:
        bridge_dead_col = random.randint(prism2_col + 2, prism1_col - 2)
        _place_bulb(grid, prism_row, bridge_dead_col, dead_colors.pop())
        bulbs += 1

    up_matches = random.choice((True, False))
    if up_matches:
        up_row = random.randint(1, prism_row - 2)
        down_row = random.randint(prism_row + 3, h - 2)
        _place_bulb(grid, up_row, prism2_col + 1, p2_right)
        _place_bulb(grid, down_row, prism2_col + 1, dead_colors.pop())
    else:
        up_row = random.randint(1, prism_row - 2)
        down_row = random.randint(prism_row + 3, h - 2)
        _place_bulb(grid, up_row, prism2_col + 1, dead_colors.pop())
        _place_bulb(grid, down_row, prism2_col + 1, p2_left)
    bulbs += 2

    if random.random() < 0.6:
        right_col = random.randint(prism1_col + 3, w - 2)
        _place_bulb(grid, prism_row, right_col, p1_right)
        bulbs += 1

    towers = 1
    if second_tower_color is not None and prism_row + 4 <= h - 4:
        free_cols = [c for c in range(2, w - 2) if c not in {tower_col, prism2_col + 1, prism1_col, prism1_col + 1}]
        if free_cols:
            second_col = random.choice(free_cols)
            bulb_row = random.randint(prism_row + 4, h - 4)
            _place_bottom_tower(grid, second_col, second_tower_color)
            _place_bulb(grid, bulb_row, second_col, second_tower_color)
            towers += 1
            bulbs += 1

    return grid, {"prisms": 2, "towers": towers, "bulbs": bulbs}


def _random_transform(grid: np.ndarray) -> np.ndarray:
    transformed = np.array(grid, copy=True)
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    k = random.randint(0, 3)
    transformed = np.rot90(transformed, k)
    return np.ascontiguousarray(transformed)


def _validate_generated(grid: np.ndarray, expected: dict[str, int]) -> bool:
    try:
        out, meta = _simulate_scene(grid, return_meta=True)
    except AssertionError:
        return False

    if len(meta["prisms"]) != expected["prisms"]:
        return False
    if len(meta["towers"]) != expected["towers"]:
        return False
    if len(meta["bulbs"]) != expected["bulbs"]:
        return False
    if len(set(meta["bulbs"].values())) != len(meta["bulbs"]):
        return False
    if not meta["lit"] or len(meta["lit"]) == len(meta["bulbs"]):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    builders = (_build_single_prism, _build_double_prism)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid_info = random.choice(builders)()
        if grid_info is None:
            continue
        grid, expected = grid_info
        transformed = _random_transform(grid)
        if _validate_generated(transformed, expected):
            return transformed
    raise RuntimeError("failed to generate a valid signal-prism puzzle")
