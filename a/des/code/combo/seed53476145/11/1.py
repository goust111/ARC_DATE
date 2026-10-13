from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
POINTER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_FROM_DELTA = {delta: name for name, delta in DIRS.items()}
POINTER_POS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}

# Every shape lives in a 3x3 bbox and keeps all four side-center cells,
# so any object can point in any cardinal direction.
BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
)

GRID_MIN = 16
GRID_MAX = 30
STEP_MIN = 4
STEP_MAX = 6
GEN_RETRIES = 300


def _dominant_non_bg_color(grid: np.ndarray) -> int:
    vals = grid[grid != BG]
    if vals.size == 0:
        return GREY
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = grid.shape
    comp_of = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == BG or int(comp_of[r, c]) != -1:
                continue
            comp_id = len(comps)
            stack = [(int(r), int(c))]
            comp_of[r, c] = comp_id
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS.values():
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG and int(comp_of[nr, nc]) == -1:
                        comp_of[nr, nc] = comp_id
                        stack.append((nr, nc))
            comps.append(cells)
    return comps, comp_of


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray]:
    grey = _dominant_non_bg_color(grid)
    comps, comp_of = _components(grid)
    sprites: list[dict] = []

    for comp_id, cells in enumerate(comps):
        pointer_cells = [(r, c) for r, c in cells if int(grid[r, c]) != grey]
        if len(pointer_cells) != 1:
            raise ValueError("each sprite must contain exactly one pointer pixel")
        pr, pc = pointer_cells[0]
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        min_r = min(rows)
        max_r = max(rows)
        min_c = min(cols)
        max_c = max(cols)
        sides = []
        if pr == min_r:
            sides.append("up")
        if pr == max_r:
            sides.append("down")
        if pc == min_c:
            sides.append("left")
        if pc == max_c:
            sides.append("right")
        if len(sides) != 1:
            raise ValueError("pointer pixel must lie on exactly one sprite side")

        sprites.append(
            {
                "id": comp_id,
                "cells": cells,
                "body": [(r, c) for r, c in cells if (r, c) != (pr, pc)],
                "pointer": (int(pr), int(pc)),
                "pointer_color": int(grid[pr, pc]),
                "direction": sides[0],
                "scan_key": min((int(c), int(r)) for r, c in cells),
            }
        )

    return grey, sprites, comp_of


def _ray_target(grid: np.ndarray, comp_of: np.ndarray, sprite: dict) -> int | None:
    h, w = grid.shape
    pr, pc = sprite["pointer"]
    dr, dc = DIRS[sprite["direction"]]
    r = int(pr + dr)
    c = int(pc + dc)
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BG:
            return int(comp_of[r, c])
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        return g.copy()
    if not np.any(g != BG):
        return g.copy()

    _grey, sprites, comp_of = _extract_sprites(g)
    sprite_by_id = {sprite["id"]: sprite for sprite in sprites}
    out = g.copy()

    for sprite in sorted(sprites, key=lambda item: item["scan_key"]):
        target_id = _ray_target(g, comp_of, sprite)
        if target_id is None or target_id == int(sprite["id"]):
            continue
        target = sprite_by_id[target_id]
        for r, c in target["body"]:
            out[r, c] = int(sprite["pointer_color"])
        tr, tc = target["pointer"]
        out[tr, tc] = int(g[tr, tc])

    return out


def _rotate_point(point: tuple[int, int]) -> tuple[int, int]:
    r, c = point
    return (int(c), int(-r))


def _flip_point(point: tuple[int, int]) -> tuple[int, int]:
    r, c = point
    return (int(r), int(-c))


def _rotate_dir(name: str) -> str:
    return DIR_FROM_DELTA[_rotate_point(DIRS[name])]


def _flip_dir(name: str) -> str:
    return DIR_FROM_DELTA[_flip_point(DIRS[name])]


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in shape}
    if random.random() < 0.5:
        pts = {(int(r), int(2 - c)) for r, c in pts}
    for _ in range(random.randint(0, 3)):
        pts = {(int(c), int(2 - r)) for r, c in pts}
    return tuple(sorted(pts))


def _line_scenario() -> tuple[list[tuple[int, int]], list[str], str]:
    count = random.randint(4, 5)
    positions = [(0, idx) for idx in range(count)]
    directions = ["right"] * (count - 1) + ["left"]
    return positions, directions, "line"


def _rectangle_scenario() -> tuple[list[tuple[int, int]], list[str], str]:
    span_r = random.randint(1, 2)
    span_c = random.randint(1, 2)
    positions = [(0, 0), (0, span_c), (span_r, span_c), (span_r, 0)]
    if random.random() < 0.5:
        directions = ["right", "down", "left", "up"]
    else:
        directions = ["down", "left", "up", "right"]
    return positions, directions, "cycle"


def _fan_scenario() -> tuple[list[tuple[int, int]], list[str], str]:
    positions = [(0, -1), (0, 0), (0, 1), (-1, 0)]
    directions = ["right", "right", "left", "down"]
    if random.random() < 0.5:
        positions.append((1, 0))
        directions.append("up")
    return positions, directions, "fan"


def _scenario_spec() -> tuple[list[tuple[int, int]], list[str], str]:
    positions, directions, kind = random.choice(
        (_line_scenario, _rectangle_scenario, _fan_scenario)
    )()
    if random.random() < 0.5:
        positions = [_flip_point(pos) for pos in positions]
        directions = [_flip_dir(name) for name in directions]
    for _ in range(random.randint(0, 3)):
        positions = [_rotate_point(pos) for pos in positions]
        directions = [_rotate_dir(name) for name in directions]
    min_r = min(r for r, _c in positions)
    min_c = min(c for _r, c in positions)
    positions = [(int(r - min_r), int(c - min_c)) for r, c in positions]
    return positions, directions, kind


def _place_centers(positions: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_rel_r = max(r for r, _c in positions)
    max_rel_c = max(c for _r, c in positions)
    row_step = random.randint(STEP_MIN, STEP_MAX)
    col_step = random.randint(STEP_MIN, STEP_MAX)
    min_h = int(max_rel_r * row_step + 5)
    min_w = int(max_rel_c * col_step + 5)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    h = random.randint(max(GRID_MIN, min_h), GRID_MAX)
    w = random.randint(max(GRID_MIN, min_w), GRID_MAX)
    origin_r = random.randint(2, h - 3 - max_rel_r * row_step)
    origin_c = random.randint(2, w - 3 - max_rel_c * col_step)
    centers = [
        (int(origin_r + rel_r * row_step), int(origin_c + rel_c * col_step))
        for rel_r, rel_c in positions
    ]
    return h, w, centers


def _render_scene(
    h: int,
    w: int,
    centers: list[tuple[int, int]],
    directions: list[str],
    colors: list[int],
) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    for center, direction, color in zip(centers, directions, colors):
        shape = _transform_shape(random.choice(BASE_SHAPES))
        cr, cc = center
        top = int(cr - 1)
        left = int(cc - 1)
        pointer = POINTER_POS[direction]
        if pointer not in shape:
            return None
        for dr, dc in shape:
            rr = int(top + dr)
            cc2 = int(left + dc)
            if not (0 <= rr < h and 0 <= cc2 < w):
                return None
            if int(grid[rr, cc2]) != BG:
                return None
            grid[rr, cc2] = GREY
        pr = int(top + pointer[0])
        pc = int(left + pointer[1])
        grid[pr, pc] = int(color)
    return grid


def _indegrees(targets: list[int | None], count: int) -> list[int]:
    deg = [0] * count
    for target in targets:
        if target is not None:
            deg[int(target)] += 1
    return deg


def _valid_branch(kind: str, indegrees: list[int]) -> bool:
    if kind == "line":
        return min(indegrees) == 0 and max(indegrees) >= 2
    if kind == "cycle":
        return min(indegrees) == 1 and max(indegrees) == 1
    return max(indegrees) >= 2 and min(indegrees) == 0


def _gen_attempt() -> np.ndarray | None:
    positions, directions, kind = _scenario_spec()
    placed = _place_centers(positions)
    if placed is None:
        return None
    h, w, centers = placed
    colors = random.sample(POINTER_COLORS, len(centers))
    grid = _render_scene(h, w, centers, directions, colors)
    if grid is None:
        return None

    try:
        _grey, sprites, comp_of = _extract_sprites(grid)
    except ValueError:
        return None
    if len(sprites) != len(centers):
        return None

    targets = [_ray_target(grid, comp_of, sprite) for sprite in sprites]
    if any(target is None for target in targets):
        return None
    indegrees = _indegrees(targets, len(sprites))
    if not _valid_branch(kind, indegrees):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
