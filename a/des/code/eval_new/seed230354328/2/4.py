from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
LINE_COUNT_RANGE = (2, 5)
LINE_LENGTH_RANGE = (2, 6)

GEN_ATTEMPTS = 400
SCENE_ATTEMPTS = 200
ADD_ATTEMPTS = 240


def _in_bounds(h: int, w: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _line_cells(seed: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(seed[0] + dr * step, seed[1] + dc * step) for step in range(1, length + 1)]


def _grey_neighbors(grid: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    r, c = seed
    hits: list[tuple[int, int, int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == GREY:
            hits.append((rr, cc, dr, dc))
    return hits


def _decode_instruction(grid: np.ndarray, seed: tuple[int, int]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    hits = _grey_neighbors(grid, seed)
    assert len(hits) == 1, "each seed must touch exactly one grey endpoint"

    rr, cc, dr, dc = hits[0]
    cells: list[tuple[int, int]] = []
    h, w = grid.shape
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == GREY:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return (dr, dc), cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored line segments from seed pixels using adjacent grey template segments, mixing overlaps into maroon.

    Input structure:
    - bg = black.
    - Foreground objects are single-pixel colored seeds and grey straight line templates.
    - Each seed is 4-neighbor adjacent to exactly one endpoint of a horizontal or vertical grey segment.
    - Grey templates may cross one another, but seeds do not overlap grey pixels.

    Rule:
    - Read each seed together with its touching grey segment as one growth instruction.
    - The touching grey pixel gives the growth direction, and the full straight grey run gives the growth length.
    - Remove all grey template pixels, keep the seeds, then repaint each decoded run using the seed color.
    - If multiple repainted runs claim the same cell, that cell becomes maroon and stays maroon.

    Key insight:
    - The grey segment is not part of the final object; it is a hidden instruction whose direction and length are transferred to its adjacent seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    seeds = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != BG) & (g != GREY))
    ]
    assert seeds, "expected at least one colored seed"

    out = g.copy()
    out[out == GREY] = BG

    for r, c, color in sorted(seeds):
        (_dr, _dc), cells = _decode_instruction(g, (r, c))
        for rr, cc in cells:
            current = int(out[rr, cc])
            if current == BG:
                out[rr, cc] = color
            elif current != MAROON:
                out[rr, cc] = MAROON

    return out


def _make_instruction(
    h: int,
    w: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    color: int,
) -> dict[str, object] | None:
    if not _in_bounds(h, w, seed):
        return None
    cells = _line_cells(seed, direction, length)
    if any(not _in_bounds(h, w, cell) for cell in cells):
        return None
    return {
        "seed": seed,
        "dir": direction,
        "length": length,
        "color": color,
        "cells": cells,
    }


def _render_input(h: int, w: int, instructions: list[dict[str, object]]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)

    for inst in instructions:
        for rr, cc in inst["cells"]:
            grid[rr, cc] = GREY

    for inst in instructions:
        rr, cc = inst["seed"]
        if int(grid[rr, cc]) != BG:
            return None
        grid[rr, cc] = int(inst["color"])

    return grid


def _scene_is_valid(h: int, w: int, instructions: list[dict[str, object]], expect_collision: bool) -> bool:
    if not instructions:
        return False

    seed_positions: set[tuple[int, int]] = set()
    cell_owners: dict[tuple[int, int], list[int]] = {}

    for idx, inst in enumerate(instructions):
        seed = inst["seed"]
        if seed in seed_positions:
            return False
        seed_positions.add(seed)

        for cell in inst["cells"]:
            if cell in seed_positions:
                return False
            cell_owners.setdefault(cell, []).append(idx)

    if any(cell in seed_positions for cell in cell_owners):
        return False

    seeds = list(seed_positions)
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            if abs(seed_a[0] - seed_b[0]) + abs(seed_a[1] - seed_b[1]) == 1:
                return False

    overlap_cells = [cell for cell, owners in cell_owners.items() if len(owners) > 1]
    if bool(overlap_cells) != expect_collision:
        return False

    for cell in overlap_cells:
        owners = cell_owners[cell]
        if len(owners) != 2:
            return False
        dirs = [instructions[idx]["dir"] for idx in owners]
        if dirs[0][0] == dirs[1][0]:
            return False

    grid = _render_input(h, w, instructions)
    if grid is None:
        return False

    traced_cells: set[tuple[int, int]] = set()
    for inst in instructions:
        seed = inst["seed"]
        try:
            direction, cells = _decode_instruction(grid, seed)
        except AssertionError:
            return False
        if direction != inst["dir"]:
            return False
        if cells != inst["cells"]:
            return False
        traced_cells.update(cells)

    if traced_cells != set(cell_owners):
        return False

    if len(np.argwhere((grid != BG) & (grid != GREY))) != len(instructions):
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    has_collision = bool(np.any(out == MAROON))
    if has_collision != expect_collision:
        return False

    return True


def _random_instruction(h: int, w: int, color: int) -> dict[str, object] | None:
    direction = random.choice(DIRS4)
    length = random.randint(*LINE_LENGTH_RANGE)
    dr, dc = direction

    if dr == 1:
        row_lo, row_hi = 0, h - 1 - length
    elif dr == -1:
        row_lo, row_hi = length, h - 1
    else:
        row_lo, row_hi = 0, h - 1

    if dc == 1:
        col_lo, col_hi = 0, w - 1 - length
    elif dc == -1:
        col_lo, col_hi = length, w - 1
    else:
        col_lo, col_hi = 0, w - 1

    if row_lo > row_hi or col_lo > col_hi:
        return None

    seed = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
    return _make_instruction(h, w, seed, direction, length, color)


def _cross_pair(h: int, w: int, colors: tuple[int, int]) -> list[dict[str, object]] | None:
    horiz_dirs = [(0, -1), (0, 1)]
    vert_dirs = [(-1, 0), (1, 0)]

    for _ in range(SCENE_ATTEMPTS):
        cross_r = random.randrange(h)
        cross_c = random.randrange(w)

        h_dir = random.choice(horiz_dirs)
        h_len = random.randint(3, LINE_LENGTH_RANGE[1])
        h_step = random.randint(2, h_len - 1)
        h_seed = (cross_r - h_dir[0] * h_step, cross_c - h_dir[1] * h_step)
        inst_h = _make_instruction(h, w, h_seed, h_dir, h_len, colors[0])
        if inst_h is None:
            continue

        v_dir = random.choice(vert_dirs)
        v_len = random.randint(3, LINE_LENGTH_RANGE[1])
        v_step = random.randint(2, v_len - 1)
        v_seed = (cross_r - v_dir[0] * v_step, cross_c - v_dir[1] * v_step)
        inst_v = _make_instruction(h, w, v_seed, v_dir, v_len, colors[1])
        if inst_v is None:
            continue

        h_cells = set(inst_h["cells"])
        v_cells = set(inst_v["cells"])
        if h_cells & v_cells != {(cross_r, cross_c)}:
            continue
        if inst_h["seed"] == inst_v["seed"]:
            continue
        if inst_h["seed"] in v_cells or inst_v["seed"] in h_cells:
            continue
        if _scene_is_valid(h, w, [inst_h, inst_v], expect_collision=True):
            return [inst_h, inst_v]

    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    line_count = random.randint(*LINE_COUNT_RANGE)
    expect_collision = random.choice([False, True])

    colors = random.sample(SEED_COLORS, line_count)
    instructions: list[dict[str, object]] = []
    color_index = 0

    if expect_collision:
        pair = _cross_pair(h, w, (colors[0], colors[1]))
        if pair is None:
            return None
        instructions.extend(pair)
        color_index = 2

    while color_index < line_count:
        placed = False
        for _ in range(ADD_ATTEMPTS):
            inst = _random_instruction(h, w, colors[color_index])
            if inst is None:
                continue
            candidate = instructions + [inst]
            if _scene_is_valid(h, w, candidate, expect_collision=expect_collision):
                instructions = candidate
                placed = True
                break
        if not placed:
            return None
        color_index += 1

    grid = _render_input(h, w, instructions)
    if grid is None:
        return None
    if not _scene_is_valid(h, w, instructions, expect_collision=expect_collision):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-template growth puzzle")
