from __future__ import annotations

import random

import numpy as np


ORANGE = 7
PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)
RAY_NEW_LEN = 4  # Total ray length is 5 including the tip pixel already in the sprite.
GENERATE_ATTEMPTS = 400
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)
ARROW_COUNT_RANGE = (3, 6)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
STOP_TYPES = ("free", "boundary", "obstacle")


_PATTERN_DATA = {
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}): ((0, 1), UP),
    frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}): ((2, 1), DOWN),
    frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}): ((1, 0), LEFT),
    frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}): ((1, 2), RIGHT),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    out: list[dict[str, object]] = []

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        mask = grid == int(color)
        seen = np.zeros((h, w), dtype=bool)
        for r0, c0 in np.argwhere(mask):
            r0 = int(r0)
            c0 = int(c0)
            if seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = int(r + dr)
                        cc = int(c + dc)
                        if (
                            0 <= rr < h
                            and 0 <= cc < w
                            and not seen[rr, cc]
                            and bool(mask[rr, cc])
                        ):
                            seen[rr, cc] = True
                            stack.append((rr, cc))

            out.append(
                {
                    "color": int(color),
                    "coords": np.array(coords, dtype=int),
                }
            )

    return out


def _parse_arrow(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    assert coords.shape == (5, 2), "each arrow must contain exactly five pixels"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "each arrow must fit in a 3x3 box"

    rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    assert rel in _PATTERN_DATA, "foreground objects must be one of the four arrow rotations"
    tip_rel, direction = _PATTERN_DATA[rel]
    tip = (int(r0 + tip_rel[0]), int(c0 + tip_rel[1]))
    return tip, direction


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = map(int, tip)
    dr, dc = map(int, direction)
    if (dr, dc) == UP:
        return {(r, c), (r, c - 1), (r, c + 1), (r + 1, c), (r + 2, c)}
    if (dr, dc) == DOWN:
        return {(r, c), (r, c - 1), (r, c + 1), (r - 1, c), (r - 2, c)}
    if (dr, dc) == LEFT:
        return {(r, c), (r - 1, c), (r + 1, c), (r, c + 1), (r, c + 2)}
    assert (dr, dc) == RIGHT
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c - 2)}


def _ray_outcome(
    tip: tuple[int, int],
    direction: tuple[int, int],
    occupied: np.ndarray,
) -> tuple[str, list[tuple[int, int]]]:
    h, w = occupied.shape
    tr, tc = map(int, tip)
    dr, dc = map(int, direction)
    cells: list[tuple[int, int]] = []

    for step in range(1, RAY_NEW_LEN + 1):
        rr = int(tr + dr * step)
        cc = int(tc + dc * step)
        if not (0 <= rr < h and 0 <= cc < w):
            return "boundary", cells
        if bool(occupied[rr, cc]):
            return "obstacle", cells
        cells.append((rr, cc))

    return "free", cells


def _candidate_tips(h: int, w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == UP:
        return [(r, c) for r in range(0, h - 2) for c in range(1, w - 1)]
    if direction == DOWN:
        return [(r, c) for r in range(2, h) for c in range(1, w - 1)]
    if direction == LEFT:
        return [(r, c) for r in range(1, h - 1) for c in range(0, w - 2)]
    assert direction == RIGHT
    return [(r, c) for r in range(1, h - 1) for c in range(2, w)]


def _sample_directions(n: int) -> list[tuple[int, int]]:
    if n <= 4:
        dirs = random.sample(list(DIRECTIONS), n)
    else:
        dirs = list(DIRECTIONS) + [random.choice(DIRECTIONS) for _ in range(n - 4)]
    random.shuffle(dirs)
    return dirs


def _sample_targets(n: int) -> list[str]:
    targets = ["free", "boundary", "obstacle"]
    while len(targets) < n:
        targets.append(random.choice(STOP_TYPES))
    random.shuffle(targets)
    return targets


def _priority(stop_type: str) -> int:
    if stop_type == "free":
        return 0
    if stop_type == "boundary":
        return 1
    return 2


def _paint_grid(
    shape: tuple[int, int],
    records: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, ORANGE, dtype=int)
    for record in records:
        color = int(record["color"])
        for r, c in record["cells"]:
            grid[int(r), int(c)] = color
    return grid


def _validate_generated_grid(
    grid: np.ndarray,
    records: list[dict[str, object]],
) -> bool:
    try:
        solved = main(grid)
    except Exception:
        return False

    occupied = grid != ORANGE
    seen_ray_cells: set[tuple[int, int]] = set()
    for record in records:
        tip = record["tip"]
        direction = record["direction"]
        target = str(record["target"])
        planned = list(record["ray_cells"])
        actual_type, actual_cells = _ray_outcome(tip, direction, occupied)
        if actual_type != target or actual_cells != planned:
            return False
        for cell in actual_cells:
            if cell in seen_ray_cells:
                return False
            seen_ray_cells.add(cell)

    if not seen_ray_cells:
        return False
    return bool(np.array_equal(solved, _expected_output(grid)))


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    occupied = g != bg
    out = g.copy()

    for comp in _find_components(g, bg):
        tip, direction = _parse_arrow(comp["coords"])
        color = int(comp["color"])
        for r, c in _ray_outcome(tip, direction, occupied)[1]:
            out[r, c] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_arrows = random.randint(*ARROW_COUNT_RANGE)

    dirs = _sample_directions(n_arrows)
    targets = _sample_targets(n_arrows)
    colors = random.sample(PALETTE, n_arrows)

    specs = [
        {"direction": direction, "target": target, "color": color}
        for direction, target, color in zip(dirs, targets, colors)
    ]
    specs.sort(key=lambda spec: (_priority(str(spec["target"])), random.random()))

    occupied = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)
    records: list[dict[str, object]] = []

    for spec in specs:
        direction = spec["direction"]
        target = str(spec["target"])
        candidates: list[dict[str, object]] = []
        tips = _candidate_tips(h, w, direction)
        random.shuffle(tips)

        for tip in tips:
            cells = _arrow_cells(tip, direction)
            if any(bool(occupied[r, c] or reserved[r, c]) for r, c in cells):
                continue

            outcome, ray_cells = _ray_outcome(tip, direction, occupied)
            if outcome != target:
                continue
            if not ray_cells:
                continue
            if any(bool(reserved[r, c]) for r, c in ray_cells):
                continue

            candidates.append(
                {
                    "tip": tip,
                    "direction": direction,
                    "target": target,
                    "color": int(spec["color"]),
                    "cells": sorted(cells),
                    "ray_cells": list(ray_cells),
                }
            )

        if not candidates:
            return None

        record = random.choice(candidates)
        for r, c in record["cells"]:
            occupied[r, c] = True
        for r, c in record["ray_cells"]:
            reserved[r, c] = True
        records.append(record)

    grid = _paint_grid((h, w), records)
    if not _validate_generated_grid(grid, records):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each arrow sprite and extend a same-color ray in the facing direction.

    Input structure:
    - Background is orange.
    - Each non-background monochrome component is a 5-pixel arrow:
      a 3-pixel shaft plus a 2-pixel fork at one end, forming a 3x3 T-shape.

    Rule:
    - Recover the arrow's direction from which side contains the fork.
    - Starting from the arrow tip, extend a ray of total length 5, so at most
      four new pixels are painted beyond the sprite.
    - The ray stops when the next cell is outside the grid or is already
      non-background in the original input.
    - The arrow itself remains unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    occupied = g != bg
    out = g.copy()

    for comp in _find_components(g, bg):
        tip, direction = _parse_arrow(comp["coords"])
        color = int(comp["color"])
        for r, c in _ray_outcome(tip, direction, occupied)[1]:
            out[r, c] = color

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray puzzle input")
