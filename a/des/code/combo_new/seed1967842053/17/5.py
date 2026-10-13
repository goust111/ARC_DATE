from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _find_wall(grid: np.ndarray):
    rows, cols = grid.shape

    horizontal = []
    for r in range(rows - 2):
        if not (np.all(grid[r, :] == GREY) and np.all(grid[r + 2, :] == GREY)):
            continue
        mid = grid[r + 1, :]
        holes = np.flatnonzero((mid != GREY) & (mid != BLACK))
        if holes.size and np.all(mid != BLACK):
            horizontal.append((holes.size, r, holes))

    vertical = []
    for c in range(cols - 2):
        if not (np.all(grid[:, c] == GREY) and np.all(grid[:, c + 2] == GREY)):
            continue
        mid = grid[:, c + 1]
        holes = np.flatnonzero((mid != GREY) & (mid != BLACK))
        if holes.size and np.all(mid != BLACK):
            vertical.append((holes.size, c, holes))

    if horizontal and (not vertical or horizontal[0][0] >= vertical[0][0]):
        _, r, holes = horizontal[0]
        return "horizontal", r, holes
    if vertical:
        _, c, holes = vertical[0]
        return "vertical", c, holes
    return None


def _paint_until_blocked(out: np.ndarray, cells, color: int) -> None:
    for r, c in cells:
        if out[r, c] != BLACK:
            break
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each path through the grey filter wall and recolor the exit side by the aligned hole.

    Input structure:
    A black grid contains one full-width horizontal or full-height vertical grey wall that is three
    pixels thick.  The middle line of the wall contains colored one-pixel holes, and one-pixel-wide
    colored paths approach those holes orthogonally from the grid edge on either side of the wall.

    Rule:
    Keep the wall, hole pixels, and entrance-side path pixels unchanged.  For every path touching a
    wall surface at the same row or column as a hole, continue that path straight through to the
    opposite side of the wall.  Paint only the newly created exit-side segment with the hole color,
    stopping before the grid edge or before the first non-black cell already present.

    Key insight:
    The path's original color identifies the incoming strand, but the aligned hole is the visible
    filter: its color determines the color of the path after it leaves the wall.
    """
    out = np.array(grid, copy=True)
    found = _find_wall(grid)
    if found is None:
        return out

    rows, cols = grid.shape
    orientation, start, holes = found

    if orientation == "horizontal":
        wall_top = start
        wall_bottom = start + 2
        for c in holes:
            color = int(grid[wall_top + 1, c])
            if wall_top > 0 and grid[wall_top - 1, c] not in (BLACK, GREY):
                _paint_until_blocked(out, ((r, c) for r in range(wall_bottom + 1, rows)), color)
            if wall_bottom + 1 < rows and grid[wall_bottom + 1, c] not in (BLACK, GREY):
                _paint_until_blocked(out, ((r, c) for r in range(wall_top - 1, -1, -1)), color)
    else:
        wall_left = start
        wall_right = start + 2
        for r in holes:
            color = int(grid[r, wall_left + 1])
            if wall_left > 0 and grid[r, wall_left - 1] not in (BLACK, GREY):
                _paint_until_blocked(out, ((r, c) for c in range(wall_right + 1, cols)), color)
            if wall_right + 1 < cols and grid[r, wall_right + 1] not in (BLACK, GREY):
                _paint_until_blocked(out, ((r, c) for c in range(wall_left - 1, -1, -1)), color)

    return out


def _spaced_positions(length: int, count: int) -> list[int]:
    candidates = list(range(1, length - 1))
    for _ in range(200):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:])):
            return picks
    return sorted(random.sample(candidates, count))


def _side_plan(hole_count: int, paired: bool) -> list[list[int]]:
    if paired:
        pair_at = random.randrange(hole_count)
        plan: list[list[int]] = []
        for i in range(hole_count):
            if i == pair_at:
                plan.append([-1, 1])
            else:
                plan.append([random.choice([-1, 1])])
        return plan

    sides = [-1, 1] + [random.choice([-1, 1]) for _ in range(hole_count - 2)]
    random.shuffle(sides)
    return [[side] for side in sides]


def generate_input() -> np.ndarray:
    orientation = random.choice(["horizontal", "vertical"])
    paired = random.choice([False, True])
    hole_count = random.randint(2, 3 if paired else 4)

    path_count = hole_count + (1 if paired else 0)
    colors = random.sample(PALETTE, hole_count + path_count)
    hole_colors = colors[:hole_count]
    path_colors = colors[hole_count:]
    path_color_i = 0
    side_plan = _side_plan(hole_count, paired)

    if orientation == "horizontal":
        rows = random.randint(11, 30)
        cols = random.randint(12, 30)
        wall_top = rows // 2 - 1
        positions = _spaced_positions(cols, hole_count)
        grid = np.zeros((rows, cols), dtype=int)
        grid[wall_top:wall_top + 3, :] = GREY
        for c, hole_color, sides in zip(positions, hole_colors, side_plan):
            grid[wall_top + 1, c] = hole_color
            for side in sides:
                path_color = path_colors[path_color_i]
                path_color_i += 1
                if side < 0:
                    grid[:wall_top, c] = path_color
                else:
                    grid[wall_top + 3:, c] = path_color
        return grid

    rows = random.randint(12, 30)
    cols = random.randint(11, 30)
    wall_left = cols // 2 - 1
    positions = _spaced_positions(rows, hole_count)
    grid = np.zeros((rows, cols), dtype=int)
    grid[:, wall_left:wall_left + 3] = GREY
    for r, hole_color, sides in zip(positions, hole_colors, side_plan):
        grid[r, wall_left + 1] = hole_color
        for side in sides:
            path_color = path_colors[path_color_i]
            path_color_i += 1
            if side < 0:
                grid[r, :wall_left] = path_color
            else:
                grid[r, wall_left + 3:] = path_color
    return grid
