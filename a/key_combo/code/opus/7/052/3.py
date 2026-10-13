from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
TEAL = 8

GEN_SIDE_OPTIONS = (3, 4)
GEN_FG_MIN = 3
GEN_FG_MAX = 8
GEN_MAX_ATTEMPTS = 300

_GEN_SIDE_BAG: list[int] = []
_GEN_FG_BAGS = {side: [] for side in GEN_SIDE_OPTIONS}


def _build_pinwheel(grid: np.ndarray) -> np.ndarray:
    top = np.hstack((grid, np.rot90(grid, 1)))
    bottom = np.hstack((np.rot90(grid, 2), np.rot90(grid, -1)))
    return np.vstack((top, bottom))


def _ordered_boundary_pairs(side: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    items = []
    for col in range(2 * side):
        anchor = (side - 1, col, 0)
        cells = ((side - 1, col), (side, col))
        items.append((anchor, cells))
    for row in range(2 * side):
        anchor = (row, side - 1, 1)
        cells = ((row, side - 1), (row, side))
        items.append((anchor, cells))

    items.sort(key=lambda item: item[0])
    return [cells for _, cells in items]


def _apply_boundary_dominoes(pinwheel: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    side = pinwheel.shape[0] // 2
    out = pinwheel.copy()
    consumed: set[tuple[int, int]] = set()
    actual_pairs = 0
    same_color_pairs = 0
    red_green_degree: dict[tuple[int, int], int] = {}

    for cell_a, cell_b in _ordered_boundary_pairs(side):
        color_a = int(pinwheel[cell_a])
        color_b = int(pinwheel[cell_b])

        if color_a != BLACK and color_a == color_b:
            same_color_pairs += 1

        if {color_a, color_b} == {RED, GREEN}:
            red_green_degree[cell_a] = red_green_degree.get(cell_a, 0) + 1
            red_green_degree[cell_b] = red_green_degree.get(cell_b, 0) + 1

        if cell_a in consumed or cell_b in consumed:
            continue
        if {color_a, color_b} != {RED, GREEN}:
            continue

        actual_pairs += 1
        consumed.add(cell_a)
        consumed.add(cell_b)

        green_cell = cell_a if color_a == GREEN else cell_b
        red_cell = cell_a if color_a == RED else cell_b
        out[green_cell] = TEAL
        out[red_cell] = BLACK

    stats = {
        "actual_pair_count": actual_pairs,
        "same_color_pair_count": same_color_pairs,
        "max_red_green_degree": max(red_green_degree.values(), default=0),
        "remaining_rg_count": int(np.count_nonzero((out == RED) | (out == GREEN))),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a 2x2 pinwheel from the input sprite, then resolve cross-quadrant red-green boundary dominoes.

    Input structure:
    - The input is a square black-background sprite.
    - Foreground pixels use only red and green, with at least one of each color.
    - The relevant interactions happen only after the sprite is copied into four rotated quadrants.

    Rule:
    - Put the original sprite in the upper-left, its 90-degree counterclockwise rotation in the upper-right,
      its 180-degree rotation in the lower-left, and its 90-degree clockwise rotation in the lower-right.
    - Scan only the horizontal boundary between the two block rows and the vertical boundary between the two
      block columns, in row-major anchor order with the horizontal pair first at the central crossing.
    - Whenever an unchecked boundary pair is one red and one green, recolor the green cell to teal and erase
      the red cell to black; cells already consumed by an earlier pair cannot be reused.

    Key insight:
    - The input colors do not interact directly; the quarter-turn pinwheel creates new cross-quadrant adjacencies,
      and only those boundary contacts trigger the domino recolor-and-erase rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] > 0, "expected a nonempty square grid"

    present = set(map(int, np.unique(g)))
    assert present <= {BLACK, RED, GREEN}, "unexpected input colors"
    assert RED in present and GREEN in present, "expected at least one red pixel and one green pixel"

    pinwheel = _build_pinwheel(g)
    out, _ = _apply_boundary_dominoes(pinwheel)
    return out


def _draw_from_bag(bag: list[int], values: list[int]) -> int:
    if not bag:
        bag.extend(random.sample(values, len(values)))
    return int(bag.pop())


def _sample_side() -> int:
    return _draw_from_bag(_GEN_SIDE_BAG, list(GEN_SIDE_OPTIONS))


def _sample_fg_count(side: int) -> int:
    bag = _GEN_FG_BAGS[side]
    values = list(range(GEN_FG_MIN, min(GEN_FG_MAX, side * side) + 1))
    return _draw_from_bag(bag, values)


def _edge_cells(side: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(side) for c in range(side) if r in (0, side - 1) or c in (0, side - 1)]


def _sample_grid(side: int, fg_count: int) -> np.ndarray:
    cells = [(r, c) for r in range(side) for c in range(side)]
    edge = _edge_cells(side)

    min_edge = 2 if fg_count == 3 else 3
    edge_lo = max(min_edge, fg_count - 2)
    edge_hi = min(fg_count, len(edge))
    if edge_lo > edge_hi:
        edge_lo = edge_hi
    edge_count = random.randint(edge_lo, edge_hi)

    chosen = random.sample(edge, edge_count)
    remainder = fg_count - edge_count
    pool = [cell for cell in cells if cell not in chosen]
    chosen.extend(random.sample(pool, remainder))

    red_count = random.randint(1, fg_count - 1)
    colors = [RED] * red_count + [GREEN] * (fg_count - red_count)
    random.shuffle(colors)

    grid = np.zeros((side, side), dtype=int)
    for cell, color in zip(chosen, colors):
        grid[cell] = int(color)
    return grid


def _accept_grid(grid: np.ndarray) -> bool:
    if np.array_equal(grid, np.rot90(grid, 1)):
        return False

    pinwheel = _build_pinwheel(grid)
    expected, stats = _apply_boundary_dominoes(pinwheel)

    if stats["actual_pair_count"] < 2:
        return False
    if stats["actual_pair_count"] > 5:
        return False
    if stats["same_color_pair_count"] < 1:
        return False
    if stats["max_red_green_degree"] > 1:
        return False
    if stats["remaining_rg_count"] < 2:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        side = _sample_side()
        fg_count = _sample_fg_count(side)
        grid = _sample_grid(side, fg_count)
        if _accept_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
