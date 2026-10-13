from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GRID_H_RANGE = (9, 16)
GRID_W_RANGE = (11, 18)
ACTIVE_ROW_MIN = 6
MAX_RETRIES = 400

PIERCE = "pierce"
MISMATCH = "mismatch"
LEFT_ONLY = "left_only"
RIGHT_ONLY = "right_only"
BRANCHES = (PIERCE, MISMATCH, LEFT_ONLY, RIGHT_ONLY)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_wall(grid: np.ndarray, bg: int) -> tuple[int, int]:
    counts = Counter(map(int, grid.ravel()))
    foreground = {color: count for color, count in counts.items() if color != bg}
    assert foreground, "expected foreground pixels"
    wall_color, wall_count = max(foreground.items(), key=lambda item: item[1])
    assert list(foreground.values()).count(wall_count) == 1, "wall color must be uniquely most frequent"

    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == wall_color)]
    assert len(cols) == 1, "expected exactly one complete vertical wall"
    wall_col = int(cols[0])
    assert 0 < wall_col < grid.shape[1] - 1, "wall must have space on both sides"
    assert wall_count == grid.shape[0], "wall must be one cell thick with no extra wall pixels"
    return int(wall_color), wall_col


def _row_markers(row: np.ndarray, wall_col: int, bg: int, wall_color: int) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    left = [(c, int(row[c])) for c in range(wall_col) if int(row[c]) not in (bg, wall_color)]
    right = [(c, int(row[c])) for c in range(wall_col + 1, len(row)) if int(row[c]) not in (bg, wall_color)]
    assert len(left) <= 1 and len(right) <= 1, "each row may have at most one marker per side"
    return (left[0] if left else None), (right[0] if right else None)


def _solve_row(row: np.ndarray, wall_col: int, bg: int, wall_color: int) -> np.ndarray:
    out = np.full(len(row), bg, dtype=int)
    out[wall_col] = wall_color
    left, right = _row_markers(row, wall_col, bg, wall_color)

    if left is None and right is None:
        return out

    if left is not None and right is not None and left[1] == right[1]:
        out[wall_col] = bg
        out[0] = left[1]
        out[-1] = right[1]
        return out

    if left is not None:
        out[wall_col - 1] = left[1]
    if right is not None:
        out[wall_col + 1] = right[1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marker pixels toward a complete vertical wall so only same-color cross-wall pairs punch a hole and pass through to the far edges.

    Input structure:
    - The background is the dominant color.
    - One non-background color forms a complete one-cell-thick vertical wall spanning every row.
    - Every other foreground pixel is a single marker, with each row containing at most one marker on each side of the wall.

    Rule:
    - For any row whose left and right markers both exist and have the same color, erase that row's wall cell, then send both markers through the hole to the opposite outer edge cells.
    - For any other marked row, slide each marker straight toward the wall and stop it in the cell adjacent to the wall on its own side; the wall stays intact.
    - Rows without markers remain unchanged apart from keeping their wall cell.

    Key insight:
    - Piercing is a synchronized row-level event triggered only by a same-color pair across the wall; otherwise motion is just independent stopping against the wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    wall_color, wall_col = _find_wall(g, bg)

    out = np.full_like(g, bg)
    for r in range(g.shape[0]):
        out[r] = _solve_row(g[r], wall_col, bg, wall_color)
    return out


def _branch_counts(branches: list[str]) -> Counter:
    return Counter(branches)


def _choose_colors() -> tuple[int, list[int]]:
    palette = list(range(1, 10))
    wall_color = random.choice(palette)
    palette.remove(wall_color)
    marker_count = random.randint(3, min(5, len(palette)))
    marker_colors = random.sample(palette, marker_count)
    return wall_color, marker_colors


def _candidate_cols(wall_col: int, width: int) -> tuple[list[int], list[int]]:
    left = list(range(0, wall_col - 1))
    right = list(range(wall_col + 2, width))
    assert left and right
    return left, right


def _paint_row(grid: np.ndarray, row: int, wall_col: int, branch: str, marker_colors: list[int], pierce_colors: list[int]) -> str:
    left_cols, right_cols = _candidate_cols(wall_col, grid.shape[1])
    left_c = random.choice(left_cols)
    right_c = random.choice(right_cols)

    if branch == PIERCE:
        if pierce_colors:
            color = pierce_colors.pop(0)
        else:
            color = random.choice(marker_colors)
        grid[row, left_c] = color
        grid[row, right_c] = color
        return branch

    if branch == MISMATCH:
        left_color, right_color = random.sample(marker_colors, 2)
        grid[row, left_c] = left_color
        grid[row, right_c] = right_color
        return branch

    color = random.choice(marker_colors)
    if branch == LEFT_ONLY:
        grid[row, left_c] = color
        return branch
    if branch == RIGHT_ONLY:
        grid[row, right_c] = color
        return branch
    raise ValueError(branch)


def _classify_row(row: np.ndarray, wall_col: int, bg: int, wall_color: int) -> str | None:
    left, right = _row_markers(row, wall_col, bg, wall_color)
    if left is None and right is None:
        return None
    if left is not None and right is not None:
        return PIERCE if left[1] == right[1] else MISMATCH
    return LEFT_ONLY if left is not None else RIGHT_ONLY


def _semantic_check(inp: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2
    assert max(inp.shape) <= 30

    bg = _get_bg(inp)
    wall_color, wall_col = _find_wall(inp, bg)
    counts = Counter(map(int, inp.ravel()))
    assert counts[bg] > counts[wall_color], "background must remain dominant"

    branches: list[str] = []
    used_rows = 0
    for r in range(inp.shape[0]):
        branch = _classify_row(inp[r], wall_col, bg, wall_color)
        if branch is not None:
            branches.append(branch)
            used_rows += 1
            left, right = _row_markers(inp[r], wall_col, bg, wall_color)
            if left is not None:
                assert left[0] <= wall_col - 2, "input markers must start away from the wall"
            if right is not None:
                assert right[0] >= wall_col + 2, "input markers must start away from the wall"

    branch_counter = _branch_counts(branches)
    assert used_rows >= ACTIVE_ROW_MIN
    for branch in BRANCHES:
        assert branch_counter[branch] >= 1, f"missing branch coverage: {branch}"
    assert branch_counter[PIERCE] >= 2, "need multiple piercing rows"
    assert sum(1 for c, n in counts.items() if c not in (bg, wall_color) and n > 0) >= 3
    assert max(counts[c] for c in counts if c not in (bg, wall_color)) < counts[wall_color], "wall must stay uniquely most frequent"

    out = main(inp)
    assert out.shape == inp.shape
    assert not np.array_equal(inp, out), "transform must do visible work"
    assert np.any(out[:, wall_col] == bg), "at least one hole row is required"
    assert np.any(out[:, wall_col] == wall_color), "at least one intact wall row is required"

    for r in range(inp.shape[0]):
        row_branch = _classify_row(inp[r], wall_col, bg, wall_color)
        expected = _solve_row(inp[r], wall_col, bg, wall_color)
        assert np.array_equal(out[r], expected), "row semantics mismatch"
        if row_branch == PIERCE:
            left, right = _row_markers(inp[r], wall_col, bg, wall_color)
            assert left is not None and right is not None
            color = left[1]
            assert out[r, 0] == color and out[r, -1] == color
            assert out[r, wall_col] == bg
        elif row_branch == MISMATCH:
            left, right = _row_markers(inp[r], wall_col, bg, wall_color)
            assert left is not None and right is not None and left[1] != right[1]
            assert out[r, wall_col] == wall_color
            assert out[r, wall_col - 1] == left[1]
            assert out[r, wall_col + 1] == right[1]
        elif row_branch == LEFT_ONLY:
            left, _right = _row_markers(inp[r], wall_col, bg, wall_color)
            assert left is not None
            assert out[r, wall_col - 1] == left[1]
            assert out[r, wall_col] == wall_color
        elif row_branch == RIGHT_ONLY:
            _left, right = _row_markers(inp[r], wall_col, bg, wall_color)
            assert right is not None
            assert out[r, wall_col + 1] == right[1]
            assert out[r, wall_col] == wall_color


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(max(11, height + 1), GRID_W_RANGE[1])
    wall_col = random.randint(3, width - 4)
    wall_color, marker_colors = _choose_colors()

    grid = np.full((height, width), BG, dtype=int)
    grid[:, wall_col] = wall_color

    active_rows = random.randint(ACTIVE_ROW_MIN, min(height - 1, 10))
    rows = random.sample(range(height), active_rows)
    random.shuffle(rows)

    branches = [PIERCE, PIERCE, MISMATCH, LEFT_ONLY, RIGHT_ONLY]
    while len(branches) < active_rows:
        branches.append(random.choice(BRANCHES))
    random.shuffle(branches)

    distinct_pierce = random.sample(marker_colors, min(2, len(marker_colors)))
    pierce_colors = distinct_pierce[:]
    while len(pierce_colors) < branches.count(PIERCE):
        pierce_colors.append(random.choice(marker_colors))

    for row, branch in zip(rows, branches):
        _paint_row(grid, row, wall_col, branch, marker_colors, pierce_colors)

    try:
        _semantic_check(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
