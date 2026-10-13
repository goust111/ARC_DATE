from __future__ import annotations

import random

import numpy as np


BG = 5
PALETTE = [1, 2, 3]
MAX_ATTEMPTS = 400


def _door_centers(grid: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    doors: dict[tuple[int, int], int] = {}
    for r in range(1, h - 1):
        for c in range(w):
            color = int(g[r - 1, c])
            if color == bg:
                continue
            if int(g[r, c]) != bg:
                continue
            if int(g[r + 1, c]) != color:
                continue
            doors[(r, c)] = color
    return doors


def _door_posts(doors: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    posts: set[tuple[int, int]] = set()
    for (r, c) in doors:
        posts.add((r - 1, c))
        posts.add((r + 1, c))
    return posts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each ball rightward through its first color-filter gate and stop it at the
    second gate or at the right edge.

    Input structure:
    - The background is grey.
    - A gate is a same-color vertical pair with a one-cell background channel between.
    - A ball is a single colored pixel on the gate's center row, to the left of the gates.

    Rule:
    - The ball advances to the first gate; if the gate color differs, the ball changes
      to that gate color.
    - The ball then continues right and stops at the second gate if present, otherwise
      at the row's right edge.
    - Erase the original ball and all gate pixels; keep only the final ball.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    h, w = g.shape
    assert h >= 3 and w >= 4, "grid is too small for this task"

    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])

    doors = _door_centers(g, bg)
    assert doors, "expected at least one gate"
    posts = _door_posts(doors)

    balls: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or (r, c) in posts:
                continue
            balls.append((r, c, color))

    assert balls, "expected at least one ball"
    out = np.full_like(g, bg)

    for r, c, color in balls:
        row_doors = sorted(col for (rr, col), _gate_color in doors.items() if rr == r and col > c)
        assert 1 <= len(row_doors) <= 2, "each ball row must contain one or two gates"
        first_col = row_doors[0]
        current = color
        first_gate = doors[(r, first_col)]
        if first_gate != current:
            current = first_gate

        final_col = row_doors[1] if len(row_doors) == 2 else w - 1
        out[r, final_col] = current

    return out


def _spaced_rows(height: int, count: int) -> list[int] | None:
    rows = list(range(1, height - 1))
    random.shuffle(rows)
    chosen: list[int] = []
    for r in rows:
        if all(abs(r - prev) >= 3 for prev in chosen):
            chosen.append(r)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _row_specs(count: int) -> list[tuple[bool, bool]]:
    categories = [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ]
    random.shuffle(categories)
    specs = categories[:count]
    if count < 4:
        while len(specs) < count:
            specs.append(random.choice(categories))
    random.shuffle(specs)
    return specs


def _choose_door_columns(width: int, two_doors: bool) -> list[int] | None:
    first_min = 2
    first_max = width - (3 if two_doors else 2)
    if first_min > first_max:
        return None
    first = random.randint(first_min, first_max)
    if not two_doors:
        return [first]

    second_min = first + 2
    second_max = width - 2
    if second_min > second_max:
        return None
    second = random.randint(second_min, second_max)
    return [first, second]


def _ball_and_gate_colors(same_color: bool) -> tuple[int, int]:
    ball = random.choice(PALETTE)
    if same_color:
        return ball, ball
    gate = random.choice([color for color in PALETTE if color != ball])
    return ball, gate


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(7, 15)
    width = random.randint(8, 18)
    max_rows = min(4, (height - 2 + 2) // 3)
    if max_rows < 2:
        return None

    row_count = random.randint(2, max_rows)
    rows = _spaced_rows(height, row_count)
    if rows is None:
        return None

    specs = _row_specs(row_count)
    grid = np.full((height, width), BG, dtype=int)

    for r, (same_color, two_doors) in zip(rows, specs):
        door_cols = _choose_door_columns(width, two_doors)
        if door_cols is None:
            return None

        ball_color, first_gate_color = _ball_and_gate_colors(same_color)
        ball_col = random.choice([0, 1]) if door_cols[0] >= 3 else 0
        if ball_col >= door_cols[0]:
            return None
        grid[r, ball_col] = ball_color

        gate_colors = [first_gate_color]
        if two_doors:
            gate_colors.append(random.choice(PALETTE))

        for c, gate_color in zip(door_cols, gate_colors):
            grid[r - 1, c] = gate_color
            grid[r + 1, c] = gate_color

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
