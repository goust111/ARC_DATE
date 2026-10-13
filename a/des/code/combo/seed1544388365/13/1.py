from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

SEED_COLORS = (RED, BLUE, YELLOW, GREEN)
MIX_COLORS = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
    frozenset((RED, GREEN)): MAROON,
    frozenset((BLUE, GREEN)): AZURE,
}

PIPE_COUNT_OPTIONS = [
    (h_count, v_count)
    for h_count in range(1, 6)
    for v_count in range(1, 6)
    if 3 <= h_count + v_count <= 6 and 2 <= h_count * v_count <= 6
]
GRID_MAX_SIDE = 25
MIN_PIPE_GAP = 3
MAX_GENERATION_ATTEMPTS = 500


def _horizontal_pipes(mask: np.ndarray) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    height, width = mask.shape
    pipe_id = np.full((height, width), -1, dtype=int)
    pipes: list[tuple[int, int, int]] = []
    for row in range(height):
        col = 0
        while col < width:
            if not mask[row, col]:
                col += 1
                continue
            start = col
            while col + 1 < width and mask[row, col + 1]:
                col += 1
            end = col
            if end - start + 1 >= 2:
                idx = len(pipes)
                pipes.append((row, start, end))
                pipe_id[row, start : end + 1] = idx
            col += 1
    return pipes, pipe_id


def _vertical_pipes(mask: np.ndarray) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    height, width = mask.shape
    pipe_id = np.full((height, width), -1, dtype=int)
    pipes: list[tuple[int, int, int]] = []
    for col in range(width):
        row = 0
        while row < height:
            if not mask[row, col]:
                row += 1
                continue
            start = row
            while row + 1 < height and mask[row + 1, col]:
                row += 1
            end = row
            if end - start + 1 >= 2:
                idx = len(pipes)
                pipes.append((col, start, end))
                pipe_id[start : end + 1, col] = idx
            row += 1
    return pipes, pipe_id


def _seed_color(value: int) -> int | None:
    color = int(value)
    if color in (BLACK, GREY):
        return None
    return color


def _pipe_seed_horizontal(grid: np.ndarray, pipe: tuple[int, int, int]) -> int | None:
    row, start, end = pipe
    colors: list[int] = []
    if start - 1 >= 0:
        left = _seed_color(grid[row, start - 1])
        if left is not None:
            colors.append(left)
    if end + 1 < grid.shape[1]:
        right = _seed_color(grid[row, end + 1])
        if right is not None:
            colors.append(right)
    assert len(colors) <= 1, "each pipe may have at most one seed"
    return colors[0] if colors else None


def _pipe_seed_vertical(grid: np.ndarray, pipe: tuple[int, int, int]) -> int | None:
    col, start, end = pipe
    colors: list[int] = []
    if start - 1 >= 0:
        top = _seed_color(grid[start - 1, col])
        if top is not None:
            colors.append(top)
    if end + 1 < grid.shape[0]:
        bottom = _seed_color(grid[end + 1, col])
        if bottom is not None:
            colors.append(bottom)
    assert len(colors) <= 1, "each pipe may have at most one seed"
    return colors[0] if colors else None


def _mixed_color(color_a: int, color_b: int) -> int:
    if color_a == color_b:
        return int(color_a)
    pair = frozenset((int(color_a), int(color_b)))
    assert pair in MIX_COLORS, f"unsupported mixed pair: {sorted(pair)}"
    return int(MIX_COLORS[pair])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate seed colors along grey pipes and mix colors at intersections.

    Input structure:
    - Black background.
    - One-pixel-wide grey horizontal and vertical pipes, possibly crossing.
    - Some pipe endpoints have one adjacent colored seed pixel outside the pipe.

    Rule:
    - A seed recolors its whole pipe.
    - At each crossing, if both directions are colored, replace the crossing with
      the fixed mixed color. If only one direction is colored, keep that color.
      If neither direction is colored, keep grey.
    - Erase the seed pixels to black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_mask = g == GREY
    h_pipes, h_id = _horizontal_pipes(grey_mask)
    v_pipes, v_id = _vertical_pipes(grey_mask)

    h_colors = [_pipe_seed_horizontal(g, pipe) for pipe in h_pipes]
    v_colors = [_pipe_seed_vertical(g, pipe) for pipe in v_pipes]
    detected = [color for color in h_colors + v_colors if color is not None]
    assert len(detected) == len(set(detected)), "seed colors must be unique across pipes"

    out = np.full(g.shape, BLACK, dtype=int)
    for row, col in np.argwhere(grey_mask):
        hid = int(h_id[row, col])
        vid = int(v_id[row, col])
        h_color = h_colors[hid] if hid != -1 else None
        v_color = v_colors[vid] if vid != -1 else None

        if h_color is not None and v_color is not None:
            out[row, col] = _mixed_color(h_color, v_color)
        elif h_color is not None:
            out[row, col] = int(h_color)
        elif v_color is not None:
            out[row, col] = int(v_color)
        else:
            out[row, col] = GREY

    return out


def _spaced_sample(low: int, high: int, count: int, min_gap: int) -> list[int]:
    adjusted_high = high - (count - 1) * (min_gap - 1)
    if adjusted_high < low or adjusted_high - low + 1 < count:
        raise ValueError("range too small for requested spacing")
    base = sorted(random.sample(range(low, adjusted_high + 1), count))
    return [value + idx * (min_gap - 1) for idx, value in enumerate(base)]


def _valid_color_assignment(h_colors: list[int], v_colors: list[int]) -> bool:
    for h_color in h_colors:
        for v_color in v_colors:
            if frozenset((int(h_color), int(v_color))) not in MIX_COLORS:
                return False
    return True


def _sample_seed_assignment(
    h_count: int,
    v_count: int,
) -> tuple[dict[int, int], dict[int, int]] | None:
    total_pipes = h_count + v_count
    seed_total = random.randint(2, min(4, total_pipes))
    splits = [
        (seed_h, seed_total - seed_h)
        for seed_h in range(1, min(h_count, seed_total - 1) + 1)
        if 1 <= seed_total - seed_h <= v_count
    ]
    if not splits:
        return None

    seed_h, seed_v = random.choice(splits)
    h_indices = random.sample(range(h_count), seed_h)
    v_indices = random.sample(range(v_count), seed_v)

    for _ in range(200):
        chosen = random.sample(SEED_COLORS, seed_total)
        random.shuffle(chosen)
        h_colors = chosen[:seed_h]
        v_colors = chosen[seed_h:]
        if not _valid_color_assignment(h_colors, v_colors):
            continue
        return dict(zip(h_indices, h_colors)), dict(zip(v_indices, v_colors))

    return None


def _paint_seeded_network(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    h_seed_colors: dict[int, int],
    v_seed_colors: dict[int, int],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)

    min_col = min(cols)
    max_col = max(cols)
    min_row = min(rows)
    max_row = max(rows)

    h_pipes: list[tuple[int, int, int]] = []
    for row in rows:
        start = random.randint(1, min_col - 1)
        end = random.randint(max_col + 1, width - 2)
        h_pipes.append((row, start, end))
        grid[row, start : end + 1] = GREY

    v_pipes: list[tuple[int, int, int]] = []
    for col in cols:
        start = random.randint(1, min_row - 1)
        end = random.randint(max_row + 1, height - 2)
        v_pipes.append((col, start, end))
        grid[start : end + 1, col] = GREY

    for idx, color in h_seed_colors.items():
        row, start, end = h_pipes[idx]
        if random.random() < 0.5:
            grid[row, start - 1] = color
        else:
            grid[row, end + 1] = color

    for idx, color in v_seed_colors.items():
        col, start, end = v_pipes[idx]
        if random.random() < 0.5:
            grid[start - 1, col] = color
        else:
            grid[end + 1, col] = color

    return grid


def _generate_attempt() -> np.ndarray | None:
    h_count, v_count = random.choice(PIPE_COUNT_OPTIONS)

    min_height = max(8, 6 + (h_count - 1) * MIN_PIPE_GAP)
    min_width = max(8, 6 + (v_count - 1) * MIN_PIPE_GAP)
    if min_height > GRID_MAX_SIDE or min_width > GRID_MAX_SIDE:
        return None

    height = random.randint(min_height, GRID_MAX_SIDE)
    width = random.randint(min_width, GRID_MAX_SIDE)

    try:
        rows = _spaced_sample(2, height - 3, h_count, MIN_PIPE_GAP)
        cols = _spaced_sample(2, width - 3, v_count, MIN_PIPE_GAP)
    except ValueError:
        return None

    assignment = _sample_seed_assignment(h_count, v_count)
    if assignment is None:
        return None
    h_seed_colors, v_seed_colors = assignment

    grid = _paint_seeded_network(height, width, rows, cols, h_seed_colors, v_seed_colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network input")
