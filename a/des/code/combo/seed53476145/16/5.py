import random

import numpy as np


OUTPUT_HEIGHT_RANGE = (2, 9)
OUTPUT_WIDTH_RANGE = (2, 9)
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300

PATTERN_MODES = ("random", "horizontal_bands", "vertical_bands", "checker", "rectangles")
POSITION_MODES = ("random", "constant", "row_cycle", "col_cycle", "checker_cycle")
MAJORITY_MODES = ("random", "global", "row_anchor", "col_anchor")
POSITIONS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _minority_color(block: np.ndarray) -> int:
    colors, counts = np.unique(block, return_counts=True)
    assert 2 <= len(colors) <= 3, "each block must contain 2-3 colors"
    singleton_colors = colors[counts == 1]
    assert len(singleton_colors) == 1, "each block must have exactly one singleton minority"
    return int(singleton_colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0 and width % 2 == 0, "grid dimensions must be even"

    out = np.empty((height // 2, width // 2), dtype=int)
    for row in range(0, height, 2):
        for col in range(0, width, 2):
            out[row // 2, col // 2] = _minority_color(g[row : row + 2, col : col + 2])
    return out


def _random_output_pattern(height: int, width: int, colors: list[int], rng: random.Random) -> np.ndarray:
    mode = rng.choice(PATTERN_MODES)
    grid = np.empty((height, width), dtype=int)

    if mode == "random":
        for row in range(height):
            for col in range(width):
                grid[row, col] = rng.choice(colors)
        return grid

    if mode == "horizontal_bands":
        band_count = rng.randint(2, min(height, len(colors) + 2))
        cuts = sorted(rng.sample(range(1, height), band_count - 1)) if band_count > 1 else []
        starts = [0] + cuts
        ends = cuts + [height]
        for start, end in zip(starts, ends):
            color = rng.choice(colors)
            grid[start:end, :] = color
        return grid

    if mode == "vertical_bands":
        band_count = rng.randint(2, min(width, len(colors) + 2))
        cuts = sorted(rng.sample(range(1, width), band_count - 1)) if band_count > 1 else []
        starts = [0] + cuts
        ends = cuts + [width]
        for start, end in zip(starts, ends):
            color = rng.choice(colors)
            grid[:, start:end] = color
        return grid

    if mode == "checker":
        shift = rng.randrange(len(colors))
        for row in range(height):
            for col in range(width):
                grid[row, col] = colors[(row + col + shift) % len(colors)]
        return grid

    base = rng.choice(colors)
    grid.fill(base)
    rectangle_count = rng.randint(2, min(5, height * width))
    for _ in range(rectangle_count):
        r0 = rng.randrange(height)
        c0 = rng.randrange(width)
        r1 = rng.randrange(r0 + 1, height + 1)
        c1 = rng.randrange(c0 + 1, width + 1)
        color = rng.choice(colors)
        grid[r0:r1, c0:c1] = color
    return grid


def _position_grid(height: int, width: int, rng: random.Random) -> np.ndarray:
    mode = rng.choice(POSITION_MODES)
    grid = np.empty((height, width), dtype=int)

    if mode == "random":
        for row in range(height):
            for col in range(width):
                grid[row, col] = rng.randrange(4)
        return grid

    if mode == "constant":
        grid.fill(rng.randrange(4))
        return grid

    offset = rng.randrange(4)
    for row in range(height):
        for col in range(width):
            if mode == "row_cycle":
                value = row + offset
            elif mode == "col_cycle":
                value = col + offset
            else:
                value = row + col + offset
            grid[row, col] = value % 4
    return grid


def _pick_other_color(minority: int, colors: list[int], rng: random.Random, preferred: int | None = None) -> int:
    if preferred is not None and preferred != minority:
        return int(preferred)
    choices = [color for color in colors if color != minority]
    return int(rng.choice(choices))


def _majority_grid(minority_grid: np.ndarray, colors: list[int], rng: random.Random) -> np.ndarray:
    height, width = minority_grid.shape
    mode = rng.choice(MAJORITY_MODES)
    grid = np.empty((height, width), dtype=int)

    if mode == "random":
        for row in range(height):
            for col in range(width):
                grid[row, col] = _pick_other_color(int(minority_grid[row, col]), colors, rng)
        return grid

    if mode == "global":
        preferred = rng.choice(colors)
        for row in range(height):
            for col in range(width):
                grid[row, col] = _pick_other_color(int(minority_grid[row, col]), colors, rng, preferred)
        return grid

    anchors = [rng.choice(colors) for _ in range(height if mode == "row_anchor" else width)]
    for row in range(height):
        for col in range(width):
            preferred = anchors[row] if mode == "row_anchor" else anchors[col]
            grid[row, col] = _pick_other_color(int(minority_grid[row, col]), colors, rng, preferred)
    return grid


def _build_input(minority_grid: np.ndarray, position_grid: np.ndarray, majority_grid: np.ndarray) -> np.ndarray:
    out_height, out_width = minority_grid.shape
    grid = np.empty((out_height * 2, out_width * 2), dtype=int)
    for row in range(out_height):
        for col in range(out_width):
            block = np.full((2, 2), int(majority_grid[row, col]), dtype=int)
            dr, dc = POSITIONS[int(position_grid[row, col])]
            block[dr, dc] = int(minority_grid[row, col])
            grid[2 * row : 2 * row + 2, 2 * col : 2 * col + 2] = block
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    height, width = grid.shape
    if height % 2 != 0 or width % 2 != 0:
        return False
    if len(np.unique(grid)) < 2:
        return False

    minority_colors = []
    for row in range(0, height, 2):
        for col in range(0, width, 2):
            block = grid[row : row + 2, col : col + 2]
            colors, counts = np.unique(block, return_counts=True)
            if not (2 <= len(colors) <= 3):
                return False
            if np.sum(counts == 1) != 1:
                return False
            minority_colors.append(int(colors[counts == 1][0]))

    return len(set(minority_colors)) >= 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    out_height = rng.randint(*OUTPUT_HEIGHT_RANGE)
    out_width = rng.randint(*OUTPUT_WIDTH_RANGE)
    palette_size = rng.randint(2, min(6, len(PALETTE)))
    colors = rng.sample(PALETTE, palette_size)

    minority_grid = _random_output_pattern(out_height, out_width, colors, rng)
    if len(np.unique(minority_grid)) < 2:
        return None

    position_grid = _position_grid(out_height, out_width, rng)
    majority_grid = _majority_grid(minority_grid, colors, rng)
    grid = _build_input(minority_grid, position_grid, majority_grid)

    if not _valid_input(grid):
        return None
    if not np.array_equal(main(grid), minority_grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
