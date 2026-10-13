from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MASK_COLOR = 8
SEPARATOR_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 9)
PANEL_SIZES = (3, 4, 5)  # The output side is n*n, so it is at most 25.
GENERATION_RETRIES = 300


def _single_foreground_color(panel: np.ndarray, name: str) -> int:
    colors = np.unique(panel[panel != BACKGROUND])
    assert len(colors) == 1, f"{name} must use exactly one foreground color"
    return int(colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a seed sprite blockwise, using mask/seed coincidence to choose its positive or negative stamp.

    Input structure:
    - A full vertical separator divides the grid into two same-size square panels.
    - The left panel contains a one-color seed sprite on black; the right panel
      contains a different one-color placement mask on black.

    Rule:
    - Make an n-by-n array of n-by-n blocks, one block for every right-panel cell.
    - A black mask cell makes a black block. A foreground mask cell makes a copy
      of the left panel, except that coincidence with left-panel foreground at
      that same panel coordinate makes the copied occupancy inverted.
    - Both positive and inverted stamps use the seed's foreground color.

    Key insight:
    - Each panel coordinate has two scales of meaning: it selects an output block,
      while seed/mask coincidence at that coordinate selects the stamp's polarity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, width = g.shape
    assert 2 * n + 1 == width, "expected two n-by-n panels and one separator column"
    assert 1 <= n <= 5, "panel size would make an ARC output larger than 30"

    separator = g[:, n]
    assert np.all(separator == separator[0]), "separator must be a full monochromatic bar"
    separator_color = int(separator[0])
    assert separator_color != BACKGROUND, "separator must be visible"

    left = g[:, :n]
    right = g[:, n + 1 :]
    seed_color = _single_foreground_color(left, "left panel")
    mask_color = _single_foreground_color(right, "right panel")
    assert len({seed_color, mask_color, separator_color, BACKGROUND}) == 4, (
        "seed, mask, separator, and background colors must be distinct"
    )

    seed = left == seed_color
    placements = right == mask_color
    assert np.all((left == BACKGROUND) | seed), "left panel contains an unexpected color"
    assert np.all((right == BACKGROUND) | placements), "right panel contains an unexpected color"

    positive_stamp = np.where(seed, seed_color, BACKGROUND)
    negative_stamp = np.where(seed, BACKGROUND, seed_color)
    out = np.full((n * n, n * n), BACKGROUND, dtype=int)

    for block_r, block_c in np.argwhere(placements):
        block_r, block_c = int(block_r), int(block_c)
        stamp = negative_stamp if seed[block_r, block_c] else positive_stamp
        r0, c0 = block_r * n, block_c * n
        out[r0 : r0 + n, c0 : c0 + n] = stamp

    return out


def _random_connected_seed(n: int) -> np.ndarray:
    """Grow a sparse 4-connected full-panel sprite with no dihedral symmetry."""
    max_area = (n * n - 1) // 2
    for _ in range(100):
        target_area = random.randint(3, max_area)
        cells = {(random.randrange(n), random.randrange(n))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        seed = np.zeros((n, n), dtype=bool)
        for r, c in cells:
            seed[r, c] = True

        variants = [np.rot90(seed, k) for k in range(1, 4)]
        variants += [np.fliplr(np.rot90(seed, k)) for k in range(4)]
        if len(cells) == target_area and all(not np.array_equal(seed, v) for v in variants):
            return seed
    raise RuntimeError("failed to grow a non-symmetric seed")


def _independent_oracle(seed: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    """Direct cellwise oracle, intentionally separate from main's stamp arrays."""
    n = seed.shape[0]
    expected = np.zeros((n * n, n * n), dtype=int)
    for macro_r in range(n):
        for macro_c in range(n):
            if not mask[macro_r, macro_c]:
                continue
            invert = bool(seed[macro_r, macro_c])
            for local_r in range(n):
                for local_c in range(n):
                    occupied = bool(seed[local_r, local_c]) != invert
                    if occupied:
                        expected[macro_r * n + local_r, macro_c * n + local_c] = color
    return expected


def _semantically_valid(inp: np.ndarray, seed: np.ndarray, mask: np.ndarray, color: int) -> bool:
    """Reject shortcuts: all branches and both inactive coordinate classes must exist."""
    overlap = mask & seed
    ordinary = mask & ~seed
    inactive_seed = ~mask & seed
    inactive_background = ~mask & ~seed
    if not all(np.any(x) for x in (overlap, ordinary, inactive_seed, inactive_background)):
        return False

    expected = _independent_oracle(seed, mask, color)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    # Coincidence must be necessary: printing a positive stamp at an overlap differs.
    r, c = map(int, np.argwhere(overlap)[0])
    n = seed.shape[0]
    positive = np.where(seed, color, BACKGROUND)
    if np.array_equal(expected[r * n : (r + 1) * n, c * n : (c + 1) * n], positive):
        return False

    # Placement masking must be necessary: an inactive coordinate really leaves a black block.
    r, c = map(int, np.argwhere(inactive_background)[0])
    return not np.any(expected[r * n : (r + 1) * n, c * n : (c + 1) * n])


def _generation_attempt() -> np.ndarray | None:
    n = random.choice(PANEL_SIZES)
    try:
        seed = _random_connected_seed(n)
    except RuntimeError:
        return None

    seed_coords = [tuple(map(int, p)) for p in np.argwhere(seed)]
    background_coords = [tuple(map(int, p)) for p in np.argwhere(~seed)]

    # Sample each polarity separately and leave examples of each coordinate class inactive.
    overlap_count = random.randint(1, min(3, len(seed_coords) - 1))
    ordinary_count = random.randint(1, min(3, len(background_coords) - 1))
    mask = np.zeros((n, n), dtype=bool)
    for r, c in random.sample(seed_coords, overlap_count):
        mask[r, c] = True
    for r, c in random.sample(background_coords, ordinary_count):
        mask[r, c] = True

    seed_color = random.choice(SEED_COLORS)
    inp = np.zeros((n, 2 * n + 1), dtype=int)
    inp[:, :n][seed] = seed_color
    inp[:, n] = SEPARATOR_COLOR
    inp[:, n + 1 :][mask] = MASK_COLOR

    return inp if _semantically_valid(inp, seed, mask, seed_color) else None


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate polarity-stamp input."""
    for _ in range(GENERATION_RETRIES):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid polarity-stamp input")
