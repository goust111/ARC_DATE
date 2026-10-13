from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

SEED_SIZE = 3
INPUT_SHAPE = (7, 3)
OUTPUT_SHAPE = (9, 9)
MAX_GENERATION_ATTEMPTS = 300


def _find_divider_row(grid: np.ndarray) -> int:
    full_orange_rows = np.flatnonzero(np.all(grid == ORANGE, axis=1))
    candidates = [
        int(r)
        for r in full_orange_rows
        if r >= SEED_SIZE and grid.shape[0] - r - 1 >= SEED_SIZE
    ]
    assert len(candidates) == 1, "expected one orange divider row with a 3x3 region on each side"
    return candidates[0]


def _extract_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] == SEED_SIZE, "expected a 3-column grid"

    divider_r = _find_divider_row(g)
    top = g[divider_r - SEED_SIZE : divider_r, :]
    bottom = g[divider_r + 1 : divider_r + 1 + SEED_SIZE, :]
    assert top.shape == (SEED_SIZE, SEED_SIZE)
    assert bottom.shape == (SEED_SIZE, SEED_SIZE)
    assert set(map(int, np.unique(top))) <= {BLACK, BLUE}
    assert set(map(int, np.unique(bottom))) <= {BLACK, RED}
    return top, bottom


def _seed_mask_from_regions(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    return (top == BLACK) & (bottom == BLACK)


def _kronecker_self_mask(mask: np.ndarray) -> np.ndarray:
    grown = np.kron(mask.astype(np.uint8), mask.astype(np.uint8))
    return np.where(grown != 0, GREEN, BLACK).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the double-background residue of two aligned 3x3 masks into a self-similar 9x9 expansion.

    Input structure:
    - A narrow grid contains one full-width orange divider row with a 3x3 region directly above it
      and another 3x3 region directly below it.
    - The upper region uses blue foreground pixels on black background; the lower region uses red
      foreground pixels on black background.
    - The two 3x3 regions are aligned cell-by-cell across the divider.

    Rule:
    - Remove the divider row and align the upper and lower 3x3 regions by position.
    - Build a 3x3 binary seed mask whose cells are on exactly where both aligned input cells are black.
    - Apply a Kronecker self-product to that seed: each on cell expands to a full copy of the seed,
      and each off cell expands to a 3x3 black block.
    - Color the on cells of the 9x9 result green and leave all off cells black.

    Key insight:
    - The boolean double-background mask is only an intermediate control pattern; its spatial layout
      becomes the final answer only after self-similar Kronecker expansion.
    """
    top, bottom = _extract_regions(grid)
    seed_mask = _seed_mask_from_regions(top, bottom)
    out = _kronecker_self_mask(seed_mask)
    assert out.shape == OUTPUT_SHAPE
    return out


def _build_category_options() -> dict[str, list[tuple[int, int, int, int]]]:
    options: dict[str, list[tuple[int, int, int, int]]] = {
        "mixed": [],
        "disjoint": [],
        "top_subset": [],
        "bottom_subset": [],
        "identical": [],
    }
    for n00 in range(3, 7):
        remaining = 9 - n00
        for n10 in range(remaining + 1):
            for n01 in range(remaining - n10 + 1):
                n11 = remaining - n10 - n01
                top_count = n10 + n11
                bottom_count = n01 + n11
                if not (3 <= top_count <= 6 and 3 <= bottom_count <= 6):
                    continue
                counts = (n00, n10, n01, n11)
                if n11 == 0 and n10 > 0 and n01 > 0:
                    options["disjoint"].append(counts)
                elif n10 == 0 and n01 == 0 and n11 > 0:
                    options["identical"].append(counts)
                elif n10 == 0 and n01 > 0 and n11 > 0:
                    options["top_subset"].append(counts)
                elif n01 == 0 and n10 > 0 and n11 > 0:
                    options["bottom_subset"].append(counts)
                elif n10 > 0 and n01 > 0 and n11 > 0:
                    options["mixed"].append(counts)
    return options


CATEGORY_OPTIONS = _build_category_options()
MODE_NAMES = np.array(["mixed", "disjoint", "top_subset", "bottom_subset", "identical"], dtype=object)
MODE_PROBS = np.array([0.35, 0.15, 0.2, 0.2, 0.1], dtype=float)


def _sample_category_grid(rng: np.random.Generator) -> np.ndarray:
    mode = str(rng.choice(MODE_NAMES, p=MODE_PROBS))
    options = CATEGORY_OPTIONS[mode]
    n00, n10, n01, n11 = options[int(rng.integers(len(options)))]

    flat = np.array(
        [0] * n00 +
        [1] * n10 +
        [2] * n01 +
        [3] * n11,
        dtype=int,
    )
    rng.shuffle(flat)
    return flat.reshape(SEED_SIZE, SEED_SIZE)


def _category_grid_to_regions(categories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = np.full((SEED_SIZE, SEED_SIZE), BLACK, dtype=int)
    bottom = np.full((SEED_SIZE, SEED_SIZE), BLACK, dtype=int)
    top[(categories == 1) | (categories == 3)] = BLUE
    bottom[(categories == 2) | (categories == 3)] = RED
    return top, bottom


def _render_input(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    grid = np.full(INPUT_SHAPE, BLACK, dtype=int)
    grid[:SEED_SIZE, :] = top
    grid[SEED_SIZE, :] = ORANGE
    grid[SEED_SIZE + 1 :, :] = bottom
    return grid


def _expected_from_categories(categories: np.ndarray) -> np.ndarray:
    seed_mask = categories == 0
    return _kronecker_self_mask(seed_mask)


def _valid_regions(top: np.ndarray, bottom: np.ndarray, categories: np.ndarray) -> bool:
    top_count = int(np.count_nonzero(top == BLUE))
    bottom_count = int(np.count_nonzero(bottom == RED))
    complement_count = int(np.count_nonzero(categories == 0))
    union_count = 9 - complement_count
    return (
        3 <= top_count <= 6
        and 3 <= bottom_count <= 6
        and 3 <= complement_count <= 6
        and 0 < union_count < 9
    )


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    categories = _sample_category_grid(rng)
    top, bottom = _category_grid_to_regions(categories)
    if not _valid_regions(top, bottom, categories):
        return None

    inp = _render_input(top, bottom)
    expected = _expected_from_categories(categories)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
