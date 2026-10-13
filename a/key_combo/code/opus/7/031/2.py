from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_SIZE = 9
MAX_SIZE = 13
MIN_SEEDS = 3
MAX_SEEDS = 5
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask as coord lists."""
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)

    return comps


def _fill_transposed_regions(transposed: np.ndarray) -> np.ndarray:
    """Fill each non-wall connected region if it contains exactly one seed color."""
    out = transposed.copy()
    passable = transposed != WALL

    for comp in _components(passable):
        colors = {
            int(transposed[r, c])
            for r, c in comp
            if int(transposed[r, c]) not in (BACKGROUND, WALL)
        }
        assert len(colors) <= 1, "each post-transpose region must contain at most one seed"
        if not colors:
            continue
        fill_color = next(iter(colors))
        for r, c in comp:
            out[r, c] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transpose the square grid and fill each post-transpose wall-bounded region from its seed.

    Input structure:
    - A square black-background grid contains a full grey main diagonal.
    - Additional grey horizontal and vertical wall segments attach to the diagonal or to outer edges.
    - Several single-cell colored seeds lie on non-grey cells inside the wall-defined regions.

    Rule:
    - Transpose the entire grid along the main diagonal.
    - In the transposed grid, treat grey cells as walls and split all non-grey cells into 4-connected regions.
    - If a region contains exactly one colored seed, fill that entire region with the seed color; regions with no seed stay black.

    Key insight:
    - The transpose changes both the wall layout and the seed locations, so the final fill depends on the new region ownership after transposition, not on the original regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height == width, "expected a square grid"

    transposed = g.T.copy()
    return _fill_transposed_regions(transposed)


def _sample_spaced(candidates: list[int], count: int, min_gap: int) -> list[int]:
    if count <= 0 or not candidates:
        return []
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[int] = []
    for value in shuffled:
        if all(abs(value - prior) >= min_gap for prior in chosen):
            chosen.append(value)
            if len(chosen) == count:
                break
    return sorted(chosen)


def _build_post_transpose_grid(
    n: int,
    upper_verticals: list[int],
    upper_horizontals: list[int],
    lower_verticals: list[int],
    lower_horizontals: list[int],
) -> np.ndarray:
    grid = np.full((n, n), BACKGROUND, dtype=int)
    np.fill_diagonal(grid, WALL)

    for c in upper_verticals:
        grid[:c, c] = WALL
    for r in upper_horizontals:
        grid[r, r + 1 :] = WALL
    for c in lower_verticals:
        grid[c + 1 :, c] = WALL
    for r in lower_horizontals:
        grid[r, :r] = WALL

    return grid


def _pick_wall_sets(n: int) -> tuple[list[int], list[int], list[int], list[int]] | None:
    inner_cols = list(range(2, n - 2))
    inner_rows = list(range(1, n - 3))

    upper_verticals = _sample_spaced(inner_cols, random.randint(1, 2), min_gap=2)
    upper_horizontals = _sample_spaced(inner_rows, random.randint(0, 2), min_gap=2)
    lower_verticals = _sample_spaced(inner_rows, random.randint(0, 2), min_gap=2)
    lower_horizontals = _sample_spaced(inner_cols, random.randint(1, 2), min_gap=2)

    total_walls = (
        len(upper_verticals)
        + len(upper_horizontals)
        + len(lower_verticals)
        + len(lower_horizontals)
    )
    if total_walls < 4:
        return None

    symmetric = upper_verticals == lower_horizontals and upper_horizontals == lower_verticals
    if symmetric:
        return None

    return upper_verticals, upper_horizontals, lower_verticals, lower_horizontals


def _valid_regions(post: np.ndarray) -> list[list[tuple[int, int]]] | None:
    comps = _components(post != WALL)
    if not (4 <= len(comps) <= 10):
        return None

    sizable = [comp for comp in comps if len(comp) >= 2]
    if len(sizable) < 4:
        return None

    if max(len(comp) for comp in comps) < 5:
        return None

    return comps


def _place_seeds(post: np.ndarray, comps: list[list[tuple[int, int]]]) -> bool:
    sizable = [comp for comp in comps if len(comp) >= 2]
    max_seed_count = min(MAX_SEEDS, len(sizable), len(comps) - 1)
    if max_seed_count < MIN_SEEDS:
        return False

    seed_count = random.randint(MIN_SEEDS, max_seed_count)
    seeded_components = random.sample(sizable, seed_count)
    colors = random.sample(SEED_COLORS, seed_count)

    for comp, color in zip(seeded_components, colors):
        r, c = random.choice(comp)
        post[r, c] = color

    return True


def _attempt_generate_post() -> np.ndarray | None:
    n = random.randint(MIN_SIZE, MAX_SIZE)
    wall_sets = _pick_wall_sets(n)
    if wall_sets is None:
        return None

    post = _build_post_transpose_grid(n, *wall_sets)
    comps = _valid_regions(post)
    if comps is None:
        return None

    if not _place_seeds(post, comps):
        return None

    filled = _fill_transposed_regions(post)
    if np.array_equal(filled, post):
        return None
    if not np.any(filled == BACKGROUND):
        return None

    return post


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        post = _attempt_generate_post()
        if post is None:
            continue
        return post.T.copy()
    raise RuntimeError("failed to generate a valid input for puzzle 2")
