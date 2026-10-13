from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np


GREY = 5
FOREGROUND_CHOICES = (1, 2, 3)
SCALE_CHOICES = (2, 3)
SIDE_CHOICES = (3, 4)

MAX_GRID_SIDE = 30
GEN_ATTEMPTS = 600

_GEN_CALL_COUNT = 0


def _repeat_tile(tile: np.ndarray, reps_r: int, reps_c: int) -> np.ndarray:
    return np.tile(tile, (int(reps_r), int(reps_c)))


def _scale_tile(tile: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(tile, np.ones((factor, factor), dtype=int))


def _is_repetition(block: np.ndarray, tile: np.ndarray) -> bool:
    h, w = block.shape
    side = tile.shape[0]
    if tile.shape[0] != tile.shape[1]:
        return False
    if h % side != 0 or w % side != 0:
        return False
    return bool(np.array_equal(block, _repeat_tile(tile, h // side, w // side)))


def _minimal_square_period(tile: np.ndarray) -> int:
    side = int(tile.shape[0])
    for cand in range(1, side):
        if side % cand != 0:
            continue
        base = tile[:cand, :cand]
        if np.array_equal(tile, _repeat_tile(base, side // cand, side // cand)):
            return cand
    return side


def _parse_structure(grid: np.ndarray) -> tuple[np.ndarray, int, int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 9 and g.shape[1] >= 9, "grid too small"
    assert np.all(g[0, :] == GREY) and np.all(g[-1, :] == GREY), "missing outer grey frame"
    assert np.all(g[:, 0] == GREY) and np.all(g[:, -1] == GREY), "missing outer grey frame"

    inner = g[1:-1, 1:-1]
    grey_rows = [r for r in range(inner.shape[0]) if np.all(inner[r, :] == GREY)]
    grey_cols = [c for c in range(inner.shape[1]) if np.all(inner[:, c] == GREY)]
    assert len(grey_rows) == 1 and len(grey_cols) == 1, "expected one internal grey row and one internal grey column"

    split_r = grey_rows[0]
    split_c = grey_cols[0]
    assert split_r == split_c, "top-left compartment must be square"
    assert split_r > 0, "empty compartment"

    arrays = [
        inner[:split_r, split_c + 1 :],
        inner[split_r + 1 :, :split_c],
        inner[split_r + 1 :, split_c + 1 :],
    ]
    arrays = [arr.copy() for arr in arrays if arr.size > 0]
    assert arrays, "missing tiled array region"
    for arr in arrays:
        assert GREY not in arr, "grey separators must not appear inside array regions"
    return inner, split_r, split_c, arrays


def _enumerate_solutions(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    inner, split_r, split_c, arrays = _parse_structure(grid)
    compartment = inner[:split_r, :split_c]
    max_side = min(min(arr.shape) for arr in arrays)
    solutions: list[tuple[int, int, np.ndarray]] = []

    for side in range(1, max_side + 1):
        tile = None
        ok = True
        for arr in arrays:
            h, w = arr.shape
            if h % side != 0 or w % side != 0:
                ok = False
                break
            candidate = arr[:side, :side]
            if tile is None:
                tile = candidate
            elif not np.array_equal(tile, candidate):
                ok = False
                break
            if not _is_repetition(arr, tile):
                ok = False
                break
        if not ok or tile is None:
            continue

        unique_colors = set(map(int, np.unique(tile)))
        if not unique_colors <= {0, 1, 2, 3}:
            continue
        if len(unique_colors) != 2 or 0 not in unique_colors:
            continue

        for k in SCALE_CHOICES:
            if compartment.shape != (side * k, side * k):
                continue
            if np.array_equal(compartment, _scale_tile(tile, k)):
                solutions.append((side, k, tile.copy()))
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the repeated base tile, read the enlarged copy's scale factor, and downsample the base tile by that factor.

    Input structure:
    - A grey single-pixel frame surrounds the grid, and one internal grey row plus one internal grey column carve out a square top-left compartment.
    - The top-left compartment contains an enlarged copy of one square binary tile, using black and a single foreground color.
    - The other three quadrants are rectangular arrays made by repeating that same base tile.

    Rule:
    - Identify the unique smallest square tile whose copies exactly fill every non-grey array quadrant.
    - Compare that tile with the top-left compartment to determine whether the compartment is a 2x or 3x nearest-neighbor enlargement.
    - Starting from the tile's top-left corner, take every K-th row and every K-th column from the base tile and return the sampled result.

    Key insight:
    - The tiled quadrants reveal the true base tile, while the enlarged compartment acts as an instruction that encodes the sampling stride K.
    """
    solutions = _enumerate_solutions(grid)
    assert len(solutions) == 1, f"expected exactly one valid interpretation, found {len(solutions)}"
    _side, k, tile = solutions[0]
    out = tile[::k, ::k]
    return out.copy()


def _sampled_output(tile: np.ndarray, k: int) -> np.ndarray:
    return tile[::k, ::k].copy()


def _tile_is_good(tile: np.ndarray, k: int) -> bool:
    side = int(tile.shape[0])
    if tile.shape != (side, side):
        return False
    unique = set(map(int, np.unique(tile)))
    if len(unique) != 2 or 0 not in unique:
        return False
    if np.count_nonzero(tile == 0) < 2 or np.count_nonzero(tile != 0) < 2:
        return False
    if _minimal_square_period(tile) != side:
        return False

    out = _sampled_output(tile, k)
    if out.shape != (2, 2):
        return False
    if len(set(map(int, np.unique(out)))) != 2:
        return False

    alt = 3 if k == 2 else 2
    alt_out = _sampled_output(tile, alt)
    if alt_out.shape == out.shape and np.array_equal(alt_out, out):
        return False

    if np.all(tile[0, :] == tile[0, 0]) and np.all(tile[:, 0] == tile[0, 0]):
        return False
    return True


def _make_tile(side: int, color: int, k: int) -> np.ndarray:
    for _ in range(300):
        tile = np.random.choice([0, color], size=(side, side), p=[0.5, 0.5]).astype(int)
        if _tile_is_good(tile, k):
            return tile
    raise RuntimeError("failed to generate a non-degenerate base tile")


def _preferred_combo() -> tuple[int, int]:
    global _GEN_CALL_COUNT
    combos = ((3, 2), (4, 2), (4, 3))
    combo = combos[_GEN_CALL_COUNT % len(combos)]
    _GEN_CALL_COUNT += 1
    if random.random() < 0.35:
        combo = random.choice(combos)
    return combo


def _max_repeats(side: int, k: int) -> int:
    scaled = side * k
    return (MAX_GRID_SIDE - scaled - 3) // side


def _build_input(tile: np.ndarray, k: int, reps_r: int, reps_c: int) -> np.ndarray:
    side = int(tile.shape[0])
    scaled_side = side * k
    bottom_h = reps_r * side
    right_w = reps_c * side

    inner = np.zeros((scaled_side + 1 + bottom_h, scaled_side + 1 + right_w), dtype=int)
    inner[:scaled_side, :scaled_side] = _scale_tile(tile, k)
    inner[scaled_side, :] = GREY
    inner[:, scaled_side] = GREY

    inner[:scaled_side, scaled_side + 1 :] = _repeat_tile(tile, k, reps_c)
    inner[scaled_side + 1 :, :scaled_side] = _repeat_tile(tile, reps_r, k)
    inner[scaled_side + 1 :, scaled_side + 1 :] = _repeat_tile(tile, reps_r, reps_c)

    grid = np.full((inner.shape[0] + 2, inner.shape[1] + 2), GREY, dtype=int)
    grid[1:-1, 1:-1] = inner
    return grid


def _generator_checks(grid: np.ndarray, tile: np.ndarray, k: int) -> bool:
    try:
        solutions = _enumerate_solutions(grid)
    except Exception:
        return False
    if len(solutions) != 1:
        return False

    solved = main(grid)
    expected = _sampled_output(tile, k)
    if not np.array_equal(solved, expected):
        return False

    side, found_k, found_tile = solutions[0]
    if side != tile.shape[0] or found_k != k or not np.array_equal(found_tile, tile):
        return False

    arrays = _parse_structure(grid)[3]
    if any(_minimal_square_period(arr[:side, :side]) != side for arr in arrays):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        side, k = _preferred_combo()
        color = random.choice(FOREGROUND_CHOICES)
        tile = _make_tile(side, color, k)

        max_rep = _max_repeats(side, k)
        if max_rep < 2:
            continue
        reps_r = random.randint(2, max_rep)
        reps_c = random.randint(2, max_rep)

        grid = _build_input(tile, k, reps_r, reps_c)
        if max(grid.shape) > MAX_GRID_SIDE:
            continue
        if _generator_checks(grid, tile, k):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 8")
