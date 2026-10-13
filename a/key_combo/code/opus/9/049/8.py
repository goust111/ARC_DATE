from __future__ import annotations

import hashlib
import random
from itertools import combinations

import numpy as np


BG = 0
WALL = 5
TEAL = 8
INPUT_SHAPE = (3, 4)
MOVABLE_COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 500
_SEEN_INPUT_SIGNATURES: set[str] = set()


def _seed_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BG) & (grid != WALL) & (grid != TEAL)


def _tile_2x2(grid: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(grid, dtype=int), (2, 2))


def _apply_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = g.shape

    for c in range(width):
        seed_rows = [int(r) for r in np.flatnonzero(_seed_mask(g[:, c]))]
        if not seed_rows:
            continue

        out[seed_rows, c] = BG
        wall_rows = [int(r) for r in np.flatnonzero(g[:, c] == WALL)]
        starts = [0] + [r + 1 for r in wall_rows]
        ends = [r - 1 for r in wall_rows] + [height - 1]

        for start, end in zip(starts, ends):
            if start > end:
                continue
            segment_rows = [r for r in seed_rows if start <= r <= end]
            if not segment_rows:
                continue

            colors = [int(g[r, c]) for r in sorted(segment_rows, reverse=True)]
            for offset, color in enumerate(colors):
                out[end - offset, c] = color

    return out


def _decorate_diagonals(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    height, width = out.shape

    for r, c in np.argwhere(_seed_mask(out)):
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < height and 0 <= cc < width and int(out[rr, cc]) == BG:
                out[rr, cc] = TEAL

    return out


def _diagonal_mark_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in np.argwhere(np.asarray(grid, dtype=int) == TEAL)}


def _diagonal_seed_pairs(grid: np.ndarray) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    seed_cells = {tuple(map(int, cell)) for cell in np.argwhere(_seed_mask(grid))}
    for r, c in seed_cells:
        for dc in (-1, 1):
            nbr = (r + 1, c + dc)
            if nbr in seed_cells:
                pair = tuple(sorted(((r, c), nbr)))
                pairs.add(pair)
    return pairs


def _count_seeds_by_column(grid: np.ndarray) -> dict[int, int]:
    coords = np.argwhere(_seed_mask(grid))
    if len(coords) == 0:
        return {}
    cols, counts = np.unique(coords[:, 1], return_counts=True)
    return {int(c): int(k) for c, k in zip(cols, counts)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile a seed-and-wall grid, let the tiled seeds fall under gravity, then teal-mark black diagonal neighbors.

    Input structure:
    - The input is a small black grid whose movable objects are single-pixel colored seeds.
    - Grey cells are fixed walls that do not move; teal does not appear as an input seed color.
    - In the intended task instances the base grid is 3x4, with seeds above the bottom-row walls.

    Rule:
    - First tile the whole input grid into a 2x2 arrangement.
    - In the tiled grid, move every non-black, non-grey, non-teal seed straight downward within its column.
    - Walls stay fixed, original seed cells become black, and seeds stop at the bottom, on top of a wall, or on top of an already settled seed.
    - After all seeds settle, inspect each settled seed's four diagonal neighbors and paint any black one teal without overwriting non-black cells.

    Key insight:
    - Tiling duplicates the seed layout before gravity, so the fall creates new column stacks and new diagonal relationships that only the final teal decoration reveals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tiled = _tile_2x2(g)
    fallen = _apply_gravity(tiled)
    return _decorate_diagonals(fallen)


def _input_is_valid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.shape != INPUT_SHAPE:
        return False
    if np.any((g < 0) | (g > 9)):
        return False
    if np.any(g[:-1, :] == WALL) or np.any(g[:-1, :] == TEAL):
        return False
    if np.any(g[-1, :] == TEAL):
        return False

    seed_positions = np.argwhere(_seed_mask(g))
    wall_positions = np.argwhere(g == WALL)
    if not (2 <= len(seed_positions) <= 3):
        return False
    if not (1 <= len(wall_positions) <= 2):
        return False
    if np.any(seed_positions[:, 0] == INPUT_SHAPE[0] - 1):
        return False
    if len({int(g[r, c]) for r, c in seed_positions}) != len(seed_positions):
        return False

    for wr, wc in wall_positions:
        if int(wr) != INPUT_SHAPE[0] - 1:
            return False
        for sr, sc in seed_positions:
            if max(abs(int(sr) - int(wr)), abs(int(sc) - int(wc))) <= 1:
                return False
    return True


def _semantic_example_is_good(grid: np.ndarray) -> bool:
    if not _input_is_valid(grid):
        return False

    tiled = _tile_2x2(grid)
    fallen = _apply_gravity(tiled)
    decorated_after_fall = _decorate_diagonals(fallen)
    decorated_without_fall = _decorate_diagonals(tiled)

    tiled_seed_cells = {tuple(map(int, cell)) for cell in np.argwhere(_seed_mask(tiled))}
    fallen_seed_cells = {tuple(map(int, cell)) for cell in np.argwhere(_seed_mask(fallen))}
    if tiled_seed_cells == fallen_seed_cells:
        return False
    if np.array_equal(tiled, fallen):
        return False
    if np.array_equal(fallen, decorated_after_fall):
        return False
    if np.array_equal(decorated_after_fall, decorated_without_fall):
        return False

    before_marks = _diagonal_mark_cells(decorated_without_fall)
    after_marks = _diagonal_mark_cells(decorated_after_fall)
    if not after_marks:
        return False
    if not (after_marks - before_marks):
        return False

    before_diag_pairs = _diagonal_seed_pairs(tiled)
    after_diag_pairs = _diagonal_seed_pairs(fallen)
    if not (after_diag_pairs - before_diag_pairs):
        return False

    seeds = np.argwhere(_seed_mask(grid))
    wall_cols = {int(c) for c in np.flatnonzero(grid[-1, :] == WALL)}
    seed_cols = [int(c) for _r, c in seeds]
    if not any(c in wall_cols for c in seed_cols):
        return False
    if not any(c not in wall_cols for c in seed_cols):
        return False

    tiled_col_counts = _count_seeds_by_column(fallen)
    if max(tiled_col_counts.values()) < 2:
        return False

    return True


def _candidate_seed_positions(wall_cols: set[int]) -> list[tuple[int, int]]:
    positions = []
    for r in range(INPUT_SHAPE[0] - 1):
        for c in range(INPUT_SHAPE[1]):
            if r == 1 and any(abs(c - wc) <= 1 for wc in wall_cols):
                continue
            positions.append((r, c))
    return positions


def _choose_positions(wall_cols: set[int], seed_count: int) -> list[tuple[int, int]] | None:
    candidates = _candidate_seed_positions(wall_cols)
    target_repeat_column = random.random() < 0.45
    target_has_row1 = random.random() < 0.6

    good: list[tuple[int, list[tuple[int, int]], tuple[int, int, int]]] = []
    for combo in combinations(candidates, seed_count):
        cols = [c for _r, c in combo]
        features = (
            int(len(set(cols)) < len(cols)),
            int(any(r == 1 for r, _c in combo)),
            int(any(c in wall_cols for _r, c in combo)),
        )

        if not any(c in wall_cols for _r, c in combo):
            continue
        if not any(c not in wall_cols for _r, c in combo):
            continue

        trial = np.zeros(INPUT_SHAPE, dtype=int)
        trial[-1, list(wall_cols)] = WALL
        for color, (r, c) in zip(MOVABLE_COLORS, combo):
            trial[r, c] = color

        if _semantic_example_is_good(trial):
            score = 0
            if bool(features[0]) == target_repeat_column:
                score += 2
            if bool(features[1]) == target_has_row1:
                score += 1
            score += features[2]
            good.append((score, list(combo), features))

    if not good:
        return None

    good.sort(key=lambda item: (item[0], item[2]), reverse=True)
    best_score = good[0][0]
    top = [combo for score, combo, _features in good if score == best_score]
    return random.choice(top)


def _generate_attempt() -> np.ndarray | None:
    grid = np.zeros(INPUT_SHAPE, dtype=int)

    wall_count = random.choice((1, 2))
    wall_cols = set(random.sample(range(INPUT_SHAPE[1]), wall_count))
    grid[-1, list(wall_cols)] = WALL

    seed_count = random.choice((2, 3))
    positions = _choose_positions(wall_cols, seed_count)
    if positions is None:
        return None

    colors = random.sample(MOVABLE_COLORS, seed_count)
    random.shuffle(positions)
    for color, (r, c) in zip(colors, positions):
        grid[r, c] = color

    if not _semantic_example_is_good(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is None:
            continue

        sig = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
        if sig in _SEEN_INPUT_SIGNATURES and len(_SEEN_INPUT_SIGNATURES) < 256:
            continue
        _SEEN_INPUT_SIGNATURES.add(sig)
        return grid
    raise RuntimeError("failed to generate a valid tiled-gravity ARC input")
