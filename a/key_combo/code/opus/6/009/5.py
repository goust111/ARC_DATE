import random
from typing import Iterable

import numpy as np


class Color:
    BLACK = 0
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    ORANGE = 7
    TEAL = 8


BG = Color.ORANGE
RAY = Color.TEAL
PALETTE = (Color.GREEN, Color.YELLOW, Color.BLACK, Color.GREY, Color.RED)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _paint_rays(shape: tuple[int, int], horiz_rows: list[int], vert_cols: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    for r in horiz_rows:
        out[int(r), :] = RAY
    for c in vert_cols:
        out[:, int(c)] = RAY
    return out


def _distance_colors(
    shape: tuple[int, int],
    seeds: list[tuple[int, int]],
    ray_mask: np.ndarray,
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    rr, cc = np.indices((h, w))
    best = np.full((h, w), h + w + 10, dtype=int)
    for sr, sc in seeds:
        best = np.minimum(best, np.abs(rr - int(sr)) + np.abs(cc - int(sc)))
    non_ray = ~ray_mask
    out[non_ray] = np.take(np.array(PALETTE, dtype=int), best[non_ray] % len(PALETTE))
    out[ray_mask] = RAY
    return out


def _parse_markers(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    h, w = map(int, grid.shape)

    top = [(0, c) for c in range(1, w - 1) if int(grid[0, c]) == RAY]
    bottom = [(h - 1, c) for c in range(1, w - 1) if int(grid[h - 1, c]) == RAY]
    left = [(r, 0) for r in range(1, h - 1) if int(grid[r, 0]) == RAY]
    right = [(r, w - 1) for r in range(1, h - 1) if int(grid[r, w - 1]) == RAY]

    markers = top + bottom + left + right
    vert_cols = sorted({c for _r, c in top + bottom})
    horiz_rows = sorted({r for r, _c in left + right})

    return markers, horiz_rows, vert_cols


def _validate_input_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert 3 <= h <= 30 and 3 <= w <= 30
    assert np.all((g == BG) | (g == RAY))

    markers, horiz_rows, vert_cols = _parse_markers(g)
    assert 4 <= len(markers) <= 8
    assert horiz_rows and vert_cols
    assert len(horiz_rows) == len({int(r) for r in horiz_rows})
    assert len(vert_cols) == len({int(c) for c in vert_cols})
    assert all(
        _in_bounds(g.shape, r, c)
        and (r in (0, h - 1) or c in (0, w - 1))
        and (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        for r, c in markers
    )
    assert int(g[0, 0]) == BG and int(g[0, w - 1]) == BG
    assert int(g[h - 1, 0]) == BG and int(g[h - 1, w - 1]) == BG

    for r in range(1, h - 1):
        assert not (int(g[r, 0]) == RAY and int(g[r, w - 1]) == RAY)
    for c in range(1, w - 1):
        assert not (int(g[0, c]) == RAY and int(g[h - 1, c]) == RAY)

    return markers, horiz_rows, vert_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw full rays from edge markers and color the remaining cells by Manhattan distance to the ray intersections.

    Input structure:
    - The grid is an orange background with teal marker pixels placed on the border but never on corners.
    - Top or bottom markers determine vertical teal rays; left or right markers determine horizontal teal rays.
    - No two vertical-ray markers share a column, and no two horizontal-ray markers share a row.

    Rule:
    - Extend each top or bottom marker into a full-height teal column, and each left or right marker into a full-width teal row.
    - Every crossing between one horizontal ray and one vertical ray is an interior seed point.
    - Keep every teal ray cell unchanged, including crossings.
    - Recolor every non-teal cell by the Manhattan distance D to its nearest crossing: D mod 5 = 0 -> green, 1 -> yellow, 2 -> black, 3 -> grey, 4 -> red.

    Key insight:
    - The distance-field seeds are not the input markers themselves; they must first be inferred as the intersections created after the edge markers are expanded into full rays.
    """
    g = np.asarray(grid, dtype=int)
    _markers, horiz_rows, vert_cols = _validate_input_structure(g)

    ray_grid = _paint_rays(g.shape, horiz_rows, vert_cols)
    seeds = [(r, c) for r in horiz_rows for c in vert_cols]
    assert seeds
    return _distance_colors(g.shape, seeds, ray_grid == RAY)


def _sample_positions(limit: int, count: int, rng: random.Random) -> list[int] | None:
    positions = list(range(1, int(limit) - 1))
    if len(positions) < count:
        return None
    for gap in (3, 2, 1):
        for _ in range(60):
            chosen = sorted(rng.sample(positions, count))
            if all(chosen[i + 1] - chosen[i] >= gap for i in range(len(chosen) - 1)):
                return chosen
    return None


def _split_between_sides(
    coords: list[int],
    rng: random.Random,
    force_both_prob: float,
) -> tuple[list[int], list[int]]:
    if len(coords) == 1:
        if rng.random() < 0.5:
            return [coords[0]], []
        return [], [coords[0]]

    use_both = rng.random() < force_both_prob
    if use_both:
        split = list(coords)
        rng.shuffle(split)
        pivot = rng.randint(1, len(split) - 1)
        return split[:pivot], split[pivot:]

    if rng.random() < 0.5:
        return list(coords), []
    return [], list(coords)


def _alt_output_from_marker_seeds(grid: np.ndarray) -> np.ndarray:
    markers, horiz_rows, vert_cols = _validate_input_structure(grid)
    ray_grid = _paint_rays(grid.shape, horiz_rows, vert_cols)
    return _distance_colors(grid.shape, markers, ray_grid == RAY)


def _alt_output_from_ray_seeds(grid: np.ndarray) -> np.ndarray:
    _markers, horiz_rows, vert_cols = _validate_input_structure(grid)
    ray_grid = _paint_rays(grid.shape, horiz_rows, vert_cols)
    ray_coords = [tuple(map(int, rc)) for rc in np.argwhere(ray_grid == RAY)]
    return _distance_colors(grid.shape, ray_coords, ray_grid == RAY)


def _distance_residues(shape: tuple[int, int], seeds: Iterable[tuple[int, int]], ray_mask: np.ndarray) -> set[int]:
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    best = np.full((h, w), h + w + 10, dtype=int)
    for sr, sc in seeds:
        best = np.minimum(best, np.abs(rr - int(sr)) + np.abs(cc - int(sc)))
    return {int(v) % 5 for v in best[~ray_mask].tolist()}


def _is_semantically_valid(grid: np.ndarray) -> bool:
    try:
        markers, horiz_rows, vert_cols = _validate_input_structure(grid)
    except AssertionError:
        return False

    ray_grid = _paint_rays(grid.shape, horiz_rows, vert_cols)
    ray_mask = ray_grid == RAY
    seeds = [(r, c) for r in horiz_rows for c in vert_cols]
    if len(seeds) < 2:
        return False
    if np.count_nonzero(~ray_mask) < 40:
        return False
    if np.count_nonzero(ray_mask) * 2 >= grid.size:
        return False
    if _distance_residues(grid.shape, seeds, ray_mask) != {0, 1, 2, 3, 4}:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if len(np.unique(out)) < 6:
        return False
    if np.array_equal(out, _alt_output_from_marker_seeds(grid)):
        return False
    if np.array_equal(out, _alt_output_from_ray_seeds(grid)):
        return False

    expected_markers = {tuple(map(int, rc)) for rc in markers}
    actual_markers = {tuple(map(int, rc)) for rc in np.argwhere(grid == RAY)}
    if expected_markers != actual_markers:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(16, 23)
        w = rng.randint(16, 23)

        total = rng.randint(4, 8)
        possible_splits = [
            (n_vert, total - n_vert)
            for n_vert in range(1, min(4, total - 1) + 1)
            if 1 <= total - n_vert <= 4
        ]
        n_vert, n_horiz = rng.choice(possible_splits)
        if n_vert == 1 or n_horiz == 1:
            if rng.random() < 0.6:
                continue

        cols = _sample_positions(w, n_vert, rng)
        rows = _sample_positions(h, n_horiz, rng)
        if cols is None or rows is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        top_cols, bottom_cols = _split_between_sides(cols, rng, force_both_prob=0.65)
        left_rows, right_rows = _split_between_sides(rows, rng, force_both_prob=0.65)

        for c in top_cols:
            grid[0, int(c)] = RAY
        for c in bottom_cols:
            grid[h - 1, int(c)] = RAY
        for r in left_rows:
            grid[int(r), 0] = RAY
        for r in right_rows:
            grid[int(r), w - 1] = RAY

        if _is_semantically_valid(grid):
            return grid

    raise RuntimeError("generate_input: failed to build a valid ray-intersection distance-field task")
