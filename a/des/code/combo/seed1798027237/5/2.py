import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
SIZE_CHOICES = (9, 11, 13, 15, 17)
MAX_STEPS = 2
MAX_GENERATION_ATTEMPTS = 400


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    row_candidates = [
        r
        for r in range(h)
        if np.all((grid[r] == GREY) | (grid[r] == MAROON))
        and int(np.count_nonzero(grid[r] == MAROON)) == 1
    ]
    col_candidates = [
        c
        for c in range(w)
        if np.all((grid[:, c] == GREY) | (grid[:, c] == MAROON))
        and int(np.count_nonzero(grid[:, c] == MAROON)) == 1
    ]
    assert len(row_candidates) == 1, "expected one horizontal divider"
    assert len(col_candidates) == 1, "expected one vertical divider"
    mid_r = int(row_candidates[0])
    mid_c = int(col_candidates[0])
    assert int(grid[mid_r, mid_c]) == GREY, "divider intersection must stay grey"
    return mid_r, mid_c


def _decode_direction(grid: np.ndarray, mid_r: int, mid_c: int) -> tuple[int, int]:
    h_marks = np.flatnonzero(grid[mid_r] == MAROON)
    v_marks = np.flatnonzero(grid[:, mid_c] == MAROON)
    assert len(h_marks) == 1 and int(h_marks[0]) != mid_c, "expected one horizontal arrow marker"
    assert len(v_marks) == 1 and int(v_marks[0]) != mid_r, "expected one vertical arrow marker"
    dc = -1 if int(h_marks[0]) < mid_c else 1
    dr = -1 if int(v_marks[0]) < mid_r else 1
    return dr, dc


def _blocked(r: int, c: int, h: int, w: int, mid_r: int, mid_c: int) -> bool:
    return not (0 <= r < h and 0 <= c < w) or r == mid_r or c == mid_c


def _grow_l_cells(
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    h: int,
    w: int,
    mid_r: int,
    mid_c: int,
) -> tuple[list[tuple[int, int]], int, int]:
    cells = [(start_r, start_c)]
    end_r, end_c = start_r, start_c
    h_steps = 0
    for _ in range(MAX_STEPS):
        next_r, next_c = end_r, end_c + dc
        if _blocked(next_r, next_c, h, w, mid_r, mid_c):
            break
        cells.append((next_r, next_c))
        end_r, end_c = next_r, next_c
        h_steps += 1

    v_steps = 0
    for _ in range(MAX_STEPS):
        next_r, next_c = end_r + dr, end_c
        if _blocked(next_r, next_c, h, w, mid_r, mid_c):
            break
        cells.append((next_r, next_c))
        end_r, end_c = next_r, next_c
        v_steps += 1

    return cells, h_steps, v_steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into an L-shaped branch using the direction encoded by
    maroon markers on the grey dividers. Branches stop before crossing a divider
    or leaving the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    mid_r, mid_c = _find_dividers(g)
    dr, dc = _decode_direction(g, mid_r, mid_c)

    seed_mask = (g != BLACK) & (g != GREY) & (g != MAROON)
    seeds = np.argwhere(seed_mask)
    out = g.copy()

    for r, c in seeds.tolist():
        r = int(r)
        c = int(c)
        assert r != mid_r and c != mid_c, "seed cannot lie on a divider"
        color = int(g[r, c])
        cells, _h_steps, _v_steps = _grow_l_cells(r, c, dr, dc, h, w, mid_r, mid_c)
        for rr, cc in cells:
            out[rr, cc] = color

    return out


def _quadrant_id(r: int, c: int, mid_r: int, mid_c: int) -> int:
    if r < mid_r and c < mid_c:
        return 0
    if r < mid_r and c > mid_c:
        return 1
    if r > mid_r and c < mid_c:
        return 2
    assert r > mid_r and c > mid_c
    return 3


def _paint_neighborhood(cells: tuple[tuple[int, int], ...], h: int, w: int) -> set[tuple[int, int]]:
    neighborhood: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    neighborhood.add((rr, cc))
    return neighborhood


def _candidate_buckets(
    h: int, w: int, mid_r: int, mid_c: int, dr: int, dc: int
) -> dict[int, dict[str, list[dict]]]:
    buckets = {
        quadrant: {"clear": [], "hblock": [], "vblock": []}
        for quadrant in range(4)
    }
    for r in range(h):
        if r == mid_r:
            continue
        for c in range(w):
            if c == mid_c:
                continue
            cells, h_steps, v_steps = _grow_l_cells(r, c, dr, dc, h, w, mid_r, mid_c)
            category = "clear"
            if h_steps < MAX_STEPS:
                category = "hblock"
            elif v_steps < MAX_STEPS:
                category = "vblock"
            buckets[_quadrant_id(r, c, mid_r, mid_c)][category].append(
                {
                    "seed": (r, c),
                    "cells": tuple(cells),
                    "added": len(cells) - 1,
                    "category": category,
                }
            )
    return buckets


def _place_dividers(grid: np.ndarray, mid_r: int, mid_c: int, dr: int, dc: int) -> None:
    h, w = grid.shape
    grid[mid_r, :] = GREY
    grid[:, mid_c] = GREY

    if dc < 0:
        h_marker_c = random.randint(0, mid_c - 1)
    else:
        h_marker_c = random.randint(mid_c + 1, w - 1)
    if dr < 0:
        v_marker_r = random.randint(0, mid_r - 1)
    else:
        v_marker_r = random.randint(mid_r + 1, h - 1)

    grid[mid_r, h_marker_c] = MAROON
    grid[v_marker_r, mid_c] = MAROON


def _choose_seed_layout(
    buckets: dict[int, dict[str, list[dict]]], h: int, w: int, n_seeds: int
) -> list[dict] | None:
    quadrants = random.sample(range(4), n_seeds)
    quadrants.sort(
        key=lambda quadrant: sum(len(items) for items in buckets[quadrant].values())
    )

    placements: list[dict] = []
    forbidden: set[tuple[int, int]] = set()
    categories_used: set[str] = set()

    for quadrant in quadrants:
        category_names = [name for name, items in buckets[quadrant].items() if items]
        random.shuffle(category_names)
        chosen = None
        for category in category_names:
            compatible = [
                cand
                for cand in buckets[quadrant][category]
                if all(cell not in forbidden for cell in cand["cells"])
            ]
            if compatible:
                chosen = random.choice(compatible)
                break
        if chosen is None:
            return None
        placements.append(chosen)
        categories_used.add(chosen["category"])
        forbidden.update(_paint_neighborhood(chosen["cells"], h, w))

    if not any(item["added"] > 0 for item in placements):
        return None

    has_truncation_available = any(
        buckets[quadrant]["hblock"] or buckets[quadrant]["vblock"] for quadrant in quadrants
    )
    if has_truncation_available and categories_used == {"clear"} and random.random() < 0.7:
        return None

    return placements


def _generate_attempt() -> np.ndarray | None:
    h = random.choice(SIZE_CHOICES)
    w = random.choice(SIZE_CHOICES)
    mid_r = h // 2
    mid_c = w // 2
    dr, dc = random.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))
    n_seeds = random.randint(1, 3)

    buckets = _candidate_buckets(h, w, mid_r, mid_c, dr, dc)
    placements = _choose_seed_layout(buckets, h, w, n_seeds)
    if placements is None:
        return None

    colors = random.sample(SEED_COLORS, n_seeds)
    grid = np.full((h, w), BLACK, dtype=int)
    _place_dividers(grid, mid_r, mid_c, dr, dc)

    for color, placement in zip(colors, placements, strict=False):
        r, c = placement["seed"]
        assert int(grid[r, c]) == BLACK
        grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
