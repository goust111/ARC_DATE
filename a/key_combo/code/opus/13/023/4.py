from __future__ import annotations

import random

import numpy as np


BG = 3
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
MOVER_COLORS = (BLUE, RED, YELLOW, PINK)

GRID_H_RANGE = (13, 20)
GRID_W_RANGE = (13, 22)
TARGET_LEVELS = (1, 2, 3, 4, 5)
TARGET_LEVEL_WEIGHTS = (1, 3, 4, 3, 2)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _children_between_gapped_pairs(active: np.ndarray) -> np.ndarray:
    children = np.zeros_like(active, dtype=bool)
    children[1:-1] = active[:-2] & ~active[1:-1] & active[2:]
    return children


def _grow_barrier(seed_mask: np.ndarray, height: int, bg: int) -> np.ndarray:
    barrier = np.full((height, seed_mask.size), bg, dtype=int)
    active = seed_mask.astype(bool).copy()
    row = height - 1
    barrier[row, active] = ORANGE
    next_color = PINK

    while row - 2 >= 0:
        children = _children_between_gapped_pairs(active)
        if not np.any(children):
            break
        row -= 2
        barrier[row, children] = next_color
        active = children
        next_color = ORANGE if next_color == PINK else PINK

    return barrier


def _extract_movers(grid: np.ndarray, bg: int, seed_mask: np.ndarray) -> list[tuple[int, int, int]]:
    movers: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value == bg:
                continue
            if r == h - 1 and seed_mask[c] and value == ORANGE:
                continue
            movers.append((r, c, value))
    movers.sort(key=lambda item: (-item[0], item[1]))
    return movers


def _drop_movers(
    barrier: np.ndarray, movers: list[tuple[int, int, int]], bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    out = np.full_like(barrier, bg)
    barrier_mask = barrier != bg
    out[barrier_mask] = barrier[barrier_mask]

    landings: list[tuple[int, int, int, int, int]] = []
    for src_r, src_c, color in movers:
        dst_r = src_r
        while dst_r + 1 < out.shape[0] and int(out[dst_r + 1, src_c]) == bg:
            dst_r += 1
        out[dst_r, src_c] = color
        landings.append((src_r, src_c, dst_r, src_c, color))

    return out, landings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an alternating upward barrier from the bottom seed row, then drop loose colored pixels onto it.

    Input structure:
    - The grid has a green background, a bottom row of orange seed pixels, and several isolated single
      pixels in the upper area colored blue, red, yellow, or pink.
    - Seed pixels are separated by green gaps; at least some adjacent seeds have exactly one background
      cell between them, enabling upward midpoint growth every other row.
    - The scattered upper pixels are movable, while the grown structure becomes a fixed collision barrier.

    Rule:
    - Starting from the bottom seed row, repeatedly create child pixels two rows above each
      active-background-active triplet, coloring generations pink, orange, pink, ... until no children can be made.
    - Keep the seeds and all generated children as fixed barrier pixels.
    - Process the scattered pixels from lower rows to higher rows, and left-to-right within a row; each
      pixel falls straight down in its column until the cell below is occupied by a barrier or by an
      already-settled earlier pixel, otherwise it reaches the bottom row.

    Key insight:
    - The exact one-cell gaps among the bottom seeds determine where midpoint descendants appear, so the
      grown alternating scaffold defines the collision profile that later falling pixels stack against.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    h, w = g.shape

    seed_mask = g[h - 1] == ORANGE
    assert np.count_nonzero(seed_mask) >= 1, "expected orange seeds on the bottom row"

    barrier = _grow_barrier(seed_mask, h, bg)
    movers = _extract_movers(g, bg, seed_mask)
    out, _ = _drop_movers(barrier, movers, bg)
    return out


def _seed_positions_from_clusters(
    width: int, cluster_sizes: list[int], rng: random.Random
) -> list[int] | None:
    gaps_between = [rng.randint(3, 6) for _ in range(max(0, len(cluster_sizes) - 1))]
    span = sum(2 * size - 1 for size in cluster_sizes) + sum(gaps_between)
    if span > width:
        return None

    left_margin = rng.randint(0, width - span)
    positions: list[int] = []
    cursor = left_margin
    for i, size in enumerate(cluster_sizes):
        for j in range(size):
            positions.append(cursor + 2 * j)
        cursor += 2 * size - 1
        if i < len(gaps_between):
            cursor += gaps_between[i]
    return positions


def _make_seed_mask(width: int, levels: int, rng: random.Random) -> np.ndarray | None:
    main_cluster = levels + 1
    extra_clusters = rng.randint(0, 2)
    remaining_budget = max(0, 7 - main_cluster)
    cluster_sizes = [main_cluster]
    for _ in range(extra_clusters):
        if remaining_budget <= 0:
            break
        size = rng.randint(1, min(3, remaining_budget) + 1)
        cluster_sizes.append(size)
        remaining_budget -= size
    rng.shuffle(cluster_sizes)

    positions = _seed_positions_from_clusters(width, cluster_sizes, rng)
    if positions is None or len(positions) < 3:
        return None

    seed_mask = np.zeros(width, dtype=bool)
    seed_mask[positions] = True
    children = _children_between_gapped_pairs(seed_mask)
    if not np.any(children):
        return None
    return seed_mask


def _highest_barrier_row_per_col(barrier: np.ndarray, bg: int) -> list[int | None]:
    heights: list[int | None] = []
    for c in range(barrier.shape[1]):
        rows = np.flatnonzero(barrier[:, c] != bg)
        heights.append(int(rows[0]) if rows.size else None)
    return heights


def _is_isolated(coord: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    r, c = coord
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if (nr, nc) in used:
            return False
    return True


def _place_column_pixels(
    rows: list[int], col: int, count: int, used: set[tuple[int, int]], rng: random.Random
) -> list[tuple[int, int]] | None:
    choices = rows[:]
    rng.shuffle(choices)
    picked: list[tuple[int, int]] = []
    for r in sorted(choices):
        coord = (r, col)
        if not _is_isolated(coord, used):
            continue
        if any(abs(r - pr) == 1 for pr, pc in picked if pc == col):
            continue
        picked.append(coord)
        if len(picked) == count:
            return picked
    return None


def _landings_summary(
    barrier: np.ndarray, landings: list[tuple[int, int, int, int, int]], bg: int
) -> tuple[bool, bool, bool]:
    barrier_mask = barrier != bg
    open_branch = False
    barrier_branch = False
    stacked_branch = False
    occupied: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(barrier_mask)
    }
    for _src_r, _src_c, dst_r, dst_c, _color in landings:
        below = (dst_r + 1, dst_c)
        if dst_r == barrier.shape[0] - 1:
            open_branch = True
        elif below in occupied and barrier_mask[below]:
            barrier_branch = True
        elif below in occupied:
            stacked_branch = True
        occupied.add((dst_r, dst_c))
    return open_branch, barrier_branch, stacked_branch


def _build_input_candidate(rng: random.Random) -> np.ndarray | None:
    levels = rng.choices(TARGET_LEVELS, weights=TARGET_LEVEL_WEIGHTS, k=1)[0]
    min_h = max(GRID_H_RANGE[0], 2 * levels + 6)
    if min_h > GRID_H_RANGE[1]:
        return None
    height = rng.randint(min_h, GRID_H_RANGE[1])
    width = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])

    seed_mask = _make_seed_mask(width, levels, rng)
    if seed_mask is None:
        return None

    barrier = _grow_barrier(seed_mask, height, BG)
    top_rows = np.flatnonzero(np.any(barrier != BG, axis=1))
    top_barrier_row = int(top_rows[0])
    if top_barrier_row < 3:
        return None

    source_row_max = min(7, top_barrier_row - 1)
    if source_row_max < 2:
        return None
    source_rows = list(range(0, source_row_max + 1))

    barrier_top = _highest_barrier_row_per_col(barrier, BG)
    catch_cols = [
        c
        for c, top in enumerate(barrier_top)
        if top is not None and top < height - 1 and top > source_row_max
    ]
    if not catch_cols:
        catch_cols = [
            c for c, top in enumerate(barrier_top) if top is not None and top > source_row_max
        ]
    open_cols = [c for c, top in enumerate(barrier_top) if top is None]
    if not catch_cols or not open_cols:
        return None

    used: set[tuple[int, int]] = set()
    movers: list[tuple[int, int, int]] = []

    stack_col = rng.choice(catch_cols if rng.random() < 0.65 else open_cols)
    stack_count = rng.randint(2, 3)
    stacked = _place_column_pixels(source_rows, stack_col, stack_count, used, rng)
    if stacked is None:
        return None
    for coord in stacked:
        used.add(coord)
        movers.append((coord[0], coord[1], rng.choice(MOVER_COLORS)))

    extra_catch_col_choices = [c for c in catch_cols if c != stack_col]
    if extra_catch_col_choices:
        extra_catch = _place_column_pixels(source_rows, rng.choice(extra_catch_col_choices), 1, used, rng)
        if extra_catch is None:
            return None
        coord = extra_catch[0]
        used.add(coord)
        movers.append((coord[0], coord[1], rng.choice(MOVER_COLORS)))

    extra_open_col_choices = [c for c in open_cols if c != stack_col]
    if extra_open_col_choices:
        extra_open = _place_column_pixels(source_rows, rng.choice(extra_open_col_choices), 1, used, rng)
        if extra_open is None:
            return None
        coord = extra_open[0]
        used.add(coord)
        movers.append((coord[0], coord[1], rng.choice(MOVER_COLORS)))

    target_count = rng.randint(4, 8)
    all_cols = catch_cols + open_cols
    while len(movers) < target_count:
        col = rng.choice(all_cols)
        row = rng.choice(source_rows)
        coord = (row, col)
        if not _is_isolated(coord, used):
            continue
        used.add(coord)
        movers.append((row, col, rng.choice(MOVER_COLORS)))

    colors_used = {color for _, _, color in movers}
    if len(colors_used) == 1:
        idx = rng.randrange(len(movers))
        row, col, color = movers[idx]
        alternatives = [c for c in MOVER_COLORS if c != color]
        movers[idx] = (row, col, rng.choice(alternatives))

    grid = np.full((height, width), BG, dtype=int)
    grid[height - 1, seed_mask] = ORANGE
    for row, col, color in movers:
        if barrier[row, col] != BG:
            return None
        grid[row, col] = color

    out, landings = _drop_movers(barrier, _extract_movers(grid, BG, seed_mask), BG)
    open_branch, barrier_branch, stacked_branch = _landings_summary(barrier, landings, BG)
    if not (open_branch and barrier_branch and stacked_branch):
        return None

    if np.array_equal(grid, out):
        return None
    if np.count_nonzero(grid == ORANGE) < 3:
        return None

    bottom_diffs = np.count_nonzero(out[-1] != barrier[-1])
    if bottom_diffs < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
