from __future__ import annotations

import random
from typing import Literal

import numpy as np


BG = 0
SEP = 5
WORK_TOP = 2
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (15, 19)
GRID_W_RANGE = (15, 19)
PALETTE_LEN_RANGE = (4, 7)
SEED_COUNT_RANGE = (2, 4)
EXTRA_OBS_RANGE = (6, 12)
MIN_FALL = 2
GEN_ATTEMPTS = 600


def _read_palette(row: np.ndarray) -> np.ndarray:
    nz = np.flatnonzero(row != BG)
    assert len(nz) > 0 and int(nz[0]) == 0, "palette must start at the left edge"
    palette = np.asarray(row[: int(nz[-1]) + 1], dtype=int)
    assert int(palette[0]) != BG, "palette[0] must be non-background"
    return palette


def _parse_initial_seeds(grid: np.ndarray, seed_color: int) -> list[tuple[int, int]]:
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(grid[WORK_TOP:] == seed_color)]
    seeds = [(r + WORK_TOP, c) for r, c in seeds]
    assert 2 <= len(seeds) <= 4, "expected 2-4 seeds"
    assert len({c for _, c in seeds}) == len(seeds), "seed columns must be unique"
    return sorted(seeds, key=lambda rc: (rc[1], rc[0]))


def _landing_row(grid: np.ndarray, start_r: int, c: int, sep_color: int) -> int:
    below = np.flatnonzero(grid[start_r + 1 :, c] == sep_color)
    if len(below) == 0:
        return int(grid.shape[0] - 1)
    first_obstacle_r = int(start_r + 1 + below[0])
    return first_obstacle_r - 1


def _metric_distance(
    r: int,
    c: int,
    center_r: int,
    center_c: int,
    metric: Literal["chebyshev", "manhattan"],
) -> int:
    if metric == "chebyshev":
        return max(abs(r - center_r), abs(c - center_c))
    return abs(r - center_r) + abs(c - center_c)


def _render_from_centers(
    grid: np.ndarray,
    palette: np.ndarray,
    seed_color: int,
    centers: list[tuple[int, int]],
    *,
    metric: Literal["chebyshev", "manhattan"] = "chebyshev",
    collect_meta: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    seed_mask = np.zeros_like(out, dtype=bool)
    seed_mask[WORK_TOP:] = out[WORK_TOP:] == seed_color
    out[seed_mask] = BG
    for r, c in centers:
        out[r, c] = seed_color

    meta = {
        "overlap_cells": 0,
        "strict_conflict_cells": 0,
        "tie_cells": 0,
        "blank_winner_cells": 0,
        "winning_seeds": set(),
        "winner_distances": {},
    }

    radius = len(palette)
    H, W = out.shape
    for r in range(WORK_TOP, H):
        for c in range(W):
            if int(out[r, c]) != BG:
                continue

            candidates: list[tuple[int, int, int]] = []
            for idx, (cr, cc) in enumerate(centers):
                d = _metric_distance(r, c, cr, cc, metric)
                if d < radius:
                    candidates.append((d, cc, idx))

            if not candidates:
                continue

            candidates.sort()
            best_d, _best_col, best_idx = candidates[0]
            if len(candidates) >= 2:
                meta["overlap_cells"] += 1
                second_d = candidates[1][0]
                if second_d > best_d:
                    meta["strict_conflict_cells"] += 1
                if second_d == best_d:
                    meta["tie_cells"] += 1

            meta["winner_distances"][(r, c)] = best_d
            if int(palette[best_d]) == BG:
                meta["blank_winner_cells"] += 1
                continue

            out[r, c] = int(palette[best_d])
            meta["winning_seeds"].add(best_idx)

    if collect_meta:
        return out, meta
    return out


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    H, W = g.shape
    assert H >= 6 and W >= 6
    assert np.all(g[1] == SEP), "row 1 must be the separator row"

    palette = _read_palette(g[0])
    assert np.count_nonzero(palette) >= 2, "palette needs at least two visible entries"
    nonzero_palette = [int(v) for v in palette if int(v) != BG]
    assert len(nonzero_palette) == len(set(nonzero_palette)), "non-background palette colors must be distinct"
    assert SEP not in nonzero_palette, "palette cannot use separator color"

    seed_color = int(palette[0])
    work_colors = set(map(int, np.unique(g[WORK_TOP:])))
    assert work_colors <= {BG, SEP, seed_color}, "work area may only contain bg, obstacles, and seeds"

    initial_seeds = _parse_initial_seeds(g, seed_color)
    landed = []
    bottom_seed_count = 0
    stopped_seed_count = 0
    fall_distances = []
    for r, c in initial_seeds:
        land_r = _landing_row(g, r, c, SEP)
        assert land_r >= r, "seed cannot move upward"
        landed.append((land_r, c))
        fall_distances.append(land_r - r)
        if land_r == H - 1:
            bottom_seed_count += 1
        else:
            assert int(g[land_r + 1, c]) == SEP
            stopped_seed_count += 1

    out, meta = _render_from_centers(
        g,
        palette,
        seed_color,
        landed,
        metric="chebyshev",
        collect_meta=True,
    )
    meta.update(
        {
            "palette": palette,
            "seed_color": seed_color,
            "initial_seeds": initial_seeds,
            "landed_seeds": landed,
            "fall_distances": fall_distances,
            "bottom_seed_count": bottom_seed_count,
            "stopped_seed_count": stopped_seed_count,
        }
    )
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """Drop the suspended seeds onto obstacle-defined floors, then grow Chebyshev-distance rings from the landing sites using the top palette.

    Input structure:
    - Row 0 is a left-aligned palette whose entries index ring distance; row 1 is a full grey separator.
    - Rows 2..end form the work area with black background, scattered grey obstacle pixels, and 2-4 seed pixels.
    - All seeds share the palette's first color and occupy distinct columns.

    Rule:
    - Keep the palette row, separator row, and every grey obstacle unchanged.
    - Drop each seed straight downward until it stops immediately above the first grey obstacle in its column, or on the bottom row if none exists; erase its original location.
    - For every remaining background cell in the work area, compute its Chebyshev distance to every landed seed.
    - If the smallest such distance is d and d is within the palette, color the cell with palette[d]; leave it black when palette[d] is black. Break equal-distance ties by the smaller seed column.

    Key insight:
    - The obstacles do not just block paint: they first decide the ring centers by gravity, and only those landed centers make the later multi-source Chebyshev competition solvable.
    """
    out, _meta = _solve_with_meta(grid)
    return out


def _sample_palette(width: int) -> np.ndarray:
    L = random.randint(PALETTE_LEN_RANGE[0], min(PALETTE_LEN_RANGE[1], width))
    interior_positions = list(range(1, L - 1))
    zero_count = 0
    if interior_positions and random.random() < 0.55:
        zero_count = random.randint(1, min(2, len(interior_positions)))
    zero_positions = set(random.sample(interior_positions, zero_count)) if zero_count else set()

    nonzero_needed = L - zero_count
    colors = random.sample(COLOR_POOL, nonzero_needed)
    palette = []
    color_idx = 0
    for i in range(L):
        if i in zero_positions:
            palette.append(BG)
        else:
            palette.append(colors[color_idx])
            color_idx += 1
    return np.array(palette, dtype=int)


def _sample_columns(width: int, seed_count: int, radius: int, want_tie: bool) -> tuple[list[int], tuple[int, int] | None] | None:
    all_cols = list(range(width))
    for _ in range(120):
        cols = sorted(random.sample(all_cols, seed_count))
        if max(cols) - min(cols) > max(radius * 2 + 3, width - 1):
            continue
        overlap_pairs = [(i, j) for i in range(seed_count) for j in range(i + 1, seed_count) if cols[j] - cols[i] <= 2 * radius + 2]
        if not overlap_pairs:
            continue
        tie_pair = None
        if want_tie:
            tie_pairs = [(i, j) for i, j in overlap_pairs if (cols[j] - cols[i]) % 2 == 0]
            if not tie_pairs:
                continue
            tie_pair = random.choice(tie_pairs)
        return cols, tie_pair
    return None


def _place_extra_obstacles(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    target_count: int,
) -> bool:
    H, W = grid.shape
    placed = {
        (r, c)
        for r in range(WORK_TOP, H)
        for c in range(W)
        if int(grid[r, c]) == SEP
    }
    candidates = [
        (r, c)
        for r in range(WORK_TOP, H)
        for c in range(W)
        if (r, c) not in forbidden and int(grid[r, c]) == BG
    ]
    random.shuffle(candidates)

    count = 0
    for r, c in candidates:
        if any((r + dr, c + dc) in placed for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            continue
        grid[r, c] = SEP
        placed.add((r, c))
        count += 1
        if count == target_count:
            return True
    return count == target_count


def _semantic_ok(inp: np.ndarray, want_tie: bool) -> bool:
    try:
        out, meta = _solve_with_meta(inp)
    except AssertionError:
        return False

    palette = meta["palette"]
    if max(meta["fall_distances"]) < 3:
        return False
    if min(meta["fall_distances"]) < 1:
        return False
    if meta["stopped_seed_count"] < 1:
        return False
    if len(meta["winning_seeds"]) < 2:
        return False
    if meta["overlap_cells"] < 8 or meta["strict_conflict_cells"] < 4:
        return False
    if want_tie and meta["tie_cells"] < 1:
        return False
    if np.any(palette[1:] == BG) and meta["blank_winner_cells"] < 2:
        return False

    work = out[WORK_TOP:]
    paint_ratio = np.count_nonzero(work != BG) / work.size
    if not (0.12 <= paint_ratio <= 0.68):
        return False

    no_gravity = _render_from_centers(
        inp,
        palette,
        meta["seed_color"],
        meta["initial_seeds"],
        metric="chebyshev",
    )
    if np.array_equal(out, no_gravity):
        return False

    manhattan = _render_from_centers(
        inp,
        palette,
        meta["seed_color"],
        meta["landed_seeds"],
        metric="manhattan",
    )
    if np.array_equal(out, manhattan):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = np.zeros((H, W), dtype=int)
    grid[1, :] = SEP

    palette = _sample_palette(W)
    grid[0, : len(palette)] = palette
    seed_color = int(palette[0])
    radius = len(palette) - 1

    seed_count = random.randint(*SEED_COUNT_RANGE)
    want_tie = random.random() < 0.6
    col_pick = _sample_columns(W, seed_count, radius, want_tie)
    if col_pick is None:
        return None
    cols, tie_pair = col_pick

    landings: list[int | None] = [None] * seed_count
    uses_obstacle = [True] * seed_count

    if tie_pair is not None:
        shared_row = random.randint(WORK_TOP + 4, H - 2)
        i, j = tie_pair
        landings[i] = shared_row
        landings[j] = shared_row
        uses_obstacle[i] = True
        uses_obstacle[j] = True

    remaining = [idx for idx, v in enumerate(landings) if v is None]
    bottom_budget = random.randint(0, max(0, len(remaining) - 1))
    if seed_count >= 3 and random.random() < 0.55:
        bottom_budget = max(bottom_budget, 1)
    bottom_indices = set(random.sample(remaining, bottom_budget)) if bottom_budget else set()

    for idx in remaining:
        if idx in bottom_indices:
            landings[idx] = H - 1
            uses_obstacle[idx] = False
        else:
            landings[idx] = random.randint(WORK_TOP + 4, H - 2)
            uses_obstacle[idx] = True

    landings = [int(v) for v in landings]
    if len(set(landings)) < 2:
        return None

    forbidden: set[tuple[int, int]] = set()
    seed_columns = set(cols)
    for col in cols:
        for r in range(WORK_TOP, H):
            forbidden.add((r, col))

    for idx, (col, landing) in enumerate(zip(cols, landings)):
        if uses_obstacle[idx]:
            stopper = (landing + 1, col)
            grid[stopper] = SEP
            forbidden.discard(stopper)

    extra_obs = random.randint(EXTRA_OBS_RANGE[0], min(EXTRA_OBS_RANGE[1], (H - WORK_TOP) * W // 8))
    if not _place_extra_obstacles(grid, forbidden, extra_obs):
        return None

    seed_rows = []
    for col, landing in zip(cols, landings):
        max_start = landing - MIN_FALL
        if max_start < WORK_TOP:
            return None
        start_r = random.randint(WORK_TOP, max_start)
        seed_rows.append(start_r)
        grid[start_r, col] = seed_color

    if any(_landing_row(grid, r, c, SEP) != landing for (r, c), landing in zip(zip(seed_rows, cols), landings)):
        return None

    if not _semantic_ok(grid, want_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
