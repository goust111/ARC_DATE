from __future__ import annotations

import random

import numpy as np


BG = 0
AVAILABLE_COLORS = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
GRID_SIZE_RANGE = (20, 25)
LEGEND_COLOR_COUNT_RANGE = (3, 5)
LEGEND_GAP = 1
MIN_CORE_GAP = 3
EXTRA_TOWER_COUNT_RANGE = (0, 2)
GEN_RETRY_LIMIT = 400


def _legend_rows_and_priority(grid: np.ndarray) -> tuple[list[int], dict[int, int]]:
    """Read the left-edge legend as a top-to-bottom priority table."""
    h, _w = grid.shape
    rows = [r for r in range(h) if int(grid[r, 0]) != BG]
    assert 3 <= len(rows) <= 5, "expected 3-5 legend entries in the left column"

    colors = [int(grid[r, 0]) for r in rows]
    assert len(set(colors)) == len(colors), "legend colors must be unique"

    priority = {color: idx for idx, color in enumerate(colors)}
    return rows, priority


def _find_towers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != BG)
    towers: list[tuple[int, int, int]] = []
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        if c == 0:
            continue
        towers.append((r, c, int(grid[r, c])))
    assert towers, "expected at least one signal tower outside the legend column"
    return towers


def _signal_overlay(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Cast rays from every tower using legend priority for overlap resolution.

    Returns:
    - best_color: winning color for each paintable cell
    - color_bits: bitset of all colors that reached each cell
    - blocked_rays: number of rays that stopped at another tower
    - edge_rays: number of rays that ran to the main-area boundary
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    _legend_rows, priority = _legend_rows_and_priority(g)
    towers = _find_towers(g)
    tower_set = {(r, c) for r, c, _color in towers}
    assert all(color in priority for _r, _c, color in towers), "tower color missing from legend"

    best_rank = np.full((h, w), len(priority) + 1, dtype=int)
    best_color = np.zeros((h, w), dtype=int)
    color_bits = np.zeros((h, w), dtype=np.int64)
    blocked_rays = 0
    edge_rays = 0

    for r, c, color in towers:
        rank = priority[color]
        bit = 1 << color
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            while 0 <= rr < h and 1 <= cc < w:
                if (rr, cc) in tower_set:
                    blocked_rays += 1
                    break
                if int(g[rr, cc]) == BG:
                    color_bits[rr, cc] |= bit
                    if rank < int(best_rank[rr, cc]):
                        best_rank[rr, cc] = rank
                        best_color[rr, cc] = color
                rr += dr
                cc += dc
            else:
                edge_rays += 1

    return best_color, color_bits, blocked_rays, edge_rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the left-column legend as signal priority and broadcast cross rays.

    Input structure:
    - The background is black.
    - The leftmost column contains a vertical color legend; higher entries have
      higher priority.
    - Every non-black pixel outside the leftmost column is a single-pixel signal
      tower whose color appears in the legend.

    Rule:
    - Each tower emits rays up, down, left, and right.
    - Rays paint only original black cells.
    - A ray stops at the main-area boundary or just before another original
      tower.
    - If multiple colors reach one cell, keep the color whose legend entry is
      higher.
    - Tower pixels and the entire legend column remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    best_color, _color_bits, _blocked_rays, _edge_rays = _signal_overlay(g)

    out = g.copy()
    paint_mask = (g == BG) & (best_color != BG)
    paint_mask[:, 0] = False
    out[paint_mask] = best_color[paint_mask]
    return out


def _sample_spaced_positions(
    candidates: list[int],
    count: int,
    min_gap: int,
    retries: int = 200,
) -> list[int] | None:
    if len(candidates) < count:
        return None
    for _ in range(retries):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _is_far_enough(
    r: int,
    c: int,
    occupied: set[tuple[int, int]],
    moat: int = 1,
) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) > moat for rr, cc in occupied)


def _find_isolated_spot(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    legend_rows: set[int],
    blocked_rows: set[int],
    blocked_cols: set[int],
) -> tuple[int, int] | None:
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(2, w - 1)
        if r not in legend_rows
        and r not in blocked_rows
        and c not in blocked_cols
        and (r, c) not in occupied
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        if _is_far_enough(r, c, occupied):
            return r, c
    return None


def _count_overlap_pixels(color_bits: np.ndarray) -> int:
    return sum(int(value).bit_count() > 1 for value in color_bits.flat)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)

    n_colors = random.randint(*LEGEND_COLOR_COUNT_RANGE)
    legend_colors = random.sample(AVAILABLE_COLORS, n_colors)

    legend_height = n_colors + (n_colors - 1) * LEGEND_GAP
    start_max = h - legend_height
    if start_max < 0:
        return None
    legend_start = random.randint(0, start_max)
    legend_rows = [legend_start + 2 * idx for idx in range(n_colors)]
    legend_row_set = set(legend_rows)

    row_candidates = [r for r in range(1, h - 1) if r not in legend_row_set]
    col_candidates = list(range(2, w - 1))
    core_rows = _sample_spaced_positions(row_candidates, 3, MIN_CORE_GAP)
    core_cols = _sample_spaced_positions(col_candidates, 3, MIN_CORE_GAP)
    if core_rows is None or core_cols is None:
        return None

    top_row, mid_row, bot_row = core_rows
    left_col, mid_col, right_col = core_cols

    if random.random() < 0.5:
        r1, r2, r4 = top_row, mid_row, bot_row
    else:
        r1, r2, r4 = bot_row, mid_row, top_row

    if random.random() < 0.5:
        c1, c2, c3 = left_col, mid_col, right_col
    else:
        c1, c2, c3 = right_col, mid_col, left_col

    core_palette = random.sample(legend_colors, len(legend_colors))
    color_a, color_b, color_c = core_palette[:3]
    if len(core_palette) >= 4 and random.random() < 0.5:
        color_d = core_palette[3]
    else:
        color_d = random.choice([color_a, color_b, color_c])

    towers: list[tuple[int, int, int]] = [
        (r1, c1, color_a),
        (r2, c2, color_b),
        (r1, c3, color_c),
        (r4, c2, color_d),
    ]
    occupied = {(r, c) for r, c, _color in towers}
    used_colors = {color for _r, _c, color in towers}

    blocked_rows = {r1, r2, r4}
    blocked_cols = {c1, c2, c3}

    for color in legend_colors:
        if color in used_colors:
            continue
        spot = _find_isolated_spot(h, w, occupied, legend_row_set, blocked_rows, blocked_cols)
        if spot is None:
            return None
        towers.append((spot[0], spot[1], color))
        occupied.add(spot)
        used_colors.add(color)

    extra_tower_count = random.randint(*EXTRA_TOWER_COUNT_RANGE)
    for _ in range(extra_tower_count):
        spot = _find_isolated_spot(h, w, occupied, legend_row_set, blocked_rows, blocked_cols)
        if spot is None:
            break
        towers.append((spot[0], spot[1], random.choice(legend_colors)))
        occupied.add(spot)

    grid = np.zeros((h, w), dtype=int)
    for row, color in zip(legend_rows, legend_colors):
        grid[row, 0] = color
    for r, c, color in towers:
        grid[r, c] = color

    try:
        out = main(grid)
        best_color, color_bits, blocked_rays, edge_rays = _signal_overlay(grid)
    except AssertionError:
        return None

    painted_pixels = int(np.count_nonzero(best_color))
    overlap_pixels = _count_overlap_pixels(color_bits)
    changed_pixels = int(np.count_nonzero(out != grid))

    if painted_pixels != changed_pixels:
        return None
    if overlap_pixels < 2:
        return None
    if blocked_rays < 2 or edge_rays < 2:
        return None
    if changed_pixels < 12:
        return None
    if len(used_colors) != len(legend_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRY_LIMIT} attempts")
