from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
PALETTE = tuple(range(10))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver helpers ---------------


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            seen[r, c] = True
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _extract_objects(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg != GREY, "background must differ from grey wall color"

    wall_mask = g == GREY
    seed_cells = []
    for r, c in np.argwhere((g != bg) & (g != GREY)):
        seed_cells.append((int(r), int(c), int(g[r, c])))
    seed_cells.sort(key=lambda item: (item[0], item[1]))

    assert len(seed_cells) == 2, "expected exactly two colored singleton seeds"
    assert seed_cells[0][2] != seed_cells[1][2], "seed colors must be distinct"

    other_colors = {int(v) for v in np.unique(g) if int(v) not in {bg, GREY}}
    assert len(other_colors) == 2, "only the two seed colors may appear besides background and walls"
    for color in other_colors:
        pts = np.argwhere(g == color)
        assert pts.shape[0] == 1, "each seed color must appear exactly once"

    return g, bg, wall_mask, seed_cells


def _bfs_distances(wall_mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = wall_mask.shape
    sr, sc = start
    dist = np.full((h, w), -1, dtype=int)
    if wall_mask[sr, sc]:
        return dist

    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not wall_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _paint_from_distances(
    grid: np.ndarray,
    bg: int,
    seed_cells: list[tuple[int, int, int]],
    dist_a: np.ndarray,
    dist_b: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    out = np.array(grid, copy=True)
    bg_mask = grid == bg

    big = grid.size + 1
    da = np.where(dist_a >= 0, dist_a, big)
    db = np.where(dist_b >= 0, dist_b, big)

    closer_a = bg_mask & (da < db)
    closer_b = bg_mask & (db < da)
    even_a = closer_a & ((da % 2) == 0)
    even_b = closer_b & ((db % 2) == 0)
    odd_a = closer_a & ((da % 2) == 1)
    odd_b = closer_b & ((db % 2) == 1)
    tie = bg_mask & (dist_a >= 0) & (dist_b >= 0) & (dist_a == dist_b)

    out[even_a] = seed_cells[0][2]
    out[even_b] = seed_cells[1][2]
    return out, {
        "closer_a": closer_a,
        "closer_b": closer_b,
        "even_a": even_a,
        "even_b": even_b,
        "odd_a": odd_a,
        "odd_b": odd_b,
        "tie": tie,
    }


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g, bg, wall_mask, seed_cells = _extract_objects(grid)
    dist_a = _bfs_distances(wall_mask, seed_cells[0][:2])
    dist_b = _bfs_distances(wall_mask, seed_cells[1][:2])
    out, masks = _paint_from_distances(g, bg, seed_cells, dist_a, dist_b)
    return {
        "grid": g,
        "bg": bg,
        "wall_mask": wall_mask,
        "seed_cells": seed_cells,
        "dist_a": dist_a,
        "dist_b": dist_b,
        "output": out,
        **masks,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint two obstacle-aware ripple fields, keeping only even-distance cells owned by the nearer seed.

    Input structure:
    - Background is the most common color.
    - Grey pixels form one or more wall segments that block four-neighbor travel.
    - Exactly two other colors appear, each as a single isolated seed pixel.

    Rule:
    - For every background cell, compute its four-connected shortest-path distance to each seed while forbidding paths through grey wall cells.
    - If one seed is strictly nearer and that shorter distance is even, recolor the cell with that seed's color.
    - If the shorter distance is odd, or if the two seed distances are equal, leave the cell as background.
    - Keep the grey walls and the two seed pixels unchanged.

    Key insight:
    - The visible pattern is the overlap of two obstacle-bent geodesic Voronoi regions with an even/odd distance filter, so walls warp the rings and ties erase the midline.
    """
    return _analyze_grid(np.asarray(grid, dtype=int))["output"]


# --------------- generator helpers ---------------

GRID_H_RANGE = (13, 23)
GRID_W_RANGE = (15, 27)
MAX_GENERATION_ATTEMPTS = 500


def _sample_barrier_parts(inner_h: int, gap_count: int) -> tuple[list[int], list[int]]:
    options = []
    if gap_count == 1:
        for seg1 in range(2, inner_h - 2):
            for gap1 in range(1, 4):
                seg2 = inner_h - seg1 - gap1
                if seg2 >= 2:
                    options.append(([seg1, seg2], [gap1]))
    else:
        for seg1 in range(2, inner_h - 5):
            for gap1 in range(1, 4):
                for seg2 in range(2, inner_h - seg1 - gap1 - 2):
                    for gap2 in range(1, 4):
                        seg3 = inner_h - seg1 - gap1 - seg2 - gap2
                        if seg3 >= 2:
                            options.append(([seg1, seg2, seg3], [gap1, gap2]))
    return random.choice(options)


def _build_main_barrier(h: int, w: int) -> tuple[np.ndarray, dict[str, object]]:
    wall_mask = np.zeros((h, w), dtype=bool)
    thickness = random.randint(1, 2)
    c0_min = max(3, w // 3)
    c0_max = min(w - thickness - 4, (2 * w) // 3)
    c0 = random.randint(c0_min, c0_max)
    gap_count = random.choice((1, 2))
    segments, gaps = _sample_barrier_parts(h - 2, gap_count)

    gap_rows = []
    cursor = 1
    for i, seg_len in enumerate(segments):
        wall_mask[cursor : cursor + seg_len, c0 : c0 + thickness] = True
        cursor += seg_len
        if i < len(gaps):
            gap_rows.append((cursor, cursor + gaps[i] - 1))
            cursor += gaps[i]

    bands = []
    band_start = 0
    for gap_start, gap_end in gap_rows:
        bands.append((band_start, gap_start - 1))
        band_start = gap_end + 1
    bands.append((band_start, h - 1))
    bands = [band for band in bands if band[0] <= band[1]]

    return wall_mask, {
        "c0": c0,
        "thickness": thickness,
        "bands": bands,
    }


def _pick_seed_rows(bands: list[tuple[int, int]]) -> tuple[int, int]:
    if len(bands) == 2:
        left_band, right_band = random.choice(((0, 1), (1, 0)))
    else:
        options = ((0, 2), (2, 0), (0, 1), (1, 2), (1, 0), (2, 1))
        left_band, right_band = random.choice(options)
    left_lo, left_hi = bands[left_band]
    right_lo, right_hi = bands[right_band]
    return random.randint(left_lo, left_hi), random.randint(right_lo, right_hi)


def _reserve_seed_neighborhood(seeds: list[tuple[int, int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    reserved: set[tuple[int, int]] = set()
    for r, c, _ in seeds:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    reserved.add((rr, cc))
    return reserved


def _segment_cells(r0: int, c0: int, dr: int, dc: int, length: int) -> list[tuple[int, int]]:
    return [(r0 + i * dr, c0 + i * dc) for i in range(length)]


def _can_place_extra(mask: np.ndarray, cells: list[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = mask.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if mask[r, c] or (r, c) in reserved:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in cell_set:
                return False
    return True


def _add_extra_segments(
    wall_mask: np.ndarray,
    barrier_info: dict[str, object],
    seeds: list[tuple[int, int, int]],
) -> None:
    h, w = wall_mask.shape
    c0 = int(barrier_info["c0"])
    thickness = int(barrier_info["thickness"])
    left_max = c0 - 2
    right_min = c0 + thickness + 1
    if left_max < 1 and right_min > w - 2:
        return

    reserved = _reserve_seed_neighborhood(seeds, wall_mask.shape)
    target_count = random.randint(0, 3)
    added = 0
    for _ in range(120):
        if added >= target_count:
            break
        side = random.choice(("left", "right"))
        vertical = random.choice((False, True))
        length = random.randint(2, 4)
        if side == "left":
            col_lo, col_hi = 1, left_max
        else:
            col_lo, col_hi = right_min, w - 2
        if col_lo > col_hi:
            continue
        if vertical:
            row_lo, row_hi = 1, h - length - 1
            if row_lo > row_hi:
                continue
            r0 = random.randint(row_lo, row_hi)
            c0_seg = random.randint(col_lo, col_hi)
            cells = _segment_cells(r0, c0_seg, 1, 0, length)
        else:
            span_hi = col_hi - length + 1
            if col_lo > span_hi:
                continue
            r0 = random.randint(1, h - 2)
            c0_seg = random.randint(col_lo, span_hi)
            cells = _segment_cells(r0, c0_seg, 0, 1, length)
        if not _can_place_extra(wall_mask, cells, reserved):
            continue
        for r, c in cells:
            wall_mask[r, c] = True
        added += 1


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _build_candidate() -> np.ndarray:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice([color for color in PALETTE if color != GREY])
    seed_colors = random.sample([color for color in PALETTE if color not in {bg, GREY}], 2)

    grid = np.full((h, w), bg, dtype=int)
    wall_mask, barrier_info = _build_main_barrier(h, w)
    c0 = int(barrier_info["c0"])
    thickness = int(barrier_info["thickness"])
    bands = barrier_info["bands"]

    left_row, right_row = _pick_seed_rows(bands)
    left_col = random.randint(1, c0 - 2)
    right_col = random.randint(c0 + thickness + 1, w - 2)
    seeds = [(left_row, left_col, seed_colors[0]), (right_row, right_col, seed_colors[1])]

    _add_extra_segments(wall_mask, barrier_info, seeds)

    grid[wall_mask] = GREY
    for r, c, color in seeds:
        grid[r, c] = color
    return _random_symmetry(grid)


def _no_wall_output(grid: np.ndarray, bg: int, seed_cells: list[tuple[int, int, int]]) -> np.ndarray:
    open_mask = np.zeros_like(grid, dtype=bool)
    dist_a = _bfs_distances(open_mask, seed_cells[0][:2])
    dist_b = _bfs_distances(open_mask, seed_cells[1][:2])
    return _paint_from_distances(grid, bg, seed_cells, dist_a, dist_b)[0]


def _valid_candidate(grid: np.ndarray) -> bool:
    try:
        meta = _analyze_grid(grid)
    except AssertionError:
        return False

    wall_mask = meta["wall_mask"]
    bg = int(meta["bg"])
    seed_cells = meta["seed_cells"]
    bg_mask = grid == bg
    out = meta["output"]

    even_a = int(np.count_nonzero(meta["even_a"]))
    even_b = int(np.count_nonzero(meta["even_b"]))
    odd_a = int(np.count_nonzero(meta["odd_a"]))
    odd_b = int(np.count_nonzero(meta["odd_b"]))
    ties = int(np.count_nonzero(meta["tie"]))
    changed = int(np.count_nonzero(out != grid))
    wall_components = _count_components(wall_mask)

    if wall_components < 2:
        return False
    if even_a < 4 or even_b < 4:
        return False
    if odd_a < 2 or odd_b < 2:
        return False
    if ties < 1:
        return False
    if changed < 12 or changed > int(np.count_nonzero(bg_mask) * 0.65):
        return False

    no_wall_out = _no_wall_output(grid, bg, seed_cells)
    obstacle_effect = int(np.count_nonzero((out != no_wall_out) & bg_mask))
    if obstacle_effect < 6:
        return False

    if not np.array_equal(main(grid), out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if _valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid ripple-with-walls input")
