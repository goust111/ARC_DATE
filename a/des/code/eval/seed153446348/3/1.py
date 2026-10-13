from __future__ import annotations

from collections import deque
import random

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

ROW_COUNT_RANGE = (3, 4)
COL_COUNT_RANGE = (3, 4)
CELL_HEIGHT_RANGE = (2, 5)
CELL_WIDTH_RANGE = (2, 5)
GRID_MIN_SIZE = 10
GRID_MAX_SIZE = 24
SEED_COUNT_CHOICES = (2, 2, 3, 3, 4)
GEN_RETRY_LIMIT = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _detect_background(grid: np.ndarray) -> int:
    if np.any(grid == BG_COLOR):
        return BG_COLOR
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(grid[grid != bg], return_counts=True)
    assert vals.size > 0, "expected at least one wall color"
    return int(vals[int(np.argmax(cnts))])


def _extract_regions(grid: np.ndarray, bg: int, wall: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    region_id = np.full((h, w), -1, dtype=int)
    regions: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if int(grid[r0, c0]) == wall or int(region_id[r0, c0]) != -1:
                continue

            rid = len(regions)
            q = deque([(r0, c0)])
            region_id[r0, c0] = rid
            cells: list[tuple[int, int]] = []
            seed_pixels: list[tuple[int, int, int]] = []

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                v = int(grid[r, c])
                if v not in (bg, wall):
                    seed_pixels.append((r, c, v))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != wall and int(region_id[nr, nc]) == -1:
                        region_id[nr, nc] = rid
                        q.append((nr, nc))

            seed = None
            if seed_pixels:
                colors = {col for _r, _c, col in seed_pixels}
                assert len(seed_pixels) == 1 and len(colors) == 1, "each seeded region must contain one seed pixel"
                seed = seed_pixels[0]
            regions.append({"cells": cells, "seed": seed})

    return regions, region_id


def _build_region_graph(
    grid: np.ndarray, wall: int, region_id: np.ndarray, region_count: int
) -> tuple[list[set[int]], dict[tuple[int, int], int], dict[tuple[int, int], set[tuple[int, int]]]]:
    h, w = grid.shape
    neighbors = [set() for _ in range(region_count)]
    shared_lengths: dict[tuple[int, int], int] = {}
    boundary_walls: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def add_pair(a: int, b: int, wr: int, wc: int) -> None:
        pair = (a, b) if a < b else (b, a)
        shared_lengths[pair] = shared_lengths.get(pair, 0) + 1
        boundary_walls.setdefault(pair, set()).add((wr, wc))
        neighbors[a].add(b)
        neighbors[b].add(a)

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != wall:
                continue

            if 0 < c < w - 1:
                left = int(region_id[r, c - 1])
                right = int(region_id[r, c + 1])
                if left >= 0 and right >= 0 and left != right:
                    add_pair(left, right, r, c)

            if 0 < r < h - 1:
                up = int(region_id[r - 1, c])
                down = int(region_id[r + 1, c])
                if up >= 0 and down >= 0 and up != down:
                    add_pair(up, down, r, c)

    return neighbors, shared_lengths, boundary_walls


def _seed_to_boundary_distance(seed_pos: tuple[int, int], wall_cells: set[tuple[int, int]]) -> int:
    sr, sc = seed_pos
    return min(abs(sr - r) + abs(sc - c) for r, c in wall_cells)


def _propagate_region_colors(
    regions: list[dict],
    neighbors: list[set[int]],
    shared_lengths: dict[tuple[int, int], int],
    boundary_walls: dict[tuple[int, int], set[tuple[int, int]]],
) -> tuple[list[int], list[int], dict]:
    region_colors = [BG_COLOR] * len(regions)
    seed_by_color: dict[int, tuple[int, int]] = {}
    rounds = [-1] * len(regions)

    for rid, region in enumerate(regions):
        if region["seed"] is None:
            continue
        sr, sc, color = region["seed"]
        color = int(color)
        region_colors[rid] = color
        rounds[rid] = 0
        seed_by_color[color] = (int(sr), int(sc))

    decision_meta = {
        "has_tie_break": False,
        "has_length_win": False,
        "ambiguous": False,
        "round_count": 0,
        "decision_count": 0,
    }

    frontier_round = 0
    while any(round_idx < 0 for round_idx in rounds):
        frontier_round += 1
        updates: dict[int, int] = {}
        for rid, round_idx in enumerate(rounds):
            if round_idx >= 0:
                continue

            candidates: list[tuple[int, int, int, int, int, int]] = []
            for nid in neighbors[rid]:
                ncolor = int(region_colors[nid])
                if rounds[nid] < 0:
                    continue
                pair = (rid, nid) if rid < nid else (nid, rid)
                seed_r, seed_c = seed_by_color[ncolor]
                length = int(shared_lengths[pair])
                dist = _seed_to_boundary_distance((seed_r, seed_c), boundary_walls[pair])
                candidates.append((length, dist, seed_r, seed_c, nid, ncolor))

            if not candidates:
                continue

            decision_meta["decision_count"] += 1
            best = min(candidates, key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
            best_len, best_dist, _best_sr, _best_sc, _best_nid, best_color = best

            other_colors = {item[5] for item in candidates if item[5] != best_color}
            if other_colors:
                if any(item[0] == best_len and item[1] > best_dist and item[5] != best_color for item in candidates):
                    decision_meta["has_tie_break"] = True
                if any(item[0] < best_len and item[5] != best_color for item in candidates):
                    decision_meta["has_length_win"] = True
                if any(item[0] == best_len and item[1] == best_dist and item[5] != best_color for item in candidates):
                    decision_meta["ambiguous"] = True

            updates[rid] = best_color

        if not updates:
            raise ValueError("color propagation stalled before all regions were assigned")

        for rid, color in updates.items():
            region_colors[rid] = int(color)
            rounds[rid] = frontier_round

    decision_meta["round_count"] = frontier_round
    return region_colors, rounds, decision_meta


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], np.ndarray, list[int], list[int], dict, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _detect_background(g)
    wall = _detect_wall_color(g, bg)
    regions, region_id = _extract_regions(g, bg, wall)
    neighbors, shared_lengths, boundary_walls = _build_region_graph(g, wall, region_id, len(regions))
    colors, rounds, meta = _propagate_region_colors(regions, neighbors, shared_lengths, boundary_walls)
    return regions, region_id, colors, rounds, meta, bg, wall


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color seedless rectangular regions by propagating seed colors across region adjacency.

    Regions are separated by one-pixel walls. Each round, an uncolored region adopts the color of the
    already-colored neighboring region sharing the longest boundary with it. Ties are broken by choosing
    the color whose original seed pixel is closer to that shared wall segment.
    """
    g = np.asarray(grid, dtype=int)
    regions, _region_id, colors, _rounds, _meta, bg, wall = _analyze_grid(g)
    out = g.copy()

    for rid, region in enumerate(regions):
        color = int(colors[rid])
        for r, c in region["cells"]:
            if int(out[r, c]) == bg:
                out[r, c] = color

    out[g == wall] = wall
    return out


def _axis_lengths(count: int, low: int, high: int) -> list[int]:
    return [random.randint(low, high) for _ in range(count)]


def _line_positions(lengths: list[int]) -> list[int]:
    positions = [0]
    cur = 0
    for size in lengths:
        cur += size + 1
        positions.append(cur)
    return positions


def _region_boxes(row_heights: list[int], col_widths: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    row_lines = _line_positions(row_heights)
    col_lines = _line_positions(col_widths)
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rr, height in enumerate(row_heights):
        for cc, width in enumerate(col_widths):
            r0 = row_lines[rr] + 1
            r1 = r0 + height - 1
            c0 = col_lines[cc] + 1
            c1 = c0 + width - 1
            boxes[(rr, cc)] = (r0, c0, r1, c1)
    return boxes


def _build_partition_grid(row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    grid = np.full((h, w), BG_COLOR, dtype=int)

    for r in _line_positions(row_heights):
        grid[r, :] = WALL_COLOR
    for c in _line_positions(col_widths):
        grid[:, c] = WALL_COLOR
    return grid


def _midpoint(lo: int, hi: int) -> int:
    return (lo + hi) // 2


def _pick_position(
    box: tuple[int, int, int, int],
    toward: str | None = None,
    near: bool = True,
    fixed_row: int | None = None,
    fixed_col: int | None = None,
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if fixed_row is None:
        fixed_row = random.randint(r0, r1)
    if fixed_col is None:
        fixed_col = random.randint(c0, c1)

    if toward == "left":
        c = c0 if near else c1
        return fixed_row, c
    if toward == "right":
        c = c1 if near else c0
        return fixed_row, c
    if toward == "top":
        r = r0 if near else r1
        return r, fixed_col
    if toward == "bottom":
        r = r1 if near else r0
        return r, fixed_col
    return fixed_row, fixed_col


def _region_index(rr: int, cc: int, cols: int) -> int:
    return rr * cols + cc


def _seed_nonfocus_regions(
    rows: int,
    cols: int,
    used: set[tuple[int, int]],
    focus: tuple[int, int],
    extra_count: int,
) -> list[tuple[int, int]]:
    candidates = []
    for rr in range(rows):
        for cc in range(cols):
            pos = (rr, cc)
            if pos in used:
                continue
            if abs(rr - focus[0]) + abs(cc - focus[1]) <= 1:
                continue
            candidates.append(pos)
    random.shuffle(candidates)
    return candidates[:extra_count]


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_COUNT_RANGE)
    cols = random.randint(*COL_COUNT_RANGE)
    row_heights = _axis_lengths(rows, *CELL_HEIGHT_RANGE)
    col_widths = _axis_lengths(cols, *CELL_WIDTH_RANGE)

    grid_h = sum(row_heights) + rows + 1
    grid_w = sum(col_widths) + cols + 1
    if not (GRID_MIN_SIZE <= grid_h <= GRID_MAX_SIZE and GRID_MIN_SIZE <= grid_w <= GRID_MAX_SIZE):
        return None

    focus_options = [(rr, cc) for rr in range(1, rows - 1) for cc in range(1, cols - 1)]
    if not focus_options:
        return None

    require_tie = random.random() < 0.5
    boxes = _region_boxes(row_heights, col_widths)
    grid = _build_partition_grid(row_heights, col_widths)
    focus = random.choice(focus_options)
    fr, fc = focus
    used_regions: set[tuple[int, int]] = {focus}
    plan: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if require_tie:
        if random.random() < 0.5:
            left = (fr, fc - 1)
            right = (fr, fc + 1)
            center_row = _midpoint(boxes[focus][0], boxes[focus][2])
            winner_is_left = random.random() < 0.5
            plan.append((left, _pick_position(boxes[left], toward="right", near=winner_is_left, fixed_row=center_row)))
            plan.append((right, _pick_position(boxes[right], toward="left", near=not winner_is_left, fixed_row=center_row)))
            used_regions.update((left, right))
        else:
            up = (fr - 1, fc)
            down = (fr + 1, fc)
            center_col = _midpoint(boxes[focus][1], boxes[focus][3])
            winner_is_up = random.random() < 0.5
            plan.append((up, _pick_position(boxes[up], toward="bottom", near=winner_is_up, fixed_col=center_col)))
            plan.append((down, _pick_position(boxes[down], toward="top", near=not winner_is_up, fixed_col=center_col)))
            used_regions.update((up, down))
    else:
        length_options = [pos for pos in focus_options if row_heights[pos[0]] != col_widths[pos[1]]]
        if not length_options:
            return None
        focus = random.choice(length_options)
        fr, fc = focus
        used_regions = {focus}
        left = (fr, fc - 1)
        up = (fr - 1, fc)
        row_mid = _midpoint(boxes[focus][0], boxes[focus][2])
        col_mid = _midpoint(boxes[focus][1], boxes[focus][3])
        plan.append((left, _pick_position(boxes[left], toward="right", near=True, fixed_row=row_mid)))
        plan.append((up, _pick_position(boxes[up], toward="bottom", near=True, fixed_col=col_mid)))
        used_regions.update((left, up))

    max_seed_count = min(len(SEED_COLORS), rows * cols - 1)
    seed_count = min(random.choice(SEED_COUNT_CHOICES), max_seed_count)
    if seed_count < len(plan):
        seed_count = len(plan)
    extra_count = seed_count - len(plan)
    extras = _seed_nonfocus_regions(rows, cols, used_regions, focus, extra_count)
    if len(extras) < extra_count:
        return None

    colors = random.sample(SEED_COLORS, seed_count)
    all_regions = [item[0] for item in plan] + extras
    positions = [item[1] for item in plan] + [_pick_position(boxes[pos]) for pos in extras]
    for (rr, cc), (pr, pc), color in zip(all_regions, positions, colors, strict=False):
        assert (rr, cc) != focus
        grid[pr, pc] = int(color)

    try:
        regions, _region_id, region_colors, rounds, meta, bg, wall = _analyze_grid(grid)
    except Exception:
        return None

    if bg != BG_COLOR or wall != WALL_COLOR:
        return None
    if len(regions) != rows * cols:
        return None
    if any(color == BG_COLOR for color in region_colors):
        return None
    if meta["ambiguous"]:
        return None
    if meta["round_count"] < 2:
        return None
    if meta["decision_count"] < 2:
        return None
    if require_tie and not meta["has_tie_break"]:
        return None
    if not require_tie and not meta["has_length_win"]:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the propagation puzzle")
