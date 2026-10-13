from __future__ import annotations

from collections import Counter
import random

import numpy as np


ANCHOR_COLOR = 5
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PATH_COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]

GRID_ROWS_RANGE = (11, 18)
GRID_COLS_RANGE = (11, 18)
SEGMENT_COUNT_RANGE = (4, 6)
SEGMENT_LENGTH_POOL = tuple(range(1, 9))
TOTAL_PATH_LEN_RANGE = (14, 26)
TURN_COUNT_RANGE = (2, 8)
GEOM_RUN_MAX = 6
GEN_ATTEMPTS = 500


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _diag_neighbors(r: int, c: int) -> list[Coord]:
    return [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _path_degrees(cells: set[Coord]) -> dict[Coord, int]:
    return {cell: sum(nei in cells for nei in _neighbors4(*cell)) for cell in cells}


def _traverse_path(cells: set[Coord], start: Coord) -> list[Coord]:
    path: list[Coord] = []
    prev: Coord | None = None
    cur = start
    while True:
        path.append(cur)
        nxts = [nei for nei in _neighbors4(*cur) if nei in cells and nei != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must be unique"
        prev, cur = cur, nxts[0]
    assert len(path) == len(cells), "path must visit every cell exactly once"
    return path


def _segments_on_path(grid: np.ndarray, path: list[Coord]) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    colors: list[int] = []
    lengths: list[int] = []
    spans: list[tuple[int, int]] = []
    start = 0
    cur_color = int(grid[path[0]])
    run = 1
    for idx in range(1, len(path)):
        color = int(grid[path[idx]])
        if color == cur_color:
            run += 1
            continue
        colors.append(cur_color)
        lengths.append(run)
        spans.append((start, start + run - 1))
        start += run
        cur_color = color
        run = 1
    colors.append(cur_color)
    lengths.append(run)
    spans.append((start, start + run - 1))
    return colors, lengths, spans


def _turn_indices(path: list[Coord]) -> set[int]:
    turns: set[int] = set()
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            turns.add(i)
    return turns


def _paint_by_segments(base: np.ndarray, path: list[Coord], colors: list[int], lengths: list[int]) -> np.ndarray:
    assert sum(lengths) == len(path)
    out = base.copy()
    idx = 0
    for color, length in zip(colors, lengths):
        for _ in range(length):
            r, c = path[idx]
            out[r, c] = int(color)
            idx += 1
    return out


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    anchor_coords = np.argwhere(g == ANCHOR_COLOR)
    assert len(anchor_coords) == 1, "expected exactly one gray anchor"
    anchor = tuple(map(int, anchor_coords[0]))
    assert int(g[anchor]) != bg, "anchor must be distinct from the background"

    path_cells = {
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (g != ANCHOR_COLOR))
    }
    assert path_cells, "expected a non-empty path"
    assert anchor not in path_cells, "anchor must not lie on the path"

    degrees = _path_degrees(path_cells)
    assert all(deg in (1, 2) for deg in degrees.values()), "path must be branchless"
    endpoints = sorted([cell for cell, deg in degrees.items() if deg == 1])
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    anchored = [ep for ep in endpoints if abs(ep[0] - anchor[0]) == 1 and abs(ep[1] - anchor[1]) == 1]
    assert len(anchored) == 1, "anchor must pick exactly one endpoint"
    start = anchored[0]
    path = _traverse_path(path_cells, start)
    end = path[-1]
    assert end in endpoints and end != start

    colors, lengths, spans = _segments_on_path(g, path)
    assert len(colors) >= 2
    assert len(set(lengths)) == len(lengths), "segment lengths must be pairwise distinct"
    assert all(colors[i] != colors[i + 1] for i in range(len(colors) - 1))

    return {
        "grid": g,
        "bg": bg,
        "anchor": anchor,
        "path": path,
        "path_cells": path_cells,
        "start": start,
        "end": end,
        "segment_colors": colors,
        "segment_lengths": lengths,
        "segment_spans": spans,
        "turns": _turn_indices(path),
    }


def _apply_length_reversal(grid: np.ndarray, path: list[Coord]) -> np.ndarray:
    colors, lengths, _spans = _segments_on_path(grid, path)
    return _paint_by_segments(grid, path, colors, list(reversed(lengths)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the color order on a single anchored path fixed while reversing only its segment-length sequence.

    Input structure:
    - The background is the majority color.
    - All foreground pixels except one gray anchor form a single 4-connected, one-pixel-wide, branchless path with exactly two endpoints.
    - Along the path, consecutive monochromatic segments have pairwise distinct lengths, and the gray anchor is diagonally adjacent to exactly one endpoint.

    Rule:
    - Read the path from the anchor-indicated endpoint to the opposite endpoint.
    - Extract the segment color sequence c1..ck and the segment length sequence L1..Lk.
    - Keep the color sequence fixed, replace the lengths by Lk..L1, and repaint the same path cells from the anchored endpoint with those new (color, length) pairs.

    Key insight:
    - The path carries two independent channels: a fixed color order and a reversible length order; only the length channel flips, and the anchor determines the reading direction.
    """
    info = _analyze_grid(grid)
    return _paint_by_segments(
        info["grid"],
        info["path"],
        info["segment_colors"],
        list(reversed(info["segment_lengths"])),
    )


def _sample_segment_lengths() -> list[int]:
    for _ in range(200):
        k = random.randint(*SEGMENT_COUNT_RANGE)
        lengths = random.sample(SEGMENT_LENGTH_POOL, k)
        total = sum(lengths)
        if TOTAL_PATH_LEN_RANGE[0] <= total <= TOTAL_PATH_LEN_RANGE[1]:
            random.shuffle(lengths)
            return lengths
    raise RuntimeError("failed to sample segment lengths")


def _sample_run_lengths(total_steps: int) -> list[int]:
    feasible_turns = [
        turns
        for turns in range(TURN_COUNT_RANGE[0], TURN_COUNT_RANGE[1] + 1)
        if turns + 1 <= total_steps <= GEOM_RUN_MAX * (turns + 1)
    ]
    turns = random.choice(feasible_turns)
    n_runs = turns + 1
    remaining = total_steps
    runs: list[int] = []
    for i in range(n_runs):
        slots = n_runs - i - 1
        lo = max(1, remaining - GEOM_RUN_MAX * slots)
        hi = min(GEOM_RUN_MAX, remaining - slots)
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    shape: tuple[int, int],
) -> tuple[list[Coord], set[Coord]] | None:
    dr, dc = direction
    cur_r, cur_c = path[-1]
    new_path = list(path)
    new_used = set(used)

    for _ in range(steps):
        nr, nc = cur_r + dr, cur_c + dc
        if not (1 <= nr < shape[0] - 1 and 1 <= nc < shape[1] - 1):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (cur_r, cur_c) for ar, ac in _neighbors4(nr, nc)):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        cur_r, cur_c = nr, nc

    return new_path, new_used


def _turn_dirs(prev_dir: Coord | None) -> list[Coord]:
    if prev_dir is None:
        dirs = list(DIRS4)
        random.shuffle(dirs)
        return dirs
    dr, dc = prev_dir
    dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _build_path(shape: tuple[int, int], path_len: int) -> list[Coord] | None:
    run_lengths = _sample_run_lengths(path_len - 1)
    starts = [(r, c) for r in range(1, shape[0] - 1) for c in range(1, shape[1] - 1)]
    random.shuffle(starts)

    def search(seg_i: int, prev_dir: Coord | None, path: list[Coord], used: set[Coord]) -> list[Coord] | None:
        if seg_i == len(run_lengths):
            return path
        for direction in _turn_dirs(prev_dir):
            extended = _extend_segment(path, used, direction, run_lengths[seg_i], shape)
            if extended is None:
                continue
            found = search(seg_i + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts:
        found = search(0, None, [start], {start})
        if found is not None:
            return found
    return None


def _anchor_candidates(endpoint: Coord, path_cells: set[Coord], shape: tuple[int, int]) -> list[Coord]:
    candidates: list[Coord] = []
    for ar, ac in _diag_neighbors(*endpoint):
        if not _in_bounds(shape, ar, ac):
            continue
        if (ar, ac) in path_cells:
            continue
        if any(nei in path_cells for nei in _neighbors4(ar, ac)):
            continue
        diag_hits = [p for p in _diag_neighbors(ar, ac) if p in path_cells]
        if diag_hits == [endpoint]:
            candidates.append((ar, ac))
    return candidates


def _segment_layout_ok(path: list[Coord], lengths: list[int]) -> bool:
    turns = _turn_indices(path)
    if not turns:
        return False
    spans: list[tuple[int, int]] = []
    start = 0
    for length in lengths:
        spans.append((start, start + length - 1))
        start += length
    has_internal_turn = any(any(a < t < b for t in turns) for a, b in spans)
    boundary_end_indices = [b for _a, b in spans[:-1]]
    has_non_turn_boundary = any(idx not in turns for idx in boundary_end_indices)
    return has_internal_turn and has_non_turn_boundary


def _full_cell_reverse(grid: np.ndarray, path: list[Coord]) -> np.ndarray:
    vals = [int(grid[r, c]) for r, c in path]
    out = np.asarray(grid, dtype=int).copy()
    for (r, c), val in zip(path, reversed(vals)):
        out[r, c] = val
    return out


def _semantic_accept(grid: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    if len(info["segment_colors"]) < SEGMENT_COUNT_RANGE[0]:
        return False
    if not (TURN_COUNT_RANGE[0] <= len(info["turns"]) <= TURN_COUNT_RANGE[1]):
        return False
    if not _segment_layout_ok(info["path"], info["segment_lengths"]):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    wrong_full_reverse = _full_cell_reverse(info["grid"], info["path"])
    if np.array_equal(out, wrong_full_reverse):
        return False

    out_info = _analyze_grid(out)
    if out_info["segment_colors"] != info["segment_colors"]:
        return False
    if out_info["segment_lengths"] != list(reversed(info["segment_lengths"])):
        return False
    if out_info["path"] != info["path"]:
        return False
    if out_info["anchor"] != info["anchor"]:
        return False
    return True


def _generate_one() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    bg = random.choice(PATH_COLOR_POOL)

    seg_lengths = _sample_segment_lengths()
    path_len = sum(seg_lengths)
    path = _build_path((rows, cols), path_len)
    if path is None:
        return None

    turns = _turn_indices(path)
    if not (TURN_COUNT_RANGE[0] <= len(turns) <= TURN_COUNT_RANGE[1]):
        return None

    path_cells = set(path)
    degree_map = _path_degrees(path_cells)
    endpoints = [cell for cell, deg in degree_map.items() if deg == 1]
    random.shuffle(endpoints)

    anchor_cell = None
    anchor_endpoint = None
    for endpoint in endpoints:
        candidates = _anchor_candidates(endpoint, path_cells, (rows, cols))
        if candidates:
            anchor_endpoint = endpoint
            anchor_cell = random.choice(candidates)
            break
    if anchor_cell is None or anchor_endpoint is None:
        return None

    colors = random.sample([c for c in PATH_COLOR_POOL if c != bg], len(seg_lengths))
    ordered_path = path if path[0] == anchor_endpoint else list(reversed(path))
    if not _segment_layout_ok(ordered_path, seg_lengths):
        return None

    grid = np.full((rows, cols), bg, dtype=int)
    idx = 0
    for color, length in zip(colors, seg_lengths):
        for _ in range(length):
            r, c = ordered_path[idx]
            grid[r, c] = int(color)
            idx += 1
    grid[anchor_cell] = ANCHOR_COLOR

    if not _semantic_accept(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored length-reversal path")
