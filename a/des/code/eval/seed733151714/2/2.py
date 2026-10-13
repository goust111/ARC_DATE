from __future__ import annotations

from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
VERTICAL_DIRS = [(-1, 0), (1, 0)]
HORIZONTAL_DIRS = [(0, -1), (0, 1)]

MIN_LEGEND_LEN = 4
MAX_LEGEND_LEN = 6
MAX_PATHS = 3
MAX_SEGMENTS = 4
MAX_PATH_BBOX_H = 6
MAX_PATH_BBOX_W = 12
PATH_ROW_GAP = 1
MAX_GRID_H = 22
MAX_GRID_W = 40
MIN_GRID_H = 10
PATH_BUILD_RETRIES = 200
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_legend(bottom_row: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    w = int(bottom_row.size)
    while c < w:
        if int(bottom_row[c]) == bg:
            c += 1
            continue
        color = int(bottom_row[c])
        start = c
        while c < w and int(bottom_row[c]) == color:
            c += 1
        length = c - start
        assert length not in mapping, "legend lengths must be unique"
        mapping[length] = color
    assert mapping, "expected a legend on the bottom row"
    max_len = max(mapping)
    assert set(mapping) == set(range(1, max_len + 1)), "legend must cover lengths 1..k"
    return mapping


def _trace_open_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(coords) == 1:
        return [coords[0]]

    cells = set(coords)
    nbrs = {
        cell: [n for n in _neighbors4(cell) if n in cells]
        for cell in cells
    }
    degrees = {cell: len(adj) for cell, adj in nbrs.items()}
    assert all(deg in (1, 2) for deg in degrees.values()), "paths must be non-branching"
    endpoints = sorted(cell for cell, deg in degrees.items() if deg == 1)
    assert len(endpoints) == 2, "expected an open path with two endpoints"

    ordered = [endpoints[0]]
    prev: tuple[int, int] | None = None
    cur = endpoints[0]
    while True:
        next_cells = [n for n in nbrs[cur] if n != prev]
        if not next_cells:
            break
        nxt = next_cells[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    assert len(ordered) == len(coords), "path traversal must visit every cell once"
    return ordered


def _split_segments(ordered: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if len(ordered) <= 1:
        return [ordered]

    directions = [
        (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
        for i in range(len(ordered) - 1)
    ]
    starts = [0]
    for i in range(1, len(ordered) - 1):
        if directions[i - 1] != directions[i]:
            starts.append(i)

    ends = starts[1:] + [len(ordered)]
    segments = [ordered[s:e] for s, e in zip(starts, ends)]
    assert all(segment for segment in segments)
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each single-pixel orthogonal path at turns, then recolor each segment by legend length.

    Input structure:
    - bg is the dominant color.
    - The bottom row contains a legend of horizontal bars whose lengths are 1,2,3,...
    - Above the legend are one or more 4-connected, one-pixel-wide orthogonal paths.

    Rule:
    - Split each path wherever its direction changes.
    - The corner pixel belongs to the segment that continues after the turn.
    - Recolor each segment by the legend color associated with that segment length.
    - Leave the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a bottom legend row"

    bg = _get_bg(g)
    legend = _extract_legend(g[-1], bg)

    out = g.copy()
    path_mask = g[:-1] != bg
    components = _components(path_mask)
    assert components, "expected at least one path above the legend"

    for comp in components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each path component must be monochrome"

        ordered = _trace_open_path(comp)
        for segment in _split_segments(ordered):
            length = len(segment)
            assert length in legend, "every segment length must exist in the legend"
            color = legend[length]
            for r, c in segment:
                out[r, c] = color

    return out


def _composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if total <= 0:
        return [0] * parts
    probs = np.full(parts, 1.0 / parts)
    return [int(x) for x in rng.multinomial(total, probs)]


def _random_directions(seg_count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    first_pool = VERTICAL_DIRS if rng.random() < 0.5 else HORIZONTAL_DIRS
    directions = [first_pool[int(rng.integers(len(first_pool)))]]
    for _ in range(1, seg_count):
        prev = directions[-1]
        pool = HORIZONTAL_DIRS if prev in VERTICAL_DIRS else VERTICAL_DIRS
        directions.append(pool[int(rng.integers(len(pool)))])
    return directions


def _cells_from_segments(lengths: list[int], directions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cur = (0, 0)
    cells = [cur]

    for _ in range(lengths[0] - 1):
        cur = (cur[0] + directions[0][0], cur[1] + directions[0][1])
        cells.append(cur)

    for idx in range(1, len(lengths)):
        prev_dr, prev_dc = directions[idx - 1]
        cur = (cur[0] + prev_dr, cur[1] + prev_dc)
        cells.append(cur)
        dr, dc = directions[idx]
        for _ in range(lengths[idx] - 1):
            cur = (cur[0] + dr, cur[1] + dc)
            cells.append(cur)

    return cells


def _is_simple_open_path(cells: list[tuple[int, int]]) -> bool:
    if len(set(cells)) != len(cells):
        return False
    if len(cells) == 1:
        return True

    cell_set = set(cells)
    degrees = []
    for cell in cell_set:
        deg = sum((nbr in cell_set) for nbr in _neighbors4(cell))
        if deg not in (1, 2):
            return False
        degrees.append(deg)
    return degrees.count(1) == 2


def _build_path(seg_count: int, max_len: int, rng: np.random.Generator) -> dict[str, object] | None:
    for _ in range(PATH_BUILD_RETRIES):
        lengths = [int(x) for x in rng.integers(1, max_len + 1, size=seg_count)]
        directions = _random_directions(seg_count, rng)
        cells = _cells_from_segments(lengths, directions)
        if not _is_simple_open_path(cells):
            continue

        arr = np.array(cells, dtype=int)
        min_r, min_c = map(int, arr.min(axis=0))
        max_r, max_c = map(int, arr.max(axis=0))
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        if height > MAX_PATH_BBOX_H or width > MAX_PATH_BBOX_W:
            continue

        normalized = [(int(r - min_r), int(c - min_c)) for r, c in cells]
        return {
            "cells": normalized,
            "height": height,
            "width": width,
        }
    return None


def _segment_count_plan(num_paths: int, rng: np.random.Generator) -> list[int]:
    mode = int(rng.integers(3))
    if mode == 0:
        return [1] * num_paths
    if mode == 1:
        counts = [int(rng.integers(1, MAX_SEGMENTS + 1)) for _ in range(num_paths)]
        counts[0] = int(rng.integers(2, MAX_SEGMENTS + 1))
        return counts

    if num_paths == 1:
        return [int(rng.integers(2, MAX_SEGMENTS + 1))]

    counts = [1] * num_paths
    turning_idx = int(rng.integers(num_paths))
    counts[turning_idx] = int(rng.integers(2, MAX_SEGMENTS + 1))
    for idx in range(num_paths):
        if idx == turning_idx:
            continue
        if rng.random() < 0.5:
            counts[idx] = int(rng.integers(2, MAX_SEGMENTS + 1))
    return counts


def _draw_legend(grid: np.ndarray, bg: int, colors: list[int], rng: np.random.Generator) -> None:
    k = len(colors)
    min_width = k * (k + 1) // 2 + (k - 1)
    extra = grid.shape[1] - min_width
    gaps = _composition(extra, k + 1, rng)

    c = gaps[0]
    for idx, length in enumerate(range(1, k + 1)):
        grid[-1, c : c + length] = int(colors[idx])
        c += length
        if idx < k - 1:
            c += 1 + gaps[idx + 1]

    assert np.all((grid[-1] == bg) | (grid[-1] >= 0))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    legend_len = int(rng.integers(MIN_LEGEND_LEN, MAX_LEGEND_LEN + 1))
    num_paths = int(rng.integers(1, MAX_PATHS + 1))
    seg_counts = _segment_count_plan(num_paths, rng)

    specs: list[dict[str, object]] = []
    for seg_count in seg_counts:
        spec = _build_path(seg_count, legend_len, rng)
        if spec is None:
            return None
        specs.append(spec)

    rng.shuffle(specs)
    path_heights = [int(spec["height"]) for spec in specs]
    path_widths = [int(spec["width"]) for spec in specs]

    required_path_h = sum(path_heights) + PATH_ROW_GAP * max(0, num_paths - 1)
    path_area_h = required_path_h + int(rng.integers(1, 6))
    grid_h = max(path_area_h + 2, MIN_GRID_H)
    if grid_h > MAX_GRID_H:
        return None

    legend_min_w = legend_len * (legend_len + 1) // 2 + (legend_len - 1)
    grid_w = max(
        legend_min_w + int(rng.integers(2, 11)),
        max(path_widths) + int(rng.integers(2, 9)),
    )
    if grid_w > MAX_GRID_W:
        return None

    palette_size = 1 + legend_len + num_paths
    colors = rng.choice(np.arange(10), size=palette_size, replace=False).tolist()
    bg = int(colors[0])
    legend_colors = [int(c) for c in colors[1 : 1 + legend_len]]
    path_colors = [int(c) for c in colors[1 + legend_len :]]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    _draw_legend(grid, bg, legend_colors, rng)

    extra_h = path_area_h - required_path_h
    vertical_gaps = _composition(extra_h, num_paths + 1, rng)
    row = vertical_gaps[0]
    for idx, spec in enumerate(specs):
        top = row
        left = int(rng.integers(0, grid_w - int(spec["width"]) + 1))
        for r, c in spec["cells"]:
            grid[top + int(r), left + int(c)] = path_colors[idx]
        row += int(spec["height"])
        if idx < num_paths - 1:
            row += PATH_ROW_GAP + vertical_gaps[idx + 1]

    if Counter(grid.ravel()).most_common(1)[0][0] != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[-1], grid[-1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GENERATION_RETRIES} attempts")
