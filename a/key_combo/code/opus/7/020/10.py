from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, components_mask, in_bounds, neighbors4


PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
GRID_H_RANGE = (13, 18)
GRID_W_RANGE = (13, 18)
SOURCE_COUNT_CHOICES = (3, 4)
GEN_ATTEMPTS = 1500
MIN_SOURCE_GAP = 2
MIN_REGION_COUNT = 4
MIN_REGION_AREA = 3
LONG_SPAN_MIN = 3
TIE_PROBABILITY = 0.4


def _extract_specs(grid: np.ndarray) -> list[tuple[int, int, tuple[int, int] | None]]:
    top_cols = [int(c) for c in np.flatnonzero(grid[0] != 0)]
    assert 3 <= len(top_cols) <= 4, "expected 3-4 colored source pixels on the top row"

    top_colors = [int(grid[0, c]) for c in top_cols]
    assert len(set(top_colors)) == len(top_colors), "top-row source colors must be distinct"

    specs = []
    endpoint_rows: list[int] = []
    source_cols = set(top_cols)
    for src_col in top_cols:
        color = int(grid[0, src_col])
        points = np.argwhere(grid == color)
        top_points = [tuple(map(int, rc)) for rc in points if int(rc[0]) == 0]
        end_points = [tuple(map(int, rc)) for rc in points if int(rc[0]) != 0]
        assert len(top_points) == 1, "each path color must have exactly one top-row source"
        assert len(end_points) <= 1, "each path color must have at most one interior endpoint"

        endpoint = end_points[0] if end_points else None
        if endpoint is not None:
            er, ec = endpoint
            assert 0 < er < grid.shape[0] - 1, "endpoint must stay off the top and bottom borders"
            assert ec not in source_cols, "endpoints must not share a column with any source"
            endpoint_rows.append(er)
        specs.append((color, src_col, endpoint))

    assert len(endpoint_rows) == len(set(endpoint_rows)), "endpoint rows must be distinct"
    return specs


def _draw_path_network(
    specs: list[tuple[int, int, tuple[int, int] | None]],
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=int)

    for color, src_col, endpoint in specs:
        end_row, end_col = (h - 1, src_col) if endpoint is None else endpoint
        out[: end_row + 1, src_col] = int(color)
        c0, c1 = sorted((src_col, end_col))
        out[end_row, c0 : c1 + 1] = int(color)
    return out


def _region_vote(
    path_grid: np.ndarray,
    coords: np.ndarray,
) -> tuple[int, Counter[int], bool]:
    adjacent: set[tuple[int, int]] = set()
    h, w = path_grid.shape

    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if in_bounds((h, w), nr, nc) and int(path_grid[nr, nc]) != 0:
                adjacent.add((int(nr), int(nc)))

    assert adjacent, "every background region must touch the path network"
    ordered = sorted(adjacent)
    counts = Counter(int(path_grid[r, c]) for r, c in ordered)
    max_count = max(counts.values())
    tied = {color for color, count in counts.items() if count == max_count}
    fill_color = next(int(path_grid[r, c]) for r, c in ordered if int(path_grid[r, c]) in tied)
    return fill_color, counts, len(tied) > 1


def _nearest_path_color(path_grid: np.ndarray, coords: np.ndarray) -> int:
    path_pixels = [
        (int(r), int(c), int(path_grid[r, c]))
        for r, c in np.argwhere(path_grid != 0)
    ]
    best = min(
        (
            min(abs(pr - int(rr)) + abs(pc - int(cc)) for rr, cc in coords),
            pr,
            pc,
            color,
        )
        for pr, pc, color in path_pixels
    )
    return int(best[3])


def _analyze_regions(path_grid: np.ndarray) -> list[dict]:
    regions = []
    for coords in components_mask(path_grid == 0, connectivity=4):
        fill_color, counts, tie = _region_vote(path_grid, coords)
        regions.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "fill_color": int(fill_color),
                "counts": counts,
                "tie": bool(tie),
                "nearest_color": _nearest_path_color(path_grid, coords),
            }
        )
    return regions


def _render_output(path_grid: np.ndarray) -> np.ndarray:
    out = path_grid.copy()
    for region in _analyze_regions(path_grid):
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = int(region["fill_color"])
    return out


def _noncrossing(specs: list[tuple[int, int, tuple[int, int] | None]]) -> bool:
    rows = [endpoint[0] for _color, _src_col, endpoint in specs if endpoint is not None]
    if len(rows) != len(set(rows)):
        return False

    for i, (_color_i, src_i, endpoint_i) in enumerate(specs):
        end_i = endpoint_i if endpoint_i is not None else (10**9, src_i)
        row_i, col_i = end_i
        lo_i, hi_i = sorted((src_i, col_i))
        for j, (_color_j, src_j, endpoint_j) in enumerate(specs):
            if i == j:
                continue
            end_j = endpoint_j if endpoint_j is not None else (10**9, src_j)
            row_j, col_j = end_j
            if row_i < row_j and lo_i <= src_j <= hi_i and src_j != src_i:
                return False
            if row_i == row_j:
                lo_j, hi_j = sorted((src_j, col_j))
                if max(lo_i, lo_j) <= min(hi_i, hi_j):
                    return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw endpoint-guided L paths from the top edge and flood-fill each enclosed region by the majority boundary-path color.

    Input structure:
    - The black background contains 3 or 4 distinct colored source pixels on the top row.
    - For each source color there is at most one same-colored interior endpoint pixel; the training generator always provides one.
    - Endpoint rows are distinct, and endpoint columns do not match any source column.

    Rule:
    - For each top-row source, draw a vertical path of the same color downward from the source column.
    - If a same-colored endpoint exists, stop on that endpoint row and extend horizontally to the endpoint column; otherwise extend straight to the bottom row.
    - Treat all drawn path pixels as walls, split the remaining black cells into 4-connected regions, and count the distinct 4-neighbor path pixels touching each region.
    - Fill each region with the most frequent touching path color, breaking ties by the earliest adjacent path pixel in row-major order; keep the path pixels unchanged.

    Key insight:
    - The answer depends on the whole L-shaped path network: endpoint positions control the partition geometry, and each region is colored by a majority vote over all touching boundary path pixels rather than by the nearest path alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    specs = _extract_specs(g)
    path_grid = _draw_path_network(specs, g.shape)
    return _render_output(path_grid)


def _sample_spaced_columns(width: int, count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    for _ in range(100):
        cols = sorted(random.sample(range(lo, hi + 1), count))
        if all(cols[i + 1] - cols[i] >= min_gap for i in range(count - 1)):
            return cols
    return None


def _sample_endpoint_columns(
    width: int,
    source_cols: list[int],
    endpoint_rows: list[int],
) -> list[int] | None:
    source_set = set(source_cols)
    used: set[int] = set()
    chosen = [None] * len(source_cols)
    order = sorted(range(len(source_cols)), key=lambda i: endpoint_rows[i])

    for idx in order:
        src_col = source_cols[idx]
        active = [source_cols[j] for j in range(len(source_cols)) if endpoint_rows[j] > endpoint_rows[idx]]
        left_block = max([0] + [c for c in active if c < src_col])
        right_block = min([width - 1] + [c for c in active if c > src_col])
        candidates = [
            c
            for c in range(left_block + 1, right_block)
            if c not in source_set and c not in used
        ]
        if not candidates:
            return None

        long_candidates = [c for c in candidates if abs(c - src_col) >= LONG_SPAN_MIN]
        if long_candidates and (random.random() < 0.7 or idx == order[-1]):
            candidates = long_candidates

        distances = [abs(c - src_col) for c in candidates]
        best_distance = max(distances)
        preferred = [c for c in candidates if abs(c - src_col) >= max(1, best_distance - 1)]
        pick_pool = preferred if preferred and random.random() < 0.8 else candidates
        end_col = random.choice(pick_pool)
        chosen[idx] = int(end_col)
        used.add(int(end_col))

    return [int(c) for c in chosen if c is not None]


def _make_input_grid(
    height: int,
    width: int,
    specs: list[tuple[int, int, tuple[int, int] | None]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for color, src_col, endpoint in specs:
        grid[0, src_col] = int(color)
        if endpoint is not None:
            er, ec = endpoint
            grid[er, ec] = int(color)
    return grid


def _valid_candidate(
    specs: list[tuple[int, int, tuple[int, int] | None]],
    shape: tuple[int, int],
    require_tie: bool,
) -> bool:
    if not _noncrossing(specs):
        return False

    endpoints = [endpoint for _color, _src_col, endpoint in specs if endpoint is not None]
    assert len(endpoints) == len(specs)

    turns = {int(np.sign(endpoint[1] - src_col)) for _color, src_col, endpoint in specs if endpoint is not None}
    if turns != {-1, 1}:
        return False

    if max(abs(endpoint[1] - src_col) for _color, src_col, endpoint in specs if endpoint is not None) < LONG_SPAN_MIN:
        return False

    path_grid = _draw_path_network(specs, shape)
    regions = _analyze_regions(path_grid)
    if len(regions) < MIN_REGION_COUNT:
        return False
    if min(region["area"] for region in regions) < MIN_REGION_AREA:
        return False

    mismatches = [region for region in regions if region["fill_color"] != region["nearest_color"]]
    if not mismatches:
        return False

    if require_tie and not any(region["tie"] for region in regions):
        return False

    inp = _make_input_grid(shape[0], shape[1], specs)
    expected = _render_output(path_grid)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt(require_tie: bool) -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    count = random.choice(SOURCE_COUNT_CHOICES)
    colors = random.sample(PALETTE, count)

    source_cols = _sample_spaced_columns(width, count, 2, width - 3, MIN_SOURCE_GAP)
    if source_cols is None:
        return None

    endpoint_rows = random.sample(range(3, height - 2), count)
    endpoint_cols = _sample_endpoint_columns(width, source_cols, endpoint_rows)
    if endpoint_cols is None:
        return None

    specs = [
        (int(color), int(src_col), (int(end_row), int(end_col)))
        for color, src_col, end_row, end_col in zip(colors, source_cols, endpoint_rows, endpoint_cols)
    ]

    if not _valid_candidate(specs, (height, width), require_tie=require_tie):
        return None
    return _make_input_grid(height, width, specs)


def generate_input() -> np.ndarray:
    tie_plan = [random.random() < TIE_PROBABILITY, False]
    seen: set[bool] = set()
    for require_tie in tie_plan:
        if require_tie in seen:
            continue
        seen.add(require_tie)
        for _ in range(GEN_ATTEMPTS):
            grid = _attempt(require_tie=require_tie)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid endpoint-turn majority-fill input")
