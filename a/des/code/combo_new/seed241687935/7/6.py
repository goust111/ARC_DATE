import bisect
import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PATH_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _mode_count(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(counts.max()) if len(counts) else 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _intervals(boundary_cols: list[int], width: int) -> list[tuple[int, int]]:
    out = []
    left = 0
    for col in boundary_cols:
        out.append((left, col - 1))
        left = col + 1
    out.append((left, width - 1))
    return out


def _separator_row(grid: np.ndarray, bg: int) -> int:
    row_scores = np.sum(grid != bg, axis=1)
    best = int(row_scores.max())
    candidates = [int(r) for r in np.where(row_scores == best)[0]]
    return max(candidates, key=lambda r: _mode_count(grid[r]))


def _detect_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    sep_r = _separator_row(g, bg)

    row_vals = g[sep_r]
    non_bg = row_vals[row_vals != bg]
    assert non_bg.size > 0, "separator row must contain non-background pixels"
    grey = int(np.bincount(non_bg, minlength=10).argmax())

    def find_half_boundaries(row0: int, row1: int) -> list[int]:
        cols = []
        span = g[row0:row1, :]
        for c in range(w):
            col = span[:, c]
            grey_count = int(np.sum(col == grey))
            if grey_count <= len(col) // 2:
                continue
            if np.all(col != bg):
                cols.append(c)
        return cols

    top_cols = find_half_boundaries(0, sep_r + 1)
    bottom_cols = find_half_boundaries(sep_r, h)
    assert top_cols, "top half must contain partition columns"
    assert bottom_cols, "bottom half must contain partition columns"

    return bg, grey, sep_r, top_cols, bottom_cols


def _region_id(
    r: int,
    c: int,
    sep_r: int,
    top_cols: list[int],
    bottom_cols: list[int],
    top_set: set[int],
    bottom_set: set[int],
):
    if r == sep_r:
        return None
    if r < sep_r:
        if c in top_set:
            return None
        return 0, bisect.bisect_left(top_cols, c)
    if c in bottom_set:
        return None
    return 1, bisect.bisect_left(bottom_cols, c)


def _trace_path(path_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    pts = [tuple(map(int, rc)) for rc in np.argwhere(path_mask)]
    assert pts, "path must be non-empty"
    pt_set = set(pts)

    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pts:
        neighbors = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pt_set:
                neighbors.append((nr, nc))
        nbrs[(r, c)] = neighbors

    ends = [p for p, xs in nbrs.items() if len(xs) == 1]
    assert len(ends) == 2, "path must have exactly two endpoints"

    left_ends = [p for p in ends if p[1] == 0]
    right_ends = [p for p in ends if p[1] == w - 1]
    assert len(left_ends) == 1 and len(right_ends) == 1, "path must start/end on left/right edges"

    start = left_ends[0]
    out = [start]
    prev = None
    cur = start
    while True:
        nxts = [p for p in nbrs[cur] if p != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch or self-touch"
        prev, cur = cur, nxts[0]
        out.append(cur)

    assert len(out) == len(pts), "traced path must cover every path pixel exactly once"
    assert out[-1] == right_ends[0], "path must terminate on the right edge"
    return out


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, grey, sep_r, top_cols, bottom_cols = _detect_structure(g)
    top_set = set(top_cols)
    bottom_set = set(bottom_cols)

    path_mask = (g != bg) & (g != grey)
    trace = _trace_path(path_mask)

    region_color: dict[tuple[int, int], int] = {}
    for r, c in trace:
        rid = _region_id(r, c, sep_r, top_cols, bottom_cols, top_set, bottom_set)
        if rid is None:
            continue
        col = int(g[r, c])
        prev = region_color.get(rid)
        if prev is None:
            region_color[rid] = col
        else:
            assert prev == col, "every visited region must use a single color"

    top_intervals = _intervals(top_cols, w)
    bottom_intervals = _intervals(bottom_cols, w)
    out = g.copy()
    for (half, idx), color in region_color.items():
        if half == 0:
            c0, c1 = top_intervals[idx]
            region = out[:sep_r, c0 : c1 + 1]
        else:
            c0, c1 = bottom_intervals[idx]
            region = out[sep_r + 1 :, c0 : c1 + 1]
        region[region == bg] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular subregion visited by the multicolored path with that region's path color.

    Input structure:
    - The background is black.
    - One complete horizontal grey separator splits the grid into a top half and a bottom half.
    - Each half is further partitioned by full-height grey vertical segments, creating rectangular black subregions.
    - A 4-connected one-pixel-wide path starts on the left edge, ends on the right edge, and can cross both the separator and the vertical partitions; crossing cells overwrite grey with path colors.

    Rule:
    - Detect the structural grey separator row and the grey partition columns in the top and bottom halves.
    - Trace the colored path from its left-edge endpoint to its right-edge endpoint and record which subregion each non-boundary path pixel belongs to.
    - For every visited subregion, fill all black cells in that rectangle with the unique color used by the path inside that subregion.
    - Keep the original path colors unchanged, preserve uncovered grey wall pixels, and leave unvisited subregions black.

    Key insight:
    - The path color is attached to the rectangular subregion it passes through, so the solver must reconstruct the partition geometry and follow the whole left-to-right path to recover the fill color of each visited region.
    """
    return _solve_grid(grid)


def _sample_partition(width: int, rng: random.Random) -> list[int] | None:
    max_regions = min(4, (width + 1) // 4)
    if max_regions < 2:
        return None
    n_regions = rng.randint(2, max_regions)
    region_total = width - (n_regions - 1)
    if region_total < 3 * n_regions:
        return None

    widths = [3] * n_regions
    extra = region_total - 3 * n_regions
    for _ in range(extra):
        widths[rng.randrange(n_regions)] += 1

    cols = []
    pos = widths[0]
    for width_part in widths[1:]:
        cols.append(pos)
        pos += 1 + width_part
    return cols


def _sample_spaced(
    candidates: list[int],
    count: int,
    min_gap: int,
    rng: random.Random,
) -> list[int] | None:
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(120):
        choice = sorted(rng.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(choice, choice[1:])):
            return choice
    return None


def _rows_for_half(half: int, sep_r: int, height: int) -> list[int]:
    if half == 0:
        return list(range(0, sep_r))
    return list(range(sep_r + 1, height))


def _choose_row(rows: list[int], current: int | None, rng: random.Random, prefer_change: bool) -> int:
    if current is None or len(rows) == 1:
        return int(rng.choice(rows))
    others = [r for r in rows if r != current]
    if prefer_change and others:
        return int(rng.choice(others))
    if others and rng.random() < 0.45:
        return int(rng.choice(others))
    return int(current)


def _build_span_points(
    x0: int,
    x1: int,
    r0: int,
    r1: int,
    blocked_cols: set[int],
    valid_rows: list[int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    assert x1 > x0
    turn_candidates = [c for c in range(x0 + 1, x1) if c not in blocked_cols]

    need_changes = 0 if r0 == r1 else 1
    extra_changes = 0
    if len(turn_candidates) >= need_changes + 2 and len(valid_rows) >= 2 and rng.random() < 0.5:
        extra_changes = 2 if r0 == r1 else 1
    n_changes = need_changes + extra_changes

    if n_changes == 0:
        return [(r0, x1)]

    change_cols = _sample_spaced(turn_candidates, n_changes, 2, rng)
    if change_cols is None:
        if r0 == r1:
            return [(r0, x1)]
        change_cols = _sample_spaced(turn_candidates, 1, 1, rng)
        assert change_cols is not None
        return [(r0, change_cols[0]), (r1, change_cols[0]), (r1, x1)]

    if n_changes == 1:
        return [(r0, change_cols[0]), (r1, change_cols[0]), (r1, x1)]

    if r0 == r1:
        mids = [r for r in valid_rows if r != r0]
        assert mids
        mid = int(rng.choice(mids))
        rows = [r0, mid, r1]
    else:
        mids = [r for r in valid_rows if r not in (r0, r1)]
        if not mids:
            mids = [r for r in valid_rows if r != r0]
        assert mids
        mid = int(rng.choice(mids))
        rows = [r0, mid, r1]

    points = []
    for col, ra, rb in zip(change_cols, rows[:-1], rows[1:]):
        points.append((ra, col))
        points.append((rb, col))
    points.append((r1, x1))
    return points


def _rasterize(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for i, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            segment = [(r0, c) for c in range(c0, c1 + step, step)]
        else:
            assert c0 == c1
            step = 1 if r1 >= r0 else -1
            segment = [(r, c0) for r in range(r0, r1 + step, step)]
        if i > 0:
            segment = segment[1:]
        coords.extend(segment)
    return coords


def _validate_simple_path(coords: list[tuple[int, int]], height: int, width: int) -> bool:
    if not coords:
        return False
    if len(coords) != len(set(coords)):
        return False
    if coords[0][1] != 0 or coords[-1][1] != width - 1:
        return False
    pt_set = set(coords)
    degrees = []
    for r, c in coords:
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) in pt_set:
                deg += 1
        degrees.append(deg)
    return degrees.count(1) == 2 and all(d in (1, 2) for d in degrees)


def _plan_expected_output(
    grid: np.ndarray,
    sep_r: int,
    top_cols: list[int],
    bottom_cols: list[int],
    region_color: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    width = grid.shape[1]
    top_intervals = _intervals(top_cols, width)
    bottom_intervals = _intervals(bottom_cols, width)
    for (half, idx), color in region_color.items():
        if half == 0:
            c0, c1 = top_intervals[idx]
            area = out[:sep_r, c0 : c1 + 1]
        else:
            c0, c1 = bottom_intervals[idx]
            area = out[sep_r + 1 :, c0 : c1 + 1]
        area[area == BG] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(10, 24)
    width = rng.randint(14, 28)
    sep_r = rng.randint(3, height - 4)

    top_cols = _sample_partition(width, rng)
    bottom_cols = _sample_partition(width, rng)
    if top_cols is None or bottom_cols is None:
        return None

    top_rows = _rows_for_half(0, sep_r, height)
    bottom_rows = _rows_for_half(1, sep_r, height)
    top_set = set(top_cols)
    bottom_set = set(bottom_cols)
    union_boundaries = top_set | bottom_set

    safe_sep = [c for c in range(2, width - 2) if c not in union_boundaries]
    feasible_crossings = []
    for k in range(4):
        xs = _sample_spaced(safe_sep, k, 4, rng)
        if xs is not None:
            spans = [b - a for a, b in zip([0] + xs, xs + [width - 1])]
            if min(spans) >= 4:
                feasible_crossings.append((k, xs))
    if not feasible_crossings:
        return None
    crossing_count, sep_crossings = rng.choice(feasible_crossings)

    start_half = rng.randrange(2)
    current_half = start_half
    current_row = _choose_row(_rows_for_half(current_half, sep_r, height), None, rng, prefer_change=False)

    points = [(current_row, 0)]
    all_spans = sep_crossings + [width - 1]
    for i, x1 in enumerate(all_spans):
        span_rows = _rows_for_half(current_half, sep_r, height)
        blocked = top_set if current_half == 0 else bottom_set
        is_last = i == len(all_spans) - 1
        exit_row = _choose_row(span_rows, current_row, rng, prefer_change=not is_last)
        points.extend(_build_span_points(points[-1][1], x1, current_row, exit_row, blocked, span_rows, rng))
        current_row = exit_row
        if not is_last:
            current_half = 1 - current_half
            next_rows = _rows_for_half(current_half, sep_r, height)
            next_row = _choose_row(next_rows, None, rng, prefer_change=False)
            points.append((next_row, x1))
            current_row = next_row

    coords = _rasterize(points)
    if not _validate_simple_path(coords, height, width):
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_r, :] = GREY
    for c in top_cols:
        grid[: sep_r + 1, c] = GREY
    for c in bottom_cols:
        grid[sep_r:, c] = GREY

    def region_at(r: int, c: int):
        return _region_id(r, c, sep_r, top_cols, bottom_cols, top_set, bottom_set)

    trace_regions = [region_at(r, c) for r, c in coords]
    visited_order = []
    for rid in trace_regions:
        if rid is not None and (not visited_order or visited_order[-1] != rid):
            visited_order.append(rid)

    unique_regions = list(dict.fromkeys(visited_order))
    if not (3 <= len(unique_regions) <= len(PATH_PALETTE)):
        return None

    region_color = {
        rid: color
        for rid, color in zip(unique_regions, rng.sample(PATH_PALETTE, len(unique_regions)))
    }

    next_region = [None] * len(coords)
    future = None
    for i in range(len(coords) - 1, -1, -1):
        if trace_regions[i] is not None:
            future = trace_regions[i]
        next_region[i] = future

    prev_region = None
    for (r, c), rid, nxt in zip(coords, trace_regions, next_region):
        use_rid = rid if rid is not None else (nxt if nxt is not None else prev_region)
        if use_rid is None:
            return None
        grid[r, c] = region_color[use_rid]
        if rid is not None:
            prev_region = rid

    solved = _solve_grid(grid)
    expected = _plan_expected_output(grid, sep_r, top_cols, bottom_cols, region_color)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
