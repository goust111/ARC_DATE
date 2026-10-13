from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_opus_6_004_3",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4


BG = Color.BLACK
PATH_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
BARRIER = Color.GREY
MARKER = Color.PINK
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


def _path_cells(grid: np.ndarray) -> set[Coord]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(np.isin(grid, np.asarray(PATH_COLORS, dtype=int)))
    }


def _ordered_path(path: set[Coord]) -> list[Coord]:
    degrees = {
        cell: sum((nbr in path) for nbr in neighbors4(*cell))
        for cell in path
    }
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees.values()), "path must be branch-free"

    start = min(endpoints, key=lambda rc: (rc[1], rc[0]))
    order = [start]
    prev: Coord | None = None
    cur = start
    while True:
        nxt = [nbr for nbr in neighbors4(*cur) if nbr in path and nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unique"
        prev, cur = cur, nxt[0]
        order.append(cur)

    assert len(order) == len(path), "path traversal must cover every path cell"
    return order


def _marker_path_neighbor(grid: np.ndarray, marker: Coord, path: set[Coord]) -> Coord:
    rr, cc = marker
    neighbors = [
        (nr, nc)
        for nr, nc in neighbors4(rr, cc)
        if in_bounds(grid.shape, nr, nc) and (nr, nc) in path
    ]
    assert len(neighbors) == 1, "each marker must touch exactly one path pixel"
    return neighbors[0]


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == int(BG), "background must be black"

    colors = {int(v) for v in np.unique(g)}
    assert colors <= {BG, BARRIER, MARKER, *PATH_COLORS}

    barrier_coords = np.argwhere(g == int(BARRIER))
    assert barrier_coords.size > 0, "expected one grey barrier segment"
    barrier_cols = {int(c) for _, c in barrier_coords}
    assert len(barrier_cols) == 1, "barrier must be vertical"
    barrier_col = barrier_cols.pop()
    barrier_rows_present = sorted(int(r) for r, _ in barrier_coords)
    r0, r1 = barrier_rows_present[0], barrier_rows_present[-1]
    full_rows = list(range(r0, r1 + 1))
    gap_rows = [r for r in full_rows if int(g[r, barrier_col]) != int(BARRIER)]
    assert len(gap_rows) == 1, "barrier must have exactly one gap"
    gap_row = gap_rows[0]
    assert 5 <= (r1 - r0 + 1) <= 8, "barrier length must be 5-8"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == int(MARKER))]
    assert len(markers) == 2, "expected exactly two markers"
    left_markers = [m for m in markers if m[1] < barrier_col]
    right_markers = [m for m in markers if m[1] > barrier_col]
    assert len(left_markers) == 1 and len(right_markers) == 1, "markers must straddle the barrier"
    left_marker = left_markers[0]
    right_marker = right_markers[0]

    path = _path_cells(g)
    assert len(path) >= 10, "path must be long enough"
    ordered = _ordered_path(path)
    if ordered[0][1] > ordered[-1][1]:
        ordered = list(reversed(ordered))
    assert ordered[0][1] == 0 and ordered[-1][1] == g.shape[1] - 1, "path must span left to right"

    barrier_hits = [cell for cell in ordered if cell[1] == barrier_col]
    assert barrier_hits == [(gap_row, barrier_col)], "path must cross the barrier through the unique gap"

    left_anchor = _marker_path_neighbor(g, left_marker, path)
    right_anchor = _marker_path_neighbor(g, right_marker, path)
    assert left_anchor[1] < barrier_col and right_anchor[1] > barrier_col

    index = {cell: i for i, cell in enumerate(ordered)}
    li = index[left_anchor]
    ri = index[right_anchor]
    assert li < ri, "left marker anchor must appear before the right marker anchor"

    segment = ordered[li : ri + 1]
    colors_on_segment = [int(g[r, c]) for r, c in segment]
    assert len(segment) >= 4, "reversed segment must have at least 4 cells"
    assert li >= 2 and len(ordered) - 1 - ri >= 2, "path must extend on both sides of the markers"
    assert colors_on_segment != list(reversed(colors_on_segment)), "reversal must change the segment colors"

    for (r1a, c1a), (r2a, c2a) in zip(ordered, ordered[1:]):
        assert abs(r1a - r2a) + abs(c1a - c2a) == 1
        assert int(g[r1a, c1a]) != int(g[r2a, c2a]), "adjacent path colors must differ"

    return {
        "grid": g,
        "ordered": ordered,
        "segment": segment,
        "left_marker": left_marker,
        "right_marker": right_marker,
        "left_anchor_index": li,
        "right_anchor_index": ri,
        "barrier_col": barrier_col,
        "gap_row": gap_row,
    }


def _expected_output(grid: np.ndarray, info: dict[str, object]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    segment = info["segment"]
    colors = [int(g[r, c]) for r, c in segment]
    for (r, c), color in zip(segment, reversed(colors)):
        out[r, c] = int(color)
    out[info["left_marker"]] = int(BG)
    out[info["right_marker"]] = int(BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reverse the path colors between the paired cross-barrier markers, then erase the markers.

    Input structure:
    - Black background with one vertical grey barrier segment that has exactly one gap.
    - A single 4-connected one-pixel-wide multicolor path runs from the left edge to the right edge and crosses the barrier at the gap.
    - Two magenta marker pixels lie on opposite sides of the barrier, each touching exactly one path pixel.

    Rule:
    - Identify the left and right markers and the unique path pixel adjacent to each of them.
    - Traverse the path from the left marker's adjacent path pixel to the right marker's adjacent path pixel.
    - Reverse only that subpath's color sequence while keeping the path geometry fixed.
    - Erase the two magenta markers; keep the barrier, the rest of the path, and the background unchanged.

    Key insight:
    - The paired markers do not move anything themselves; they define the exact path interval whose colors must be reversed across the obstacle crossing.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    return _expected_output(np.asarray(grid, dtype=int), info)


def _append_line(path: list[Coord], target: Coord, seen: set[Coord]) -> None:
    r, c = path[-1]
    tr, tc = target
    assert r == tr or c == tc
    if r == tr:
        step = 1 if tc > c else -1
        for cc in range(c + step, tc + step, step):
            cell = (r, cc)
            assert cell not in seen
            path.append(cell)
            seen.add(cell)
    else:
        step = 1 if tr > r else -1
        for rr in range(r + step, tr + step, step):
            cell = (rr, c)
            assert cell not in seen
            path.append(cell)
            seen.add(cell)


def _build_path(height: int, width: int, barrier_col: int, gap_row: int) -> list[Coord] | None:
    if barrier_col < 4 or barrier_col > width - 5:
        return None

    start_row = random.randint(1, height - 2)
    end_row = random.randint(1, height - 2)
    approach_col = random.randint(2, barrier_col - 1)
    post_col = random.randint(barrier_col + 1, width - 3)

    max_left_internal = min(2, approach_col - 1)
    left_internal_count = random.randint(0, max_left_internal)
    left_pool = list(range(1, approach_col))
    left_cols = sorted(random.sample(left_pool, left_internal_count)) if left_internal_count else []

    right_pool = list(range(post_col + 1, width - 1))
    max_right_internal = min(2, len(right_pool))
    right_internal_count = random.randint(0, max_right_internal)
    right_cols = sorted(random.sample(right_pool, right_internal_count)) if right_internal_count else []

    path = [(start_row, 0)]
    seen = {path[0]}
    current_row = start_row

    for col in left_cols:
        _append_line(path, (current_row, col), seen)
        next_row_choices = [r for r in range(1, height - 1) if r != current_row]
        next_row = random.choice(next_row_choices)
        _append_line(path, (next_row, col), seen)
        current_row = next_row

    _append_line(path, (current_row, approach_col), seen)
    if current_row != gap_row:
        _append_line(path, (gap_row, approach_col), seen)
        current_row = gap_row

    _append_line(path, (gap_row, post_col), seen)
    current_row = gap_row

    for col in right_cols:
        _append_line(path, (current_row, col), seen)
        next_row_choices = [r for r in range(1, height - 1) if r != current_row]
        next_row = random.choice(next_row_choices)
        _append_line(path, (next_row, col), seen)
        current_row = next_row

    _append_line(path, (current_row, width - 1), seen)
    if current_row != end_row:
        _append_line(path, (end_row, width - 1), seen)

    degrees = {
        cell: sum((nbr in seen) for nbr in neighbors4(*cell))
        for cell in seen
    }
    if sum(deg == 1 for deg in degrees.values()) != 2 or any(deg not in (1, 2) for deg in degrees.values()):
        return None
    return path


def _marker_candidate(
    path: list[Coord],
    barrier_col: int,
    delta_c: int,
    grid_shape: tuple[int, int],
) -> list[tuple[int, Coord, Coord]]:
    path_set = set(path)
    out: list[tuple[int, Coord, Coord]] = []
    for idx, (r, c) in enumerate(path):
        marker = (r, c + delta_c)
        if not in_bounds(grid_shape, marker[0], marker[1]):
            continue
        if marker in path_set:
            continue
        if delta_c == 1 and not (marker[1] < barrier_col):
            continue
        if delta_c == -1 and not (marker[1] > barrier_col):
            continue
        if any(
            in_bounds(grid_shape, nr, nc) and (nr, nc) in path_set
            for nr, nc in neighbors4(*marker)
            if (nr, nc) != (r, c)
        ):
            continue
        out.append((idx, (r, c), marker))
    return out


def _choose_markers(path: list[Coord], barrier_col: int, grid_shape: tuple[int, int]) -> tuple[Coord, Coord] | None:
    left_candidates = [
        item
        for item in _marker_candidate(path, barrier_col, 1, grid_shape)
        if item[1][1] <= barrier_col - 2
    ]
    right_candidates = [
        item
        for item in _marker_candidate(path, barrier_col, -1, grid_shape)
        if item[1][1] >= barrier_col + 2
    ]
    valid_pairs: list[tuple[Coord, Coord]] = []
    for li, _left_anchor, left_marker in left_candidates:
        if li < 2:
            continue
        for ri, _right_anchor, right_marker in right_candidates:
            if ri <= li or len(path) - 1 - ri < 2:
                continue
            if ri - li + 1 < 4:
                continue
            valid_pairs.append((left_marker, right_marker))
    if not valid_pairs:
        return None
    return random.choice(valid_pairs)


def _assign_path_colors(path: list[Coord], segment: list[Coord]) -> dict[Coord, int] | None:
    path_set = set(path)
    segment_set = set(segment)
    for _ in range(80):
        colors: dict[Coord, int] = {}
        last_color = None
        for cell in path:
            choices = [color for color in PATH_COLORS if color != last_color]
            color = random.choice(choices)
            colors[cell] = int(color)
            last_color = color
        segment_colors = [colors[cell] for cell in segment]
        if segment_colors == list(reversed(segment_colors)):
            continue
        if any(colors[a] == colors[b] for a, b in zip(path, path[1:])):
            continue
        for cell in segment_set:
            assert cell in path_set
        return colors
    return None


def _build_expected_from_structure(
    grid: np.ndarray,
    path: list[Coord],
    left_marker: Coord,
    right_marker: Coord,
) -> np.ndarray:
    info = _analyze_grid(grid)
    assert info["left_marker"] == left_marker and info["right_marker"] == right_marker
    return _expected_output(grid, info)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 16)
    width = random.randint(15, 18)

    center = width // 2
    barrier_choices = [c for c in (center - 2, center - 1, center + 1, center + 2) if 4 <= c <= width - 5]
    if not barrier_choices:
        return None
    barrier_col = random.choice(barrier_choices)
    gap_row = random.randint(2, height - 3)

    path = _build_path(height, width, barrier_col, gap_row)
    if path is None:
        return None

    markers = _choose_markers(path, barrier_col, (height, width))
    if markers is None:
        return None
    left_marker, right_marker = markers

    path_index = {cell: i for i, cell in enumerate(path)}
    left_anchor = next(nbr for nbr in neighbors4(*left_marker) if nbr in path_index)
    right_anchor = next(nbr for nbr in neighbors4(*right_marker) if nbr in path_index)
    li = path_index[left_anchor]
    ri = path_index[right_anchor]
    segment = path[li : ri + 1]

    colors = _assign_path_colors(path, segment)
    if colors is None:
        return None

    barrier_len = random.randint(5, min(8, height - 2))
    top_lo = max(1, gap_row - barrier_len + 1)
    top_hi = min(gap_row, height - barrier_len - 1)
    if top_lo > top_hi:
        return None
    barrier_top = random.randint(top_lo, top_hi)

    grid = np.full((height, width), int(BG), dtype=int)
    for r in range(barrier_top, barrier_top + barrier_len):
        if r != gap_row:
            grid[r, barrier_col] = int(BARRIER)
    for cell, color in colors.items():
        grid[cell] = int(color)
    grid[left_marker] = int(MARKER)
    grid[right_marker] = int(MARKER)

    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return None

    expected = _build_expected_from_structure(grid, path, left_marker, right_marker)
    if not np.array_equal(expected, main(grid)):
        return None

    out = expected
    if int(np.count_nonzero(grid == int(MARKER))) != 2 or int(np.count_nonzero(out == int(MARKER))) != 0:
        return None
    if np.array_equal(grid, out):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-barrier reverse-path input")


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
