from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


WALL = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield int(r) + dr, int(c) + dc


def _label_background(g: np.ndarray, bg: int) -> tuple[np.ndarray, list[list[Coord]], list[bool]]:
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[Coord]] = []
    touches_boundary: list[bool] = []

    for sr, sc in zip(*np.where(g == bg), strict=False):
        sr = int(sr)
        sc = int(sc)
        if labels[sr, sc] != -1:
            continue

        cid = len(comps)
        q: deque[Coord] = deque([(sr, sc)])
        labels[sr, sc] = cid
        pts: list[Coord] = []
        touches = False

        while q:
            r, c = q.popleft()
            pts.append((r, c))
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                touches = True
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(g.shape, nr, nc) and labels[nr, nc] == -1 and int(g[nr, nc]) == bg:
                    labels[nr, nc] = cid
                    q.append((nr, nc))

        comps.append(pts)
        touches_boundary.append(touches)

    return labels, comps, touches_boundary


def _chamber_labels(
    g: np.ndarray, bg: int
) -> tuple[np.ndarray, list[list[Coord]], np.ndarray, list[bool]]:
    bg_labels, bg_comps, touches = _label_background(g, bg)
    chamber_labels = np.full(g.shape, -1, dtype=int)
    chambers: list[list[Coord]] = []
    for old_id, pts in enumerate(bg_comps):
        if touches[old_id]:
            continue
        cid = len(chambers)
        chambers.append(pts)
        for r, c in pts:
            chamber_labels[r, c] = cid
    return chamber_labels, chambers, bg_labels, touches


def _marker_direction(
    g: np.ndarray,
    marker: Coord,
    bg: int,
    bg_labels: np.ndarray,
    touches_boundary: list[bool],
) -> Coord:
    wall_nbrs = [(nr, nc) for nr, nc in _neighbors4(*marker) if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == WALL]
    assert len(wall_nbrs) == 1, "each marker must touch exactly one wall cell"

    wr, wc = wall_nbrs[0]
    step = (int(wr - marker[0]), int(wc - marker[1]))

    back = (marker[0] - step[0], marker[1] - step[1])
    if _in_bounds(g.shape, *back):
        assert int(g[back]) == bg, "the side behind a marker must be exterior background"
        bid = int(bg_labels[back])
        assert bid >= 0 and touches_boundary[bid], "marker must sit outside the structure"

    return step


def _ray_chambers(chamber_labels: np.ndarray, marker: Coord, step: Coord) -> list[int]:
    h, w = chamber_labels.shape
    order: list[int] = []
    seen: set[int] = set()
    last = -1
    r, c = marker

    while True:
        r += step[0]
        c += step[1]
        if not (0 <= r < h and 0 <= c < w):
            break

        cid = int(chamber_labels[r, c])
        if cid < 0:
            last = -1
            continue
        if cid == last:
            continue
        assert cid not in seen, "a marker ray must not re-enter the same chamber"
        order.append(cid)
        seen.add(cid)
        last = cid

    assert order, "each marker ray must pass through at least one enclosed chamber"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint every other enclosed chamber crossed by each outside marker ray.

    Input structure:
    - The background is the dominant color, and grey wall cells form one outer
      structure with internal grey partitions.
    - The chambers are enclosed 4-connected background regions inside the walls.
    - Colored marker pixels sit outside the structure, each touching exactly one
      outer wall cell.

    Rule:
    - A marker looks straight through its adjacent wall cell, in the direction
      from marker to wall.
    - Along that ray, list the enclosed background chambers in first-entry order.
    - Fill the 1st, 3rd, 5th, ... chambers in that list with the marker color;
      leave even-indexed chambers, walls, markers, and exterior background unchanged.

    Key insight:
    The ray only orders whole enclosed chambers; the painted object is the full
    chamber region, not the one-cell-wide ray path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    assert bg != WALL and np.any(g == WALL), "expected grey walls on a non-grey background"

    chamber_id, chambers, bg_labels, touches_boundary = _chamber_labels(g, bg)
    assert chambers, "expected at least one enclosed chamber"

    markers = [tuple(map(int, p)) for p in np.argwhere((g != bg) & (g != WALL))]
    assert markers, "expected outside marker pixels"

    decisions: dict[int, int | None] = {}
    for marker in markers:
        marker_color = int(g[marker])
        step = _marker_direction(g, marker, bg, bg_labels, touches_boundary)
        order = _ray_chambers(chamber_id, marker, step)
        for idx, cid in enumerate(order):
            decision = marker_color if idx % 2 == 0 else None
            if cid in decisions:
                assert decisions[cid] == decision, "overlapping marker rays must agree"
            else:
                decisions[cid] = decision

    out = g.copy()
    for cid, color in decisions.items():
        if color is None:
            continue
        for r, c in chambers[cid]:
            out[r, c] = int(color)
    return out


# --------------- generator ---------------

BG_CHOICES = (0, 7, 8)
MARKER_PALETTE = tuple(c for c in range(10) if c != WALL)
MAX_ATTEMPTS = 500
CHAMBER_SIZE_RANGE = (2, 4)
ROW_COL_COUNT_RANGE = (2, 5)
MARKER_COUNT_MAX = 4


def _wall_positions(sizes: list[int]) -> list[int]:
    pos = [0]
    cur = 0
    for size in sizes:
        cur += int(size) + 1
        pos.append(cur)
    return pos


def _chamber_ranges(sizes: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 1
    for size in sizes:
        end = start + int(size) - 1
        ranges.append((start, end))
        start = end + 2
    return ranges


def _sample_sizes(count: int) -> list[int]:
    lo, hi = CHAMBER_SIZE_RANGE
    return [random.randint(lo, hi) for _ in range(int(count))]


def _draw_wall_grid(
    grid: np.ndarray,
    r0: int,
    c0: int,
    row_sizes: list[int],
    col_sizes: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]:
    sh = sum(row_sizes) + len(row_sizes) + 1
    sw = sum(col_sizes) + len(col_sizes) + 1
    row_walls = _wall_positions(row_sizes)
    col_walls = _wall_positions(col_sizes)

    for rr in row_walls:
        grid[r0 + rr, c0 : c0 + sw] = WALL
    for cc in col_walls:
        grid[r0 : r0 + sh, c0 + cc] = WALL

    row_ranges = [(r0 + a, r0 + b) for a, b in _chamber_ranges(row_sizes)]
    col_ranges = [(c0 + a, c0 + b) for a, b in _chamber_ranges(col_sizes)]
    return row_ranges, col_ranges, sh, sw


def _paint_expected(
    out: np.ndarray,
    mode: str,
    side: str,
    markers: list[tuple[int, int, int]],
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> None:
    if mode == "H":
        order = list(range(len(col_ranges)))
        if side == "right":
            order.reverse()
        for row_idx, _marker_pos, color in markers:
            rr0, rr1 = row_ranges[row_idx]
            for rank, col_idx in enumerate(order):
                if rank % 2 != 0:
                    continue
                cc0, cc1 = col_ranges[col_idx]
                out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)
    else:
        order = list(range(len(row_ranges)))
        if side == "bottom":
            order.reverse()
        for col_idx, _marker_pos, color in markers:
            cc0, cc1 = col_ranges[col_idx]
            for rank, row_idx in enumerate(order):
                if rank % 2 != 0:
                    continue
                rr0, rr1 = row_ranges[row_idx]
                out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)


def _ray_cells_for_marker(
    marker: Coord,
    step: Coord,
    shape: tuple[int, int],
) -> set[Coord]:
    cells: set[Coord] = set()
    r, c = marker
    while True:
        r += step[0]
        c += step[1]
        if not _in_bounds(shape, r, c):
            return cells
        cells.add((r, c))


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    mode: str,
    side: str,
    markers: list[tuple[int, int, int]],
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> bool:
    changed = {tuple(map(int, p)) for p in np.argwhere(inp != expected)}
    if not changed:
        return False

    ray_cells: set[Coord] = set()
    for _idx, marker, _color in markers:
        if mode == "H":
            step = (0, 1) if side == "left" else (0, -1)
        else:
            step = (1, 0) if side == "top" else (-1, 0)
        ray_cells |= _ray_cells_for_marker(marker, step, inp.shape)

    if not any(p not in ray_cells for p in changed):
        return False

    if mode == "H":
        chamber_count = len(col_ranges)
        axis_ranges = col_ranges
    else:
        chamber_count = len(row_ranges)
        axis_ranges = row_ranges
    if chamber_count < 3:
        return False

    # At least one even chamber on every ray must stay background, which rules out
    # "fill all chambers seen by the marker" shortcuts.
    bg = _get_bg(inp)
    for idx, _marker, _color in markers:
        even_rank = 1
        if mode == "H":
            cols = list(range(len(col_ranges)))
            if side == "right":
                cols.reverse()
            cc0, cc1 = axis_ranges[cols[even_rank]]
            rr0, rr1 = row_ranges[idx]
            if not np.all(expected[rr0 : rr1 + 1, cc0 : cc1 + 1] == bg):
                return False
        else:
            rows = list(range(len(row_ranges)))
            if side == "bottom":
                rows.reverse()
            rr0, rr1 = axis_ranges[rows[even_rank]]
            cc0, cc1 = col_ranges[idx]
            if not np.all(expected[rr0 : rr1 + 1, cc0 : cc1 + 1] == bg):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    mode = random.choice(("H", "V"))

    if mode == "H":
        n_rows = random.randint(*ROW_COL_COUNT_RANGE)
        n_cols = random.randint(3, ROW_COL_COUNT_RANGE[1])
    else:
        n_rows = random.randint(3, ROW_COL_COUNT_RANGE[1])
        n_cols = random.randint(*ROW_COL_COUNT_RANGE)

    row_sizes = _sample_sizes(n_rows)
    col_sizes = _sample_sizes(n_cols)
    sh = sum(row_sizes) + len(row_sizes) + 1
    sw = sum(col_sizes) + len(col_sizes) + 1

    top_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    left_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)
    h = sh + top_margin + bottom_margin
    w = sw + left_margin + right_margin
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), bg, dtype=int)
    row_ranges, col_ranges, _sh, _sw = _draw_wall_grid(
        grid,
        top_margin,
        left_margin,
        row_sizes,
        col_sizes,
    )

    available_colors = [c for c in MARKER_PALETTE if c != bg]
    if mode == "H":
        marker_count = random.randint(2, min(MARKER_COUNT_MAX, n_rows))
        chosen = sorted(random.sample(range(n_rows), marker_count))
        side = random.choice(("left", "right"))
    else:
        marker_count = random.randint(2, min(MARKER_COUNT_MAX, n_cols))
        chosen = sorted(random.sample(range(n_cols), marker_count))
        side = random.choice(("top", "bottom"))

    colors = random.sample(available_colors, marker_count)
    markers: list[tuple[int, Coord, int]] = []
    for idx, color in zip(chosen, colors, strict=False):
        if mode == "H":
            rr = random.randint(*row_ranges[idx])
            marker = (rr, left_margin - 1) if side == "left" else (rr, left_margin + sw)
        else:
            cc = random.randint(*col_ranges[idx])
            marker = (top_margin - 1, cc) if side == "top" else (top_margin + sh, cc)
        grid[marker] = int(color)
        markers.append((idx, marker, int(color)))

    expected = grid.copy()
    _paint_expected(expected, mode, side, markers, row_ranges, col_ranges)

    if _get_bg(grid) != bg:
        return None
    if not _semantic_ok(grid, expected, mode, side, markers, row_ranges, col_ranges):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
