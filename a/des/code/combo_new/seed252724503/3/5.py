import random
from dataclasses import dataclass

import numpy as np


BG = 0
RED = 2
GRAY = 5
PATH_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


@dataclass(frozen=True)
class PathInfo:
    color: int
    orientation: str  # "h" or "v"
    fixed: int        # row for horizontal, column for vertical
    lo: int
    hi: int
    start_low: bool   # start is at the low-coordinate end


def _endpoint_distance(r: int, c: int, h: int, w: int) -> int:
    return min(r, c, h - 1 - r, w - 1 - c)


def _read_legend(grid: np.ndarray) -> tuple[list[int], int, int]:
    cells = np.argwhere(grid != BG)
    assert cells.size > 0, "expected at least one non-background cell"

    legend_c = int(cells[:, 1].max())
    non_bg_rows = np.flatnonzero(grid[:, legend_c] != BG)
    assert non_bg_rows.size > 0, "expected a right-edge legend column"

    legend_bottom = int(non_bg_rows.max())
    legend_top = legend_bottom
    while legend_top > 0 and int(grid[legend_top - 1, legend_c]) != BG:
        legend_top -= 1

    colors = [int(v) for v in grid[legend_top : legend_bottom + 1, legend_c]]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return colors, legend_top, legend_c


def _extract_paths(grid: np.ndarray, legend_colors: list[int], legend_c: int) -> dict[int, PathInfo]:
    h, w = grid.shape
    paths: dict[int, PathInfo] = {}
    for color in legend_colors:
        mask = grid == color
        mask[:, legend_c] = False
        coords = np.argwhere(mask)
        assert coords.size > 0, f"legend color {color} must appear on a path"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            fixed = int(rows[0])
            lo = int(cols.min())
            hi = int(cols.max())
            d_lo = _endpoint_distance(fixed, lo, h, w)
            d_hi = _endpoint_distance(fixed, hi, h, w)
            start_low = d_lo <= d_hi
            info = PathInfo(color=color, orientation="h", fixed=fixed, lo=lo, hi=hi, start_low=start_low)
        else:
            assert len(cols) == 1, f"path color {color} must form one row or one column"
            fixed = int(cols[0])
            lo = int(rows.min())
            hi = int(rows.max())
            d_lo = _endpoint_distance(lo, fixed, h, w)
            d_hi = _endpoint_distance(hi, fixed, h, w)
            start_low = d_lo <= d_hi
            info = PathInfo(color=color, orientation="v", fixed=fixed, lo=lo, hi=hi, start_low=start_low)
        paths[color] = info
    return paths


def _distance_from_start(path: PathInfo, point: tuple[int, int]) -> int:
    axis = point[1] if path.orientation == "h" else point[0]
    return axis - path.lo if path.start_low else path.hi - axis


def _step_away_from_start(path: PathInfo) -> tuple[int, int]:
    if path.orientation == "h":
        return (0, 1 if path.start_low else -1)
    return (1 if path.start_low else -1, 0)


def _marker_cell(grid: np.ndarray, path: PathInfo, point: tuple[int, int]) -> tuple[int, int]:
    h, w = grid.shape
    dr, dc = _step_away_from_start(path)
    r, c = point
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        if int(grid[rr, cc]) == path.color:
            return rr, cc
        rr += dr
        cc += dc
    raise AssertionError(f"could not find stop marker cell for path color {path.color}")


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_colors, _legend_top, legend_c = _read_legend(g)
    rank = {color: i for i, color in enumerate(legend_colors)}
    paths = _extract_paths(g, legend_colors, legend_c)

    horizontal_by_row: dict[int, PathInfo] = {}
    vertical_by_col: dict[int, PathInfo] = {}
    for path in paths.values():
        if path.orientation == "h":
            assert path.fixed not in horizontal_by_row, "expected at most one horizontal path per row"
            horizontal_by_row[path.fixed] = path
        else:
            assert path.fixed not in vertical_by_col, "expected at most one vertical path per column"
            vertical_by_col[path.fixed] = path

    out = g.copy()
    total_crossings = {color: 0 for color in paths}
    losing_crossings = {color: [] for color in paths}

    for r, c in map(tuple, np.argwhere(g == GRAY)):
        r = int(r)
        c = int(c)
        hpath = horizontal_by_row.get(r)
        vpath = vertical_by_col.get(c)
        assert hpath is not None and vpath is not None, "every gray cell must be a row/column crossing"
        assert hpath.lo <= c <= hpath.hi
        assert vpath.lo <= r <= vpath.hi

        total_crossings[hpath.color] += 1
        total_crossings[vpath.color] += 1

        if rank[hpath.color] < rank[vpath.color]:
            winner, loser = hpath, vpath
        else:
            winner, loser = vpath, hpath

        out[r, c] = winner.color
        losing_crossings[loser.color].append((r, c))

    for color, losses in losing_crossings.items():
        if not losses:
            continue
        path = paths[color]
        losses = sorted(losses, key=lambda pt: _distance_from_start(path, pt))
        if len(losses) == total_crossings[color]:
            losses = [losses[-1]]
        for point in losses:
            mr, mc = _marker_cell(g, path, point)
            out[mr, mc] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve crossing priorities between colored paths and mark the losing branches with red stop pixels.

    Input structure:
    - The background is black.
    - Several single-pixel-wide colored paths run horizontally or vertically across most of the grid, and their crossings are shown as gray pixels.
    - A small legend column sits in the lower-right corner; its colors read from top to bottom give the path-priority order.

    Rule:
    - Read the lower-right legend column and rank path colors by height in that column.
    - Reconstruct each path from the non-legend cells of its color; every path is a single row or a single column interrupted only by gray crossings.
    - At each gray crossing, compare the horizontal and vertical path colors: the higher-priority color replaces the gray pixel.
    - For every losing path branch, mark the first same-color pixel just beyond that blocked crossing as red; if one path loses at all of its crossings, keep only the red mark for the loss farthest from that path's start end.

    Key insight:
    - The gray cells are not final colors; they are unresolved intersections whose winner is determined entirely by the bottom-right priority legend, while red marks encode where the lower-priority path finally gets cut.
    """
    return _solve(grid)


def _sample_spaced(rng: random.Random, low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if count <= 0 or len(values) < count:
        return None
    for _ in range(200):
        picks = sorted(rng.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _build_horizontal(rng: random.Random, row: int, width: int, color: int) -> PathInfo:
    if rng.random() < 0.5:
        lo = 1
        hi = width - 1 - rng.randint(2, 4)
        return PathInfo(color=color, orientation="h", fixed=row, lo=lo, hi=hi, start_low=True)
    lo = rng.randint(3, 5)
    hi = width - 3
    return PathInfo(color=color, orientation="h", fixed=row, lo=lo, hi=hi, start_low=False)


def _build_vertical(rng: random.Random, col: int, height: int, color: int) -> PathInfo:
    if rng.random() < 0.5:
        lo = 1
        hi = height - 1 - rng.randint(2, 4)
        return PathInfo(color=color, orientation="v", fixed=col, lo=lo, hi=hi, start_low=True)
    lo = rng.randint(3, 5)
    hi = height - 2
    return PathInfo(color=color, orientation="v", fixed=col, lo=lo, hi=hi, start_low=False)


def _validate_generated(grid: np.ndarray, expect_losers: int, expect_multi_loss: bool) -> bool:
    legend_colors, _legend_top, legend_c = _read_legend(grid)
    paths = _extract_paths(grid, legend_colors, legend_c)
    horizontals = [p for p in paths.values() if p.orientation == "h"]
    verticals = [p for p in paths.values() if p.orientation == "v"]
    if not horizontals or not verticals:
        return False

    gray_count = int(np.count_nonzero(grid == GRAY))
    if gray_count != len(horizontals) * len(verticals):
        return False

    out = main(grid)
    red_count = int(np.count_nonzero(out == RED))
    if red_count != expect_losers:
        return False
    if np.array_equal(out, grid):
        return False

    per_path_losses = []
    for color, path in paths.items():
        axis_losses = 0
        if path.orientation == "h":
            for vpath in verticals:
                if path.lo <= vpath.fixed <= path.hi and vpath.lo <= path.fixed <= vpath.hi:
                    axis_losses += 1
        else:
            for hpath in horizontals:
                if path.lo <= hpath.fixed <= path.hi and hpath.lo <= path.fixed <= hpath.hi:
                    axis_losses += 1
        if axis_losses:
            per_path_losses.append(axis_losses)
    if expect_multi_loss and max(per_path_losses) < 2:
        return False
    if not expect_multi_loss and min(per_path_losses) != 1:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    winner_orientation = rng.choice(("h", "v"))
    multi_loss = rng.choice((False, True))
    winner_count = rng.randint(2, 3) if multi_loss else 1
    loser_count = rng.randint(1, min(3, 6 - winner_count))
    if winner_orientation == "h":
        num_h, num_v = winner_count, loser_count
    else:
        num_h, num_v = loser_count, winner_count
    total_paths = num_h + num_v

    min_h = max(14, total_paths + 6, 2 * num_h + 10)
    min_w = max(16, 2 * num_v + 10)
    height = rng.randint(min_h, 28)
    width = rng.randint(min_w, 29)

    rows = _sample_spaced(rng, 6, height - 6, num_h, 2)
    cols = _sample_spaced(rng, 6, width - 6, num_v, 2)
    if rows is None or cols is None:
        return None

    colors = rng.sample(PATH_COLORS, total_paths)
    h_colors = colors[:num_h]
    v_colors = colors[num_h:]

    horizontal_paths = [_build_horizontal(rng, row, width, color) for row, color in zip(rows, h_colors)]
    vertical_paths = [_build_vertical(rng, col, height, color) for col, color in zip(cols, v_colors)]

    if winner_orientation == "h":
        winners = [p.color for p in horizontal_paths]
        losers = [p.color for p in vertical_paths]
    else:
        winners = [p.color for p in vertical_paths]
        losers = [p.color for p in horizontal_paths]
    legend_colors = rng.sample(winners, len(winners)) + rng.sample(losers, len(losers))

    grid = np.full((height, width), BG, dtype=int)
    for path in horizontal_paths:
        grid[path.fixed, path.lo : path.hi + 1] = path.color
    for path in vertical_paths:
        grid[path.lo : path.hi + 1, path.fixed] = path.color

    for hpath in horizontal_paths:
        for vpath in vertical_paths:
            if hpath.lo <= vpath.fixed <= hpath.hi and vpath.lo <= hpath.fixed <= vpath.hi:
                grid[hpath.fixed, vpath.fixed] = GRAY

    grid[:, width - 2] = BG
    legend_top = height - total_paths
    for i, color in enumerate(legend_colors):
        grid[legend_top + i, width - 1] = color

    if not _validate_generated(grid, expect_losers=loser_count, expect_multi_loss=multi_loss):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-priority input")
