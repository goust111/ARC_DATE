from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER_COLORS = (1, 2, 3, 4)
WALL = 5
START = 9
GEN_RETRY_LIMIT = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _separator_rows(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    rows = []
    for r in range(1, h - 1):
        inner = grid[r, 1 : w - 1]
        wall_count = int(np.sum(inner == WALL))
        non_walls = int(inner.size - wall_count)
        if wall_count >= w - 3 and non_walls <= 1:
            rows.append(r)
    assert rows, "expected at least one horizontal separator row"
    return rows


def _divider_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    wall_counts = np.sum(grid == WALL, axis=0)
    col = max(range(1, w - 1), key=lambda c: int(wall_counts[c]))
    assert int(wall_counts[col]) >= h // 2, "expected one dominant vertical divider"
    return int(col)


def _trace_path(grid: np.ndarray, start_col: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    r = h - 1
    c = int(start_col)
    path = [(r, c)]
    while True:
        if r == 0:
            break
        if int(grid[r - 1, c]) == WALL:
            moved = False
            for dc in (1, -1):
                nc = c + dc
                if 0 <= nc < w and int(grid[r, nc]) != WALL:
                    c = nc
                    path.append((r, c))
                    moved = True
                    break
            if not moved:
                break
        else:
            r -= 1
            path.append((r, c))
    return path


def _band_index(row: int, separators: list[int], height: int) -> int | None:
    bounds = [-1] + separators + [height]
    for i in range(len(bounds) - 1):
        if bounds[i] < row < bounds[i + 1]:
            return i
    return None


def _chamber_id(
    row: int, col: int, separators: list[int], divider: int, shape: tuple[int, int]
) -> tuple[int, int] | None:
    h, _w = shape
    band = _band_index(row, separators, h)
    if band is None or col == divider:
        return None
    return band, 0 if col < divider else 1


def _chamber_ranges(
    chamber: tuple[int, int], separators: list[int], divider: int, shape: tuple[int, int]
) -> tuple[range, range]:
    h, w = shape
    band, side = chamber
    bounds = [-1] + separators + [h]
    rows = range(bounds[band] + 1, bounds[band + 1])
    cols = range(0, divider) if side == 0 else range(divider + 1, w)
    return rows, cols


def _majority_color(values: list[int]) -> int:
    counts = Counter(values)
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique majority marker color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the maroon path upward and fill only the chambers it visits with each chamber's majority wall-adjacent marker color.

    Input structure:
    - The grid has black background, gray internal walls, one maroon start pixel on the bottom row, and colored marker pixels.
    - A dominant vertical divider and one or two dominant horizontal separator rows partition the canvas into 4 or 6 rectangular chambers.
    - Marker pixels sit inside chambers but orthogonally adjacent to nearby wall cells, and each chamber has a unique majority marker color.

    Rule:
    - Start from the maroon bottom pixel and try to move upward one cell at a time.
    - If the cell above is gray, sidestep right by one cell; if that is blocked by a wall or the grid edge, sidestep left by one cell instead; stop if neither sidestep is possible.
    - Record which wall-defined chambers contain path pixels. For each visited chamber, collect all orthogonally adjacent marker pixels touching that chamber and fill the chamber's background cells with the unique majority marker color.
    - Leave unvisited chambers black, preserve the wall and marker pixels, and paint every path cell maroon.

    Key insight:
    - The path does not decide the fill color; it decides which chambers are activated, while each activated chamber gets its own color from the local marker majority on its walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    start_cols = [c for c in range(w) if int(g[h - 1, c]) == START]
    assert len(start_cols) == 1, "expected exactly one bottom start pixel"

    separators = _separator_rows(g)
    divider = _divider_col(g)
    path = _trace_path(g, start_cols[0])
    path_set = set(path)

    visited: set[tuple[int, int]] = set()
    for r, c in path:
        chamber = _chamber_id(r, c, separators, divider, g.shape)
        if chamber is not None:
            visited.add(chamber)

    marker_set = set(MARKER_COLORS)
    out = g.copy()
    for chamber in visited:
        rows, cols = _chamber_ranges(chamber, separators, divider, g.shape)
        marker_hits: dict[tuple[int, int], int] = {}
        for r in rows:
            for c in cols:
                if int(g[r, c]) in marker_set or int(g[r, c]) == WALL:
                    continue
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in marker_set:
                        marker_hits[(nr, nc)] = int(g[nr, nc])
        assert marker_hits, "each visited chamber must touch marker pixels"
        fill = _majority_color(list(marker_hits.values()))
        for r in rows:
            for c in cols:
                if int(g[r, c]) == BG and (r, c) not in path_set:
                    out[r, c] = fill

    for r, c in path:
        out[r, c] = START
    return out


def _sample_band_heights(height: int, bands: int) -> list[int] | None:
    usable = height - (bands - 1)
    if usable < 3 * bands:
        return None
    if bands == 2:
        top = random.randint(3, usable - 3)
        return [top, usable - top]
    top = random.randint(3, usable - 6)
    middle = random.randint(3, usable - top - 3)
    bottom = usable - top - middle
    if bottom < 3:
        return None
    return [top, middle, bottom]


def _separator_layout(height: int, bands: int) -> list[int] | None:
    band_heights = _sample_band_heights(height, bands)
    if band_heights is None:
        return None
    rows = []
    cursor = band_heights[0]
    for i in range(bands - 1):
        rows.append(cursor)
        cursor += 1
        cursor += band_heights[i + 1]
    return rows


def _path_plan(
    separators: list[int], side_lo: int, side_hi: int
) -> tuple[int, list[dict[str, int | str]]] | None:
    if side_lo > side_hi:
        return None
    for _ in range(40):
        current = random.randint(side_lo, side_hi)
        start_col = current
        bottom_up_plan: list[dict[str, int | str]] = []
        shifted = False
        for _sep in reversed(separators):
            gap = random.randint(current, side_hi)
            if gap > current:
                shifted = True
            bottom_up_plan.append({"branch": "right", "gap": gap, "entry": current})
            current = gap
        if shifted or side_hi == start_col:
            return start_col, list(reversed(bottom_up_plan))
    return None


def _build_wall_grid(height: int, width: int) -> tuple[np.ndarray, int, list[int], int, list[dict[str, int | str]]] | None:
    bands = random.choice([2, 3])
    separators = _separator_layout(height, bands)
    if separators is None:
        return None

    divider = random.randint(3, width - 4)
    side = random.choice([0, 1])
    side_lo = 1 if side == 0 else divider + 1
    side_hi = divider - 1 if side == 0 else width - 3
    if side_hi - side_lo < 1:
        return None

    planned = _path_plan(separators, side_lo, side_hi)
    if planned is None:
        return None
    start_col, plan = planned

    grid = np.full((height, width), BG, dtype=int)
    grid[1 : height - 1, divider] = WALL

    for idx, row in enumerate(separators):
        grid[row, 1 : width - 1] = WALL
        gap = int(plan[idx]["gap"])
        grid[row, gap] = BG

    grid[height - 1, start_col] = START
    return grid, divider, separators, start_col, plan


def _expand_reserved(path: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    del shape
    return set(path)


def _candidate_marker_cells(
    grid: np.ndarray, chamber: tuple[int, int], separators: list[int], divider: int, reserved: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    h, w = grid.shape
    rows, cols = _chamber_ranges(chamber, separators, divider, grid.shape)
    out = []
    for r in rows:
        for c in cols:
            if int(grid[r, c]) != BG or (r, c) in reserved:
                continue
            if any(
                0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == WALL
                for nr, nc in _neighbors4(r, c)
            ):
                out.append((r, c))
    random.shuffle(out)
    return out


def _assign_majorities(chambers: list[tuple[int, int]], visited: set[tuple[int, int]]) -> dict[tuple[int, int], int] | None:
    for _ in range(200):
        majors = {chamber: random.choice(MARKER_COLORS) for chamber in chambers}
        if len(set(majors.values())) < 2:
            continue
        visited_colors = {majors[chamber] for chamber in visited}
        if len(visited) >= 2 and len(visited_colors) < 2:
            continue
        return majors
    return None


def _marker_palette_for_chamber(majority: int, count: int) -> list[int]:
    return [majority] * count


def _place_markers_for_chamber(
    grid: np.ndarray,
    chamber: tuple[int, int],
    separators: list[int],
    divider: int,
    reserved: set[tuple[int, int]],
    majority: int,
) -> bool:
    candidates = _candidate_marker_cells(grid, chamber, separators, divider, reserved)
    if len(candidates) < 2:
        return False

    count = 3 if len(candidates) >= 3 else 2
    if len(candidates) >= 4 and random.random() < 0.35:
        count = 4

    chosen = candidates[:count]
    colors = _marker_palette_for_chamber(majority, count)
    for (r, c), color in zip(chosen, colors):
        grid[r, c] = color
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        height = random.randint(13, 15)
        width = random.randint(9, 10)
        built = _build_wall_grid(height, width)
        if built is None:
            continue
        grid, divider, separators, start_col, _plan = built

        path = _trace_path(grid, start_col)
        path_set = set(path)
        visited = {
            chamber
            for r, c in path
            if (chamber := _chamber_id(r, c, separators, divider, grid.shape)) is not None
        }

        bounds = [-1] + separators + [height]
        chambers = [
            (band, side)
            for band in range(len(bounds) - 1)
            for side in (0, 1)
        ]
        if len(visited) < 2 or len(visited) == len(chambers):
            continue

        majorities = _assign_majorities(chambers, visited)
        if majorities is None:
            continue

        reserved = _expand_reserved(path_set, grid.shape)
        ok = True
        for chamber in chambers:
            if not _place_markers_for_chamber(
                grid, chamber, separators, divider, reserved, majorities[chamber]
            ):
                ok = False
                break
        if not ok:
            continue

        # Static semantic guard: every chamber must have a unique locally visible majority.
        for chamber in chambers:
            rows, cols = _chamber_ranges(chamber, separators, divider, grid.shape)
            seen: dict[tuple[int, int], int] = {}
            for r in rows:
                for c in cols:
                    if int(grid[r, c]) in MARKER_COLORS or int(grid[r, c]) == WALL:
                        continue
                    for nr, nc in _neighbors4(r, c):
                        if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) in MARKER_COLORS:
                            seen[(nr, nc)] = int(grid[nr, nc])
            if not seen:
                ok = False
                break
            if _majority_color(list(seen.values())) != majorities[chamber]:
                ok = False
                break
        if not ok:
            continue

        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
