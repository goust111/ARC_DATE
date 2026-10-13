from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
MARKER_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

GRID_SIZE_RANGE = (13, 16)
MARKER_COUNT_OPTIONS = (3, 4)
MARKER_MARGIN = 2
MARKER_MIN_GAP = 3
WALL_COUNT_RANGE = (3, 4)
GEN_ATTEMPTS = 1500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(cells)
    return parts


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers = []
    for r, c in np.argwhere((grid != BLACK) & (grid != GREY) & (grid != RED)):
        markers.append((int(r), int(c), int(grid[r, c])))
    markers.sort()
    return markers


def _trace_rays(grid: np.ndarray) -> tuple[np.ndarray, list[list[set[int]]], int]:
    base = np.asarray(grid, dtype=int)
    h, w = base.shape
    markers = _extract_markers(base)
    marker_colors = {(r, c): color for r, c, color in markers}
    coverage = [[set() for _ in range(w)] for _ in range(h)]
    blocked_ray_count = 0

    for r, c, color in markers:
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            while _in_bounds(base.shape, rr, cc):
                if base[rr, cc] == GREY:
                    blocked_ray_count += 1
                    break
                coverage[rr][cc].add(color)
                rr += dr
                cc += dc

    painted = base.copy()
    for r in range(h):
        for c in range(w):
            if base[r, c] == GREY:
                continue
            if (r, c) in marker_colors:
                painted[r, c] = marker_colors[(r, c)]
                continue
            if not coverage[r][c]:
                continue
            painted[r, c] = RED if len(coverage[r][c]) >= 2 else next(iter(coverage[r][c]))

    return painted, coverage, blocked_ray_count


def _solve_with_options(
    grid: np.ndarray,
    *,
    red_counts_as_owner: bool = False,
) -> dict:
    ray_grid, _coverage, blocked_ray_count = _trace_rays(grid)
    out = ray_grid.copy()
    regions = []

    for cells in _components(ray_grid == BLACK):
        region_set = set(cells)
        boundary_colors: set[int] = set()
        touches_red = False
        touches_grey = False
        touches_edge = any(
            r == 0 or c == 0 or r == ray_grid.shape[0] - 1 or c == ray_grid.shape[1] - 1
            for r, c in cells
        )
        for r, c in cells:
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if not _in_bounds(ray_grid.shape, nr, nc) or (nr, nc) in region_set:
                    continue
                value = int(ray_grid[nr, nc])
                if value == RED:
                    touches_red = True
                    if red_counts_as_owner:
                        boundary_colors.add(RED)
                elif value == GREY:
                    touches_grey = True
                elif value != BLACK:
                    boundary_colors.add(value)

        fill_color = next(iter(boundary_colors)) if len(boundary_colors) == 1 else None
        if fill_color is not None:
            for r, c in cells:
                out[r, c] = fill_color

        regions.append(
            {
                "cells": cells,
                "size": len(cells),
                "boundary_colors": set(boundary_colors) - ({RED} if red_counts_as_owner else set()),
                "touches_red": touches_red,
                "touches_grey": touches_grey,
                "touches_edge": touches_edge,
                "fill_color": None if fill_color == RED else fill_color,
            }
        )

    return {
        "ray_grid": ray_grid,
        "output": out,
        "regions": regions,
        "markers": _extract_markers(grid),
        "red_count": int(np.count_nonzero(ray_grid == RED)),
        "blocked_ray_count": blocked_ray_count,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend wall-blocked marker rays, mark cross-source overlaps red, then flood-fill only uniquely owned black regions.

    Input structure:
    - A black background contains 3-4 isolated single-pixel markers in distinct non-red, non-grey colors.
    - Marker rows and columns are all distinct, and grey wall pixels form short horizontal or vertical blocker segments.
    - No other foreground objects are present in the input.

    Rule:
    - From each marker, extend rays up, down, left, and right until a grey wall or the grid edge stops the ray.
    - Paint each reached cell with that marker color; if a cell is reached by different markers, paint it red, while marker pixels themselves keep their original colors.
    - Treat the resulting rays, red crossings, grey walls, and the grid boundary as region boundaries. For each black 4-connected region, collect the boundary's non-grey, non-red ray colors. If exactly one marker color appears, flood-fill that region with that color; otherwise leave it black.

    Key insight:
    - Grey walls matter twice: they stop rays and they change which ray colors appear on each region boundary, while red crossings are visible boundaries but never count as ownership evidence.
    """
    return _solve_with_options(np.asarray(grid, dtype=int))["output"]


def _spaced_positions(
    rng: random.Random,
    start: int,
    stop: int,
    count: int,
    min_gap: int,
) -> list[int]:
    combos = [
        combo
        for combo in itertools.combinations(range(start, stop + 1), count)
        if all(b - a >= min_gap for a, b in zip(combo, combo[1:]))
    ]
    if not combos:
        raise ValueError("no spaced position combination available")
    return list(rng.choice(combos))


def _cells_clear_for_wall(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    marker_positions: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    wall_cells = set(map(tuple, np.argwhere(grid == GREY)))
    for r, c in cells:
        if not _in_bounds((h, w), r, c) or grid[r, c] != BLACK:
            return False
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if (nr, nc) in marker_positions or (nr, nc) in wall_cells:
                return False
    return True


def _try_place_wall(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    marker_index: int,
    orientation: str,
    side: str,
    rng: random.Random,
    preferred_values: list[int] | None = None,
) -> bool:
    h, w = grid.shape
    r, c, _color = markers[marker_index]
    marker_positions = {(mr, mc) for mr, mc, _ in markers}

    if orientation == "v":
        columns = list(range(1, c)) if side == "left" else list(range(c + 1, w - 1))
        rng.shuffle(columns)
        if preferred_values is not None:
            preferred = [value for value in preferred_values if value in columns]
            others = [value for value in columns if value not in preferred]
            columns = preferred + others
        lengths = [2, 3, 4]
        rng.shuffle(lengths)
        for wall_c in columns:
            for length in lengths:
                r0_lo = max(1, r - length + 1)
                r0_hi = min(r, h - length - 1)
                if r0_lo > r0_hi:
                    continue
                starts = list(range(r0_lo, r0_hi + 1))
                rng.shuffle(starts)
                for r0 in starts:
                    cells = [(r0 + i, wall_c) for i in range(length)]
                    if _cells_clear_for_wall(grid, cells, marker_positions):
                        for rr, cc in cells:
                            grid[rr, cc] = GREY
                        return True
        return False

    rows = list(range(1, r)) if side == "up" else list(range(r + 1, h - 1))
    rng.shuffle(rows)
    if preferred_values is not None:
        preferred = [value for value in preferred_values if value in rows]
        others = [value for value in rows if value not in preferred]
        rows = preferred + others
    lengths = [2, 3, 4]
    rng.shuffle(lengths)
    for wall_r in rows:
        for length in lengths:
            c0_lo = max(1, c - length + 1)
            c0_hi = min(c, w - length - 1)
            if c0_lo > c0_hi:
                continue
            starts = list(range(c0_lo, c0_hi + 1))
            rng.shuffle(starts)
            for c0 in starts:
                cells = [(wall_r, c0 + i) for i in range(length)]
                if _cells_clear_for_wall(grid, cells, marker_positions):
                    for rr, cc in cells:
                        grid[rr, cc] = GREY
                    return True
    return False


def _place_walls(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    wall_count: int,
    rng: random.Random,
) -> bool:
    last = len(markers) - 1
    descending = markers[0][1] > markers[last][1]
    top_inward_v = (0, "v", "left" if descending else "right")
    top_inward_h = (0, "h", "down")
    bottom_inward_v = (last, "v", "right" if descending else "left")
    bottom_inward_h = (last, "h", "up")

    chosen_specs = [top_inward_v, bottom_inward_h]
    optional_specs = [top_inward_h, bottom_inward_v]
    for idx in range(1, last):
        optional_specs.extend(
            [
                (idx, "v", "left"),
                (idx, "v", "right"),
                (idx, "h", "up"),
                (idx, "h", "down"),
            ]
        )
    rng.shuffle(optional_specs)
    for spec in optional_specs:
        if len(chosen_specs) >= wall_count:
            break
        if spec not in chosen_specs:
            chosen_specs.append(spec)

    if len(chosen_specs) < wall_count:
        return False

    def preferred_values(spec: tuple[int, str, str]) -> list[int] | None:
        idx, orientation, side = spec
        if idx == 0 and last >= 1:
            ref_r, ref_c, _ = markers[1]
        elif idx == last and last >= 1:
            ref_r, ref_c, _ = markers[last - 1]
        else:
            return None

        if orientation == "v":
            if side == "left":
                return list(range(1, ref_c))
            return list(range(ref_c + 1, grid.shape[1] - 1))
        if side == "up":
            return list(range(1, ref_r))
        return list(range(ref_r + 1, grid.shape[0] - 1))

    for idx, orientation, side in chosen_specs:
        if not _try_place_wall(
            grid,
            markers,
            idx,
            orientation,
            side,
            rng,
            preferred_values=preferred_values((idx, orientation, side)),
        ):
            return False
    return True


def _wall_components_are_valid(grid: np.ndarray) -> bool:
    comps = _components(grid == GREY)
    if not comps:
        return False
    for comp in comps:
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(comp) < 2 or len(comp) > 4:
            return False
        if len(rows) != 1 and len(cols) != 1:
            return False
        if len(rows) == 1:
            ordered = sorted(c for _, c in comp)
            if ordered != list(range(ordered[0], ordered[-1] + 1)):
                return False
        if len(cols) == 1:
            ordered = sorted(r for r, _ in comp)
            if ordered != list(range(ordered[0], ordered[-1] + 1)):
                return False
    return True


def _markers_are_valid(grid: np.ndarray) -> bool:
    markers = _extract_markers(grid)
    if len(markers) not in MARKER_COUNT_OPTIONS:
        return False
    rows = [r for r, _, _ in markers]
    cols = [c for _, c, _ in markers]
    colors = [color for _, _, color in markers]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols) or len(set(colors)) != len(colors):
        return False
    for r, c, _ in markers:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BLACK:
                return False
    return True


def _semantic_checks(grid: np.ndarray) -> bool:
    if not _markers_are_valid(grid) or not _wall_components_are_valid(grid):
        return False

    solved = _solve_with_options(grid)
    unique_regions = [region for region in solved["regions"] if region["fill_color"] is not None]
    ambiguous_regions = [region for region in solved["regions"] if region["fill_color"] is None]
    unique_colors = {region["fill_color"] for region in unique_regions}

    if solved["red_count"] < 1 or solved["blocked_ray_count"] < 2:
        return False
    if len(unique_regions) < 2 or len(ambiguous_regions) < 1 or len(unique_colors) < 2:
        return False
    if np.count_nonzero(solved["output"] != solved["ray_grid"]) < 6:
        return False
    if not any(region["touches_grey"] for region in unique_regions):
        return False
    if not any(len(region["boundary_colors"]) >= 2 for region in ambiguous_regions):
        return False

    wrong_red = _solve_with_options(grid, red_counts_as_owner=True)["output"]
    if np.array_equal(wrong_red, solved["output"]):
        return False

    no_walls = np.where(grid == GREY, BLACK, grid)
    if np.array_equal(main(no_walls), solved["output"]):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    marker_count = rng.choice(MARKER_COUNT_OPTIONS)

    try:
        rows = _spaced_positions(rng, MARKER_MARGIN, h - MARKER_MARGIN - 1, marker_count, MARKER_MIN_GAP)
        cols = _spaced_positions(rng, MARKER_MARGIN, w - MARKER_MARGIN - 1, marker_count, MARKER_MIN_GAP)
    except ValueError:
        return None

    descending = rng.random() < 0.5
    paired_cols = list(reversed(cols)) if descending else list(cols)
    colors = rng.sample(MARKER_COLORS, marker_count)

    grid = np.zeros((h, w), dtype=int)
    markers = []
    for idx in range(marker_count):
        r, c, color = rows[idx], paired_cols[idx], colors[idx]
        grid[r, c] = color
        markers.append((r, c, color))

    wall_count = rng.randint(*WALL_COUNT_RANGE)
    if not _place_walls(grid, markers, wall_count, rng):
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition flood-fill input")
