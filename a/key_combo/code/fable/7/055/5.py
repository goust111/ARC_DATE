from __future__ import annotations

from collections import Counter
import random

import numpy as np


PATH_COLOR = 3
ALL_COLORS = tuple(range(10))
NON_PATH_COLORS = tuple(c for c in ALL_COLORS if c != PATH_COLOR)
GEN_ATTEMPTS = 400
WALL_COUNT_RANGE = (2, 4)
WIDTH_RANGE = (12, 20)
BAND_HEIGHT_RANGE = (3, 5)
BOTTOM_BAND_HEIGHT_RANGE = (2, 5)
GAP_COUNT_RANGE = (3, 5)
DISTRACTOR_COUNT_RANGE = (2, 5)


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color, matching the shared ARC toolbox convention."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _paint_segment(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    color = int(color)
    if r0 == r1:
        a, b = sorted((c0, c1))
        out[r0, a : b + 1] = color
        return
    if c0 == c1:
        a, b = sorted((r0, r1))
        out[a : b + 1, c0] = color
        return
    raise ValueError("segments must be axis-aligned")


def _segment_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    if r0 == r1:
        a, b = sorted((c0, c1))
        return [(r0, c) for c in range(a, b + 1)]
    if c0 == c1:
        a, b = sorted((r0, r1))
        return [(r, c0) for r in range(a, b + 1)]
    raise ValueError("segments must be axis-aligned")


def _find_walls(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    walls: list[dict] = []
    for r in range(h):
        row = g[r]
        fg = sorted(set(map(int, row)) - {int(bg)})
        if len(fg) != 1:
            continue
        wall_color = fg[0]
        gap_cols = np.flatnonzero(row == int(bg))
        if len(gap_cols) < 2:
            continue
        if int(np.count_nonzero(row == wall_color)) <= w // 2:
            continue
        walls.append({"row": int(r), "color": int(wall_color), "gaps": [int(c) for c in gap_cols]})

    assert walls, "expected at least one full-row wall"
    wall_rows = [wall["row"] for wall in walls]
    assert wall_rows == sorted(wall_rows), "wall rows must be ordered"
    assert len({wall["color"] for wall in walls}) == len(walls), "wall colors must be distinct"
    return walls


def _guide_for_band(grid: np.ndarray, band_top: int, wall_row: int, wall_color: int) -> tuple[int, int]:
    band = np.asarray(grid[band_top:wall_row, :], dtype=int)
    coords = np.argwhere(band == int(wall_color))
    assert len(coords) == 1, "each band must contain exactly one guide matching the next wall color"
    gr, gc = map(int, coords[0])
    return int(band_top + gr), int(gc)


def _trace_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    bg = get_bg(g)
    assert bg != PATH_COLOR, "background must differ from the green path color"

    start_cols = np.flatnonzero(g[0] == PATH_COLOR)
    assert len(start_cols) == 1, "expected exactly one green start pixel on the top edge"
    start_col = int(start_cols[0])

    walls = _find_walls(g, bg)
    assert walls[-1]["row"] < h - 1, "there must be a bottom band below the last wall"

    current_r = 0
    current_c = start_col
    path: list[tuple[int, int]] = []
    metadata: list[dict] = []

    for i, wall in enumerate(walls):
        band_top = 1 if i == 0 else walls[i - 1]["row"] + 1
        wall_row = int(wall["row"])
        wall_color = int(wall["color"])
        assert band_top < wall_row, "each wall must have a non-empty band above it"

        guide_r, guide_c = _guide_for_band(g, band_top, wall_row, wall_color)
        gaps = list(wall["gaps"])
        assert guide_c in gaps, "the matching guide must align with a gap in the next wall"

        path.extend(_segment_coords(current_r, current_c, guide_r, current_c))
        path.extend(_segment_coords(guide_r, current_c, guide_r, guide_c)[1:])
        path.extend(_segment_coords(guide_r, guide_c, wall_row + 1, guide_c)[1:])

        metadata.append(
            {
                "band_top": int(band_top),
                "band_bottom": int(wall_row - 1),
                "entry_col": int(current_c),
                "guide_row": int(guide_r),
                "guide_col": int(guide_c),
                "wall_row": int(wall_row),
                "wall_color": int(wall_color),
                "gaps": gaps,
            }
        )
        current_r = wall_row + 1
        current_c = guide_c

    path.extend(_segment_coords(current_r, current_c, h - 1, current_c)[1:])
    return path, metadata


def _output_from_path(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in path:
        out[int(r), int(c)] = int(PATH_COLOR)
    return out


def _nearest_gap_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    walls = _find_walls(g, bg)
    start_col = int(np.flatnonzero(g[0] == PATH_COLOR)[0])
    h = g.shape[0]

    out = g.copy()
    current_r = 0
    current_c = start_col
    for wall in walls:
        wall_row = int(wall["row"])
        band_top = current_r + 1
        chosen = min(wall["gaps"], key=lambda c: (abs(int(c) - current_c), int(c)))
        _paint_segment(out, current_r, current_c, band_top, current_c, PATH_COLOR)
        _paint_segment(out, band_top, current_c, band_top, chosen, PATH_COLOR)
        _paint_segment(out, band_top, chosen, wall_row + 1, chosen, PATH_COLOR)
        current_r = wall_row + 1
        current_c = int(chosen)
    _paint_segment(out, current_r, current_c, h - 1, current_c, PATH_COLOR)
    return out


def _straight_first_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    walls = _find_walls(g, bg)
    start_col = int(np.flatnonzero(g[0] == PATH_COLOR)[0])
    h = g.shape[0]

    out = g.copy()
    current_r = 0
    current_c = start_col
    for wall in walls:
        wall_row = int(wall["row"])
        band_top = current_r + 1
        if current_c in wall["gaps"]:
            chosen = int(current_c)
        else:
            chosen = min(wall["gaps"], key=lambda c: (abs(int(c) - current_c), int(c)))
        _paint_segment(out, current_r, current_c, band_top, current_c, PATH_COLOR)
        _paint_segment(out, band_top, current_c, band_top, chosen, PATH_COLOR)
        _paint_segment(out, band_top, chosen, wall_row + 1, chosen, PATH_COLOR)
        current_r = wall_row + 1
        current_c = int(chosen)
    _paint_segment(out, current_r, current_c, h - 1, current_c, PATH_COLOR)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a green route that uses same-colored guide pixels to choose gaps in successive full-row walls.

    Input structure:
    - Several distinct-colored full-width walls split the grid into horizontal bands; each wall row has multiple background gaps.
    - A single green start pixel sits on the top edge.
    - In every band above a wall, exactly one singleton has the same color as that wall, and its column matches one of that wall's gaps; other singletons are distractors.

    Rule:
    - Start from the top green pixel and descend into the first band.
    - In each band, continue down to the unique guide whose color matches the next wall, shift horizontally through that guide to its column, then descend through the aligned gap into the next band.
    - After the last wall, continue straight down to the bottom edge.
    - Paint the entire route green, overwriting the matched guides and leaving all other pixels unchanged.

    Key insight:
    - Geometry alone is insufficient: each wall's correct gap is keyed by the band-local pixel with the same color as that wall, creating a layer-by-layer color relay.
    """
    path, metadata = _trace_path(grid)
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    guide_cells = {(band["guide_row"], band["guide_col"]) for band in metadata}

    overwritten = {
        (int(r), int(c))
        for r, c in path
        if int(g[int(r), int(c)]) not in {int(bg), int(PATH_COLOR)}
    }
    assert overwritten <= guide_cells, "only matched guides may be covered by the route"
    return _output_from_path(g, path)


def _choose_gap_plan(width: int, entry_col: int, gap_count: int, force_straight_gap: bool) -> tuple[list[int], int, int]:
    cols = list(range(int(width)))
    nearest_distance = min(abs(c - entry_col) for c in cols if c != entry_col)
    nearest_options = [c for c in cols if c != entry_col and abs(c - entry_col) == nearest_distance]
    nearest_gap = random.choice(nearest_options)
    chosen_options = [c for c in cols if c != entry_col and abs(c - entry_col) > nearest_distance]
    chosen_gap = random.choice(chosen_options)

    gaps = {int(nearest_gap), int(chosen_gap)}
    if force_straight_gap:
        gaps.add(int(entry_col))
    while len(gaps) < int(gap_count):
        gaps.add(random.randrange(width))
    return sorted(gaps), int(chosen_gap), int(nearest_gap)


def _planned_path_coords(height: int, start_col: int, walls: list[dict]) -> set[tuple[int, int]]:
    current_r = 0
    current_c = int(start_col)
    coords = {(0, int(start_col))}
    for wall in walls:
        guide_r = int(wall["guide_row"])
        guide_c = int(wall["guide_col"])
        wall_row = int(wall["wall_row"])
        coords.update(_segment_coords(current_r, current_c, guide_r, current_c))
        coords.update(_segment_coords(guide_r, current_c, guide_r, guide_c))
        coords.update(_segment_coords(guide_r, guide_c, wall_row + 1, guide_c))
        current_r = wall_row + 1
        current_c = guide_c
    coords.update(_segment_coords(current_r, current_c, height - 1, current_c))
    return coords


def _sample_total_height(wall_count: int, band_heights: list[int], bottom_height: int) -> int:
    return 1 + sum(int(h) for h in band_heights) + int(wall_count) + int(bottom_height)


def _candidate_rows_without_path(path_cells: set[tuple[int, int]], rows: list[int], col: int) -> list[int]:
    return [int(r) for r in rows if (int(r), int(col)) not in path_cells]


def _place_band_distractors(
    grid: np.ndarray,
    bg: int,
    wall_plan: dict,
    wall_colors: list[int],
    path_cells: set[tuple[int, int]],
    used_cells: set[tuple[int, int]],
) -> None:
    rows = list(range(int(wall_plan["band_top"]), int(wall_plan["band_bottom"]) + 1))
    guide_row = int(wall_plan["guide_row"])
    guide_col = int(wall_plan["guide_col"])
    wall_color = int(wall_plan["wall_color"])
    aligned_decoy_col = int(wall_plan["nearest_gap"])
    free_colors = [c for c in wall_colors if c != wall_color]
    if bg in free_colors:
        free_colors.remove(bg)
    if PATH_COLOR in free_colors:
        free_colors.remove(PATH_COLOR)
    if not free_colors:
        free_colors = [c for c in NON_PATH_COLORS if c not in {bg, wall_color}]

    decoy_rows = _candidate_rows_without_path(path_cells, [r for r in rows if r != guide_row], aligned_decoy_col)
    assert decoy_rows, "need a non-path row for the nearest-gap distractor"
    decoy_r = random.choice(decoy_rows)
    decoy_color = random.choice(free_colors)
    grid[decoy_r, aligned_decoy_col] = int(decoy_color)
    used_cells.add((decoy_r, aligned_decoy_col))

    if wall_plan["entry_col"] in wall_plan["gaps"] and wall_plan["entry_col"] != guide_col:
        straight_rows = _candidate_rows_without_path(path_cells, rows, int(wall_plan["entry_col"]))
        straight_rows = [r for r in straight_rows if (r, int(wall_plan["entry_col"])) not in used_cells]
        if straight_rows:
            grid[random.choice(straight_rows), int(wall_plan["entry_col"])] = int(random.choice(free_colors))

    extra_count = random.randint(*DISTRACTOR_COUNT_RANGE) - 1
    available = [
        (r, c)
        for r in rows
        for c in range(grid.shape[1])
        if (r, c) not in path_cells
        and (r, c) not in used_cells
        and int(grid[r, c]) == int(bg)
    ]
    random.shuffle(available)
    for r, c in available[: max(0, extra_count)]:
        color_pool = [c2 for c2 in free_colors if c2 != wall_color]
        grid[r, c] = int(random.choice(color_pool))
        used_cells.add((r, c))


def _build_case() -> np.ndarray | None:
    bg = random.choice(NON_PATH_COLORS)
    wall_count = random.randint(*WALL_COUNT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    band_heights = [random.randint(*BAND_HEIGHT_RANGE) for _ in range(wall_count)]
    bottom_height = random.randint(*BOTTOM_BAND_HEIGHT_RANGE)
    height = _sample_total_height(wall_count, band_heights, bottom_height)
    if height > 30 or width > 30:
        return None

    wall_colors = random.sample([c for c in NON_PATH_COLORS if c != bg], wall_count)
    grid = np.full((height, width), int(bg), dtype=int)
    start_col = random.randrange(width)
    grid[0, start_col] = int(PATH_COLOR)

    force_straight_band = random.randrange(wall_count)
    plans: list[dict] = []
    cursor = 1
    entry_col = int(start_col)
    for i in range(wall_count):
        band_top = cursor
        band_bottom = cursor + band_heights[i] - 1
        wall_row = band_bottom + 1
        gap_count = random.randint(*GAP_COUNT_RANGE)
        gaps, chosen_gap, nearest_gap = _choose_gap_plan(width, entry_col, gap_count, i == force_straight_band)
        guide_row = random.randint(band_top, band_bottom)
        plan = {
            "band_top": int(band_top),
            "band_bottom": int(band_bottom),
            "wall_row": int(wall_row),
            "wall_color": int(wall_colors[i]),
            "gaps": gaps,
            "guide_row": int(guide_row),
            "guide_col": int(chosen_gap),
            "nearest_gap": int(nearest_gap),
            "entry_col": int(entry_col),
        }
        plans.append(plan)
        entry_col = int(chosen_gap)
        cursor = wall_row + 1

    path_cells = _planned_path_coords(height, start_col, plans)
    used_cells = {(0, start_col)}

    for plan in plans:
        wall_row = int(plan["wall_row"])
        wall_color = int(plan["wall_color"])
        grid[wall_row, :] = wall_color
        grid[wall_row, plan["gaps"]] = int(bg)
        grid[int(plan["guide_row"]), int(plan["guide_col"])] = wall_color
        used_cells.add((int(plan["guide_row"]), int(plan["guide_col"])))

    for plan in plans:
        _place_band_distractors(grid, bg, plan, wall_colors, path_cells, used_cells)

    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    try:
        path, metadata = _trace_path(g)
    except AssertionError:
        return False

    bg = get_bg(g)
    if bg == PATH_COLOR:
        return False

    guide_cells = {(band["guide_row"], band["guide_col"]) for band in metadata}
    overwritten = {
        (int(r), int(c))
        for r, c in path
        if int(g[int(r), int(c)]) not in {int(bg), int(PATH_COLOR)}
    }
    if overwritten != guide_cells:
        return False

    saw_straight_decoy = False
    for band in metadata:
        entry_col = int(band["entry_col"])
        guide_col = int(band["guide_col"])
        if guide_col == entry_col:
            return False
        if min(abs(c - entry_col) for c in band["gaps"] if c != guide_col) >= abs(guide_col - entry_col):
            return False

        band_slice = g[int(band["band_top"]) : int(band["wall_row"]), :]
        aligned = [
            (int(r), int(c), int(band_slice[r, c]))
            for r, c in np.argwhere(band_slice != int(bg))
            if int(c) in set(map(int, band["gaps"]))
        ]
        match_count = sum(color == int(band["wall_color"]) for _r, _c, color in aligned)
        if len(aligned) < 2 or match_count != 1:
            return False

        if entry_col in band["gaps"] and entry_col != guide_col:
            saw_straight_decoy = True

    true_out = main(g)
    if np.array_equal(true_out, _nearest_gap_output(g)):
        return False
    if saw_straight_decoy and np.array_equal(true_out, _straight_first_output(g)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        if grid is None:
            continue
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid color-chain portal input")
