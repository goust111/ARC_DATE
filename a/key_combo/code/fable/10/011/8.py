from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _contiguous_runs(cols: list[int]) -> list[tuple[int, int]]:
    if not cols:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))
    return runs


def _detect_wall_and_sources(top_row: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    counts = Counter(map(int, top_row))
    wall_color, wall_count = counts.most_common(1)[0]
    assert wall_count == len(top_row) - 2, "top wall row must have exactly two source gaps"

    sources = [(int(c), int(top_row[c])) for c in np.flatnonzero(top_row != wall_color)]
    assert len(sources) == 2, "expected exactly two top-row sources"
    assert sources[0][1] != sources[1][1], "the two source colors must differ"
    return int(wall_color), sources


def _detect_legend(grid: np.ndarray, source_colors: set[int]) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    assert w >= 3, "grid too narrow for a 1x3 legend"

    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for c0 in (0, w - 3):
        vals = tuple(int(x) for x in grid[h - 1, c0 : c0 + 3])
        if (
            vals[0] != vals[2]
            and {vals[0], vals[2]} == source_colors
            and vals[1] not in source_colors
        ):
            cells = [(h - 1, c0 + dc) for dc in range(3)]
            candidates.append((vals[1], cells))

    assert len(candidates) == 1, "expected one bottom-corner 1x3 legend"
    return candidates[0]


def _identify_roles(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], int, int, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    wall_color, sources = _detect_wall_and_sources(g[0])
    source_colors = {color for _col, color in sources}
    overlap_color, legend_cells = _detect_legend(g, source_colors)

    remaining = set(map(int, np.unique(g))) - {bg, wall_color, overlap_color, *source_colors}
    assert len(remaining) == 1, "expected exactly one obstacle color"
    obstacle_color = remaining.pop()
    return bg, wall_color, sources, overlap_color, obstacle_color, set(legend_cells)


def _propagate_light(grid: np.ndarray, start_col: int, obstacle_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    lit = np.zeros((h, w), dtype=bool)
    lit[0, int(start_col)] = True

    for r in range(h - 1):
        for c in np.flatnonzero(lit[r]):
            c = int(c)
            for nc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                if int(g[r + 1, nc]) != obstacle_color:
                    lit[r + 1, nc] = True
    return lit


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, _wall_color, sources, overlap_color, obstacle_color, _legend_cells = _identify_roles(g)

    (c1, color1), (c2, color2) = sources
    light1 = _propagate_light(g, c1, obstacle_color)
    light2 = _propagate_light(g, c2, obstacle_color)

    out = g.copy()
    bg_mask = g == bg
    out[bg_mask & light1 & ~light2] = int(color1)
    out[bg_mask & light2 & ~light1] = int(color2)
    out[bg_mask & light1 & light2] = int(overlap_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate two cone-shaped light beams downward from the wall gaps and recolor their overlap by the corner legend.

    Input structure:
    - The top row is a solid wall strip with exactly two differently colored source pixels replacing two gaps.
    - A single obstacle color marks dark blocking pixels scattered in the interior.
    - A 1x3 legend sits in one bottom corner; its two end cells match the two source colors and its center gives the overlap color.

    Rule:
    - Track each source separately. If a cell is lit on row r, then on row r+1 the light reaches the three candidate cells directly below, down-left, and down-right, provided the target cell is not an obstacle.
    - Background cells lit by only one source are recolored to that source's color.
    - Background cells lit by both sources are recolored to the legend's middle color.
    - The wall row, the legend, all obstacles, and every unlit cell stay unchanged.

    Key insight:
    - The spread is row-by-row cone propagation rather than flood fill: obstacle cells do not just stay dark themselves, they also cast shadows because blocked cells cannot continue the next-row expansion.
    """
    return _transform(grid)


GEN_ATTEMPTS = 500
GEN_H_RANGE = (12, 20)
GEN_W_RANGE = (12, 22)
GEN_MIN_SOURCE_GAP = 4
GUIDED_DIMS = ((1, 1), (1, 2), (2, 1), (1, 3), (2, 2))
FILLER_DIMS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2))


def _paint_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = True


def _can_place(mask: np.ndarray, reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    if reserved[r0 : r0 + h, c0 : c0 + w].any():
        return False
    return not mask[r0 : r0 + h, c0 : c0 + w].any()


def _place_guided_block(
    obstacle_mask: np.ndarray,
    reserved: np.ndarray,
    candidate_mask: np.ndarray,
    dims: tuple[tuple[int, int], ...],
) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(candidate_mask)]
    random.shuffle(cells)
    cells.sort(key=lambda rc: (rc[0], random.random()))

    for r, c in cells:
        options: list[tuple[int, int, int, int]] = []
        for h, w in dims:
            for dr in range(h):
                for dc in range(w):
                    r0 = r - dr
                    c0 = c - dc
                    if _can_place(obstacle_mask, reserved, r0, c0, h, w):
                        options.append((r0, c0, h, w))
        if options:
            r0, c0, h, w = random.choice(options)
            _paint_rect(obstacle_mask, r0, c0, h, w)
            return True
    return False


def _random_fill_block(obstacle_mask: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = obstacle_mask.shape
    dims = list(FILLER_DIMS)
    random.shuffle(dims)
    for block_h, block_w in dims:
        cells = [
            (r0, c0)
            for r0 in range(2, h - 1)
            for c0 in range(0, w)
            if _can_place(obstacle_mask, reserved, r0, c0, block_h, block_w)
        ]
        if cells:
            r0, c0 = random.choice(cells)
            _paint_rect(obstacle_mask, r0, c0, block_h, block_w)
            return True
    return False


def _build_base_grid() -> tuple[np.ndarray, dict[str, object]]:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    bg, wall, src1, src2, mix, obstacle = random.sample(range(10), 6)

    c1 = random.randint(2, w - GEN_MIN_SOURCE_GAP - 3)
    c2 = random.randint(c1 + GEN_MIN_SOURCE_GAP, w - 3)

    grid = np.full((h, w), bg, dtype=int)
    grid[0, :] = wall
    grid[0, c1] = src1
    grid[0, c2] = src2

    legend_right = bool(random.getrandbits(1))
    legend_vals = [src1, mix, src2]
    if random.random() < 0.5:
        legend_vals = [src2, mix, src1]

    legend_c0 = w - 3 if legend_right else 0
    grid[h - 1, legend_c0 : legend_c0 + 3] = np.array(legend_vals, dtype=int)

    reserved = np.zeros((h, w), dtype=bool)
    reserved[0, :] = True
    reserved[h - 1, legend_c0 : legend_c0 + 3] = True

    return grid, {
        "bg": bg,
        "wall": wall,
        "src1": src1,
        "src2": src2,
        "mix": mix,
        "obstacle": obstacle,
        "c1": c1,
        "c2": c2,
        "reserved": reserved,
    }


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        bg, _wall, sources, overlap_color, obstacle_color, _legend_cells = _identify_roles(grid)
    except AssertionError:
        return False

    out = _transform(grid)
    h, w = grid.shape
    assert out.shape == (h, w)

    (c1, color1), (c2, color2) = sources
    light1 = _propagate_light(grid, c1, obstacle_color)
    light2 = _propagate_light(grid, c2, obstacle_color)

    unobstructed = grid.copy()
    unobstructed[unobstructed == obstacle_color] = bg
    full1 = _propagate_light(unobstructed, c1, obstacle_color)
    full2 = _propagate_light(unobstructed, c2, obstacle_color)

    bg_mask = grid == bg
    single1 = int(np.count_nonzero(bg_mask & light1 & ~light2))
    single2 = int(np.count_nonzero(bg_mask & light2 & ~light1))
    both = int(np.count_nonzero(bg_mask & light1 & light2))
    lit_total = int(np.count_nonzero(bg_mask & (light1 | light2)))

    if min(single1, single2) < 4 or both < 2 or lit_total < 18:
        return False

    if np.count_nonzero((full1 & ~light1) & bg_mask) < 4:
        return False
    if np.count_nonzero((full2 & ~light2) & bg_mask) < 4:
        return False

    if np.count_nonzero(grid == obstacle_color) < 3:
        return False

    gap_row_found = False
    for r in range(1, h):
        open_cols = np.flatnonzero(bg_mask[r] & (full1[r] | full2[r])).tolist()
        actual_cols = np.flatnonzero(bg_mask[r] & (light1[r] | light2[r])).tolist()
        if not open_cols or not actual_cols:
            continue
        if len(_contiguous_runs(open_cols)) == 1 and len(_contiguous_runs(actual_cols)) >= 2:
            gap_row_found = True
            break
    if not gap_row_found:
        return False

    if overlap_color not in out:
        return False
    if color1 not in out or color2 not in out:
        return False

    if np.array_equal(out, grid):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    grid, meta = _build_base_grid()
    h, w = grid.shape
    reserved = np.array(meta["reserved"], copy=True)
    obstacle_color = int(meta["obstacle"])

    empty1 = _propagate_light(grid, int(meta["c1"]), obstacle_color)
    empty2 = _propagate_light(grid, int(meta["c2"]), obstacle_color)

    bg_mask = grid == int(meta["bg"])
    obstacle_mask = np.zeros((h, w), dtype=bool)

    source1_only = bg_mask & empty1 & ~empty2
    source2_only = bg_mask & empty2 & ~empty1
    overlap = bg_mask & empty1 & empty2

    source1_only[:2, :] = False
    source2_only[:2, :] = False
    overlap[:3, :] = False

    if not _place_guided_block(obstacle_mask, reserved, source1_only, GUIDED_DIMS):
        return None
    if not _place_guided_block(obstacle_mask, reserved, source2_only, GUIDED_DIMS):
        return None

    if random.random() < 0.8:
        _place_guided_block(obstacle_mask, reserved, overlap, GUIDED_DIMS)

    for _ in range(random.randint(0, 2)):
        _random_fill_block(obstacle_mask, reserved)

    grid[obstacle_mask] = obstacle_color

    try:
        if not _validate_semantics(grid):
            return None
        out = main(grid)
        independent = _transform(grid)
        if not np.array_equal(out, independent):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-cone light input")
