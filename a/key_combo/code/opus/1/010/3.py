from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG_COLOR = 0
LATTICE_COLOR = 5
OVERLAP_COLOR = 6
PALETTE_COLOR_POOL = (1, 2, 3, 4, 7, 8, 9)
CELL_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400
SEED_SEARCH_ATTEMPTS = 300
_RNG = random.Random()


def _cell_bands(line_indices: list[int]) -> list[tuple[int, int]]:
    bands = []
    for start, end in zip(line_indices, line_indices[1:]):
        assert end - start >= 2, "adjacent lattice lines must enclose a cell"
        bands.append((start + 1, end - 1))
    assert bands, "expected at least one enclosed cell band"
    return bands


def _parse_layout(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    palette = [int(v) for v in g[0] if int(v) != BG_COLOR]
    assert 3 <= len(palette) <= 4, "palette row must contain 3 or 4 colors"

    lattice_top = next((r for r in range(1, g.shape[0]) if np.all(g[r] == LATTICE_COLOR)), None)
    assert lattice_top is not None, "expected a full lattice row below the palette"

    wall_rows = [r for r in range(lattice_top, g.shape[0]) if np.all(g[r] == LATTICE_COLOR)]
    wall_cols = [c for c in range(g.shape[1]) if np.all(g[lattice_top:, c] == LATTICE_COLOR)]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "lattice must define at least one cell"

    row_bands = _cell_bands(wall_rows)
    col_bands = _cell_bands(wall_cols)
    return palette, row_bands, col_bands


def _find_seeds(
    grid: np.ndarray,
    seed_color: int,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    seeds: list[tuple[int, int]] = []
    for gr, (r0, r1) in enumerate(row_bands):
        for gc, (c0, c1) in enumerate(col_bands):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            non_bg = np.argwhere(cell != BG_COLOR)
            if non_bg.size == 0:
                continue
            assert non_bg.shape[0] == 1, "each occupied input cell must contain exactly one seed pixel"
            rr, cc = map(int, non_bg[0])
            assert int(cell[rr, cc]) == seed_color, "all seeds must match palette[0]"
            seeds.append((gr, gc))
    assert 2 <= len(seeds) <= 3, "expected 2 or 3 seed cells"
    return seeds


def _coverage_info(
    rows: int,
    cols: int,
    seeds: list[tuple[int, int]],
    palette_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    cover_count = np.zeros((rows, cols), dtype=int)
    unique_dist = np.full((rows, cols), -1, dtype=int)

    for gr in range(rows):
        for gc in range(cols):
            active = []
            for sr, sc in seeds:
                dist = max(abs(gr - sr), abs(gc - sc))
                if dist < palette_len:
                    active.append(dist)
            cover_count[gr, gc] = len(active)
            if len(active) == 1:
                unique_dist[gr, gc] = active[0]

    return cover_count, unique_dist


def _render_solution(
    base_grid: np.ndarray,
    palette: list[int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    seeds: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(base_grid, dtype=int).copy()
    rows, cols = len(row_bands), len(col_bands)
    cover_count, unique_dist = _coverage_info(rows, cols, seeds, len(palette))

    for gr, (r0, r1) in enumerate(row_bands):
        for gc, (c0, c1) in enumerate(col_bands):
            if cover_count[gr, gc] == 0:
                fill_color = BG_COLOR
            elif cover_count[gr, gc] == 1:
                fill_color = palette[int(unique_dist[gr, gc])]
            else:
                fill_color = OVERLAP_COLOR
            out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed across whole lattice cells by Chebyshev distance, using the palette row and painting overlaps magenta.

    Input structure:
    - The top row is a palette row: 3 or 4 non-background pixels from left to right encode the colors for distances 0, 1, 2, ...
    - Below it, grey lattice lines partition the canvas into regular rectangular cells whose interiors are otherwise black.
    - Two or three same-colored seed pixels sit inside distinct lattice cells, one seed per occupied cell.

    Rule:
    - Treat each lattice cell as one atomic position in a coarse grid.
    - For every seed, cells at lattice-cell Chebyshev distance d from that seed are covered when d is within the palette length.
    - If exactly one seed covers a cell, fill the entire cell interior with palette[d]; if two or more seeds cover it, fill the whole cell interior with magenta.
    - Cells outside every seed's palette radius stay black, while the palette row and grey lattice lines remain unchanged.

    Key insight:
    - Distance is measured between lattice cells rather than pixels, so the whole interior of each cell changes color together.
    """
    g = np.asarray(grid, dtype=int)
    palette, row_bands, col_bands = _parse_layout(g)
    seeds = _find_seeds(g, palette[0], row_bands, col_bands)
    return _render_solution(g, palette, row_bands, col_bands, seeds)


def _latent_bands(rows: int, cols: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_lines = [1 + i * (CELL_SIZE + 1) for i in range(rows + 1)]
    col_lines = [i * (CELL_SIZE + 1) for i in range(cols + 1)]
    return _cell_bands(row_lines), _cell_bands(col_lines)


def _render_input_case(case: dict) -> np.ndarray:
    rows = int(case["rows"])
    cols = int(case["cols"])
    palette = list(case["palette"])
    seeds = list(case["seeds"])
    top_start = int(case["palette_start"])
    top_step = int(case["palette_step"])

    height = 1 + rows * CELL_SIZE + (rows + 1)
    width = cols * CELL_SIZE + (cols + 1)
    grid = np.full((height, width), BG_COLOR, dtype=int)

    row_lines = [1 + i * (CELL_SIZE + 1) for i in range(rows + 1)]
    col_lines = [i * (CELL_SIZE + 1) for i in range(cols + 1)]
    grid[row_lines, :] = LATTICE_COLOR
    grid[1:, col_lines] = LATTICE_COLOR

    for i, color in enumerate(palette):
        grid[0, top_start + i * top_step] = int(color)

    row_bands, col_bands = _latent_bands(rows, cols)
    for gr, gc in seeds:
        r0, r1 = row_bands[gr]
        c0, c1 = col_bands[gc]
        grid[(r0 + r1) // 2, (c0 + c1) // 2] = palette[0]

    return grid


def _expected_from_case(case: dict, input_grid: np.ndarray) -> np.ndarray:
    row_bands, col_bands = _latent_bands(int(case["rows"]), int(case["cols"]))
    return _render_solution(input_grid, list(case["palette"]), row_bands, col_bands, list(case["seeds"]))


def _arrangement_ok(seeds: list[tuple[int, int]], want_aligned: bool) -> bool:
    aligned_pairs = sum(int(r1 == r2 or c1 == c2) for (r1, c1), (r2, c2) in combinations(seeds, 2))
    return aligned_pairs >= 1 if want_aligned else aligned_pairs < len(seeds) - 1


def _quality_ok(rows: int, cols: int, seeds: list[tuple[int, int]], palette_len: int) -> bool:
    cover_count, unique_dist = _coverage_info(rows, cols, seeds, palette_len)
    total_cells = rows * cols
    overlap_cells = int(np.count_nonzero(cover_count >= 2))
    unique_cells = int(np.count_nonzero(cover_count == 1))
    untouched_cells = int(np.count_nonzero(cover_count == 0))

    if overlap_cells == 0 or unique_cells == 0 or untouched_cells == 0:
        return False
    if overlap_cells * 2 >= total_cells:
        return False
    if unique_cells <= overlap_cells:
        return False

    visible_unique_distances = {int(v) for v in unique_dist.ravel() if int(v) >= 0}
    if len(visible_unique_distances) < min(3, palette_len):
        return False
    if palette_len >= 2 and int(np.count_nonzero(unique_dist == 1)) == 0:
        return False
    if palette_len >= 3 and int(np.count_nonzero(unique_dist == palette_len - 1)) == 0:
        return False

    return True


def _generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = _RNG if rng is None else rng

    for _ in range(MAX_GENERATION_ATTEMPTS):
        palette_len = rng.choice((3, 4))
        min_extent = 4 if palette_len == 3 else 6
        rows = rng.randint(min_extent, 7)
        cols = rng.randint(min_extent, 7)
        seed_count = 2 if rng.random() < 0.55 else 3
        palette = rng.sample(list(PALETTE_COLOR_POOL), palette_len)
        want_aligned = rng.random() < 0.35

        cells = [(r, c) for r in range(rows) for c in range(cols)]
        seeds = None
        for _ in range(SEED_SEARCH_ATTEMPTS):
            trial = sorted(rng.sample(cells, seed_count))
            if not _arrangement_ok(trial, want_aligned):
                continue
            if not _quality_ok(rows, cols, trial, palette_len):
                continue
            seeds = trial
            break

        if seeds is None:
            continue

        top_step = rng.choice((2, 3))
        top_start = rng.randint(1, 3)
        case = {
            "rows": rows,
            "cols": cols,
            "palette": palette,
            "seeds": seeds,
            "palette_start": top_start,
            "palette_step": top_step,
        }
        input_grid = _render_input_case(case)
        expected = _expected_from_case(case, input_grid)
        return input_grid, expected, case

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
