import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
SEP = 5
HOLE = 7
BAR = 9
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, grid.flat))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(sep_rows) == 1 and len(sep_cols) == 1
    split_r = int(sep_rows[0])
    split_c = int(sep_cols[0])
    assert split_r == grid.shape[0] - split_r - 1
    assert split_c == grid.shape[1] - split_c - 1
    return split_r, split_c


def _region_bounds(grid: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    split_r, split_c = _find_cross(grid)
    return [
        ("tl", (0, split_r, 0, split_c)),
        ("tr", (0, split_r, split_c + 1, grid.shape[1])),
        ("bl", (split_r + 1, grid.shape[0], 0, split_c)),
        ("br", (split_r + 1, grid.shape[0], split_c + 1, grid.shape[1])),
    ]


def _bar_columns(c0: int, c1: int) -> tuple[int, int]:
    width = int(c1 - c0)
    assert width >= 2 and width % 2 == 0
    left = int(c0 + (width - 2) // 2)
    return left, left + 1


def _future_bar_cells(region_h: int, region_w: int, height: int) -> set[tuple[int, int]]:
    left = (region_w - 2) // 2
    return {
        (r, c)
        for r in range(region_h - height, region_h)
        for c in range(left, left + 2)
    }


def _infer_region(region: np.ndarray) -> tuple[int, tuple[int, int]]:
    palette_pixels = [int(v) for v in region.flat if int(v) in PALETTE]
    present = set(palette_pixels)
    assert len(palette_pixels) == 3
    assert len(present) == 3
    missing = list(set(PALETTE) - present)
    assert len(missing) == 1

    hole_coords = np.argwhere(region == HOLE)
    assert hole_coords.shape == (1, 2)
    hole_r, hole_c = map(int, hole_coords[0])
    return int(missing[0]), (hole_r, hole_c)


def _solve_regions(grid: np.ndarray) -> list[dict]:
    records = []
    for name, (r0, r1, c0, c1) in _region_bounds(grid):
        region = grid[r0:r1, c0:c1]
        missing, hole_local = _infer_region(region)
        records.append(
            {
                "name": name,
                "bounds": (r0, r1, c0, c1),
                "missing": missing,
                "hole_local": hole_local,
            }
        )
    return records


def _render_output(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rec in records:
        r0, r1, c0, c1 = rec["bounds"]
        missing = int(rec["missing"])
        hole_r, hole_c = rec["hole_local"]
        out[r0 + hole_r, c0 + hole_c] = missing

    for rec in records:
        r0, r1, c0, c1 = rec["bounds"]
        height = int(rec["missing"])
        left, right = _bar_columns(c0, c1)
        out[r1 - height : r1, left : right + 1] = BAR

    split_r, split_c = _find_cross(grid)
    out[split_r, :] = SEP
    out[:, split_c] = SEP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing color in each separator-defined quadrant, fill its hole, and convert that color into a bottom terrain pillar.

    Input structure:
    - The background is black, with one full grey row and one full grey column forming a centered cross that splits the grid into four equal rectangles.
    - Each rectangle contains exactly three distinct colors from {blue(1), red(2), green(3), yellow(4)} plus one orange hole marker.
    - Across the four rectangles, the missing colors are exactly the four palette colors in some order.

    Rule:
    - In each quadrant, compare the visible palette colors against {1,2,3,4} to infer the unique missing color.
    - Replace the orange hole marker with that inferred color.
    - Then draw a solid maroon pillar of width 2 at the horizontal center of the same quadrant, bottom-aligned within that quadrant, with height equal to the inferred color value.
    - Pillars overwrite any pixels they cover, while the grey separator cross stays unchanged.

    Key insight:
    - The orange hole is only the first step: the inferred missing palette member also determines a geometric height, so color completion and terrain generation must both be applied quadrant by quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG
    assert set(colors_present(g, exclude={BG})) <= {1, 2, 3, 4, SEP, HOLE}
    records = _solve_regions(g)
    assert sorted(int(rec["missing"]) for rec in records) == [1, 2, 3, 4]
    return _render_output(g, records)


def _sample_positions(
    rng: random.Random,
    pool: list[tuple[int, int]],
    count: int,
    *,
    min_manhattan: int,
) -> list[tuple[int, int]] | None:
    cells = pool[:]
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= min_manhattan for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _visible_hole_pool(region_h: int, region_w: int) -> list[tuple[int, int]]:
    bar_shadow = _future_bar_cells(region_h, region_w, 4)
    return [
        (r, c)
        for r in range(1, region_h - 1)
        for c in range(1, region_w - 1)
        if (r, c) not in bar_shadow
    ]


def _ordinary_pool(region_h: int, region_w: int, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(1, region_h - 1)
        for c in range(1, region_w - 1)
        if (r, c) not in forbidden
    ]


def _build_region_cells(
    rng: random.Random,
    region_h: int,
    region_w: int,
    missing: int,
    hole_local: tuple[int, int],
) -> dict[tuple[int, int], int] | None:
    present_colors = [color for color in PALETTE if color != missing]
    cells: dict[tuple[int, int], int] = {hole_local: HOLE}
    occupied = {hole_local}
    bar_shadow = _future_bar_cells(region_h, region_w, missing)
    bar_cells = list(bar_shadow - occupied)

    if bar_cells and rng.random() < 0.65:
        covered_cell = rng.choice(bar_cells)
        covered_color = rng.choice(present_colors)
        cells[covered_cell] = covered_color
        occupied.add(covered_cell)
        present_colors.remove(covered_color)

    pool = _ordinary_pool(region_h, region_w, occupied | bar_shadow)
    chosen = _sample_positions(rng, pool, len(present_colors), min_manhattan=2)
    if chosen is None:
        return None
    for pos, color in zip(chosen, present_colors, strict=False):
        cells[pos] = color
    return cells


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_h = rng.randint(6, 10)
    region_w = rng.choice([6, 8, 10, 12])
    grid = np.full((2 * region_h + 1, 2 * region_w + 1), BG, dtype=int)
    grid[region_h, :] = SEP
    grid[:, region_w] = SEP

    hole_candidates = _visible_hole_pool(region_h, region_w)
    hole_positions = _sample_positions(rng, hole_candidates, 4, min_manhattan=3)
    if hole_positions is None:
        return None

    missing_by_region = list(PALETTE)
    rng.shuffle(missing_by_region)
    local_records: list[tuple[tuple[int, int], int, dict[tuple[int, int], int] | None]] = []
    for hole_local, missing in zip(hole_positions, missing_by_region, strict=False):
        cells = _build_region_cells(rng, region_h, region_w, missing, hole_local)
        local_records.append((hole_local, missing, cells))
    if any(cells is None for _, _, cells in local_records):
        return None

    for (name, (r0, r1, c0, c1)), (_, _, cells) in zip(_region_bounds(grid), local_records, strict=False):
        assert r1 - r0 == region_h and c1 - c0 == region_w
        for (rr, cc), color in cells.items():
            grid[r0 + rr, c0 + cc] = int(color)

    if not _validate_generated_case(grid):
        return None
    return grid


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        split_r, split_c = _find_cross(grid)
        assert 0 < split_r < grid.shape[0] - 1
        assert 0 < split_c < grid.shape[1] - 1

        records = _solve_regions(grid)
        assert sorted(int(rec["missing"]) for rec in records) == [1, 2, 3, 4]
        assert len({tuple(map(int, rec["hole_local"])) for rec in records}) == 4

        out = main(grid)
        assert out.shape == grid.shape
        assert np.any(out != grid)
        assert np.all(out[split_r, :] == SEP)
        assert np.all(out[:, split_c] == SEP)

        for rec in records:
            r0, r1, c0, c1 = rec["bounds"]
            missing = int(rec["missing"])
            hole_r, hole_c = rec["hole_local"]
            assert (hole_r, hole_c) not in _future_bar_cells(r1 - r0, c1 - c0, 4)
            assert int(out[r0 + hole_r, c0 + hole_c]) == missing
            left, right = _bar_columns(c0, c1)
            pillar = out[r1 - missing : r1, left : right + 1]
            assert pillar.shape == (missing, 2)
            assert np.all(pillar == BAR)
            if r1 - missing - 1 >= r0:
                assert np.all(out[r0 : r1 - missing, left : right + 1] != BAR)
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
