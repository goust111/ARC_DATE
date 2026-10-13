from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_GRID_SIZE = 13
MAX_GRID_SIZE = 29
MIN_TOTAL_SEEDS = 3
MAX_TOTAL_SEEDS = 8
MAX_COLOR_COUNT = 4
MAX_GEN_ATTEMPTS = 400

QUADRANT_DIRS = {
    0: (-1, -1),  # top-left
    1: (-1, 1),   # top-right
    2: (1, -1),   # bottom-left
    3: (1, 1),    # bottom-right
}


def _find_cross(g: np.ndarray) -> tuple[int, int]:
    grey_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    grey_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert len(grey_rows) == 1, "expected exactly one full grey row"
    assert len(grey_cols) == 1, "expected exactly one full grey column"

    cross_r = int(grey_rows[0])
    cross_c = int(grey_cols[0])
    assert 0 < cross_r < g.shape[0] - 1
    assert 0 < cross_c < g.shape[1] - 1

    cross_mask = np.zeros_like(g, dtype=bool)
    cross_mask[cross_r, :] = True
    cross_mask[:, cross_c] = True
    assert np.array_equal(g == GREY, cross_mask), "grey must appear only on the separator cross"
    return cross_r, cross_c


def _quadrant_of(row: int, col: int, cross_r: int, cross_c: int) -> int:
    if row < cross_r and col < cross_c:
        return 0
    if row < cross_r and col > cross_c:
        return 1
    if row > cross_r and col < cross_c:
        return 2
    if row > cross_r and col > cross_c:
        return 3
    raise AssertionError("seed pixels must not lie on the grey cross")


def _rectangle_bounds(
    row: int,
    col: int,
    size: int,
    quadrant: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    dr, dc = QUADRANT_DIRS[quadrant]

    if dr < 0:
        r0, r1 = max(0, row - size + 1), row
    else:
        r0, r1 = row, min(height - 1, row + size - 1)

    if dc < 0:
        c0, c1 = max(0, col - size + 1), col
    else:
        c0, c1 = col, min(width - 1, col + size - 1)

    return int(r0), int(c0), int(r1), int(c1)


def _extract_seed_specs(g: np.ndarray, cross_r: int, cross_c: int) -> list[tuple[int, int, int, int, int]]:
    seed_mask = (g != BLACK) & (g != GREY)
    coords = np.argwhere(seed_mask)
    assert len(coords) > 0, "expected at least one colored seed"

    color_counts = Counter(map(int, g[seed_mask].reshape(-1)))
    specs: list[tuple[int, int, int, int, int]] = []
    for row, col in coords:
        row = int(row)
        col = int(col)
        color = int(g[row, col])
        quadrant = _quadrant_of(row, col, cross_r, cross_c)
        specs.append((row, col, color, quadrant, int(color_counts[color])))
    return specs


def _paint_from_seed_specs(
    specs: list[tuple[int, int, int, int, int]],
    shape: tuple[int, int],
    cross_r: int,
    cross_c: int,
) -> dict[tuple[int, int], int]:
    height, width = map(int, shape)
    painted: dict[tuple[int, int], int] = {}

    for row, col, color, quadrant, size in specs:
        r0, c0, r1, c1 = _rectangle_bounds(row, col, size, quadrant, height, width)
        assert not (r0 <= cross_r <= r1), "rectangles must grow away from the horizontal separator"
        assert not (c0 <= cross_c <= c1), "rectangles must grow away from the vertical separator"

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                prev = painted.get((rr, cc))
                assert prev in (None, color), "different colors must not claim the same output cell"
                painted[(rr, cc)] = color

    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a quadrant-directed square whose side length equals that color's global seed count.

    Input structure:
    - Black background with exactly one full grey horizontal line and one full grey vertical line.
    - The grey cross splits the grid into four quadrants.
    - Every non-black, non-grey cell is a single-pixel colored seed inside one quadrant.

    Rule:
    - Count the total number N of seeds of each color across the whole grid.
    - For each seed, paint a solid N x N rectangle of that seed's color, clipped by the grid boundary.
    - The seed is the corner nearest the grey cross, so top-left seeds grow up-left, top-right seeds
      grow up-right, bottom-left seeds grow down-left, and bottom-right seeds grow down-right.
    - Keep the grey cross unchanged.

    Key insight:
    - Rectangle size comes from a global per-color count, while rectangle direction comes from the
      seed's local quadrant, so solving requires combining counting with quadrant-specific geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_r, cross_c = _find_cross(g)
    specs = _extract_seed_specs(g, cross_r, cross_c)
    painted = _paint_from_seed_specs(specs, g.shape, cross_r, cross_c)

    out = g.copy()
    for (row, col), color in painted.items():
        out[row, col] = color
    out[cross_r, :] = GREY
    out[:, cross_c] = GREY
    return out


def _sample_grid_shape(max_count: int, total_seeds: int) -> tuple[int, int, int, int]:
    min_dim = max(MIN_GRID_SIZE, 2 * max_count + 7, total_seeds + 9)
    min_dim = min(min_dim, MAX_GRID_SIZE)

    for _ in range(100):
        height = random.randint(min_dim, MAX_GRID_SIZE)
        width = random.randint(min_dim, MAX_GRID_SIZE)
        cross_r = random.randint(4, height - 5)
        cross_c = random.randint(4, width - 5)
        if min(cross_r, height - cross_r - 1, cross_c, width - cross_c - 1) >= 4:
            return height, width, cross_r, cross_c
    raise RuntimeError("failed to sample a valid grid shape")


def _sample_color_counts() -> list[tuple[int, int]] | None:
    num_colors = random.choice([2, 2, 3, 3, 4])
    colors = random.sample(PALETTE, num_colors)

    for _ in range(80):
        counts = [random.randint(1, MAX_COLOR_COUNT) for _ in range(num_colors)]
        total = sum(counts)
        if not (MIN_TOTAL_SEEDS <= total <= MAX_TOTAL_SEEDS):
            continue
        if max(counts) == 1:
            continue
        if len(set(counts)) == 1 and random.random() < 0.65:
            continue
        return list(zip(colors, counts))
    return None


def _make_seed_jobs(color_counts: list[tuple[int, int]]) -> list[dict[str, int]] | None:
    jobs = [
        {"color": int(color), "size": int(size)}
        for color, size in color_counts
        for _ in range(size)
    ]
    total = len(jobs)
    used_quadrants = random.sample(range(4), random.randint(2, min(4, total)))
    quadrant_pool = used_quadrants + [random.choice(used_quadrants) for _ in range(total - len(used_quadrants))]

    split_candidates = [int(color) for color, count in color_counts if count >= 2]
    force_split_color = random.choice(split_candidates) if split_candidates else None

    for _ in range(80):
        random.shuffle(jobs)
        random.shuffle(quadrant_pool)
        if force_split_color is None:
            break
        split_quadrants = {
            int(quadrant)
            for job, quadrant in zip(jobs, quadrant_pool)
            if int(job["color"]) == force_split_color
        }
        if len(split_quadrants) >= 2:
            break
    else:
        return None

    for job, quadrant in zip(jobs, quadrant_pool):
        job["quadrant"] = int(quadrant)
    return jobs


def _quadrant_ranges(
    quadrant: int,
    height: int,
    width: int,
    cross_r: int,
    cross_c: int,
) -> tuple[range, range]:
    if quadrant == 0:
        return range(0, cross_r), range(0, cross_c)
    if quadrant == 1:
        return range(0, cross_r), range(cross_c + 1, width)
    if quadrant == 2:
        return range(cross_r + 1, height), range(0, cross_c)
    if quadrant == 3:
        return range(cross_r + 1, height), range(cross_c + 1, width)
    raise AssertionError("unknown quadrant")


def _is_clipped(bounds: tuple[int, int, int, int], size: int) -> bool:
    r0, c0, r1, c1 = bounds
    return (r1 - r0 + 1) != size or (c1 - c0 + 1) != size


def _buffered_overlap(mask: np.ndarray, bounds: tuple[int, int, int, int], margin: int = 1) -> bool:
    height, width = mask.shape
    r0, c0, r1, c1 = bounds
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(height - 1, r1 + margin)
    cc1 = min(width - 1, c1 + margin)
    return bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_jobs(
    jobs: list[dict[str, int]],
    height: int,
    width: int,
    cross_r: int,
    cross_c: int,
    need_clipped: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((height, width), BLACK, dtype=int)
    inp[cross_r, :] = GREY
    inp[:, cross_c] = GREY

    expected = inp.copy()
    seed_mask = np.zeros((height, width), dtype=bool)
    footprint_mask = np.zeros((height, width), dtype=bool)
    placed_specs: list[tuple[int, int, int, int, int]] = []
    clipped_count = 0

    random.shuffle(jobs)
    jobs = sorted(jobs, key=lambda job: -int(job["size"]))

    for job in jobs:
        color = int(job["color"])
        size = int(job["size"])
        quadrant = int(job["quadrant"])
        rows, cols = _quadrant_ranges(quadrant, height, width, cross_r, cross_c)

        unclipped: list[tuple[int, int, tuple[int, int, int, int]]] = []
        clipped: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for row in rows:
            for col in cols:
                bounds = _rectangle_bounds(row, col, size, quadrant, height, width)
                item = (int(row), int(col), bounds)
                if _is_clipped(bounds, size):
                    clipped.append(item)
                else:
                    unclipped.append(item)

        random.shuffle(unclipped)
        random.shuffle(clipped)
        prefer_clipped = random.random() < 0.4
        candidates = (clipped + unclipped) if prefer_clipped else (unclipped + clipped)

        placed = False
        for row, col, bounds in candidates:
            if _buffered_overlap(seed_mask, (row, col, row, col)):
                continue
            if _buffered_overlap(footprint_mask, bounds):
                continue

            r0, c0, r1, c1 = bounds
            inp[row, col] = color
            expected[r0 : r1 + 1, c0 : c1 + 1] = color
            seed_mask[row, col] = True
            footprint_mask[r0 : r1 + 1, c0 : c1 + 1] = True
            placed_specs.append((row, col, color, quadrant, size))
            clipped_count += int(_is_clipped(bounds, size))
            placed = True
            break

        if not placed:
            return None

    if need_clipped and clipped_count == 0:
        return None

    solver_expected = expected.copy()
    try:
        solver_actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(solver_actual, solver_expected):
        return None

    specs_from_input = _extract_seed_specs(inp, cross_r, cross_c)
    if sorted(placed_specs) != sorted(specs_from_input):
        return None

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    color_counts = _sample_color_counts()
    if color_counts is None:
        return None

    total_seeds = sum(count for _color, count in color_counts)
    max_count = max(count for _color, count in color_counts)
    height, width, cross_r, cross_c = _sample_grid_shape(max_count, total_seeds)

    jobs = _make_seed_jobs(color_counts)
    if jobs is None:
        return None

    need_clipped = random.random() < 0.5
    placed = _place_jobs(jobs, height, width, cross_r, cross_c, need_clipped)
    if placed is None:
        return None

    inp, _expected = placed
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
