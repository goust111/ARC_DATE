from __future__ import annotations

import numpy as np

from arc_utils import Color


def _reflection_orbit(row: int, col: int, height: int, width: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (row, col),
                (row, width - 1 - col),
                (height - 1 - row, col),
                (height - 1 - row, width - 1 - col),
            }
        )
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the grid by replacing every green placeholder with the color required by two-axis reflection symmetry.

    Input structure:
    - A rectangular pattern is symmetric under left-right and top-bottom reflection once the green placeholder pixels are ignored.
    - Every reflection orbit containing green has exactly one visible non-green color across its remaining cells.

    Rule:
    - For each cell orbit under horizontal and vertical mirror reflections, preserve the shared non-green color.
    - Recolor any green cell in that orbit to that shared color.

    Key insight:
    - Green is not part of the final design; it marks damaged cells whose values are recoverable from the same-position cells in the reflected copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == Color.GREEN), "expected green placeholder pixels"

    height, width = map(int, g.shape)
    out = g.copy()
    seen: set[tuple[tuple[int, int], ...]] = set()
    repaired = 0

    for row in range(height):
        for col in range(width):
            orbit = _reflection_orbit(row, col, height, width)
            if orbit in seen:
                continue
            seen.add(orbit)

            values = [int(g[r, c]) for r, c in orbit]
            visible_colors = {color for color in values if color != Color.GREEN}
            assert len(visible_colors) == 1, "each reflection orbit must have one visible color"
            fill_color = visible_colors.pop()

            for r, c in orbit:
                if int(g[r, c]) == Color.GREEN:
                    out[r, c] = fill_color
                    repaired += 1

    assert repaired == int(np.count_nonzero(g == Color.GREEN)), "all green pixels must be repairable"
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_SIZE = 16
HALF_GRID = GRID_SIZE // 2
OUTPUT_PALETTE = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
OUTPUT_COLOR_COUNT_RANGE = (5, 9)  # rng upper bound is exclusive.
COPY_PROBABILITY_RANGE = (0.15, 0.45)
FUNDAMENTAL_DOMAIN = tuple((row, col) for row in range(HALF_GRID) for col in range(row, HALF_GRID))
FUNDAMENTAL_NEIGHBOR_OFFSETS = ((-1, 0), (0, -1), (-1, -1), (1, -1))

# Green damage rectangles.
BLOB_SIDE_RANGE = range(2, 5)  # rng-style upper bound is exclusive.
BLOB_AREA_MAX = 12
TOTAL_GREEN_RANGE = (10, 20)
GREEN_ORBIT_CAP = 2

# Retry limits.
BLOB_PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 300


def _build_blob_size_pairs_by_total() -> dict[int, list[tuple[tuple[int, int], tuple[int, int]]]]:
    dims = [(h, w) for h in BLOB_SIDE_RANGE for w in BLOB_SIDE_RANGE if h * w <= BLOB_AREA_MAX]
    pairs_by_total: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for first in dims:
        for second in dims:
            total = first[0] * first[1] + second[0] * second[1]
            if TOTAL_GREEN_RANGE[0] <= total <= TOTAL_GREEN_RANGE[1]:
                pairs_by_total.setdefault(total, []).append((first, second))
    return pairs_by_total


BLOB_SIZE_PAIRS_BY_TOTAL = _build_blob_size_pairs_by_total()
BLOB_TOTALS = np.array(sorted(BLOB_SIZE_PAIRS_BY_TOTAL), dtype=int)


def _gen_d4_orbit(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (row, col),
                (col, row),
                (row, GRID_SIZE - 1 - col),
                (GRID_SIZE - 1 - row, col),
                (GRID_SIZE - 1 - row, GRID_SIZE - 1 - col),
                (col, GRID_SIZE - 1 - row),
                (GRID_SIZE - 1 - col, row),
                (GRID_SIZE - 1 - col, GRID_SIZE - 1 - row),
            }
        )
    )


def _gen_reflection_orbit(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (row, col),
                (row, GRID_SIZE - 1 - col),
                (GRID_SIZE - 1 - row, col),
                (GRID_SIZE - 1 - row, GRID_SIZE - 1 - col),
            }
        )
    )


def _gen_rect_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    row, col, height, width = rect
    return [
        (r, c)
        for r in range(row, row + height)
        for c in range(col, col + width)
    ]


def _gen_rects_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar <= br + bh and br <= ar + ah and ac <= bc + bw and bc <= ac + aw


def _gen_orbit_cap_ok(rects: tuple[tuple[int, int, int, int], ...]) -> bool:
    counts: dict[tuple[tuple[int, int], ...], int] = {}
    for rect in rects:
        for row, col in _gen_rect_cells(rect):
            orbit = _gen_reflection_orbit(row, col)
            counts[orbit] = counts.get(orbit, 0) + 1
            if counts[orbit] > GREEN_ORBIT_CAP:
                return False
    return True


def _gen_output(rng: np.random.Generator) -> np.ndarray:
    n_colors = int(rng.integers(*OUTPUT_COLOR_COUNT_RANGE))
    colors = rng.choice(OUTPUT_PALETTE, size=n_colors, replace=False)
    values: dict[tuple[int, int], int] = {}
    copy_probability = float(rng.uniform(*COPY_PROBABILITY_RANGE))

    for row, col in FUNDAMENTAL_DOMAIN:
        neighbors = [
            values[pos]
            for dr, dc in FUNDAMENTAL_NEIGHBOR_OFFSETS
            if (pos := (row + dr, col + dc)) in values
        ]
        if neighbors and rng.random() < copy_probability:
            color = int(rng.choice(neighbors))
        else:
            color = int(rng.choice(colors))
        values[(row, col)] = color

    forced_indices = rng.choice(len(FUNDAMENTAL_DOMAIN), size=n_colors, replace=False)
    forced_colors = list(map(int, colors))
    rng.shuffle(forced_colors)
    for index, color in zip(forced_indices, forced_colors):
        values[FUNDAMENTAL_DOMAIN[int(index)]] = color

    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for (row, col), color in values.items():
        for r, c in _gen_d4_orbit(row, col):
            out[r, c] = color
    return out


def _gen_blob_rects(rng: np.random.Generator) -> tuple[tuple[int, int, int, int], ...] | None:
    """Sample two separated green rectangles satisfying the LR/UD orbit cap."""
    for _ in range(BLOB_PLACEMENT_ATTEMPTS):
        total = int(rng.choice(BLOB_TOTALS))
        pairs = BLOB_SIZE_PAIRS_BY_TOTAL[total]
        (h1, w1), (h2, w2) = pairs[int(rng.integers(len(pairs)))]
        rect1 = (
            int(rng.integers(0, GRID_SIZE + 1 - h1)),
            int(rng.integers(0, GRID_SIZE + 1 - w1)),
            h1,
            w1,
        )
        rect2 = (
            int(rng.integers(0, GRID_SIZE + 1 - h2)),
            int(rng.integers(0, GRID_SIZE + 1 - w2)),
            h2,
            w2,
        )
        rects = (rect1, rect2)
        if not _gen_rects_touch(rect1, rect2) and _gen_orbit_cap_ok(rects):
            return rects
    return None


def _apply_green_damage(
    out: np.ndarray, rects: tuple[tuple[int, int, int, int], ...]
) -> np.ndarray:
    inp = out.copy()
    for row, col, height, width in rects:
        inp[row : row + height, col : col + width] = Color.GREEN
    return inp


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    out = _gen_output(rng)
    rects = _gen_blob_rects(rng)
    if rects is None:
        return None

    inp = _apply_green_damage(out, rects)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
