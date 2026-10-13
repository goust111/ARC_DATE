from __future__ import annotations

import random

import numpy as np


N_COLORS = 10
GENERATION_ATTEMPTS = 500
GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
DIVIDER_COUNT_RANGE = (2, 4)
PAIR_COUNT_RANGE = (2, 5)


def _background(g: np.ndarray) -> int:
    values, counts = np.unique(g.astype(int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _divider_specs(g: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int], list[int]]:
    colors = [int(v) for v in np.unique(g) if int(v) != int(bg)]
    counts = {color: int(np.sum(g == color)) for color in colors}
    pair_colors = [color for color in colors if counts[color] == 2]
    divider_colors = [color for color in colors if counts[color] > 2]
    assert divider_colors, "expected at least one full-row or full-column divider"
    assert len(pair_colors) + len(divider_colors) == len(colors), "unexpected foreground color count"

    divider_set = set(divider_colors)
    horizontal: dict[int, int] = {}
    vertical: dict[int, int] = {}

    for color in divider_colors:
        row_counts = np.sum(g == color, axis=1)
        col_counts = np.sum(g == color, axis=0)
        best_row = int(np.argmax(row_counts))
        best_col = int(np.argmax(col_counts))
        row_score = int(row_counts[best_row])
        col_score = int(col_counts[best_col])
        assert row_score != col_score, "divider orientation must be visible from its long axis"

        if row_score > col_score:
            assert best_row not in horizontal, "two dividers cannot share a row"
            assert all(int(v) in divider_set for v in g[best_row, :]), "divider row must span the grid"
            horizontal[best_row] = int(color)
        else:
            assert best_col not in vertical, "two dividers cannot share a column"
            assert all(int(v) in divider_set for v in g[:, best_col]), "divider column must span the grid"
            vertical[best_col] = int(color)

    return horizontal, vertical, pair_colors


def _paint_segmented_line(
    out: np.ndarray,
    source: np.ndarray,
    fixed: int,
    lo: int,
    hi: int,
    dividers: dict[int, int],
    horizontal: bool,
    bg: int,
) -> None:
    crossed = [(pos, color) for pos, color in dividers.items() if lo < pos < hi]
    crossed.sort()
    assert crossed, "each pair must have at least one divider between its endpoints"

    crossed_pos = {pos: color for pos, color in crossed}
    first_pos, first_color = crossed[0]
    prev_color = first_color
    k = 0

    for p in range(lo + 1, hi):
        r, c = (fixed, p) if horizontal else (p, fixed)
        if p in crossed_pos:
            assert int(source[r, c]) == int(crossed_pos[p]), "divider cell must keep its divider color"
            k = next(i for i, (pos, _color) in enumerate(crossed) if pos == p)
            prev_color = crossed[k][1]
            continue

        assert int(source[r, c]) == int(bg), "line interiors may pass only through background or dividers"
        assert int(out[r, c]) == int(bg), "connector paths must not collide"

        if p < first_pos:
            fill = first_color
        else:
            while k + 1 < len(crossed) and crossed[k + 1][0] < p:
                k += 1
                prev_color = crossed[k][1]
            fill = prev_color
        out[r, c] = int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored pixel pairs with divider-colored segmented straight lines.

    Input structure:
    - The background is the dominant color.
    - One-cell-wide divider bands span complete rows or complete columns, and each divider band has its own color.
    - Every other non-background color appears exactly twice as a pair of pixels in one shared row or one shared column.
    - Each pair has at least one divider band between its two endpoints along that row or column.

    Rule:
    - For every same-colored pixel pair, draw the straight horizontal or vertical connection between the two endpoints.
    - Pixels that lie on divider bands keep the divider color already present in the input.
    - Background pixels before the first crossed divider use that first divider color; pixels between two crossed dividers use the divider color on the start side; pixels after the last crossed divider use that last divider color.
    - The two paired endpoint pixels keep their original pair color.

    Key insight:
    - The pair color only identifies which endpoints connect; the visible divider bands split the connector into directional color segments from the lower row or column endpoint toward the other endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)
    horizontal_dividers, vertical_dividers, pair_colors = _divider_specs(g, bg)
    assert pair_colors, "expected at least one colored pixel pair"

    out = g.copy()
    for color in pair_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each non-divider color must occur exactly twice"
        (r0, c0), (r1, c1) = [(int(r), int(c)) for r, c in coords]

        if r0 == r1:
            lo, hi = sorted((c0, c1))
            assert r0 not in horizontal_dividers, "pair cannot lie on a divider row"
            _paint_segmented_line(out, g, r0, lo, hi, vertical_dividers, True, bg)
        else:
            assert c0 == c1, "paired pixels must align in a row or column"
            lo, hi = sorted((r0, r1))
            assert c0 not in vertical_dividers, "pair cannot lie on a divider column"
            _paint_segmented_line(out, g, c0, lo, hi, horizontal_dividers, False, bg)

    return out


def _sample_dividers(length: int, count: int, rng: random.Random) -> list[int] | None:
    candidates = list(range(2, length - 2))
    for _ in range(100):
        coords = sorted(rng.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(coords, coords[1:])):
            return coords
    return None


def _compartments(dividers: list[int], length: int) -> list[list[int]]:
    parts: list[list[int]] = [list(range(0, dividers[0]))]
    for a, b in zip(dividers, dividers[1:]):
        parts.append(list(range(a + 1, b)))
    parts.append(list(range(dividers[-1] + 1, length)))
    return parts


def _endpoint_axis_positions(
    dividers: list[int],
    length: int,
    rng: random.Random,
    full_span: bool,
) -> tuple[int, int] | None:
    parts = _compartments(dividers, length)
    if any(len(part) == 0 for part in parts):
        return None

    if full_span:
        left_i, right_i = 0, len(parts) - 1
    elif rng.random() < 0.5:
        left_i = rng.randrange(len(dividers))
        right_i = left_i + 1
    else:
        left_i = rng.randrange(len(parts) - 1)
        right_i = rng.randrange(left_i + 1, len(parts))

    return int(rng.choice(parts[left_i])), int(rng.choice(parts[right_i]))


def _render_parallel_case(
    h: int,
    w: int,
    bg: int,
    divider_colors: list[int],
    pair_colors: list[int],
    orientation: str,
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.full((h, w), int(bg), dtype=int)
    divider_count = len(divider_colors)

    if orientation == "vertical":
        dividers = _sample_dividers(w, divider_count, rng)
        if dividers is None or len(pair_colors) > h:
            return None
        for c, color in zip(dividers, divider_colors):
            grid[:, c] = int(color)

        rows = rng.sample(range(h), len(pair_colors))
        for i, (row, color) in enumerate(zip(rows, pair_colors)):
            endpoints = _endpoint_axis_positions(dividers, w, rng, full_span=(i == 0))
            if endpoints is None:
                return None
            c0, c1 = endpoints
            grid[row, c0] = int(color)
            grid[row, c1] = int(color)
        return grid

    assert orientation == "horizontal"
    dividers = _sample_dividers(h, divider_count, rng)
    if dividers is None or len(pair_colors) > w:
        return None
    for r, color in zip(dividers, divider_colors):
        grid[r, :] = int(color)

    cols = rng.sample(range(w), len(pair_colors))
    for i, (col, color) in enumerate(zip(cols, pair_colors)):
        endpoints = _endpoint_axis_positions(dividers, h, rng, full_span=(i == 0))
        if endpoints is None:
            return None
        r0, r1 = endpoints
        grid[r0, col] = int(color)
        grid[r1, col] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)

    divider_count = rng.randint(*DIVIDER_COUNT_RANGE)
    max_pairs = min(PAIR_COUNT_RANGE[1], N_COLORS - 1 - divider_count)
    if max_pairs < PAIR_COUNT_RANGE[0]:
        return None
    pair_count = rng.randint(PAIR_COUNT_RANGE[0], max_pairs)

    colors = [c for c in range(N_COLORS) if c != bg]
    rng.shuffle(colors)
    divider_colors = [int(c) for c in colors[:divider_count]]
    pair_colors = [int(c) for c in colors[divider_count : divider_count + pair_count]]
    orientation = rng.choice(["vertical", "horizontal"])

    grid = _render_parallel_case(h, w, bg, divider_colors, pair_colors, orientation, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
