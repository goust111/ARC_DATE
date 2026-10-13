from __future__ import annotations

import random
import numpy as np


def _split_runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = idx
        prev = idx
    runs.append((start, prev))
    return runs


def _single_run_bands(lines: list[tuple[int, ...]]) -> list[tuple[int, int]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for idx, line in enumerate(lines):
        groups.setdefault(line, []).append(idx)

    bands: list[tuple[int, int]] = []
    for indices in groups.values():
        runs = _split_runs(indices)
        if len(runs) == 1:
            bands.append(runs[0])
    bands.sort()
    return bands


def _infer_stripes(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    row_lines = [tuple(map(int, g[r])) for r in range(height)]
    col_lines = [tuple(map(int, g[:, c])) for c in range(width)]
    horizontal_bands = _single_run_bands(row_lines)
    vertical_bands = _single_run_bands(col_lines)

    assert horizontal_bands, "expected at least one horizontal stripe"
    assert vertical_bands, "expected at least one vertical stripe"

    row_mask = np.zeros(height, dtype=bool)
    col_mask = np.zeros(width, dtype=bool)
    for r0, r1 in horizontal_bands:
        row_mask[r0 : r1 + 1] = True
    for c0, c1 in vertical_bands:
        col_mask[c0 : c1 + 1] = True

    assert bool((~col_mask).any()), "vertical stripes must leave some uncovered columns"
    assert bool((~row_mask).any()), "horizontal stripes must leave some uncovered rows"

    horizontal: list[tuple[int, int, int]] = []
    for r0, r1 in horizontal_bands:
        visible = np.unique(g[r0 : r1 + 1, ~col_mask])
        assert visible.size == 1, "horizontal stripe color must be readable outside intersections"
        horizontal.append((int(r0), int(r1), int(visible[0])))

    vertical: list[tuple[int, int, int]] = []
    for c0, c1 in vertical_bands:
        visible = np.unique(g[~row_mask, c0 : c1 + 1])
        assert visible.size == 1, "vertical stripe color must be readable outside intersections"
        vertical.append((int(c0), int(c1), int(visible[0])))

    h_heights = {r1 - r0 + 1 for r0, r1, _color in horizontal}
    v_widths = {c1 - c0 + 1 for c0, c1, _color in vertical}
    assert len(h_heights) == 1, "horizontal stripes must have equal thickness"
    assert len(v_widths) == 1, "vertical stripes must have equal thickness"

    colors = [color for _a, _b, color in horizontal] + [color for _a, _b, color in vertical]
    assert len(colors) == len(set(colors)), "stripe colors must be unique"
    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode an overlaid stripe lattice and recolor every crossing with the smaller stripe color.

    Input structure:
    - Several equal-thickness horizontal stripes of distinct colors.
    - Several equal-thickness vertical stripes of distinct colors.
    - One orientation was painted over the other, so each crossing shows only the top stripe.

    Rule:
    - Recover all horizontal and vertical stripe positions and colors.
    - For each crossing, compare the total area of the two whole stripes.
    - Paint the crossing with the color of the smaller-area stripe.
    - Keep all non-crossing cells unchanged.

    Key insight:
    - The hidden stripe layout is recoverable from repeated whole-row and whole-column patterns.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    horizontal, vertical = _infer_stripes(g)

    out = g.copy()
    for r0, r1, h_color in horizontal:
        h_area = (r1 - r0 + 1) * width
        for c0, c1, v_color in vertical:
            v_area = (c1 - c0 + 1) * height
            if h_area < v_area:
                out[r0 : r1 + 1, c0 : c1 + 1] = h_color
            elif v_area < h_area:
                out[r0 : r1 + 1, c0 : c1 + 1] = v_color

    return out


# --------------- generator (att1) ---------------

GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
STRIPE_COUNT_RANGE = (2, 4)
H_THICKNESS_RANGE = (1, 3)
V_THICKNESS_RANGE = (1, 3)
COLOR_COUNT = 10
GEN_RETRY_LIMIT = 400
GAP_RETRY_LIMIT = 60


def _positive_composition(total: int, parts: int) -> list[int] | None:
    if total < parts:
        return None
    for _ in range(GAP_RETRY_LIMIT):
        gaps = [1] * parts
        for _ in range(total - parts):
            gaps[random.randrange(parts)] += 1
        random.shuffle(gaps)
        if parts == 1 or len(set(gaps)) > 1:
            return gaps
    return None


def _layout_intervals(length: int, count: int, thickness: int) -> list[tuple[int, int]] | None:
    gaps = _positive_composition(length - count * thickness, count + 1)
    if gaps is None:
        return None

    intervals: list[tuple[int, int]] = []
    pos = gaps[0]
    for i in range(count):
        intervals.append((pos, pos + thickness - 1))
        pos += thickness + gaps[i + 1]
    return intervals


def _render(
    height: int,
    width: int,
    bg: int,
    horizontal: list[tuple[int, int, int]],
    vertical: list[tuple[int, int, int]],
    top_orientation: str,
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    if top_orientation == "h":
        for c0, c1, color in vertical:
            grid[:, c0 : c1 + 1] = color
        for r0, r1, color in horizontal:
            grid[r0 : r1 + 1, :] = color
        return grid

    assert top_orientation == "v"
    for r0, r1, color in horizontal:
        grid[r0 : r1 + 1, :] = color
    for c0, c1, color in vertical:
        grid[:, c0 : c1 + 1] = color
    return grid


def _thickness_options(height: int, width: int, n_h: int, n_v: int, winner: str) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for h_thick in range(H_THICKNESS_RANGE[0], H_THICKNESS_RANGE[1] + 1):
        for v_thick in range(V_THICKNESS_RANGE[0], V_THICKNESS_RANGE[1] + 1):
            free_rows = height - n_h * h_thick
            free_cols = width - n_v * v_thick
            if free_rows < n_h + 1 or free_cols < n_v + 1:
                continue
            if free_cols <= h_thick or free_rows <= v_thick:
                continue

            h_area = h_thick * width
            v_area = v_thick * height
            if winner == "h" and h_area < v_area:
                options.append((h_thick, v_thick))
            if winner == "v" and v_area < h_area:
                options.append((h_thick, v_thick))
    return options


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_h = random.randint(*STRIPE_COUNT_RANGE)
    n_v = random.randint(*STRIPE_COUNT_RANGE)
    top_orientation = random.choice(["h", "v"])
    winner = random.choice(["h", "v"])

    thicknesses = _thickness_options(height, width, n_h, n_v, winner)
    if not thicknesses:
        return None
    h_thick, v_thick = random.choice(thicknesses)

    row_intervals = _layout_intervals(height, n_h, h_thick)
    col_intervals = _layout_intervals(width, n_v, v_thick)
    if row_intervals is None or col_intervals is None:
        return None

    if 1 + n_h + n_v > COLOR_COUNT:
        return None
    colors = random.sample(range(COLOR_COUNT), 1 + n_h + n_v)
    bg = int(colors[0])
    h_colors = [int(color) for color in colors[1 : 1 + n_h]]
    v_colors = [int(color) for color in colors[1 + n_h :]]

    horizontal = [(r0, r1, color) for (r0, r1), color in zip(row_intervals, h_colors)]
    vertical = [(c0, c1, color) for (c0, c1), color in zip(col_intervals, v_colors)]

    inp = _render(height, width, bg, horizontal, vertical, top_orientation)
    out = _render(height, width, bg, horizontal, vertical, winner)

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stripe-overlay input")
