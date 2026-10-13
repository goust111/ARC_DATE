from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel()).most_common()
    assert counts, "expected a non-empty grid"
    return int(counts[0][0])


def _parse_instance(grid: np.ndarray) -> tuple[int, np.ndarray, list[int], list[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    non_bg_cols = np.flatnonzero(np.any(g != bg, axis=0))
    assert non_bg_cols.size > 0, "expected a legend column"

    legend_col = int(non_bg_cols[-1])
    assert legend_col >= 2, "legend column must leave room for canvas and separator"
    assert np.all(g[:, legend_col + 1 :] == bg), "expected the legend to be the rightmost non-background structure"

    separator_col = legend_col - 1
    assert np.all(g[:, separator_col] == bg), "expected a blank separator column before the legend"

    legend_rows = np.flatnonzero(g[:, legend_col] != bg)
    assert legend_rows.size > 0, "legend must contain at least one color"
    assert np.all(np.diff(legend_rows) == 1), "legend colors must form one contiguous vertical strip"
    legend = [int(g[r, legend_col]) for r in legend_rows]

    canvas = g[:, :separator_col].copy()
    seed_coords_raw = np.argwhere(canvas != bg)
    assert seed_coords_raw.shape == (2, 2), "expected exactly two seed pixels in the main canvas"
    seed_coords = [(int(r), int(c)) for r, c in seed_coords_raw]
    seed_colors = [int(canvas[r, c]) for r, c in seed_coords]
    assert seed_colors[0] != seed_colors[1], "the two seeds must have different colors"

    return bg, canvas, legend, seed_coords, seed_colors


def _distance_grid(shape: tuple[int, int], seed: tuple[int, int], metric: str) -> np.ndarray:
    h, w = shape
    rr = np.arange(h, dtype=int)[:, None]
    cc = np.arange(w, dtype=int)[None, :]
    dr = np.abs(rr - int(seed[0]))
    dc = np.abs(cc - int(seed[1]))
    if metric == "chebyshev":
        return np.maximum(dr, dc)
    if metric == "manhattan":
        return dr + dc
    raise ValueError(f"unsupported metric: {metric}")


def _solve_canvas(
    canvas_shape: tuple[int, int],
    bg: int,
    legend: list[int],
    seed_coords: list[tuple[int, int]],
    seed_colors: list[int],
    metric: str = "chebyshev",
) -> np.ndarray:
    h, w = canvas_shape
    out = np.full((h, w), bg, dtype=int)

    (r1, c1), (r2, c2) = seed_coords
    out[r1, c1] = int(seed_colors[0])
    out[r2, c2] = int(seed_colors[1])

    d1 = _distance_grid(canvas_shape, seed_coords[0], metric=metric)
    d2 = _distance_grid(canvas_shape, seed_coords[1], metric=metric)
    nearest = np.minimum(d1, d2)
    tie = d1 == d2

    if legend:
        palette = np.asarray(legend, dtype=int)
        fill_mask = (~tie) & (nearest >= 1) & (nearest <= len(legend))
        out[fill_mask] = palette[nearest[fill_mask] - 1]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow ordered color rings inside each seed's nearest Chebyshev territory, leaving equal-distance gaps blank.

    Input structure:
    - The left side is a background canvas containing exactly two isolated seed pixels of different colors.
    - On the far right, a vertical legend strip lists ring colors from top to bottom.
    - A blank separator column sits between the canvas and the legend strip.

    Rule:
    - Remove the separator and legend, keeping only the main canvas in the output.
    - For every background canvas cell, compare its Chebyshev distance to the two seeds.
    - If one seed is strictly nearer and that distance is d >= 1, paint the cell with the legend's d-th color; the seeds themselves stay unchanged.
    - Cells farther than the legend length stay background, and cells tied between the two seeds also stay background.

    Key insight:
    - The legend is an ordered distance palette, while the two seeds first partition the canvas by nearest-seed ownership; ties become a visible background seam.
    """
    bg, canvas, legend, seed_coords, seed_colors = _parse_instance(grid)
    return _solve_canvas(canvas.shape, bg, legend, seed_coords, seed_colors, metric="chebyshev")


# --------------- generator ---------------

PALETTE = tuple(range(10))
MIN_LEGEND_LEN = 3
MAX_LEGEND_LEN = 6
MIN_CANVAS_SIDE = 12
MAX_CANVAS_HEIGHT = 24
MAX_CANVAS_WIDTH = 27  # total width adds separator + legend
MIN_SEED_SEP = 4
MAX_GEN_ATTEMPTS = 600
MIN_VISIBLE_TERRITORY = 6
MIN_VISIBLE_TIE = 2
MIN_VISIBLE_OVERLAP = 6
MIN_FAR_BG = 6


def _sample_canvas_params() -> tuple[int, int, int, int]:
    legend_len = random.randint(MIN_LEGEND_LEN, MAX_LEGEND_LEN)
    min_side = max(MIN_CANVAS_SIDE, legend_len + 6)
    h = random.randint(min_side, MAX_CANVAS_HEIGHT)
    w = random.randint(min_side, min(MAX_CANVAS_WIDTH, 28))
    if w + 2 > 30:
        w = 28
    legend_top = random.randint(0, h - legend_len)
    return h, w, legend_len, legend_top


def _sample_seed_coords(h: int, w: int, legend_len: int, mode: str) -> list[tuple[int, int]] | None:
    max_sep = min(2 * legend_len, max(h, w) - 3)
    row_choices = list(range(1, h - 1))
    col_choices = list(range(1, w - 1))
    if len(row_choices) < 2 or len(col_choices) < 2:
        return None

    for _ in range(300):
        if mode == "same_row":
            r = random.choice(row_choices)
            c1, c2 = sorted(random.sample(col_choices, 2))
            p1, p2 = (r, c1), (r, c2)
        elif mode == "same_col":
            c = random.choice(col_choices)
            r1, r2 = sorted(random.sample(row_choices, 2))
            p1, p2 = (r1, c), (r2, c)
        else:
            r1, r2 = random.sample(row_choices, 2)
            c1, c2 = random.sample(col_choices, 2)
            p1, p2 = (r1, c1), (r2, c2)

        cheb_sep = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        if not (MIN_SEED_SEP <= cheb_sep <= max_sep):
            continue

        border_margin = min(
            p1[0],
            p1[1],
            h - 1 - p1[0],
            w - 1 - p1[1],
            p2[0],
            p2[1],
            h - 1 - p2[0],
            w - 1 - p2[1],
        )
        if border_margin < 2:
            continue

        return [p1, p2]
    return None


def _build_input(
    h: int,
    w: int,
    bg: int,
    legend_top: int,
    legend: list[int],
    seed_coords: list[tuple[int, int]],
    seed_colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w + 2), bg, dtype=int)
    for (r, c), color in zip(seed_coords, seed_colors):
        grid[r, c] = int(color)
    legend_col = w + 1
    for offset, color in enumerate(legend):
        grid[legend_top + offset, legend_col] = int(color)
    return grid


def _semantic_checks(grid: np.ndarray, mode: str) -> bool:
    bg, canvas, legend, seed_coords, seed_colors = _parse_instance(grid)
    out = main(grid)
    oracle = _solve_canvas(canvas.shape, bg, legend, seed_coords, seed_colors, metric="chebyshev")
    if not np.array_equal(out, oracle):
        return False

    h, w = canvas.shape
    d1 = _distance_grid((h, w), seed_coords[0], metric="chebyshev")
    d2 = _distance_grid((h, w), seed_coords[1], metric="chebyshev")
    nearest = np.minimum(d1, d2)
    tie = d1 == d2
    owned1 = d1 < d2
    owned2 = d2 < d1

    ring_mask = (~tie) & (nearest >= 1) & (nearest <= len(legend))
    tie_gap = tie & (d1 >= 1) & (d1 <= len(legend))
    overlap_mask = (d1 >= 1) & (d1 <= len(legend)) & (d2 >= 1) & (d2 <= len(legend)) & (~tie)
    far_bg = nearest > len(legend)

    if np.count_nonzero(ring_mask & owned1) < MIN_VISIBLE_TERRITORY:
        return False
    if np.count_nonzero(ring_mask & owned2) < MIN_VISIBLE_TERRITORY:
        return False
    if np.count_nonzero(tie_gap) < MIN_VISIBLE_TIE:
        return False
    if np.count_nonzero(overlap_mask) < MIN_VISIBLE_OVERLAP:
        return False
    if np.count_nonzero(far_bg) < MIN_FAR_BG:
        return False

    used_distances = {int(d) for d in nearest[ring_mask].ravel()}
    if len(used_distances) < min(3, len(legend)):
        return False

    used_ring_colors = set(int(v) for v in out.ravel()) - {bg, seed_colors[0], seed_colors[1]}
    if len(used_ring_colors) < min(3, len(legend)):
        return False

    manhattan_out = _solve_canvas(canvas.shape, bg, legend, seed_coords, seed_colors, metric="manhattan")
    if mode == "skew" and np.array_equal(out, manhattan_out):
        return False

    return not np.array_equal(canvas, out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        h, w, legend_len, legend_top = _sample_canvas_params()
        bg = random.choice(PALETTE)
        pool = [color for color in PALETTE if color != bg]
        random.shuffle(pool)
        seed_colors = pool[:2]
        legend = pool[2 : 2 + legend_len]

        mode = random.choices(
            population=["skew", "same_row", "same_col"],
            weights=[4, 2, 2],
            k=1,
        )[0]
        seed_coords = _sample_seed_coords(h, w, legend_len, mode)
        if seed_coords is None:
            continue

        grid = _build_input(h, w, bg, legend_top, legend, seed_coords, seed_colors)
        if _semantic_checks(grid, mode):
            return grid

    raise RuntimeError("failed to generate a valid nearest-partition ring puzzle input")
