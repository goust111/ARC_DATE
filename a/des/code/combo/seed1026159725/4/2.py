from __future__ import annotations

import numpy as np


GREY = 5
ORANGE = 7
COLOR_VALUES = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bresenham_points(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Return the rasterized line between two points, including both endpoints."""
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    step_r = 1 if r0 < r1 else -1
    step_c = 1 if c0 < c1 else -1
    err = dc - dr

    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        doubled = 2 * err
        if doubled > -dr:
            err -= dr
            c += step_c
        if doubled < dc:
            err += dc
            r += step_r
    return points


def _extract_bars(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 5, "grid is too small for baseline and bars"

    baseline_color = int(g[-1, 0])
    assert baseline_color == GREY, "bottom baseline must be grey"
    assert np.all(g[-1] == baseline_color), "bottom row must be a solid baseline"

    upper = g[:-1]
    bg = _dominant_color(upper)
    assert bg not in {baseline_color, ORANGE}, "background must differ from the reserved baseline and line colors"

    bar_cols = [c for c in range(w) if np.any(upper[:, c] != bg)]
    assert bar_cols, "expected at least one vertical bar"
    assert np.all(np.diff(bar_cols) >= 2), "bars must be separated by at least one background column"

    bars: list[dict[str, int]] = []
    seen_colors: set[int] = set()
    for c in bar_cols:
        col = upper[:, c]
        filled_rows = np.flatnonzero(col != bg)
        assert filled_rows.size > 0
        assert np.array_equal(filled_rows, np.arange(filled_rows[0], h - 1)), "each bar must be vertically contiguous"

        colors = np.unique(col[filled_rows])
        assert colors.size == 1, "each bar must be monochrome"
        color = int(colors[0])
        assert color not in {bg, baseline_color, ORANGE}, "bars must use non-background, non-reserved colors"
        assert color not in seen_colors, "bars must have distinct colors"
        seen_colors.add(color)

        bars.append(
            {
                "col": int(c),
                "height": int(h - 1 - filled_rows[0]),
                "color": color,
            }
        )

    assert len({bar["height"] for bar in bars}) == len(bars), "bar heights must be distinct"
    return bg, baseline_color, bars


def _render_sorted_output(
    shape: tuple[int, int], bg: int, baseline_color: int, bars: list[dict[str, int]]
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    out[-1, :] = baseline_color

    slot_cols = sorted(bar["col"] for bar in bars)
    ordered = sorted(bars, key=lambda bar: (bar["height"], bar["col"]))

    top_points: list[tuple[int, int]] = []
    for slot_c, bar in zip(slot_cols, ordered):
        height = int(bar["height"])
        color = int(bar["color"])
        top_r = h - 1 - height
        out[top_r : h - 1, slot_c] = color
        top_points.append((top_r, slot_c))

    for (r0, c0), (r1, c1) in zip(top_points, top_points[1:]):
        for r, c in _bresenham_points(r0, c0, r1, c1):
            if c0 < c < c1:
                out[r, c] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort bottom-aligned vertical bars by height and connect sorted bar tops with an orange line.

    Input structure
    - The bottom row is a solid grey baseline.
    - Above the baseline are one-pixel-wide, bottom-aligned vertical bars.
    - Bars use distinct colors, distinct heights, and are separated by at least one background column.

    Rule
    - Read each bar's height.
    - Reorder the bars from shortest to tallest, placing them back into the original left-to-right bar slots.
    - Draw an orange diagonal line segment between the tops of each pair of consecutive sorted bars.
    - Keep the grey baseline unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, baseline_color, bars = _extract_bars(g)
    return _render_sorted_output(g.shape, bg, baseline_color, bars)


def _sample_bars(rng: np.random.Generator) -> tuple[tuple[int, int], int, list[dict[str, int]]]:
    n_bars = int(rng.integers(2, 7))

    left_margin = int(rng.integers(0, 4))
    right_margin = int(rng.integers(0, 4))
    gaps = [int(rng.integers(1, 5)) for _ in range(n_bars - 1)]
    cols = [left_margin]
    for gap in gaps:
        cols.append(cols[-1] + gap + 1)
    width = cols[-1] + right_margin + 1

    max_height = int(rng.integers(n_bars + 1, n_bars + 8))
    heights = sorted(int(v) for v in rng.choice(np.arange(1, max_height + 1), size=n_bars, replace=False))
    top_margin = int(rng.integers(1, 4))
    height = top_margin + max_height + 1

    bg_choices = [c for c in COLOR_VALUES if c not in {GREY, ORANGE}]
    bg = int(rng.choice(bg_choices))
    bar_colors = list(map(int, rng.choice([c for c in bg_choices if c != bg], size=n_bars, replace=False)))

    sorted_bars = [{"height": h, "color": color} for h, color in zip(heights, bar_colors)]
    order = list(map(int, rng.permutation(n_bars)))
    if order == list(range(n_bars)):
        order = order[1:] + order[:1]

    bars = []
    for col, idx in zip(cols, order):
        bars.append(
            {
                "col": int(col),
                "height": int(sorted_bars[idx]["height"]),
                "color": int(sorted_bars[idx]["color"]),
            }
        )
    return (height, width), bg, bars


def _render_input(shape: tuple[int, int], bg: int, bars: list[dict[str, int]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), bg, dtype=int)
    grid[-1, :] = GREY
    for bar in bars:
        top_r = h - 1 - int(bar["height"])
        grid[top_r : h - 1, int(bar["col"])] = int(bar["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    shape, bg, bars = _sample_bars(rng)
    grid = _render_input(shape, bg, bars)

    try:
        out = main(grid)
    except Exception:
        return None

    expected = _render_sorted_output(grid.shape, bg, GREY, bars)
    if not np.array_equal(out, expected):
        return None

    values, counts = np.unique(grid, return_counts=True)
    color_counts = {int(v): int(n) for v, n in zip(values, counts)}
    if color_counts[bg] <= max(color_counts.get(GREY, 0), max(bar["height"] for bar in bars)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
