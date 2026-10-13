from __future__ import annotations

import numpy as np


BACKGROUND_COLOR = 0
COLOR_POOL = np.arange(1, 10)
WIDTH_CHOICES = np.arange(1, 5)

MIN_HEIGHT = 3
MAX_HEIGHT = 10
MIN_DISTINCT_STRIPES = 3
MAX_DISTINCT_STRIPES = 4
MIN_TIED_STRIPES = 4
MAX_TIED_STRIPES = 8

WIDTH_SAMPLE_RETRIES = 80
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _extract_vertical_stripes(g: np.ndarray) -> list[dict[str, int]]:
    """Return full-height vertical stripes in their original left-to-right order."""
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a non-empty grid"

    column_colors = np.asarray(g[0], dtype=int)
    assert np.array_equal(g, np.broadcast_to(column_colors, (height, width)))
    assert np.all(column_colors != BACKGROUND_COLOR), "stripes must be foreground colors"

    stripes: list[dict[str, int]] = []
    start = 0
    current = int(column_colors[0])
    for col in range(1, width):
        color = int(column_colors[col])
        if color != current:
            stripe_width = col - start
            assert 1 <= stripe_width <= 4
            stripes.append({"color": current, "width": stripe_width, "col": start})
            start = col
            current = color

    stripe_width = width - start
    assert 1 <= stripe_width <= 4
    stripes.append({"color": current, "width": stripe_width, "col": start})

    colors = [stripe["color"] for stripe in stripes]
    assert len(colors) == len(set(colors)), "stripe colors must be unique"
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """Turn vertical stripes into width-sorted horizontal stripes."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    stripes = _extract_vertical_stripes(g)
    ordered = sorted(stripes, key=lambda stripe: stripe["width"])

    out = np.full((width, height), BACKGROUND_COLOR, dtype=int)
    row = 0
    for stripe in ordered:
        band_height = int(stripe["width"])
        out[row : row + band_height, :] = int(stripe["color"])
        row += band_height

    assert row == width
    return out


def _render_input(height: int, widths: list[int], colors: list[int]) -> np.ndarray:
    grid = np.full((height, int(sum(widths))), BACKGROUND_COLOR, dtype=int)
    col = 0
    for stripe_width, color in zip(widths, colors):
        grid[:, col : col + stripe_width] = int(color)
        col += stripe_width
    return grid


def _expected_output(height: int, widths: list[int], colors: list[int]) -> np.ndarray:
    ordered = sorted(zip(widths, colors), key=lambda item: item[0])
    out = np.full((int(sum(widths)), height), BACKGROUND_COLOR, dtype=int)
    row = 0
    for stripe_width, color in ordered:
        out[row : row + stripe_width, :] = int(color)
        row += stripe_width
    return out


def _sample_distinct_widths() -> list[int] | None:
    for _ in range(WIDTH_SAMPLE_RETRIES):
        count = int(_RNG.integers(MIN_DISTINCT_STRIPES, MAX_DISTINCT_STRIPES + 1))
        widths = _RNG.choice(WIDTH_CHOICES, size=count, replace=False).astype(int).tolist()
        if widths != sorted(widths):
            return widths
    return None


def _sample_tied_widths() -> list[int] | None:
    for _ in range(WIDTH_SAMPLE_RETRIES):
        count = int(_RNG.integers(MIN_TIED_STRIPES, MAX_TIED_STRIPES + 1))
        widths = _RNG.integers(1, 5, size=count).astype(int).tolist()
        unique_widths = set(widths)
        if len(unique_widths) == count or len(unique_widths) == 1:
            continue
        if widths == sorted(widths):
            continue
        return widths
    return None


def _gen_attempt() -> np.ndarray | None:
    use_ties = bool(_RNG.integers(0, 2))
    widths = _sample_tied_widths() if use_ties else _sample_distinct_widths()
    if widths is None:
        return None

    height = int(_RNG.integers(MIN_HEIGHT, MAX_HEIGHT + 1))
    colors = _RNG.choice(COLOR_POOL, size=len(widths), replace=False).astype(int).tolist()

    grid = _render_input(height, widths, colors)
    expected = _expected_output(height, widths, colors)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
