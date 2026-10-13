import random

import numpy as np


MAX_ATTEMPTS = 300
HEIGHT_RANGE = (8, 18)
WIDTH_RANGE = (10, 22)
BAND_COUNT_RANGE = (2, 5)
BAND_WIDTH_RANGE = (1, 4)
GAP_WIDTH_RANGE = (0, 3)
BAND_DENSITY_RANGE = (0.35, 0.8)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _band_specs(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        rows, cols = np.where(grid == color)
        assert rows.size > 0, "every foreground color must appear"
        left = int(cols.min())
        right = int(cols.max())
        band = grid[:, left : right + 1]
        assert np.all((band == bg) | (band == color)), "band contains an unexpected color"
        specs.append((left, right, color))

    specs.sort(key=lambda item: item[0])
    for (left, right, _color), (next_left, _next_right, _next_color) in zip(specs, specs[1:]):
        assert right < next_left, "bands must be disjoint and ordered left-to-right"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each vertical foreground band's background gaps with the next band's color.

    Bands are ordered by their leftmost column. Within each band's bounding
    column interval, background cells become the next band's color, with the
    last band wrapping around to the first band's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    bands = _band_specs(g, bg)
    assert len(bands) >= 2, "expected at least two foreground bands"

    out = g.copy()
    fill_colors = [bands[(i + 1) % len(bands)][2] for i in range(len(bands))]
    for (left, right, _color), fill_color in zip(bands, fill_colors):
        mask = out[:, left : right + 1] == bg
        out[:, left : right + 1][mask] = fill_color

    return out


def _sample_band_layout(width: int, band_count: int) -> list[tuple[int, int]]:
    usable = width - band_count
    assert usable >= band_count - 1

    widths = [random.randint(*BAND_WIDTH_RANGE) for _ in range(band_count)]
    total_width = sum(widths)
    remaining = width - total_width
    if remaining < 0:
        return []

    min_gaps = [0] * (band_count + 1)
    inner_gap_count = band_count - 1
    for i in range(inner_gap_count):
        if remaining <= 0:
            break
        gap = random.randint(GAP_WIDTH_RANGE[0], min(GAP_WIDTH_RANGE[1], remaining))
        min_gaps[i + 1] = gap
        remaining -= gap

    slack_slots = band_count + 1
    for _ in range(remaining):
        min_gaps[random.randrange(slack_slots)] += 1

    spans = []
    col = min_gaps[0]
    for i, band_width in enumerate(widths):
        left = col
        right = left + band_width - 1
        spans.append((left, right))
        col = right + 1 + min_gaps[i + 1]
    return spans


def _random_band_mask(height: int, width: int, density: float) -> np.ndarray:
    mask = np.random.random((height, width)) < density
    if not mask.any():
        mask[random.randrange(height), random.randrange(width)] = True
    if mask.all():
        mask[random.randrange(height), random.randrange(width)] = False

    # Keep each column represented so the full column interval is visibly part of the band.
    for c in range(width):
        if not mask[:, c].any():
            mask[random.randrange(height), c] = True

    if mask.all():
        mask[random.randrange(height), random.randrange(width)] = False
    return mask


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    band_count = random.randint(*BAND_COUNT_RANGE)
    if width < 2 * band_count:
        return None

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    colors = random.sample(palette, band_count)

    spans = _sample_band_layout(width, band_count)
    if len(spans) != band_count:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for color, (left, right) in zip(colors, spans):
        band_width = right - left + 1
        density = random.uniform(*BAND_DENSITY_RANGE)
        mask = _random_band_mask(height, band_width, density)
        grid[:, left : right + 1][mask] = color

    if np.count_nonzero(grid == bg) <= max(1, band_count):
        return None
    if _background_color(grid) != bg:
        return None

    bands = _band_specs(grid, bg)
    if len(bands) != band_count:
        return None

    # Ensure every band has at least one visible gap so the rule is readable.
    for left, right, color in bands:
        band = grid[:, left : right + 1]
        if not np.any(band == bg):
            return None
        if not np.any(band == color):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
