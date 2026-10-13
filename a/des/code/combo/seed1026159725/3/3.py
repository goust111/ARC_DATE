import numpy as np


GREY = 5
INPUT_COLORS = tuple(color for color in range(10) if color != GREY)
MAX_GENERATION_ATTEMPTS = 300
MIN_BANDS = 2
MAX_BANDS = 6
MAX_BAND_WIDTH = 5
MIN_THICKNESS = 2
MAX_THICKNESS = 7
MAX_MARGIN = 4


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_runs(profile: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(profile) + 1):
        if idx == len(profile) or int(profile[idx]) != int(profile[start]):
            runs.append((int(profile[start]), idx - start))
            start = idx
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode a striped rectangle as alternating color markers and grey count bars.

    Input structure:
    - The dominant color is the background.
    - One rectangular non-background region contains horizontal or vertical bands.
    - The bands are uniform across their short axis and vary only along one axis.

    Rule:
    - Read the color and width of each contiguous band in order.
    - Replace each band by one pixel of its color followed by a grey run whose
      length equals the original band width.
    - Keep the output strip aligned to the original band direction.

    Key insight:
    - The output is an encoded legend for the stripe widths rather than a crop
      of the original picture.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    bg = _dominant_color(g)
    mask = g != bg
    assert mask.any(), "grid must contain a non-background banded object"

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "banded object must fill its bounding box"
    obj = g[r0 : r1 + 1, c0 : c1 + 1]

    row_uniform = bool(np.all(obj == obj[:, [0]]))
    col_uniform = bool(np.all(obj == obj[[0], :]))
    assert row_uniform ^ col_uniform, "object must contain either horizontal or vertical bands"

    if row_uniform:
        profile = obj[:, 0]
        vertical_output = True
    else:
        profile = obj[0, :]
        vertical_output = False

    runs = _extract_runs(profile)
    assert len(runs) >= 2, "expected multiple color bands"
    assert len({color for color, _ in runs}) >= 2, "expected multiple colors"

    encoded = []
    for color, width in runs:
        encoded.append(color)
        encoded.extend([GREY] * width)

    out = np.array(encoded, dtype=int)
    if vertical_output:
        return out.reshape((-1, 1))
    return out.reshape((1, -1))


def _sample_band_colors(rng: np.random.Generator, count: int, bg: int) -> list[int]:
    usable = [color for color in INPUT_COLORS if color != bg]
    pool_size = int(rng.integers(2, min(5, count, len(usable)) + 1))
    pool = [int(color) for color in rng.choice(usable, size=pool_size, replace=False)]

    colors = [int(rng.choice(pool))]
    for _ in range(1, count):
        choices = [color for color in pool if color != colors[-1]]
        colors.append(int(rng.choice(choices)))
    return colors


def _make_grid(
    orientation: int,
    widths: np.ndarray,
    thickness: int,
    colors: list[int],
    bg: int,
    margins: tuple[int, int, int, int],
) -> np.ndarray:
    top, bottom, left, right = margins
    span = int(widths.sum())
    if orientation == 0:
        obj_h, obj_w = thickness, span
    else:
        obj_h, obj_w = span, thickness

    grid = np.full((top + obj_h + bottom, left + obj_w + right), bg, dtype=int)

    cursor = 0
    for color, width in zip(colors, widths):
        width = int(width)
        if orientation == 0:
            grid[top : top + obj_h, left + cursor : left + cursor + width] = color
        else:
            grid[top + cursor : top + cursor + width, left : left + obj_w] = color
        cursor += width
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = int(rng.integers(0, 2))
    bg = int(rng.choice(INPUT_COLORS))
    band_count = int(rng.integers(MIN_BANDS, MAX_BANDS + 1))
    widths = rng.integers(1, MAX_BAND_WIDTH + 1, size=band_count)
    thickness = int(rng.integers(MIN_THICKNESS, MAX_THICKNESS + 1))
    colors = _sample_band_colors(rng, band_count, bg)

    top, bottom, left, right = [int(x) for x in rng.integers(0, MAX_MARGIN + 1, size=4)]
    grid = _make_grid(orientation, widths, thickness, colors, bg, (top, bottom, left, right))

    color_areas: dict[int, int] = {}
    for color, width in zip(colors, widths):
        color_areas[color] = color_areas.get(color, 0) + int(width) * thickness

    bg_count = int(np.count_nonzero(grid == bg))
    needed_bg = max(color_areas.values()) + 1
    if bg_count < needed_bg:
        extra = needed_bg - bg_count
        if grid.shape[0] <= grid.shape[1]:
            add_rows = (extra + grid.shape[1] - 1) // grid.shape[1]
            extra_block = np.full((add_rows, grid.shape[1]), bg, dtype=int)
            if rng.random() < 0.5:
                grid = np.vstack([extra_block, grid])
            else:
                grid = np.vstack([grid, extra_block])
        else:
            add_cols = (extra + grid.shape[0] - 1) // grid.shape[0]
            extra_block = np.full((grid.shape[0], add_cols), bg, dtype=int)
            if rng.random() < 0.5:
                grid = np.hstack([extra_block, grid])
            else:
                grid = np.hstack([grid, extra_block])

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if orientation == 0 and out.shape[0] != 1:
        return None
    if orientation == 1 and out.shape[1] != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid striped-band input")
