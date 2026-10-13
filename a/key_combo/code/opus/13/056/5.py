import random

import numpy as np


BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
SIGNAL_COLORS = (RED, GREY)

WIDTH_CHOICES = (5, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _swap_signal(color: int) -> int:
    assert color in SIGNAL_COLORS, color
    return GREY if int(color) == RED else RED


def _swap_signal_array(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=int)
    assert np.all(np.isin(vals, SIGNAL_COLORS))
    return np.where(vals == RED, GREY, RED).astype(int)


def _parse_top_band(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    row0 = np.asarray(grid[0], dtype=int)
    signal_mask = np.isin(row0, SIGNAL_COLORS)
    assert signal_mask.any(), "expected a top-row red/grey band"

    cols = np.flatnonzero(signal_mask)
    left = int(cols[0])
    right = int(cols[-1])
    assert bool(np.all(signal_mask[left : right + 1])), "top-row band must be contiguous"
    assert not bool(signal_mask[:left].any())
    assert not bool(signal_mask[right + 1 :].any())

    band = row0[left : right + 1].copy()
    assert band.size % 2 == 1 and band.size >= 5, "expected an odd band width of at least 5"
    assert bool(np.all(np.isin(band, SIGNAL_COLORS)))
    assert bool(np.all(band[1:] != band[:-1])), "top-row band must alternate red and grey"
    return left, right, band


def _parse_obstacle(grid: np.ndarray) -> tuple[int, int, int]:
    blue = np.argwhere(grid == BLUE)
    assert blue.size > 0, "expected a blue obstacle segment"
    rows = blue[:, 0]
    assert bool(np.all(rows == rows[0])), "blue obstacle must lie on a single row"

    row = int(rows[0])
    left = int(blue[:, 1].min())
    right = int(blue[:, 1].max())
    assert int(blue.shape[0]) == right - left + 1, "blue obstacle must be contiguous"
    return row, left, right


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    band_left, band_right, band = _parse_top_band(g)
    obstacle_row, obstacle_left, obstacle_right = _parse_obstacle(g)

    mask = np.ones_like(g, dtype=bool)
    mask[0, band_left : band_right + 1] = False
    mask[obstacle_row, obstacle_left : obstacle_right + 1] = False
    assert bool(np.all(g[mask] == ORANGE)), "unexpected extra foreground outside the band and obstacle"

    out = g.copy()
    span_width = int(band.size)
    tip_depth = span_width // 2

    for r in range(1, min(height, tip_depth + 1)):
        lo = band_left + r
        hi = band_right - r
        if lo > hi:
            break
        colors = band[r : span_width - r]
        if r % 2 == 1:
            colors = _swap_signal_array(colors)
        out[r, lo : hi + 1] = colors

    if obstacle_row == tip_depth + 1:
        for d in range(1, height - obstacle_row):
            source_depth = tip_depth - d + 1
            if source_depth < 0:
                break
            lo = band_left + source_depth
            hi = band_right - source_depth
            colors = band[source_depth : span_width - source_depth]
            if (source_depth + 1) % 2 == 1:
                colors = _swap_signal_array(colors)
            out[obstacle_row + d, lo : hi + 1] = colors

    return out


def _semantic_validate(inp: np.ndarray, out: np.ndarray) -> dict:
    g = np.asarray(inp, dtype=int)
    result = np.asarray(out, dtype=int)
    assert g.ndim == 2 and result.ndim == 2

    band_left, band_right, band = _parse_top_band(g)
    obstacle_row, obstacle_left, obstacle_right = _parse_obstacle(g)
    span_width = int(band.size)
    tip_depth = span_width // 2

    assert obstacle_row == tip_depth + 1, "generator must place the obstacle exactly at the collision row"
    assert obstacle_left <= band_left and obstacle_right >= band_right, "obstacle must span the whole band"
    assert g.shape[0] > obstacle_row + 1, "obstacle must leave room for the rebound"
    assert np.array_equal(result, _expected_output(g))

    assert np.array_equal(result[0], g[0]), "top row must stay unchanged"
    assert np.array_equal(result[obstacle_row], g[obstacle_row]), "obstacle row must stay unchanged"

    for r in range(1, tip_depth + 1):
        lo = band_left + r
        hi = band_right - r
        expected = band[r : span_width - r]
        if r % 2 == 1:
            expected = _swap_signal_array(expected)
        assert np.array_equal(result[r, lo : hi + 1], expected)

    bounce_rows = int(g.shape[0] - obstacle_row - 1)
    for d in range(1, bounce_rows + 1):
        source_depth = tip_depth - d + 1
        if source_depth < 0:
            break
        lo = band_left + source_depth
        hi = band_right - source_depth
        mirrored = band[source_depth : span_width - source_depth]
        if source_depth % 2 == 1:
            mirrored = _swap_signal_array(mirrored)
        bounced = result[obstacle_row + d, lo : hi + 1]
        assert np.array_equal(bounced, _swap_signal_array(mirrored)), "bounce colors must continue with one extra flip"
        assert not np.array_equal(bounced, mirrored), "bounce must not be a plain vertical mirror"

    return {
        "span_width": span_width,
        "full_rebound": bounce_rows >= tip_depth + 1,
        "bounce_rows": bounce_rows,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a shrinking top-row wedge, then reflect it across a blue collision row into an expanding rebound wedge.

    Input structure:
    - The background is orange.
    - Row 0 contains one contiguous odd-width band of alternating red and grey cells.
    - Lower in the grid there is one contiguous horizontal blue segment acting as an obstacle.

    Rule:
    - Copy the input.
    - For each row r below the top band, fill the narrowed interval [L+r, R-r] using the top-row band colors in the same columns, flipping red and grey once per row.
    - When the band has shrunk to its tip, if the next row is the blue obstacle, continue below the obstacle by expanding the wedge again.
    - Each rebound row uses the symmetric shrinking row's colors with one additional red/grey swap, while the blue obstacle itself remains unchanged.

    Key insight:
    - The lower wedge is not a plain mirror of the upper wedge: the collision inserts one extra color flip, so rebound rows must be derived from the shrinking history rather than copied directly.
    """
    return _expected_output(grid)


def _build_band(width: int, start_color: int) -> np.ndarray:
    band = np.empty(int(width), dtype=int)
    color = int(start_color)
    for i in range(int(width)):
        band[i] = color
        color = _swap_signal(color)
    return band


def _generate_attempt() -> np.ndarray | None:
    span_width = int(random.choice(WIDTH_CHOICES))
    tip_depth = span_width // 2

    left_pad = random.randint(0, 6)
    right_pad = random.randint(0, 6)
    total_width = span_width + left_pad + right_pad
    if total_width > 30:
        return None

    full_rebound = random.random() < 0.5
    if full_rebound:
        bounce_rows = tip_depth + 1
        extra_bottom = random.randint(0, 4)
    else:
        bounce_rows = random.randint(1, tip_depth)
        extra_bottom = 0

    height = (tip_depth + 1) + bounce_rows + 1 + extra_bottom
    if height > 30:
        return None

    grid = np.full((height, total_width), ORANGE, dtype=int)

    band_left = left_pad
    band_right = band_left + span_width - 1
    band = _build_band(span_width, random.choice(SIGNAL_COLORS))
    grid[0, band_left : band_right + 1] = band

    obstacle_row = tip_depth + 1
    obstacle_left = band_left - random.randint(0, left_pad)
    obstacle_right = band_right + random.randint(0, right_pad)
    grid[obstacle_row, obstacle_left : obstacle_right + 1] = BLUE

    out = main(grid)
    info = _semantic_validate(grid, out)

    if full_rebound != info["full_rebound"]:
        return None
    if info["span_width"] != span_width or info["bounce_rows"] != bounce_rows + extra_bottom:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    seen = set()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is None:
            continue
        key = (tuple(map(int, grid.shape)), grid.tobytes())
        if key in seen:
            continue
        seen.add(key)
        return grid
    raise RuntimeError("failed to generate a valid diagonal-wedge rebound input")
