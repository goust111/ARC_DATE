from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GRID_MIN_SIZE = 12
GRID_MAX_SIZE = 24
INNER_MIN = 6
INNER_MAX = 10
MARGIN_MIN = 1
MARGIN_MAX = 4
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_markers(marker_coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = marker_coords[:, 0]
    cols = marker_coords[:, 1]
    r_min = int(rows.min())
    r_max = int(rows.max())
    c_min = int(cols.min())
    c_max = int(cols.max())

    assert np.sum(rows == r_min) == 1, "expected a unique top marker"
    assert np.sum(rows == r_max) == 1, "expected a unique bottom marker"
    assert np.sum(cols == c_min) == 1, "expected a unique left marker"
    assert np.sum(cols == c_max) == 1, "expected a unique right marker"
    return r_min, c_min, r_max, c_max


def _parse_grid(grid: np.ndarray) -> dict[str, int | tuple[int, int] | np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, GREEN}), "unexpected colors in input"

    marker_coords = np.argwhere(g == RED)
    assert marker_coords.shape == (4, 2), "expected exactly four red markers"

    blue_coords = np.argwhere(g == BLUE)
    green_coords = np.argwhere(g == GREEN)
    assert blue_coords.shape == (1, 2), "expected exactly one blue seed"
    assert green_coords.shape == (1, 2), "expected exactly one green seed"

    r_min, c_min, r_max, c_max = _bbox_from_markers(marker_coords)
    frame_r0, frame_c0 = r_min + 1, c_min + 1
    frame_r1, frame_c1 = r_max - 1, c_max - 1
    inner_r0, inner_c0 = r_min + 2, c_min + 2
    inner_r1, inner_c1 = r_max - 2, c_max - 2

    assert 0 <= frame_r0 < frame_r1 < g.shape[0], "marker rows must enclose a frame"
    assert 0 <= frame_c0 < frame_c1 < g.shape[1], "marker cols must enclose a frame"
    assert inner_r0 <= inner_r1 and inner_c0 <= inner_c1, "yellow frame must have a nonempty interior"
    assert inner_r1 - inner_r0 + 1 >= INNER_MIN, "inner area height must be at least 6"
    assert inner_c1 - inner_c0 + 1 >= INNER_MIN, "inner area width must be at least 6"

    blue = tuple(map(int, blue_coords[0]))
    green = tuple(map(int, green_coords[0]))
    for seed in (blue, green):
        assert inner_r0 <= seed[0] <= inner_r1 and inner_c0 <= seed[1] <= inner_c1, "seeds must lie strictly inside the yellow frame"
    assert blue[0] != green[0], "seeds must not share a row"
    assert blue[1] != green[1], "seeds must not share a column"

    return {
        "grid": g,
        "markers": marker_coords,
        "blue": blue,
        "green": green,
        "frame_r0": frame_r0,
        "frame_c0": frame_c0,
        "frame_r1": frame_r1,
        "frame_c1": frame_c1,
        "inner_r0": inner_r0,
        "inner_c0": inner_c0,
        "inner_r1": inner_r1,
        "inner_c1": inner_c1,
    }


def _stripe_spec(parsed: dict[str, int | tuple[int, int] | np.ndarray]) -> dict[str, int | str]:
    blue_r, blue_c = parsed["blue"]  # type: ignore[misc]
    green_r, green_c = parsed["green"]  # type: ignore[misc]
    inner_r0 = int(parsed["inner_r0"])
    inner_c0 = int(parsed["inner_c0"])

    blue_dr = blue_r - inner_r0
    blue_dc = blue_c - inner_c0
    green_dr = green_r - inner_r0
    green_dc = green_c - inner_c0

    row_gap = abs(blue_dr - green_dr)
    col_gap = abs(blue_dc - green_dc)
    assert row_gap > 0 and col_gap > 0, "seeds must differ on both axes"

    if row_gap > col_gap:
        if blue_dr < green_dr:
            first_color = BLUE
            phase = blue_dr
        else:
            first_color = GREEN
            phase = green_dr
        return {
            "orientation": "horizontal",
            "period": row_gap,
            "phase": phase,
            "first_color": first_color,
            "second_color": GREEN if first_color == BLUE else BLUE,
        }

    if blue_dc < green_dc:
        first_color = BLUE
        phase = blue_dc
    else:
        first_color = GREEN
        phase = green_dc
    return {
        "orientation": "vertical",
        "period": col_gap,
        "phase": phase,
        "first_color": first_color,
        "second_color": GREEN if first_color == BLUE else BLUE,
    }


def _paint_frame(out: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    out[r0, c0 : c1 + 1] = YELLOW
    out[r1, c0 : c1 + 1] = YELLOW
    out[r0 : r1 + 1, c0] = YELLOW
    out[r0 : r1 + 1, c1] = YELLOW


def _oracle_transform(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_grid(grid)
    spec = _stripe_spec(parsed)

    out = parsed["grid"].copy()  # type: ignore[union-attr]
    frame_r0 = int(parsed["frame_r0"])
    frame_c0 = int(parsed["frame_c0"])
    frame_r1 = int(parsed["frame_r1"])
    frame_c1 = int(parsed["frame_c1"])
    inner_r0 = int(parsed["inner_r0"])
    inner_c0 = int(parsed["inner_c0"])
    inner_r1 = int(parsed["inner_r1"])
    inner_c1 = int(parsed["inner_c1"])
    _paint_frame(out, frame_r0, frame_c0, frame_r1, frame_c1)

    period = int(spec["period"])
    phase = int(spec["phase"])
    first_color = int(spec["first_color"])
    second_color = int(spec["second_color"])

    if spec["orientation"] == "horizontal":
        for rel_r, r in enumerate(range(inner_r0, inner_r1 + 1)):
            color = first_color if ((rel_r - phase) % (2 * period)) < period else second_color
            out[r, inner_c0 : inner_c1 + 1] = color
    else:
        for rel_c, c in enumerate(range(inner_c0, inner_c1 + 1)):
            color = first_color if ((rel_c - phase) % (2 * period)) < period else second_color
            out[inner_r0 : inner_r1 + 1, c] = color

    out[tuple(parsed["markers"].T)] = RED  # type: ignore[union-attr]
    return out


def _phase_zero_shortcut(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_grid(grid)
    spec = _stripe_spec(parsed)

    out = parsed["grid"].copy()  # type: ignore[union-attr]
    frame_r0 = int(parsed["frame_r0"])
    frame_c0 = int(parsed["frame_c0"])
    frame_r1 = int(parsed["frame_r1"])
    frame_c1 = int(parsed["frame_c1"])
    inner_r0 = int(parsed["inner_r0"])
    inner_c0 = int(parsed["inner_c0"])
    inner_r1 = int(parsed["inner_r1"])
    inner_c1 = int(parsed["inner_c1"])
    _paint_frame(out, frame_r0, frame_c0, frame_r1, frame_c1)

    period = int(spec["period"])
    first_color = int(spec["first_color"])
    second_color = int(spec["second_color"])
    if spec["orientation"] == "horizontal":
        for rel_r, r in enumerate(range(inner_r0, inner_r1 + 1)):
            color = first_color if (rel_r % (2 * period)) < period else second_color
            out[r, inner_c0 : inner_c1 + 1] = color
    else:
        for rel_c, c in enumerate(range(inner_c0, inner_c1 + 1)):
            color = first_color if (rel_c % (2 * period)) < period else second_color
            out[inner_r0 : inner_r1 + 1, c] = color

    out[tuple(parsed["markers"].T)] = RED  # type: ignore[union-attr]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the inset frame from four red markers, then fill its interior with seed-phased blue/green stripes.

    Input structure:
    - A black background contains exactly four red marker pixels with unique top, bottom, left, and right extrema.
    - Exactly one blue seed and one green seed lie strictly inside the rectangle implied by those red extrema.
    - The red extrema are far enough apart that the yellow frame one cell inward still encloses a sizable interior.

    Rule:
    - Read the red extrema `(r_min, r_max, c_min, c_max)` and draw a yellow rectangular frame at rows `r_min+1`, `r_max-1` and cols `c_min+1`, `c_max-1`.
    - Let `I` be the strict interior of that yellow frame and reinterpret both seeds in `I`'s local `(row, col)` coordinate system.
    - If the seeds differ more in local row than in local column, fill `I` with horizontal stripes; otherwise fill it with vertical stripes.
    - The stripe period is the larger local-axis separation, and the phase is anchored at the seed with the smaller local row/col on the stripe axis, so the two seed rows or columns land in opposite colors.
    - Preserve the red markers and yellow frame; the seeds themselves are overwritten by the stripe fill.

    Key insight:
    - The seeds do not only choose colors and period; their positions must be read relative to the frame-defined local interior, because that local offset fixes the stripe phase.
    """
    parsed = _parse_grid(grid)
    spec = _stripe_spec(parsed)

    out = parsed["grid"].copy()  # type: ignore[union-attr]
    frame_r0 = int(parsed["frame_r0"])
    frame_c0 = int(parsed["frame_c0"])
    frame_r1 = int(parsed["frame_r1"])
    frame_c1 = int(parsed["frame_c1"])
    inner_r0 = int(parsed["inner_r0"])
    inner_c0 = int(parsed["inner_c0"])
    inner_r1 = int(parsed["inner_r1"])
    inner_c1 = int(parsed["inner_c1"])

    _paint_frame(out, frame_r0, frame_c0, frame_r1, frame_c1)

    period = int(spec["period"])
    phase = int(spec["phase"])
    first_color = int(spec["first_color"])
    second_color = int(spec["second_color"])
    inner_h = inner_r1 - inner_r0 + 1
    inner_w = inner_c1 - inner_c0 + 1

    if spec["orientation"] == "horizontal":
        rel_rows = np.arange(inner_h, dtype=int)
        row_colors = np.where(((rel_rows - phase) % (2 * period)) < period, first_color, second_color)
        out[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] = row_colors[:, None]
    else:
        rel_cols = np.arange(inner_w, dtype=int)
        col_colors = np.where(((rel_cols - phase) % (2 * period)) < period, first_color, second_color)
        out[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] = col_colors[None, :]

    out[tuple(parsed["markers"].T)] = RED  # type: ignore[union-attr]
    return out


def _build_canvas(
    inner_h: int,
    inner_w: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> tuple[np.ndarray, dict[str, int]]:
    outer_h = inner_h + 4
    outer_w = inner_w + 4
    height = top_margin + outer_h + bottom_margin
    width = left_margin + outer_w + right_margin
    assert GRID_MIN_SIZE <= height <= GRID_MAX_SIZE
    assert GRID_MIN_SIZE <= width <= GRID_MAX_SIZE

    r_min = top_margin
    c_min = left_margin
    r_max = r_min + outer_h - 1
    c_max = c_min + outer_w - 1

    grid = np.zeros((height, width), dtype=int)
    return grid, {"r_min": r_min, "c_min": c_min, "r_max": r_max, "c_max": c_max}


def _sample_horizontal_seeds(
    rng: np.random.Generator, inner_h: int, inner_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    max_period = min(5, inner_h - 2)
    if max_period < 2:
        raise ValueError("horizontal branch needs enough inner rows")
    period = int(rng.integers(2, max_period + 1))
    start_r = int(rng.integers(1, inner_h - period))
    col_gap = int(rng.integers(1, min(period, inner_w)))
    col_start = int(rng.integers(0, inner_w - col_gap))
    if bool(rng.integers(0, 2)):
        c1, c2 = col_start, col_start + col_gap
    else:
        c1, c2 = col_start + col_gap, col_start
    return (start_r, c1), (start_r + period, c2)


def _sample_vertical_seeds(
    rng: np.random.Generator, inner_h: int, inner_w: int
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[bool, int]]:
    tie_case = bool(rng.integers(0, 2))
    if tie_case:
        max_period = min(5, inner_h - 1, inner_w - 2)
        if max_period < 2:
            tie_case = False
    if tie_case:
        period = int(rng.integers(2, max_period + 1))
        start_c = int(rng.integers(1, inner_w - period))
        start_r = int(rng.integers(0, inner_h - period))
        return ((start_r, start_c), (start_r + period, start_c + period)), (True, period)

    max_period = min(5, inner_w - 2)
    if max_period < 2:
        raise ValueError("vertical branch needs enough inner cols")
    period = int(rng.integers(2, max_period + 1))
    start_c = int(rng.integers(1, inner_w - period))
    row_gap = int(rng.integers(1, min(period, inner_h)))
    start_r = int(rng.integers(0, inner_h - row_gap))
    if bool(rng.integers(0, 2)):
        r1, r2 = start_r, start_r + row_gap
    else:
        r1, r2 = start_r + row_gap, start_r
    return ((r1, start_c), (r2, start_c + period)), (False, period)


def _place_markers(grid: np.ndarray, bounds: dict[str, int], rng: np.random.Generator) -> None:
    r_min = bounds["r_min"]
    c_min = bounds["c_min"]
    r_max = bounds["r_max"]
    c_max = bounds["c_max"]
    top = (r_min, int(rng.integers(c_min + 1, c_max)))
    bottom = (r_max, int(rng.integers(c_min + 1, c_max)))
    left = (int(rng.integers(r_min + 1, r_max)), c_min)
    right = (int(rng.integers(r_min + 1, r_max)), c_max)
    for r, c in (top, bottom, left, right):
        grid[r, c] = RED


def _make_input(
    rng: np.random.Generator,
    orientation: str,
    inner_h: int,
    inner_w: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> np.ndarray:
    grid, bounds = _build_canvas(inner_h, inner_w, top_margin, bottom_margin, left_margin, right_margin)
    _place_markers(grid, bounds, rng)

    inner_r0 = bounds["r_min"] + 2
    inner_c0 = bounds["c_min"] + 2
    smaller_is_blue = bool(rng.integers(0, 2))

    if orientation == "horizontal":
        small_seed_rel, large_seed_rel = _sample_horizontal_seeds(rng, inner_h, inner_w)
        smaller_seed = (inner_r0 + small_seed_rel[0], inner_c0 + small_seed_rel[1])
        larger_seed = (inner_r0 + large_seed_rel[0], inner_c0 + large_seed_rel[1])
    else:
        (small_seed_rel, large_seed_rel), _meta = _sample_vertical_seeds(rng, inner_h, inner_w)
        smaller_seed = (inner_r0 + small_seed_rel[0], inner_c0 + small_seed_rel[1])
        larger_seed = (inner_r0 + large_seed_rel[0], inner_c0 + large_seed_rel[1])

    if smaller_is_blue:
        blue_seed, green_seed = smaller_seed, larger_seed
    else:
        green_seed, blue_seed = smaller_seed, larger_seed

    grid[blue_seed] = BLUE
    grid[green_seed] = GREEN
    return grid


def _valid_sample(grid: np.ndarray) -> bool:
    try:
        expected = _oracle_transform(grid)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False

    try:
        parsed = _parse_grid(grid)
        spec = _stripe_spec(parsed)
    except Exception:
        return False

    if int(spec["phase"]) == 0:
        return False
    if np.array_equal(expected, _phase_zero_shortcut(grid)):
        return False

    blue = tuple(map(int, parsed["blue"]))  # type: ignore[arg-type]
    green = tuple(map(int, parsed["green"]))  # type: ignore[arg-type]
    row_gap = abs(blue[0] - green[0])
    col_gap = abs(blue[1] - green[1])
    if spec["orientation"] == "horizontal" and not (row_gap > col_gap):
        return False
    if spec["orientation"] == "vertical" and not (row_gap <= col_gap):
        return False

    return True


def _attempt_generate() -> np.ndarray:
    rng = np.random.default_rng()
    orientation = random.choice(["horizontal", "vertical"])
    inner_h = int(rng.integers(INNER_MIN + 1, INNER_MAX + 1))
    inner_w = int(rng.integers(INNER_MIN + 1, INNER_MAX + 1))
    top_margin = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    bottom_margin = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    left_margin = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    right_margin = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    return _make_input(
        rng,
        orientation,
        inner_h,
        inner_w,
        top_margin,
        bottom_margin,
        left_margin,
        right_margin,
    )


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if _valid_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
