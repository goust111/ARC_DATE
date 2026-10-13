from __future__ import annotations

import numpy as np

from arc_utils import colors_present, draw_line, get_bg
import random


def _full_width_bar(g: np.ndarray, bg: int) -> tuple[int, int]:
    bars: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        row_colors = set(map(int, g[r]))
        if len(row_colors) == 1:
            color = row_colors.pop()
            if color != bg:
                bars.append((r, color))

    assert len(bars) == 1, "expected exactly one full-width foreground bar"
    return bars[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each non-bar foreground pixel one row downward, and splash diagonal rays upward when it lands directly above the bar.

    Input structure
    - The background is the dominant color.
    - One foreground color forms a single full-width horizontal bar.
    - The only other foreground color appears as isolated moving pixels off the bar.

    Rule
    - Clear the moving pixels from their original locations and paint each one at the same column one row lower.
    - If a moved pixel lands immediately above the horizontal bar, also paint both upward diagonals from that landing pixel until they leave the grid.
    - The full-width bar and all background cells outside those shifted/splashed paths remain unchanged.

    Key insight
    - The bar is a trigger line: pixels always shift down one step, but only pixels that become adjacent to the bar emit the diagonal splash.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H > 1 and W > 0, "grid must have room for downward movement"

    bg = get_bg(g)
    bar_r, bar_color = _full_width_bar(g, bg)
    assert bar_r > 0, "bar must have a row above it for splash detection"

    moving_colors = [color for color in colors_present(g, exclude={bg, bar_color})]
    assert len(moving_colors) == 1, "expected exactly one moving foreground color"
    moving_color = moving_colors[0]

    source_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == moving_color)]
    assert source_coords, "expected at least one moving pixel"
    assert all(r != bar_r for r, _c in source_coords), "moving pixels cannot lie on the bar"

    out = g.copy()
    out[g == moving_color] = bg

    paint: set[tuple[int, int]] = set()
    for r, c in source_coords:
        moved_r = r + 1
        assert moved_r < H, "moving pixel cannot shift beyond the bottom edge"
        assert moved_r != bar_r, "moving pixel cannot shift into the bar"

        paint.add((moved_r, c))
        if moved_r == bar_r - 1:
            for dc in (-1, 1):
                rr = moved_r - 1
                cc = c + dc
                while 0 <= rr < H and 0 <= cc < W:
                    paint.add((rr, cc))
                    rr -= 1
                    cc += dc

    for r, c in paint:
        assert out[r, c] == bg, "painted cells must be background after clearing moving pixels"
        out[r, c] = moving_color

    return out

# --------------- generator (o5x) ---------------

# Colors.
BG_COLOR, BAR_COLOR, SOURCE_COLOR = 0, 2, 4

# Canvas and bar bounds.
MIN_HEIGHT, MAX_HEIGHT = 6, 12
MIN_WIDTH, MAX_WIDTH = 3, 8
MIN_BAR_ROW, MAX_BAR_ROW = 3, 9
BAR_BOTTOM_MARGIN = 2

# Source placement and event rates.
MIN_SOURCE_COUNT, MAX_SOURCE_COUNT = 1, 5
SPLASH_PROB, BELOW_BAR_PROB, MAX_RAY_LENGTH = 0.55, 0.25, 4

# Generation retry budget.
GENERATION_RETRIES = 300


def _gen_ray_lengths(bar_r: int, width: int, col: int) -> tuple[int, int]:
    left = min(bar_r - 1, col)
    right = min(bar_r - 1, width - 1 - col)
    return left, right


def _gen_is_isolated(coord: tuple[int, int], coords: list[tuple[int, int]]) -> bool:
    r, c = coord
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in coords)


def _gen_add_optional_splash(
    output: np.ndarray, bar_r: int
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Mutate output with one optional splash and return input sources plus painted output cells."""
    width = output.shape[1]
    splash_cols = [
        c
        for c in range(width)
        if max(_gen_ray_lengths(bar_r, width, c)) <= MAX_RAY_LENGTH
    ]
    use_splash = bool(splash_cols) and random.random() < SPLASH_PROB
    if not use_splash:
        return [], set()

    splash_col = random.choice(splash_cols)
    output[bar_r - 1, splash_col] = SOURCE_COLOR
    for dc in (-1, 1):
        draw_line(
            output,
            bar_r - 2,
            splash_col + dc,
            direction=(-1, dc),
            color=SOURCE_COLOR,
        )
    output_paint = {tuple(map(int, rc)) for rc in np.argwhere(output == SOURCE_COLOR)}
    return [(bar_r - 2, splash_col)], output_paint


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    max_bar_r = min(height - BAR_BOTTOM_MARGIN, MAX_BAR_ROW)
    if max_bar_r < MIN_BAR_ROW:
        return None

    bar_r = random.randint(MIN_BAR_ROW, max_bar_r)
    count = random.randint(MIN_SOURCE_COUNT, min(MAX_SOURCE_COUNT, width))

    output = np.full((height, width), BG_COLOR, dtype=int)
    output[bar_r, :] = BAR_COLOR
    source_coords, output_paint = _gen_add_optional_splash(output, bar_r)
    used_cols = {c for _r, c in source_coords}
    below_count = 0

    remaining = count - len(source_coords)
    available_cols = [c for c in range(width) if c not in used_cols]
    if remaining > len(available_cols):
        return None

    for col in random.sample(available_cols, remaining):
        above_rows = list(range(1, bar_r - 1))
        below_rows = list(range(bar_r + 2, height)) if below_count == 0 else []

        above_choices: list[tuple[int, int]] = []
        below_choices: list[tuple[int, int]] = []
        for out_r in above_rows + below_rows:
            src = (out_r - 1, col)
            shifted = (out_r, col)
            if shifted in output_paint or shifted in source_coords:
                continue
            if src in output_paint:
                continue
            if not _gen_is_isolated(src, source_coords):
                continue
            if out_r > bar_r:
                below_choices.append(shifted)
            else:
                above_choices.append(shifted)

        if below_choices and (not above_choices or random.random() < BELOW_BAR_PROB):
            shifted = random.choice(below_choices)
            below_count += 1
        elif above_choices:
            shifted = random.choice(above_choices)
        else:
            return None

        source_coords.append((shifted[0] - 1, shifted[1]))
        output[shifted] = SOURCE_COLOR
        output_paint.add(shifted)

    inp = np.full((height, width), BG_COLOR, dtype=int)
    inp[bar_r, :] = BAR_COLOR
    for r, c in source_coords:
        inp[r, c] = SOURCE_COLOR

    try:
        return inp if np.array_equal(main(inp), output) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 73c3b0d8 input")
