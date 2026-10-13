import random

import numpy as np


BLACK = 0
GREY = 5
STRIPE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SAMPLE_SIZE = 4
MAX_GENERATION_ATTEMPTS = 300


def _group_consecutive_pairs(indices: list[int]) -> list[int]:
    if not indices or len(indices) % 2 != 0:
        raise AssertionError("grey rows/cols must form consecutive pairs")

    tops: list[int] = []
    i = 0
    while i < len(indices):
        first = int(indices[i])
        second = int(indices[i + 1])
        if second != first + 1:
            raise AssertionError("grey rows/cols must have thickness 2")
        tops.append(first)
        i += 2
        if i < len(indices) and indices[i] <= second + 1:
            raise AssertionError("adjacent stripe pairs must be separated")
    return tops


def _stripe_color_from_rows(grid: np.ndarray, top: int) -> int:
    colors = {int(v) for v in grid[top : top + 2, :].reshape(-1) if int(v) not in (BLACK, GREY)}
    if len(colors) != 1:
        raise AssertionError(f"expected one horizontal stripe color on rows {top}:{top + 2}, got {colors}")
    return next(iter(colors))


def _stripe_color_from_cols(grid: np.ndarray, left: int) -> int:
    colors = {int(v) for v in grid[:, left : left + 2].reshape(-1) if int(v) not in (BLACK, GREY)}
    if len(colors) != 1:
        raise AssertionError(f"expected one vertical stripe color on cols {left}:{left + 2}, got {colors}")
    return next(iter(colors))


def _find_sample_phase(grid: np.ndarray, horizontal_colors: set[int], vertical_colors: set[int]) -> bool:
    h, w = grid.shape
    for r in range(h - SAMPLE_SIZE + 1):
        for c in range(w - SAMPLE_SIZE + 1):
            block = grid[r : r + SAMPLE_SIZE, c : c + SAMPLE_SIZE]
            a = int(block[0, 0])
            b = int(block[0, 2])
            if a == b or BLACK in (a, b) or GREY in (a, b):
                continue
            if not np.all(block[:2, :2] == a):
                continue
            if not np.all(block[:2, 2:] == b):
                continue
            if not np.all(block[2:, :2] == b):
                continue
            if not np.all(block[2:, 2:] == a):
                continue
            if a in horizontal_colors and b in vertical_colors:
                return True
            if a in vertical_colors and b in horizontal_colors:
                return False
    raise AssertionError("failed to locate the weave sample block")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve grey 2x2 crossings of horizontal and vertical color bands into a woven over-under pattern.

    Input structure:
    - Background is black.
    - Several horizontal bands and several vertical bands, each exactly 2 cells thick, use distinct non-grey colors.
    - Every horizontal band crosses every vertical band; each crossing is a 2x2 grey block.
    - A small top-left sample shows an alternating 2x2 checkerboard made from one horizontal-band color and one vertical-band color.

    Rule:
    - Identify the horizontal band row-pairs and vertical band column-pairs from the grey 2x2 crossing grid.
    - Recover each band's color from its non-grey, non-black cells.
    - Read the sample: if its top-left quadrant uses a horizontal-band color then the first crossing has the horizontal band on top; otherwise the vertical band is on top.
    - Fill every grey crossing block with the winning band's color, alternating by checkerboard parity across columns and flipping the starting winner on each lower horizontal band.

    Key insight:
    - The grey crossing lattice reveals the band geometry, while the sample's top-left color determines the global parity of the weave.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    grey_positions = np.argwhere(g == GREY)
    if len(grey_positions) == 0:
        raise AssertionError("expected at least one grey crossing")

    grey_rows = sorted({int(r) for r, _ in grey_positions})
    grey_cols = sorted({int(c) for _, c in grey_positions})
    row_tops = _group_consecutive_pairs(grey_rows)
    col_tops = _group_consecutive_pairs(grey_cols)

    horizontal_colors = [_stripe_color_from_rows(g, top) for top in row_tops]
    vertical_colors = [_stripe_color_from_cols(g, left) for left in col_tops]
    if len(set(horizontal_colors + vertical_colors)) != len(horizontal_colors) + len(vertical_colors):
        raise AssertionError("stripe colors must be unique across both directions")

    for top in row_tops:
        for left in col_tops:
            if not np.all(g[top : top + 2, left : left + 2] == GREY):
                raise AssertionError("grey crossings must form a full Cartesian product")

    horizontal_first = _find_sample_phase(g, set(horizontal_colors), set(vertical_colors))

    out = g.copy()
    for row_index, top in enumerate(row_tops):
        for col_index, left in enumerate(col_tops):
            even = (row_index + col_index) % 2 == 0
            horizontal_wins = horizontal_first if even else not horizontal_first
            color = horizontal_colors[row_index] if horizontal_wins else vertical_colors[col_index]
            out[top : top + 2, left : left + 2] = color

    return out


def _paired_positions(start: int, gaps: list[int]) -> list[int]:
    positions = [int(start)]
    current = int(start)
    for gap in gaps:
        current += 2 + int(gap)
        positions.append(current)
    return positions


def _paint_sample(grid: np.ndarray, top: int, left: int, top_left_color: int, other_color: int) -> None:
    grid[top : top + 2, left : left + 2] = top_left_color
    grid[top : top + 2, left + 2 : left + 4] = other_color
    grid[top + 2 : top + 4, left : left + 2] = other_color
    grid[top + 2 : top + 4, left + 2 : left + 4] = top_left_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    sample_top = rng.randint(0, 1)
    sample_left = rng.randint(0, 1)

    horizontal_count = rng.randint(2, 4)
    vertical_count = rng.randint(2, 4)
    if horizontal_count + vertical_count > len(STRIPE_PALETTE):
        return None

    row_gaps = [rng.randint(1, 3) for _ in range(horizontal_count - 1)]
    col_gaps = [rng.randint(1, 3) for _ in range(vertical_count - 1)]
    weave_height = 2 * horizontal_count + sum(row_gaps)
    weave_width = 2 * vertical_count + sum(col_gaps)

    min_weave_top = sample_top + SAMPLE_SIZE + 2
    min_weave_left = sample_left + SAMPLE_SIZE + 2
    if min_weave_top + weave_height > 30 or min_weave_left + weave_width > 30:
        return None

    weave_top = rng.randint(min_weave_top, 30 - weave_height)
    weave_left = rng.randint(min_weave_left, 30 - weave_width)
    height = rng.randint(weave_top + weave_height, 30)
    width = rng.randint(weave_left + weave_width, 30)

    grid = np.full((height, width), BLACK, dtype=int)
    row_tops = _paired_positions(weave_top, row_gaps)
    col_tops = _paired_positions(weave_left, col_gaps)

    colors = rng.sample(list(STRIPE_PALETTE), horizontal_count + vertical_count)
    horizontal_colors = colors[:horizontal_count]
    vertical_colors = colors[horizontal_count:]

    for top, color in zip(row_tops, horizontal_colors):
        grid[top : top + 2, weave_left : weave_left + weave_width] = color
    for left, color in zip(col_tops, vertical_colors):
        grid[weave_top : weave_top + weave_height, left : left + 2] = color
    for top in row_tops:
        for left in col_tops:
            grid[top : top + 2, left : left + 2] = GREY

    horizontal_first = bool(rng.getrandbits(1))
    sample_horizontal = rng.choice(horizontal_colors)
    sample_vertical = rng.choice(vertical_colors)
    top_left_color = sample_horizontal if horizontal_first else sample_vertical
    other_color = sample_vertical if horizontal_first else sample_horizontal
    _paint_sample(grid, sample_top, sample_left, top_left_color, other_color)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving input")
