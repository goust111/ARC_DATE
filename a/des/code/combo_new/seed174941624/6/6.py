import random
from typing import Sequence

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

MAIN_PALETTE = (1, 4, 6, 7, 8, 9)

REGION_COUNT_RANGE = (2, 5)
REGION_WIDTH_RANGE = (3, 6)
GRID_HEIGHT_RANGE = (8, 20)
MAX_GENERATION_ATTEMPTS = 300


def _region_intervals(width: int, sep_cols: Sequence[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for sep in sep_cols:
        assert 0 < sep < width - 1, "separator cannot touch the side border"
        intervals.append((start, sep - 1))
        start = sep + 1
    intervals.append((start, width - 1))
    return intervals


def _parse_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 4 and width >= 3, "grid is too small"

    top = g[0]
    sep_cols = np.where(top == GREY)[0].tolist()
    intervals = _region_intervals(width, sep_cols)
    region_widths = {c1 - c0 + 1 for c0, c1 in intervals}
    assert len(region_widths) == 1, "column regions must have equal width"

    if sep_cols:
        for c in sep_cols:
            assert np.all(g[2:, c] == GREY), "vertical separators must align with top-row grey pixels"

    grey_mask = g == GREY
    allowed_grey = np.zeros_like(grey_mask, dtype=bool)
    allowed_grey[0, sep_cols] = True
    if sep_cols:
        allowed_grey[2:, sep_cols] = True
    assert not np.any(grey_mask & ~allowed_grey), "grey is reserved for separators"

    signal_cols: list[int] = []
    signals: list[int] = []
    for c0, c1 in intervals:
        region_signals = [c for c in range(c0, c1 + 1) if top[c] in (RED, GREEN)]
        assert len(region_signals) == 1, "each column region must contain exactly one signal light"
        signal_col = region_signals[0]
        signal_cols.append(signal_col)
        signals.append(int(top[signal_col]))

        for c in range(c0, c1 + 1):
            if c != signal_col:
                assert top[c] == BLACK, "top row may only contain lights, separators, and black background"

    if height > 1:
        assert np.all(g[1] == BLACK), "the spacer row under the lights must be black"

    return intervals, signal_cols, signals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-row traffic lights to decide which column regions keep their foreground patterns.

    Input structure:
    - The background is black.
    - The top row contains one red or green signal light inside each column region, and grey separator pixels between neighboring regions.
    - Starting at row 3, the main area is split into equal-width column regions by grey vertical separator lines aligned with the top-row grey pixels.
    - Each column region contains arbitrary non-grey foreground pixels on black background.

    Rule:
    - Read each region's signal light from the top row.
    - If the signal is green, keep every pixel in the corresponding column region below the spacer row.
    - If the signal is red, erase that entire column region to black.
    - The full top row and all grey separator lines are erased to black in the output, while the grid size stays unchanged.

    Key insight:
    - The grey columns establish the alignment between the one-pixel traffic lights and the wider regions below, so solving requires matching each signal to its column block before applying the keep-versus-erase meaning of green and red.
    """
    g = np.asarray(grid, dtype=int)
    intervals, _signal_cols, signals = _parse_layout(g)

    out = np.full_like(g, BLACK)
    for (c0, c1), signal in zip(intervals, signals):
        if signal == GREEN:
            out[2:, c0 : c1 + 1] = g[2:, c0 : c1 + 1]
        else:
            assert signal == RED, "signals must be red or green"

    return out


def _paint_rectangle(region: np.ndarray, occupied: np.ndarray, color: int, rng: random.Random) -> bool:
    height, width = region.shape
    rect_h = rng.randint(1, min(5, height))
    rect_w = rng.randint(1, min(3, width))
    r0 = rng.randint(0, height - rect_h)
    c0 = rng.randint(0, width - rect_w)
    sl = np.s_[r0 : r0 + rect_h, c0 : c0 + rect_w]
    if occupied[sl].any():
        return False
    region[sl] = color
    occupied[sl] = True
    return True


def _paint_hbar(region: np.ndarray, occupied: np.ndarray, color: int, rng: random.Random) -> bool:
    height, width = region.shape
    length = rng.randint(2, width)
    row = rng.randint(0, height - 1)
    c0 = rng.randint(0, width - length)
    sl = np.s_[row : row + 1, c0 : c0 + length]
    if occupied[sl].any():
        return False
    region[sl] = color
    occupied[sl] = True
    return True


def _paint_vbar(region: np.ndarray, occupied: np.ndarray, color: int, rng: random.Random) -> bool:
    height, width = region.shape
    length = rng.randint(2, min(6, height))
    col = rng.randint(0, width - 1)
    r0 = rng.randint(0, height - length)
    sl = np.s_[r0 : r0 + length, col : col + 1]
    if occupied[sl].any():
        return False
    region[sl] = color
    occupied[sl] = True
    return True


def _paint_diagonal(region: np.ndarray, occupied: np.ndarray, color: int, rng: random.Random) -> bool:
    height, width = region.shape
    length = rng.randint(2, min(5, height, width))
    r0 = rng.randint(0, height - length)
    c0 = rng.randint(0, width - length)
    coords = []
    direction = rng.choice((-1, 1))
    for i in range(length):
        rr = r0 + i
        cc = c0 + i if direction == 1 else c0 + (length - 1 - i)
        coords.append((rr, cc))
    if any(occupied[r, c] for r, c in coords):
        return False
    for r, c in coords:
        region[r, c] = color
        occupied[r, c] = True
    return True


def _paint_cluster(region: np.ndarray, occupied: np.ndarray, color: int, rng: random.Random) -> bool:
    shapes = (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2)),
    )
    shape = rng.choice(shapes)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    height, width = region.shape
    if max_r >= height or max_c >= width:
        return False
    r0 = rng.randint(0, height - max_r - 1)
    c0 = rng.randint(0, width - max_c - 1)
    coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
    if any(occupied[r, c] for r, c in coords):
        return False
    for r, c in coords:
        region[r, c] = color
        occupied[r, c] = True
    return True


def _make_region_pattern(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    region = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    painters = (
        _paint_rectangle,
        _paint_hbar,
        _paint_vbar,
        _paint_diagonal,
        _paint_cluster,
    )

    primitive_count = rng.randint(2, 4)
    used_colors: list[int] = []
    for _ in range(primitive_count):
        color = rng.choice(MAIN_PALETTE)
        if len(used_colors) < 3 and color not in used_colors and rng.random() < 0.7:
            used_colors.append(color)
        elif used_colors:
            color = rng.choice(used_colors)
        else:
            used_colors.append(color)

        success = False
        for _ in range(20):
            if rng.choice(painters)(region, occupied, color, rng):
                success = True
                break
        if not success:
            break

    if not occupied.any():
        return None

    # Ensure every region contains a clearly visible pattern with at least two cells.
    if int(occupied.sum()) == 1:
        return None
    return region


def _choose_dimensions(rng: random.Random) -> tuple[int, int, int]:
    region_count = rng.randint(*REGION_COUNT_RANGE)
    valid_widths = [
        region_width
        for region_width in range(REGION_WIDTH_RANGE[0], REGION_WIDTH_RANGE[1] + 1)
        if region_count * region_width + (region_count - 1) <= 30
    ]
    region_width = rng.choice(valid_widths)
    total_width = region_count * region_width + (region_count - 1)
    grid_height = rng.randint(*GRID_HEIGHT_RANGE)
    return region_count, region_width, min(grid_height, 30)


def _choose_signals(region_count: int, rng: random.Random) -> list[int]:
    green_count = rng.randint(1, region_count - 1)
    signals = [GREEN] * green_count + [RED] * (region_count - green_count)
    rng.shuffle(signals)
    return signals


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count, region_width, height = _choose_dimensions(rng)
    width = region_count * region_width + (region_count - 1)
    main_height = height - 2
    if main_height < 3:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    signals = _choose_signals(region_count, rng)

    sep_cols = [region_width + i * (region_width + 1) for i in range(region_count - 1)]
    for c in sep_cols:
        grid[0, c] = GREY
        grid[2:, c] = GREY

    intervals = _region_intervals(width, sep_cols)
    for (c0, c1), signal in zip(intervals, signals):
        signal_col = rng.randint(c0, c1)
        grid[0, signal_col] = signal

        pattern = _make_region_pattern(main_height, region_width, rng)
        if pattern is None:
            return None
        grid[2:, c0 : c1 + 1] = pattern

    try:
        out = main(grid)
    except AssertionError:
        return None

    # Reject uninformative cases: both branches must visibly matter.
    changed_mask = out != grid
    if not changed_mask.any():
        return None
    if not any(signal == RED for signal in signals):
        return None
    if not any(signal == GREEN for signal in signals):
        return None

    red_pixels_erased = 0
    green_pixels_kept = 0
    for (c0, c1), signal in zip(intervals, signals):
        region_pixels = int(np.count_nonzero(grid[2:, c0 : c1 + 1]))
        if signal == RED:
            red_pixels_erased += region_pixels
        else:
            green_pixels_kept += region_pixels
    if red_pixels_erased == 0 or green_pixels_kept == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
