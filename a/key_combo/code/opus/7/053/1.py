import random
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
FLOOR = 5
BAR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (12, 19)
GRID_WIDTH_RANGE = (10, 16)
BAR_COUNT_RANGE = (4, 6)
BAR_WIDTH_RANGE = (2, 4)
GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Bar:
    color: int
    row: int
    left: int
    right: int

    @property
    def width(self) -> int:
        return self.right - self.left + 1


def _extract_bars(grid: np.ndarray) -> list[Bar]:
    bars: list[Bar] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BACKGROUND, FLOOR)):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        rows = {int(r) for r, _ in coords}
        cols = sorted(int(c) for _, c in coords)
        assert len(rows) == 1, "each colored object must be one row tall"
        row = rows.pop()
        left = cols[0]
        right = cols[-1]
        assert len(cols) == right - left + 1, "each colored object must be a contiguous bar"
        bars.append(Bar(color=color, row=row, left=left, right=right))
    assert 4 <= len(bars) <= 6, "expected 4-6 colored bars"
    return bars


def _landing_order(bars: list[Bar], floor_row: int, width: int) -> list[Bar]:
    order = sorted(bars, key=lambda bar: (-bar.row, bar.left))
    top_support = [floor_row] * width
    landed: list[Bar] = []
    for bar in order:
        landing_row = min(top_support[c] for c in range(bar.left, bar.right + 1)) - 1
        landed_bar = Bar(color=bar.color, row=landing_row, left=bar.left, right=bar.right)
        landed.append(landed_bar)
        for c in range(bar.left, bar.right + 1):
            top_support[c] = landing_row
    return landed


def _layered_output(colors: list[int]) -> np.ndarray:
    n = len(colors)
    side = 2 * n if n % 2 == 0 else 2 * n - 1
    out = np.empty((side, side), dtype=int)
    for layer, color in enumerate(colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate falling horizontal bars and recolor concentric square layers by landing rank.

    Input structure:
    - The background is black.
    - The entire bottom row is a grey floor.
    - Above the floor sit 4-6 monochrome horizontal bars, each one row tall, width 2-4, and in a distinct non-black/non-grey color.

    Rule:
    - Read every colored bar, then process them from lower initial row to higher initial row; ties break by leftmost column.
    - Each bar falls straight down until the next row would hit either the floor or the top of an already landed bar in one of its covered columns.
    - Record the bars in that landing sequence and build the smallest concentric square that shows one visible layer per rank: even counts give a 2x2 center, odd counts give a 1x1 center.
    - Color the outermost layer with the first landed bar's color, the next layer with the second color, and continue outside-in until the last landed color fills the center.

    Key insight:
    - Overlapping column spans let earlier bars become supports for later ones, so the rank-colored concentric output must be driven by an explicit gravity simulation rather than by reading colors alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert np.all(g[-1] == FLOOR), "bottom row must be the grey floor"

    bars = _extract_bars(g)
    landed = _landing_order(bars, floor_row=height - 1, width=width)
    return _layered_output([bar.color for bar in landed])


def _overlap_pairs(intervals: list[tuple[int, int]]) -> int:
    pairs = 0
    for i, (l0, r0) in enumerate(intervals):
        for l1, r1 in intervals[i + 1 :]:
            if max(l0, l1) <= min(r0, r1):
                pairs += 1
    return pairs


def _sample_intervals(width: int, bar_widths: list[int]) -> list[tuple[int, int]] | None:
    intervals: list[tuple[int, int]] = []
    for i, bar_width in enumerate(bar_widths):
        must_overlap = i < 3
        target_overlap = must_overlap or random.random() < 0.65
        placed = False
        for _ in range(80):
            if intervals and target_overlap:
                base_left, base_right = random.choice(intervals)
                left_min = max(0, base_left - bar_width + 1)
                left_max = min(width - bar_width, base_right)
                if left_min > left_max:
                    continue
                left = random.randint(left_min, left_max)
            else:
                left = random.randint(0, width - bar_width)
            interval = (left, left + bar_width - 1)
            trial = intervals + [interval]
            pairs = _overlap_pairs(trial)
            if i == 1 and pairs < 1:
                continue
            if i == 2 and pairs < 2:
                continue
            intervals = trial
            placed = True
            break
        if not placed:
            return None
    if _overlap_pairs(intervals) < 2:
        return None
    return intervals


def _build_input(height: int, width: int, bars: list[Bar]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[-1, :] = FLOOR
    for bar in bars:
        grid[bar.row, bar.left : bar.right + 1] = bar.color
    return grid


def _sample_bars(height: int, width: int, bar_count: int) -> list[Bar] | None:
    rows = sorted(random.sample(range(1, height - 2), bar_count), reverse=True)
    bar_widths = [random.randint(*BAR_WIDTH_RANGE) for _ in range(bar_count)]
    intervals = _sample_intervals(width, bar_widths)
    if intervals is None:
        return None

    colors = random.sample(BAR_COLORS, bar_count)
    bars = [
        Bar(color=colors[i], row=rows[i], left=intervals[i][0], right=intervals[i][1])
        for i in range(bar_count)
    ]

    landed = _landing_order(bars, floor_row=height - 1, width=width)
    if all(bar.row == height - 2 for bar in landed):
        return None

    stack_levels = {bar.row for bar in landed}
    if len(stack_levels) == 1:
        return None

    return bars


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bar_count = random.randint(*BAR_COUNT_RANGE)

    bars = _sample_bars(height, width, bar_count)
    if bars is None:
        return None

    grid = _build_input(height, width, bars)

    try:
        out = main(grid)
    except Exception:
        return None

    expected = _layered_output(
        [bar.color for bar in _landing_order(_extract_bars(grid), floor_row=height - 1, width=width)]
    )
    if not np.array_equal(out, expected):
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
