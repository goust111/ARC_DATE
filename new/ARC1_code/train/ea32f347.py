from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg
import random


def _bar_length(obj: dict) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    h = bbox_h((r0, c0, r1, c1))
    w = bbox_w((r0, c0, r1, c1))
    area = int(obj["area"])
    assert h == 1 or w == 1, "each gray object must be a straight horizontal or vertical bar"
    assert area == max(h, w), "each bar must completely fill its one-cell-thick bbox"
    return area


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the three gray bars by length rank: longest blue, middle yellow, shortest red.

    Input structure
    - The background is the dominant color, and the only foreground color is gray.
    - The gray pixels form exactly three 4-connected straight one-cell-thick bars.
    - The three bars have distinct lengths.

    Rule
    - Preserve the background pixels.
    - Recolor the longest bar blue, the middle-length bar yellow, and the shortest bar red.

    Key insight
    - The bar colors are determined by relative length rank, independent of bar
      position or orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    fg_colors = set(map(int, np.unique(g))) - {bg}
    assert fg_colors == {Color.GREY}, "expected gray bars on a background"

    bars = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[Color.GREY])
    assert len(bars) == 3, "expected exactly three gray bars"

    lengths = [_bar_length(bar) for bar in bars]
    assert len(set(lengths)) == 3, "bar lengths must be distinct"

    out = g.copy()
    for bar, color in zip(sorted(bars, key=_bar_length, reverse=True), (Color.BLUE, Color.YELLOW, Color.RED)):
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out

# --------------- generator (o5x) ---------------
# CFG: canvas and colors.
GEN_SHAPE = (10, 10)
BACKGROUND = 0
BAR_COLOR = Color.GREY
RANK_COLORS = (Color.BLUE, Color.YELLOW, Color.RED)

# CFG: bar sampling, placement, and retry limit.
BAR_COUNT = 3
BAR_LENGTHS = range(2, 8)
ORIENTATIONS = ("H", "V")
MAX_GENERATION_ATTEMPTS = 300
NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_bar_cells(r: int, c: int, length: int, orient: str) -> list[tuple[int, int]]:
    if orient == "H":
        return [(r, c + dc) for dc in range(length)]
    return [(r + dr, c) for dr in range(length)]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    grid[tuple(zip(*cells))] = color


def _gen_can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    rows, cols = grid.shape
    for r, c in cells:
        if grid[r, c] != BACKGROUND:
            return False
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != BACKGROUND:
                return False
    return True


def _gen_valid_starts(grid: np.ndarray, length: int, orient: str) -> list[tuple[int, int]]:
    rows, cols = grid.shape
    max_r = rows if orient == "H" else rows - length + 1
    max_c = cols - length + 1 if orient == "H" else cols
    starts = []
    for r in range(max_r):
        for c in range(max_c):
            cells = _gen_bar_cells(r, c, length, orient)
            if _gen_can_place(grid, cells):
                starts.append((r, c))
    return starts


def _gen_attempt() -> np.ndarray | None:
    inp = np.zeros(GEN_SHAPE, dtype=int)
    out = inp.copy()

    lengths = random.sample(BAR_LENGTHS, BAR_COUNT)
    random.shuffle(lengths)
    bars: list[tuple[int, list[tuple[int, int]]]] = []

    for length in lengths:
        orient = random.choice(ORIENTATIONS)
        starts = _gen_valid_starts(inp, length, orient)
        if not starts:
            return None
        r, c = random.choice(starts)
        cells = _gen_bar_cells(r, c, length, orient)
        _paint_cells(inp, cells, BAR_COLOR)
        bars.append((length, cells))

    for rank, (_length, cells) in enumerate(sorted(bars, key=lambda item: item[0], reverse=True)):
        _paint_cells(out, cells, RANK_COLORS[rank])

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ea32f347 input")
