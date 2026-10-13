import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

SOURCE_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, AZURE, MAROON]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _inside(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _pipe_width(grid, r, c, dr, dc):
    if not _inside(grid.shape, r, c) or int(grid[r, c]) != GREY:
        return 0
    if dr:
        offsets = [(0, 0), (0, -1), (0, 1)]
    else:
        offsets = [(0, 0), (-1, 0), (1, 0)]
    return sum(
        _inside(grid.shape, r + rr, c + cc) and int(grid[r + rr, c + cc]) == GREY
        for rr, cc in offsets
    )


def _neighbors(grid, r, c):
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if _inside(grid.shape, rr, cc):
            yield dr, dc, rr, cc, int(grid[rr, cc])


def _trace_one(grid, source_pos):
    sr, sc = source_pos
    start_options = [
        (dr, dc, rr, cc)
        for dr, dc, rr, cc, color in _neighbors(grid, sr, sc)
        if color == GREY
    ]
    if not start_options:
        return None
    dr, dc, r, c = start_options[0]
    prev = (sr, sc)
    direction = (dr, dc)
    seen = set()

    for _ in range(grid.size * 4):
        state = (r, c, direction)
        if state in seen:
            return None
        seen.add(state)

        for ndr, ndc, nr, nc, color in _neighbors(grid, r, c):
            if (nr, nc) != prev and color == ORANGE:
                return nr, nc

        candidates = []
        for ndr, ndc, nr, nc, color in _neighbors(grid, r, c):
            if (nr, nc) == prev or color != GREY:
                continue
            width = _pipe_width(grid, nr, nc, ndr, ndc)
            straight = int((ndr, ndc) == direction)
            candidates.append((width, straight, -DIRS.index((ndr, ndc)), ndr, ndc, nr, nc))
        if not candidates:
            return None
        width, straight, _, ndr, ndc, nr, nc = max(candidates)
        prev = (r, c)
        r, c = nr, nc
        direction = (ndr, ndc)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    sources = []
    for r, c in zip(*np.where((grid != BLACK) & (grid != GREY) & (grid != ORANGE))):
        color = int(grid[r, c])
        if any(int(grid[rr, cc]) == GREY for _, _, rr, cc, _ in _neighbors(grid, r, c)):
            sources.append((r, c, color))

    for r, c, color in sources:
        receiver = _trace_one(grid, (r, c))
        if receiver is not None:
            out[receiver] = color
    return out


def _paint_h(grid, r, c0, c1):
    if c0 > c1:
        c0, c1 = c1, c0
    grid[r, c0 : c1 + 1] = GREY


def _paint_v(grid, c, r0, r1):
    if r0 > r1:
        r0, r1 = r1, r0
    grid[r0 : r1 + 1, c] = GREY


def _draw_network(grid, band_top, band_bottom, color):
    r = random.randint(band_top + 2, band_bottom - 3)
    j = random.randint(6, 9)
    right_end = random.randint(j + 5, grid.shape[1] - 4)
    down_len = min(random.randint(3, 5), band_bottom - r - 2)
    if down_len < 3:
        down_len = 3
        r = band_bottom - down_len - 2

    grid[r, 1] = color
    _paint_h(grid, r, 2, j)

    mode = random.choice(["straight_tie", "straight_wide", "down_wide"])
    _paint_h(grid, r, j + 1, right_end)
    _paint_v(grid, j, r + 1, r + down_len)

    if mode == "straight_wide":
        _paint_h(grid, r + 1, j + 1, right_end)
        grid[r, right_end + 1] = ORANGE
    elif mode == "down_wide":
        _paint_v(grid, j + 1, r + 1, r + down_len)
        grid[r + down_len + 1, j] = ORANGE
    else:
        grid[r, right_end + 1] = ORANGE


def generate_input() -> np.ndarray:
    networks = random.randint(1, 3)
    band_h = 8
    h = networks * band_h + 2
    w = random.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(SOURCE_COLORS, networks)
    for i, color in enumerate(colors):
        top = i * band_h
        bottom = top + band_h - 1
        _draw_network(grid, top, bottom, color)
    return grid


if __name__ == "__main__":
    for _ in range(100):
        sample = generate_input()
        result = main(sample)
        assert isinstance(sample, np.ndarray)
        assert isinstance(result, np.ndarray)
