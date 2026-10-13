import random

import numpy as np


BG = 0
WALL = 5
MARKER = 7
PALETTE = [1, 2, 3, 4, 6, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _legend(grid: np.ndarray):
    top_colors = [int(v) for v in grid[0] if v not in (BG, WALL, MARKER)]
    left_colors = [int(v) for v in grid[:, 0] if v not in (BG, WALL, MARKER)]
    if len(top_colors) >= len(left_colors) and len(top_colors) >= 2:
        return top_colors, {(0, c) for c, v in enumerate(grid[0]) if v in top_colors}
    return left_colors, {(r, 0) for r, v in enumerate(grid[:, 0]) if v in left_colors}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the legend color indexed by its number of orange markers.

    Input structure:
    - The background is black and a one-dimensional color legend lies on the top row or the left column.
    - Grey walls partition the remaining area into several enclosed regions.
    - Each enclosed region contains some orange marker pixels on black background.

    Rule:
    - Count the orange marker pixels inside each enclosed region.
    - Use that count as a one-based index into the legend strip.
    - Recolor the entire region interior, including the marker pixels, with the selected legend color while keeping the walls and legend unchanged.

    Key insight:
    - The orange pixels do not supply the fill color directly; they encode an index into a visible lookup strip.
    """
    g = np.asarray(grid, dtype=int)
    legend, legend_cells = _legend(g)
    fillable = (g == BG) | (g == MARKER)
    for r, c in legend_cells:
        fillable[r, c] = False

    out = g.copy()
    for comp in _components(fillable):
        if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in comp):
            continue
        count = sum(1 for r, c in comp if g[r, c] == MARKER)
        color = legend[count - 1]
        for r, c in comp:
            out[r, c] = color
    return out


def _room(grid: np.ndarray, top: int, left: int, h: int, w: int):
    grid[top, left : left + w] = WALL
    grid[top + h - 1, left : left + w] = WALL
    grid[top : top + h, left] = WALL
    grid[top : top + h, left + w - 1] = WALL


def _gen_attempt(rng: random.Random):
    legend_len = rng.randint(3, 6)
    legend = rng.sample(PALETTE, legend_len)
    top_or_left = rng.choice(["top", "left"])
    h = rng.randint(16, 24)
    w = rng.randint(16, 24)
    grid = np.full((h, w), BG, dtype=int)

    if top_or_left == "top":
        if legend_len > w:
            return None
        grid[0, :legend_len] = legend
        row_offset, col_offset = 2, 0
    else:
        if legend_len > h:
            return None
        grid[:legend_len, 0] = legend
        row_offset, col_offset = 0, 2

    room_count = rng.randint(2, 4)
    rooms = []
    attempts = 0
    while len(rooms) < room_count and attempts < 100:
        attempts += 1
        rh = rng.randint(4, 7)
        rw = rng.randint(4, 7)
        top = rng.randint(row_offset, h - rh - 1)
        left = rng.randint(col_offset, w - rw - 1)
        if any(not (top + rh + 1 <= t or t + hh + 1 <= top or left + rw + 1 <= l or l + ww + 1 <= left) for t, l, hh, ww in rooms):
            continue
        rooms.append((top, left, rh, rw))

    if len(rooms) != room_count:
        return None

    for top, left, rh, rw in rooms:
        _room(grid, top, left, rh, rw)
        cells = [(r, c) for r in range(top + 1, top + rh - 1) for c in range(left + 1, left + rw - 1)]
        count = rng.randint(1, min(legend_len, len(cells)))
        rng.shuffle(cells)
        for r, c in cells[:count]:
            grid[r, c] = MARKER
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
