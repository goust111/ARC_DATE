import random

import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _arrow_at(grid, r, c):
    if r < 1 or c < 1 or r + 2 >= grid.shape[0] or c + 2 >= grid.shape[1]:
        return None
    if not np.all(grid[r:r + 2, c:c + 2] == GREY):
        return None
    checks = [
        ("up", (r - 1, c)),
        ("down", (r + 2, c)),
        ("left", (r, c - 1)),
        ("right", (r, c + 2)),
    ]
    hits = [(name, cell) for name, cell in checks if grid[cell] == GREY]
    if len(hits) != 1:
        return None
    name, cell = hits[0]
    return {"block": {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}, "indicator": cell, "dir": DIRS[name], "anchor": (r, c)}


def _arrows(grid):
    found = []
    seen = set()
    for r in range(1, grid.shape[0] - 2):
        for c in range(1, grid.shape[1] - 2):
            arr = _arrow_at(grid, r, c)
            if arr and not (arr["block"] & seen):
                seen |= arr["block"]
                found.append(arr)
    return found


def _source_blocks(grid):
    blocks = []
    for color in set(int(v) for v in grid.ravel()) - {BG, GREY}:
        for r in range(grid.shape[0] - 1):
            for c in range(grid.shape[1] - 1):
                if np.all(grid[r:r + 2, c:c + 2] == color):
                    blocks.append({"color": color, "anchor": (r, c)})
    blocks.sort(key=lambda b: (b["anchor"][1], b["anchor"][0]))
    return blocks


def _nearest_arrow_from_source(src, arrows):
    sr, sc = src["anchor"]
    candidates = []
    for i, arr in enumerate(arrows):
        ar, ac = arr["anchor"]
        if ar == sr or ac == sc:
            candidates.append((abs(ar - sr) + abs(ac - sc), i))
    return min(candidates)[1] if candidates else None


def _next_arrow(idx, arrows):
    arr = arrows[idx]
    ar, ac = arr["anchor"]
    dr, dc = arr["dir"]
    candidates = []
    for j, other in enumerate(arrows):
        if j == idx:
            continue
        br, bc = other["anchor"]
        if dr and bc == ac and (br - ar) * dr > 0:
            candidates.append((abs(br - ar), j))
        if dc and br == ar and (bc - ac) * dc > 0:
            candidates.append((abs(bc - ac), j))
    return min(candidates)[1] if candidates else None


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate source block colors through chains of grey arrows.

    Input structure:
    Grey arrows consist of a 2x2 grey block plus one adjacent grey direction pixel.
    Non-grey 2x2 source blocks sit in the same row or column as their first arrow.

    Rule:
    Sources are processed left to right, then top to bottom.  Each source color colors
    the 2x2 block of the nearest aligned arrow, then follows each arrow's direction to
    the next aligned arrow.  Later sources overwrite earlier arrivals.  Arrow direction
    pixels and source blocks remain unchanged.

    Key insight:
    Local arrow orientation defines a global directed chain, and conflict resolution is
    simply the processing order of the source blocks.
    """
    out = np.array(grid, copy=True)
    arrows = _arrows(grid)
    for src in _source_blocks(grid):
        idx = _nearest_arrow_from_source(src, arrows)
        seen = set()
        while idx is not None and idx not in seen:
            seen.add(idx)
            for r, c in arrows[idx]["block"]:
                out[r, c] = src["color"]
            idx = _next_arrow(idx, arrows)
    return out


def _draw_arrow(grid, r, c, direction):
    ind = {"up": (r - 1, c), "down": (r + 2, c), "left": (r, c - 1), "right": (r, c + 2)}[direction]
    cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1), ind]
    if any(grid[rr, cc] != BG for rr, cc in cells):
        return False
    for rr, cc in cells:
        grid[rr, cc] = GREY
    return True


def _draw_source(grid, r, c, color):
    if np.any(grid[r:r + 2, c:c + 2] != BG):
        return False
    grid[r:r + 2, c:c + 2] = color
    return True


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.zeros((h, w), dtype=int)
        row = random.randint(5, h - 6)
        cols = sorted(random.sample(range(5, w - 4), random.randint(3, 5)))
        ok = True
        for i, c in enumerate(cols):
            direction = "right" if i < len(cols) - 1 else random.choice(["up", "down"])
            if not _draw_arrow(grid, row, c, direction):
                ok = False
                break
        if not ok:
            continue
        src_colors = random.sample(COLORS, random.randint(2, 3))
        for i, color in enumerate(src_colors):
            c = max(1, cols[0] - 4 - i * 3)
            if not _draw_source(grid, row, c, color):
                ok = False
                break
        if not ok:
            continue
        if len(cols) >= 4:
            col = cols[-1]
            extra_rows = [row + 5, row + 9] if row + 10 < h else [row - 5, row - 9]
            for rr in extra_rows:
                if 2 <= rr < h - 3:
                    _draw_arrow(grid, rr, col, "down" if rr > row else "up")
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid arrow-chain puzzle")
