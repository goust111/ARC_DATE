import random
from collections import deque

import numpy as np


BG = 0
FRAME = 1
STRIPE = 7
DIR_COLORS = {"up": 2, "down": 3, "left": 4, "right": 6}


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _legend(grid: np.ndarray):
    mapping = {}
    for r in range(grid.shape[0]):
        if grid[r, 0] in DIR_COLORS.values():
            seq = [int(v) for v in grid[r, 1:] if v != BG]
            direction = next(k for k, v in DIR_COLORS.items() if v == grid[r, 0])
            mapping[direction] = seq
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each frame's interior stripes using the legend row indexed by the side that contains the border gap.

    Input structure:
    - The grid has a black background, several single-color rectangular frames, and a legend made of edge rows.
    - Each frame has exactly one one-pixel gap on one side.
    - Inside each frame are parallel grey stripes, horizontal for top/bottom gaps or vertical for left/right gaps.
    - Each legend row starts with a direction marker color and then a cyclic sequence of fill colors.

    Rule:
    - Detect the gap side of each frame.
    - Read the matching legend color sequence for that direction.
    - Order the grey stripes from the side opposite the gap toward the gap side.
    - Recolor the stripes cyclically along that order, while keeping the frame and the gap unchanged.

    Key insight:
    - The gap side is both a visual cue and a legend key, while the stripe ordering starts from the opposite side rather than from the gap itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend = _legend(g)
    for comp in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        top_missing = [c for c in range(c0 + 1, c1) if g[r0, c] != FRAME]
        bot_missing = [c for c in range(c0 + 1, c1) if g[r1, c] != FRAME]
        left_missing = [r for r in range(r0 + 1, r1) if g[r, c0] != FRAME]
        right_missing = [r for r in range(r0 + 1, r1) if g[r, c1] != FRAME]
        if top_missing:
            direction = "up"
        elif bot_missing:
            direction = "down"
        elif left_missing:
            direction = "left"
        else:
            direction = "right"
        stripe_cells = []
        for stripe in _components(g[r0 + 1 : r1, c0 + 1 : c1] == STRIPE):
            stripe = [(r + r0 + 1, c + c0 + 1) for r, c in stripe]
            stripe_cells.append(stripe)
        if direction in ("up", "down"):
            stripe_cells.sort(key=lambda cells: min(r for r, _ in cells), reverse=(direction == "up"))
        else:
            stripe_cells.sort(key=lambda cells: min(c for _, c in cells), reverse=(direction == "left"))
        seq = legend[direction]
        for i, cells in enumerate(stripe_cells):
            color = seq[i % len(seq)]
            for r, c in cells:
                out[r, c] = color
    return out


def _draw_frame(grid, top, left, height, width, direction):
    grid[top, left : left + width] = FRAME
    grid[top + height - 1, left : left + width] = FRAME
    grid[top : top + height, left] = FRAME
    grid[top : top + height, left + width - 1] = FRAME
    if direction == "up":
        grid[top, left + width // 2] = BG
        for r in range(top + 2, top + height - 1, 2):
            grid[r, left + 1 : left + width - 1] = STRIPE
    elif direction == "down":
        grid[top + height - 1, left + width // 2] = BG
        for r in range(top + 1, top + height - 2, 2):
            grid[r, left + 1 : left + width - 1] = STRIPE
    elif direction == "left":
        grid[top + height // 2, left] = BG
        for c in range(left + 2, left + width - 1, 2):
            grid[top + 1 : top + height - 1, c] = STRIPE
    else:
        grid[top + height // 2, left + width - 1] = BG
        for c in range(left + 1, left + width - 2, 2):
            grid[top + 1 : top + height - 1, c] = STRIPE


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(20, 28), random.randint(20, 28)
        grid = np.full((h, w), BG, dtype=int)
        for i, direction in enumerate(["up", "down", "left", "right"]):
            seq = random.sample([1, 3, 4, 6, 8, 9], random.randint(2, 4))
            grid[h - 4 + i, 0] = DIR_COLORS[direction]
            grid[h - 4 + i, 1 : 1 + len(seq)] = seq
        occ = np.zeros((h, w), dtype=bool)
        occ[h - 5 :, :] = True
        ok = True
        for _ in range(random.randint(2, 4)):
            direction = random.choice(list(DIR_COLORS))
            placed = False
            for _ in range(100):
                hh, ww = random.randint(6, 9), random.randint(6, 9)
                top = random.randint(1, h - hh - 6)
                left = random.randint(2, w - ww - 2)
                if occ[max(0, top - 1) : min(h, top + hh + 1), max(0, left - 1) : min(w, left + ww + 1)].any():
                    continue
                _draw_frame(grid, top, left, hh, ww, direction)
                occ[max(0, top - 1) : min(h, top + hh + 1), max(0, left - 1) : min(w, left + ww + 1)] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
