import random
from collections import deque

import numpy as np


BG = 0
WALL = 5


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
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each solid interior fill block out through the open side of its U-shaped grey container.

    Input structure:
    - Grey U-shaped rectangular containers sit on a black background.
    - Each container holds one solid colored interior block.
    - Exactly one side of the surrounding rectangle is missing.

    Rule:
    - Clear the fill from the container interior.
    - Translate the whole fill block outward through the opening until the next step would hit a non-background cell or leave the grid.
    - Leave the grey walls unchanged.

    Key insight:
    - The missing side of the rectangle is the pour direction; once detected, the whole packed fill slides as one block.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    for cells in _components(g == WALL):
        r0, c0, r1, c1 = _bbox(cells)
        top = np.all(g[r0, c0 : c1 + 1] == WALL)
        bot = np.all(g[r1, c0 : c1 + 1] == WALL)
        lef = np.all(g[r0 : r1 + 1, c0] == WALL)
        rig = np.all(g[r0 : r1 + 1, c1] == WALL)
        if sum([top, bot, lef, rig]) != 3:
            continue
        direction = "up" if not top else "down" if not bot else "left" if not lef else "right"
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = [v for v in np.unique(inner) if v != BG]
        if not colors:
            continue
        color = int(colors[0])
        pts = list(zip(*np.where(inner == color), strict=False))
        ir0, ic0 = min(r for r, _ in pts), min(c for _, c in pts)
        ir1, ic1 = max(r for r, _ in pts), max(c for _, c in pts)
        block = [(r0 + 1 + r, c0 + 1 + c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
        for r, c in block:
            out[r, c] = BG
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
        cur = block[:]
        while True:
            nxt = [(r + dr, c + dc) for r, c in cur]
            if not all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in nxt):
                break
            if any(g[r, c] != BG and (r, c) not in block for r, c in nxt):
                break
            cur = nxt
        for r, c in cur:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7], random.randint(2, 3)):
        fh, fw = random.randint(5, 8), random.randint(5, 8)
        for _ in range(200):
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            if np.any(g[r0 - 1 : r0 + fh + 1, c0 - 1 : c0 + fw + 1]):
                continue
            direction = random.choice(["up", "down", "left", "right"])
            g[r0 : r0 + fh, c0] = g[r0 : r0 + fh, c0 + fw - 1] = WALL
            g[r0, c0 : c0 + fw] = WALL
            g[r0 + fh - 1, c0 : c0 + fw] = WALL
            if direction == "up":
                g[r0, c0 : c0 + fw] = BG
            elif direction == "down":
                g[r0 + fh - 1, c0 : c0 + fw] = BG
            elif direction == "left":
                g[r0 : r0 + fh, c0] = BG
            else:
                g[r0 : r0 + fh, c0 + fw - 1] = BG
            g[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = color
            break
    return g
