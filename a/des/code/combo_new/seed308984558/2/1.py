import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
DIRS = {"up": 0, "right": 1, "down": 2, "left": 3}


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
    Recolor each gapped frame interior from the bottom legend column chosen by gap direction.

    Input structure:
    - A black background contains several one-pixel-thick grey rectangular frames with one missing border pixel.
    - Each frame encloses a solid fill color.
    - The bottom two rows contain four vertical legend pairs ordered as up, right, down, left.

    Rule:
    - Detect which side of each frame contains the single gap.
    - Use that direction to select the legend column and replace the frame's fill color by the legend's bottom color if the top color matches.
    - Leave the frame border, background, and legend unchanged.

    Key insight:
    - The frame gap acts as a lookup-table selector, and the interior color is remapped only after choosing the correct legend column.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    cols = [c for c in range(g.shape[1]) if g[-2, c] != BG and g[-1, c] != BG]
    cols.sort()
    mapping = {i: (int(g[-2, c]), int(g[-1, c])) for i, c in enumerate(cols[:4])}
    for cells in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(cells)
        missing = []
        for c in range(c0, c1 + 1):
            if g[r0, c] != FRAME:
                missing.append("up")
            if g[r1, c] != FRAME:
                missing.append("down")
        for r in range(r0, r1 + 1):
            if g[r, c0] != FRAME:
                missing.append("left")
            if g[r, c1] != FRAME:
                missing.append("right")
        if len(missing) != 1:
            continue
        direction = missing[0]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        fill = int([v for v in np.unique(inner) if v != BG][0])
        src, dst = mapping[DIRS[direction]]
        if fill == src:
            out[r0 + 1 : r1, c0 + 1 : c1][inner == fill] = dst
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(16, 24), random.randint(16, 24)
    g = np.zeros((h, w), dtype=int)
    top_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    bot_colors = random.sample([v for v in [1, 2, 3, 4, 6, 7, 8, 9] if v not in top_colors + [FRAME]], 4)
    positions = sorted(random.sample(range(1, w - 1), 4))
    for i, c in enumerate(positions):
        g[-2, c] = top_colors[i]
        g[-1, c] = bot_colors[i]
    for _ in range(random.randint(2, 4)):
        direction = random.choice(list(DIRS))
        fill = top_colors[DIRS[direction]]
        fh, fw = random.randint(4, 7), random.randint(4, 7)
        for _ in range(200):
            r0 = random.randint(0, h - fh - 3)
            c0 = random.randint(0, w - fw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)]):
                continue
            g[r0 : r0 + fh, c0] = g[r0 : r0 + fh, c0 + fw - 1] = FRAME
            g[r0, c0 : c0 + fw] = g[r0 + fh - 1, c0 : c0 + fw] = FRAME
            gap = {
                "up": (r0, random.randint(c0 + 1, c0 + fw - 2)),
                "down": (r0 + fh - 1, random.randint(c0 + 1, c0 + fw - 2)),
                "left": (random.randint(r0 + 1, r0 + fh - 2), c0),
                "right": (random.randint(r0 + 1, r0 + fh - 2), c0 + fw - 1),
            }[direction]
            g[gap] = BG
            g[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = fill
            break
    return g
