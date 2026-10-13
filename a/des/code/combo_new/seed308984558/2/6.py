import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
CYCLE = {1: 3, 3: 2, 2: 4, 4: 1}


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


def _arrow_info(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    rc = {r: rows.count(r) for r in set(rows)}
    cc = {c: cols.count(c) for c in set(cols)}
    if rc[min(rc)] == 3:
        return "up", (max(rc), int(np.median(cols)))
    if rc[max(rc)] == 3:
        return "down", (min(rc), int(np.median(cols)))
    if cc[min(cc)] == 3:
        return "left", (int(np.median(rows)), max(cc))
    return "right", (int(np.median(rows)), min(cc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 2x2 color block along its grey arrow and cycle its color once per crossed prism line.

    Input structure:
    - A black background contains grey 5-pixel arrows, 2x2 colored blocks at their tails, and maroon full-span prism lines.
    - Prism lines are either full rows or full columns.
    - Each arrow and its block are isolated from the others.

    Rule:
    - Erase the arrow and move the adjacent 2x2 block in the arrow direction until the next step would hit a non-background cell or leave the grid.
    - Count how many maroon full-row or full-column prisms the block crosses while moving.
    - Advance the block color through the fixed cycle once per crossed prism and paint the block at its final location.

    Key insight:
    - The grey arrow determines the travel direction, but the final color depends on counting crossed prism lines along that path.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    pris_r = {r for r in range(g.shape[0]) if np.all(g[r] == MAROON)}
    pris_c = {c for c in range(g.shape[1]) if np.all(g[:, c] == MAROON)}
    for cells in _components(g == GREY):
        if len(cells) != 5:
            continue
        direction, tail = _arrow_info(cells)
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
        tr, tc = tail
        br, bc = tr - dr, tc - dc
        block = [(br + i, bc + j) for i in [0, 1] for j in [0, 1]]
        if not all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in block):
            continue
        color = int(g[block[0]])
        if color in (BG, GREY, MAROON):
            continue
        for r, c in cells + block:
            out[r, c] = BG if g[r, c] != MAROON else MAROON
        steps = 0
        cur = block[:]
        while True:
            nxt = [(r + dr, c + dc) for r, c in cur]
            if not all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in nxt):
                break
            if any(g[r, c] not in (BG, MAROON) and (r, c) not in block for r, c in nxt):
                break
            cur = nxt
            steps += 1
        cross = 0
        if dr:
            lo, hi = sorted([block[0][0], cur[0][0]])
            cross = sum(lo < r <= hi for r in pris_r)
        else:
            lo, hi = sorted([block[0][1], cur[0][1]])
            cross = sum(lo < c <= hi for c in pris_c)
        for _ in range(cross):
            color = CYCLE[color]
        for r, c in cur:
            out[r, c] = color
    return out


def _arrow_mask(direction):
    base = np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int)
    return {"right": base, "down": np.rot90(base, 3), "left": np.rot90(base, 2), "up": np.rot90(base, 1)}[direction]


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    g = np.zeros((h, w), dtype=int)
    for r in random.sample(range(4, h - 4), random.randint(1, 2)):
        g[r, :] = MAROON
    for c in random.sample(range(4, w - 4), random.randint(1, 2)):
        g[:, c] = MAROON
    for color in random.sample([1, 2, 3, 4], random.randint(2, 3)):
        direction = random.choice(["up", "down", "left", "right"])
        mask = _arrow_mask(direction)
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
        for _ in range(200):
            r0 = random.randint(2, h - 5)
            c0 = random.randint(2, w - 5)
            if np.any(~np.isin(g[r0 - 2 : r0 + 5, c0 - 2 : c0 + 5], [BG, MAROON])):
                continue
            tip = {"up": (r0, c0 + 1), "down": (r0 + 2, c0 + 1), "left": (r0 + 1, c0), "right": (r0 + 1, c0 + 2)}[direction]
            tail = (tip[0] - 2 * dr, tip[1] - 2 * dc)
            br, bc = tail[0] - dr, tail[1] - dc
            block = [(br + i, bc + j) for i in [0, 1] for j in [0, 1]]
            if not all(0 <= r < h and 0 <= c < w for r, c in block):
                continue
            if any(g[r, c] not in (BG, MAROON) for r, c in block):
                continue
            for r in range(3):
                for c in range(3):
                    if mask[r, c]:
                        g[r0 + r, c0 + c] = GREY
            for r, c in block:
                g[r, c] = color
            break
    return g
