import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


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


def _arrow(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    rc = {r: rows.count(r) for r in set(rows)}
    cc = {c: cols.count(c) for c in set(cols)}
    if rc[min(rc)] == 3:
        return "up", (max(rc), int(np.median(cols))), (min(rc), int(np.median(cols)))
    if rc[max(rc)] == 3:
        return "down", (min(rc), int(np.median(cols))), (max(rc), int(np.median(cols)))
    if cc[min(cc)] == 3:
        return "left", (int(np.median(rows)), max(cc)), (int(np.median(rows)), min(cc))
    return "right", (int(np.median(rows)), min(cc)), (int(np.median(rows)), max(cc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the target object named by each grey arrow and redraw it one cell beyond the arrow tip.

    Input structure:
    - Grey arrows and colored target objects share the same row band or column band.
    - The target object lies behind the arrow tail, opposite the arrow tip.
    - There is empty space beyond the tip for the rotated copy.

    Rule:
    - Use the arrow direction to choose a quarter-turn amount.
    - Find the nearest colored object behind the tail.
    - Rotate that object and place it one blank cell beyond the tip, keeping the original arrow and target unchanged.

    Key insight:
    - The arrow simultaneously identifies which object to use and how much to rotate it before re-placing it in front.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    objs = []
    for color in [v for v in np.unique(g) if v not in (BG, GREY)]:
        for cells in _components(g == int(color)):
            objs.append((int(color), cells))
    for cells in _components(g == GREY):
        if len(cells) != 5:
            continue
        d, tail, tip = _arrow(cells)
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[d]
        target = None
        best = None
        for color, ocells in objs:
            r0, c0, r1, c1 = _bbox(ocells)
            if d in ("up", "down") and not (c1 < min(c for _, c in cells) or c0 > max(c for _, c in cells)):
                dist = tail[0] - r1 if d == "up" else r0 - tail[0]
            elif d in ("left", "right") and not (r1 < min(r for r, _ in cells) or r0 > max(r for r, _ in cells)):
                dist = tail[1] - c1 if d == "left" else c0 - tail[1]
            else:
                continue
            if dist > 0 and (best is None or dist < best):
                best = dist
                target = (color, ocells)
        if target is None:
            continue
        color, ocells = target
        r0, c0, r1, c1 = _bbox(ocells)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in ocells:
            patch[r - r0, c - c0] = color
        patch = np.rot90(patch, {"right": 0, "down": 3, "left": 2, "up": 1}[d])
        nr, nc = tip[0] + dr * 2, tip[1] + dc * 2
        rr = nr if d in ("down", "up") and dr > 0 else nr - patch.shape[0] + 1 if d == "up" else nr - patch.shape[0] // 2
        cc = nc if d == "right" else nc - patch.shape[1] + 1 if d == "left" else nc - patch.shape[1] // 2
        rr = max(0, min(rr, g.shape[0] - patch.shape[0]))
        cc = max(0, min(cc, g.shape[1] - patch.shape[1]))
        out[rr : rr + patch.shape[0], cc : cc + patch.shape[1]][patch != 0] = color
    return out


def _arrow_mask(direction):
    base = np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int)
    return {"right": base, "down": np.rot90(base, 3), "left": np.rot90(base, 2), "up": np.rot90(base, 1)}[direction]


SHAPE = np.array([[1, 1], [1, 0]])


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6], random.randint(2, 3)):
        direction = random.choice(["up", "down", "left", "right"])
        mask = _arrow_mask(direction)
        for _ in range(200):
            r0 = random.randint(3, h - 6)
            c0 = random.randint(3, w - 6)
            if np.any(g[r0 - 3 : r0 + 6, c0 - 3 : c0 + 6]):
                continue
            for r in range(3):
                for c in range(3):
                    if mask[r, c]:
                        g[r0 + r, c0 + c] = GREY
            if direction == "right":
                g[r0 + 1 : r0 + 3, c0 - 3 : c0 - 1][SHAPE == 1] = color
            elif direction == "left":
                g[r0 + 1 : r0 + 3, c0 + 4 : c0 + 6][SHAPE == 1] = color
            elif direction == "up":
                g[r0 + 4 : r0 + 6, c0 + 1 : c0 + 3][SHAPE == 1] = color
            else:
                g[r0 - 3 : r0 - 1, c0 + 1 : c0 + 3][SHAPE == 1] = color
            break
    return g
