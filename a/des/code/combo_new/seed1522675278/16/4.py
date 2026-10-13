import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


def _components(mask, eight=False):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if eight:
        neigh = neigh + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _frame_bboxes(grid):
    frames = []
    for cells in _components(grid == GREY):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        if set(cells) == expected:
            frames.append((r0, r1, c0, c1))
    return frames


def _arrow_dir(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    rel = {(r - r0, c - c0) for r, c in cells}
    if rel == {(0, 0), (1, 1), (2, 0)}:
        return (0, 1)
    if rel == {(0, 1), (1, 0), (2, 1)}:
        return (0, -1)
    if rel == {(0, 0), (1, 1), (0, 2)}:
        return (1, 0)
    return (-1, 0)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    frames = _frame_bboxes(grid)
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY):
            continue
        for cells in _components(grid == color, eight=True):
            if len(cells) != 3:
                continue
            dr, dc = _arrow_dir(cells)
            tip = max(cells, key=lambda p: p[0] * dr + p[1] * dc)
            rr, cc = tip[0] + dr, tip[1] + dc
            target = None
            while 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if grid[rr, cc] == GREY:
                    for bbox in frames:
                        r0, r1, c0, c1 = bbox
                        if rr in (r0, r1) and c0 <= cc <= c1 or cc in (c0, c1) and r0 <= rr <= r1:
                            target = bbox
                            break
                    break
                rr += dr
                cc += dc
            if target is None:
                continue
            r0, r1, c0, c1 = target
            if dr == 0:
                cols = range(c0 + 1, c1) if dc == 1 else range(c1 - 1, c0, -1)
                for idx, c in enumerate(cols):
                    if idx % 2 == 0:
                        out[r0 + 1 : r1, c] = color
            else:
                rows = range(r0 + 1, r1) if dr == 1 else range(r1 - 1, r0, -1)
                for idx, r in enumerate(rows):
                    if idx % 2 == 0:
                        out[r, c0 + 1 : c1] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    occupied = np.zeros((h, w), dtype=bool)
    for color in colors:
        fh, fw = random.randint(5, 7), random.randint(5, 7)
        for _ in range(200):
            top = random.randint(2, h - fh - 3)
            left = random.randint(2, w - fw - 3)
            if occupied[top - 2 : top + fh + 2, left - 2 : left + fw + 2].any():
                continue
            bottom, right = top + fh - 1, left + fw - 1
            grid[top, left : right + 1] = GREY
            grid[bottom, left : right + 1] = GREY
            grid[top : bottom + 1, left] = GREY
            grid[top : bottom + 1, right] = GREY
            occupied[top : bottom + 1, left : right + 1] = True
            side = random.choice(["left", "right", "top", "bottom"])
            mid_r = (top + bottom) // 2
            mid_c = (left + right) // 2
            if side == "right":
                coords = [(mid_r - 1, left - 1), (mid_r, left), (mid_r + 1, left - 1)]
            elif side == "left":
                coords = [(mid_r - 1, right + 1), (mid_r, right), (mid_r + 1, right + 1)]
            elif side == "down":
                coords = [(top - 1, mid_c - 1), (top, mid_c), (top - 1, mid_c + 1)]
            else:
                coords = [(bottom + 1, mid_c - 1), (bottom, mid_c), (bottom + 1, mid_c + 1)]
            for r, c in coords:
                if 0 <= r < h and 0 <= c < w:
                    grid[r, c] = color
                    occupied[r, c] = True
            break
    return grid
