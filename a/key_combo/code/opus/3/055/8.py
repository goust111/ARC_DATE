from collections import Counter, deque
import random

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _find_sprite(grid, sep):
    for r in range(sep - 2):
        for c in range(grid.shape[1] - 2):
            win = grid[r:r + 3, c:c + 3]
            border = np.r_[win[0, :], win[2, :], win[1, 0], win[1, 2]]
            counts = Counter(int(x) for x in border)
            color, n = counts.most_common(1)[0]
            if color not in (0, 5) and n == 8 and win[1, 1] not in (0, 5, color):
                return r, c, win.copy(), int(win[1, 1]), color
    raise ValueError("no sprite found")


def main(grid: np.ndarray) -> np.ndarray:
    """Choose the growing marker color, stamp the centered sprite toward its matching rectangle, then tint lower background."""
    sep = next(r for r in range(grid.shape[0]) if np.all(grid[r] == 5))
    sr, sc, sprite, center_color, border_color = _find_sprite(grid, sep)
    excluded = {0, 5, center_color, border_color}
    colors = [int(x) for x in np.unique(grid) if int(x) not in excluded]
    best = None
    for color in colors:
        upper = int(np.sum(grid[:sep] == color))
        lower = int(np.sum(grid[sep + 1:] == color))
        item = (lower - upper, lower, color)
        if best is None or item > best:
            best = item
    _, nstamps, winner = best
    comps = _components((grid == center_color) & (np.arange(grid.shape[0])[:, None] > sep))
    rect = max(comps, key=len)
    rr0, rc0, rr1, rc1 = _bbox(rect)
    sprite_center = (sr + 1, sc + 1)
    rect_center = ((rr0 + rr1) // 2, (rc0 + rc1) // 2)
    dr, dc = rect_center[0] - sprite_center[0], rect_center[1] - sprite_center[1]
    if abs(dr) >= abs(dc):
        step = (3 if dr >= 0 else -3, 0)
    else:
        step = (0, 3 if dc >= 0 else -3)
    out = grid.copy()
    cr, cc = sprite_center
    for _ in range(nstamps):
        cr += step[0]
        cc += step[1]
        top, left = cr - 1, cc - 1
        overlaps_rect = not (top + 2 < rr0 or top > rr1 or left + 2 < rc0 or left > rc1)
        if 0 <= top and top + 2 < out.shape[0] and 0 <= left and left + 2 < out.shape[1]:
            out[top:top + 3, left:left + 3] = sprite
        if overlaps_rect:
            break
    lower = out[sep + 1:]
    lower[lower == 0] = winner
    out[sep + 1:] = lower
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    grid[7, :] = 5
    sprite = np.full((3, 3), 7, dtype=int)
    sprite[1, 1] = 4
    grid[1:4, 6:9] = sprite
    grid[12:15, 6:9] = 4
    counts = {1: (1, 4), 2: (2, 3), 3: (3, 2)}
    occupied = set(map(tuple, np.argwhere(grid != 0)))
    for color, (upper, lower) in counts.items():
        for area, amount in ((range(0, 7), upper), (range(8, 16), lower)):
            placed = 0
            while placed < amount:
                r = random.choice(list(area))
                c = random.randrange(16)
                if (r, c) not in occupied:
                    occupied.add((r, c))
                    grid[r, c] = color
                    placed += 1
    return grid
