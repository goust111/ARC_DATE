from collections import deque
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


def _sprites(grid):
    ans = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            win = grid[r:r + 3, c:c + 3]
            border = np.r_[win[0, :], win[2, :], win[1, 0], win[1, 2]]
            vals = set(int(x) for x in border)
            if len(vals) == 1 and 0 not in vals and win[1, 1] == 0:
                ans.append((r, c, win.copy()))
    return ans


def main(grid: np.ndarray) -> np.ndarray:
    """Select the guided hollow sprite, pack bottom rectangles diagonally, and stamp the sprite on each joint."""
    guide = np.argwhere(grid == 2)
    gr0, gc0 = guide.min(axis=0)
    gr1, gc1 = guide.max(axis=0)
    selected = None
    for r, c, sprite in _sprites(grid):
        if gr0 <= r and c >= gc0 and r + 2 <= gr1 and c + 2 <= gc1:
            selected = sprite
            break
    if selected is None:
        selected = _sprites(grid)[0][2]
    rects = []
    for comp in _components(np.isin(grid, [5, 8, 9, 4])):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 == grid.shape[0] - 1:
            vals = {int(grid[p]) for p in comp}
            if len(vals) == 1 and len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1):
                rects.append((c0, r1 - r0 + 1, c1 - c0 + 1, vals.pop()))
    rects.sort()
    out = np.zeros_like(grid)
    r = c = 0
    joints = []
    for i, (_, h, w, color) in enumerate(rects):
        out[r:r + h, c:c + w] = color
        if i < len(rects) - 1:
            joints.append((r + h - 1, c + w - 1))
        r, c = r + h - 1, c + w - 1
    for jr, jc in joints:
        for sr in range(3):
            for sc in range(3):
                if selected[sr, sc] != 0:
                    rr, cc = jr + sr - 1, jc + sc - 1
                    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                        out[rr, cc] = selected[sr, sc]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 20), dtype=int)
    sprite_positions = [(1, 2), (1, 9), (5, 5), (5, 13)]
    colors = random.sample([1, 3, 7, 6], 4)
    for (r, c), color in zip(sprite_positions, colors):
        grid[r:r + 3, c:c + 3] = color
        grid[r + 1, c + 1] = 0
    target = random.randrange(4)
    tr, tc = sprite_positions[target]
    for p in [(tr - 1, tc - 1), (tr - 1, tc + 3), (tr + 3, tc - 1), (tr + 3, tc + 3)]:
        grid[p] = 2
    rect_specs = [(3, 2, 5), (2, 4, 8), (4, 3, 9), (2, 2, 4)]
    lefts = [1, 6, 11, 16]
    for c, (h, w, color) in zip(lefts, random.sample(rect_specs, random.randint(3, 4))):
        grid[15 - h:15, c:c + w] = color
    return grid
