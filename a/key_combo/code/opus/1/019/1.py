import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Drop singleton pixels into wall gaps and flood enclosed regions with their colors.

    Input structure: Several colored open wall objects sit on a black background,
    with isolated singleton pixels above their missing wall cells and optional
    isolated distractors elsewhere.
    Rule: Isolated singleton pixels fall vertically, lower ones first, until they
    rest above an obstacle or on the bottom row. The landed singletons become
    walls for background flood fill. Each enclosed background component bounded
    by exactly one landed singleton is filled with that singleton's color.
    Key insight: Gravity changes the topology first; only after the gaps are
    plugged can the formerly open regions be identified and colored.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    singleton_comps = [comp for comp in _components(grid != 0) if len(comp) == 1]
    singletons = [(comp[0][0], comp[0][1], int(grid[comp[0]])) for comp in singleton_comps]

    base = grid.copy()
    for r, c, _ in singletons:
        base[r, c] = 0

    landed = []
    for r, c, val in sorted(singletons, key=lambda x: x[0], reverse=True):
        rr = r
        while rr + 1 < h and base[rr + 1, c] == 0:
            rr += 1
        base[rr, c] = val
        landed.append((rr, c, val))

    blocked = base != 0
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    landed_at = {(r, c): val for r, c, val in landed}
    out = base.copy()
    enclosed = (~blocked) & (~outside)
    for comp in _components(enclosed):
        boundary_drops = {}
        for r, c in comp:
            for dr, dc in DIR4:
                pos = (r + dr, c + dc)
                if pos in landed_at:
                    boundary_drops[pos] = landed_at[pos]
        if len(boundary_drops) == 1:
            fill = next(iter(boundary_drops.values()))
            for r, c in comp:
                out[r, c] = fill
    return out


def _draw_corner_gap_box(grid, top, left, height, width, color, gap_side):
    bottom = top + height - 1
    right = left + width - 1
    for c in range(left, right + 1):
        grid[top, c] = color
        grid[bottom, c] = color
    for r in range(top, bottom + 1):
        grid[r, left] = color
        grid[r, right] = color
    if gap_side == "left":
        grid[top, left] = 0
        return top, left
    grid[top, right] = 0
    return top, right


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    layouts = [(5, 1), (5, 7), (10, 3), (10, 10)]
    random.shuffle(layouts)
    wall_colors = random.sample([1, 2, 3, 5, 8, 9], random.randint(2, 3))
    used_wall = set(wall_colors)
    available_drops = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in used_wall]
    drop_colors = random.sample(available_drops, min(len(available_drops), len(wall_colors) + 2))

    for idx, color in enumerate(wall_colors):
        top, left = layouts[idx]
        height = random.choice([4, 5])
        width = random.choice([4, 5])
        gap_side = random.choice(["left", "right"])
        gr, gc = _draw_corner_gap_box(grid, top, left, height, width, color, gap_side)
        drop_r = random.randint(0, gr - 2)
        grid[drop_r, gc] = drop_colors[idx]

    for color in drop_colors[len(wall_colors):len(wall_colors) + random.randint(0, 2)]:
        for _ in range(50):
            r = random.randint(0, 4)
            c = random.randint(0, 15)
            if grid[r, c] == 0 and grid[min(15, r + 1), c] == 0:
                grid[r, c] = color
                break
    return grid
