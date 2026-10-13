import random
from collections import deque

import numpy as np


def _bbox(mask):
    rs, cs = np.where(mask)
    return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())


def _eight_components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    full_sep = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5)]
    if len(full_sep) < 2:
        return grid.copy()
    s1, s2 = full_sep[0], full_sep[-1]
    left = grid[:, :s1]
    middle = grid[:, s1 + 1:s2]
    right = grid[:, s2 + 1:]

    lr0, lr1, lc0, lc1 = _bbox(left == 5)
    mask = left[lr0:lr1 + 1, lc0:lc1 + 1] == 5
    clean = mask.copy()
    for comp in _eight_components(mask):
        if len(comp) == 1:
            clean[comp[0]] = False

    mr0, mr1, mc0, mc1 = _bbox(middle != 0)
    tile = middle[mr0:mr1 + 1, mc0:mc1 + 1]
    rr, cc = np.where(right != 0)
    order = sorted(zip(rr, cc), key=lambda x: x[0])
    bg_color = int(right[order[-1]]) if order else 3
    fg_color = int(right[order[0]]) if order else 4
    tile_colors = []
    for v in tile.ravel():
        iv = int(v)
        if iv not in tile_colors:
            tile_colors.append(iv)
    mapped = tile.copy()
    if tile_colors:
        mapped[tile == tile_colors[0]] = fg_color
    if len(tile_colors) > 1:
        mapped[tile == tile_colors[1]] = bg_color

    th, tw = mapped.shape
    out = np.full((clean.shape[0] * th, clean.shape[1] * tw), bg_color, dtype=int)
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            if clean[r, c]:
                out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = mapped
    return out


def generate_input() -> np.ndarray:
    mh = random.randint(6, 8)
    mw = random.randint(6, 8)
    th, tw = random.choice([(2, 2), (2, 3), (3, 2)])
    h = max(mh + 2, th + 2, 8)
    w = mw + 1 + tw + 1 + 3
    grid = np.zeros((h, w), dtype=int)
    s1 = mw
    s2 = mw + 1 + tw
    grid[:, s1] = 5
    grid[:, s2] = 5

    mask = np.zeros((mh, mw), dtype=int)
    starts = [(1, 1), (mh - 3, mw - 3), (1, mw - 3), (mh - 3, 1)]
    random.shuffle(starts)
    for r, c in starts[: random.randint(2, 4)]:
        shape = random.choice([[(0, 0), (0, 1)], [(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1)]])
        for dr, dc in shape:
            if 0 <= r + dr < mh and 0 <= c + dc < mw:
                mask[r + dr, c + dc] = 5
    placed = 0
    tries = 0
    while placed < 2 and tries < 100:
        tries += 1
        r, c = random.randrange(mh), random.randrange(mw)
        if mask[r, c] != 0:
            continue
        if any(0 <= r + dr < mh and 0 <= c + dc < mw and mask[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
            continue
        mask[r, c] = 5
        placed += 1
    grid[:mh, :mw] = mask

    colors = random.sample([1, 2, 6, 7, 8, 9], 2)
    tile = np.zeros((th, tw), dtype=int)
    for r in range(th):
        for c in range(tw):
            tile[r, c] = colors[(r + c + random.randrange(2)) % 2]
    if len(set(tile.ravel())) < 2:
        tile[0, 0] = colors[0]
        tile[-1, -1] = colors[1]
    grid[:th, s1 + 1:s2] = tile
    grid[1, s2 + 2] = 4
    grid[3, s2 + 2] = 3
    return grid
