import random

import numpy as np


def _find_sprites(g):
    h, w = g.shape
    sprites = []
    for r in range(h - 2):
        for c in range(w - 2):
            if r < 2 and c < 2:
                continue
            block = g[r:r + 3, c:c + 3]
            outer = [block[0, 0], block[0, 1], block[0, 2], block[1, 0],
                     block[1, 2], block[2, 0], block[2, 1], block[2, 2]]
            if block[1, 1] == 0 and outer[0] not in (0, 5) and all(x == outer[0] for x in outer):
                sprites.append((r, c, int(outer[0])))
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    sprites = _find_sprites(g)
    sprite_cells = set()
    for r, c, _ in sprites:
        for dr in range(3):
            for dc in range(3):
                if not (dr == 1 and dc == 1):
                    sprite_cells.add((r + dr, c + dc))
    guide = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] != 0 and not (r < 2 and c < 2) and (r, c) not in sprite_cells:
                guide.append((r, c))
    gr = [r for r, _ in guide]
    gc = [c for _, c in guide]
    br0, br1, bc0, bc1 = min(gr), max(gr), min(gc), max(gc)
    selected = [(r, c, color) for r, c, color in sprites
                if br0 <= r and r + 2 <= br1 and bc0 <= c and c + 2 <= bc1][0]
    pairs = {}
    for row in range(2):
        a, b = int(g[row, 0]), int(g[row, 1])
        pairs[a] = b
        pairs[b] = a
    out = np.zeros((3, 3), dtype=int)
    out[:] = pairs[selected[2]]
    out[1, 1] = 0
    return out


def _overlap(a, b):
    ar, ac = a
    br, bc = b
    return not (ar + 2 < br - 1 or br + 2 < ar - 1 or ac + 2 < bc - 1 or bc + 2 < ac - 1)


def generate_input() -> np.ndarray:
    h = w = random.randint(13, 16)
    g = np.zeros((h, w), dtype=int)
    g[:2, :2] = np.array([[1, 2], [3, 4]])
    colors = random.sample([1, 2, 3, 4], random.randint(3, 4))
    positions = []
    tries = 0
    while len(positions) < len(colors) and tries < 500:
        tries += 1
        pos = (random.randint(3, h - 4), random.randint(3, w - 4))
        if all(not _overlap(pos, old) for old in positions):
            positions.append(pos)
    for (r, c), color in zip(positions, colors):
        g[r:r + 3, c:c + 3] = color
        g[r + 1, c + 1] = 0
    idx = random.randrange(len(positions))
    sr, sc = positions[idx]
    br0, br1 = max(2, sr - 1), min(h - 1, sr + 3)
    bc0, bc1 = max(2, sc - 1), min(w - 1, sc + 3)
    for rr, cc in ((br0, bc0), (br0, bc1), (br1, bc0), (br1, bc1)):
        if g[rr, cc] == 0:
            g[rr, cc] = 5
    return g
