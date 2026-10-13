import random
import numpy as np


GLYPHS = {
    ((0, 1), (1, 1), (2, 1)): 0,
    ((1, 0), (1, 1), (1, 2)): 1,
    ((0, 0), (1, 1), (2, 2)): 2,
    ((0, 2), (1, 1), (2, 0)): 3,
}


def _glyph_key(block):
    pts = np.argwhere(block == 5)
    return tuple(sorted(map(tuple, pts.tolist())))


def _regions(grid):
    sep = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5)]
    bounds = [-1] + sep + [grid.shape[1]]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i] + 1 < bounds[i + 1]]


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    regs = _regions(grid)
    if len(regs) < 4:
        return out
    c0, c1 = regs[0]
    template = np.where((grid[:, c0:c1] != 0) & (grid[:, c0:c1] != 5), grid[:, c0:c1], 0)
    for c0, c1 in regs[1:4]:
        width = c1 - c0
        r0 = max(0, grid.shape[0] // 2 - 1)
        cc0 = c0 + width // 2 - 1
        glyph = grid[r0:r0 + 3, cc0:cc0 + 3]
        turns = GLYPHS.get(_glyph_key(glyph), 0)
        rot = np.rot90(template, -turns)
        patch = np.zeros((grid.shape[0], width), dtype=int)
        rh, rw = rot.shape
        patch[:min(rh, patch.shape[0]), :min(rw, width)] = rot[:min(rh, patch.shape[0]), :min(rw, width)]
        out[:, c0:c1] = patch
    return out


def generate_input() -> np.ndarray:
    h, region_w = 5, 5
    w = region_w * 4 + 3
    g = np.zeros((h, w), dtype=int)
    for c in (5, 11, 17):
        g[:, c] = 5
    template = np.array([
        [0, 1, 0, 0, 0],
        [2, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 4, 4],
        [0, 0, 0, 0, 4],
    ])
    g[:, :5] = template
    glyphs = list(GLYPHS.keys())
    for idx, start in enumerate((6, 12, 18)):
        key = random.choice(glyphs)
        for r, c in key:
            g[1 + r, start + 1 + c] = 5
    return g
