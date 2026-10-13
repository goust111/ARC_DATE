import random
from collections import deque

import numpy as np


BOUNDARY = 5


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            touch = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touch = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((cells, touch))
    return comps


def _legend(grid: np.ndarray, bg: int):
    mapping = {}
    mask = np.zeros_like(grid, dtype=bool)
    r = 0
    while r < grid.shape[0]:
        if grid[r, 0] in (bg, BOUNDARY):
            r += 1
            continue
        key = int(grid[r, 0])
        size = None
        for k in (3, 2):
            if r + k <= grid.shape[0] and 1 + k <= grid.shape[1]:
                tex = grid[r : r + k, 1 : 1 + k]
                if np.all(tex != bg) and np.all(tex != BOUNDARY):
                    size = k
                    break
        assert size is not None
        mapping[key] = grid[r : r + size, 1 : 1 + size].copy()
        mask[r : r + size, 0 : 1 + size] = True
        r += size + 1
    return mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each interior seed color as a key into the edge legend, then tile that legend texture across the whole enclosed region.

    Input structure:
    - The grid has a dominant background color plus several closed grey-bordered regions.
    - Each region contains exactly one non-background seed pixel and otherwise background interior cells.
    - Along one grid edge, each legend entry consists of a single key color cell next to a 2x2 or 3x3 texture patch.

    Rule:
    - Read the legend as a mapping from seed color to texture patch.
    - Find every enclosed non-border region that contains one seed pixel.
    - Replace the entire region interior, including the seed, by tiling the matching texture from the region's top-left corner and clipping at the region boundary.
    - Keep the grey borders and the legend area unchanged.

    Key insight:
    - The lone seed pixel is not output content; it is an index into a visible color-to-texture lookup table on the edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = g.copy()
    mapping, legend_mask = _legend(g, bg)
    passable = (g != BOUNDARY) & (~legend_mask)
    for cells, touch in _components(passable):
        if touch:
            continue
        seed_colors = {int(g[r, c]) for r, c in cells if g[r, c] != bg}
        if len(seed_colors) != 1:
            continue
        seed = next(iter(seed_colors))
        tex = mapping[seed]
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        region = set(cells)
        for r, c in cells:
            tr = (r - r0) % tex.shape[0]
            tc = (c - c0) % tex.shape[1]
            out[r, c] = tex[tr, tc]
    return out


def _draw_rect_region(grid, top, left, height, width, color):
    grid[top, left : left + width] = BOUNDARY
    grid[top + height - 1, left : left + width] = BOUNDARY
    grid[top : top + height, left] = BOUNDARY
    grid[top : top + height, left + width - 1] = BOUNDARY
    sr = random.randint(top + 1, top + height - 2)
    sc = random.randint(left + 1, left + width - 2)
    grid[sr, sc] = color


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(18, 28), random.randint(18, 28)
        bg = 0
        grid = np.full((h, w), bg, dtype=int)
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
        row = 0
        textures = {}
        ok = True
        for color in colors:
            size = random.choice([2, 3])
            if row + size > h - 8:
                ok = False
                break
            tex_colors = random.sample([c for c in range(10) if c not in (bg, BOUNDARY, color)], 2)
            tex = np.array(
                [[tex_colors[(r + c + random.randint(0, 1)) % 2] for c in range(size)] for r in range(size)],
                dtype=int,
            )
            textures[color] = tex
            grid[row, 0] = color
            grid[row : row + size, 1 : 1 + size] = tex
            row += size + 1
        if not ok:
            continue
        occ = np.zeros((h, w), dtype=bool)
        occ[:, :5] = True
        for _ in range(random.randint(2, 4)):
            color = random.choice(colors)
            placed = False
            for _ in range(100):
                hh, ww = random.randint(5, 8), random.randint(5, 8)
                top = random.randint(1, h - hh - 1)
                left = random.randint(6, w - ww - 1)
                if occ[max(0, top - 1) : min(h, top + hh + 1), max(0, left - 1) : min(w, left + ww + 1)].any():
                    continue
                _draw_rect_region(grid, top, left, hh, ww, color)
                occ[max(0, top - 1) : min(h, top + hh + 1), max(0, left - 1) : min(w, left + ww + 1)] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
