import random
import numpy as np


BG = 0
GREY = 5
RED = 2


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Rank enclosed regions by red marker count and fill from a bottom palette.

    Input structure: Grey linework encloses several black regions. Each enclosed
    region contains a unique number of red marker pixels. The bottom row has a
    left-to-right palette with one color per region.
    Rule: Count red markers in every enclosed region, sort regions from fewest
    to most markers, and assign palette colors by rank. Fill both background
    and red marker cells inside a region with its ranked palette color, while
    preserving grey boundaries and the palette.
    Key insight: Marker counts define an ordering first; only then does the
    bottom strip act as the rank-to-color lookup.
    """
    out = grid.copy()
    h, w = grid.shape
    palette = [int(grid[h - 1, c]) for c in range(w) if int(grid[h - 1, c]) not in (BG, GREY, RED)]
    passable = grid != GREY
    passable[h - 1, :] = False
    regions = []
    for comp in _components(passable):
        if any(r in (0, h - 2) or c in (0, w - 1) for r, c in comp):
            continue
        k = sum(grid[r, c] == RED for r, c in comp)
        if k:
            regions.append((k, comp))
    regions.sort(key=lambda x: x[0])
    for idx, (_, comp) in enumerate(regions):
        if idx >= len(palette):
            break
        for r, c in comp:
            out[r, c] = palette[idx]
    return out


def generate_input() -> np.ndarray:
    h = w = 24
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 3, 4, 6, 7, 8, 9], 5)
    grid[h - 1, :5] = colors
    frames = [(2, 2), (2, 11), (10, 3), (11, 13)]
    counts = random.sample(range(1, 7), len(frames))
    for (r0, c0), k in zip(frames, counts):
        rh, rw = random.randint(5, 7), random.randint(5, 8)
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        grid[r0, c0:c1 + 1] = GREY
        grid[r1, c0:c1 + 1] = GREY
        grid[r0:r1 + 1, c0] = GREY
        grid[r0:r1 + 1, c1] = GREY
        spots = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        for r, c in random.sample(spots, k):
            grid[r, c] = RED
    return grid
