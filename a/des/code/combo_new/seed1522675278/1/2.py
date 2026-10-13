import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the fragment color inventory from the palette box and output the missing fragment color as a full rectangle.

    Input structure:
    - Black background with several disconnected colored fragment components.
    - A grey bordered palette box contains one cell for every fragment color plus one extra color.
    - Fragment positions are noisy; only their colors and total span matter.

    Rule:
    - The color present in the palette but absent from all fragments is the missing color.
    - The target rectangle width is the horizontal span of all fragment pixels.
    - The target area is fragment area plus one missing cell, so height is inferred from that total.

    Key insight:
    - The fragments need not be reassembled; the palette is only a color inventory, and the missing piece has area one.
    """
    g = np.asarray(grid, dtype=int)
    grey_pts = np.argwhere(g == GREY)
    r0, c0 = grey_pts.min(axis=0)
    r1, c1 = grey_pts.max(axis=0)
    palette_colors = {int(g[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] != BLACK}
    frag_mask = (g != BLACK) & (g != GREY)
    frag_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(frag_mask)
    frag_colors = {int(g[pts[0]]) for pts in comps}
    missing = list(palette_colors - frag_colors)[0]
    cols = [c for pts in comps for _, c in pts]
    width = max(cols) - min(cols) + 1
    area = sum(len(pts) for pts in comps) + 1
    height = area // width
    return np.full((height, width), missing, dtype=int)


def generate_input() -> np.ndarray:
    while True:
        width = random.randint(3, 6)
        height = random.randint(2, 5)
        rect = np.zeros((height, width), dtype=int)
        miss_r = random.randrange(height)
        miss_c = random.randrange(width)
        colors = random.sample(COLORS, random.randint(2, min(5, width * height - 1)))
        fill = [(r, c) for r in range(height) for c in range(width) if (r, c) != (miss_r, miss_c)]
        random.shuffle(fill)
        splits = [0]
        remain = len(fill)
        while remain:
            step = random.randint(1, min(4, remain))
            splits.append(splits[-1] + step)
            remain -= step
        if len(splits) - 1 < 2:
            continue
        groups = [fill[splits[i] : splits[i + 1]] for i in range(len(splits) - 1)]
        random.shuffle(colors)
        for color, cells in zip(colors * 10, groups):
            for r, c in cells:
                rect[r, c] = color
        if len(np.unique(rect[rect != 0])) < 2:
            continue
        h = height + 8
        w = width + 8
        g = np.zeros((h, w), dtype=int)
        for r in range(height):
            for c in range(width):
                if rect[r, c]:
                    g[1 + r + random.randint(0, 2), 1 + c] = rect[r, c]
        comps = _components(g != 0)
        if sum(len(c) for c in comps) != len(fill):
            continue
        pal_colors = list({int(x) for x in rect[rect != 0]})
        missing = random.choice([c for c in COLORS if c not in pal_colors])
        pal = pal_colors + [missing]
        random.shuffle(pal)
        pw = len(pal) + 2
        ph = 3
        pr0 = h - ph - 1
        pc0 = w - pw - 1
        g[pr0, pc0 : pc0 + pw] = GREY
        g[pr0 + ph - 1, pc0 : pc0 + pw] = GREY
        g[pr0 : pr0 + ph, pc0] = GREY
        g[pr0 : pr0 + ph, pc0 + pw - 1] = GREY
        for i, c in enumerate(pal):
            g[pr0 + 1, pc0 + 1 + i] = c
        return g
