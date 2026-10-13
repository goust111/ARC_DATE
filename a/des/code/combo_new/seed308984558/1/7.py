import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


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
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter one color at a time from the template and concatenate the cropped results by filter order.

    Input structure:
    - A black background contains one large multicolor template component.
    - Several 2x2 filter blocks contain three grey pixels and one colored selector pixel.
    - Each selector color appears somewhere inside the template.

    Rule:
    - For each filter, extract all template pixels of the selector color and crop them to their minimal bounding box.
    - Order filters from left to right by their original position.
    - Concatenate the cropped monochrome extracts horizontally with one blank column between them and bottom-align the pieces.

    Key insight:
    - The small 2x2 blocks act as color selectors, and the output order comes from filter position rather than from template geometry.
    """
    g = np.array(grid, dtype=int)
    filters = []
    filter_mask = np.zeros_like(g, dtype=bool)
    for cells in _components(g != BG):
        r0, c0, r1, c1 = _bbox(cells)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2) or len(cells) != 4:
            continue
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.count_nonzero(block == GREY) == 3:
            color = int([v for v in block.ravel() if v not in (BG, GREY)][0])
            filters.append((c0, color))
            filter_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    template_cells = max(_components((g != BG) & ~filter_mask), key=len)
    r0, c0, r1, c1 = _bbox(template_cells)
    template = g[r0 : r1 + 1, c0 : c1 + 1]
    crops = []
    for _, color in sorted(filters):
        pts = np.argwhere(template == color)
        rr0, cc0 = pts.min(axis=0)
        rr1, cc1 = pts.max(axis=0)
        crop = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=int)
        crop[template[rr0 : rr1 + 1, cc0 : cc1 + 1] == color] = color
        crops.append(crop)
    height = max(c.shape[0] for c in crops)
    width = sum(c.shape[1] for c in crops) + len(crops) - 1
    out = np.zeros((height, width), dtype=int)
    cur = 0
    for crop in crops:
        out[height - crop.shape[0] :, cur : cur + crop.shape[1]] = crop
        cur += crop.shape[1] + 1
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4], 3)
    th, tw = random.randint(5, 7), random.randint(5, 6)
    r0 = random.randint(4, h - th - 4)
    c0 = random.randint(4, w - tw - 4)
    for r in range(th):
        for c in range(tw):
            g[r0 + r, c0 + c] = random.choice(colors)
    for color in colors:
        if color not in g[r0 : r0 + th, c0 : c0 + tw]:
            g[r0 + random.randint(0, th - 1), c0 + random.randint(0, tw - 1)] = color
    start = random.randint(1, max(1, w - (len(colors) * 4)))
    xs = [start + i * 4 for i in range(len(colors))]
    fr = r0 + th + 3 if r0 + th + 5 <= h else max(0, r0 - 4)
    for x, color in zip(xs, colors):
        g[fr : fr + 2, x : x + 2] = GREY
        rr, cc = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        g[fr + rr, x + cc] = color
    return g
