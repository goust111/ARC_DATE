import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ANCHORS = {
    (0, 0): "tl",
    (0, 1): "tr",
    (1, 0): "bl",
    (1, 1): "br",
}


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _perimeter_bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey frame with a largest anchored square decoded from the top legend.

    Input structure:
    - The top two rows contain separated 2x2 legend blocks on a black background.
    - Each legend block has one black corner and three same-colored cells; the color
      matches exactly one seed color inside a grey hollow rectangle below.
    - Below the legend are disjoint one-pixel-thick grey rectangular frames whose
      interiors are otherwise black except for one colored seed pixel each.

    Rule:
    - Decode every legend block as color -> missing-corner anchor.
    - Remove the top legend rows from the output canvas.
    - For each grey frame, find its seed color, look up the anchor, compute the
      interior side length S = min(interior_height, interior_width), and fill an
      SxS solid square of the seed color aligned to that anchor inside the frame.
    - Grey frame borders remain unchanged.

    Key insight:
    - The legend's missing corner encodes geometry rather than color: seed color
      picks the legend entry, and the black corner determines which frame corner
      should anchor the maximal filled square.
    """
    legend = {}
    for c in range(grid.shape[1] - 1):
        patch = grid[:2, c : c + 2]
        vals = [int(v) for v in patch.ravel()]
        nz = [v for v in vals if v != BLACK]
        if len(nz) == 3 and len(set(nz)) == 1:
            miss = tuple(np.argwhere(patch == BLACK)[0])
            legend[nz[0]] = ANCHORS[miss]

    canvas = grid[2:].copy()
    for comp in _components(canvas == GREY):
        r0, c0, r1, c1 = _perimeter_bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        inner = canvas[r0 + 1 : r1, c0 + 1 : c1]
        seeds = [int(v) for v in np.unique(inner) if v not in (BLACK, GREY)]
        if len(seeds) != 1:
            continue
        color = seeds[0]
        side = min(inner.shape)
        anchor = legend[color]
        if anchor == "tl":
            rr = slice(r0 + 1, r0 + 1 + side)
            cc = slice(c0 + 1, c0 + 1 + side)
        elif anchor == "tr":
            rr = slice(r0 + 1, r0 + 1 + side)
            cc = slice(c1 - side, c1)
        elif anchor == "bl":
            rr = slice(r1 - side, r1)
            cc = slice(c0 + 1, c0 + 1 + side)
        else:
            rr = slice(r1 - side, r1)
            cc = slice(c1 - side, c1)
        canvas[rr, cc] = color
    return canvas


def generate_input() -> np.ndarray:
    while True:
        n = random.randint(2, 4)
        colors = random.sample(COLORS, n)
        missing = random.sample(list(ANCHORS.keys()), n)
        widths = [2] * n
        gaps = [random.randint(1, 3) for _ in range(n - 1)]
        legend_w = sum(widths) + sum(gaps)
        frame_specs = []
        x = 1
        max_h = 0
        for color in colors:
            ih = random.randint(3, 6)
            iw = random.randint(3, 6)
            fh, fw = ih + 2, iw + 2
            frame_specs.append((color, ih, iw, fh, fw))
            x += fw + random.randint(1, 3)
            max_h = max(max_h, fh)
        w = max(legend_w + 2, x + 1)
        h = 2 + 1 + max_h + random.randint(0, 2)
        if h > 30 or w > 30:
            continue
        grid = np.zeros((h, w), dtype=int)
        c = random.randint(0, w - legend_w)
        for idx, (color, miss) in enumerate(zip(colors, missing)):
            grid[:2, c : c + 2] = color
            grid[miss[0], c + miss[1]] = BLACK
            c += 2
            if idx < len(gaps):
                c += gaps[idx]

        cur_c = random.randint(0, 2)
        base_r = 3
        for color, ih, iw, fh, fw in frame_specs:
            r0 = base_r + random.randint(0, max(0, h - base_r - fh))
            c0 = cur_c
            if c0 + fw > w:
                break
            grid[r0 : r0 + fh, c0 : c0 + fw] = GREY
            grid[r0 + 1 : r0 + 1 + ih, c0 + 1 : c0 + 1 + iw] = BLACK
            sr = random.randint(r0 + 1, r0 + ih)
            sc = random.randint(c0 + 1, c0 + iw)
            grid[sr, sc] = color
            cur_c += fw + random.randint(1, 3)
        try:
            main(grid)
        except Exception:
            continue
        if np.any(grid == GREY):
            return grid
