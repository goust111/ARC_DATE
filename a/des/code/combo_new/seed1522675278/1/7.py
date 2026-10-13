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
    Classify seed layouts inside grey rectangles and draw the corresponding cross, line, or full fill.

    Input structure:
    - Black background with multiple disjoint grey rectangular borders.
    - Each border encloses 1 to 3 same-colored seed pixels.
    - Seed layouts are singleton, horizontal pair, vertical pair, or L-shape triple.

    Rule:
    - A singleton seed grows a full cross to the border-adjacent interior.
    - A horizontal pair paints a horizontal line across the room; a vertical pair paints a vertical line.
    - A three-seed L layout fills the whole interior.

    Key insight:
    - The geometry of the seed arrangement determines the fill mode.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for box in _components(g == GREY):
        rs = [r for r, _ in box]
        cs = [c for _, c in box]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        pts = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if g[r, c] not in (BLACK, GREY)]
        if not pts:
            continue
        color = int(g[pts[0]])
        rows = {r for r, _ in pts}
        cols = {c for _, c in pts}
        if len(pts) == 1:
            r, c = pts[0]
            out[r0 + 1 : r1, c] = color
            out[r, c0 + 1 : c1] = color
        elif len(pts) == 2 and len(rows) == 1:
            out[list(rows)[0], c0 + 1 : c1] = color
        elif len(pts) == 2 and len(cols) == 1:
            out[r0 + 1 : r1, list(cols)[0]] = color
        else:
            out[r0 + 1 : r1, c0 + 1 : c1] = color
    out[g == GREY] = GREY
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    for _ in range(random.randint(2, 4)):
        for _ in range(50):
            bh = random.randint(4, 8)
            bw = random.randint(4, 8)
            r0 = random.randint(1, h - bh - 2)
            c0 = random.randint(1, w - bw - 2)
            r1, c1 = r0 + bh - 1, c0 + bw - 1
            if np.any(g[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] != BLACK):
                continue
            g[r0, c0 : c1 + 1] = GREY
            g[r1, c0 : c1 + 1] = GREY
            g[r0 : r1 + 1, c0] = GREY
            g[r0 : r1 + 1, c1] = GREY
            color = random.choice(COLORS)
            mode = random.choice(["single", "h", "v", "L"])
            if mode == "single":
                g[random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)] = color
            elif mode == "h":
                rr = random.randint(r0 + 1, r1 - 1)
                a, b = sorted(random.sample(range(c0 + 1, c1), 2))
                g[rr, a] = color
                g[rr, b] = color
            elif mode == "v":
                cc = random.randint(c0 + 1, c1 - 1)
                a, b = sorted(random.sample(range(r0 + 1, r1), 2))
                g[a, cc] = color
                g[b, cc] = color
            else:
                rr = random.randint(r0 + 1, r1 - 2)
                cc = random.randint(c0 + 1, c1 - 2)
                g[rr, cc] = color
                g[rr, cc + 1] = color
                g[rr + 1, cc] = color
            break
    return g
