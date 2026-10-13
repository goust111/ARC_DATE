import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _corner_count(cells):
    s = set(cells)
    rs = [r for r, _ in s]
    cs = [c for _, c in s]
    count = 0
    for vr in range(min(rs), max(rs) + 2):
        for vc in range(min(cs), max(cs) + 2):
            occ = 0
            for dr, dc in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
                if (vr + dr, vc + dc) in s:
                    occ += 1
            if occ in (1, 3):
                count += 1
    return count


def _palette(g):
    h, w = g.shape
    mapping = {}
    mask = (g != BG) & (g != GREY)
    for pts in _components(mask):
        if not any(r == h - 1 or c == w - 1 for r, c in pts):
            continue
        colors = {int(g[p]) for p in pts}
        if len(colors) != 1:
            continue
        color = colors.pop()
        mapping[len(pts)] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill closed line regions by matching their side count to palette-block area.

    Input structure:
    - Black background with grey orthogonal closed line boundaries and paired colored anchor pixels on those boundaries.
    - Each enclosed black region is separated from the outside by non-black boundary pixels.
    - Solid palette blocks on the bottom or right edge have distinct pixel areas.

    Rule:
    - Remove the edge palette blocks after reading their area-to-color mapping.
    - Find every enclosed black component, count its orthogonal boundary corners, and use that count as the side key.
    - Fill the enclosed black component with the palette color whose block area equals that side count.

    Key insight:
    - Region geometry selects a palette color indirectly through side count, while anchors and grey lines remain as the visible boundary.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    pal = _palette(g)
    h, w = g.shape
    for pts in _components((g != BG) & (g != GREY)):
        if any(r == h - 1 or c == w - 1 for r, c in pts):
            for r, c in pts:
                out[r, c] = BG
    for pts in _components(g == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        sides = _corner_count(pts)
        if sides in pal:
            for r, c in pts:
                out[r, c] = pal[sides]
    return out


def _shape_cells(rng, kind, h, w):
    if kind == "rect":
        return {(r, c) for r in range(h) for c in range(w)}
    cells = {(r, c) for r in range(h) for c in range(w)}
    cut_h = rng.randint(1, max(1, h // 2))
    cut_w = rng.randint(1, max(1, w // 2))
    corner = rng.choice(["tl", "tr", "bl", "br"])
    for r in range(cut_h):
        for c in range(cut_w):
            rr = r if "t" in corner else h - 1 - r
            cc = c if "l" in corner else w - 1 - c
            cells.discard((rr, cc))
    return cells


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(20, 30)
        w = rng.randint(20, 30)
        g = np.zeros((h, w), dtype=int)
        n = rng.randint(2, 3)
        fill_colors = rng.sample(colors, n)
        anchor_colors = rng.sample([c for c in colors if c not in fill_colors], n)
        occupied = set()
        side_keys = []
        ok = True
        for i in range(n):
            placed = False
            for _try in range(100):
                ih = rng.randint(3, 6)
                iw = rng.randint(3, 6)
                cells = _shape_cells(rng, rng.choice(["rect", "l"]), ih, iw)
                r0 = rng.randint(2, h - ih - 6)
                c0 = rng.randint(2, w - iw - 3)
                interior = {(r0 + r, c0 + c) for r, c in cells}
                boundary = set()
                for r, c in interior:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if (r + dr, c + dc) not in interior:
                            boundary.add((r + dr, c + dc))
                footprint = interior | boundary
                if footprint & occupied or any(not (0 <= r < h - 4 and 0 <= c < w) for r, c in footprint):
                    continue
                occupied |= footprint
                for r, c in boundary:
                    g[r, c] = GREY
                b_list = list(boundary)
                for r, c in rng.sample(b_list, 2):
                    g[r, c] = anchor_colors[i]
                side_keys.append(_corner_count(interior))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok or len(set(side_keys)) != len(side_keys):
            continue
        col = 1
        for key, color in zip(side_keys, fill_colors):
            if col + key >= w:
                ok = False
                break
            g[h - 1, col : col + key] = color
            col += key + 1
        if ok:
            return g
    raise RuntimeError("generate_input failed")
