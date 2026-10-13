import random
from collections import deque

import numpy as np


BG = 0


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
    Slide each same-color fragment into the missing border cells of its matching rectangular ring.

    Input structure:
    - Each foreground color forms one hollow rectangular ring with one gap and one detached fragment of the same color.
    - The fragment shape matches the missing border cells exactly.
    - Fragment and gap are aligned horizontally or vertically.

    Rule:
    - For each color, identify the ring bbox and the missing border coordinates.
    - Remove the detached fragment.
    - Paint the missing border coordinates with that color to close the ring.

    Key insight:
    - Color identifies the pair, but the real job is finding the ring's missing border geometry and matching it to the detached fragment.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    for color in [v for v in np.unique(g) if v != BG]:
        comps = _components(g == int(color))
        ring = max(comps, key=len)
        frag = min(comps, key=len)
        r0, c0, r1, c1 = _bbox(ring)
        need = []
        for c in range(c0, c1 + 1):
            if g[r0, c] != color:
                need.append((r0, c))
            if g[r1, c] != color:
                need.append((r1, c))
        for r in range(r0, r1 + 1):
            if g[r, c0] != color:
                need.append((r, c0))
            if g[r, c1] != color:
                need.append((r, c1))
        for r, c in frag:
            out[r, c] = BG
        for r, c in need:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 2, 3, 4, 6, 7], random.randint(2, 3)):
        fh, fw = random.randint(5, 8), random.randint(5, 8)
        for _ in range(200):
            r0 = random.randint(0, h - fh)
            c0 = random.randint(0, w - fw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)]):
                continue
            border = []
            for r in range(r0, r0 + fh):
                border += [(r, c0), (r, c0 + fw - 1)]
            for c in range(c0, c0 + fw):
                border += [(r0, c), (r0 + fh - 1, c)]
            gap_side = random.choice(["top", "bottom", "left", "right"])
            if gap_side in ("top", "bottom"):
                cols = random.sample(range(c0 + 1, c0 + fw - 1), random.randint(1, 2))
                gap = [(r0 if gap_side == "top" else r0 + fh - 1, c) for c in cols]
            else:
                rows = random.sample(range(r0 + 1, r0 + fh - 1), random.randint(1, 2))
                gap = [(r, c0 if gap_side == "left" else c0 + fw - 1) for r in rows]
            for r, c in border:
                if (r, c) not in gap:
                    g[r, c] = color
            gr0, gc0, gr1, gc1 = min(r for r, _ in gap), min(c for _, c in gap), max(r for r, _ in gap), max(c for _, c in gap)
            for _ in range(200):
                if gap_side in ("top", "bottom"):
                    rr = gr0 + random.choice([-3, 3])
                    cc = gc0
                else:
                    rr = gr0
                    cc = gc0 + random.choice([-3, 3])
                if rr < 0 or cc < 0 or rr + (gr1 - gr0 + 1) > h or cc + (gc1 - gc0 + 1) > w:
                    continue
                if np.any(g[max(0, rr - 1) : min(h, rr + (gr1 - gr0 + 1) + 1), max(0, cc - 1) : min(w, cc + (gc1 - gc0 + 1) + 1)]):
                    continue
                for r, c in gap:
                    g[rr + r - gr0, cc + c - gc0] = color
                break
            break
    return g
