import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5


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
    Project the selected colored border segment inward as a perpendicular bar chosen by the interior marker color.

    Input structure:
    - Grey hollow rectangles contain one interior marker pixel.
    - Their borders may include short colored segments replacing part of one edge.
    - The marker color matches exactly one border segment on the same frame.

    Rule:
    - Use the marker color to choose the border segment.
    - Erase the marker.
    - Convert a horizontal segment into a vertical inward bar of the same length at the segment midpoint, and convert a vertical segment analogously into a horizontal inward bar.

    Key insight:
    - The interior marker selects which border segment is active, and the chosen segment's orientation determines the projection direction.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    for cells in _components(g == FRAME):
        r0, c0, r1, c1 = _bbox(cells)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        mark = [(r0 + 1 + r, c0 + 1 + c, int(inner[r, c])) for r in range(inner.shape[0]) for c in range(inner.shape[1]) if inner[r, c] != BG]
        if not mark:
            continue
        mr, mc, color = mark[0]
        out[mr, mc] = BG
        seg = [(r, c) for r in [r0, r1] for c in range(c0, c1 + 1) if g[r, c] == color]
        if not seg:
            seg = [(r, c) for c in [c0, c1] for r in range(r0, r1 + 1) if g[r, c] == color]
        if len({r for r, _ in seg}) == 1:
            mid = sorted(c for _, c in seg)[len(seg) // 2]
            step = 1 if seg[0][0] == r0 else -1
            for k in range(1, len(seg) + 1):
                rr = seg[0][0] + step * k
                if r0 < rr < r1:
                    out[rr, mid] = color
        else:
            mid = sorted(r for r, _ in seg)[len(seg) // 2]
            step = 1 if seg[0][1] == c0 else -1
            for k in range(1, len(seg) + 1):
                cc = seg[0][1] + step * k
                if c0 < cc < c1:
                    out[mid, cc] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 4))
    for color in colors:
        fh, fw = random.randint(6, 9), random.randint(6, 9)
        for _ in range(200):
            r0 = random.randint(0, h - fh)
            c0 = random.randint(0, w - fw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)]):
                continue
            g[r0 : r0 + fh, c0] = g[r0 : r0 + fh, c0 + fw - 1] = FRAME
            g[r0, c0 : c0 + fw] = g[r0 + fh - 1, c0 : c0 + fw] = FRAME
            if random.choice([True, False]):
                length = random.randint(2, fw - 2)
                start = random.randint(c0 + 1, c0 + fw - length - 1)
                row = random.choice([r0, r0 + fh - 1])
                g[row, start : start + length] = color
                g[r0 + fh // 2, c0 + fw // 2] = color
            else:
                length = random.randint(2, fh - 2)
                start = random.randint(r0 + 1, r0 + fh - length - 1)
                col = random.choice([c0, c0 + fw - 1])
                g[start : start + length, col] = color
                g[r0 + fh // 2, c0 + fw // 2] = color
            break
    return g
