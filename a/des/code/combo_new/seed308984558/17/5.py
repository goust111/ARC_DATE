import random

import numpy as np


def _bbox(mask: np.ndarray):
    rs, cs = np.where(mask)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def main(grid: np.ndarray) -> np.ndarray:
    """Pair each tag with its nearest rectangle, then recolor and rotate if orientation mismatches."""
    g = np.array(grid, dtype=int)
    out = np.zeros_like(g)
    rects = []
    for color in [5]:
        mask = g == color
        seen = np.zeros_like(mask, dtype=bool)
        for r, c in zip(*np.where(mask)):
            if seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rects.append(_bbox(np.isin(np.indices(g.shape).transpose(1, 2, 0), pts).all(axis=2)))
    tags = [(int(r), int(c), int(g[r, c])) for r, c in zip(*np.where((g != 0) & (g != 5)))]
    out[:, :] = 0
    for r0, c0, r1, c1 in rects:
        best = None
        for tr, tc, color in tags:
            dist = max(r0 - tr, 0, tr - r1) + max(c0 - tc, 0, tc - c1)
            if best is None or dist < best[0]:
                best = (dist, tr, tc, color)
        _, tr, tc, color = best
        h, w = r1 - r0 + 1, c1 - c0 + 1
        vertical = h > w
        side = "up" if tr < r0 else "down" if tr > r1 else "left" if tc < c0 else "right"
        if (side in ("up", "down")) ^ vertical:
            nh, nw = w, h
        else:
            nh, nw = h, w
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        nr0 = cr - nh // 2
        nc0 = cc - nw // 2
        out[nr0 : nr0 + nh, nc0 : nc0 + nw] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    rects = []
    for color in random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4)):
        for _ in range(100):
            h, w = random.choice([(2, 4), (2, 5), (4, 2), (5, 2), (3, 5), (5, 3)])
            r0 = random.randint(1, g.shape[0] - h - 2)
            c0 = random.randint(1, g.shape[1] - w - 2)
            box = (r0, c0, r0 + h - 1, c0 + w - 1, color)
            if all(box[2] + 1 < b[0] or b[2] + 1 < box[0] or box[3] + 1 < b[1] or b[3] + 1 < box[1] for b in rects):
                rects.append(box)
                g[r0 : r0 + h, c0 : c0 + w] = 5
                side = random.choice(["up", "down", "left", "right"])
                rr = r0 - 1 if side == "up" else r0 + h if side == "down" else random.randint(r0, r0 + h - 1)
                cc = c0 - 1 if side == "left" else c0 + w if side == "right" else random.randint(c0, c0 + w - 1)
                g[rr, cc] = color
                break
    return g
