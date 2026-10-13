import random

import numpy as np


BLACK = 0
FRAME = 3
SEED = 2


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
            out.append(sorted(pts))
    return out


def _runs(xs):
    xs = sorted(xs)
    if not xs:
        return []
    out = [[xs[0], xs[0]]]
    for x in xs[1:]:
        if x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return [(a, b) for a, b in out]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair interior seed shapes with frame-edge gaps by ordering, erase the seeds, and extrude each shape outward.

    Input structure:
    - Black background with one or more monochrome hollow rectangular frames.
    - Each frame contains several small seed components in another color.
    - The frame border contains the same number of contiguous gaps.

    Rule:
    - Seeds are ordered by reading order inside the frame.
    - Gaps are ordered clockwise starting from the top edge.
    - Each seed is translated outside the frame through its matched gap without rotation, and the original seed is erased.

    Key insight:
    - The pairing uses two different orderings: reading order for seeds and perimeter order for gaps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for comp in _components(g == FRAME):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        seeds = sorted(_components((g == SEED)[r0 + 1 : r1, c0 + 1 : c1]), key=lambda pts: (pts[0][0], pts[0][1]))
        gaps = []
        for a, b in _runs([c for c in range(c0 + 1, c1) if g[r0, c] == BLACK]):
            gaps.append(("top", a, b))
        for a, b in _runs([r for r in range(r0 + 1, r1) if g[r, c1] == BLACK]):
            gaps.append(("right", a, b))
        for a, b in reversed(_runs([c for c in range(c0 + 1, c1) if g[r1, c] == BLACK])):
            gaps.append(("bottom", a, b))
        for a, b in reversed(_runs([r for r in range(r0 + 1, r1) if g[r, c0] == BLACK])):
            gaps.append(("left", a, b))
        for seed, (side, a, b) in zip(seeds, gaps):
            seed = [(r + r0 + 1, c + c0 + 1) for r, c in seed]
            sr0 = min(r for r, _ in seed)
            sc0 = min(c for _, c in seed)
            norm = [(r - sr0, c - sc0) for r, c in seed]
            for r, c in seed:
                out[r, c] = BLACK
            if side == "top":
                base_r, base_c = r0 - (max(r for r, _ in norm) + 1), a
            elif side == "bottom":
                base_r, base_c = r1 + 1, a
            elif side == "left":
                base_r, base_c = a, c0 - (max(c for _, c in norm) + 1)
            else:
                base_r, base_c = a, c1 + 1
            for dr, dc in norm:
                rr, cc = base_r + dr, base_c + dc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = SEED
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    g = np.zeros((h, w), dtype=int)
    for _ in range(random.randint(1, 2)):
        for _ in range(50):
            bh, bw = random.randint(7, 10), random.randint(7, 10)
            r0, c0 = random.randint(3, h - bh - 3), random.randint(3, w - bw - 3)
            r1, c1 = r0 + bh - 1, c0 + bw - 1
            if np.any(g[r0 - 2 : r1 + 3, c0 - 2 : c1 + 3] != BLACK):
                continue
            g[r0, c0 : c1 + 1] = FRAME
            g[r1, c0 : c1 + 1] = FRAME
            g[r0 : r1 + 1, c0] = FRAME
            g[r0 : r1 + 1, c1] = FRAME
            count = random.randint(2, 3)
            sides = random.sample(["top", "right", "bottom", "left"], count)
            shapes = []
            for side in sides:
                if side in ("top", "bottom"):
                    length = random.randint(2, 3)
                    a = random.randint(c0 + 1, c1 - length)
                    for c in range(a, a + length):
                        g[r0 if side == "top" else r1, c] = BLACK
                    shape = [(0, i) for i in range(length)]
                else:
                    length = random.randint(2, 3)
                    a = random.randint(r0 + 1, r1 - length)
                    for r in range(a, a + length):
                        g[r, c0 if side == "left" else c1] = BLACK
                    shape = [(i, 0) for i in range(length)]
                shapes.append(shape)
            for shape in shapes:
                placed = False
                for _ in range(50):
                    pr = random.randint(r0 + 1, r1 - 3)
                    pc = random.randint(c0 + 1, c1 - 3)
                    pts = {(pr + dr, pc + dc) for dr, dc in shape}
                    if all(r0 < r < r1 and c0 < c < c1 and g[r, c] == BLACK for r, c in pts):
                        for r, c in pts:
                            g[r, c] = SEED
                        placed = True
                        break
                if not placed:
                    break
            return g
    return g
