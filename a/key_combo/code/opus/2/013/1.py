import numpy as np
from collections import Counter

RNG = np.random.default_rng(101)


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            stack = [(r, c)]
            vis[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Task 1: scan-hull filling by overlap depth of hollow rectangles.

Input:
    Colored hollow rectangles with colors 1..5 and magenta(6) distractors.
Output:
    All non-border non-distractor pixels recolored by max covering depth.

Reasoning:
    Detect rectangles from color components, assign depth by bbox area desc.
    For each row maintain active rectangles by column events and fill with depth color.

Complexity:
    O(H*W*N), N=#rectangles.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    rects = []
    for color in (1, 2, 3, 4, 5):
        for comp in _components(g == color):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            top, bottom = int(min(rs)), int(max(rs))
            left, right = int(min(cs)), int(max(cs))
            area = (bottom - top + 1) * (right - left + 1)
            rects.append((top, bottom, left, right, area))

    if not rects:
        return out

    rects.sort(key=lambda x: x[4], reverse=True)
    for i, rect in enumerate(rects):
        pass

    colors_set = {1, 2, 3, 4, 5}
    contour = np.isin(g, list(colors_set))
    dist_map = np.zeros((h, w), dtype=int)

    # assign depth by area order
    depth_for_pos = {}
    for i, (top, bottom, left, right, _area) in enumerate(rects):
        d = i + 1
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                depth_for_pos.setdefault((r, c), []).append(d)

    for row in range(h):
        starts = [[] for _ in range(w + 1)]
        ends = [[] for _ in range(w + 1)]
        for i, (top, bottom, left, right, _area) in enumerate(rects):
            d = i + 1
            if top < row < bottom:
                if 0 <= left + 1 < w + 1:
                    starts[left + 1].append(d)
                if 0 <= right < w + 1:
                    ends[right].append(d)

        active = Counter()
        for c in range(w):
            for d in starts[c]:
                active[d] += 1
            for d in ends[c]:
                if d in active:
                    active[d] -= 1
                    if active[d] <= 0:
                        del active[d]

            if contour[r, c] or g[r, c] == 6:
                continue

            if active:
                d = max(active)
                out[r, c] = d

    return out


def _rect_intersects(r1, b1, l1, rr1, r2, b2, l2, rr2):
    return not (rr1 < r2 or r1 > b2 or rr2 < l1 or l1 > b2)


def generate_input() -> np.ndarray:
    h = 18
    w = 18
    g = np.zeros((h, w), dtype=int)

    n = int(RNG.integers(3, 6))
    colors = [1, 2, 3, 4, 5][:n]
    chosen = []

    for color in colors:
        placed = False
        for _ in range(800):
            rh = int(RNG.integers(4, 11))
            rw = int(RNG.integers(4, 11))
            top = int(RNG.integers(1, h - rh))
            left = int(RNG.integers(1, w - rw))
            bottom = top + rh - 1
            right = left + rw - 1

            # no complete containment
            if any((top >= t and bottom <= b and left >= l and right <= r) or (t >= top and b <= bottom and l >= left and r <= right) for t, b, l, r, _ in chosen):
                continue

            if chosen:
                touch = False
                for t, b, l, r, _ in chosen:
                    if not (bottom < t or top > b or right < l or left > r):
                        touch = True
                        break
                if not touch:
                    continue

            chosen.append((top, bottom, left, right, color))
            placed = True
            break

        if not placed:
            top = 2
            rw = 6
            rh = 6
            left = int(RNG.integers(1, w - rw))
            bottom = top + rh - 1
            right = left + rw - 1
            chosen.append((top, bottom, left, right, color))

    # draw larger to smaller -> smaller on top
    for top, bottom, left, right, color in sorted(chosen, key=lambda x: (x[1] - x[0] + 1) * (x[3] - x[2] + 1)):
        g[top, left:right + 1] = color
        g[bottom, left:right + 1] = color
        g[top:bottom + 1, left] = color
        g[top:bottom + 1, right] = color

    # add distractors
    empt = [(r, c) for r in range(h) for c in range(w) if g[r, c] == 0]
    if len(empt) >= 3:
        for idx in RNG.choice(len(empt), size=3, replace=False):
            r, c = empt[int(idx)]
            g[r, c] = 6

    return g
