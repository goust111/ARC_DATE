import random
from collections import deque

import numpy as np


PAIR = {3: 6, 8: 4, 2: 1}


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _rect_bbox(grid):
    comp = max(_components(grid == 4), key=len)
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _landing(r, c, box):
    r0, c0, r1, c1 = box
    if r0 <= r <= r1:
        if c < c0:
            return c0 - c, (r, c0), (r, c0 - 1)
        return c - c1, (r, c1), (r, c1 + 1)
    if c0 <= c <= c1:
        if r < r0:
            return r0 - r, (r0, c), (r0 - 1, c)
        return r - r1, (r1, c), (r1 + 1, c)
    raise ValueError("marker is not row- or column-aligned with the rectangle")


def main(grid: np.ndarray) -> np.ndarray:
    """Slide aligned markers to the yellow rectangle, recoloring the nearest entrant and outlining the rest."""
    out = grid.copy()
    box = _rect_bbox(grid)
    markers = []
    for r, c in np.argwhere(np.isin(grid, list(PAIR))):
        r, c = int(r), int(c)
        color = int(grid[r, c])
        dist, inside, outside = _landing(r, c, box)
        markers.append((dist, r, c, color, inside, outside))
        out[r, c] = 0
    entrant = min(markers, key=lambda x: x[0])
    er, ec = entrant[4]
    fill = PAIR[entrant[3]]
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = fill
    out[er, ec] = fill
    h, w = grid.shape
    for marker in markers:
        if marker is entrant:
            continue
        _, _, _, color, _, (lr, lc) = marker
        outline = PAIR[color]
        for rr in range(lr - 1, lr + 2):
            for cc in range(lc - 1, lc + 2):
                if 0 <= rr < h and 0 <= cc < w and (rr != lr or cc != lc):
                    out[rr, cc] = outline
        if 0 <= lr < h and 0 <= lc < w:
            out[lr, lc] = color
    return out


def generate_input() -> np.ndarray:
    h = w = 13
    grid = np.zeros((h, w), dtype=int)
    rh, rw = random.choice([(3, 4), (4, 3)])
    r0 = random.randint(4, 6)
    c0 = random.randint(4, 6)
    grid[r0 : r0 + rh, c0 : c0 + rw] = 4
    box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
    sides = ["left", "right", "top", "bottom"]
    random.shuffle(sides)
    colors = [3, 8, 2, random.choice([3, 8, 2])]
    used = set()
    for side, color in zip(sides, colors):
        for _ in range(50):
            if side == "left":
                r = random.randint(box[0], box[2])
                c = random.randint(0, box[1] - 3)
            elif side == "right":
                r = random.randint(box[0], box[2])
                c = random.randint(box[3] + 3, w - 1)
            elif side == "top":
                r = random.randint(0, box[0] - 3)
                c = random.randint(box[1], box[3])
            else:
                r = random.randint(box[2] + 3, h - 1)
                c = random.randint(box[1], box[3])
            _, _, land = _landing(r, c, box)
            if land not in used:
                used.add(land)
                grid[r, c] = color
                break
    return grid


if __name__ == "__main__":
    for _ in range(100):
        result = main(generate_input())
        assert isinstance(result, np.ndarray)
