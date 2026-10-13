import random
from collections import deque

import numpy as np


BG = 0
ARROW = 2


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


def _dir(cells):
    s = set(cells)
    for r, c in cells:
        if {(r + 1, c - 1), (r + 1, c + 1)} <= s:
            return "up"
        if {(r - 1, c - 1), (r - 1, c + 1)} <= s:
            return "down"
        if {(r - 1, c + 1), (r + 1, c + 1)} <= s:
            return "left"
        if {(r - 1, c - 1), (r + 1, c - 1)} <= s:
            return "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each object inside its own bbox using the axis encoded by the paired red arrow direction.

    Input structure:
    - A black background contains several colored objects and separate 3-pixel red arrows.
    - Vertical arrows pair with objects whose column span overlaps the arrow.
    - Horizontal arrows pair with objects whose row span overlaps the arrow.

    Rule:
    - Up/down arrows trigger a vertical flip inside the object's bbox.
    - Left/right arrows trigger a horizontal flip inside the object's bbox.
    - Keep the object in place and erase the arrows.

    Key insight:
    - The same arrow direction determines both how pairing is resolved and which mirror axis to apply.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    objs = []
    for color in [v for v in np.unique(g) if v not in (BG, ARROW)]:
        for cells in _components(g == int(color)):
            objs.append((int(color), cells))
    for cells in _components(g == ARROW):
        d = _dir(cells)
        r0, c0, r1, c1 = _bbox(cells)
        target = None
        for color, ocells in objs:
            a0, b0, a1, b1 = _bbox(ocells)
            if d in ("up", "down") and not (b1 < c0 or b0 > c1):
                target = (color, ocells)
            if d in ("left", "right") and not (a1 < r0 or a0 > r1):
                target = (color, ocells)
        for r, c in cells:
            out[r, c] = BG
        if target is None:
            continue
        color, ocells = target
        a0, b0, a1, b1 = _bbox(ocells)
        box = np.zeros((a1 - a0 + 1, b1 - b0 + 1), dtype=int)
        for r, c in ocells:
            box[r - a0, c - b0] = color
            out[r, c] = BG
        box = np.flipud(box) if d in ("up", "down") else np.fliplr(box)
        out[a0 : a1 + 1, b0 : b1 + 1][box != 0] = color
    return out


ARROWS = {
    "up": [(0, 1), (1, 0), (1, 2)],
    "down": [(1, 1), (0, 0), (0, 2)],
    "left": [(1, 0), (0, 1), (2, 1)],
    "right": [(1, 1), (0, 0), (2, 0)],
}
SHAPES = [np.array([[1, 1], [1, 0]]), np.array([[1, 1, 1], [0, 1, 0]])]


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    for color in random.sample([1, 3, 4, 5, 6, 7, 8], random.randint(2, 4)):
        shape = random.choice(SHAPES)
        sh, sw = shape.shape
        for _ in range(200):
            r0 = random.randint(2, h - sh - 3)
            c0 = random.randint(2, w - sw - 3)
            if np.any(g[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1]):
                continue
            g[r0 : r0 + sh, c0 : c0 + sw][shape == 1] = color
            direction = random.choice(list(ARROWS))
            if direction in ("up", "down"):
                ar = r0 - 2 if direction == "up" else r0 + sh
                ac = c0
            else:
                ar = r0
                ac = c0 - 2 if direction == "left" else c0 + sw
            if ar < 0 or ac < 0 or ar + 3 > h or ac + 3 > w:
                continue
            for dr, dc in ARROWS[direction]:
                g[ar + dr, ac + dc] = ARROW
            break
    return g
