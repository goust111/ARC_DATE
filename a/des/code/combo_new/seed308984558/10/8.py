import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components_of_mask(mask):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each grey object across its adjacent colored mirror line and recolor it.

    Input structure:
    - Colored one-pixel-wide horizontal or vertical line segments act as mirrors.
    - Exactly one grey source object is flush against one side of each mirror.

    Rule:
    - Detect the mirror orientation and the side holding the grey source object.
    - Reflect the grey object across the mirror axis onto the empty side.
    - Paint the reflected copy with the mirror's color while leaving the grey source
      object and mirror unchanged.

    Key insight:
    - Each mirror line encodes both the geometric axis of reflection and the final
      color of the reflected copy.
    """
    out = grid.copy()
    for color in [int(v) for v in np.unique(grid) if v not in (BLACK, GREY)]:
        for comp in _components_of_mask(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
            if r0 == r1:
                axis = "h"
                upper = {(r0 - 1, c) for c in range(c0, c1 + 1) if r0 - 1 >= 0 and grid[r0 - 1, c] == GREY}
                lower = {(r0 + 1, c) for c in range(c0, c1 + 1) if r0 + 1 < grid.shape[0] and grid[r0 + 1, c] == GREY}
                start = next(iter(upper or lower), None)
                if start is None:
                    continue
                source = set()
                q = deque([start])
                source.add(start)
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == GREY and (nr, nc) not in source:
                            source.add((nr, nc))
                            q.append((nr, nc))
                for sr, sc in source:
                    rr = 2 * r0 - sr
                    if 0 <= rr < grid.shape[0] and grid[rr, sc] == BLACK:
                        out[rr, sc] = color
            else:
                start = next(((r, c0 - 1) for r in range(r0, r1 + 1) if c0 - 1 >= 0 and grid[r, c0 - 1] == GREY), None)
                if start is None:
                    start = next(((r, c0 + 1) for r in range(r0, r1 + 1) if c0 + 1 < grid.shape[1] and grid[r, c0 + 1] == GREY), None)
                if start is None:
                    continue
                source = set()
                q = deque([start])
                source.add(start)
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == GREY and (nr, nc) not in source:
                            source.add((nr, nc))
                            q.append((nr, nc))
                for sr, sc in source:
                    cc = 2 * c0 - sc
                    if 0 <= cc < grid.shape[1] and grid[sr, cc] == BLACK:
                        out[sr, cc] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample(COLORS, random.randint(2, 4)):
        if random.random() < 0.5:
            r = random.randint(3, h - 4)
            c0 = random.randint(2, w - 6)
            c1 = random.randint(c0 + 3, min(w - 3, c0 + 6))
            grid[r, c0 : c1 + 1] = color
            side = random.choice([-1, 1])
            patch = np.zeros((2, random.randint(2, 3)), dtype=int)
            patch[0, 0] = GREY
            patch[0, -1] = GREY
            patch[1, 1 if patch.shape[1] > 2 else 0] = GREY
            rr = r + side * patch.shape[0] if side < 0 else r + 1
            cc = random.randint(c0, c1 - patch.shape[1] + 1)
            grid[rr : rr + patch.shape[0], cc : cc + patch.shape[1]] = np.maximum(grid[rr : rr + patch.shape[0], cc : cc + patch.shape[1]], patch)
        else:
            c = random.randint(3, w - 4)
            r0 = random.randint(2, h - 6)
            r1 = random.randint(r0 + 3, min(h - 3, r0 + 6))
            grid[r0 : r1 + 1, c] = color
            side = random.choice([-1, 1])
            patch = np.zeros((random.randint(2, 3), 2), dtype=int)
            patch[0, 0] = GREY
            patch[-1, 0] = GREY
            patch[1 if patch.shape[0] > 2 else 0, 1] = GREY
            rr = random.randint(r0, r1 - patch.shape[0] + 1)
            cc = c - patch.shape[1] if side < 0 else c + 1
            grid[rr : rr + patch.shape[0], cc : cc + patch.shape[1]] = np.maximum(grid[rr : rr + patch.shape[0], cc : cc + patch.shape[1]], patch)
    return grid
