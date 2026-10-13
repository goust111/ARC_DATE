from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
OBJECT_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in points]
    cs = [c for _r, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep each object's upper half in place, then mirror its lower half horizontally and translate it downward.

    Input structure:
    - Black is the background.
    - Each foreground object is single-color and 4-connected.
    - Every object's bounding-box height is even, so it can be split into equal upper and lower halves.

    Rule:
    - Split each object across the horizontal midpoint of its bounding box.
    - Copy upper-half pixels unchanged.
    - Erase the original lower half, mirror those lower-half pixels left-to-right within the bounding box, and translate
      them downward by the upper-half height, clipping at grid edges.

    Key insight: The bounding box midpoint anchors a two-step transform: reflect the lower half, then drop it by half height.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BLACK)
    for comp in _components(g != BLACK):
        minr, minc, maxr, maxc = _bbox(comp)
        height = maxr - minr + 1
        if height % 2 != 0:
            continue
        half = height // 2
        color = int(g[comp[0]])
        split = minr + half
        for r, c in comp:
            if r < split:
                out[r, c] = color
            else:
                nr = r + half
                nc = minc + maxc - c
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = color
    return out


def _random_mask(height: int, width: int) -> np.ndarray:
    half = height // 2
    mask = np.zeros((height, width), dtype=bool)
    mask[:half, :] = True
    left, right = 0, width - 1
    for r in range(half, height):
        if r == half:
            left, right = 0, width - 1
        else:
            left = random.randint(0, max(0, right - 1))
            right = random.randint(max(left + 1, width // 2), width - 1)
        mask[r, left : right + 1] = True
    if width >= 4 and height >= 6:
        mask[height - 1, random.choice([0, width - 1])] = False
    return mask


def _can_place(occupied: np.ndarray, r0: int, c0: int, hh: int, ww: int, extra_down: int) -> bool:
    h, w = occupied.shape
    r0p = max(0, r0 - 1)
    c0p = max(0, c0 - 1)
    r1p = min(h, r0 + hh + extra_down + 1)
    c1p = min(w, c0 + ww + 1)
    return not bool(occupied[r0p:r1p, c0p:c1p].any())


def generate_input() -> np.ndarray:
    rows = random.randint(16, 28)
    cols = random.randint(14, 28)
    grid = np.full((rows, cols), BLACK, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)
    colors = random.sample(OBJECT_COLORS, k=random.randint(2, 4))

    for color in colors:
        height = random.choice([4, 6, 8])
        width = random.randint(3, 7)
        half = height // 2
        if rows - height - half <= 0 or cols - width <= 0:
            continue
        mask = _random_mask(height, width)
        for _attempt in range(100):
            r0 = random.randint(0, rows - height - half)
            c0 = random.randint(0, cols - width)
            if not _can_place(occupied, r0, c0, height, width, half):
                continue
            sub = grid[r0 : r0 + height, c0 : c0 + width]
            sub[mask] = int(color)
            occupied[r0 : r0 + height + half, c0 : c0 + width] = True
            break
    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
