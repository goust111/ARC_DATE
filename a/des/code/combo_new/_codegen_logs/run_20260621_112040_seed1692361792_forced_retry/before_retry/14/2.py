import random
from collections import deque

import numpy as np

AZURE = 8
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize each color's directed line-segment chain as a single sorted color bar.

    Input structure:
    - Azure background with scattered monochrome horizontal or vertical line segments.
    - Each segment has one maroon marker one step beyond its head endpoint.
    - Every non-maroon color appears in several segments belonging to the same chain group.

    Rule:
    - Group the line segments by color and read each segment length.
    - Within a color group, the maroon markers define the head endpoints, while the path assembly order is
      the segment lengths from short to long.
    - The output is an N-by-1 column, one cell per color group, sorted from largest total chain length to
      smallest total chain length.
    - Each output cell is painted with that group's segment color, and the maroon markers disappear.

    Key insight:
    - The maroon pixel gives directionality to each segment, but the final output depends on aggregating
      the per-color lengths and then sorting the groups by their total path lengths.
    """
    totals = []
    for color in np.unique(grid):
        if color in (AZURE, MAROON):
            continue
        total = 0
        for comp in _components(grid == color):
            total += len(comp)
        totals.append((total, int(color)))
    totals.sort(key=lambda item: (-item[0], item[1]))
    return np.array([[color] for _, color in totals], dtype=int)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        h = rng.randint(18, 26)
        w = rng.randint(18, 26)
        grid = np.full((h, w), AZURE, dtype=int)
        used = np.zeros((h, w), dtype=bool)

        num_colors = rng.randint(2, 4)
        colors = rng.sample(PALETTE, num_colors)
        totals = []
        specs = []
        ok = True

        for color in colors:
            seg_count = rng.randint(2, 4)
            lengths = [rng.randint(2, 6) for _ in range(seg_count)]
            total = sum(lengths)
            totals.append(total)
            specs.append((color, lengths))
        if len(set(totals)) != len(totals):
            continue

        for color, lengths in specs:
            for length in lengths:
                placed = False
                for _ in range(200):
                    horiz = rng.choice([True, False])
                    head_at_start = rng.choice([True, False])
                    if horiz:
                        r = rng.randint(1, h - 2)
                        c = rng.randint(1, w - length - 2)
                        cells = [(r, c + i) for i in range(length)]
                        marker = (r, c - 1) if head_at_start else (r, c + length)
                    else:
                        r = rng.randint(1, h - length - 2)
                        c = rng.randint(1, w - 2)
                        cells = [(r + i, c) for i in range(length)]
                        marker = (r - 1, c) if head_at_start else (r + length, c)

                    rr0 = max(0, min(r for r, _ in cells + [marker]) - 1)
                    rr1 = min(h, max(r for r, _ in cells + [marker]) + 2)
                    cc0 = max(0, min(c for _, c in cells + [marker]) - 1)
                    cc1 = min(w, max(c for _, c in cells + [marker]) + 2)
                    if used[rr0:rr1, cc0:cc1].any():
                        continue
                    for rr, cc in cells:
                        grid[rr, cc] = color
                    grid[marker] = MAROON
                    used[rr0:rr1, cc0:cc1] = True
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
