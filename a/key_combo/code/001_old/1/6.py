from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color pairs into walls and flood-fill each resulting region from its singleton seed.

    Input structure:
    - A black grid contains colors that appear exactly twice as aligned endpoint pairs.
    - Other non-black colors appear exactly once and act as seeds.
    - Pair segments do not cross seeds or each other.

    Rule:
    - Draw a full horizontal or vertical segment between every same-color aligned pair.
    - Treat those segments as walls and split the remaining cells into 4-connected regions.
    - Fill any non-wall region containing a singleton seed with that seed color; regions without seeds remain black.

    Key insight:
    Color multiplicity first assigns roles, then the pair walls create the regions that determine how singleton seeds spread.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    counts = Counter(int(v) for v in g.ravel() if int(v) != 0)
    wall_colors: set[int] = set()
    seeds: dict[tuple[int, int], int] = {}
    out = np.zeros_like(g)

    for color, count in counts.items():
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        if count == 2 and (pts[0][0] == pts[1][0] or pts[0][1] == pts[1][1]):
            wall_colors.add(color)
            (r0, c0), (r1, c1) = pts
            if r0 == r1:
                for c in range(min(c0, c1), max(c0, c1) + 1):
                    out[r0, c] = color
            else:
                for r in range(min(r0, r1), max(r0, r1) + 1):
                    out[r, c0] = color
        elif count == 1:
            seeds[pts[0]] = color

    wall_mask = out != 0
    seen = np.zeros((h, w), dtype=bool)
    for sr in range(h):
        for sc in range(w):
            if wall_mask[sr, sc] or seen[sr, sc]:
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            region: list[tuple[int, int]] = []
            seed_color = None
            seen[sr, sc] = True
            while q:
                r, c = q.popleft()
                region.append((r, c))
                if (r, c) in seeds:
                    seed_color = seeds[(r, c)]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and not wall_mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if seed_color is not None:
                for r, c in region:
                    out[r, c] = seed_color
    return out


def generate_input() -> np.ndarray:
    for _attempt in range(200):
        h = random.randint(10, 14)
        w = random.randint(10, 14)
        g = np.zeros((h, w), dtype=int)
        colors = random.sample(range(1, 10), 8)
        orientation = random.choice(["vertical", "horizontal"])
        wall_count = random.randint(2, 4)
        if orientation == "vertical":
            cols = sorted(random.sample(range(2, w - 2), wall_count))
            for i, c in enumerate(cols):
                g[0, c] = colors[i]
                g[h - 1, c] = colors[i]
            bands = []
            prev = 0
            for c in cols:
                bands.append((prev, c - 1))
                prev = c + 1
            bands.append((prev, w - 1))
            valid_bands = [(a, b) for a, b in bands if a <= b]
            seed_bands = random.sample(valid_bands, random.randint(1, min(3, len(valid_bands) - 1)))
            for j, (a, b) in enumerate(seed_bands):
                r = random.randint(1, h - 2)
                c = random.randint(max(0, a), min(w - 1, b))
                g[r, c] = colors[wall_count + j]
        else:
            rows = sorted(random.sample(range(2, h - 2), wall_count))
            for i, r in enumerate(rows):
                g[r, 0] = colors[i]
                g[r, w - 1] = colors[i]
            bands = []
            prev = 0
            for r in rows:
                bands.append((prev, r - 1))
                prev = r + 1
            bands.append((prev, h - 1))
            valid_bands = [(a, b) for a, b in bands if a <= b]
            seed_bands = random.sample(valid_bands, random.randint(1, min(3, len(valid_bands) - 1)))
            for j, (a, b) in enumerate(seed_bands):
                r = random.randint(max(0, a), min(h - 1, b))
                c = random.randint(1, w - 2)
                g[r, c] = colors[wall_count + j]
        return g
    raise RuntimeError("failed to generate puzzle input")
