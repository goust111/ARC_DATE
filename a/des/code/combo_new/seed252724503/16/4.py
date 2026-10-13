import random
from collections import deque

import numpy as np


OUTLINE = 1


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _interior_components(passable: np.ndarray):
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touch = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touch = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touch:
                parts.append(cells)
    return parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the first N interior layers of each closed outline, where N is the number of same-color seeds inside.

    Input structure:
    - The grid has a dominant background color and several blue closed outlines.
    - Inside each outline, black interior cells contain one to four detached seed pixels of a single non-blue color.
    - Different outlined regions may use different seed colors.

    Rule:
    - For each outlined region, count its seed pixels to obtain N and read the shared seed color.
    - Starting from the interior cells adjacent to the blue wall, expand inward by Manhattan layers.
    - Recolor the first N layers with the seed color, clipping if the region is shallower than N layers.
    - Keep the blue outline and the seed pixels themselves unchanged.

    Key insight:
    - The seeds encode two independent facts at once: their color is the fill color, and their count is the fill depth.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = g.copy()
    passable = g != OUTLINE
    for region in _interior_components(passable):
        seed_cells = [(r, c) for r, c in region if g[r, c] not in (bg, OUTLINE)]
        if not seed_cells:
            continue
        seed_color = int(g[seed_cells[0]])
        depth = len(seed_cells)
        region_set = set(region)
        dist = {}
        q = deque()
        for r, c in region:
            if any(
                0 <= r + dr < g.shape[0]
                and 0 <= c + dc < g.shape[1]
                and g[r + dr, c + dc] == OUTLINE
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                dist[(r, c)] = 1
                q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in region_set and (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))
        for (r, c), d in dist.items():
            if d <= depth and g[r, c] == bg:
                out[r, c] = seed_color
    return out


def _draw_rect(grid: np.ndarray, top: int, left: int, height: int, width: int):
    grid[top, left : left + width] = OUTLINE
    grid[top + height - 1, left : left + width] = OUTLINE
    grid[top : top + height, left] = OUTLINE
    grid[top : top + height, left + width - 1] = OUTLINE


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(16, 26), random.randint(16, 26)
        bg = 0
        grid = np.full((h, w), bg, dtype=int)
        occ = np.zeros((h, w), dtype=bool)
        colors = random.sample([2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
        ok = True
        for color in colors:
            placed = False
            for _ in range(100):
                hh, ww = random.randint(5, 9), random.randint(5, 9)
                top = random.randint(1, h - hh - 1)
                left = random.randint(1, w - ww - 1)
                if occ[top - 1 : top + hh + 1, left - 1 : left + ww + 1].any():
                    continue
                _draw_rect(grid, top, left, hh, ww)
                occ[top - 1 : top + hh + 1, left - 1 : left + ww + 1] = True
                seed_count = random.randint(1, 4)
                interior = [(r, c) for r in range(top + 1, top + hh - 1) for c in range(left + 1, left + ww - 1)]
                for r, c in random.sample(interior, seed_count):
                    grid[r, c] = color
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
