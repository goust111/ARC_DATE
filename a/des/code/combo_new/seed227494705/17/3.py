from collections import deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _frames(grid):
    out = []
    for cells in _components(grid == GREY):
        box = _bbox(cells)
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
            out.append(box)
    return out


def _lens_colors(grid):
    mapping = {}
    for r, c in map(tuple, np.argwhere(grid == GREEN)):
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                v = int(grid[nr, nc])
                if v not in (BLACK, RED, GREEN, GREY):
                    mapping[(r, c)] = v
                    break
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Fill target rectangles with the final color of the first ray that reaches them.

    Input structure: Black background contains red signal towers, green lens
    pixels each tagged by an adjacent filter-color pixel, and hollow grey target
    rectangles.
    Rule: Each tower casts invisible orthogonal rays initially colored red.  A ray
    that crosses a green lens changes to that lens's tag color and continues.  On
    hitting a grey target border it stops; the target interior is filled with the
    ray color from the shortest tower-to-hit distance among all hits.
    Key insight: The output color is not drawn along the ray; it is the ray state
    after all lens color changes, with distance breaking multiple hits.
    """
    out = grid.copy()
    frames = _frames(grid)
    lenses = _lens_colors(grid)
    best = {}
    for tr, tc in map(tuple, np.argwhere(grid == RED)):
        for dr, dc in DIRS:
            color = RED
            r, c, dist = tr + dr, tc + dc, 1
            while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                v = int(grid[r, c])
                if v == GREEN and (r, c) in lenses:
                    color = lenses[(r, c)]
                if v == GREY:
                    for box in frames:
                        r0, c0, r1, c1 = box
                        if r0 <= r <= r1 and c0 <= c <= c1:
                            if box not in best or dist < best[box][0]:
                                best[box] = (dist, color)
                            break
                    break
                r += dr
                c += dc
                dist += 1
    for (r0, c0, r1, c1), (_, color) in best.items():
        out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _draw_frame(grid, r0, c0, h, w):
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 28), random.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)
    tower = (h // 2, w // 2)
    grid[tower] = RED
    colors = random.sample([2, 4, 6, 7, 8, 9], 4)
    targets = [
        ("up", (2, tower[1] - 2), (-1, 0)),
        ("down", (h - 6, tower[1] - 2), (1, 0)),
        ("left", (tower[0] - 2, 2), (0, -1)),
        ("right", (tower[0] - 2, w - 6), (0, 1)),
    ]
    for i, (_, pos, vec) in enumerate(targets):
        if random.random() < 0.75:
            _draw_frame(grid, pos[0], pos[1], 4, 4)
            lr = tower[0] + vec[0] * random.randint(2, 4)
            lc = tower[1] + vec[1] * random.randint(2, 4)
            if grid[lr, lc] == BLACK:
                grid[lr, lc] = GREEN
                if vec[0] == 0:
                    grid[lr - 1, lc] = colors[i]
                else:
                    grid[lr, lc + 1] = colors[i]
    return grid
