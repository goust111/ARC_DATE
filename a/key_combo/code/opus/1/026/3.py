import math
import random
import numpy as np


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    anchor_pos = np.argwhere(grid == 5)[0]
    ar, ac = int(anchor_pos[0]), int(anchor_pos[1])
    sprite = [(int(r), int(c)) for r, c in np.argwhere(grid == 2)]
    markers = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in (0, 2, 5):
                angle = math.atan2(c - ac, -(r - ar)) % (2 * math.pi)
                markers.append((angle, r, c, color))
    markers.sort()

    cells = sprite
    final_color = 2
    for _, tr, tc, color in markers:
        cr = _round_half_up(sum(r for r, _ in cells) / len(cells))
        cc = _round_half_up(sum(c for _, c in cells) / len(cells))
        dr, dc = tr - cr, tc - cc
        cells = [(r + dr, c + dc) for r, c in cells]
        final_color = color

    out = np.zeros_like(grid)
    out[ar, ac] = 5
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = final_color
    return out


def _shape(n: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.add((r + dr, c + dc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def generate_input() -> np.ndarray:
    marker_colors = [1, 3, 4, 6, 7, 9]
    for _ in range(500):
        h = w = random.randint(12, 15)
        grid = np.zeros((h, w), dtype=int)
        ar, ac = random.randint(3, h - 4), random.randint(3, w - 4)
        grid[ar, ac] = 5
        shape = _shape(random.randint(3, 5))
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        sr = random.randint(1, h - sh - 2)
        sc = random.randint(1, w - sw - 2)
        sprite = [(sr + r, sc + c) for r, c in shape]
        if (ar, ac) in sprite:
            continue
        for r, c in sprite:
            grid[r, c] = 2

        centroid_r = _round_half_up(sum(r for r, _ in sprite) / len(sprite))
        centroid_c = _round_half_up(sum(c for _, c in sprite) / len(sprite))
        count = random.randint(4, 6)
        colors = random.sample(marker_colors, count)
        used_dirs = set()
        occupied = {(ar, ac), *sprite}
        markers = []
        candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        random.shuffle(candidates)
        for r, c in candidates:
            if len(markers) == count:
                break
            if (r, c) in occupied or (r, c) == (ar, ac):
                continue
            dr, dc = r - ar, c - ac
            g = math.gcd(abs(dr), abs(dc))
            key = (dr // g, dc // g)
            if key in used_dirs:
                continue
            tr, tc = r - centroid_r, c - centroid_c
            moved = [(rr + tr, cc + tc) for rr, cc in sprite]
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in moved):
                continue
            if (ar, ac) in moved:
                continue
            used_dirs.add(key)
            occupied.add((r, c))
            markers.append((r, c))
        if len(markers) == count:
            for (r, c), color in zip(markers, colors):
                grid[r, c] = color
            return grid
    return grid
