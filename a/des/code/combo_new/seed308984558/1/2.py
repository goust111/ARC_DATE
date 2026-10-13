import random

import numpy as np


BG = 0
HIT = 2


def _arrow_dir(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    row_counts = {r: rows.count(r) for r in set(rows)}
    col_counts = {c: cols.count(c) for c in set(cols)}
    if row_counts[min(row_counts)] == 3:
        return "up", (min(row_counts), int(np.median(cols)))
    if row_counts[max(row_counts)] == 3:
        return "down", (max(row_counts), int(np.median(cols)))
    if col_counts[min(col_counts)] == 3:
        return "left", (int(np.median(rows)), min(col_counts))
    return "right", (int(np.median(rows)), max(col_counts))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend fixed-length rays from each 3x3 arrow and mark ray collisions in red.

    Input structure:
    - A black background contains several disconnected monochrome arrow sprites.
    - Every arrow occupies a 3x3 box and points up, down, left, or right.
    - Arrow colors are distinct and no arrow touches another arrow.

    Rule:
    - Keep every arrow unchanged.
    - From the tip-center pixel, draw a same-colored ray in the arrow direction with length equal to the arrow area.
    - Truncate the ray at the grid edge.
    - Any cell hit by two different rays becomes red.

    Key insight:
    - The arrow geometry provides both the direction and the length source, and collisions override the normal ray color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    ray_colors = {}
    for color in [v for v in np.unique(g) if v != BG]:
        coords = list(zip(*np.where(g == int(color)), strict=False))
        if len(coords) != 5:
            continue
        direction, tip = _arrow_dir(coords)
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
        r, c = tip
        for k in range(1, 6):
            rr, cc = r + dr * k, c + dc * k
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                break
            ray_colors.setdefault((rr, cc), set()).add(int(color))
    for (r, c), colors in ray_colors.items():
        out[r, c] = HIT if len(colors) > 1 else next(iter(colors))
    return out


def _arrow_mask(direction):
    base = np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int)
    return {"right": base, "down": np.rot90(base, 3), "left": np.rot90(base, 2), "up": np.rot90(base, 1)}[direction]


def generate_input() -> np.ndarray:
    h, w = random.randint(14, 24), random.randint(14, 24)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
    for color in colors:
        direction = random.choice(["up", "down", "left", "right"])
        mask = _arrow_mask(direction)
        for _ in range(200):
            r0 = random.randint(1, h - 4)
            c0 = random.randint(1, w - 4)
            if np.any(g[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4]):
                continue
            dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
            tip = {
                "up": (r0, c0 + 1),
                "down": (r0 + 2, c0 + 1),
                "left": (r0 + 1, c0),
                "right": (r0 + 1, c0 + 2),
            }[direction]
            rr, cc = tip
            if not (0 <= rr + dr * 5 < h and 0 <= cc + dc * 5 < w):
                continue
            for r in range(3):
                for c in range(3):
                    if mask[r, c]:
                        g[r0 + r, c0 + c] = color
            break
    return g
