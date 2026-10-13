import random
import numpy as np


BG = 0
YELLOW = 4
ORANGE = 7


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shine light straight into each room from its window gaps and color beam crossings orange.

    Input structure:
    - Colored rectangular room borders sit on a black background.
    - Some single border pixels are missing, and a yellow source pixel sits just outside each gap.

    Rule:
    - For every gap, determine which wall it belongs to and cast a straight yellow beam
      inward until the opposite wall.
    - If multiple beams hit the same cell, recolor that cell orange.
    - Remove the outside yellow source pixels and keep the room borders unchanged.

    Key insight:
    - A window is defined jointly by the border gap and the outside source; the gap's wall
      orientation controls beam direction, while crossings need a separate merge rule.
    """
    out = grid.copy()
    out[out == YELLOW] = BG
    hit = {}
    for color in [int(c) for c in np.unique(grid) if c not in (BG, YELLOW)]:
        pts = np.argwhere(grid == color)
        if len(pts) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        for c in range(c0 + 1, c1):
            if grid[r0, c] != color and r0 > 0 and grid[r0 - 1, c] == YELLOW:
                for r in range(r0 + 1, r1):
                    hit[r, c] = hit.get((r, c), 0) + 1
            if grid[r1, c] != color and r1 < grid.shape[0] - 1 and grid[r1 + 1, c] == YELLOW:
                for r in range(r0 + 1, r1):
                    hit[r, c] = hit.get((r, c), 0) + 1
        for r in range(r0 + 1, r1):
            if grid[r, c0] != color and c0 > 0 and grid[r, c0 - 1] == YELLOW:
                for c in range(c0 + 1, c1):
                    hit[r, c] = hit.get((r, c), 0) + 1
            if grid[r, c1] != color and c1 < grid.shape[1] - 1 and grid[r, c1 + 1] == YELLOW:
                for c in range(c0 + 1, c1):
                    hit[r, c] = hit.get((r, c), 0) + 1
    for (r, c), n in hit.items():
        out[r, c] = ORANGE if n > 1 else YELLOW
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 6, 8, 9], random.randint(2, 4))
    used = []
    for color in colors:
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        for _ in range(200):
            r = random.randint(2, 24 - h - 2)
            c = random.randint(2, 24 - w - 2)
            if all(r + h + 2 <= rr or rr + hh + 2 <= r or c + w + 2 <= cc or cc + ww + 2 <= c for rr, cc, hh, ww in used):
                used.append((r, c, h, w))
                grid[r, c : c + w] = color
                grid[r + h - 1, c : c + w] = color
                grid[r : r + h, c] = color
                grid[r : r + h, c + w - 1] = color
                for side in random.sample(["u", "d", "l", "r"], random.randint(1, 3)):
                    if side == "u":
                        cc = random.randint(c + 1, c + w - 2)
                        grid[r, cc] = BG
                        grid[r - 1, cc] = YELLOW
                    elif side == "d":
                        cc = random.randint(c + 1, c + w - 2)
                        grid[r + h - 1, cc] = BG
                        grid[r + h, cc] = YELLOW
                    elif side == "l":
                        rr = random.randint(r + 1, r + h - 2)
                        grid[rr, c] = BG
                        grid[rr, c - 1] = YELLOW
                    else:
                        rr = random.randint(r + 1, r + h - 2)
                        grid[rr, c + w - 1] = BG
                        grid[rr, c + w] = YELLOW
                break
    return grid
