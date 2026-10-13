import numpy as np


BLACK = 0
GREY = 5
MAROON = 9


def _clockwise(dr, dc):
    return dc, -dr


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each interior color straight across the corridor, with maroon wall hits reflecting the direction clockwise once.

    Input structure:
    - A single grey rectangular corridor encloses black interior cells.
    - Some wall pixels are maroon mirrors instead of ordinary grey wall pixels.
    - Colored singleton pixels sit one cell inside a wall, so each source is adjacent to exactly one wall.

    Rule:
    - For each colored source, take the inward direction perpendicular to the adjacent wall.
    - Advance through black cells until the ray reaches the opposite wall.
    - If that wall pixel is grey, place a same-colored copy on the last interior cell before the wall; if it is maroon, rotate the travel direction 90 degrees clockwise and continue from that interior cell until the next wall.
    - Keep the original colored sources and leave the path itself unpainted.

    Key insight:
    - Maroon wall pixels are not just another wall color: they change the beam direction, so predicting the copy location requires simulating the wall interaction.
    """
    out = grid.copy()
    for r, c in np.argwhere((grid != BLACK) & (grid != GREY) & (grid != MAROON)):
        color = int(grid[r, c])
        if grid[r - 1, c] in (GREY, MAROON):
            dr, dc = 1, 0
        elif grid[r + 1, c] in (GREY, MAROON):
            dr, dc = -1, 0
        elif grid[r, c - 1] in (GREY, MAROON):
            dr, dc = 0, 1
        else:
            dr, dc = 0, -1
        cr, cc = r, c
        while True:
            nr, nc = cr + dr, cc + dc
            if grid[nr, nc] in (GREY, MAROON):
                if grid[nr, nc] == GREY:
                    out[cr, cc] = color
                    break
                dr, dc = _clockwise(dr, dc)
            else:
                cr, cc = nr, nc
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((24, 24), dtype=int)
    r0, c0, h, w = 3, 3, 18, 18
    grid[r0 : r0 + h, c0 : c0 + w] = GREY
    grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = BLACK
    mirror_cols = rng.choice(np.arange(c0 + 2, c0 + w - 2), size=2, replace=False)
    mirror_rows = rng.choice(np.arange(r0 + 2, r0 + h - 2), size=2, replace=False)
    grid[r0 + h - 1, mirror_cols[0]] = MAROON
    grid[r0, mirror_cols[1]] = MAROON
    grid[mirror_rows[0], c0 + w - 1] = MAROON
    grid[mirror_rows[1], c0] = MAROON
    colors = list(rng.choice([1, 2, 3, 4, 6, 7, 8], size=int(rng.integers(3, 6)), replace=False))
    placements = [
        (r0 + 1, int(mirror_cols[0]), "reflect"),
        (r0 + h - 2, int(mirror_cols[1]), "reflect"),
        (int(mirror_rows[0]), c0 + 1, "reflect"),
        (int(mirror_rows[1]), c0 + w - 2, "reflect"),
        (r0 + 1, c0 + 2, "direct"),
        (r0 + h - 2, c0 + w - 3, "direct"),
    ]
    rng.shuffle(placements)
    for color, (r, c, _) in zip(colors, placements):
        grid[r, c] = int(color)
    return grid
