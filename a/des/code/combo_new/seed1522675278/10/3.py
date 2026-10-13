import random
import numpy as np


BG = 0
GREY = 5
OBS = 7


ARROWS = {
    "r": [(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)],
    "l": [(1, 2), (1, 1), (1, 0), (0, 0), (2, 0)],
    "u": [(2, 1), (1, 1), (0, 1), (0, 0), (0, 2)],
    "d": [(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)],
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each tail-attached colored bar along its grey arrow and keep the whole swept strip.

    Input structure:
    - Grey arrows indicate one of four cardinal directions.
    - A colored bar touches each arrow on its tail side.
    - Orange obstacle pixels block extension.

    Rule:
    - Determine the arrow direction and remove both the arrow and the original bar.
    - Starting just beyond the arrow tip, extend a same-colored strip along that direction
      until the next step would hit an obstacle or leave the grid.
    - Keep the obstacles unchanged.

    Key insight:
    - The moved payload is the bar attached to the arrow tail, while the output is the
      entire path it sweeps out, not only the endpoint.
    """
    out = grid.copy()
    for d, pts in ARROWS.items():
        pass
    # generator uses one arrow-bar pair per color and 1-cell-thick bars, so scan colors
    for color in [int(c) for c in np.unique(grid) if c not in (BG, GREY, OBS)]:
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        # find nearby arrow
        search = np.argwhere(grid == GREY)
        sr, sc = search.min(axis=0)
        # robust enough for this generator by checking nearest tip direction around the bar
        if r0 > 3 and np.any(grid[r0 - 3 : r0, c0 : c1 + 1] == GREY):
            d = "d"
            row = r0
            while row < grid.shape[0] and np.all(grid[row, c0 : c1 + 1] != OBS):
                out[row, c0 : c1 + 1] = color
                row += 1
        elif r1 < grid.shape[0] - 4 and np.any(grid[r1 + 1 : r1 + 4, c0 : c1 + 1] == GREY):
            d = "u"
            row = r1
            while row >= 0 and np.all(grid[row, c0 : c1 + 1] != OBS):
                out[row, c0 : c1 + 1] = color
                row -= 1
        elif c0 > 3 and np.any(grid[r0 : r1 + 1, c0 - 3 : c0] == GREY):
            col = c0
            while col < grid.shape[1] and np.all(grid[r0 : r1 + 1, col] != OBS):
                out[r0 : r1 + 1, col] = color
                col += 1
        else:
            col = c1
            while col >= 0 and np.all(grid[r0 : r1 + 1, col] != OBS):
                out[r0 : r1 + 1, col] = color
                col -= 1
    out[grid == GREY] = BG
    out[grid == OBS] = OBS
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 22), dtype=int)
    for _ in range(random.randint(4, 7)):
        grid[random.randint(1, 20), random.randint(1, 20)] = OBS
    colors = random.sample([1, 2, 3, 4, 6, 8, 9], random.randint(2, 4))
    for color in colors:
        d = random.choice(["r", "l", "u", "d"])
        if d in ("r", "l"):
            r = random.randint(4, 17)
            c = random.randint(6, 10)
            grid[r, c : c + random.randint(2, 4)] = color
            if d == "r":
                for dr, dc in ARROWS["r"]:
                    grid[r - 1 + dr, c - 3 + dc] = GREY
            else:
                w = np.count_nonzero(grid[r] == color)
                for dr, dc in ARROWS["l"]:
                    grid[r - 1 + dr, c + w + dc] = GREY
        else:
            r = random.randint(6, 10)
            c = random.randint(4, 17)
            grid[r : r + random.randint(2, 4), c] = color
            if d == "u":
                h = np.count_nonzero(grid[:, c] == color)
                for dr, dc in ARROWS["u"]:
                    grid[r + h + dr, c - 1 + dc] = GREY
            else:
                for dr, dc in ARROWS["d"]:
                    grid[r - 3 + dr, c - 1 + dc] = GREY
    return grid
