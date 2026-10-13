import random
import numpy as np


BG = 0
GREY = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _missing_side(grid, r0, c0, r1, c1):
    border = []
    for c in range(c0, c1 + 1):
        border.append((r0, c, "top"))
        border.append((r1, c, "bottom"))
    for r in range(r0 + 1, r1):
        border.append((r, c0, "left"))
        border.append((r, c1, "right"))
    miss = [side for r, c, side in border if grid[r, c] == BG]
    return miss[0] if len(miss) == 1 else None


def main(grid: np.ndarray) -> np.ndarray:
    """Fill grey frames from a left legend whose notch corner encodes direction.

    Input structure: The left side contains legend entries made of a 3x3 color
    square and an adjacent 2x2 grey square with one missing corner. The right
    side contains grey hollow rectangular frames with one missing side-midpoint.
    Rule: Map frame side to legend corner by the clockwise code left->upper
    left, top->upper right, right->lower right, bottom->lower left. Fill each
    frame interior with the color square from the matching legend entry.
    Key insight: A corner notch in the legend and a side notch in the frame are
    two visual direction codes connected by a fixed clockwise correspondence.
    """
    out = grid.copy()
    legend = {}
    for col in [int(x) for x in np.unique(grid) if x not in (BG, GREY)]:
        for comp in _components(grid == col):
            r0, c0, r1, c1 = _bbox(comp)
            if c0 <= 2 and r1 - r0 == 2 and c1 - c0 == 2 and len(comp) == 9:
                gr, gc = r0, c1 + 1
                if gc + 1 < grid.shape[1]:
                    corners = {(0, 0): "ul", (0, 1): "ur", (1, 0): "ll", (1, 1): "lr"}
                    for (dr, dc), name in corners.items():
                        block = grid[gr:gr + 2, gc:gc + 2]
                        if block.shape == (2, 2) and block[dr, dc] == BG and np.count_nonzero(block == GREY) == 3:
                            legend[name] = col
                break
    side_to_corner = {"left": "ul", "top": "ur", "right": "lr", "bottom": "ll"}
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        side = _missing_side(grid, r0, c0, r1, c1)
        if side and side_to_corner[side] in legend:
            fill = legend[side_to_corner[side]]
            out[r0 + 1:r1, c0 + 1:c1] = fill
    return out


def generate_input() -> np.ndarray:
    h, w = 24, 24
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    corners = {"ul": (0, 0), "ur": (0, 1), "lr": (1, 1), "ll": (1, 0)}
    for idx, (name, miss) in enumerate(corners.items()):
        r = 1 + idx * 5
        col = colors[idx]
        grid[r:r + 3, 0:3] = col
        grid[r:r + 2, 3:5] = GREY
        grid[r + miss[0], 3 + miss[1]] = BG
    frames = [(2, 9), (2, 16), (12, 9), (13, 16)]
    sides = ["left", "top", "right", "bottom"]
    for r0, c0 in frames[: random.randint(2, 4)]:
        rh, rw = random.randint(5, 7), random.randint(5, 7)
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        grid[r0, c0:c1 + 1] = GREY
        grid[r1, c0:c1 + 1] = GREY
        grid[r0:r1 + 1, c0] = GREY
        grid[r0:r1 + 1, c1] = GREY
        side = random.choice(sides)
        if side == "left":
            grid[(r0 + r1) // 2, c0] = BG
        elif side == "right":
            grid[(r0 + r1) // 2, c1] = BG
        elif side == "top":
            grid[r0, (c0 + c1) // 2] = BG
        else:
            grid[r1, (c0 + c1) // 2] = BG
    return grid
