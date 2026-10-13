import random
import numpy as np


BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each damaged 3x3 color block in the direction and distance encoded by its missing edge cells.

    Input structure:
    - The background is black.
    - Each object is a colored 3x3 block with 1 to 3 erased cells on exactly one edge.

    Rule:
    - Determine which edge contains the missing cells: that edge sets the slide direction.
    - The number of missing cells sets the slide distance.
    - Remove the old damaged block and draw a complete 3x3 block at the slid position,
      clamped to the grid boundary if needed.

    Key insight:
    - The defect carries two signals at once: its side determines direction and its
      size determines distance, while the output restores the full block.
    """
    out = grid.copy()
    colors = [int(c) for c in np.unique(grid) if c != BG]
    for color in colors:
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rr = slice(max(0, r0 - 1), min(grid.shape[0], r1 + 2))
        cc = slice(max(0, c0 - 1), min(grid.shape[1], c1 + 2))
        box = grid[rr, cc]
        best = None
        for tr in range(box.shape[0] - 2):
            for tc in range(box.shape[1] - 2):
                sub = box[tr : tr + 3, tc : tc + 3]
                hits = sub == color
                count = hits.sum()
                if 6 <= count <= 8:
                    miss = []
                    for i in range(3):
                        for j in range(3):
                            if not hits[i, j]:
                                miss.append((i, j))
                    rows = {i for i, _ in miss}
                    cols = {j for _, j in miss}
                    if len(rows) == 1 or len(cols) == 1:
                        best = (rr.start + tr, cc.start + tc, miss)
        br, bc, miss = best
        out[br : br + 3, bc : bc + 3] = BG
        if all(i == 0 for i, _ in miss):
            dr, dc = -len(miss), 0
        elif all(i == 2 for i, _ in miss):
            dr, dc = len(miss), 0
        elif all(j == 0 for _, j in miss):
            dr, dc = 0, -len(miss)
        else:
            dr, dc = 0, len(miss)
        nr = min(max(0, br + dr), grid.shape[0] - 3)
        nc = min(max(0, bc + dc), grid.shape[1] - 3)
        out[nr : nr + 3, nc : nc + 3] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((random.randint(16, 24), random.randint(16, 24)), dtype=int)
    colors = random.sample(range(1, 10), random.randint(3, 5))
    placed = []
    for color in colors:
        side = random.choice(["u", "d", "l", "r"])
        miss_n = random.randint(1, 3)
        miss_idx = random.sample(range(3), miss_n)
        for _ in range(200):
            r = random.randint(0, grid.shape[0] - 4)
            c = random.randint(0, grid.shape[1] - 4)
            bbox = (r, c, 3, 3)
            if all(r + 5 <= rr or rr + hh + 2 <= r or c + 5 <= cc or cc + ww + 2 <= c for rr, cc, hh, ww in placed):
                placed.append(bbox)
                grid[r : r + 3, c : c + 3] = color
                if side == "u":
                    for j in miss_idx:
                        grid[r, c + j] = BG
                elif side == "d":
                    for j in miss_idx:
                        grid[r + 2, c + j] = BG
                elif side == "l":
                    for i in miss_idx:
                        grid[r + i, c] = BG
                else:
                    for i in miss_idx:
                        grid[r + i, c + 2] = BG
                break
    return grid
