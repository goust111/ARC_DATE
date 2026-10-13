import random

import numpy as np

BLACK = 0
MAROON = 9


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed block straight inward across its room while keeping the maroon partition grid fixed.

    Input structure:
    - Black background partitioned into rectangular rooms by full maroon separator rows and columns.
    - Each room contains one small colored seed block touching exactly one room side.
    - Seed colors are distinct from black and maroon.

    Rule:
    - For each room, detect which side the seed block touches.
    - If the seed touches the top or bottom, extend a same-colored vertical strip covering the seed's columns across the room interior.
    - If the seed touches the left or right, extend a same-colored horizontal strip covering the seed's rows across the room interior.

    Key insight:
    - The touched boundary determines the growth direction, while the seed block's own width or height determines the strip thickness.
    """
    out = grid.copy()
    line_rows = [-1] + [r for r in range(grid.shape[0]) if np.all(grid[r, :] == MAROON)] + [grid.shape[0]]
    line_cols = [-1] + [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAROON)] + [grid.shape[1]]
    for r0, r1 in zip(line_rows, line_rows[1:]):
        for c0, c1 in zip(line_cols, line_cols[1:]):
            rs = slice(r0 + 1, r1)
            cs = slice(c0 + 1, c1)
            if rs.stop <= rs.start or cs.stop <= cs.start:
                continue
            room = grid[rs, cs]
            vals = [int(v) for v in np.unique(room) if v not in (BLACK, MAROON)]
            if not vals:
                continue
            color = vals[0]
            rr, cc = np.where(room == color)
            minr, maxr = int(rr.min()), int(rr.max())
            minc, maxc = int(cc.min()), int(cc.max())
            if minr == 0 or maxr == room.shape[0] - 1:
                out[r0 + 1:r1, c0 + 1 + minc:c0 + 2 + maxc] = color
            else:
                out[r0 + 1 + minr:r0 + 2 + maxr, c0 + 1:c1] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 5, 6, 7, 8]
    for _ in range(400):
        room_rows = rng.randint(2, 3)
        room_cols = rng.randint(2, min(3, len(colors) // room_rows))
        heights = [rng.randint(4, 6) for _ in range(room_rows)]
        widths = [rng.randint(4, 6) for _ in range(room_cols)]
        h = sum(heights) + room_rows - 1
        w = sum(widths) + room_cols - 1
        if h > 30 or w > 30:
            continue
        grid = np.zeros((h, w), dtype=int)
        row_edges = []
        acc = 0
        for i, hh in enumerate(heights[:-1]):
            acc += hh
            row_edges.append(acc)
            acc += 1
        col_edges = []
        acc = 0
        for i, ww in enumerate(widths[:-1]):
            acc += ww
            col_edges.append(acc)
            acc += 1
        for r in row_edges:
            grid[r, :] = MAROON
        for c in col_edges:
            grid[:, c] = MAROON
        color_iter = iter(rng.sample(colors, room_rows * room_cols))
        r_start = 0
        for rh in heights:
            c_start = 0
            for cw in widths:
                color = next(color_iter)
                side = rng.choice(["top", "bottom", "left", "right"])
                if side in ("top", "bottom"):
                    block_h = rng.randint(1, min(2, rh))
                    block_w = rng.randint(1, min(2, cw))
                    col0 = c_start + rng.randint(0, cw - block_w)
                    row0 = r_start if side == "top" else r_start + rh - block_h
                else:
                    block_h = rng.randint(1, min(2, rh))
                    block_w = rng.randint(1, min(2, cw))
                    row0 = r_start + rng.randint(0, rh - block_h)
                    col0 = c_start if side == "left" else c_start + cw - block_w
                grid[row0:row0 + block_h, col0:col0 + block_w] = color
                c_start += cw + 1
            r_start += rh + 1
        return grid
    raise RuntimeError("failed to generate task 13/4")
