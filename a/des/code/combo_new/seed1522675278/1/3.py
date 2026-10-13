import random

import numpy as np


MARK_TO_K = {2: 0, 3: 3, 1: 2, 7: 1}
MARKS = list(MARK_TO_K)
STRIPES = [4, 8]


def _restore_block(block):
    colors = [c for c in np.unique(block) if c not in MARKS]
    a, b = colors
    h, w = block.shape
    horiz = True
    for r in range(h):
        vals = {int(block[r, c]) for c in range(w) if not (r == 0 and c == 0)}
        if len(vals) > 1:
            horiz = False
            break
    if horiz:
        out = np.zeros_like(block)
        first = block[0, 1]
        second = a if b == first else b
        for r in range(h):
            out[r, :] = first if r % 2 == 0 else second
        return out
    out = np.zeros_like(block)
    first = block[1, 0]
    second = a if b == first else b
    for c in range(w):
        out[:, c] = first if c % 2 == 0 else second
    return out


def _block_size(g):
    h, w = g.shape
    for bs in range(2, min(h, w) + 1):
        if h % bs or w % bs:
            continue
        ok = True
        for r in range(0, h, bs):
            for c in range(0, w, bs):
                if int(g[r, c]) not in MARKS:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return bs
    raise ValueError("no block size")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode per-block rotation markers, restore the overwritten stripe cell, then rotate each striped block.

    Input structure:
    - The grid is tiled by equal square blocks with no separators.
    - Each block contains alternating horizontal or vertical stripes using the same two colors.
    - The top-left cell of every block is replaced by one marker color.

    Rule:
    - Marker colors encode the rotation angle for that block.
    - The top-left cell is restored from the stripe pattern before rotation.
    - Each block is rotated independently and stitched back into the full output grid.

    Key insight:
    - The marker cell is a control token, not part of the stripe pattern itself.
    """
    g = np.asarray(grid, dtype=int)
    bs = _block_size(g)
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(0, h, bs):
        for c in range(0, w, bs):
            block = g[r : r + bs, c : c + bs].copy()
            k = MARK_TO_K[int(block[0, 0])]
            block = _restore_block(block)
            out[r : r + bs, c : c + bs] = np.rot90(block, k)
    return out


def generate_input() -> np.ndarray:
    n = random.randint(2, 4)
    bs = random.randint(3, 5)
    g = np.zeros((n * bs, n * bs), dtype=int)
    for br in range(n):
        for bc in range(n):
            r = br * bs
            c = bc * bs
            horizontal = random.choice([True, False])
            phase = random.choice([0, 1])
            if horizontal:
                for rr in range(bs):
                    g[r + rr, c : c + bs] = STRIPES[(rr + phase) % 2]
            else:
                for cc in range(bs):
                    g[r : r + bs, c + cc] = STRIPES[(cc + phase) % 2]
            g[r, c] = random.choice(MARKS)
    return g
