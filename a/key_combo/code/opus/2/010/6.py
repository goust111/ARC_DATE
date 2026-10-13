import random

import numpy as np


SPRITE = np.array(
    [
        [1, 4, 1, 1],
        [1, 1, 4, 1],
        [4, 1, 1, 1],
        [1, 1, 1, 4],
    ],
    dtype=int,
)


def _separators(g):
    rows = [r for r in range(g.shape[0]) if np.all(np.isin(g[r, :], [0, 5]))]
    cols = [c for c in range(g.shape[1]) if np.all(np.isin(g[:, c], [0, 5]))]
    return rows, cols


def _ranges(n, seps):
    starts = [0] + [s + 1 for s in seps]
    ends = seps + [n]
    return [(a, b) for a, b in zip(starts, ends) if a < b]


def _match(a, b):
    return a.shape == b.shape and bool(np.all((a == 5) | (a == b)))


def _blank(a):
    return bool(np.all((a == 5) | (a == 4) | (a == 2)))


def _invert_piece(piece, sprite_piece):
    clean = piece.copy()
    clean[clean == 2] = 5
    if _match(clean, sprite_piece):
        return np.full(piece.shape, 1, dtype=int)
    if _blank(clean):
        return sprite_piece.copy()
    return np.where(clean == 5, sprite_piece, clean)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    sep_rows, sep_cols = _separators(g)
    row_ranges = _ranges(g.shape[0], sep_rows)
    col_ranges = _ranges(g.shape[1], sep_cols)
    sprite = None
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            block = g[r0:r1, c0:c1]
            if 2 not in block and 5 not in block and len(set(block.ravel())) > 1:
                sprite = block.copy()
                break
        if sprite is not None:
            break
    if sprite is None:
        sprite = SPRITE

    block_rows = []
    for bi, (r0, r1) in enumerate(row_ranges):
        pieces = []
        for bj, (c0, c1) in enumerate(col_ranges):
            block = g[r0:r1, c0:c1]
            reds = np.argwhere(block == 2)
            if len(reds) and len(set(reds[:, 0])) == 1:
                rr = int(reds[0, 0])
                top = _invert_piece(block[:rr, :], sprite[:rr, :])
                bot = _invert_piece(block[rr + 1 :, :], sprite[rr + 1 :, :])
                outb = np.vstack([top, np.zeros((2, block.shape[1]), dtype=int), bot])
            elif len(reds) and len(set(reds[:, 1])) == 1:
                cc = int(reds[0, 1])
                left = _invert_piece(block[:, :cc], sprite[:, :cc])
                right = _invert_piece(block[:, cc + 1 :], sprite[:, cc + 1 :])
                outb = np.hstack([left, np.zeros((block.shape[0], 2), dtype=int), right])
            elif _match(block, sprite):
                outb = np.full(block.shape, 1, dtype=int)
            elif _blank(block):
                outb = sprite.copy()
            else:
                outb = np.where(block == 5, sprite, block)
            pieces.append(outb)
        maxh = max(p.shape[0] for p in pieces)
        padded = [np.pad(p, ((0, maxh - p.shape[0]), (0, 0))) for p in pieces]
        sep = np.zeros((maxh, 1), dtype=int)
        row = padded[0]
        for p in padded[1:]:
            row = np.hstack([row, sep, p])
        block_rows.append(row)
    maxw = max(r.shape[1] for r in block_rows)
    block_rows = [np.pad(r, ((0, 0), (0, maxw - r.shape[1]))) for r in block_rows]
    sep = np.zeros((1, maxw), dtype=int)
    out = block_rows[0]
    for r in block_rows[1:]:
        out = np.vstack([out, sep, r])
    return out


def generate_input() -> np.ndarray:
    b = 4
    g = np.zeros((3 * b + 2, 3 * b + 2), dtype=int)
    states = [True, True, True, False, False, False, True, False, random.choice([True, False])]
    random.shuffle(states)
    cracks = set(random.sample(range(9), random.randint(1, 2)))
    for idx, is_sprite in enumerate(states):
        br, bc = divmod(idx, 3)
        r, c = br * (b + 1), bc * (b + 1)
        g[r : r + b, c : c + b] = SPRITE if is_sprite else 4
        if idx in cracks:
            if random.random() < 0.5:
                g[r + b // 2, c : c + b] = 2
            else:
                g[r : r + b, c + b // 2] = 2
    for _ in range(random.randint(1, 3)):
        r, c = random.randrange(g.shape[0]), random.randrange(g.shape[1])
        if g[r, c] != 2:
            g[r, c] = 5
    return g
