import random

import numpy as np


def _variant(tile, kind):
    if kind == "h":
        return np.fliplr(tile)
    if kind == "v":
        return np.flipud(tile)
    return tile.copy()


def _normalize(block, kind):
    return _variant(block, kind)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape
    bw = w // 4
    base = g[:, :bw]
    out = np.zeros((h, bw), dtype=int)
    variants = {"d": base, "h": np.fliplr(base), "v": np.flipud(base)}
    for i in range(1, 4):
        block = g[:, i * bw : (i + 1) * bw]
        kind = min(variants, key=lambda k: int(np.sum(block != variants[k])))
        norm = _normalize(block, kind)
        out[norm != base] = 3
    return out


def _base_tile():
    tile = np.zeros((12, 4), dtype=int)
    cells = random.sample([(r, c) for r in range(12) for c in range(4)], random.randint(8, 12))
    for r, c in cells:
        tile[r, c] = random.choice([1, 2])
    if min(np.sum(tile != np.fliplr(tile)), np.sum(tile != np.flipud(tile))) < 4:
        return _base_tile()
    return tile


def generate_input() -> np.ndarray:
    base = _base_tile()
    kinds = ["d", "h", "v"]
    tail = [random.choice(kinds) for _ in range(3)]
    if len(set(tail)) < 2:
        tail[0], tail[1] = "h", "v"
    blocks = [base.copy()] + [_variant(base, k) for k in tail]
    bad = random.randint(1, 3)
    for _ in range(random.randint(1, 2)):
        r = random.randrange(12)
        c = random.randrange(4)
        old = blocks[bad][r, c]
        choices = [0, 1, 2]
        choices.remove(int(old))
        blocks[bad][r, c] = random.choice(choices)
    return np.concatenate(blocks, axis=1)
