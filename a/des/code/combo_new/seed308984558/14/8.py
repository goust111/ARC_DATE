from __future__ import annotations
import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a noisy fully-filled periodic color pattern by majority voting over 2x2 or 3x3 residue classes.

    Input structure:
    - A fully filled rectangular color grid with no background.
    - The true pattern repeats every 2 or 3 rows and every 2 or 3 columns, with a few corrupted cells.

    Rule:
    - Try all period pairs (P,Q) in {2,3}x{2,3}.
    - For each residue class, choose its majority color to define a candidate tile.
    - Select the candidate with the fewest mismatches to the input.
    - Replace every cell by the color predicted by that chosen periodic tile.

    Key insight:
    - The correct period is the one that globally minimizes disagreement after residue-class majority voting.
    """
    g = np.asarray(grid, dtype=int)
    best=None
    for q in (2,3):
        for p in (2,3):
            tile = np.zeros((q,p), dtype=int)
            mism=0
            for r in range(q):
                for c in range(p):
                    vals = g[r::q, c::p].ravel()
                    colors, counts = np.unique(vals, return_counts=True)
                    tile[r,c] = int(colors[np.argmax(counts)])
            pred = np.fromfunction(lambda r,c: tile[r % q, c % p], g.shape, dtype=int)
            mism = int(np.sum(pred != g))
            if best is None or mism < best[0]: best=(mism,pred)
    return best[1]


def generate_input() -> np.ndarray:
    q,p = random.choice([2,3]), random.choice([2,3])
    h,w = random.randint(10,18), random.randint(10,18)
    tile = np.array([[random.choice([1,2,3,4,6,7,8,9]) for _ in range(p)] for __ in range(q)], dtype=int)
    g = np.fromfunction(lambda r,c: tile[r % q, c % p], (h,w), dtype=int)
    for _ in range(max(1, h*w//12)):
        r,c = random.randrange(h), random.randrange(w)
        g[r,c] = random.choice([1,2,3,4,6,7,8,9])
    return g
