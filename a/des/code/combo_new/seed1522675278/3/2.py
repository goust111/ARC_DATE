import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair L-fragments by red-mark count, choose the base fragment by the mark opposite its missing corner, and complete a 2x2 square there.

    Input structure:
    - All fragments occupy 3 cells of a 2x2 box using a common foreground color.
    - Some occupied cells are red markers instead of the base color.
    - Each red-count class contains exactly two fragments.

    Rule:
    - Pair fragments sharing the same number of red marks.
    - The fragment with a red mark on the cell opposite its missing corner is the base.
    - Fill the full 2x2 square at the base position with the base color and erase the other fragment.

    Key insight:
    - Red count finds the partner, but red position chooses where the square is completed.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    comps = _components(g != BLACK)
    groups = {}
    for comp in comps:
        reds = sum(g[r, c] == RED for r, c in comp)
        groups.setdefault(reds, []).append(comp)
    for pair in groups.values():
        base = pair[0]
        for comp in pair:
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r0, c0 = min(rs), min(cs)
            occ = {(r - r0, c - c0) for r, c in comp}
            miss = list({(0, 0), (0, 1), (1, 0), (1, 1)} - occ)[0]
            opp = (1 - miss[0], 1 - miss[1])
            if g[r0 + opp[0], c0 + opp[1]] == RED:
                base = comp
        rs = [r for r, _ in base]
        cs = [c for _, c in base]
        r0, c0 = min(rs), min(cs)
        out[r0 : r0 + 2, c0 : c0 + 2] = BLUE
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((12, 12), dtype=int)
    pairs = [0, 1, 2]
    positions = [(1, 1), (1, 7), (6, 1), (6, 7), (3, 4), (8, 4)]
    shapes = [{(0, 0), (0, 1), (1, 0)}, {(0, 1), (1, 0), (1, 1)}]
    for idx, marks in enumerate(pairs):
        for k in range(2):
            r0, c0 = positions[2 * idx + k]
            occ = shapes[k]
            for r, c in occ:
                g[r0 + r, c0 + c] = BLUE
            miss = list({(0, 0), (0, 1), (1, 0), (1, 1)} - occ)[0]
            opp = (1 - miss[0], 1 - miss[1])
            red_cells = [(r0 + opp[0], c0 + opp[1])]
            others = [(r0 + r, c0 + c) for r, c in occ if (r, c) != opp]
            for cell in random.sample(red_cells + others, marks):
                g[cell] = RED
    return g
