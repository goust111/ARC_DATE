from __future__ import annotations
import numpy as np
from arc_utils import draw_line

def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate water pixels falling under gravity into basins formed by colored obstacles.

    Input structure:
    - bg = 0. Water pixels = color 1 (blue).
    - Other colors form static solid structures (walls, steps, containers).
    - Water starts in unstable positions above or beside basins.

    Rule:
    - Each water pixel falls straight down through bg cells.
    - When blocked below, it slides diagonally, preferring the side with a deeper drop (tie → right).
    - When resting on a surface, it slides horizontally toward the nearest gap where it can fall again.
    - Process repeats until no water pixel moves.

    Key insight: Water seeks the deepest reachable basin via diagonal and horizontal sliding, not just vertical fall.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1

    def fall_depth(r: int, c: int) -> int:
        d = 0
        rr = int(r)
        cc = int(c)
        while rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
            rr += 1
            d += 1
        return int(d)

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            # Down
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                continue

            # Diagonals (choose by deepest fall; tie -> right)
            opts = []
            if rr + 1 < H and cc - 1 >= 0 and int(g[rr + 1, cc - 1]) == int(bg):
                opts.append((int(rr + 1), int(cc - 1)))
            if rr + 1 < H and cc + 1 < W and int(g[rr + 1, cc + 1]) == int(bg):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                if len(opts) == 1:
                    rr, cc = opts[0]
                else:
                    (rL, cL), (rR, cR) = opts
                    dL = fall_depth(rL, cL)
                    dR = fall_depth(rR, cR)
                    if dR > dL:
                        rr, cc = int(rR), int(cR)
                    elif dL > dR:
                        rr, cc = int(rL), int(cL)
                    else:
                        rr, cc = int(rR), int(cR)
                continue

            # Horizontal slide only when supported.
            if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                # Empty run [L,R] in this row
                L = int(cc)
                while L - 1 >= 0 and int(g[rr, L - 1]) == int(bg):
                    L -= 1
                R = int(cc)
                while R + 1 < W and int(g[rr, R + 1]) == int(bg):
                    R += 1

                # Candidate drop columns within the run.
                drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
                if not drops:
                    break

                best = min(drops, key=lambda x: (abs(int(x - cc)), int(x)))  # tie -> left
                if best == int(cc):
                    break
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                if int(g[rr, nx]) == int(bg):
                    cc = nx
                    continue

            break
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(water):
                continue
            g[int(r), int(c)] = int(bg)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(water)
        if not moved:
            break

    return g
