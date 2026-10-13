from __future__ import annotations
from collections import Counter
import numpy as np

def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair noisy periodic patterns inside frame-delimited grid regions.

    Input structure:
    - Frame color (most common) forms full rows/cols, dividing the grid into rectangular regions.
    - bg (second most common) fills empty space; each region contains a small colored tile repeated periodically with a few corrupted pixels.

    Rule:
    - For each region, find the bounding box of non-bg content.
    - Infer the smallest repeating tile that explains the pattern (majority vote per residue position), requiring it repeat at least 2x per axis.
    - Replace the region with the clean tiled version.

    Key insight: Tile dimensions are unknown and must be inferred; prefer the smallest tile with genuine repetition over one that memorizes the entire region.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    frame = int(cnt.most_common(1)[0][0])
    bg = None
    for col, _k in cnt.most_common():
        if int(col) != int(frame):
            bg = int(col)
            break
    assert not (bg is None)

    fr = [int(r) for r in range(H) if bool((g[int(r), :] == int(frame)).all())]
    fc = [int(c) for c in range(W) if bool((g[:, int(c)] == int(frame)).all())]
    fr = sorted(set(fr))
    fc = sorted(set(fc))

    assert fr
    assert fc

    if fr[0] != 0:
        fr = [-1] + fr
    if fr[-1] != (H - 1):
        fr = fr + [H]
    else:
        fr = fr + [H]

    if fc[0] != 0:
        fc = [-1] + fc
    if fc[-1] != (W - 1):
        fc = fc + [W]
    else:
        fc = fc + [W]

    def reconstruct(core: np.ndarray, maxp: int = 12) -> np.ndarray:
        h, w = core.shape
        best_all = None  # ((mism, area), B)
        best_ok = None

        for ph in range(1, min(int(maxp), int(h)) + 1):
            for pw in range(1, min(int(maxp), int(w)) + 1):
                tile = np.zeros((int(ph), int(pw)), dtype=int)
                for i in range(int(ph)):
                    for j in range(int(pw)):
                        vals = core[int(i) :: int(ph), int(j) :: int(pw)].flatten()
                        tile[int(i), int(j)] = int(Counter(map(int, vals)).most_common(1)[0][0])

                B = np.tile(tile, ((h + ph - 1) // ph, (w + pw - 1) // pw))[:h, :w]
                mism = int(np.sum(B != core))
                area = int(ph * pw)
                key = (mism, area)

                if best_all is None or key < best_all[0]:
                    best_all = (key, B)

                # Require some repetition (avoid memorizing the entire core).
                ok = True
                if int(ph) == int(h) and int(pw) == int(w):
                    ok = False
                if int(ph) < int(h) and (int(h) // int(ph)) < 2:
                    ok = False
                if int(pw) < int(w) and (int(w) // int(pw)) < 2:
                    ok = False
                if ok and (best_ok is None or key < best_ok[0]):
                    best_ok = (key, B)

        return (best_ok or best_all)[1]

    out = g.copy()
    for i in range(len(fr) - 1):
        r0, r1 = int(fr[i]), int(fr[i + 1])
        if int(r1 - r0) <= 1:
            continue
        rr0, rr1 = int(r0 + 1), int(r1 - 1)

        for j in range(len(fc) - 1):
            c0, c1 = int(fc[j]), int(fc[j + 1])
            if int(c1 - c0) <= 1:
                continue
            cc0, cc1 = int(c0 + 1), int(c1 - 1)

            A = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            m = A != int(bg)
            if not bool(m.any()):
                continue

            rs = np.where(m.any(axis=1))[0]
            cs = np.where(m.any(axis=0))[0]
            rmin, rmax = int(rs.min()), int(rs.max())
            cmin, cmax = int(cs.min()), int(cs.max())

            core = A[rmin : rmax + 1, cmin : cmax + 1]
            rec = reconstruct(core)

            A2 = A.copy()
            A2[rmin : rmax + 1, cmin : cmax + 1] = rec
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = A2

    return out


def generate_input() -> np.ndarray:
    bg = 0
    frame = 5

    # Region grid (2x2 or 3x2) with 1px frame separators.
    n_r = int(np.random.randint(2, 4))
    n_c = int(np.random.randint(2, 3))

    r_sizes = [int(np.random.randint(6, 11)) for _ in range(n_r)]
    c_sizes = [int(np.random.randint(6, 11)) for _ in range(n_c)]

    H = int(sum(r_sizes) + (n_r + 1))
    W = int(sum(c_sizes) + (n_c + 1))

    g = np.full((H, W), int(frame), dtype=int)

    # Fill each region with bg and optionally a repeating tile with noise.
    any_pattern = False
    r0 = 1
    for i, rh in enumerate(r_sizes):
        c0 = 1
        for j, cw in enumerate(c_sizes):
            region = np.full((int(rh), int(cw)), int(bg), dtype=int)

            if bool(np.random.rand() < 0.85) or not any_pattern:
                any_pattern = True
                ph = int(np.random.randint(1, min(4, rh) + 1))
                pw = int(np.random.randint(1, min(4, cw) + 1))

                pal = [int(c) for c in np.random.choice(range(1, 10), size=int(np.random.randint(2, 5)), replace=False)]
                pal = [c for c in pal if int(c) != int(frame)]
                if not pal:
                    pal = [2, 3]

                tile = np.full((ph, pw), int(bg), dtype=int)
                # Ensure at least one non-bg in the tile.
                tile[int(np.random.randint(0, ph)), int(np.random.randint(0, pw))] = int(pal[0])
                # Add a few more colored cells.
                k = int(np.random.randint(0, ph * pw))
                idx = np.random.choice(ph * pw, size=int(k), replace=False)
                for t in idx:
                    tile.flat[int(t)] = int(pal[int(np.random.randint(0, len(pal)))])

                tiled = np.tile(tile, ((rh + ph - 1) // ph, (cw + pw - 1) // pw))[:rh, :cw]
                region[:, :] = tiled

                # Add small noise (flip a few cells) so reconstruction does real work.
                n_noise = int(np.random.randint(0, max(1, (rh * cw) // 20)))
                for _ in range(n_noise):
                    rr = int(np.random.randint(0, rh))
                    cc = int(np.random.randint(0, cw))
                    if int(region[rr, cc]) != int(bg):
                        region[rr, cc] = int(pal[int(np.random.randint(0, len(pal)))])

            g[int(r0) : int(r0 + rh), int(c0) : int(c0 + cw)] = region
            c0 += int(cw) + 1
        r0 += int(rh) + 1

    return g

