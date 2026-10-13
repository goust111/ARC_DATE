import numpy as np

_RNG = np.random.default_rng(3108)


def main(grid: np.ndarray):
    """Task 8 transform.

    1) For each seed pixel, find a same-color framed rectangle in the top area.
    2) Compute interior area A of that frame.
    3) Extend a vertical column of seed color upward by exactly A cells.
    4) Keep top frames unchanged and clip extension at canvas boundary.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    seed_row = h - 2
    if seed_row < 0:
        return out

    for c in range(w):
        seed_color = int(g[seed_row, c])
        if seed_color in (0, 5):
            continue

        # find frame by same color in non-zero pixels above separator
        comp_mask = g[:seed_row] == seed_color
        rs, cs = np.where(comp_mask)
        if rs.size == 0:
            continue
        # choose component connected to frame candidate using bbox from all same-color pixels
        rmin, rmax = int(rs.min()), int(rs.max())
        cmin, cmax = int(cs.min()), int(cs.max())
        if rmax - rmin < 2 or cmax - cmin < 2:
            continue

        # interior area measured by bounding rectangle
        A = max(0, (rmax - rmin - 1) * (cmax - cmin - 1))
        if A <= 0:
            continue

        for k in range(A):
            rr = seed_row - k
            if rr < 0:
                break
            out[rr, c] = seed_color

    return out


def generate_input():
    """Task 8 input generator.

    1) Create 15x18 canvas with 3-4 colored rectangular frames in top area.
    2) Put same colors on one seed at bottom row, with one seed per frame.
    3) Interior areas of frames are distinct positive integers.
    """

    h, w = 15, 18
    g = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3, 4]
    _RNG.shuffle(colors)

    frame_colors = colors[: int(_RNG.integers(3, 5))]
    used_areas = []
    attempts = 0

    # place small non-overlap frames in upper half
    occ = np.zeros((h, w), dtype=bool)
    for idx, col in enumerate(frame_colors):
        area = int(_RNG.integers(1, 8))
        while area in used_areas:
            area = int(_RNG.integers(1, 8))
        used_areas.append(area)

        # pick dimensions producing this area when hollow interior
        candidates = []
        for hh in range(3, 8):
            for ww in range(3, 8):
                a = (hh - 2) * (ww - 2)
                if a == area:
                    candidates.append((hh, ww))
        if not candidates:
            area = max(1, area - 1)
            for hh in range(3, 8):
                for ww in range(3, 8):
                    a = (hh - 2) * (ww - 2)
                    if a == area:
                        candidates.append((hh, ww))
        if not candidates:
            continue

        hh, ww = candidates[int(_RNG.integers(len(candidates)))]
        placed = False
        for _ in range(120):
            r0 = int(_RNG.integers(0, 8))
            c0 = int(_RNG.integers(0, w - ww))
            r1 = r0 + hh - 1
            c1 = c0 + ww - 1
            if np.any(occ[max(0, r0 - 1):min(h - 1, r1 + 2), max(0, c0 - 1):min(w - 1, c1 + 2)]):
                continue
            g[r0, c0:c1 + 1] = col
            g[r1, c0:c1 + 1] = col
            g[r0:r1 + 1, c0] = col
            g[r0:r1 + 1, c1] = col
            occ[max(0, r0 - 1):min(h - 1, r1 + 2), max(0, c0 - 1):min(w - 1, c1 + 2)] = True
            placed = True
            break

        if placed:
            attempts += 1

    # seeds at bottom row
    seed_row = h - 2
    seed_cols = np.arange(1, w - 1)
    _RNG.shuffle(seed_cols)
    idx = 0
    for c in range(w):
        if c == frame_colors[0] and idx < len(seed_cols):
            pass

    # find frame colors actually on top area and use first row in order
    active_colors = list(dict.fromkeys([v for v in np.unique(g[:10]) if v != 0 and v != 5]).keys())
    for i, cc in enumerate(active_colors):
        if i >= len(seed_cols):
            break
        c = int(seed_cols[i])
        g[seed_row, c] = cc

    return g


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
