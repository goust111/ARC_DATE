import numpy as np


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            stack = [(r, c)]
            vis[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray):
    """Task 3 transform.

    1) Detect all blue occluders and rotate the whole grid for each to restore hidden content.
    2) Locate all red markers after restoration and compute their strict bounding box.
    3) Crop to the interior of that marker bounding box and remove red markers.
    4) Return the cropped recolored pattern.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    rotated = np.rot90(g, 2)
    out = g.copy()

    comps = _components(g == 1)
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        out[r0:r1 + 1, c0:c1 + 1] = rotated[r0:r1 + 1, c0:c1 + 1]

    marks = np.argwhere(out == 2)
    if marks.size == 0:
        return out

    rs = marks[:, 0]
    cs = marks[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    rr0 = r0 + 1
    rr1 = r1 - 1
    cc0 = c0 + 1
    cc1 = c1 - 1
    if rr0 > rr1 or cc0 > cc1:
        return np.zeros((1, 1), dtype=int)

    crop = out[rr0:rr1 + 1, cc0:cc1 + 1].copy()
    crop[crop == 2] = 0
    return crop


def generate_input():
    """Problem 3 input generator.

    1) Create a 17x17 180° symmetric yellow/gray pattern.
    2) Place three red markers on pattern corners of their own bbox, one covered by a blue block.
    3) Place 2-3 blue rectangular occluders and return the corrupted grid.
    """

    h = w = 17
    g = np.zeros((h, w), dtype=int)

    colors = [4, 5]
    for r in range((h + 1) // 2):
        for c in range((w + 1) // 2):
            if (r + c) % 7 == 0:
                continue
            if (r + c) % 3 == 0:
                col = colors[int(_RNG.choice(2))]
                rr = h - 1 - r
                cc = w - 1 - c
                g[r, c] = col
                g[rr, cc] = col

    r1 = int(_RNG.integers(3, 13))
    r2 = int(_RNG.integers(r1 + 1, 14))
    c1 = int(_RNG.integers(3, 13))
    c2 = int(_RNG.integers(c1 + 1, 14))
    corners = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
    pick = _RNG.choice(np.arange(4), size=3, replace=False)
    markers = [tuple(map(int, corners[int(i)] ) ) for i in pick]

    for rr, cc in markers:
        g[rr, cc] = 2

    hidden = int(_RNG.integers(3))

    occluders = int(_RNG.integers(2, 4))
    placed = []
    # first occluder must cover the hidden marker
    hr, hc = markers[hidden]
    for _ in range(40):
        size = int(_RNG.choice([2, 3]))
        r0 = int(_RNG.integers(max(0, hr - size + 1), min(hr, h - size) + 1))
        c0 = int(_RNG.integers(max(0, hc - size + 1), min(hc, w - size) + 1))
        r1 = r0 + size - 1
        c1 = c0 + size - 1
        if not (r0 <= hr <= r1 and c0 <= hc <= c1):
            continue
        # avoid full overlap with all 3 markers
        if all(not (r0 <= mr <= r1 and c0 <= mc <= c1) for mr, mc in markers):
            continue
        break
    g[r0:r1 + 1, c0:c1 + 1] = 1
    placed.append((r0, c0, r1, c1))

    for _ in range(occluders - 1):
        for _ in range(80):
            size = int(_RNG.choice([2, 3]))
            tr = int(_RNG.integers(0, h - size + 1))
            tc = int(_RNG.integers(0, w - size + 1))
            rr = tr + size - 1
            cc = tc + size - 1
            # do not cover any markers
            if any(tr <= mr <= rr and tc <= mc <= cc for mr, mc in markers):
                continue
            if any(not (rr < p0 or p2 < tr or cc < p1 or p3 < tc) for p0, p1, p2, p3 in placed):
                break
            g[tr:rr + 1, tc:cc + 1] = 1
            placed.append((tr, tc, rr, cc))
            break

    # make sure only two markers remain visible
    g_hidden_restore = g.copy()
    return g


# fixed RNG shared by generator
_RNG = np.random.default_rng(3103)
if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
