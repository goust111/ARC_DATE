import numpy as np


_BG = 0
_GRAY = 5

_generate_calls = 0


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(6000 + _generate_calls)
    _generate_calls += 1

    h = 16
    w = 16
    grid = np.zeros((h, w), dtype=int)

    r0 = 7
    c0 = 7
    grid[r0, c0] = _GRAY
    grid[r0, c0 + 1] = _GRAY
    grid[r0 + 1, c0] = _GRAY
    grid[r0 + 1, c0 + 1] = _GRAY

    base_colors = [2, 1]
    base = rng.choice(base_colors, size=(4, 4))
    if np.all(base == base[0, 0]):
        base[0, 1] = base_colors[1]
    if np.array_equal(base, np.rot90(base, -1)):
        base[0, 0], base[1, 1] = base[1, 1], base[0, 0]

    source_q = int(rng.randint(0, 4))
    placements = [(2, 2), (2, 10), (10, 10), (10, 2)]

    for qi, (rr, cc) in enumerate(placements):
        pat = base.copy()
        if qi != source_q:
            k = (qi - source_q) % 4
            pat = np.rot90(pat, -k)
            keep = int(rng.randint(10, 13))
            drop = 16 - keep
            flat = list(range(16))
            rng.shuffle(flat)
            for idx in flat[:drop]:
                y = idx // 4
                x = idx % 4
                pat[y, x] = 0
        for i in range(4):
            for j in range(4):
                if pat[i, j] != 0:
                    grid[rr + i, cc + j] = int(pat[i, j])

    return grid


def _rotate(mat, k):
    return np.rot90(mat, -k)


def main(grid: np.ndarray) -> np.ndarray:
    """1) Locate the gray 2x2 anchor and four surrounding quadrants.
2) Find the unique fully-filled source quadrant template.
3) For each other quadrant, rotate the source by quarter-turn offsets.
4) Fill only missing pixels according to the rotated template."""

    g = np.array(grid)
    out = g.copy()

    loc = np.argwhere(g == _GRAY)
    if len(loc) < 4:
        return out

    r0, c0 = int(np.min(loc[:, 0])), int(np.min(loc[:, 1]))
    anchors = {
        0: (r0 - 4, c0 - 4, r0, c0),
        1: (r0 - 4, c0 + 2, r0, c0 + 6),
        2: (r0 + 2, c0 + 2, r0 + 6, c0 + 6),
        3: (r0 + 2, c0 - 4, r0 + 6, c0),
    }

    source_q = None
    source_template = None
    for q, (ar0, ac0, ar1, ac1) in anchors.items():
        if ar0 < 0 or ac0 < 0 or ar1 > g.shape[0] or ac1 > g.shape[1]:
            continue
        box = g[ar0:ar1, ac0:ac1]
        if box.size == 16 and np.all(box != 0):
            source_q = q
            source_template = box.copy()
            break

    if source_template is None:
        return out

    for q, (ar0, ac0, ar1, ac1) in anchors.items():
        if q == source_q:
            continue
        if ar0 < 0 or ac0 < 0 or ar1 > g.shape[0] or ac1 > g.shape[1]:
            continue
        k = (q - source_q) % 4
        target = _rotate(source_template, k)
        for i in range(4):
            for j in range(4):
                rr = ar0 + i
                cc = ac0 + j
                if target[i, j] != 0 and out[rr, cc] == 0:
                    out[rr, cc] = int(target[i, j])

    return out


if __name__ == "__main__":
    print(main(generate_input()).shape)
