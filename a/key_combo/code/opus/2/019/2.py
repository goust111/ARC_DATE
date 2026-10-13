import numpy as np


def _orbit_positions(r, c):
    if (r, c) == (2, 2):
        return [(2, 2)]
    return sorted({
        (r, c),
        (c, r),
        (4 - r, 4 - c),
        (4 - c, 4 - r),
        (c, 4 - r),
        (4 - c, r),
        (r, 4 - c),
        (4 - r, c),
    })


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        A 5x5 diagonal X pattern with some gray(5) mask pixels.
    Output:
        3x3 folded result after completing each orbit color.
    Idea:
        Complete each 8-fold orbit on X positions, then map the 9 X pixels to 3x3.
    Complexity:
        O(1) space for fixed size after local scan.
    """
    inp = np.array(grid, dtype=int)

    # locate diagonal block
    H, W = inp.shape
    best = (0, 0)
    best_score = -1
    for r in range(max(0, H - 4)):
        for c in range(max(0, W - 4)):
            sub = inp[r:r + 5, c:c + 5]
            diag = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (1, 3), (3, 1), (4, 0)]
            score = sum(int(sub[a, b] != 0) for a, b in diag)
            if score > best_score:
                best = (r, c)
                best_score = score

    sub = inp[best[0]:best[0] + 5, best[1]:best[1] + 5].copy()

    diag_pos = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (1, 3), (3, 1), (4, 0)]
    for rr, cc in diag_pos:
        if sub[rr, cc] != 5:
            continue
        orbit = _orbit_positions(rr, cc)
        color = 0
        for ar, ac in orbit:
            v = sub[ar, ac]
            if v != 5 and v != 0:
                color = v
                break
        if color == 0:
            color = 1
        for ar, ac in orbit:
            sub[ar, ac] = color

    fold = {
        (2, 2): (1, 1),
        (1, 1): (0, 0),
        (0, 0): (0, 1),
        (1, 3): (0, 2),
        (0, 4): (1, 2),
        (3, 1): (2, 0),
        (4, 0): (1, 0),
        (3, 3): (2, 1),
        (4, 4): (2, 2),
    }
    out = np.zeros((3, 3), dtype=int)
    for (sr, sc), (tr, tc) in fold.items():
        out[tr, tc] = sub[sr, sc]
    return out


def generate_input() -> np.ndarray:
    """
    Random input generator for task 2.
    Returns:
        A standalone 5x5 X pattern with random 8-fold masks.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(2302)
        generate_input._i = 0
    rs = generate_input._rng
    generate_input._i += 1

    sub = np.zeros((5, 5), dtype=int)
    colors = [1, 2, 3, 4]
    orbit1 = [(0, 0), (0, 4), (4, 0), (4, 4)]
    orbit2 = [(1, 1), (1, 3), (3, 1), (3, 3)]

    for orbit in [orbit1, orbit2]:
        col = int(colors[rs.randint(0, len(colors))])
        k = int(rs.randint(1, 4))
        idxs = rs.choice(np.arange(len(orbit)), size=k, replace=False)
        masked = {orbit[int(i)] for i in idxs}
        for p in orbit:
            sub[p] = 5 if p in masked else col

    # center always visible to avoid impossible recovery
    sub[2, 2] = int(colors[rs.randint(0, len(colors))])

    return sub
