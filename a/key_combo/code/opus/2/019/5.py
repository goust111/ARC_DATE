import numpy as np


def _rotate_point(pt, n: int, k: int) -> tuple[int, int]:
    r, c = pt
    if k == 0:
        return r, c
    if k == 1:
        return c, n - 1 - r
    if k == 2:
        return n - 1 - r, n - 1 - c
    # k==3
    return n - 1 - c, r


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        n^2 x n^2 grid, top-left n x n contains an incomplete foreground motif.
    Output:
        Same size, expanded self-similar field of blocks.
    Idea:
        Complete the top-left motif by 90°-rotation closure, mark added pixels as yellow.
        For each completed motif pixel, stamp one full motif block; color by original/addition ownership.
    Complexity:
        O((n^4) + (n^2 * n^2)) with n <= 5 in generators.
    """
    inp = np.array(grid, dtype=int)
    H, W = inp.shape
    n = int(np.sqrt(H))
    if n * n != H or H != W:
        n = int(np.sqrt(min(H, W)))

    base = inp[:n, :n]
    green = [(r, c) for r in range(n) for c in range(n) if base[r, c] == 3]

    full_set = set(green)
    k = len(green)
    for (r, c) in green:
        for t in range(1, 4):
            full_set.add(_rotate_point((r, c), n, t))

    base_mask = np.zeros((n, n), dtype=bool)
    for r, c in full_set:
        base_mask[r, c] = True

    orig_set = set(green)
    new_set = full_set - orig_set

    out = np.zeros((n * n, n * n), dtype=int)

    # self-similar expansion
    for r in range(n):
        for c in range(n):
            if not base_mask[r, c]:
                continue
            color = 3 if (r, c) in orig_set else 4
            # place block
            for rr in range(n):
                for cc in range(n):
                    if base_mask[rr, cc]:
                        out[r * n + rr, c * n + cc] = color

    return out


def _rand_connected(rng: np.random.RandomState, n: int, target: int) -> list[tuple[int, int]]:
    cells = {(rng.randint(0, n), rng.randint(0, n))}
    frontier = list(cells)
    while len(cells) < target:
        if not frontier:
            return []
        x, y = frontier[rng.randint(0, len(frontier))]
        neigh = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        cand = [(a, b) for a, b in neigh if 0 <= a < n and 0 <= b < n and (a, b) not in cells]
        if not cand:
            continue
        a, b = cand[rng.randint(0, len(cand))]
        cells.add((int(a), int(b)))
        frontier.append((int(a), int(b)))
    return sorted(cells)


def _symmetry_ok(shape: set[tuple[int, int]], n: int) -> bool:
    for k in [1, 2, 3]:
        rot = {_rotate_point(p, n, k) for p in shape}
        if len(rot) == len(shape):
            continue
        # new pixels added => not fully symmetric
    return True


def generate_input() -> np.ndarray:
    """
    Random input generator for task 5.
    Returns:
        A 16x16 grid with random top-left 4x4 motif.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(5505)
        generate_input._i = 0
    rs = generate_input._rng
    generate_input._i += 1

    n = 4
    out = np.zeros((n * n, n * n), dtype=int)

    for _ in range(80):
        k = int(rs.randint(3, 7))
        cells = _rand_connected(rs, n, k)
        orig = set(cells)
        full = set(orig)
        for (r, c) in orig:
            for t in range(1, 4):
                full.add(_rotate_point((r, c), n, t))
        if len(full) <= len(orig):
            continue

        # reject accidental rotational symmetry
        bad = False
        for t in [1, 2, 3]:
            rotated = {_rotate_point(p, n, t) for p in orig}
            if rotated == orig:
                bad = True
                break
        if bad:
            continue

        for r, c in orig:
            out[r, c] = 3
        return out

    # fallback deterministic pattern
    out[0, 0] = 3
    out[1, 0] = 3
    out[0, 1] = 3
    return out
