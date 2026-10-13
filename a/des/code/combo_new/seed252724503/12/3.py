import numpy as np


BLACK = 0
GREY = 5


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords):
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
    for r, c in coords:
        mask[r - min(rs), c - min(cs)] = True
    return mask


def _rotations(mask):
    cur = mask
    for _ in range(4):
        yield cur
        cur = np.rot90(cur)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to a grey lock by rotated hole shape, then recolor the whole lock with that key's color.

    Input structure:
    - Small colored key sprites sit on the black background away from larger grey lock sprites.
    - Every lock contains exactly one enclosed black hole.
    - Each key matches exactly one lock hole up to rotation.

    Rule:
    - Extract each lock's enclosed black hole shape.
    - Find the uniquely matching key under 0, 90, 180, or 270 degree rotation.
    - Erase the key from its source position, fill the lock hole with the key color, and recolor all grey lock pixels to the same color.

    Key insight:
    - The key contributes both geometry and color: shape decides the match, and the matched key's color propagates to the entire lock body.
    """
    out = grid.copy()
    keys = []
    for color in np.unique(grid):
        if color in (BLACK, GREY):
            continue
        for comp in _components(grid == color):
            keys.append((color, _normalize(comp), comp))
    locks = []
    for comp in _components(grid == GREY):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        bbox = grid[r0 : r1 + 1, c0 : c1 + 1]
        holes = []
        for hole in _components(bbox == BLACK):
            hrs = [p[0] for p in hole]
            hcs = [p[1] for p in hole]
            if min(hrs) > 0 and max(hrs) < bbox.shape[0] - 1 and min(hcs) > 0 and max(hcs) < bbox.shape[1] - 1:
                holes.append(hole)
        if holes:
            locks.append((comp, holes[0], (r0, c0)))
    for comp, hole, (r0, c0) in locks:
        hole_mask = _normalize(hole)
        match = None
        for color, key_mask, key_comp in keys:
            if any(rot.shape == hole_mask.shape and np.array_equal(rot, hole_mask) for rot in _rotations(key_mask)):
                match = (color, key_comp)
                break
        color, key_comp = match
        for r, c in key_comp:
            out[r, c] = BLACK
        for r, c in comp:
            out[r, c] = color
        for r, c in hole:
            out[r0 + r, c0 + c] = color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((26, 26), dtype=int)
    shapes = [
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ]
    count = int(rng.integers(2, 5))
    choice = rng.choice(len(shapes), size=count, replace=False)
    colors = rng.choice([1, 2, 3, 4, 6, 7, 8, 9], size=count, replace=False)
    for idx, color, offset in zip(choice, colors, range(count)):
        key = shapes[int(idx)]
        kh, kw = key.shape
        kr = 2 + offset * 5
        kc = 1
        grid[kr : kr + kh, kc : kc + kw][key] = int(color)
        rot = np.rot90(key, int(rng.integers(0, 4)))
        rh, rw = rot.shape
        lr = 1 + offset * 5
        lc = 10 + int(rng.integers(0, 8))
        grid[lr : lr + rh + 4, lc : lc + rw + 4] = GREY
        grid[lr + 1 : lr + rh + 3, lc + 1 : lc + rw + 3] = GREY
        grid[lr + 2 : lr + 2 + rh, lc + 2 : lc + 2 + rw][rot] = BLACK
    return grid
