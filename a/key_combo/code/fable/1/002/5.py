import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5


EDGE_K = {(0, 1): 0, (1, 2): 1, (2, 1): 2, (1, 0): 3}
CORNERS = {(0, 0), (0, 2), (2, 0), (2, 2)}


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _dial_at(g: np.ndarray, r: int, c: int):
    if r + 3 > g.shape[0] or c + 3 > g.shape[1]:
        return None
    patch = g[r : r + 3, c : c + 3]
    frame = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    if patch[1, 1] != BG:
        return None
    colored = []
    for rr, cc in frame:
        v = int(patch[rr, cc])
        if v == GRAY:
            continue
        if v == BG:
            return None
        colored.append((rr, cc, v))
    if len(colored) != 1:
        return None
    if any(int(patch[rr, cc]) != GRAY for rr, cc in frame if (rr, cc) != (colored[0][0], colored[0][1])):
        return None
    return colored[0]


def _find_dials(g: np.ndarray):
    dials = []
    occupied = set()
    for r in range(g.shape[0] - 2):
        for c in range(g.shape[1] - 2):
            item = _dial_at(g, r, c)
            if item is not None:
                rr, cc, color = item
                dials.append({"bbox": (r, c, r + 2, c + 2), "pos": (rr, cc), "color": color})
                occupied.update((r + a, c + b) for a in range(3) for b in range(3))
    assert len(dials) >= 2
    return dials, occupied


def _anti_diag_reflect(patch):
    n = patch.shape[0]
    out = np.zeros_like(patch)
    for r in range(n):
        for c in range(n):
            out[n - 1 - c, n - 1 - r] = patch[r, c]
    return out


def _transform(patch, pos):
    if pos in EDGE_K:
        return np.rot90(patch, -EDGE_K[pos])
    if pos == (0, 0) or pos == (2, 2):
        return patch.T.copy()
    return _anti_diag_reflect(patch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each grey dial to the same-colored sprite via its colored needle. Edge
    needles rotate the sprite; corner needles reflect it along the indicated
    diagonal. Dials are erased from the output.
    """
    g = np.asarray(grid, dtype=int)
    dials, dial_cells = _find_dials(g)
    out = g.copy()
    for r, c in dial_cells:
        out[r, c] = BG

    dial_colors = {d["color"] for d in dials}
    for color in dial_colors:
        comps = _components((g == color) & np.array([[((r, c) not in dial_cells) for c in range(g.shape[1])] for r in range(g.shape[0])]))
        assert len(comps) == 1
        comp = comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 == c1 - c0
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = color
        dial = next(d for d in dials if d["color"] == color)
        new_patch = _transform(patch, dial["pos"])
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        mask = new_patch != BG
        out[r0 : r1 + 1, c0 : c1 + 1][mask] = color
    return out


SPRITE = np.array(
    [
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ],
    dtype=bool,
)


def _can_place(grid, mask, r0, c0):
    h, w = mask.shape
    if r0 < 1 or c0 < 1 or r0 + h >= grid.shape[0] or c0 + w >= grid.shape[1]:
        return False
    rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
    rr1, cc1 = min(grid.shape[0], r0 + h + 1), min(grid.shape[1], c0 + w + 1)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _place_sprite(grid, color):
    for _ in range(200):
        r0 = random.randint(1, grid.shape[0] - 6)
        c0 = random.randint(1, grid.shape[1] - 6)
        if _can_place(grid, SPRITE, r0, c0):
            grid[r0 : r0 + 4, c0 : c0 + 4][SPRITE] = color
            return True
    return False


def _place_dial(grid, color, pos):
    frame = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    mask = np.ones((3, 3), dtype=bool)
    for _ in range(200):
        r0 = random.randint(1, grid.shape[0] - 4)
        c0 = random.randint(1, grid.shape[1] - 4)
        if _can_place(grid, mask, r0, c0):
            for rr, cc in frame:
                grid[r0 + rr, c0 + cc] = GRAY
            grid[r0 + pos[0], c0 + pos[1]] = color
            return True
    return False


def _gen_attempt():
    h = w = random.randint(22, 29)
    grid = np.zeros((h, w), dtype=int)
    n = random.randint(3, 5)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    positions = [random.choice(list(EDGE_K)), random.choice(list(CORNERS))]
    while len(positions) < n:
        positions.append(random.choice(list(EDGE_K) + list(CORNERS)))
    random.shuffle(positions)
    for color in colors:
        if not _place_sprite(grid, color):
            return None
    for color, pos in zip(colors, positions):
        if not _place_dial(grid, color, pos):
            return None
    try:
        dials, _ = _find_dials(grid)
        if not any(d["pos"] in EDGE_K for d in dials) or not any(d["pos"] in CORNERS for d in dials):
            return None
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
