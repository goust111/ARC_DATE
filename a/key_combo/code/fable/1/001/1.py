import random
from collections import deque

import numpy as np


BG = 0
AXIS = 5


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
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
        out.append(comp)
    return out


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(cells):
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _solid_join(residual, piece):
    rset = set(residual)
    pset0 = set(piece)
    for dr in range(-8, 9):
        for dc in range(-8, 9):
            pset = {(r + dr, c + dc) for r, c in pset0}
            if rset & pset:
                continue
            union = rset | pset
            r0, c0, r1, c1 = _bbox(union)
            if len(union) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            return r0, c0, r1, c1, pset
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the mirror-XOR residual above the divider, then output the candidate
    shape that complements it into a solid rectangle.

    The vertical axis is used only to fold asymmetric foreground pixels into one
    residual mask. Candidate colors supply the output color for the complementary
    part; residual cells are rendered with the axis color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r] == AXIS)]
    assert len(sep_rows) == 1
    sep = sep_rows[0]
    top = g[:sep]
    bottom = g[sep + 1 :]

    axis_cols = [c for c in range(w) if np.all(top[:, c] == AXIS)]
    assert len(axis_cols) == 1
    axis = axis_cols[0]

    residual = []
    for r in range(top.shape[0]):
        for c in range(axis):
            mc = 2 * axis - c
            if mc >= w:
                continue
            a = int(top[r, c])
            b = int(top[r, mc])
            a_fg = a not in (BG, AXIS)
            b_fg = b not in (BG, AXIS)
            if a_fg ^ b_fg:
                residual.append((r, axis - c - 1))
    assert residual, "expected mirror residual pixels"
    residual = list(_norm(residual))

    winners = []
    for comp in _components((bottom != BG) & (bottom != AXIS)):
        colors = {int(bottom[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = next(iter(colors))
        piece = list(_norm(comp))
        if set(piece) == set(residual):
            continue
        joined = _solid_join(residual, piece)
        if joined is not None:
            winners.append((color, joined))
    assert len(winners) == 1, "expected one complementary candidate"

    color, (r0, c0, r1, c1, pset) = winners[0]
    rset = set(residual)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in rset:
        out[r - r0, c - c0] = AXIS
    for r, c in pset:
        out[r - r0, c - c0] = color
    return out


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    q = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (r + dr, c + dc)
            if p in cells and p not in seen:
                seen.add(p)
                q.append(p)
    return seen == cells


def _sample_join_pair():
    for _ in range(200):
        rh = random.randint(2, 4)
        rw = random.randint(3, 5)
        all_cells = {(r, c) for r in range(rh) for c in range(rw)}
        size = random.randint(2, min(6, rh * rw - 2))
        cells = {random.choice(tuple(all_cells))}
        while len(cells) < size:
            r, c = random.choice(tuple(cells))
            nbrs = [
                (r + dr, c + dc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (r + dr, c + dc) in all_cells and (r + dr, c + dc) not in cells
            ]
            if nbrs:
                cells.add(random.choice(nbrs))
        comp = all_cells - cells
        if _connected(cells) and _connected(comp):
            return list(_norm(cells)), list(_norm(comp))
    raise RuntimeError("failed to sample complementary shapes")


def _can_place(grid, mask, r0, c0):
    h, w = mask.shape
    if r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0], r0 + h + 1)
    cc1 = min(grid.shape[1], c0 + w + 1)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _place_mask(grid, cells, color):
    arr = np.zeros((max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1), dtype=bool)
    for r, c in cells:
        arr[r, c] = True
    for _ in range(200):
        r0 = random.randint(0, grid.shape[0] - arr.shape[0])
        c0 = random.randint(0, grid.shape[1] - arr.shape[1])
        if _can_place(grid, arr, r0, c0):
            grid[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]][arr] = color
            return True
    return False


def _random_shape():
    h, w = random.randint(2, 4), random.randint(2, 4)
    cells = {(random.randrange(h), random.randrange(w))}
    for _ in range(random.randint(2, 5)):
        r, c = random.choice(tuple(cells))
        nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= r + dr < h and 0 <= c + dc < w]
        cells.add(random.choice(nbrs))
    return list(_norm(cells))


def _gen_attempt():
    residual, piece = _sample_join_pair()
    h = random.randint(18, 24)
    w = random.randint(17, 23)
    top_h = random.randint(8, 11)
    axis = w // 2
    grid = np.zeros((h, w), dtype=int)
    grid[top_h, :] = AXIS
    grid[:top_h, axis] = AXIS

    rr0 = random.randint(1, top_h - max(r for r, _ in residual) - 2)
    dd0 = random.randint(0, axis - max(c for _, c in residual) - 2)
    used = set()
    for rr, dd in residual:
        r = rr0 + rr
        dist = dd0 + dd
        if random.random() < 0.5:
            c = axis - 1 - dist
        else:
            c = axis + 1 + dist
        if not (0 <= c < w):
            return None
        grid[r, c] = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
        used.add((r, dist))

    for _ in range(random.randint(6, 12)):
        r = random.randint(0, top_h - 1)
        dist = random.randint(0, min(axis - 1, w - axis - 2))
        if (r, dist) in used:
            continue
        color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
        grid[r, axis - 1 - dist] = color
        grid[r, axis + 1 + dist] = color

    bottom = grid[top_h + 1 :]
    masks = [piece]
    while len(masks) < random.randint(3, 5):
        shape = _random_shape()
        if _solid_join(residual, shape) is None and set(shape) != set(residual):
            masks.append(shape)
    random.shuffle(masks)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], len(masks))
    for shape, color in zip(masks, colors):
        if not _place_mask(bottom, shape, color):
            return None
    try:
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
