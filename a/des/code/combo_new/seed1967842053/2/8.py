import random
from collections import deque

import numpy as np


GRAY = 5
BLACK = 0
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _nonzero_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == BLACK:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if not seen[nr, nc] and grid[nr, nc] != BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each colored seed inside its own gray region for as many layers as that region has seeds.

    Input structure:
    The grid has black background, separated non-black connected components, and each component is a gray
    region containing one or more colored single-cell seeds. Gray uses color 5; seeds are non-black and
    non-gray cells. Different regions are separated by black cells.

    Rule:
    For each non-black component, count its seeds N. Each seed expands through gray cells in that same
    component for up to N four-neighbor layers. For every gray cell, consider all seeds that can reach it
    within N layers. The color that reaches it at the latest layer is kept; if several seeds reach it on
    that same latest layer, the conflict cancels and the cell stays gray. Unreached gray cells stay gray,
    black background stays black, and the original seed cells keep their colors.

    Key insight:
    The gray components are independent containers, and the hidden growth radius is not fixed: it is the
    number of seeds inside that same container, with equal-layer conflicts canceling to gray.
    """
    arr = np.asarray(grid)
    h, w = arr.shape
    out = arr.copy()

    for comp in _nonzero_components(arr):
        seeds = [(r, c, int(arr[r, c])) for r, c in comp if arr[r, c] not in (BLACK, GRAY)]
        if not seeds:
            continue
        layers = len(seeds)
        gray_cells = [(r, c) for r, c in comp if arr[r, c] == GRAY]
        if not gray_cells:
            continue

        best = {}
        for sr, sc, color in seeds:
            dist = {(sr, sc): 0}
            q = deque([(sr, sc)])
            while q:
                cr, cc = q.popleft()
                cd = dist[(cr, cc)]
                if cd == layers:
                    continue
                for nr, nc in _neighbors(cr, cc, h, w):
                    if (nr, nc) in dist:
                        continue
                    if arr[nr, nc] != GRAY:
                        continue
                    nd = cd + 1
                    dist[(nr, nc)] = nd
                    q.append((nr, nc))

            for cell, d in dist.items():
                if d == 0:
                    continue
                old = best.get(cell)
                if old is None or d > old[0]:
                    best[cell] = (d, color, 1)
                elif d == old[0]:
                    best[cell] = (old[0], GRAY, old[2] + 1)

        for r, c in gray_cells:
            item = best.get((r, c))
            if item is None or item[2] > 1:
                out[r, c] = GRAY
            else:
                out[r, c] = item[1]

    return out


def _is_connected(mask):
    coords = list(zip(*np.nonzero(mask)))
    if not coords:
        return False
    seen = {coords[0]}
    q = deque([coords[0]])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _make_blob(height, width):
    mask = np.ones((height, width), dtype=bool)
    edge_cells = (
        [(0, c) for c in range(width)]
        + [(height - 1, c) for c in range(width)]
        + [(r, 0) for r in range(1, height - 1)]
        + [(r, width - 1) for r in range(1, height - 1)]
    )
    random.shuffle(edge_cells)
    for r, c in edge_cells:
        if random.random() > 0.35:
            continue
        mask[r, c] = False
        if mask.sum() < height * width * 0.62 or not _is_connected(mask):
            mask[r, c] = True
    return mask


def _interior_cells(mask):
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if all(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
                cells.append((r, c))
    return cells


def _pick_seed_positions(mask, count):
    candidates = _interior_cells(mask)
    if len(candidates) < count:
        candidates = list(zip(*np.nonzero(mask)))
    random.shuffle(candidates)

    if count == 2 and random.random() < 0.45:
        by_cell = set(candidates)
        pairs = []
        for r, c in candidates:
            for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                other = (r + dr, c + dc)
                mid = (r + dr // 2, c + dc // 2)
                if other in by_cell and mid in by_cell:
                    pairs.append(((r, c), other))
        if pairs:
            return list(random.choice(pairs))

    chosen = []
    for cell in candidates:
        if all(abs(cell[0] - old[0]) + abs(cell[1] - old[1]) >= 2 for old in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    for cell in candidates:
        if cell not in chosen:
            chosen.append(cell)
            if len(chosen) == count:
                break
    return chosen


def generate_input() -> np.ndarray:
    h = random.randint(13, 30)
    w = random.randint(13, 30)
    grid = np.zeros((h, w), dtype=int)

    target_regions = random.randint(2, 4)
    placed = 0
    attempts = 0
    while placed < target_regions and attempts < 120:
        attempts += 1
        bh = random.randint(4, min(9, h - 2))
        bw = random.randint(4, min(9, w - 2))
        top = random.randint(1, h - bh - 1)
        left = random.randint(1, w - bw - 1)
        r0 = max(0, top - 2)
        r1 = min(h, top + bh + 2)
        c0 = max(0, left - 2)
        c1 = min(w, left + bw + 2)
        if np.any(grid[r0:r1, c0:c1] != BLACK):
            continue

        mask = _make_blob(bh, bw)
        cells = list(zip(*np.nonzero(mask)))
        interior = _interior_cells(mask)
        max_seeds = min(5, len(interior) if len(interior) >= 2 else len(cells))
        if max_seeds < 1:
            continue

        if placed == 0:
            seed_count = random.choice([2, 3, min(4, max_seeds)])
            seed_count = min(seed_count, max_seeds)
        elif placed == 1 and max_seeds >= 1:
            seed_count = random.choice([1, 2, min(3, max_seeds)])
            seed_count = min(seed_count, max_seeds)
        else:
            seed_count = random.randint(1, max_seeds)

        seeds = _pick_seed_positions(mask, seed_count)
        if len(seeds) != seed_count:
            continue

        sub = grid[top : top + bh, left : left + bw]
        sub[mask] = GRAY
        colors = random.sample(SEED_COLORS, k=min(seed_count, len(SEED_COLORS)))
        while len(colors) < seed_count:
            colors.append(random.choice(SEED_COLORS))
        if seed_count > 1 and random.random() < 0.18:
            colors[-1] = colors[0]
        for (rr, cc), color in zip(seeds, colors):
            sub[rr, cc] = color
        placed += 1

    if placed == 0:
        grid[3:8, 3:8] = GRAY
        grid[5, 5] = random.choice(SEED_COLORS)

    return grid
