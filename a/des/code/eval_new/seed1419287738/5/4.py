import random
from collections import Counter, deque

import numpy as np


BG = 0
BASE_MAIN = 2
BASE_SUB = 3
OCCLUDER_COLORS = [4, 6, 7, 8, 9]


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in zip(*np.where(grid == color)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, h, w):
                if not seen[nr, nc] and grid[nr, nc] == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(cells)
    return comps


def _is_rect(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _occluders(grid, base_colors):
    occ = []
    for color in set(int(v) for v in grid.ravel()) - {BG} - set(base_colors):
        for cells in _components(grid, color):
            if _is_rect(cells):
                occ.append((color, cells))
    return occ


def _pattern_box(grid, base_colors):
    coords = np.argwhere(np.isin(grid, list(base_colors)))
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    size = max(int(r1 - r0 + 1), int(c1 - c0 + 1))
    return int(r0), int(c0), size


def _rotates(r, c, n):
    return [(c, n - 1 - r), (n - 1 - r, n - 1 - c), (n - 1 - c, r)]


def main(grid: np.ndarray) -> np.ndarray:
    """Recover occluded cells of a 90-degree rotationally symmetric pattern.

    Input structure:
    A large two-color rotationally symmetric pattern lies on black background, with
    one or more solid-color rectangular occluders drawn over part of the pattern.

    Rule:
    Rectangular components whose colors are not the two base pattern colors are
    removed.  Each covered cell is restored from an unoccluded 90/180/270-degree
    counterpart.  Restored main-color cells become the occluder's color, while
    restored secondary-color cells remain secondary-colored.

    Key insight:
    The hidden content is determined by rotational symmetry, but the occluder's color
    is still meaningful because it replaces recovered main-color pixels.
    """
    out = np.array(grid, copy=True)
    counts = Counter(int(v) for v in grid.ravel() if v != BG)
    if len(counts) < 3:
        return out
    base_colors = [color for color, _ in counts.most_common(2)]
    main_color, sub_color = base_colors[0], base_colors[1]
    box = _pattern_box(grid, base_colors)
    if box is None:
        return out
    r0, c0, n = box
    for occ_color, cells in _occluders(grid, base_colors):
        for r, c in cells:
            lr, lc = r - r0, c - c0
            restored = None
            if 0 <= lr < n and 0 <= lc < n:
                for rr, cc in _rotates(lr, lc, n):
                    ar, ac = r0 + rr, c0 + cc
                    if 0 <= ar < grid.shape[0] and 0 <= ac < grid.shape[1] and int(grid[ar, ac]) in base_colors:
                        restored = int(grid[ar, ac])
                        break
            out[r, c] = occ_color if restored == main_color else (sub_color if restored == sub_color else BG)
    return out


def _orbits(n):
    seen = set()
    orbits = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = {(r, c)}
            cur = (r, c)
            for _ in range(3):
                cur = (cur[1], n - 1 - cur[0])
                orbit.add(cur)
            seen |= orbit
            orbits.append(sorted(orbit))
    return orbits


def _build_pattern(n):
    pat = np.zeros((n, n), dtype=int)
    for orbit in _orbits(n):
        color = BASE_MAIN if random.random() < 0.7 else BASE_SUB
        for r, c in orbit:
            pat[r, c] = color
    pat[0, 0] = pat[0, n - 1] = pat[n - 1, 0] = pat[n - 1, n - 1] = BASE_MAIN
    return pat


def generate_input() -> np.ndarray:
    for _ in range(1000):
        n = random.choice([9, 11, 13])
        h = random.randint(n + 6, 29)
        w = random.randint(n + 6, 29)
        top = random.randint(2, h - n - 2)
        left = random.randint(2, w - n - 2)
        grid = np.zeros((h, w), dtype=int)
        grid[top:top + n, left:left + n] = _build_pattern(n)
        n_occ = random.randint(1, 3)
        colors = random.sample(OCCLUDER_COLORS, n_occ)
        used = np.zeros((h, w), dtype=bool)
        ok = True
        for color in colors:
            rh = random.randint(2, 3)
            rw = random.randint(2, 4)
            r0 = random.randint(top + 1, top + n - rh - 1)
            c0 = random.randint(left + 1, left + n - rw - 1)
            if np.any(used[r0:r0 + rh, c0:c0 + rw]):
                ok = False
                break
            used[r0:r0 + rh, c0:c0 + rw] = True
            grid[r0:r0 + rh, c0:c0 + rw] = color
        if ok and not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid occluded-symmetry puzzle")
