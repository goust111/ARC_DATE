import random
from collections import defaultdict

import numpy as np


def _components(grid):
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] != 0:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _find_windows(grid):
    windows = []
    for cells in _components(grid):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h not in (3, 4):
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        zeros = np.argwhere(patch == 0)
        nonzeros = patch[patch != 0]
        if len(zeros) != 1 or len(nonzeros) != h * h - 1:
            continue
        colors = set(int(v) for v in nonzeros)
        if len(colors) != 1:
            continue

        hr, hc = (int(v) for v in zeros[0])
        windows.append(
            {
                "size": h,
                "hole": (hr, hc),
                "abs_hole": (r0 + hr, c0 + hc),
                "color": int(nonzeros[0]),
                "origin": (r0, c0),
            }
        )
    windows.sort(key=lambda item: item["origin"])
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each missing window cell with the border color of its rotated-position partner.

    Input structure:
    The grid has a black background with separated 3x3 or 4x4 single-color window blocks. Each window is a solid non-black square except for exactly one black cell inside its square footprint; that black cell marks the missing pane position.

    Rule:
    Identify every window, record its square size, color, and missing-cell coordinate relative to the window. Pair windows of the same size whose missing coordinates are related by a 180-degree rotation within that size. For every pair, fill each missing cell with the other window's border color and leave all other cells unchanged.

    Key insight:
    The fill color is not the window's own color; it comes from the uniquely matched partner found by the geometric complement of the missing pane position.
    """
    out = np.array(grid, copy=True)
    buckets = defaultdict(list)
    for window in _find_windows(grid):
        buckets[(window["size"], window["hole"])].append(window)

    used = set()
    for window in _find_windows(grid):
        wid = (window["origin"], window["hole"])
        if wid in used:
            continue
        n = window["size"]
        hr, hc = window["hole"]
        target_key = (n, (n - 1 - hr, n - 1 - hc))
        candidates = buckets.get(target_key, [])
        partner = None
        for candidate in candidates:
            cid = (candidate["origin"], candidate["hole"])
            if cid != wid and cid not in used:
                partner = candidate
                break
        if partner is None:
            continue

        ar, ac = window["abs_hole"]
        br, bc = partner["abs_hole"]
        out[ar, ac] = partner["color"]
        out[br, bc] = window["color"]
        used.add(wid)
        used.add((partner["origin"], partner["hole"]))
    return out


def _place_windows(grid, windows, rng):
    rows, cols = grid.shape
    placed = []
    order = list(range(len(windows)))
    rng.shuffle(order)

    for idx in order:
        size = windows[idx]["size"]
        found = False
        for _ in range(800):
            r = rng.randrange(1, rows - size)
            c = rng.randrange(1, cols - size)
            padded = (r - 1, c - 1, r + size, c + size)
            if any(
                not (padded[2] < pr0 or padded[0] > pr1 or padded[3] < pc0 or padded[1] > pc1)
                for pr0, pc0, pr1, pc1 in placed
            ):
                continue
            color = windows[idx]["color"]
            hr, hc = windows[idx]["hole"]
            grid[r : r + size, c : c + size] = color
            grid[r + hr, c + hc] = 0
            placed.append((r, c, r + size - 1, c + size - 1))
            found = True
            break
        if not found:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = list(range(1, 10))

    for _ in range(200):
        rows = rng.randint(14, 30)
        cols = rng.randint(14, 30)
        max_pairs = rng.randint(1, 5)

        orbits = []
        for n in (3, 4):
            seen = set()
            for r in range(n):
                for c in range(n):
                    p = (r, c)
                    q = (n - 1 - r, n - 1 - c)
                    if p in seen or q in seen:
                        continue
                    seen.add(p)
                    seen.add(q)
                    orbits.append((n, p, q))
        rng.shuffle(orbits)

        pair_count = rng.randint(1, min(max_pairs, len(orbits)))
        windows = []
        for n, p, q in orbits[:pair_count]:
            color_a, color_b = rng.sample(colors, 2)
            windows.append({"size": n, "hole": p, "color": color_a})
            windows.append({"size": n, "hole": q, "color": color_b})

        grid = np.zeros((rows, cols), dtype=int)
        if _place_windows(grid, windows, rng):
            return grid

    rows, cols = 16, 16
    grid = np.zeros((rows, cols), dtype=int)
    grid[2:5, 2:5] = 2
    grid[2, 2] = 0
    grid[9:12, 9:12] = 7
    grid[11, 11] = 0
    return grid
