from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 4  # yellow
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _cross_cells(h: int, w: int, cr: int, cc: int, arm_len: int) -> set[tuple[int, int]]:
    cells = {(int(cr), int(cc))}
    for d in range(1, arm_len + 1):
        for rr, cc2 in ((cr - d, cc), (cr + d, cc), (cr, cc - d), (cr, cc + d)):
            if 0 <= rr < h and 0 <= cc2 < w:
                cells.add((int(rr), int(cc2)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    out = g.copy()

    objects: list[tuple[int, list[tuple[int, int]]]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _components_of_color(g == int(color)):
            objects.append((int(color), comp))

    for _color, comp in objects:
        for r, c in comp:
            out[r, c] = int(bg)

    for color, comp in objects:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        cr = (min(rows) + max(rows)) // 2
        cc = (min(cols) + max(cols)) // 2
        arm_len = len(comp) // 2
        for r, c in _cross_cells(h, w, int(cr), int(cc), int(arm_len)):
            out[r, c] = int(color)

    return out


def _dilate(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out = set(cells)
    frontier = set(cells)
    for _ in range(radius):
        nxt = set(frontier)
        for r, c in frontier:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        nxt.add((int(rr), int(cc)))
        out |= nxt
        frontier = nxt
    return out


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(int(r), int(c))]
    if random.random() < 0.5:
        while c != tc:
            c += 1 if tc > c else -1
            cells.append((int(r), int(c)))
        while r != tr:
            r += 1 if tr > r else -1
            cells.append((int(r), int(c)))
    else:
        while r != tr:
            r += 1 if tr > r else -1
            cells.append((int(r), int(c)))
        while c != tc:
            c += 1 if tc > c else -1
            cells.append((int(r), int(c)))
    return cells


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _shape_is_irregular(cells: set[tuple[int, int]]) -> bool:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1 or len(cols) == 1:
        return False
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if len(cells) == area:
        return False
    degs = []
    for r, c in cells:
        deg = 0
        for dr, dc in DIR4:
            if (r + dr, c + dc) in cells:
                deg += 1
        degs.append(deg)
    return any(d == 1 for d in degs) and any(d >= 2 for d in degs)


def _random_connected_shape(
    h: int,
    w: int,
    center: tuple[int, int],
    n_cells: int,
) -> set[tuple[int, int]] | None:
    cr, cc = center
    top_options = [t for t in range(0, min(4, cr) + 1)]
    if not top_options:
        return None
    for _ in range(120):
        top = random.choice(top_options)
        bottom = top + random.choice([0, 1])
        if cr + bottom >= h:
            continue
        left_options = [l for l in range(0, min(4, cc) + 1)]
        if not left_options:
            continue
        left = random.choice(left_options)
        right = left + random.choice([0, 1])
        if cc + right >= w:
            continue

        r0, r1 = cr - top, cr + bottom
        c0, c1 = cc - left, cc + right
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh * bw < n_cells:
            continue

        top_side = [(0, c) for c in range(bw)]
        bottom_side = [(bh - 1, c) for c in range(bw)]
        left_side = [(r, 0) for r in range(bh)]
        right_side = [(r, bw - 1) for r in range(bh)]
        anchors = {
            random.choice(top_side),
            random.choice(bottom_side),
            random.choice(left_side),
            random.choice(right_side),
        }
        anchors.add((cr - r0, cc - c0))
        base = {next(iter(anchors))}
        for anchor in list(anchors)[1:]:
            path = _path_between(next(iter(base)), anchor)
            base.update(path)
        if len(base) > n_cells:
            continue

        cells = set(base)
        while len(cells) < n_cells:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < bh and 0 <= nc < bw and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != n_cells:
            continue

        if min(r for r, _ in cells) != 0 or max(r for r, _ in cells) != bh - 1:
            continue
        if min(c for _, c in cells) != 0 or max(c for _, c in cells) != bw - 1:
            continue
        if not _shape_is_irregular(cells):
            continue

        global_cells = {(int(r + r0), int(c + c0)) for r, c in cells}
        if len(global_cells) != n_cells:
            continue
        rows = [r for r, _ in global_cells]
        cols = [c for _, c in global_cells]
        if ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2) != (cr, cc):
            continue
        return global_cells
    return None


def _candidate_centers(h: int, w: int, arm_len: int, truncated: bool) -> list[tuple[int, int]]:
    centers = []
    for r in range(h):
        for c in range(w):
            fits = arm_len <= r <= h - 1 - arm_len and arm_len <= c <= w - 1 - arm_len
            if fits != truncated:
                continue
            centers.append((int(r), int(c)))
    random.shuffle(centers)
    return centers


def _choose_specs(count: int) -> list[tuple[int, bool]]:
    base = [(5, False), (4, True), (10, False), (7, True)]
    specs = base[:count]
    while len(specs) < count:
        n = random.randint(4, 10)
        specs.append((n, random.random() < 0.5))
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 18)
    w = random.randint(10, 18)
    grid = np.full((h, w), BG, dtype=int)

    count = random.randint(2, 4)
    colors = random.sample([c for c in range(10) if c != BG], count)
    specs = _choose_specs(count)

    reserved_output: set[tuple[int, int]] = set()
    reserved_input: set[tuple[int, int]] = set()

    for color, (n_cells, truncated) in zip(colors, specs, strict=False):
        arm_len = n_cells // 2
        placed = False
        for cr, cc in _candidate_centers(h, w, arm_len, truncated):
            cross = _cross_cells(h, w, cr, cc, arm_len)
            if reserved_output & _dilate(cross, h, w, radius=1):
                continue
            shape = _random_connected_shape(h, w, (cr, cc), n_cells)
            if shape is None:
                continue
            if reserved_input & _dilate(shape, h, w, radius=1):
                continue
            if cross & shape and len(shape) >= 8 and random.random() < 0.3:
                continue
            for r, c in shape:
                grid[r, c] = int(color)
            reserved_output |= _dilate(cross, h, w, radius=1)
            reserved_input |= _dilate(shape, h, w, radius=1)
            placed = True
            break
        if not placed:
            return None

    if len({tuple(row) for row in grid.tolist()}) < 2:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
