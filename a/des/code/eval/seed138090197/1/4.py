from __future__ import annotations

from collections import deque
import random

import numpy as np


GRAY = 5
BG = 0
LINE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
LINE_RETRIES = 120


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _path_color(grid: np.ndarray, bg: int) -> int:
    if np.any(grid == GRAY):
        return GRAY
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    assert colors, "expected a non-background path color"
    return max(colors, key=lambda c: int(np.sum(grid == c)))


def _find_path_mask(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    path_color = _path_color(g, bg)

    path = g == path_color
    for r in range(h):
        for c in range(w):
            if path[r, c] or int(g[r, c]) in (bg, path_color):
                continue
            if 0 < c < w - 1 and int(g[r, c - 1]) == path_color and int(g[r, c + 1]) == path_color:
                path[r, c] = True
                continue
            if 0 < r < h - 1 and int(g[r - 1, c]) == path_color and int(g[r + 1, c]) == path_color:
                path[r, c] = True

    return path, path_color, bg


def _ordered_path(path_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(path_mask)]
    assert coords, "expected a path"

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        nbrs = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and path_mask[nr, nc]:
                nbrs.append((int(nr), int(nc)))
        assert 1 <= len(nbrs) <= 2, "path must be a non-branching chain"
        adj[(r, c)] = nbrs

    endpoints = [p for p, nbrs in adj.items() if len(nbrs) == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start = min(endpoints)
    order = [start]
    seen = {start}
    prev = None
    curr = start
    while True:
        nxts = [p for p in adj[curr] if p != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must stay unique"
        nxt = nxts[0]
        assert nxt not in seen, "path must not loop"
        order.append(nxt)
        seen.add(nxt)
        prev, curr = curr, nxt

    assert len(order) == len(coords), "path must be connected"
    return order


def _components(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comp_id = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            cid = len(comps)
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp_id[r, c] = cid
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        comp_id[nr, nc] = cid
                        q.append((int(nr), int(nc)))
            comps.append(pts)

    return comps, comp_id


def _side_components(path: list[tuple[int, int]], comp_id: np.ndarray, comp_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = comp_id.shape
    first_left: dict[int, int] = {}
    first_right: dict[int, int] = {}

    for i in range(len(path) - 1):
        r, c = path[i]
        nr, nc = path[i + 1]
        dr, dc = nr - r, nc - c
        left = (-dc, dr)
        right = (dc, -dr)

        for x, y in ((r + left[0], c + left[1]), (nr + left[0], nc + left[1])):
            if 0 <= x < h and 0 <= y < w:
                cid = int(comp_id[x, y])
                if cid >= 0:
                    first_left.setdefault(cid, i)

        for x, y in ((r + right[0], c + right[1]), (nr + right[0], nc + right[1])):
            if 0 <= x < h and 0 <= y < w:
                cid = int(comp_id[x, y])
                if cid >= 0:
                    first_right.setdefault(cid, i)

    left_comps: list[tuple[int, int]] = []
    right_comps: list[tuple[int, int]] = []
    for cid in range(comp_count):
        li = first_left.get(cid)
        ri = first_right.get(cid)
        if li is None and ri is None:
            continue
        if ri is None or (li is not None and li <= ri):
            left_comps.append((int(li), cid))
        else:
            right_comps.append((int(ri), cid))

    left_comps.sort()
    right_comps.sort()
    return left_comps, right_comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the grey path from the lexicographically earlier endpoint, collect the
    colors of crossed perpendicular line segments, then fill the left-side
    pockets in that same order while leaving the path and the line segments
    unchanged.
    """
    g = np.asarray(grid, dtype=int)
    path_mask, path_color, bg = _find_path_mask(g)
    path = _ordered_path(path_mask)

    colors = [int(g[r, c]) for r, c in path if int(g[r, c]) != path_color]

    comps, comp_id = _components(~path_mask)
    left_comps, _right_comps = _side_components(path, comp_id, len(comps))

    out = g.copy()
    for (_first_idx, cid), color in zip(left_comps, colors, strict=False):
        for r, c in comps[cid]:
            if int(out[r, c]) == bg:
                out[r, c] = int(color)

    return out


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:], strict=False):
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for c in range(c0, c1 + step, step):
                cell = (int(r0), int(c))
                if not path or path[-1] != cell:
                    path.append(cell)
        else:
            step = 1 if r1 > r0 else -1
            for r in range(r0, r1 + step, step):
                cell = (int(r), int(c0))
                if not path or path[-1] != cell:
                    path.append(cell)
    return path


def _build_canonical_path() -> tuple[int, int, list[tuple[int, int]]]:
    sweep_count = random.choice([1, 3, 5])
    width = random.randint(12, 24)

    first_row = random.randint(2, 4)
    rows = [first_row]
    for _ in range(sweep_count - 1):
        rows.append(rows[-1] + random.randint(3, 5))
    height = rows[-1] + random.randint(2, 5) + 1
    if height < 10:
        height = 10
    if height > 30:
        raise AssertionError("canonical path too tall")

    start_col = random.randint(3, width - 4)
    side = random.choice([0, width - 1])

    points = [(0, start_col)]
    curr = points[0]
    for row in rows:
        if curr[0] != row:
            points.append((row, curr[1]))
        if curr[1] != side:
            points.append((row, side))
        curr = points[-1]
        side = width - 1 if side == 0 else 0
    if curr[0] != height - 1:
        points.append((height - 1, curr[1]))

    return height, width, _polyline(points)


def _paint_path(height: int, width: int, path: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r, c in path:
        grid[r, c] = GRAY
    return grid


def _run_capacity(path_mask: np.ndarray, used_cells: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> int:
    h, w = path_mask.shape
    cap = 0
    while cap < 3:
        nr = r + (cap + 1) * dr
        nc = c + (cap + 1) * dc
        if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
            break
        if path_mask[nr, nc] or (nr, nc) in used_cells:
            break
        cap += 1
    return cap


def _line_options(path: list[tuple[int, int]], path_mask: np.ndarray, used_cells: set[tuple[int, int]], idx: int) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    h, w = path_mask.shape
    if not (1 <= idx < len(path) - 1):
        return []

    r, c = path[idx]
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return []

    pr, pc = path[idx - 1]
    nr, nc = path[idx + 1]
    d1 = (r - pr, c - pc)
    d2 = (nr - r, nc - c)
    if d1 != d2:
        return []

    pdr, pdc = -d1[1], d1[0]
    pos_cap = _run_capacity(path_mask, used_cells, r, c, pdr, pdc)
    neg_cap = _run_capacity(path_mask, used_cells, r, c, -pdr, -pdc)

    options: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    for pos_len in range(pos_cap + 1):
        for neg_len in range(neg_cap + 1):
            if not (1 <= pos_len + neg_len <= 3):
                continue
            cells = [(r + step * pdr, c + step * pdc) for step in range(1, pos_len + 1)]
            cells += [(r - step * pdr, c - step * pdc) for step in range(1, neg_len + 1)]
            if any(cell in used_cells for cell in cells):
                continue
            options.append((cells, (pdr, pdc)))
    return options


def _collect_intersection_count(grid: np.ndarray) -> int:
    path_mask, path_color, _bg = _find_path_mask(grid)
    path = _ordered_path(path_mask)
    return sum(1 for r, c in path if int(grid[r, c]) != path_color)


def _left_pocket_count(grid: np.ndarray) -> int:
    path_mask, _path_color, _bg = _find_path_mask(grid)
    path = _ordered_path(path_mask)
    comps, comp_id = _components(~path_mask)
    left_comps, _right_comps = _side_components(path, comp_id, len(comps))
    return len(left_comps)


def _place_lines(base_grid: np.ndarray, path: list[tuple[int, int]], line_count: int) -> np.ndarray | None:
    path_mask = base_grid == GRAY

    for _ in range(LINE_RETRIES):
        grid = base_grid.copy()
        used_cells: set[tuple[int, int]] = set()
        chosen: list[int] = []

        ok = True
        for _line_idx in range(line_count):
            candidates: list[tuple[int, list[tuple[list[tuple[int, int]], tuple[int, int]]]]] = []
            for idx in range(1, len(path) - 1):
                if any(abs(idx - prev_idx) < 3 for prev_idx in chosen):
                    continue
                options = _line_options(path, path_mask, used_cells, idx)
                if options:
                    candidates.append((idx, options))
            if not candidates:
                ok = False
                break

            idx, options = random.choice(candidates)
            cells, _perp = random.choice(options)
            color = random.choice(LINE_COLORS)
            r, c = path[idx]
            grid[r, c] = color
            for rr, cc in cells:
                grid[rr, cc] = color
                used_cells.add((rr, cc))
            chosen.append(idx)

        if ok:
            return grid

    return None


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rot = random.randrange(4)
    g = np.rot90(g, rot)
    if random.choice([False, True]):
        g = np.fliplr(g)
    if random.choice([False, True]):
        g = np.flipud(g)
    return np.array(g, copy=True)


def _sample_grid() -> np.ndarray:
    height, width, path = _build_canonical_path()
    base = _paint_path(height, width, path)

    left_count = _left_pocket_count(base)
    right_count = len(_components(~(base == GRAY))[0]) - left_count
    assert left_count >= 1 and left_count == right_count, "canonical layout must balance left/right pockets"

    grid = _place_lines(base, path, left_count)
    if grid is None:
        raise AssertionError("failed to place line segments")

    return _transform(grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _sample_grid()
            if _collect_intersection_count(grid) != _left_pocket_count(grid):
                continue
            out = main(grid)
            if np.array_equal(out, grid):
                continue
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
