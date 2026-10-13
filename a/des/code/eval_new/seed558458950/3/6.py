"""Propagate a fixed palette cycle along a one-cell path, advancing color at every 90° turn.
Input structure:
- Grid uses a dominant background color.
- One single-color-gray path of width one connects two border pixels through orthogonal moves.
- The first path pixel on the border is recolored with one palette color, while the rest is gray.
- A 3- or 4-pixel color sequence is drawn on a border corner as a row or column legend.
Rule:
- Read the sequence from the legend to define a cyclic color list.
- Start at the recolored path endpoint, paint the start segment with that color, and for each 90° turn on the path paint the turn pixel and all following cells with the next palette color (wrap around).
- Keep all non-path pixels, including the legend, unchanged.
Key insight:
- Path direction changes are the only triggers for color transitions; the legend only fixes the initial palette phase and cyclic order.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors(h: int, w: int, r: int, c: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if _in_bounds(h, w, nr, nc):
            out.append((nr, nc))
    return out


def _connected_components(cells: Iterable[Tuple[int, int]], h: int, w: int) -> List[List[Tuple[int, int]]]:
    remaining = set(cells)
    components: List[List[Tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors(h, w, r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        components.append(comp)
    return components


def _detect_path_color(grid: np.ndarray, bg: int) -> int | None:
    h, w = grid.shape
    best_color = None
    best_size = 0
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        comps = _connected_components(set(zip(*np.where(grid == color))), h, w)
        for comp in comps:
            if len(comp) < 4:
                continue
            cset = set(comp)
            endpoints = 0
            max_deg = 0
            for r, c in comp:
                deg = 0
                for nr, nc in _neighbors(h, w, r, c):
                    if (nr, nc) in cset:
                        deg += 1
                max_deg = max(max_deg, deg)
                if deg == 1:
                    endpoints += 1
            if max_deg > 2 or endpoints != 2:
                continue
            if len(comp) > best_size:
                best_size = len(comp)
                best_color = int(color)
    return best_color


def _read_legend(grid: np.ndarray, bg: int, grey: int) -> List[Tuple[Tuple[int, int], int]]:
    h, w = grid.shape
    candidates: List[Tuple[List[Tuple[int, int]], int]] = []
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}

    def valid_run(run: List[Tuple[int, int]]) -> bool:
        if not (3 <= len(run) <= 4):
            return False
        vals = [int(grid[r, c]) for r, c in run]
        if bg in vals or grey in vals:
            return False
        if len(set(vals)) != len(vals):
            return False
        if not any(pt in corners for pt in run):
            return False
        return True

    def add_run(run: List[Tuple[int, int]]) -> None:
        if valid_run(run):
            # score prefers corner-touching full length runs
            score = len(run) + (1 if any(pt in corners for pt in run) else 0)
            candidates.append((run, score))

    # top / bottom rows
    for r in (0, h - 1):
        c = 0
        while c < w:
            if int(grid[r, c]) != bg and int(grid[r, c]) != grey:
                run = [(r, c)]
                c2 = c + 1
                while c2 < w and int(grid[r, c2]) != bg and int(grid[r, c2]) != grey:
                    run.append((r, c2))
                    c2 += 1
                add_run(run)
                c = c2
            else:
                c += 1

    # left / right columns
    for c in (0, w - 1):
        r = 0
        while r < h:
            if int(grid[r, c]) != bg and int(grid[r, c]) != grey:
                run = [(r, c)]
                r2 = r + 1
                while r2 < h and int(grid[r2, c]) != bg and int(grid[r2, c]) != grey:
                    run.append((r2, c))
                    r2 += 1
                add_run(run)
                r = r2
            else:
                r += 1

    if not candidates:
        # fallback: any non-bg/grey border run len 3 or 4
        for r in (0, h - 1):
            for c in range(w - 1):
                c2 = c
                run = []
                while c2 < w and int(grid[r, c2]) != bg and int(grid[r, c2]) != grey:
                    run.append((r, c2))
                    c2 += 1
                if 3 <= len(run) <= 4:
                    candidates.append((run, len(run)))
                c = c2 if c2 > c else c + 1
        for c in (0, w - 1):
            for r in range(h - 1):
                r2 = r
                run = []
                while r2 < h and int(grid[r2, c]) != bg and int(grid[r2, c]) != grey:
                    run.append((r2, c))
                    r2 += 1
                if 3 <= len(run) <= 4:
                    candidates.append((run, len(run)))
                r = r2 if r2 > r else r + 1

    if not candidates:
        return []

    candidates.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
    selected, _ = candidates[0]
    return [(pt, int(grid[pt])) for pt in selected]


def _pick_start(grid: np.ndarray, bg: int, grey: int, legend: List[Tuple[Tuple[int, int], int]]) -> Tuple[int, int] | None:
    h, w = grid.shape
    cycle = [v for _, v in legend]
    if not cycle:
        return None
    cycle_set = set(cycle)
    legend_cells = {pt for pt, _ in legend}
    start_candidates = []

    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val == bg or val == grey:
                continue
            if val not in cycle_set:
                continue
            if (r, c) in legend_cells:
                continue
            neigh_grey = 0
            for nr, nc in _neighbors(h, w, r, c):
                if int(grid[nr, nc]) == grey:
                    neigh_grey += 1
            if neigh_grey:
                on_border = r == 0 or c == 0 or r == h - 1 or c == w - 1
                start_candidates.append((on_border, r, c, neigh_grey))

    if not start_candidates:
        return None

    # prefer border candidates (the start marker is on the boundary by rule)
    on_border = [x for x in start_candidates if x[0]]
    picked = on_border[0] if on_border else start_candidates[0]
    return picked[1], picked[2]


def _trace_path(grid: np.ndarray, start: Tuple[int, int], grey: int) -> List[Tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    neigh = [(nr, nc) for nr, nc in _neighbors(h, w, sr, sc) if int(grid[nr, nc]) == grey]
    if not neigh:
        return [start]

    path = [start]
    path_set = {start}
    cur = neigh[0]
    prev = start
    path.append(cur)
    path_set.add(cur)

    while True:
        r, c = cur
        next_cells = []
        for nr, nc in _neighbors(h, w, r, c):
            if (nr, nc) in path_set:
                continue
            if int(grid[nr, nc]) != grey:
                continue
            next_cells.append((nr, nc))
        if not next_cells:
            break
        nxt = next_cells[0]
        prev = cur
        cur = nxt
        path.append(cur)
        path_set.add(cur)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return np.array(grid, copy=True)

    out = np.array(grid, copy=True)
    h, w = out.shape

    vals, counts = np.unique(out, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    grey = _detect_path_color(out, bg)
    if grey is None:
        return out

    legend = _read_legend(out, bg, grey)
    if not legend:
        return out

    start = _pick_start(out, bg, grey, legend)
    if start is None:
        return out

    path_coords = _trace_path(out, start, grey)
    if len(path_coords) < 2:
        return out

    cycle = [v for _, v in legend]
    if not cycle:
        return out

    try:
        start_color = int(out[start])
        color_idx = cycle.index(start_color)
    except ValueError:
        color_idx = 0

    dirs = []
    for i in range(1, len(path_coords)):
        r1, c1 = path_coords[i - 1]
        r2, c2 = path_coords[i]
        dirs.append((r2 - r1, c2 - c1))

    curr = color_idx
    for i, (r, c) in enumerate(path_coords):
        if i >= 2 and dirs[i - 1] != dirs[i - 2]:
            curr = (curr + 1) % len(cycle)
        out[r, c] = cycle[curr]

    return out


def _build_segmented_path(h: int, w: int, start_edge: str, rng: np.random.Generator) -> List[Tuple[int, int]] | None:
    if start_edge == "top":
        if w < 4 or h < 4:
            return None
        max_turns = min(5, h - 2)
        turns = int(rng.integers(1, max_turns + 1))
        start_row = 0
        end_row = h - 1
        interior = sorted(rng.choice(np.arange(1, h - 1), size=turns, replace=False).tolist())
        row_pts = [start_row] + interior + [end_row]
        start_col = int(rng.integers(1, w - 1))
        col_pts = [start_col]
        for _ in range(turns):
            choices = [c for c in range(1, w - 1) if c != col_pts[-1]]
            if not choices:
                return None
            col_pts.append(int(rng.choice(choices)))

        out: List[Tuple[int, int]] = [(start_row, start_col)]
        col = start_col
        for seg in range(turns + 1):
            r_from = row_pts[seg]
            r_to = row_pts[seg + 1]
            step = 1 if r_to > r_from else -1
            rr = r_from + step
            while True:
                out.append((rr, col))
                if rr == r_to:
                    break
                rr += step
            if seg < turns:
                c_to = col_pts[seg + 1]
                step_c = 1 if c_to > col else -1
                cc = col + step_c
                while True:
                    out.append((r_to, cc))
                    if cc == c_to:
                        break
                    cc += step_c
                col = c_to
        return out

    if start_edge == "bottom":
        if w < 4 or h < 4:
            return None
        max_turns = min(5, h - 2)
        turns = int(rng.integers(1, max_turns + 1))
        start_row = h - 1
        end_row = 0
        interior = sorted(rng.choice(np.arange(1, h - 1), size=turns, replace=False).tolist(), reverse=True)
        row_pts = [start_row] + interior + [end_row]
        start_col = int(rng.integers(1, w - 1))
        col_pts = [start_col]
        for _ in range(turns):
            choices = [c for c in range(1, w - 1) if c != col_pts[-1]]
            if not choices:
                return None
            col_pts.append(int(rng.choice(choices)))

        out: List[Tuple[int, int]] = [(start_row, start_col)]
        col = start_col
        for seg in range(turns + 1):
            r_from = row_pts[seg]
            r_to = row_pts[seg + 1]
            step = 1 if r_to > r_from else -1
            rr = r_from + step
            while True:
                out.append((rr, col))
                if rr == r_to:
                    break
                rr += step
            if seg < turns:
                c_to = col_pts[seg + 1]
                step_c = 1 if c_to > col else -1
                cc = col + step_c
                while True:
                    out.append((r_to, cc))
                    if cc == c_to:
                        break
                    cc += step_c
                col = c_to
        return out

    if start_edge == "left":
        if h < 4 or w < 4:
            return None
        max_turns = min(5, w - 2)
        turns = int(rng.integers(1, max_turns + 1))
        start_col = 0
        end_col = w - 1
        interior = sorted(rng.choice(np.arange(1, w - 1), size=turns, replace=False).tolist())
        col_pts = [start_col] + interior + [end_col]
        start_row = int(rng.integers(1, h - 1))
        row_pts = [start_row]
        for _ in range(turns):
            choices = [r for r in range(1, h - 1) if r != row_pts[-1]]
            if not choices:
                return None
            row_pts.append(int(rng.choice(choices)))

        out = [(start_row, start_col)]
        row = start_row
        for seg in range(turns + 1):
            c_from = col_pts[seg]
            c_to = col_pts[seg + 1]
            step = 1 if c_to > c_from else -1
            cc = c_from + step
            while True:
                out.append((row, cc))
                if cc == c_to:
                    break
                cc += step
            if seg < turns:
                r_to = row_pts[seg + 1]
                step_r = 1 if r_to > row else -1
                rr = row + step_r
                while True:
                    out.append((rr, c_to))
                    if rr == r_to:
                        break
                    rr += step_r
                row = r_to
        return out

    if start_edge == "right":
        if h < 4 or w < 4:
            return None
        max_turns = min(5, w - 2)
        turns = int(rng.integers(1, max_turns + 1))
        start_col = w - 1
        end_col = 0
        interior = sorted(rng.choice(np.arange(1, w - 1), size=turns, replace=False).tolist(), reverse=True)
        col_pts = [start_col] + interior + [end_col]
        start_row = int(rng.integers(1, h - 1))
        row_pts = [start_row]
        for _ in range(turns):
            choices = [r for r in range(1, h - 1) if r != row_pts[-1]]
            if not choices:
                return None
            row_pts.append(int(rng.choice(choices)))

        out = [(start_row, start_col)]
        row = start_row
        for seg in range(turns + 1):
            c_from = col_pts[seg]
            c_to = col_pts[seg + 1]
            step = 1 if c_to > c_from else -1
            cc = c_from + step
            while True:
                out.append((row, cc))
                if cc == c_to:
                    break
                cc += step
            if seg < turns:
                r_to = row_pts[seg + 1]
                step_r = 1 if r_to > row else -1
                rr = row + step_r
                while True:
                    out.append((rr, c_to))
                    if rr == r_to:
                        break
                    rr += step_r
                row = r_to
        return out

    return None


def _legend_candidates(h: int, w: int, L: int) -> List[List[Tuple[int, int]]]:
    candidates: List[List[Tuple[int, int]]] = []
    if w >= L:
        candidates.append([(0, c) for c in range(L)])
        candidates.append([(h - 1, c) for c in range(L)])
        candidates.append([(0, w - L + c) for c in range(L)])
        candidates.append([(h - 1, w - L + c) for c in range(L)])
    if h >= L:
        candidates.append([(r, 0) for r in range(L)])
        candidates.append([(r, w - 1) for r in range(L)])
        candidates.append([(h - L + r, 0) for r in range(L)])
        candidates.append([(h - L + r, w - 1) for r in range(L)])
    return candidates


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    color_pool = list(range(1, 10))
    max_trials = 400

    for _ in range(max_trials):
        h = int(rng.integers(12, 29))
        w = int(rng.integers(12, 29))

        bg = 0
        start_edge = rng.choice(["top", "bottom", "left", "right"])
        path = _build_segmented_path(h, w, start_edge, rng)
        if not path:
            continue

        palette_len = int(rng.choice([3, 4]))
        if len(color_pool) < palette_len + 1:
            continue
        path_grey = int(rng.choice(color_pool))
        remaining = [c for c in color_pool if c != path_grey]
        palette = rng.choice(remaining, size=palette_len, replace=False).astype(int).tolist()
        start_color = int(rng.choice(palette))

        grid = np.full((h, w), bg, dtype=np.int8)
        for i, (r, c) in enumerate(path):
            grid[r, c] = start_color if i == 0 else path_grey

        path_set = set(path)
        legend_runs = _legend_candidates(h, w, palette_len)
        rng.shuffle(legend_runs)

        placed = False
        for run in legend_runs:
            if any(pt in path_set for pt in run):
                continue
            if any(grid[r, c] != bg for r, c in run):
                continue
            run = [pt for pt in run if 0 <= pt[0] < h and 0 <= pt[1] < w]
            if len(run) != len(palette):
                continue
            for (r, c), color in zip(run, palette):
                grid[r, c] = color
            placed = True
            break

        if not placed:
            continue

        return grid

    # Fallback: tiny conservative template-like sample
    h, w = 16, 16
    grid = np.full((h, w), 0, dtype=np.int8)
    path = _build_segmented_path(h, w, "top", np.random.default_rng(0))
    palette = [1, 2, 3, 4]
    start_color = 1
    if path is None:
        path = [(0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (7, 5), (7, 6), (8, 6), (9, 6)]
    for i, (r, c) in enumerate(path):
        grid[r, c] = start_color if i == 0 else 9
    grid[0, 0:len(palette)] = palette
    return grid
