import random
from collections import deque

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_marker_colors(grid: np.ndarray) -> set[int]:
    colors = set(int(c) for c in np.unique(grid) if c not in (BLACK, RED))
    markers = set()
    for c in colors:
        comps = _components_of_color((grid == c))
        if all(len(comp) == 1 for comp in comps):
            markers.add(c)
    return markers


def _infer_period(grid: np.ndarray, markers: set[int]) -> int:
    h, w = grid.shape
    candidate_scores = []
    for q in range(3, 7):
        ok = True
        for r in range(h):
            row_pattern = {}
            for c in range(w):
                col = int(grid[r, c])
                if col in markers or col == RED or col == BLACK:
                    continue
                mod = c % q
                if mod in row_pattern and row_pattern[mod] != col:
                    ok = False
                    break
                row_pattern[mod] = col
            if not ok:
                break
        if ok:
            candidate_scores.append(q)
    if not candidate_scores:
        return 3
    return min(candidate_scores)


def _draw_path_with_markers(grid: np.ndarray, out: np.ndarray, start: int, cells: list[tuple[int, int]], marker_colors: set[int]) -> None:
    cur = RED
    for r, c in cells:
        col = int(grid[r, c])
        if col in marker_colors:
            cur = col
        out[r, c] = cur


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape

    red_cells = np.argwhere(grid == RED)
    if len(red_cells) != 1:
        return np.array([[BLACK]], dtype=int)
    sr, sc = map(int, red_cells[0])

    markers = _infer_marker_colors(grid)
    q = _infer_period(grid, markers)

    out = np.zeros((h, w), dtype=int)
    out[sr, sc] = RED

    # horizontal ray
    h_cells = [(sr, c) for c in range(sc, w)]
    _draw_path_with_markers(grid, out, sc, h_cells, markers)

    # perpendicular rays
    for c in range(sc, w, q):
        if not (0 <= c < w):
            continue
        v_cells = [(r, c) for r in range(sr + 1, h)]
        _draw_path_with_markers(grid, out, sc, v_cells, markers)

    return out


def _infer_tile_q():
    return 3


def generate_input() -> np.ndarray:
    h = 18
    w = 18
    while True:
        q = random.randint(3, 6)
        th = random.randint(3, 5)
        tw = q
        tile_colors = [BLUE, YELLOW, GREY]
        tile = np.array([[random.choice(tile_colors) for _ in range(tw)] for _ in range(th)], dtype=int)

        g = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                g[r, c] = tile[r % th, c % tw]

        # black occluders
        n_blk = random.randint(2, 4)
        for _ in range(n_blk):
            bh = random.randint(2, 4)
            bw = random.randint(2, 4)
            r0 = random.randint(0, h - bh)
            c0 = random.randint(0, w - bw)
            g[r0 : r0 + bh, c0 : c0 + bw] = BLACK

        # single red start on left edge
        sr = random.randint(2, h - 3)
        g[sr, 0] = RED

        # marker pixels
        marker_palette = [GREEN, PINK, ORANGE]
        random.shuffle(marker_palette)
        n_m = random.randint(1, 3)
        used = set()
        used.add((sr, 0))
        for i in range(n_m):
            col = marker_palette[i]
            for _ in range(500):
                r = random.randint(0, h - 1)
                c = random.randint(1, w - 1)
                if (r, c) in used:
                    continue
                if g[r, c] == BLACK:
                    continue
                # ensure marker not at start row's leftmost
                if r == sr and c == 0:
                    continue
                g[r, c] = col
                used.add((r, c))
                break

        # inferable by horizontal periodicity after removing markers
        markers = _infer_marker_colors(g)
        g2 = g.copy()
        for rm in markers | {RED}:
            g2[np.argwhere(g2 == rm)] = BLACK
        if _infer_period(g2, set()) in range(3, 7):
            return g


_generate_input_single = generate_input


def generate_inputs() -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    attempts = 0
    while len(outputs) < 20 and attempts < 200:
        attempts += 1
        g = _generate_input_single()
        arr = np.array(g, dtype=int, copy=True)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or arr.shape[0] > 30 or arr.shape[1] > 30:
            continue
        outputs.append(arr)

    while len(outputs) < 20:
        if outputs:
            outputs.append(outputs[-1].copy())
        else:
            outputs.append(np.zeros((1, 1), dtype=int))
    return outputs


    random.seed(0)
    print(main(generate_input()).shape)
