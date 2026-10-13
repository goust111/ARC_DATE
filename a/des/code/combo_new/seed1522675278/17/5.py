import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
RED = 2
BLUE = 1
PATTERN_COLOR = 8


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _frame_bboxes(grid):
    frames = []
    for cells in _components(grid == GREY):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        if expected.issubset(set(cells)):
            frames.append((r0, r1, c0, c1, len(cells)))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    frames = _frame_bboxes(grid)
    legend = max(frames, key=lambda x: (x[1], x[3], x[4]))
    lr0, lr1, lc0, lc1, _ = legend
    inner = grid[lr0 + 1 : lr1, lc0 + 1 : lc1]
    split_rows = [i for i in range(inner.shape[0]) if np.all(inner[i] == GREY)]
    bounds = [-1] + split_rows + [inner.shape[0]]
    patterns = {}
    for i in range(len(bounds) - 1):
        a, b = bounds[i] + 1, bounds[i + 1]
        if b - a <= 0:
            continue
        cell = inner[a:b]
        n = int((cell == BLUE).sum())
        mask = cell == PATTERN_COLOR
        if not mask.any():
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        patterns[n] = mask[rows][:, cols].astype(int) * PATTERN_COLOR

    for r0, r1, c0, c1, _ in frames:
        if (r0, r1, c0, c1) == legend[:4]:
            continue
        room = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere(room == RED)
        n = len(seeds)
        if n == 0:
            continue
        pat = patterns.get(n)
        if pat is None:
            continue
        center_r = (room.shape[0] - 1) / 2.0
        center_c = (room.shape[1] - 1) / 2.0
        mean_r, mean_c = seeds[:, 0].mean(), seeds[:, 1].mean()
        if mean_r < center_r and mean_c < center_c:
            rot = 0
        elif mean_r < center_r and mean_c >= center_c:
            rot = 1
        elif mean_r >= center_r and mean_c >= center_c:
            rot = 2
        else:
            rot = 3
        pat = np.rot90(pat, -rot)
        ph, pw = pat.shape
        top = r0 + 1 + room.shape[0] // 2 - ph // 2
        left = c0 + 1 + room.shape[1] // 2 - pw // 2
        patch = out[top : top + ph, left : left + pw]
        patch[pat != 0] = PATTERN_COLOR
        out[r0 + 1 : r1, c0 + 1 : c1][room == RED] = BG
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(22, 28), random.randint(22, 28)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    room_count = random.randint(2, 4)
    for _ in range(room_count):
        rh, rw = random.randint(5, 7), random.randint(5, 7)
        for _ in range(200):
            top = random.randint(1, h - rh - 10)
            left = random.randint(1, w - rw - 10)
            if occupied[top - 1 : top + rh + 1, left - 1 : left + rw + 1].any():
                continue
            bottom, right = top + rh - 1, left + rw - 1
            grid[top, left : right + 1] = GREY
            grid[bottom, left : right + 1] = GREY
            grid[top : bottom + 1, left] = GREY
            grid[top : bottom + 1, right] = GREY
            quadrant = random.choice(["ul", "ur", "ll", "lr"])
            seed_count = random.randint(1, 4)
            candidates = []
            for r in range(top + 1, bottom):
                for c in range(left + 1, right):
                    if quadrant == "ul" and r <= (top + bottom) // 2 and c <= (left + right) // 2:
                        candidates.append((r, c))
                    if quadrant == "ur" and r <= (top + bottom) // 2 and c >= (left + right) // 2:
                        candidates.append((r, c))
                    if quadrant == "ll" and r >= (top + bottom) // 2 and c <= (left + right) // 2:
                        candidates.append((r, c))
                    if quadrant == "lr" and r >= (top + bottom) // 2 and c >= (left + right) // 2:
                        candidates.append((r, c))
            random.shuffle(candidates)
            for r, c in candidates[:seed_count]:
                grid[r, c] = RED
            occupied[top : bottom + 1, left : right + 1] = True
            break

    legend_h, legend_w = 17, 8
    top, left = h - legend_h - 1, w - legend_w - 1
    bottom, right = top + legend_h - 1, left + legend_w - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    patterns = {
        1: np.array([[PATTERN_COLOR, 0], [PATTERN_COLOR, PATTERN_COLOR]], dtype=int),
        2: np.array([[PATTERN_COLOR, PATTERN_COLOR, 0], [0, PATTERN_COLOR, PATTERN_COLOR]], dtype=int),
        3: np.array([[0, PATTERN_COLOR, 0], [PATTERN_COLOR, PATTERN_COLOR, PATTERN_COLOR]], dtype=int),
        4: np.array([[PATTERN_COLOR, 0, PATTERN_COLOR], [0, PATTERN_COLOR, 0], [PATTERN_COLOR, PATTERN_COLOR, 0]], dtype=int),
    }
    cell_h = 4
    for idx in range(4):
        if idx > 0:
            grid[top + 1 + idx * cell_h - 1, left + 1 : right] = GREY
        cell_top = top + 1 + idx * cell_h
        cell_bottom = min(bottom - 1, cell_top + cell_h - 2)
        for j in range(idx + 1):
            grid[cell_top + j % max(1, cell_bottom - cell_top + 1), left + 1] = BLUE
        pat = patterns[idx + 1]
        ph, pw = pat.shape
        pt = cell_top
        pl = left + 3
        grid[pt : pt + ph, pl : pl + pw][pat != 0] = PATTERN_COLOR
    return grid
