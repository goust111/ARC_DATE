import random
from collections import deque
import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in np.argwhere(grid == color):
        sr = int(sr)
        sc = int(sc)
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
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _is_frame(comp):
    rows = [p[0] for p in comp]
    cols = [p[1] for p in comp]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    if bottom - top + 1 < 5 or right - left + 1 < 5:
        return False
    border = {
        *((top, c) for c in range(left, right + 1)),
        *((bottom, c) for c in range(left, right + 1)),
        *((r, left) for r in range(top, bottom + 1)),
        *((r, right) for r in range(top, bottom + 1)),
    }
    return set(comp) == border


def _find_frames(grid):
    frames = []
    for color in set(int(x) for x in np.unique(grid) if x != 5):
        for comp in _components(grid, color):
            if _is_frame(comp):
                rows = [p[0] for p in comp]
                cols = [p[1] for p in comp]
                frames.append((min(rows), min(cols), max(rows), max(cols), color))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    frames = _find_frames(grid)
    frame_cells = set()
    rect_areas = []
    for top, left, bottom, right, _ in frames:
        rect_areas.append((top, left, bottom, right))
        for c in range(left, right + 1):
            frame_cells.add((top, c))
            frame_cells.add((bottom, c))
        for r in range(top, bottom + 1):
            frame_cells.add((r, left))
            frame_cells.add((r, right))

    inside_any = set()
    for top, left, bottom, right in rect_areas:
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                inside_any.add((r, c))

    decoder = None
    for c in range(grid.shape[1]):
        rows = [r for r in range(grid.shape[0]) if grid[r, c] != 5 and (r, c) not in frame_cells and (r, c) not in inside_any]
        rows.sort()
        for i in range(len(rows) - 4):
            group = rows[i:i + 5]
            if group == list(range(group[0], group[0] + 5)):
                decoder = [int(grid[r, c]) for r in group]
                break
        if decoder is not None:
            break

    out = grid.copy()
    for top, left, bottom, right, _ in frames:
        pts = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right) if grid[r, c] != 5]
        key_col = max(set(c for _, c in pts), key=lambda c: sum(1 for _, cc in pts if cc == c))
        key_rows = sorted(r for r, c in pts if c == key_col)
        key_rows = key_rows[:5]
        key = [int(grid[r, key_col]) for r in key_rows]
        unlocked = [key[1], key[0], key[2], key[4], key[3]]
        if decoder is not None and unlocked == decoder:
            fill_color = key[2]
            key_cells = set(zip(key_rows, [key_col] * 5))
            for r in range(top + 1, bottom):
                for c in range(left + 1, right):
                    if (r, c) not in key_cells and out[r, c] == 5:
                        out[r, c] = fill_color

    if not frames:
        return out
    r0 = min(f[0] for f in frames)
    c0 = min(f[1] for f in frames)
    r1 = max(f[2] for f in frames)
    c1 = max(f[3] for f in frames)
    return out[r0:r1 + 1, c0:c1 + 1]


def _draw_frame(grid, top, left, h, w, color):
    bottom = top + h - 1
    right = left + w - 1
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    grid = np.full((19, 19), 5, dtype=int)
    placements = [(1, 1), (1, 11), (11, 1), (11, 11)]
    rng.shuffle(placements)
    n = rng.choice([3, 4])
    outline_colors = [1, 3, 7, 8]
    decoder = rng.sample([1, 2, 4, 6, 8, 9], 5)
    matches = set(rng.sample(range(n), rng.randint(1, n - 1)))
    for i in range(n):
        top, left = placements[i]
        color = outline_colors[i]
        _draw_frame(grid, top, left, 7, 7, color)
        if i in matches:
            key = [decoder[1], decoder[0], decoder[2], decoder[4], decoder[3]]
        else:
            while True:
                key = rng.sample([1, 2, 4, 6, 8, 9], 5)
                if [key[1], key[0], key[2], key[4], key[3]] != decoder:
                    break
        col = left + 3
        for j, value in enumerate(key):
            grid[top + 1 + j, col] = value
    dec_col = 9
    dec_top = rng.choice([1, 7, 13])
    for j, value in enumerate(decoder):
        grid[dec_top + j, dec_col] = value
    return grid
