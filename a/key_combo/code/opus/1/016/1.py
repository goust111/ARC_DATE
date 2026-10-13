import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
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
        comps.append(comp)
    return comps


def _settle(grid, frame_mask, comps):
    occupied = frame_mask.copy()
    placed = []
    for comp in sorted(comps, key=lambda p: max(r for r, _ in p), reverse=True):
        color = int(grid[comp[0]])
        pts = [(r, c) for r, c in comp]
        while True:
            moved = [(r + 1, c) for r, c in pts]
            if any(r >= grid.shape[0] or occupied[r, c] for r, c in moved):
                break
            pts = moved
        for r, c in pts:
            occupied[r, c] = True
        placed.append((pts, color))
    return placed


def _external_zero(arr):
    h, w = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if arr[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if arr[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == 0 and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    green = np.argwhere(grid == 3)
    fr0, fc0 = green.min(axis=0)
    fr1, fc1 = green.max(axis=0)
    frame_mask = grid == 3
    fragments = _components((grid != 0) & (grid != 3))
    settled = _settle(grid, frame_mask, fragments)
    out = np.zeros_like(grid)
    out[frame_mask] = 3
    for pts, color in settled:
        for r, c in pts:
            out[r, c] = color
    exterior = _external_zero(out)
    enclosed = (out == 0) & (~exterior)
    opening_cols = range(fc0 + 1, fc1)
    if np.any(enclosed):
        out[enclosed] = 6
        for pts, _color in settled:
            if any(c in opening_cols and r <= fr0 for r, c in pts):
                for r, c in pts:
                    out[r, c] = 2
    else:
        for pts, _color in settled:
            if any(fr0 <= r <= fr1 and fc0 <= c <= fc1 for r, c in pts):
                for r, c in pts:
                    if fr0 <= r <= fr1 and fc0 <= c <= fc1:
                        out[r, c] = 5
    out[frame_mask] = 3
    return out


def _draw_u(grid):
    top, bottom, left, right = 7, 12, 3, 8
    grid[top:bottom + 1, left] = 3
    grid[top:bottom + 1, right] = 3
    grid[bottom, left:right + 1] = 3
    return top, bottom, left, right


def generate_input() -> np.ndarray:
    grid = np.zeros((14, 14), dtype=int)
    top, _bottom, left, right = _draw_u(grid)
    closed = random.random() < 0.5
    colors = random.sample([1, 4, 6, 7, 8, 9], 4)
    if closed:
        row = random.randint(1, 3)
        grid[row, left:left + 3] = colors[0]
        grid[row - 1 if row > 1 else row + 1, right - 2:right + 1] = colors[1]
    else:
        row = random.randint(1, 3)
        grid[row, left + 1:left + 3] = colors[0]
        grid[row + 2, left + 3:right] = colors[1]
    grid[random.randint(1, 4), 1:3] = colors[2]
    grid[random.randint(1, 4), 10:12] = colors[3]
    return grid
