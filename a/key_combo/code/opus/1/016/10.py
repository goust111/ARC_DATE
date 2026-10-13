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


def main(grid: np.ndarray) -> np.ndarray:
    crops = []
    for comp in _components(grid != 0):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        crop = grid[r0:r1 + 1, c0:c1 + 1]
        area = int(np.sum(crop != 0))
        crops.append((area, crop))
    crops.sort(key=lambda x: x[0], reverse=True)
    max_h = max(crop.shape[0] for _, crop in crops)
    max_w = max(crop.shape[1] for _, crop in crops)
    canvas = np.zeros((max_h, max_w), dtype=int)
    for _area, crop in crops:
        r0 = (max_h - crop.shape[0]) // 2
        c0 = (max_w - crop.shape[1]) // 2
        for r in range(crop.shape[0]):
            for c in range(crop.shape[1]):
                if crop[r, c] != 0:
                    canvas[r0 + r, c0 + c] = crop[r, c]
    seen = []
    for value in canvas.ravel():
        v = int(value)
        if v != 0 and v not in seen:
            seen.append(v)
    return np.array([seen], dtype=int)


def _place(grid, top, left, arr):
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r, c]:
                grid[top + r, left + c] = arr[r, c]


def generate_input() -> np.ndarray:
    sprites = [
        np.array([[1, 0, 3], [1, 3, 3], [0, 4, 0]]),
        np.array([[8, 8, 0, 6], [0, 8, 6, 6], [8, 0, 0, 6]]),
        np.array([[5, 0], [5, 9], [9, 9], [0, 9]]),
        np.array([[2, 2, 0], [0, 2, 7], [7, 7, 7], [0, 7, 0]]),
    ]
    grid = np.zeros((14, 15), dtype=int)
    chosen = random.sample(sprites, random.randint(3, 4))
    slots = [(0, 0), (0, 9), (8, 0), (8, 9)]
    random.shuffle(slots)
    for arr, (base_r, base_c) in zip(chosen, slots):
        top = base_r + random.randint(0, max(0, 5 - arr.shape[0]))
        left = base_c + random.randint(0, max(0, 5 - arr.shape[1]))
        _place(grid, top, left, arr)
    return grid
