import random
from collections import deque
from typing import Dict, Tuple, List

import numpy as np


def _components(grid: np.ndarray, color: int):
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            q = [(r, c)]
            visited[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if nx < 0 or nx >= H or ny < 0 or ny >= W:
                        continue
                    if visited[nx, ny] or int(grid[nx, ny]) != color:
                        continue
                    visited[nx, ny] = True
                    q.append((nx, ny))
            comps.append(pts)
    return comps


def _is_source_frame(g: np.ndarray, pts):
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    color = int(g[r0, c0])
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            expected = color if (r in (r0, r1) or c in (c0, c1)) else 0
            if int(g[r, c]) != expected:
                return None
    return r0, c0, r1, c1, color


def _is_target_frame(g: np.ndarray, pts):
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            is_border = (r in (r0, r1) or c in (c0, c1))
            if is_border:
                if (r, c) not in set(pts):
                    return None
            else:
                if int(g[r, c]) != 0:
                    return None
    return r0, c0, r1, c1


def _reconstruct_full(h: int, w: int, axis: str):
    m = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                m[r, c] = True

    if axis == "v":
        mid = (w - 1) // 2
        for r in range(h):
            for c in range(w):
                if m[r, c] and c < mid:
                    c2 = 2 * mid - c
                    if 0 <= c2 < w:
                        m[r, c2] = True
    elif axis == "h":
        mid = (h - 1) // 2
        for r in range(h):
            for c in range(w):
                if m[r, c] and r < mid:
                    r2 = 2 * mid - r
                    if 0 <= r2 < h:
                        m[r2, c] = True
    return m


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = np.zeros((H, W), dtype=int)

    source_frames: Dict[Tuple[int, int], int] = {}
    for color in [1, 2, 3, 4, 7, 9]:
        if color == 0:
            continue
        for comp in _components(g, color):
            det = _is_source_frame(g, comp)
            if det is not None:
                r0, c0, r1, c1, col = det
                source_frames[(r1 - r0 + 1, c1 - c0 + 1)] = col

    used_sizes = set()
    for comp in _components(g, 5):
        det = _is_target_frame(g, comp)
        if det is None:
            continue
        r0, c0, r1, c1 = det
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if (h, w) not in source_frames:
            continue
        if (h, w) in used_sizes:
            continue
        used_sizes.add((h, w))

        color = source_frames[(h, w)]
        frame = g[r0 : r1 + 1, c0 : c1 + 1]

        # infer axis by missing side
        left = any((frame[:, 0] == 5))
        right = any((frame[:, -1] == 5))
        top = any((frame[0, :] == 5))
        bottom = any((frame[-1, :] == 5))
        axis = "v"
        if not left or not right:
            axis = "v"
        if not top or not bottom:
            axis = "h"

        full_mask = _reconstruct_full(h, w, axis)
        for rr in range(h):
            for cc in range(w):
                if full_mask[rr, cc]:
                    out[r0 + rr, c0 + cc] = color

    return out


def generate_input() -> np.ndarray:
    H = 18
    W = 18
    g = np.zeros((H, W), dtype=int)

    n = random.randint(2, 3)
    sizes = [(5, 5), (5, 7), (7, 5)]
    random.shuffle(sizes)
    sizes = sizes[:n]

    colors = random.sample([1, 2, 3, 4, 7], n)

    # source complete frames
    for i, (h, w) in enumerate(sizes):
        r0 = 1 + i * 6
        c0 = 1
        if r0 + h >= H:
            r0 = 1
            c0 = 10
        col = colors[i]
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                if r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1):
                    g[r, c] = col

    # target incomplete frames with same sizes, odd dims
    for i, (h, w) in enumerate(sizes):
        r0 = H // 2 + i * (h + 1) - 1
        c0 = 1
        if r0 + h >= H:
            r0 = H // 2
            c0 = 10
        axis = random.choice(["v", "h"])
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                if r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1):
                    if axis == "v":
                        if c >= (w - 1) // 2:
                            g[r, c] = 5
                    else:
                        if r >= (h - 1) // 2:
                            g[r, c] = 5

    return g
