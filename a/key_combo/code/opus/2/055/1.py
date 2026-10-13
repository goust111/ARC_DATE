import random
from collections import deque
from typing import List, Tuple, Dict

import numpy as np


def _find_selector(grid: np.ndarray):
    H, W = grid.shape
    centers = [(1, 1), (1, 5), (5, 1), (5, 5)]
    for r0 in range(H - 6):
        for c0 in range(W - 6):
            b = grid[r0 : r0 + 7, c0 : c0 + 7]
            if b.shape != (7, 7):
                continue
            if not np.all(b[3, :] == 0):
                continue
            if not np.all(b[:, 3] == 0):
                continue
            colors: List[int] = []
            ok = True
            for dr, dc in centers:
                v = int(b[dr, dc])
                if v in (0, 5):
                    ok = False
                    break
                colors.append(v)
            if not ok:
                continue
            if len(set(colors)) != 4:
                continue
            return r0, c0, colors
    return None


def _components_of_color(grid: np.ndarray, color: int, skip: np.ndarray):
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []

    for sr in range(H):
        for sc in range(W):
            if visited[sr, sc] or skip[sr, sc] or int(grid[sr, sc]) != color:
                continue
            q = [(sr, sc)]
            visited[sr, sc] = True
            pts: List[Tuple[int, int]] = []
            while q:
                r, c = q.pop()
                pts.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        continue
                    if visited[nr, nc] or skip[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append(pts)
    return comps


def _flood_from_selector_adjacent(grid: np.ndarray, sr: int, sc: int):
    H, W = grid.shape
    q = deque()
    visited = np.zeros((H, W), dtype=bool)

    for c in range(sc, sc + 7):
        for rr in (sr - 1, sr + 7):
            if 0 <= rr < H and int(grid[rr, c]) == 0:
                q.append((rr, c))
                visited[rr, c] = True
    for r in range(sr, sr + 7):
        for cc in (sc - 1, sc + 7):
            if 0 <= cc < W and int(grid[r, cc]) == 0:
                q.append((r, cc))
                visited[r, cc] = True

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if visited[nr, nc]:
                continue
            if int(grid[nr, nc]) != 0:
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
    return visited


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    sel = _find_selector(g)
    if sel is None:
        return np.zeros((7, 7), dtype=int)

    sr, sc, marker_colors = sel

    marker_to_quad: Dict[int, Tuple[int, int]] = {
        int(g[sr + 1, sc + 1]): (0, 0),
        int(g[sr + 1, sc + 5]): (0, 4),
        int(g[sr + 5, sc + 1]): (4, 0),
        int(g[sr + 5, sc + 5]): (4, 4),
    }

    skip = np.zeros((H, W), dtype=bool)
    skip[sr : sr + 7, sc : sc + 7] = True

    comps_by_color: Dict[int, List[Tuple[int, int]]] = {}
    for color in marker_colors:
        comps = _components_of_color(g, color, skip)
        if not comps:
            continue
        # 1~9 cells, bbox <=3x3
        cand = []
        for comp in comps:
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
            if h <= 3 and w <= 3:
                cand.append(comp)
        if cand:
            comp = max(cand, key=len)
            comps_by_color[color] = comp

    reachable = _flood_from_selector_adjacent(g, sr, sc)

    out = np.zeros((7, 7), dtype=int)
    for color, comp in comps_by_color.items():
        if color not in marker_to_quad:
            continue
        is_reachable = False
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and reachable[nr, nc]:
                    is_reachable = True
                    break
            if is_reachable:
                break

        if not is_reachable:
            continue

        qr, qc = marker_to_quad[color]
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        base_r, base_c = min(rs), min(cs)
        for r, c in comp:
            rr = qr + (r - base_r)
            cc = qc + (c - base_c)
            if 0 <= rr < 3 and 0 <= cc < 3:
                out[rr, cc] = color

    return out


def _random_sprite_mask() -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    r, c = 1, 1
    mask[r, c] = True
    frontier = {(0, 1), (2, 1), (1, 0), (1, 2)}
    target = random.randint(3, 8)
    while mask.sum() < target and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        if mask[r, c]:
            continue
        mask[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3 and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def generate_input() -> np.ndarray:
    H = W = 20
    g = np.zeros((H, W), dtype=int)

    sr = random.randint(2, 5)
    sc = random.randint(2, 5)
    # Selector 7x7
    g[sr : sr + 7, sc : sc + 7] = 5
    g[sr + 3, sc : sc + 7] = 0
    g[sr : sr + 7, sc + 3] = 0

    marker_colors = random.sample([1, 2, 3, 4, 7], 4)
    marker_cells = [(sr + 1, sc + 1), (sr + 1, sc + 5), (sr + 5, sc + 1), (sr + 5, sc + 5)]
    for (r, c), color in zip(marker_cells, marker_colors):
        g[r, c] = color

    wall_color = 9 if 1 not in marker_colors else 7
    wall_col = sc + 7 + random.randint(2, 3)
    if wall_col < W:
        g[:, wall_col] = wall_color

    sprite_positions = [
        (sr + 6, sc - 3),
        (sr + 10, sc - 3),
        (sr + 6, sc + 10),
        (sr + 10, sc + 10),
    ]
    random.shuffle(sprite_positions)

    for (r0, c0), color in zip(sprite_positions, marker_colors):
        if not (0 <= r0 and r0 + 2 < H and 0 <= c0 and c0 + 2 < W):
            continue
        if wall_col <= c0 <= wall_col:
            continue
        mask = _random_sprite_mask()
        for dr in range(3):
            for dc in range(3):
                if mask[dr, dc]:
                    g[r0 + dr, c0 + dc] = color

    return g
