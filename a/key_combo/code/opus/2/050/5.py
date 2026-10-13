"""Repaint orange noise points based on nearby rectangle neighborhood counts.

Input structure:
  A black background with 2-3 large solid rectangles in colors 1/3/9, and 6-10
  orange(7) noise points located at varying Chebyshev-neighborhood distances to the
  rectangles.

Rule:
  For each noise point, count in its 8-neighborhood how many cells belong to any
  rectangle and the rectangle color C.
    N >= 3: recolor the noise and neighboring background cells to C.
    N == 1 or 2: recolor only the noise to C.
    N == 0: erase the noise to background.
  All counts are based on the original input.

Key insight:
  The same 8-neighborhood relation both decides the class and defines the halo
  fill area for strong connection.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import numpy as np


def _components(grid: np.ndarray, mask: np.ndarray) -> List[Set[Tuple[int, int]]]:
    H, W = grid.shape
    vis = np.zeros((H, W), dtype=bool)
    comps: List[Set[Tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp: Set[Tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not vis[nr, nc] and mask[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if g.ndim != 2:
        return g.copy()
    H, W = g.shape
    bg = 0

    rect_mask = g != bg
    # remove orange points (will be processed separately)
    rect_mask = rect_mask & (g != 7)
    rect_id = np.full((H, W), -1, dtype=np.int32)
    rects = _components(g, rect_mask)
    for idx, comp in enumerate(rects):
        for r, c in comp:
            rect_id[r, c] = idx
    out = g.copy()
    noise_cells = [(r, c) for r in range(H) for c in range(W) if g[r, c] == 7]

    for r, c in noise_cells:
        ids: Set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    rid = rect_id[nr, nc]
                    if rid >= 0:
                        ids.add(int(rid))
        n = len(ids)
        if n == 0:
            out[r, c] = 0
            continue
        # choose the closest/first rectangle color encountered by scan
        rid = next(iter(ids))
        if rects[rid]:
            rr0, cc0 = next(iter(rects[rid]))
            color = int(g[rr0, cc0])
        else:
            color = 0
        # fallback: pick color by first neighboring pixel
        if color == 0:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if g[nr, nc] not in (0, 7):
                            color = int(g[nr, nc])
                            break
                if color != 0:
                    break
        if n >= 3:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 0:
                        out[nr, nc] = color
            out[r, c] = color
        elif n in (1, 2):
            out[r, c] = color
        else:
            out[r, c] = 0
    return out


def _extract_rect_color(rect: Set[Tuple[int, int]], grid: np.ndarray) -> int:
    colors = defaultdict(int)
    for r, c in rect:
        colors[int(grid[r, c])] += 1
    if not colors:
        return 0
    return max(colors.items(), key=lambda kv: kv[1])[0]


def _neighbors(mask: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    return {}


def generate_input() -> np.ndarray:
    H = 15
    W = 15
    g = np.zeros((H, W), dtype=np.int64)
    colors = [1, 3, 9]
    rect_colors = random.sample(colors, random.randint(2, 3))

    rects: List[Tuple[int, int, int, int, int]] = []
    # place 2-3 rectangles with spacing
    for color in rect_colors:
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        for _ in range(400):
            r = random.randint(1, H - h - 1)
            c = random.randint(1, W - w - 1)
            area = g[r - 1 : r + h + 1, c - 1 : c + w + 1]
            if np.any(area != 0):
                continue
            g[r : r + h, c : c + w] = color
            rects.append((r, c, h, w, color))
            break

    # helper to gather neighborhood rectangle id counts
    def nearby_count(r: int, c: int, max_n: int) -> int:
        nset = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] != 0 and g[nr, nc] != 7:
                    # tie rectangles by absolute position
                    for j, (rr, cc, hh, ww, color) in enumerate(rects):
                        if rr <= nr < rr + hh and cc <= nc < cc + ww:
                            nset.add(j)
        return len(nset)

    def add_noise(r: int, c: int, _min: int, _max: int) -> bool:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if g[r, c] != 0:
            return False
        # keep noise points from touching each other
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and g[nr, nc] == 7:
                    return False
        g[r, c] = 7
        return True

    # construct three noise categories
    noise_targets: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    noise_cells = []
    for nr in range(1, H - 1):
        for nc in range(1, W - 1):
            if g[nr, nc] != 0:
                continue
            cnt = nearby_count(nr, nc, 3)
            if 0 <= cnt <= 4:
                noise_targets[cnt].append((nr, nc))

    # 1) isolated
    random.shuffle(noise_targets[0])
    for _ in range(3):
        if noise_targets[0]:
            pos = noise_targets[0].pop()
            if add_noise(*pos, 0, 0):
                noise_cells.append(pos)
                break

    # 2) weak
    weak_ok = [p for p in noise_targets[1] + noise_targets[2] if p not in noise_cells]
    random.shuffle(weak_ok)
    if weak_ok:
        for _ in range(2):
            if not weak_ok:
                break
            pos = weak_ok.pop()
            if add_noise(*pos, 1, 2):
                noise_cells.append(pos)

    # 3) strong
    strong_ok: List[Tuple[int, int]] = []
    for k in sorted(noise_targets.keys()):
        if k >= 3:
            strong_ok.extend(noise_targets[k])
    strong_ok = [p for p in strong_ok if p not in noise_cells]
    random.shuffle(strong_ok)
    if strong_ok:
        for _ in range(3):
            if not strong_ok:
                break
            pos = strong_ok.pop()
            if add_noise(*pos, 3, 99):
                noise_cells.append(pos)

    # fill additional noise to satisfy requested count
    while len(noise_cells) < random.randint(6, 10):
        r = random.randint(1, H - 2)
        c = random.randint(1, W - 2)
        if add_noise(r, c, 0, 0):
            noise_cells.append((r, c))
    return g
