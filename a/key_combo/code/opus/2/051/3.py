from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dr, dc in _DIRS:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _is_hollow_rect(comp: List[Tuple[int, int]], h: int, w: int) -> bool:
    if not comp:
        return False
    r0, c0, r1, c1 = _bbox(comp)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return False
    perimeter = set()
    for rr in range(r0, r1 + 1):
        perimeter.add((rr, c0))
        perimeter.add((rr, c1))
    for cc in range(c0 + 1, c1):
        perimeter.add((r0, cc))
        perimeter.add((r1, cc))
    comp_set = set(comp)
    if perimeter != comp_set:
        return False
    area = hh * ww - (hh - 2) * (ww - 2)
    return len(comp) == area


def _sorted_color_cells(comp: List[Tuple[int, int]], g: np.ndarray, color: int) -> List[Tuple[float, int]]:
    cells = [(r, c) for r, c in comp if g[r, c] == color]
    if len(cells) == 0:
        return []
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    # Parse legend: contiguous 2x2 gray/noise blocks in first two rows.
    legend: List[np.ndarray] = []
    c = 0
    while c < w - 1:
        block = out[:2, c : c + 2]
        vals = set(int(v) for v in block.flatten())
        if 5 in vals and vals.issubset({0, 5}):
            legend.append((block == 5).astype(int))
            c += 3
        else:
            c += 1
    if not legend:
        return out

    # Detect candidate yellow frames.
    rects = []
    for comp in _components(g == 4):
        if not _is_hollow_rect(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        area = (r1 - r0 - 1) * (c1 - c0 - 1)
        rects.append((area, comp))
    if not rects:
        return out
    rects.sort(key=lambda x: x[0])

    for idx, (_, comp) in enumerate(rects[: len(legend)]):
        r0, c0, r1, c1 = _bbox(comp)
        pattern = legend[idx % len(legend)]
        color_cells = _sorted_color_cells(comp, g, 4)
        control_color = 2  # fallback
        if color_cells:
            # The control color isn't needed for geometry; keep existing control if present.
            pass
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                local_r = (rr - (r0 + 1)) % 2
                local_c = (cc - (c0 + 1)) % 2
                if pattern[local_r, local_c] == 1:
                    # Use first non-black interior color seen in this rectangle as template color.
                    seed = _find_seed_color(g[r0 + 1 : r1, c0 + 1 : c1])
                    out[rr, cc] = seed if seed != 0 else 1
                else:
                    out[rr, cc] = 0

    return out


def _find_seed_color(block: np.ndarray) -> int:
    for r in range(block.shape[0]):
        for c in range(block.shape[1]):
            v = int(block[r, c])
            if v != 0 and v != 4:
                return v
    return 1


def generate_input() -> np.ndarray:
    h = 20
    w = 20
    g = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3, 6]
    random.shuffle(colors)

    # Legend row blocks.
    starts = [1, 4, 7, 10]
    for i in range(3):
        c0 = starts[i]
        r0 = 0
        mask = np.zeros((2, 2), dtype=int)
        bitmask = random.randint(1, 15)
        idx = 0
        for rr in range(2):
            for cc in range(2):
                if (bitmask >> idx) & 1:
                    mask[rr, cc] = 1
                idx += 1
        g[r0 : r0 + 2, c0 : c0 + 2] = np.where(mask == 1, 5, 0)

    # Paint N yellow hollow rectangles in lower area.
    occupied = np.zeros_like(g, dtype=bool)
    used_colors = []
    rect_specs = [(2, 2), (3, 4), (4, 2)]
    for i in range(3):
        ih, iw = rect_specs[i]
        th = ih + 2
        tw = iw + 2
        placed = False
        for _ in range(200):
            r0 = random.randint(4, h - th - 2)
            c0 = random.randint(1, w - tw - 2)
            rr0, cc0, rr1, cc1 = r0, c0, r0 + th - 1, c0 + tw - 1
            # No overlap with existing border
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            used_colors.append(colors[i % len(colors)])
            for rr in range(rr0, rr1 + 1):
                g[rr, cc0] = 4
                g[rr, cc1] = 4
            for cc in range(cc0 + 1, cc1):
                g[rr0, cc] = 4
                g[rr1, cc] = 4
            seed_r = random.randint(rr0 + 1, rr1 - 1)
            seed_c = random.randint(cc0 + 1, cc1 - 1)
            g[seed_r, seed_c] = colors[i % len(colors)]
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            break
    return g
