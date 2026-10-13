from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            pts: List[Tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_from_coords(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        out[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = np.zeros((0, 0), dtype=int)

    colors = sorted(set(map(int, g.flatten())) - {bg})
    packed_blocks = []

    for col in colors:
        mask = g == col
        comps = _components(mask)
        block_comp = None
        motif_comp = None

        for comp in comps:
            if comp.size == 0:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            kh, kw = r1 - r0 + 1, c1 - c0 + 1
            area = int(comp.shape[0])
            if area == 25 and kh == 5 and kw == 5:
                block_comp = comp
            elif 2 <= area <= 4 and kh <= 3 and kw <= 3:
                motif_comp = comp
        if block_comp is None or motif_comp is None:
            continue

        br0, bc0, br1, bc1 = _bbox(block_comp)
        tr0, tc0, tr1, tc1 = _bbox(motif_comp)
        motif = np.zeros((tr1 - tr0 + 1, tc1 - tc0 + 1), dtype=int)
        for r, c in motif_comp:
            motif[int(r - tr0), int(c - tc0)] = col
        motif_points = _crop_from_coords(g, motif_comp) > 0

        # Align motif box center (floor for even side) to block center
        block_center = (br0 + 2, bc0 + 2)
        motif_center = ((tr0 + tr1) // 2, (tc0 + tc1) // 2)
        dr = block_center[0] - motif_center[0]
        dc = block_center[1] - motif_center[1]

        carved = np.full((5, 5), col, dtype=int)
        for r, c in motif_comp:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr - br0 < 5 and 0 <= cc - bc0 < 5:
                carved[rr - br0, cc - bc0] = bg

        holes = int(np.sum(carved == bg))
        holes = max(2, holes)  # fallback safety
        enlarged = np.repeat(np.repeat(carved, holes, axis=0), holes, axis=1)
        packed_blocks.append((bc0, enlarged))

    packed_blocks.sort(key=lambda x: x[0])
    if not packed_blocks:
        return g.copy()

    total_w = sum(b.shape[1] for _, b in packed_blocks) + max(0, len(packed_blocks) - 1)
    max_h = max(b.shape[0] for _, b in packed_blocks)
    out = np.full((max_h, total_w), bg, dtype=int)
    c = 0
    for _, block in packed_blocks:
        out[: block.shape[0], c : c + block.shape[1]] = block
        c += block.shape[1] + 1
    return out


_RNG = np.random.default_rng()


_TEMPLATES = {
    2: [[(0, 0), (0, 1)]],
    3: [
        [(0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
    ],
    4: [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, 1), (2, 1), (1, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
    ],
}


def _build_template(size: int) -> np.ndarray:
    choices = _TEMPLATES[size]
    coords = _RNG.choice(len(choices))
    coords = choices[int(coords)]
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    t = np.zeros((h, w), dtype=int)
    for r, c in coords:
        t[r, c] = 1
    return t


def _can_place(mask: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    h0, w0 = mask.shape
    r1, c1 = r0 + h, c0 + w
    if r1 > h0 or c1 > w0 or r0 < 0 or c0 < 0:
        return False
    dil = np.zeros((h0, w0), dtype=int)
    rr0, rr1 = max(0, r0 - margin), min(h0, r1 + margin)
    cc0, cc1 = max(0, c0 - margin), min(w0, c1 + margin)
    if np.any(mask[rr0:rr1, cc0:cc1]):
        return False
    return True


def generate_input() -> np.ndarray:
    h = int(_RNG.integers(19, 23))
    w = 25
    g = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3]
    _RNG.shuffle(colors)

    block_top = int(_RNG.integers(1, 5))
    block_cols = [1, 9, 17]
    # random jitter while keeping spacing
    block_cols = [max(0, min(w - 6, int(c + _RNG.integers(-1, 2)))) for c in block_cols]
    block_cols = sorted(set(block_cols))
    if len(block_cols) != 3:
        block_cols = [1, 9, 17]

    block_mask = np.zeros_like(g, dtype=bool)
    for col, color in zip(block_cols, colors):
        for dr in range(5):
            for dc in range(5):
                r = block_top + dr
                c = col + dc
                g[r, c] = color
                block_mask[r, c] = True

    motif_sizes = _RNG.permutation([2, 3, 4])
    motif_mask = np.zeros_like(g, dtype=bool)
    for color, size in zip(colors, motif_sizes):
        t = _build_template(int(size))
        th, tw = t.shape
        placed = False
        tries = 300
        for _ in range(tries):
            r0 = int(_RNG.integers(h // 2, h - th))
            c0 = int(_RNG.integers(1, w - tw))
            if not _can_place(g == 0, r0, c0, th, tw, margin=2):
                continue
            if not _can_place(block_mask | motif_mask, r0, c0, th, tw, margin=2):
                continue
            if r0 + th >= h or c0 + tw >= w:
                continue
            for rr in range(th):
                for cc in range(tw):
                    if t[rr, cc]:
                        g[r0 + rr, c0 + cc] = color
                        motif_mask[r0 + rr, c0 + cc] = True
            placed = True
            break
        if not placed:
            # deterministic fallback
            r0 = min(h // 2, h - th - 1)
            c0 = 1
            for rr in range(th):
                for cc in range(tw):
                    if t[rr, cc]:
                        g[r0 + rr, c0 + cc] = color
                        motif_mask[r0 + rr, c0 + cc] = True

    return g
