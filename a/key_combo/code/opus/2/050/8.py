"""Fill the center gap with two halved template-tiled textures.

Input structure:
  A black background with two same-size colored rectangles in the same row and a black
  gap between them. A small gray/black template tile exists elsewhere.

Rule:
  Split the gap by columns into left/right halves. For odd gap sizes keep the middle
  gap column background. Tile the template from each half's top-left using the
  corresponding rectangle color for gray pixels, and preserve black for template
  background cells. Keep rectangles unchanged and erase the original template region.

Key insight:
  The half-split determines which side owns each gap column, while the template tile
  controls per-cell micro-patterns independently.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np


def _connected_components(mask: np.ndarray) -> List[Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]]:
    H, W = mask.shape
    vis = np.zeros((H, W), dtype=bool)
    comps: List[Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or vis[r, c]:
                continue
            stack = [(r, c)]
            vis[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            comps.append((comp, (min(rs), min(cs), max(rs), max(cs))))
    return comps


def _is_filled_rect(cell: List[Tuple[int, int]], bbox: Tuple[int, int, int, int], mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox
    return len(cell) == (r1 - r0 + 1) * (c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if g.ndim != 2:
        return g.copy()
    H, W = g.shape
    # template is all gray pixels (5) in this generator family
    tpl_positions = np.argwhere(g == 5)
    if len(tpl_positions) == 0:
        return g.copy()

    r0t, c0t = int(tpl_positions[:, 0].min()), int(tpl_positions[:, 1].min())
    r1t, c1t = int(tpl_positions[:, 0].max()), int(tpl_positions[:, 1].max())
    template_bbox = g[r0t : r1t + 1, c0t : c1t + 1]
    tpl_h, tpl_w = template_bbox.shape
    template_mask = template_bbox == 5

    # detect rectangles with same size and same row span
    rect_mask = (g != 0) & (g != 5)
    rect_candidates = []
    for cells, bbox in _connected_components(rect_mask):
        if not cells:
            continue
        r0, c0, r1, c1 = bbox
        if not _is_filled_rect(cells, bbox, rect_mask):
            continue
        rect_candidates.append((r0, c0, r1, c1))
    if len(rect_candidates) < 2:
        return g.copy()

    # select pair in same row span and same width/height
    chosen = None
    for i, a in enumerate(rect_candidates):
        r0a, c0a, r1a, c1a = a
        ha = r1a - r0a + 1
        wa = c1a - c0a + 1
        for b in rect_candidates[i + 1 :]:
            r0b, c0b, r1b, c1b = b
            hb = r1b - r0b + 1
            wb = c1b - c0b + 1
            if ha == hb and wa == wb and r0a == r0b and r1a == r1b and c1a < c0b:
                chosen = (a, b)
                break
        if chosen:
            break
    if chosen is None:
        return g.copy()

    (r0a, c0a, r1a, c1a), (r0b, c0b, r1b, c1b) = chosen
    if c1a < c0b:
        left = (r0a, c0a, r1a, c1a)
        right = (r0b, c0b, r1b, c1b)
    else:
        left = (r0b, c0b, r1b, c1b)
        right = (r0a, c0a, r1a, c1a)
    r0, r1 = left[0], left[2]
    lg_color = int(g[left[0], left[1]])
    rd_color = int(g[right[0], right[1]])

    gap_c0 = left[3] + 1
    gap_c1 = right[1] - 1
    if gap_c0 > gap_c1:
        return g.copy()

    cols = gap_c1 - gap_c0 + 1
    half = cols // 2
    left_cols = list(range(gap_c0, gap_c0 + half))
    right_cols = list(range(gap_c0 + half + (cols % 2), gap_c1 + 1))

    # fill halves with tiled template
    for rr in range(r0, r1 + 1):
        for cc in left_cols:
            tr = (rr - r0) % tpl_h
            tc = (cc - left_cols[0]) % tpl_w if left_cols else 0
            if template_mask[tr, tc]:
                g[rr, cc] = lg_color
            else:
                g[rr, cc] = 0
        for cc in right_cols:
            tr = (rr - r0) % tpl_h
            tc = (cc - right_cols[0]) % tpl_w if right_cols else 0
            if template_mask[tr, tc]:
                g[rr, cc] = rd_color
            else:
                g[rr, cc] = 0

    # erase template region
    g[r0t : r1t + 1, c0t : c1t + 1] = 0
    return g


def _make_template() -> np.ndarray:
    h = random.choice([2, 2, 3])
    w = random.choice([2, 3])
    t = np.zeros((h, w), dtype=np.int64)
    # at least one foreground and one background cell
    fg = int(random.randint(1, (h * w) // 2))
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(fg):
        r, c = random.choice(cells)
        t[r, c] = 5
    return t


def generate_input() -> np.ndarray:
    H = 12
    W = 24
    g = np.zeros((H, W), dtype=np.int64)

    rect_h = random.randint(4, 5)
    rect_w = random.randint(3, 4)
    top = random.randint(2, H - rect_h - 2)
    row_end = top + rect_h - 1
    gap = random.randint(6, 10)
    left = random.randint(1, 3)
    right = left + rect_w + gap
    if right + rect_w >= W:
        right = W - rect_w - 1
        left = right - rect_w - gap
        if left < 1:
            left = 1
            right = left + rect_w + gap
    color_left = random.choice([1, 2, 3, 7, 8, 9])
    color_right = random.choice([1, 2, 3, 7, 8, 9, 4, 5, 6])
    while color_right == color_left:
        color_right = random.choice([1, 2, 3, 7, 8, 9, 4, 5, 6])
    g[top : row_end + 1, left : left + rect_w] = color_left
    g[top : row_end + 1, right : right + rect_w] = color_right

    # place template in a corner region
    template = _make_template()
    th, tw = template.shape
    if random.random() < 0.5:
        tr = H - th
        tc = 1
    else:
        tr = H - th
        tc = W - tw - 1
    if tr < 0:
        tr = 0
    if tc < 0:
        tc = 0
    g[tr : tr + th, tc : tc + tw] = template
    return g
