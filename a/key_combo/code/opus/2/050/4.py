"""Propagate a corner ray template to other corners and stop on collisions.

Input structure:
  A gray hollow square frame with four corner holes (black). One corner has an
  existing outward diagonal ray; this is the source. Other corners have a single
  marker pixel at the same ray-length offset along their outward diagonal.

Rule:
  Extract the ordered diagonal offsets of the source ray and apply this template to
  each marked target corner in clockwise order, drawing with the target marker
  color along its outward diagonal. A ray stops when the next pixel is already
  occupied (frame, previous ray, or other foreground). Keep the source ray and all
  original pixels except the added ray segments.

Key insight:
  All corners share the same diagonal template, and clockwise write order changes how
  later rays collide with earlier rays and become shorter.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np


def _bbox_of_color(grid: np.ndarray, color: int) -> Tuple[int, int, int, int] | None:
    rs, cs = np.where(grid == color)
    if len(rs) == 0:
        return None
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if g.ndim != 2:
        return g.copy()
    H, W = g.shape
    bg = 0

    # assume frame color is dominant non-background connected block with a square box
    colors, counts = np.unique(g, return_counts=True)
    frame_color = int(colors[np.argmax(counts)] if len(colors) else 0)
    if frame_color == bg:
        return g.copy()
    bbox = _bbox_of_color(g, frame_color)
    if bbox is None:
        return g.copy()
    r0, c0, r1, c1 = bbox
    if not (r1 > r0 and c1 > c0):
        return g.copy()

    corners = [
        ("tl", (r0, c0), (-1, -1)),
        ("tr", (r0, c1), (-1, 1)),
        ("br", (r1, c1), (1, 1)),
        ("bl", (r1, c0), (1, -1)),
    ]

    # extract source ray
    source_idx = None
    source_offsets: List[Tuple[int, int]] = []
    source_color = 0
    k = 0
    for i, (_name, corner, (dr, dc)) in enumerate(corners):
        offsets: List[Tuple[int, int]] = []
        cr, cc = corner
        rr, cc2 = cr + dr, cc + dc
        if not (0 <= rr < H and 0 <= cc2 < W):
            continue
        col = int(g[rr, cc2])
        if col == bg:
            continue
        t = 1
        while 0 <= rr < H and 0 <= cc2 < W and int(g[rr, cc2]) == col:
            offsets.append((dr * t, dc * t))
            t += 1
            rr += dr
            cc2 += dc
        if len(offsets) >= 3 and len(offsets) > len(source_offsets):
            source_idx = i
            source_offsets = offsets
            source_color = col
            k = len(offsets)

    if source_idx is None or k == 0:
        return g.copy()

    out = g.copy()
    for step in range(1, k + 1):
        dr, dc = corners[source_idx][2]
        rr = corners[source_idx][1][0] + dr * step
        cc = corners[source_idx][1][1] + dc * step
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = source_color

    marker_colors: Dict[int, int] = {}
    for i, (_name, corner, (dr, dc)) in enumerate(corners):
        if i == source_idx:
            continue
        tr, tc = corner
        mr, mc = tr + dr * k, tc + dc * k
        if 0 <= mr < H and 0 <= mc < W:
            mv = int(g[mr, mc])
            if mv != bg and mv != frame_color:
                marker_colors[i] = mv

    # process target corners clockwise from next corner of source
    for t in range(1, 4):
        idx = (source_idx + t) % 4
        if idx not in marker_colors:
            continue
        color = marker_colors[idx]
        corner, (dr, dc) = corners[idx][1], corners[idx][2]
        cr, cc = corner
        for step in range(1, k + 1):
            rr = cr + dr * step
            cc2 = cc + dc * step
            if not (0 <= rr < H and 0 <= cc2 < W):
                break
            if out[rr, cc2] != bg and (rr, cc2) != (cr + dr * k, cc + dc * k):
                break
            if (rr, cc2) == (cr + dr * k, cc + dc * k) and out[rr, cc2] != bg:
                # marker is preserved
                break
            out[rr, cc2] = color
    return out


def generate_input() -> np.ndarray:
    H = 20
    W = 20
    g = np.zeros((H, W), dtype=np.int64)
    frame_color = 5

    for _ in range(200):
        k = random.randint(3, 4)
        frame_size = random.randint(7, 9)
        margin = k + 1
        top_min = margin
        left_min = margin
        top_max = H - frame_size - margin
        left_max = W - frame_size - margin
        if top_max < top_min or left_max < left_min:
            continue
        top = random.randint(top_min, top_max)
        left = random.randint(left_min, left_max)
        break
    else:
        top = 5
        left = 5
        frame_size = 7
        k = 3

    # ensure chosen frame still within bounds under chosen parameters
    if top + frame_size > H or left + frame_size > W:
        top = max(0, H - frame_size - 1)
        left = max(0, W - frame_size - 1)

    bottom = top + frame_size - 1
    right = left + frame_size - 1

    # draw hollow frame
    g[top, left + 1 : right] = frame_color
    g[bottom, left + 1 : right] = frame_color
    g[top + 1 : bottom, left] = frame_color
    g[top + 1 : bottom, right] = frame_color

    corners = [
        (top, left, -1, -1),
        (top, right, -1, 1),
        (bottom, right, 1, 1),
        (bottom, left, 1, -1),
    ]
    src_idx = random.randint(0, 3)
    g[corners[src_idx][0], corners[src_idx][1]] = 0

    # draw source ray
    source_color = random.choice([1, 2, 3, 7, 8, 9])
    sr, sc, sdr, sdc = corners[src_idx]
    rr, cc = sr + sdr, sc + sdc
    for _ in range(k):
        if 0 <= rr < H and 0 <= cc < W:
            g[rr, cc] = source_color
        rr += sdr
        cc += sdc

    # place target markers on other corners
    target_choices = [i for i in range(4) if i != src_idx]
    n_markers = random.randint(2, 3)
    for idx in random.sample(target_choices, n_markers):
        r, c, dr, dc = corners[idx]
        mr, mc = r + dr * k, c + dc * k
        if 0 <= mr < H and 0 <= mc < W:
            g[mr, mc] = random.choice([1, 2, 3, 4, 7, 8, 9])
    return g
