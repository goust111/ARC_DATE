from __future__ import annotations

import random

import numpy as np
from arc_utils import find_components


BG = 0
GREY = 5


def _hole_shapes_in_object(g: np.ndarray, bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox
    box = g[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    seen = np.zeros(box.shape, dtype=bool)
    for r in range(box.shape[0]):
        for c in range(box.shape[1]):
            if box[r, c] != BG or seen[r, c]:
                continue
            stack = [(r, c)]
            comp = []
            seen[r, c] = True
            touch = False
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                if rr in {0, box.shape[0] - 1} or cc in {0, box.shape[1] - 1}:
                    touch = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < box.shape[0] and 0 <= nc < box.shape[1] and not seen[nr, nc] and box[nr, nc] == BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touch:
                rs = [p[0] for p in comp]
                cs = [p[1] for p in comp]
                mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
                for rr, cc in comp:
                    mask[rr - min(rs), cc - min(cs)] = True
                holes.append(mask)
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each holed object by matching its dominant hole shape to the left-side legend.

    Input structure:
    - Black background with a left legend made of grey 3x3 hole samples and adjacent color pixels.
    - Larger monochrome objects on the right contain one or more enclosed black holes.

    Rule:
    - Read every legend entry as hole-shape sample -> output color mapping.
    - For each right-side object, extract its enclosed hole shapes and choose the largest hole if there are several.
    - Match that normalized hole shape to the legend sample without rotation or reflection.
    - Recolor the object's foreground pixels to the mapped legend color, while keeping holes black.

    Key insight:
    - The left legend does not describe outer silhouettes; it keys objects only by the normalized shape of their interior cavity.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    r = 0
    while r + 2 < g.shape[0]:
        sample = g[r : r + 3, 0:3]
        if np.any(sample == GREY) and g[r + 1, 3] not in {BG, GREY}:
            hole = sample == BG
            rows, cols = np.where(hole)
            if len(rows):
                mask = np.zeros((rows.max() - rows.min() + 1, cols.max() - cols.min() + 1), dtype=bool)
                for rr, cc in zip(rows, cols, strict=False):
                    mask[rr - rows.min(), cc - cols.min()] = True
                legend[(tuple(mask.shape), tuple(mask.flatten().astype(int)))] = int(g[r + 1, 3])
            r += 4
        else:
            r += 1
    for comp in find_components(g[:, 4:], bg=BG, connectivity=4, monochromatic=True):
        bbox = comp["bbox"]
        bbox = (bbox[0], bbox[1] + 4, bbox[2], bbox[3] + 4)
        holes = _hole_shapes_in_object(g, bbox)
        if not holes:
            continue
        hole = max(holes, key=lambda m: int(m.sum()))
        key = (tuple(hole.shape), tuple(hole.flatten().astype(int)))
        if key not in legend:
            continue
        color = legend[key]
        for rr, cc in comp["coords"]:
            out[int(rr), int(cc) + 4] = color
    return out


def _place_hole_object(g: np.ndarray, r0: int, c0: int, outer_h: int, outer_w: int, hole: np.ndarray, color: int) -> None:
    g[r0 : r0 + outer_h, c0 : c0 + outer_w] = color
    hr, hc = hole.shape
    sr = r0 + (outer_h - hr) // 2
    sc = c0 + (outer_w - hc) // 2
    g[sr : sr + hr, sc : sc + hc][hole] = BG


def generate_input() -> np.ndarray:
    hole_bank = [
        np.array([[1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
    ]
    h = random.randint(18, 24)
    w = random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    for i, (color, hole) in enumerate(zip(colors, random.sample(hole_bank, len(colors)), strict=False)):
        r0 = 1 + 4 * i
        g[r0 : r0 + 3, 0:3] = GREY
        sr = r0 + (3 - hole.shape[0]) // 2
        sc = (3 - hole.shape[1]) // 2
        g[sr : sr + hole.shape[0], sc : sc + hole.shape[1]][hole] = BG
        g[r0 + 1, 3] = color
        _place_hole_object(g, random.randint(1, h - 7), random.randint(6, w - 7), random.randint(5, 7), random.randint(5, 7), hole, 7)
    return g
