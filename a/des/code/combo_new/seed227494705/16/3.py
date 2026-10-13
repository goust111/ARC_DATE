from collections import deque
import math
import random

import numpy as np

GREY = 5


def _most_common(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _same_color_components(grid, colors=None, exclude=()):
    colors = None if colors is None else set(colors)
    excluded = set(exclude)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded or (colors is not None and color not in colors):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return comps


def _component_mask(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _mask_coords(mask):
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _rotate_cw(mask, k=1):
    return np.rot90(mask, -k)


def _rect_perimeter_cells(r0, c0, r1, c1):
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _draw_frame(grid, r0, c0, h, w, color):
    for r, c in _rect_perimeter_cells(r0, c0, r0 + h - 1, c0 + w - 1):
        grid[r, c] = color


def _stamp_mask(grid, mask, top, left, color):
    for r, c in _mask_coords(mask):
        grid[top + r, left + c] = color


def _asym_mask():
    return np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool)


def _find_grey_frame(grid):
    frames = []
    for comp in _same_color_components(grid, colors=[GREY]):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 >= 3 and c1 - c0 >= 3:
            frames.append((r0, c0, r1, c1))
    return max(frames, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _sequence_colors(grid, bg, frame_col):
    best_col = None
    best = []
    for c in range(frame_col):
        vals = []
        for r in range(grid.shape[0]):
            v = int(grid[r, c])
            if v not in (bg, GREY):
                vals.append((r, v))
        if len(vals) > len(best):
            best_col, best = c, vals
    return [v for _, v in sorted(best)], best_col


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble color-ordered equal sprites into the target frame with successive rotations.

    Input structure: A black background has a right-side grey hollow frame, a
    vertical color-order strip, and one same-shape colored sprite for every strip
    color scattered to the left of the frame.
    Rule: Read the strip top to bottom.  Clear the original loose sprites.  The
    first strip color's sprite is placed unrotated in the frame interior, the next
    is rotated 90 degrees clockwise, then 180, then 270, cycling by order.  The
    rotated sprites are packed in reading order inside the frame; the frame and
    strip remain.
    Key insight: Color gives object identity, but strip position gives both the
    placement order and the rotation amount.
    """
    bg = _most_common(grid)
    out = grid.copy()
    fr = _find_grey_frame(grid)
    r0, c0, r1, c1 = fr
    order, seq_col = _sequence_colors(grid, bg, c0)
    comps_by_color = {}
    for comp in _same_color_components(grid, exclude=[bg, GREY]):
        if seq_col is not None and all(c == seq_col for _, c in comp["cells"]) and len(comp["cells"]) == 1:
            continue
        if comp["color"] in order and len(comp["cells"]) > 1:
            comps_by_color[comp["color"]] = comp

    masks = {}
    for color, comp in comps_by_color.items():
        masks[color] = _component_mask(comp["cells"])
        for rr, cc in comp["cells"]:
            out[rr, cc] = bg

    if not masks:
        return out
    slot_h = max(m.shape[0] for m in masks.values())
    slot_w = max(m.shape[1] for m in masks.values())
    inner_h, inner_w = r1 - r0 - 1, c1 - c0 - 1
    slots_per_row = max(1, inner_w // slot_w)

    for i, color in enumerate(order):
        if color not in masks:
            continue
        mask = _rotate_cw(masks[color], i % 4)
        sr = i // slots_per_row
        sc = i % slots_per_row
        top = r0 + 1 + sr * slot_h
        left = c0 + 1 + sc * slot_w
        for rr, cc in _mask_coords(mask):
            if top + rr < r1 and left + cc < c1:
                out[top + rr, left + cc] = color
    return out


def generate_input() -> np.ndarray:
    n = random.randint(3, 6)
    cols = random.choice([2, 3])
    rows = math.ceil(n / cols)
    slot = 3
    h = max(18, rows * slot + 8)
    w = 27
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    for i, color in enumerate(colors):
        grid[1 + i, 0] = color

    fh, fw = rows * slot + 2, cols * slot + 2
    fr0, fc0 = 3, w - fw - 2
    _draw_frame(grid, fr0, fc0, fh, fw, GREY)

    base = _asym_mask()
    placed = np.zeros_like(grid, dtype=bool)
    placed[: n + 3, :2] = True
    placed[fr0 : fr0 + fh, fc0 : fc0 + fw] = True
    for color in colors:
        for _ in range(500):
            r = random.randint(3, h - 5)
            c = random.randint(3, fc0 - 5)
            if not placed[r - 1 : r + 4, c - 1 : c + 4].any():
                _stamp_mask(grid, base, r, c, color)
                placed[r - 1 : r + 4, c - 1 : c + 4] = True
                break
    return grid
