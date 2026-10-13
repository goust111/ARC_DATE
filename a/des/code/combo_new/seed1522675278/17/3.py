import random
from collections import defaultdict, deque

import numpy as np


BG = 0


def _components_by_color(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _bbox_mask(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    return mask, r0, c0


def _fill_holes(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = mask.copy()
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            region = []
            touches = False
            while q:
                cr, cc = q.popleft()
                region.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches:
                for rr, cc in region:
                    out[rr, cc] = 1
    return out


def _holes(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    regions = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            region = []
            touches = False
            while q:
                cr, cc = q.popleft()
                region.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches:
                regions.append(region)
    return regions


def _canonical(mask):
    variants = []
    for flip in (False, True):
        cur = np.fliplr(mask) if flip else mask
        for k in range(4):
            rot = np.rot90(cur, k)
            rows = np.any(rot, axis=1)
            cols = np.any(rot, axis=0)
            cropped = rot[rows][:, cols]
            variants.append(tuple(map(tuple, cropped)))
    return min(variants)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    groups = defaultdict(list)
    comps = []
    for color, cells in _components_by_color(grid):
        mask, r0, c0 = _bbox_mask(cells)
        filled = _fill_holes(mask)
        groups[_canonical(filled)].append((color, cells, mask, r0, c0))
        comps.append((color, cells, mask, r0, c0))

    for members in groups.values():
        if len(members) != 2:
            continue
        (color_a, cells_a, mask_a, r0a, c0a), (color_b, cells_b, mask_b, r0b, c0b) = members
        for region in _holes(mask_a):
            for rr, cc in region:
                out[r0a + rr, c0a + cc] = color_b
        for region in _holes(mask_b):
            for rr, cc in region:
                out[r0b + rr, c0b + cc] = color_a
    return out


def _base_shape():
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    arr = np.zeros((h, w), dtype=int)
    arr[1:-1, 1:-1] = 1
    if random.random() < 0.5:
        arr[:, random.randrange(1, w - 1)] = 1
    if random.random() < 0.5:
        arr[random.randrange(1, h - 1), :] = 1
    return arr


def _carve_holes(mask):
    arr = mask.copy()
    interior = [(r, c) for r in range(1, arr.shape[0] - 1) for c in range(1, arr.shape[1] - 1)]
    random.shuffle(interior)
    for r, c in interior[: random.randint(0, min(2, len(interior)))]:
        if arr[r, c]:
            arr[r, c] = 0
    return arr


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(4, 6))
    color_idx = 0
    for _ in range(len(colors) // 2):
        base = _base_shape()
        a = _carve_holes(base)
        b = _carve_holes(base)
        b = np.rot90(b, random.randrange(4))
        if random.random() < 0.5:
            b = np.fliplr(b)
        for shape in (a, b):
            sh, sw = shape.shape
            color = colors[color_idx]
            color_idx += 1
            for _ in range(200):
                top = random.randint(0, h - sh)
                left = random.randint(0, w - sw)
                if occupied[max(0, top - 1) : min(h, top + sh + 1), max(0, left - 1) : min(w, left + sw + 1)].any():
                    continue
                patch = shape * color
                grid[top : top + sh, left : left + sw][patch != 0] = color
                occupied[top : top + sh, left : left + sw] |= patch != 0
                break
    return grid
