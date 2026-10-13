import random
from collections import defaultdict, deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _hole_count(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                cr, cc = q.popleft()
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = np.zeros_like(grid)
    template_color = int(grid[0, 0])
    template_cells = np.argwhere(grid == template_color)
    max_r = int(template_cells[:, 0].max())
    max_c = int(template_cells[:, 1].max())
    layer_cells = defaultdict(list)
    for r, c in template_cells:
        inset = min(r, c, max_r - r, max_c - c)
        layer_cells[int(inset)].append((int(r), int(c)))
    layer_keys = sorted(layer_cells)

    groups = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, template_color):
            continue
        for cells in _components(grid == color):
            holes = _hole_count(cells)
            area = len(cells)
            best = groups.get(holes)
            if best is None or area > best[0]:
                groups[holes] = (area, color)

    for inset, hole_count in zip(layer_keys, sorted(groups)):
        color = groups[hole_count][1]
        for r, c in layer_cells[inset]:
            out[r, c] = color
    return out


def _place_block(grid, top, left, color, holes):
    h = random.randint(max(5, holes * 2 + 3), max(6, holes * 2 + 5))
    w = random.randint(max(5, holes * 2 + 3), max(6, holes * 2 + 5))
    grid[top : top + h, left : left + w] = color
    candidates = [
        (r, c)
        for r in range(top + 1, top + h - 1)
        for c in range(left + 1, left + w - 1)
        if (r - top) % 2 == 1 and (c - left) % 2 == 1
    ]
    random.shuffle(candidates)
    for r, c in candidates[:holes]:
        grid[r, c] = BG


def generate_input() -> np.ndarray:
    layers = random.randint(2, 4)
    side = 4 * layers - 1
    min_extent = max(18, side + 12)
    h = random.randint(min_extent, 28)
    w = random.randint(min_extent, 28)
    grid = np.zeros((h, w), dtype=int)
    template_color = 5
    for inset in range(0, 2 * layers, 2):
        bottom = side - 1 - inset
        right = side - 1 - inset
        grid[inset, inset : right + 1] = template_color
        grid[bottom, inset : right + 1] = template_color
        grid[inset : bottom + 1, inset] = template_color
        grid[inset : bottom + 1, right] = template_color

    counts = list(range(layers))
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], layers * 2)
    occupied = grid != BG
    color_idx = 0
    for holes in counts:
        variants = random.randint(1, 2)
        big_variant = random.randrange(variants)
        for i in range(variants):
            color = colors[color_idx]
            color_idx += 1
            tries = 0
            while True:
                tries += 1
                if tries > 200:
                    break
                block_h = max(5, holes * 2 + 3) + (2 if i == big_variant else 0)
                block_w = max(5, holes * 2 + 3) + (2 if i == big_variant else 0)
                if side + 1 > h - block_h - 1 or side + 1 > w - block_w - 1:
                    continue
                top = random.randint(side + 1, h - block_h - 1)
                left = random.randint(side + 1, w - block_w - 1)
                if occupied[max(0, top - 1) : top + block_h + 1, max(0, left - 1) : left + block_w + 1].any():
                    continue
                sub = np.zeros_like(grid)
                _place_block(sub, top, left, color, holes)
                if (sub != BG).sum() == 0:
                    continue
                occupied |= sub != BG
                grid[sub != BG] = sub[sub != BG]
                break
    return grid
