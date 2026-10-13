import random
from collections import deque

import numpy as np


BG = 0


def _components(mask, eight=False):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if eight:
        neigh = neigh + ((1, 1), (1, -1), (-1, 1), (-1, -1))
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
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _frame_info(grid):
    infos = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        mask = grid == color
        for cells in _components(mask):
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            expected = set()
            for c in range(c0, c1 + 1):
                expected.add((r0, c))
                expected.add((r1, c))
            for r in range(r0, r1 + 1):
                expected.add((r, c0))
                expected.add((r, c1))
            if set(cells) == expected:
                infos.append((color, r0, r1, c0, c1))
    return infos


def _crop_component(grid, cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        arr[r - r0, c - c0] = grid[r, c]
    return arr


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    frames = _frame_info(grid)
    frame_mask = np.zeros_like(grid, dtype=bool)
    for _, r0, r1, c0, c1 in frames:
        frame_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    templates = []
    nonframe = (grid != BG) & (~frame_mask)
    for cells in _components(nonframe, eight=True):
        crop = _crop_component(grid, cells)
        colors = {int(v) for v in crop.ravel() if v != BG}
        if len(colors) == 2:
            templates.append((colors, crop))

    for frame_color, r0, r1, c0, c1 in frames:
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        fragment_colors = [int(v) for v in np.unique(inner) if v not in (BG, frame_color)]
        if not fragment_colors:
            continue
        key_color = fragment_colors[0]
        fragment_coords = set(map(tuple, np.argwhere(inner == key_color)))
        chosen = None
        for colors, template in templates:
            if key_color not in colors:
                continue
            for k in range(4):
                rot = np.rot90(template, -k)
                special = set(map(tuple, np.argwhere(rot == key_color)))
                if fragment_coords and fragment_coords.issubset(special):
                    chosen = rot
                    break
            if chosen is not None:
                break
        if chosen is None:
            continue
        h, w = chosen.shape
        out[r0 + 1 : r0 + 1 + h, c0 + 1 : c0 + 1 + w] = chosen
    return out


def _random_template():
    h = random.randint(3, 4)
    w = random.randint(3, 4)
    special = random.randint(1, 4)
    fill = random.choice([c for c in [1, 2, 3, 4, 5, 6, 7, 8, 9] if c != special])
    arr = np.full((h, w), fill, dtype=int)
    coords = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(coords)
    chosen = coords[: random.randint(3, min(5, h * w - 1))]
    for r, c in chosen:
        arr[r, c] = special
    if len({tuple(x) for x in np.argwhere(arr == special)}) < 3:
        return _random_template()
    return arr, special


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    frame_colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 3))
    templates = []

    for frame_color in frame_colors:
        template, special = _random_template()
        th, tw = template.shape
        ih = th + random.randint(0, 1)
        iw = tw + random.randint(0, 1)
        fh, fw = ih + 2, iw + 2
        for _ in range(200):
            top = random.randint(1, h - fh - 1)
            left = random.randint(1, w - fw - 1)
            if occupied[max(0, top - 1) : top + fh + 1, max(0, left - 1) : left + fw + 1].any():
                continue
            grid[top, left : left + fw] = frame_color
            grid[top + fh - 1, left : left + fw] = frame_color
            grid[top : top + fh, left] = frame_color
            grid[top : top + fh, left + fw - 1] = frame_color
            occupied[top : top + fh, left : left + fw] = True
            rot = np.rot90(template, random.randrange(4))
            special_cells = list(map(tuple, np.argwhere(rot == special)))
            keep = random.sample(special_cells, random.randint(1, len(special_cells)))
            for r, c in keep:
                grid[top + 1 + r, left + 1 + c] = special
            templates.append((template, special))
            break

    for template, special in templates:
        th, tw = template.shape
        for _ in range(200):
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)
            box = occupied[max(0, top - 1) : min(h, top + th + 1), max(0, left - 1) : min(w, left + tw + 1)]
            if box.any():
                continue
            grid[top : top + th, left : left + tw] = template
            occupied[top : top + th, left : left + tw] = True
            break
    return grid
