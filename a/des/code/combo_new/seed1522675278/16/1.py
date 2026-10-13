import random
from collections import deque

import numpy as np


BG = 0


def _components_of_color(grid, color):
    mask = grid == color
    h, w = grid.shape
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


def _crop(color, cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        arr[r - r0, c - c0] = color
    return arr, r0, c0


def _find_frame(grid):
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            continue
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        if all(r in (r0, r1) or c in (c0, c1) for r, c in coords):
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if best is None or area > best[0]:
                best = (area, color, r0, r1, c0, c1)
    return best[1:]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    frame_color, r0, r1, c0, c1 = _find_frame(grid)
    gaps = []
    for side, coords in (
        ("top", [(r0, c) for c in range(c0 + 1, c1)]),
        ("right", [(r, c1) for r in range(r0 + 1, r1)]),
        ("bottom", [(r1, c) for c in range(c0 + 1, c1)]),
        ("left", [(r, c0) for r in range(r0 + 1, r1)]),
    ):
        segment = []
        for rr, cc in coords:
            v = int(grid[rr, cc])
            if v not in (BG, frame_color):
                segment.append((rr, cc, v))
        if segment:
            color = segment[0][2]
            if side in ("top", "bottom"):
                tangential = [cc for _, cc, _ in segment]
            else:
                tangential = [rr for rr, _, _ in segment]
            gaps.append((side, color, min(tangential), max(tangential)))

    for _, color, _, _ in gaps:
        for cells in _components_of_color(grid, color):
            if all(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in cells):
                for r, c in cells:
                    out[r, c] = BG
                sprite, _, _ = _crop(color, cells)
                sh, sw = sprite.shape
                if color not in [g[1] for g in gaps]:
                    continue
                side, _, a0, _ = next(g for g in gaps if g[1] == color)
                if side == "top":
                    top, left = r1 - sh, a0
                elif side == "bottom":
                    top, left = r0 + 1, a0
                elif side == "left":
                    top, left = a0, c1 - sw
                else:
                    top, left = a0, c0 + 1
                patch = out[top : top + sh, left : left + sw]
                patch[sprite != 0] = color
                break
    return out


def _sprite(color, length, depth):
    arr = np.zeros((depth, length), dtype=int)
    arr[:, :] = color
    for _ in range(random.randint(0, max(0, depth * length // 3))):
        r = random.randrange(depth)
        c = random.randrange(length)
        if arr.sum() > color * 2:
            arr[r, c] = BG
    rows = np.any(arr != 0, axis=1)
    cols = np.any(arr != 0, axis=0)
    return arr[rows][:, cols]


def generate_input() -> np.ndarray:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    grid = np.zeros((h, w), dtype=int)
    frame_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    other_colors = [c for c in [1, 2, 3, 4, 5, 6, 7, 8, 9] if c != frame_color]
    gap_colors = random.sample(other_colors, 4)
    fh = random.randint(10, min(14, h - 8))
    fw = random.randint(10, min(14, w - 8))
    top = random.randint(4, h - fh - 4)
    left = random.randint(4, w - fw - 4)
    bottom, right = top + fh - 1, left + fw - 1
    grid[top, left : right + 1] = frame_color
    grid[bottom, left : right + 1] = frame_color
    grid[top : bottom + 1, left] = frame_color
    grid[top : bottom + 1, right] = frame_color
    gap_lengths = {
        "top": random.randint(2, 4),
        "right": random.randint(2, 4),
        "bottom": random.randint(2, 4),
        "left": random.randint(2, 4),
    }
    placements = {}
    placements["top"] = random.randint(left + 2, right - gap_lengths["top"] - 2)
    placements["bottom"] = random.randint(left + 2, right - gap_lengths["bottom"] - 2)
    placements["left"] = random.randint(top + 2, bottom - gap_lengths["left"] - 2)
    placements["right"] = random.randint(top + 2, bottom - gap_lengths["right"] - 2)
    sides = ["top", "right", "bottom", "left"]
    for side, color in zip(sides, gap_colors):
        gl = gap_lengths[side]
        start = placements[side]
        if side == "top":
            grid[top, start : start + gl] = color
            sprite = _sprite(color, gl, random.randint(2, 4))
            sh, sw = sprite.shape
            place_top = max(0, top - sh - 2)
            place_left = start
        elif side == "bottom":
            grid[bottom, start : start + gl] = color
            sprite = _sprite(color, gl, random.randint(2, 4))
            sh, sw = sprite.shape
            place_top = min(h - sh, bottom + 2)
            place_left = start
        elif side == "left":
            grid[start : start + gl, left] = color
            sprite = _sprite(color, random.randint(2, 4), gl).T
            sh, sw = sprite.shape
            place_top = start
            place_left = max(0, left - sw - 2)
        else:
            grid[start : start + gl, right] = color
            sprite = _sprite(color, random.randint(2, 4), gl).T
            sh, sw = sprite.shape
            place_top = start
            place_left = min(w - sw, right + 2)
        grid[place_top : place_top + sh, place_left : place_left + sw][sprite != 0] = color
    return grid
