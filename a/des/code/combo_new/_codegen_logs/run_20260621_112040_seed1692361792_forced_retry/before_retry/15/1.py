import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9


def _components(mask: np.ndarray):
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _crop_from_cells(cells: np.ndarray):
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[cells[:, 0] - r0, cells[:, 1] - c0] = True
    return mask, (r0, r1, c0, c1)


def _shape_key(mask: np.ndarray):
    return (mask.shape, tuple(mask.reshape(-1).tolist()))


def _notch_direction(mask: np.ndarray):
    missing = {
        "up": int((~mask[0]).sum()),
        "down": int((~mask[-1]).sum()),
        "left": int((~mask[:, 0]).sum()),
        "right": int((~mask[:, -1]).sum()),
    }
    return max(missing, key=missing.get)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()

    frame_mask = np.isin(grid, [RED, BLUE, GREEN, YELLOW])
    frame_cells = np.argwhere(frame_mask)
    r0, c0 = frame_cells.min(axis=0)
    r1, c1 = frame_cells.max(axis=0)
    side_color = {"up": grid[r0, c0 + 1], "right": grid[r0 + 1, c1], "down": grid[r1, c0 + 1], "left": grid[r0 + 1, c0]}

    inner = np.zeros_like(grid, dtype=bool)
    inner[r0 + 1 : r1, c0 + 1 : c1] = grid[r0 + 1 : r1, c0 + 1 : c1] == ORANGE
    outer = (grid == MAROON).copy()
    outer[r0 : r1 + 1, c0 : c1 + 1] = False

    inside_shapes = []
    for comp in _components(inner):
        mask, _ = _crop_from_cells(comp)
        inside_shapes.append((_shape_key(mask), _notch_direction(mask)))

    outer_lookup = {}
    for comp in _components(outer):
        mask, _ = _crop_from_cells(comp)
        outer_lookup[_shape_key(mask)] = comp

    for key, direction in inside_shapes:
        comp = outer_lookup[key]
        out[comp[:, 0], comp[:, 1]] = side_color[direction]
    return out


def _make_shape(h: int, w: int, direction: str, notch_pos: int):
    mask = np.ones((h, w), dtype=bool)
    if direction == "up":
        mask[0, notch_pos] = False
    elif direction == "down":
        mask[-1, notch_pos] = False
    elif direction == "left":
        mask[notch_pos, 0] = False
    else:
        mask[notch_pos, -1] = False
    return mask


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, row_range, col_range):
    h, w = mask.shape
    rs = list(row_range)
    cs = list(col_range)
    random.shuffle(rs)
    random.shuffle(cs)
    for r in rs:
        for c in cs:
            if r + h > grid.shape[0] or c + w > grid.shape[1]:
                continue
            rr0 = max(0, r - 1)
            rr1 = min(grid.shape[0], r + h + 1)
            cc0 = max(0, c - 1)
            cc1 = min(grid.shape[1], c + w + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            area = occupied[r : r + h, c : c + w]
            grid[r : r + h, c : c + w][mask] = color
            area |= mask
            return True
    return False


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(20, 24)
        w = random.randint(20, 24)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        inner_h = random.randint(8, 12)
        inner_w = random.randint(8, 12)
        fr = random.randint(2, h - inner_h - 4)
        fc = random.randint(2, w - inner_w - 4)
        r0, c0 = fr, fc
        r1, c1 = fr + inner_h + 1, fc + inner_w + 1

        grid[r0, c0:c1 + 1] = RED
        grid[r1, c0:c1 + 1] = GREEN
        grid[r0:r1 + 1, c0] = YELLOW
        grid[r0:r1 + 1, c1] = BLUE
        occupied[r0:r1 + 1, c0:c1 + 1] |= grid[r0:r1 + 1, c0:c1 + 1] != 0

        candidates = []
        for sh in (3, 4):
            for sw in (3, 4):
                for direction in ("up", "down", "left", "right"):
                    limit = sw - 1 if direction in ("up", "down") else sh - 1
                    for notch_pos in range(1, limit):
                        candidates.append((sh, sw, direction, notch_pos))
        random.shuffle(candidates)
        count = random.randint(2, 4)
        picks = candidates[:count]

        ok = True
        for sh, sw, direction, notch_pos in picks:
            mask = _make_shape(sh, sw, direction, notch_pos)
            if not _place_mask(grid, occupied, mask, ORANGE, range(r0 + 2, r1 - sh), range(c0 + 2, c1 - sw)):
                ok = False
                break
            outside_ranges = [
                (range(1, max(2, r0 - sh)), range(1, w - sw - 1)),
                (range(r1 + 2, h - sh - 1), range(1, w - sw - 1)),
                (range(1, h - sh - 1), range(1, max(2, c0 - sw))),
                (range(1, h - sh - 1), range(c1 + 2, w - sw - 1)),
            ]
            random.shuffle(outside_ranges)
            placed = False
            for rr, cc in outside_ranges:
                if _place_mask(grid, occupied, mask, MAROON, rr, cc):
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return grid
