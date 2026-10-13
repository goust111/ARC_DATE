import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
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


def _bbox(cells: np.ndarray):
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    return r0, r1, c0, c1


def _crop(grid: np.ndarray, cells: np.ndarray):
    r0, r1, c0, c1 = _bbox(cells)
    color = int(grid[cells[0, 0], cells[0, 1]])
    mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    return mask, (r0, r1, c0, c1), color


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    divider = int(np.where(np.all(grid == GREY, axis=1))[0][0])

    windows = []
    for frame_color in sorted(set(grid[:divider].reshape(-1).tolist()) - {BLACK, GREY, MAROON}):
        for comp in _components(grid[:divider] == frame_color):
            mask, (r0, r1, c0, c1), _ = _crop(grid, comp)
            border = np.zeros_like(mask, dtype=bool)
            border[0] = True
            border[-1] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(mask, border):
                continue
            inner_h = mask.shape[0] - 2
            inner_w = mask.shape[1] - 2
            hints = set(map(tuple, np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == MAROON)))
            windows.append(
                {
                    "bbox": (r0, r1, c0, c1),
                    "inner": (inner_h, inner_w),
                    "area": inner_h * inner_w,
                    "hints": hints,
                    "frame_color": frame_color,
                }
            )

    patterns = []
    for color in sorted(set(grid[divider + 1 :].reshape(-1).tolist()) - {BLACK, GREY}):
        for comp in _components(grid[divider + 1 :] == color):
            comp = comp.copy()
            comp[:, 0] += divider + 1
            mask, bbox, _ = _crop(grid, comp)
            patterns.append((mask, bbox, color, int(mask.sum())))

    used = set()
    for mask, bbox, color, count in patterns:
        r0, r1, c0, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1][mask] = BLACK
        match_idx = None
        for i, win in enumerate(windows):
            if i in used:
                continue
            if win["area"] == count:
                match_idx = i
                break
        if match_idx is None:
            for i, win in enumerate(windows):
                if i in used:
                    continue
                if mask.shape[0] * mask.shape[1] == win["area"]:
                    match_idx = i
                    break
        used.add(match_idx)
        win = windows[match_idx]
        inner_h, inner_w = win["inner"]
        candidates = []
        for cand in (mask, np.rot90(mask)):
            if cand.shape == (inner_h, inner_w):
                candidates.append(cand)
        chosen = candidates[0]
        if len(candidates) > 1:
            for cand in candidates:
                if all(cand[r, c] for r, c in win["hints"]):
                    chosen = cand
                    break
        wr0, wr1, wc0, wc1 = win["bbox"]
        out[wr0 + 1 : wr1, wc0 + 1 : wc1] = BLACK
        out[wr0 + 1 : wr1, wc0 + 1 : wc1][chosen] = color
    return out


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 6, 7, 8]
    frame_colors = [1, 2, 3, 4, 6]
    while True:
        h = random.randint(20, 25)
        w = random.randint(18, 24)
        divider = random.randint(8, 11)
        grid = np.zeros((h, w), dtype=int)
        grid[divider, :] = GREY
        occupied = np.zeros((h, w), dtype=bool)
        occupied[divider, :] = True

        count = random.randint(2, 3)
        dims = []
        used_areas = set()
        for _ in range(count):
            for _ in range(100):
                inner_h = random.randint(2, 4)
                inner_w = random.randint(2, 5)
                area = inner_h * inner_w
                if area not in used_areas:
                    used_areas.add(area)
                    dims.append((inner_h, inner_w))
                    break

        random.shuffle(frame_colors)
        ok = True
        windows = []
        for (inner_h, inner_w), frame_color in zip(dims, frame_colors):
            fh, fw = inner_h + 2, inner_w + 2
            placed = False
            for _ in range(200):
                r = random.randint(0, divider - fh)
                c = random.randint(0, w - fw)
                area = occupied[r : r + fh, c : c + fw]
                if np.any(area):
                    continue
                grid[r, c : c + fw] = frame_color
                grid[r + fh - 1, c : c + fw] = frame_color
                grid[r : r + fh, c] = frame_color
                grid[r : r + fh, c + fw - 1] = frame_color
                hint_count = random.randint(1, 2)
                for hr, hc in random.sample([(rr, cc) for rr in range(inner_h) for cc in range(inner_w)], hint_count):
                    grid[r + 1 + hr, c + 1 + hc] = MAROON
                area[:] = True
                windows.append((r, c, inner_h, inner_w))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        random.shuffle(colors)
        for color, (r, c, inner_h, inner_w) in zip(colors, windows):
            if random.random() < 0.5:
                ph, pw = inner_h, inner_w
            else:
                ph, pw = inner_w, inner_h
            placed = False
            for _ in range(200):
                pr = random.randint(divider + 1, h - ph)
                pc = random.randint(0, w - pw)
                area = occupied[pr : pr + ph, pc : pc + pw]
                if np.any(area):
                    continue
                grid[pr : pr + ph, pc : pc + pw] = color
                area[:] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
