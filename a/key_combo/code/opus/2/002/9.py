import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _hole_count(crop):
    mask = crop == 0
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        border = False
        while q:
            r, c = q.popleft()
            if r in (0, crop.shape[0] - 1) or c in (0, crop.shape[1] - 1):
                border = True
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < crop.shape[0] and 0 <= nc < crop.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        if not border:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    objs = []
    for comp in _components(grid == 7):
        r0, r1, c0, c1 = _bbox(comp)
        crop = np.where(grid[r0:r1, c0:c1] == 7, 7, 0)
        objs.append({"comp": comp, "bbox": (r0, r1, c0, c1), "crop": crop, "holes": _hole_count(crop)})

    markers = [tuple(map(int, p)) for p in np.argwhere((grid != 0) & (grid != 7))]
    for obj in objs:
        pts = obj["comp"]
        mr, mc = min(markers, key=lambda p: min(abs(p[0] - r) + abs(p[1] - c) for r, c in pts))
        obj["color"] = int(grid[mr, mc])

    stamps = []
    for obj in sorted(objs, key=lambda x: x["holes"]):
        stamps.append(np.where(obj["crop"] == 7, obj["color"], 0))
    height = max(s.shape[0] for s in stamps)
    width = sum(s.shape[1] for s in stamps) + len(stamps) - 1
    out = np.zeros((height, width), dtype=int)
    c = 0
    for stamp in stamps:
        out[: stamp.shape[0], c : c + stamp.shape[1]] = stamp
        c += stamp.shape[1] + 1
    return out


def _place(grid, mask, color):
    h, w = mask.shape
    spots = [(r, c) for r in range(grid.shape[0] - h - 1) for c in range(grid.shape[1] - w - 1)]
    random.shuffle(spots)
    for r, c in spots:
        if np.any(grid[max(0, r - 1) : r + h + 2, max(0, c - 1) : c + w + 2] != 0):
            continue
        grid[r : r + h, c : c + w] = np.where(mask, 7, 0)
        options = [(r - 1, c), (r + h, c + w - 1), (r, c + w), (r + h - 1, c - 1)]
        random.shuffle(options)
        for mr, mc in options:
            if 0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1] and grid[mr, mc] == 0:
                grid[mr, mc] = color
                return True
    return False


def generate_input() -> np.ndarray:
    masks = [
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
    ]
    for _ in range(200):
        grid = np.zeros((15, 15), dtype=int)
        ok = True
        for mask, color in zip(random.sample(masks, len(masks)), random.sample([1, 2, 3, 4], 3)):
            ok = ok and _place(grid, mask, color)
        if ok:
            return grid
    return grid
