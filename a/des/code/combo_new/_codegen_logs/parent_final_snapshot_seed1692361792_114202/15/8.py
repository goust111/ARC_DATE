import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2


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
    center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
    return {"cells": cells, "bbox": (r0, r1, c0, c1), "mask": mask, "color": color, "area": int(mask.sum()), "center": center}


def _mirror_center(center, axis_kind, axis_value):
    r, c = center
    if axis_kind == "row":
        return (2 * axis_value - r, c)
    return (r, 2 * axis_value - c)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    if np.any(np.all(grid == RED, axis=1)):
        axis_kind = "row"
        axis_value = int(np.where(np.all(grid == RED, axis=1))[0][0])
    else:
        axis_kind = "col"
        axis_value = int(np.where(np.all(grid == RED, axis=0))[0][0])

    comps = []
    for color in sorted(set(grid.reshape(-1).tolist()) - {BLACK, RED}):
        for comp in _components(grid == color):
            comps.append(_crop(grid, comp))
    if axis_kind == "row":
        side_a = [comp for comp in comps if comp["center"][0] < axis_value]
        side_b = [comp for comp in comps if comp["center"][0] > axis_value]
    else:
        side_a = [comp for comp in comps if comp["center"][1] < axis_value]
        side_b = [comp for comp in comps if comp["center"][1] > axis_value]

    remaining = side_b[:]
    pairs = []
    for comp in side_a:
        mr, mc = _mirror_center(comp["center"], axis_kind, axis_value)
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: abs(remaining[idx]["center"][0] - mr) + abs(remaining[idx]["center"][1] - mc),
        )
        best = remaining.pop(best_idx)
        pairs.append((comp, best))

    for a, b in pairs:
        small, large = (a, b) if a["area"] <= b["area"] else (b, a)
        for r, c in large["cells"]:
            out[r, c] = BLACK
        mirrored = np.flipud(small["mask"]) if axis_kind == "row" else np.fliplr(small["mask"])
        lr0, lr1, lc0, lc1 = large["bbox"]
        common_h = min(mirrored.shape[0], large["mask"].shape[0])
        common_w = min(mirrored.shape[1], large["mask"].shape[1])
        trimmed = np.zeros_like(large["mask"], dtype=bool)
        trimmed[:common_h, :common_w] = large["mask"][:common_h, :common_w] & mirrored[:common_h, :common_w]
        out[lr0 : lr1 + 1, lc0 : lc1 + 1][trimmed] = large["color"]
    return out


def _random_connected_mask(h: int, w: int, target: int):
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        r, c = random.choice(list(cells))
        options = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(options)
        for nr, nc in options:
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                cells.add((nr, nc))
                break
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _grow_connected(mask: np.ndarray, extra: int):
    mask = mask.copy()
    h, w = mask.shape
    while extra > 0:
        frontier = []
        for r, c in np.argwhere(mask):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
        extra -= 1
    return mask


def _mirror_top_left(r: int, c: int, h: int, w: int, axis_kind: str, axis_value: int):
    if axis_kind == "row":
        return 2 * axis_value - (r + h - 1), c
    return r, 2 * axis_value - (c + w - 1)


def generate_input() -> np.ndarray:
    colors = [1, 3, 4, 6, 7, 8, 9]
    while True:
        h = random.randint(16, 20)
        w = random.randint(16, 20)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        axis_kind = random.choice(["row", "col"])
        if axis_kind == "row":
            axis_value = h // 2
            grid[axis_value, :] = RED
            occupied[axis_value, :] = True
        else:
            axis_value = w // 2
            grid[:, axis_value] = RED
            occupied[:, axis_value] = True

        random.shuffle(colors)
        count = random.randint(2, 3)
        ok = True
        for color_small, color_large in zip(colors[::2], colors[1::2]):
            bh = random.randint(2, 4)
            bw = random.randint(2, 4)
            small_area = random.randint(max(2, (bh * bw) // 3), bh * bw - 1)
            small = _random_connected_mask(bh, bw, small_area)
            mirrored_small = np.flipud(small) if axis_kind == "row" else np.fliplr(small)
            extra_budget = random.randint(1, max(1, bh * bw - small_area))
            large = _grow_connected(mirrored_small, extra_budget)
            if axis_kind == "row":
                r = random.randint(0, axis_value - bh - 1)
                c = random.randint(0, w - bw)
            else:
                r = random.randint(0, h - bh)
                c = random.randint(0, axis_value - bw - 1)
            mr, mc = _mirror_top_left(r, c, bh, bw, axis_kind, axis_value)
            if mr < 0 or mc < 0 or mr + bh > h or mc + bw > w:
                ok = False
                break
            if np.any(occupied[r : r + bh, c : c + bw] & small):
                ok = False
                break
            if np.any(occupied[mr : mr + bh, mc : mc + bw] & large):
                ok = False
                break
            small_side_first = random.random() < 0.5
            if small_side_first:
                grid[r : r + bh, c : c + bw][small] = color_small
                grid[mr : mr + bh, mc : mc + bw][large] = color_large
            else:
                grid[r : r + bh, c : c + bw][large] = color_large
                grid[mr : mr + bh, mc : mc + bw][small] = color_small
            occupied[r : r + bh, c : c + bw] |= small | large
            occupied[mr : mr + bh, mc : mc + bw] |= small | large
            count -= 1
            if count == 0:
                break
        if ok and np.count_nonzero(grid) > (w if axis_kind == "row" else h):
            return grid
