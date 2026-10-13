import random
from collections import deque

import numpy as np


BLOCK_COLORS = (1, 4, 8)


def _neighbors4(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _slot_intervals(grid):
    row = grid.shape[0] - 2
    slots = []
    c = 0
    while c < grid.shape[1]:
        if grid[row, c] != 0:
            c += 1
            continue
        c0 = c
        while c < grid.shape[1] and grid[row, c] == 0:
            c += 1
        slots.append((c0, c))
    return slots


def _insert_blocks(grid):
    out = grid.copy()
    slot_infos = [{"width": c1 - c0, "interval": (c0, c1)} for c0, c1 in _slot_intervals(grid)]
    block_mask = np.isin(grid, BLOCK_COLORS)
    used = set()
    blocks = []
    for cells in _components(block_mask):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(grid[rows[0], cols[0]])
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        blocks.append((max(bh, bw), r0, c0, r1, c1, color))
    for width, r0, c0, r1, c1, color in sorted(blocks, reverse=True):
        out[r0 : r1 + 1, c0 : c1 + 1] = 0
        candidates = []
        for idx, info in enumerate(slot_infos):
            if idx in used:
                continue
            candidates.append((abs(info["width"] - width), info["width"], idx, info["interval"]))
        if not candidates:
            for idx, info in enumerate(slot_infos):
                candidates.append((abs(info["width"] - width), info["width"], idx, info["interval"]))
        _, _, idx, (sc0, sc1) = min(candidates)
        used.add(idx)
        out[-3:-1, sc0:sc1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summary:
    Insert each floating block into its matching slot, then fill the cavity above it and radiate upward through the opening.

    Input Model:
    A gray platform spans the bottom, black gaps on the row above are slots, green side walls rise from the slot edges, orange seeds sit inside the future cavities, and brown obstacles may stand in the upward radiation band.

    Algorithm:
    Match every floating rectangle to the unique slot with the same long side, erase the block from its source, place it as a two-row insert above the platform, fill the rectangular cavity between the green walls, and then project the filled band upward while letting each obstacle cast a one-column shadow behind itself.

    Output:
    Return the transformed grid with inserted blocks in place, seeds expanded to orange regions and upward rays, and all structural colors preserved.
    """
    out = _insert_blocks(np.asarray(grid, dtype=int))
    seeds = np.argwhere(out == 7)
    for sr, sc in seeds:
        left = sc - 1
        while left >= 0 and out[sr, left] != 3:
            left -= 1
        right = sc + 1
        while right < out.shape[1] and out[sr, right] != 3:
            right += 1
        if left < 0 or right >= out.shape[1]:
            continue
        wall_rows = [r for r in range(out.shape[0]) if out[r, left] == 3 and out[r, right] == 3]
        if not wall_rows:
            continue
        top = min(wall_rows)
        for r in range(top, out.shape[0] - 3):
            for c in range(left + 1, right):
                if out[r, c] == 0 or out[r, c] == 7:
                    out[r, c] = 7
        shadow = set()
        for r in range(top - 1, -1, -1):
            for c in range(left + 1, right):
                if out[r, c] == 9:
                    shadow.add(c)
                elif out[r, c] == 0 and c not in shadow:
                    out[r, c] = 7
    return out


def _generate_case(rng: random.Random):
    h = rng.randint(15, 19)
    slot_widths = rng.sample([2, 3, 4], rng.randint(2, 3))
    gaps = [rng.randint(1, 2) for _ in range(len(slot_widths) + 1)]
    w = sum(slot_widths) + sum(gaps)
    grid = np.zeros((h, w), dtype=int)
    grid[-1, :] = 5
    row = h - 2
    grid[row, :] = 5
    slots = []
    c = gaps[0]
    for width, gap in zip(slot_widths, gaps[1:]):
        slots.append((c, c + width))
        grid[row, c : c + width] = 0
        c += width + gap
    colors = rng.sample(BLOCK_COLORS, len(slots))
    expected = grid.copy()
    for (c0, c1), color in zip(slots, colors):
        top = rng.randint(3, 7)
        grid[top : h - 2, c0 - 1] = 3
        grid[top : h - 2, c1] = 3
        expected[top : h - 2, c0 - 1] = 3
        expected[top : h - 2, c1] = 3
        expected[-3:-1, c0:c1] = color
        seed_r = h - 5
        seed_c = (c0 + c1 - 1) // 2
        grid[seed_r, seed_c] = 7
        expected[top : h - 3, c0:c1] = 7
        obstacle_count = rng.randint(0, min(2, c1 - c0))
        obstacle_cols = rng.sample(range(c0, c1), obstacle_count)
        for oc in obstacle_cols:
            orow = rng.randint(0, max(0, top - 1))
            grid[orow, oc] = 9
            expected[orow, oc] = 9
        shadow = set(obstacle_cols)
        for r in range(top - 1, -1, -1):
            for cc in range(c0, c1):
                if expected[r, cc] == 9:
                    continue
                if cc not in shadow:
                    expected[r, cc] = 7
        shape = np.full((2, c1 - c0), color, dtype=int)
        if rng.choice((True, False)):
            shape = shape.T
        for _ in range(500):
            rh, rw = shape.shape
            r0 = rng.randint(0, h - 4 - rh)
            cstart = rng.randint(0, w - rw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, cstart - 1)
            rr1 = min(h - 1, r0 + rh + 1)
            cc1 = min(w, cstart + rw + 1)
            if np.any(grid[rr0:rr1, cc0:cc1] != 0):
                continue
            grid[r0 : r0 + rh, cstart : cstart + rw] = shape
            break
    return grid, main(grid)


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(random.randrange(1 << 30)))[0]
