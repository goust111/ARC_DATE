import random

import numpy as np


BG = 0
BLUE = 1
PALETTE_COLORS = [2, 3, 4, 6, 7, 8, 9]


def _palette(grid):
    best = None
    best_count = 0
    for r in range(grid.shape[0] - 1, -1, -1):
        cols = [int(v) for v in grid[r] if v not in (BG, BLUE)]
        if len(cols) > best_count:
            best = cols
            best_count = len(cols)
    return best or []


def _room_candidates(grid):
    h, w = grid.shape
    candidates = []
    blue = np.argwhere(grid == BLUE)
    if blue.size == 0:
        return []
    rows = sorted(set(int(r) for r, _ in blue))
    cols = sorted(set(int(c) for _, c in blue))
    for r0 in rows:
        for r1 in rows:
            if r1 <= r0 + 2:
                continue
            for c0 in cols:
                for c1 in cols:
                    if c1 <= c0 + 2:
                        continue
                    if not all(grid[r, c] == BLUE for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
                        continue
                    border = (
                        [(r0, c) for c in range(c0, c1 + 1)]
                        + [(r1, c) for c in range(c0, c1 + 1)]
                        + [(r, c0) for r in range(r0 + 1, r1)]
                        + [(r, c1) for r in range(r0 + 1, r1)]
                    )
                    vals = [int(grid[r, c]) for r, c in border]
                    if any(v not in (BG, BLUE) for v in vals):
                        continue
                    if vals.count(BLUE) < len(vals) - 4:
                        continue
                    if np.any(grid[r0 + 1:r1, c0 + 1:c1] == BLUE):
                        continue
                    candidates.append((r0, r1, c0, c1))
    # Keep minimal rectangles; generated rooms do not overlap.
    unique = []
    seen = set()
    for rect in candidates:
        if rect not in seen:
            seen.add(rect)
            unique.append(rect)
    return unique


def _gap_count(grid, rect):
    r0, r1, c0, c1 = rect
    cells = (
        [(r0, c) for c in range(c0 + 1, c1)]
        + [(r1, c) for c in range(c0 + 1, c1)]
        + [(r, c0) for r in range(r0 + 1, r1)]
        + [(r, c1) for r in range(r0 + 1, r1)]
    )
    return sum(1 for r, c in cells if grid[r, c] == BG)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill blue-bounded rooms using a bottom palette indexed by window count.

    Input structure:
    Blue line rectangles form rooms on a black background.  Some border pixels are
    missing as black one-cell windows.  A bottom palette row lists colors separated
    by blue pixels.

    Rule:
    Count each room's black gaps along its blue border.  A room with N gaps is filled
    with the Nth palette color, while rooms with no gaps stay black.  Blue borders,
    gap cells, and the palette remain unchanged.

    Key insight:
    The visually small border gaps encode an index into the ordered bottom palette.
    """
    out = np.array(grid, copy=True)
    palette = _palette(grid)
    for rect in _room_candidates(grid):
        n = _gap_count(grid, rect)
        if n <= 0 or n > len(palette):
            continue
        r0, r1, c0, c1 = rect
        interior = out[r0 + 1:r1, c0 + 1:c1]
        interior[interior == BG] = palette[n - 1]
    return out


def _draw_room(grid, r0, c0, rh, rw, gaps):
    r1, c1 = r0 + rh - 1, c0 + rw - 1
    if r1 >= grid.shape[0] - 2 or c1 >= grid.shape[1] - 1:
        return False
    if np.any(grid[r0:r1 + 1, c0:c1 + 1] != BG):
        return False
    grid[r0, c0:c1 + 1] = BLUE
    grid[r1, c0:c1 + 1] = BLUE
    grid[r0:r1 + 1, c0] = BLUE
    grid[r0:r1 + 1, c1] = BLUE
    border = (
        [(r0, c) for c in range(c0 + 1, c1)]
        + [(r1, c) for c in range(c0 + 1, c1)]
        + [(r, c0) for r in range(r0 + 1, r1)]
        + [(r, c1) for r in range(r0 + 1, r1)]
    )
    for r, c in random.sample(border, gaps):
        grid[r, c] = BG
    return True


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = random.randint(18, 29)
        w = random.randint(18, 29)
        grid = np.zeros((h, w), dtype=int)
        palette_len = random.randint(3, 5)
        start = random.randint(1, max(1, w - palette_len * 2))
        colors = random.sample(PALETTE_COLORS, palette_len)
        for i, color in enumerate(colors):
            c = start + i * 2
            grid[h - 1, c] = color
            if c + 1 < w:
                grid[h - 1, c + 1] = BLUE

        placements = [(1, 1), (1, w // 2), (h // 2, 1), (h // 2, w // 2)]
        random.shuffle(placements)
        n_rooms = random.randint(2, 4)
        ok = True
        used_gap_counts = []
        for i in range(n_rooms):
            r0, c0 = placements[i]
            rh = random.randint(4, 6)
            rw = random.randint(5, 8)
            gaps = random.randint(0, palette_len)
            if not _draw_room(grid, r0, c0, rh, rw, gaps):
                ok = False
                break
            used_gap_counts.append(gaps)
        if ok and any(g > 0 for g in used_gap_counts) and not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid window-palette room puzzle")
