import random
import numpy as np


BG = 0
MAROON = 9


def _column_groups(cols):
    groups = []
    cur = []
    last = None
    for c in sorted(cols):
        if last is None or c <= last + 1:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
        last = c
    if cur:
        groups.append(cur)
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """Grow seed pixels using same-colored top templates and maroon counts.

    Input structure: The top band contains separated square growth templates;
    each template center color identifies the seed color and surrounding
    non-black cells define one growth layer. Lower seed pixels have zero to
    three adjacent maroon count markers.
    Rule: For each lower seed, count adjacent maroon pixels, clear all maroon
    markers, and iteratively apply the matching template offsets that many
    layers. Existing colored cells are never overwritten and off-grid growth is
    clipped. The templates remain unchanged.
    Key insight: Seed color chooses the template, template geometry chooses the
    spread, and local maroon count chooses the number of iterations.
    """
    out = grid.copy()
    h, w = grid.shape
    top_rows = range(0, min(6, h))
    cols = {c for r in top_rows for c in range(w) if grid[r, c] not in (BG, MAROON)}
    templates = {}
    for group in _column_groups(cols):
        if not group:
            continue
        c0, c1 = min(group), max(group)
        rows = [r for r in top_rows for c in group if grid[r, c] not in (BG, MAROON)]
        r0, r1 = min(rows), max(rows)
        block = grid[r0:r1 + 1, c0:c1 + 1]
        cr, cc = block.shape[0] // 2, block.shape[1] // 2
        center = int(block[cr, cc])
        offsets = []
        for rr in range(block.shape[0]):
            for cc2 in range(block.shape[1]):
                if (rr, cc2) != (cr, cc) and block[rr, cc2] != BG:
                    offsets.append((rr - cr, cc2 - cc, int(block[rr, cc2])))
        if center != BG:
            templates[center] = offsets
    out[out == MAROON] = BG
    for r in range(6, h):
        for c in range(w):
            col = int(grid[r, c])
            if col not in templates:
                continue
            n = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == MAROON:
                    n += 1
            frontier = [(r, c)]
            for _ in range(n):
                new = []
                for fr, fc in frontier:
                    for dr, dc, paint in templates[col]:
                        nr, nc = fr + dr, fc + dc
                        if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == BG:
                            out[nr, nc] = paint
                            new.append((nr, nc))
                frontier = new
    return out


def generate_input() -> np.ndarray:
    h = w = 25
    grid = np.zeros((h, w), dtype=int)
    centers = random.sample([1, 2, 3, 4, 6, 7, 8], 3)
    patterns = [[(-1, 0), (1, 0), (0, 1)], [(0, -1), (0, 1), (1, 0)], [(-1, 0), (0, -1), (1, 0), (0, 1)]]
    for i, (col, offs) in enumerate(zip(centers, patterns)):
        r0, c0 = 1, 1 + i * 7
        grid[r0 + 1, c0 + 1] = col
        for dr, dc in offs:
            grid[r0 + 1 + dr, c0 + 1 + dc] = col
    seed_positions = [(10, 5), (13, 15), (19, 8), (19, 18)]
    for r, c in random.sample(seed_positions, random.randint(2, 4)):
        col = random.choice(centers)
        grid[r, c] = col
        for dr, dc in random.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], random.randint(0, 3)):
            if grid[r + dr, c + dc] == BG:
                grid[r + dr, c + dc] = MAROON
    return grid
