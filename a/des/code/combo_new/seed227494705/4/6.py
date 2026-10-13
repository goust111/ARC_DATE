import random
import numpy as np


BG = 0


def _largest_rect(cells):
    s = set(cells)
    rs = sorted({r for r, _ in cells})
    cs = sorted({c for _, c in cells})
    best = None
    for r0 in rs:
        for r1 in rs:
            if r1 < r0:
                continue
            for c0 in cs:
                for c1 in cs:
                    if c1 < c0:
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if best and area <= best[0]:
                        continue
                    if all((r, c) in s for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
                        best = (area, r0, c0, r1, c1)
    return best[1:] if best else None


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Grow colored patterns from seeds attached to same-colored soil rectangles.

    Input structure: Each color forms a solid rectangular soil region with a
    small same-colored seed shape touching one side. Seed shapes are a horizontal
    domino, vertical domino, or 2x2 square.
    Rule: The side of the soil touched by the seed gives the outward growth
    direction. A horizontal seed grows a straight line, a vertical seed grows an
    alternating zigzag, and a square seed grows an expanding triangle. Growth
    length equals the soil side length along the growth axis, and soil/seed
    pixels are preserved.
    Key insight: Direction, motif, and length come from three independent visual
    clues: seed side, seed shape, and soil dimensions.
    """
    out = grid.copy()
    h, w = grid.shape
    for col in [int(x) for x in np.unique(grid) if x != BG]:
        cells = [(r, c) for r, c in zip(*np.where(grid == col))]
        rect = _largest_rect(cells)
        if rect is None:
            continue
        r0, c0, r1, c1 = rect
        soil = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        seed_cells = [p for p in cells if p not in soil]
        if not seed_cells:
            continue
        sr0, sc0 = min(r for r, _ in seed_cells), min(c for _, c in seed_cells)
        sr1, sc1 = max(r for r, _ in seed_cells), max(c for _, c in seed_cells)
        if sr1 < r0:
            d, length = (-1, 0), r1 - r0 + 1
        elif sr0 > r1:
            d, length = (1, 0), r1 - r0 + 1
        elif sc1 < c0:
            d, length = (0, -1), c1 - c0 + 1
        else:
            d, length = (0, 1), c1 - c0 + 1
        shape = (sr1 - sr0 + 1, sc1 - sc0 + 1)
        center = (round((sr0 + sr1) / 2), round((sc0 + sc1) / 2))
        for step in range(1, length + 1):
            base_r = center[0] + d[0] * step
            base_c = center[1] + d[1] * step
            cells_to_draw = []
            if shape == (1, 2):
                cells_to_draw = [(base_r, base_c)]
            elif shape == (2, 1):
                off = step % 2
                cells_to_draw = [(base_r + (off if d[1] else 0), base_c + (off if d[0] else 0))]
            else:
                for k in range(step):
                    if d[0]:
                        cells_to_draw.append((base_r, base_c - k))
                        cells_to_draw.append((base_r, base_c + k))
                    else:
                        cells_to_draw.append((base_r - k, base_c))
                        cells_to_draw.append((base_r + k, base_c))
            for rr, cc in cells_to_draw:
                if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == BG:
                    out[rr, cc] = col
    return out


def generate_input() -> np.ndarray:
    h = w = random.choice([24, 26, 28])
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    rects = [(8, 8), (8, 16), (16, 8)]
    seed_shapes = [[(0, 0), (0, 1)], [(0, 0), (1, 0)], [(0, 0), (0, 1), (1, 0), (1, 1)]]
    for col, (r0, c0), shape in zip(colors, rects, random.sample(seed_shapes, 3)):
        rh, rw = random.randint(3, 5), random.randint(3, 5)
        grid[r0:r0 + rh, c0:c0 + rw] = col
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            sr, sc = r0 - max(dr for dr, _ in shape) - 1, c0 + 1
        elif side == "bottom":
            sr, sc = r0 + rh, c0 + 1
        elif side == "left":
            sr, sc = r0 + 1, c0 - max(dc for _, dc in shape) - 1
        else:
            sr, sc = r0 + 1, c0 + rw
        for dr, dc in shape:
            grid[sr + dr, sc + dc] = col
    return grid
