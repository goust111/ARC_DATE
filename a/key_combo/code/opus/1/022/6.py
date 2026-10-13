import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    out = grid.copy()
    bar_cols = [int(c) for c in np.where(np.all(grid == 5, axis=0))[0]]
    if len(bar_cols) < 2:
        return out
    left, right = bar_cols[0], bar_cols[-1]
    glyph = [(int(r), int(c)) for r, c in np.argwhere(grid == 1)]
    if not glyph:
        return out
    min_col = min(c for _, c in glyph)
    anchor_row = min(r for r, c in glyph if c == min_col)
    offsets = [(r - anchor_row, c - min_col) for r, c in glyph]
    for r, c in glyph:
        out[r, c] = 0

    marker_rows = []
    for r in range(h):
        vals = [int(grid[r, left + 1]), int(grid[r, right - 1])]
        if vals[0] == vals[1] and vals[0] not in (0, 1, 5):
            marker_rows.append((r, vals[0]))

    fixed = out != 0
    glyphs = []
    for anchor, color in marker_rows:
        cells = []
        for dr, dc in offsets:
            rr, cc = anchor + dr, left + 1 + dc
            if 0 <= rr < h and left < cc < right:
                out[rr, cc] = color
                cells.append((rr, cc))
        glyphs.append((anchor, color, cells))

    occupied = fixed | (out != 0)
    proposals = []
    for anchor, color, cells in glyphs:
        by_col = {}
        for r, c in cells:
            by_col[c] = max(by_col.get(c, -1), r)
        for c, r0 in by_col.items():
            for r in range(r0 + 1, h):
                if fixed[r, c] or ((r, c) not in cells and out[r, c] not in (0, color)):
                    break
                proposals.append((anchor, r, c, color))
    for _, r, c, color in sorted(proposals):
        if out[r, c] == 0:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 22), random.randint(13, 16)
    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = 5
    grid[:, w - 1] = 5
    shapes = [
        [(0, 1), (1, 1), (2, 1), (2, 0), (2, 2), (3, 2), (4, 2), (4, 3)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (4, 3)],
        [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2)],
    ]
    glyph = random.choice(shapes)
    gr, gc = 1, random.randint(2, 4)
    for dr, dc in glyph:
        grid[gr + dr, gc + dc] = 1
    rows = sorted(random.sample(range(9, h - 5), random.randint(2, 3)))
    colors = random.sample([2, 3, 4], len(rows))
    for r, color in zip(rows, colors):
        grid[r, 1] = color
        grid[r, w - 2] = color
    return grid
