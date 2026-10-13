import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
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
            out.append(cells)
    return out


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each rectangle toward the target encoded by its arrow color and side.

    Input structure:
    - A vertical grey separator splits a left palette from a right operation area.
    - The palette is a single colored pixel column; each color's row is its target row.
    - The right side contains solid colored rectangles, each with one adjacent colored
      arrow pixel whose side indicates movement direction and whose color appears in
      the palette.

    Rule:
    - Read palette color -> row mappings from the left side.
    - For each rectangle, find its adjacent arrow cell and infer direction from the
      arrow's side relative to the rectangle.
    - Move the rectangle: up targets its top edge to the palette row; down targets its
      bottom edge to the palette row; left moves until its left edge touches the
      separator; right moves until its right edge reaches the canvas boundary.
    - Remove arrow pixels and keep the separator and palette unchanged.

    Key insight:
    - The arrow color names the target row in the palette, while the arrow position
      names the movement direction; both signals are required to place each object.
    """
    h, w = grid.shape
    sep = next(c for c in range(w) if np.all(grid[:, c] == GREY))
    palette = {int(grid[r, c]): r for r in range(h) for c in range(sep) if grid[r, c] not in (BLACK, GREY)}
    out = grid.copy()
    right = grid[:, sep + 1 :]
    used = np.zeros_like(right, dtype=bool)
    for color in [int(v) for v in np.unique(right) if v not in (BLACK, GREY) and v not in palette]:
        for comp in _components(right == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
            arrow = None
            for rr in range(max(0, r0 - 1), min(right.shape[0], r1 + 2)):
                for cc in range(max(0, c0 - 1), min(right.shape[1], c1 + 2)):
                    val = int(right[rr, cc])
                    if val in palette and not (r0 <= rr <= r1 and c0 <= cc <= c1):
                        if (r0 <= rr <= r1 and abs(cc - c0) == 1) or (c0 <= cc <= c1 and abs(rr - r0) == 1) or abs(rr - r1) == 1 or abs(cc - c1) == 1:
                            arrow = (rr, cc, val)
                            break
                if arrow:
                    break
            if arrow is None:
                continue
            ar, ac, acolor = arrow
            out[ar, sep + 1 + ac] = BLACK
            out[r0 : r1 + 1, sep + 1 + c0 : sep + 1 + c1 + 1] = BLACK
            if ar == r0 - 1:
                nr0 = palette[acolor]
                nc0 = c0
            elif ar == r1 + 1:
                nr0 = palette[acolor] - (r1 - r0)
                nc0 = c0
            elif ac == c0 - 1:
                nr0 = r0
                nc0 = 1
            else:
                nr0 = r0
                nc0 = right.shape[1] - (c1 - c0 + 1)
            out[nr0 : nr0 + (r1 - r0 + 1), sep + 1 + nc0 : sep + 1 + nc0 + (c1 - c0 + 1)] = color
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(16, 24)
        w = random.randint(18, 28)
        sep = random.randint(4, 6)
        grid = np.zeros((h, w), dtype=int)
        grid[:, sep] = GREY
        palette_colors = random.sample(COLORS, random.randint(3, 5))
        rows = random.sample(range(1, h - 1), len(palette_colors))
        for color, r in zip(palette_colors, rows):
            grid[r, random.randint(0, sep - 1)] = color

        placed = []
        directions = ["up", "down", "left", "right"]
        for arrow_color in palette_colors:
            oh = random.randint(2, 3)
            ow = random.randint(2, 4)
            color = random.choice([c for c in COLORS if c != arrow_color])
            direction = random.choice(directions)
            if direction == "left":
                tr0, tc0 = random.randint(1, h - oh - 1), sep + 1
            elif direction == "right":
                tr0, tc0 = random.randint(1, h - oh - 1), w - ow
            elif direction == "up":
                tr0, tc0 = rows[palette_colors.index(arrow_color)], random.randint(sep + 2, w - ow - 2)
            else:
                tr0, tc0 = rows[palette_colors.index(arrow_color)] - oh + 1, random.randint(sep + 2, w - ow - 2)
            sr0, sc0 = tr0, tc0
            if direction == "left":
                sc0 = random.randint(sep + 3, w - ow - 2)
            elif direction == "right":
                sc0 = random.randint(sep + 1, w - ow - 4)
            elif direction == "up":
                lo, hi = tr0 + 2, h - oh - 1
                if lo > hi:
                    continue
                sr0 = random.randint(lo, hi)
            else:
                hi = tr0 - 2
                if hi < 1:
                    continue
                sr0 = random.randint(1, hi)
            if any(not (sr0 + oh - 1 < pr0 or pr1 < sr0 or sc0 + ow - 1 < pc0 or pc1 < sc0) for pr0, pc0, pr1, pc1 in placed):
                continue
            grid[sr0 : sr0 + oh, sc0 : sc0 + ow] = color
            if direction == "up":
                grid[sr0 - 1, sc0 + ow // 2] = arrow_color
            elif direction == "down":
                grid[sr0 + oh, sc0 + ow // 2] = arrow_color
            elif direction == "left":
                grid[sr0 + oh // 2, sc0 - 1] = arrow_color
            else:
                grid[sr0 + oh // 2, sc0 + ow] = arrow_color
            placed.append((sr0, sc0, sr0 + oh - 1, sc0 + ow - 1))
        if np.count_nonzero(grid[:, sep + 1 :]) > 0:
            return grid
