import numpy as np


BLACK = 0
GREY = 5


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangles(grid):
    rects = []
    for comp in _components(grid == GREY):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if len(comp) == 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4:
            rects.append((r0, r1, c0, c1))
    rects.sort()
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each room's interior dots with an exterior bar whose side and length are encoded by the dots.

    Input structure:
    - Black background with several hollow grey rectangles.
    - Each rectangle contains a small set of same-colored dots inside its border.
    - The dots are biased toward one side of the rectangle and their count is between one and four.

    Rule:
    - For each rectangle, infer the dominant side of the dot cluster from the dots' positions inside the box.
    - Remove the interior dots.
    - Draw a centered bar of the same color immediately outside that side; its length equals the number of dots.

    Key insight:
    - The dots carry two independent signals at once: count gives the bar length, and directional bias chooses which outer edge receives the bar.
    """
    out = grid.copy()
    for r0, r1, c0, c1 in _rectangles(grid):
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        coords = np.argwhere((inner != BLACK) & (inner != GREY))
        color = int(inner[tuple(coords[0])])
        count = len(coords)
        center_r = (inner.shape[0] - 1) / 2.0
        center_c = (inner.shape[1] - 1) / 2.0
        mean_r, mean_c = coords.mean(axis=0)
        if abs(mean_c - center_c) >= abs(mean_r - center_r):
            side = "left" if mean_c < center_c else "right"
        else:
            side = "top" if mean_r < center_r else "bottom"
        out[r0 + 1 : r1, c0 + 1 : c1][(inner != BLACK) & (inner != GREY)] = BLACK
        if side in ("top", "bottom"):
            row = r0 - 1 if side == "top" else r1 + 1
            start = c0 + (c1 - c0 + 1 - count) // 2
            out[row, start : start + count] = color
        else:
            col = c0 - 1 if side == "left" else c1 + 1
            start = r0 + (r1 - r0 + 1 - count) // 2
            out[start : start + count, col] = color
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    height = int(rng.integers(18, 27))
    width = int(rng.integers(18, 27))
    grid = np.zeros((height, width), dtype=int)
    rect_count = int(rng.integers(2, 5))
    used = np.zeros_like(grid, dtype=bool)
    for _ in range(rect_count):
        for _attempt in range(200):
            h = int(rng.integers(5, 8))
            w = int(rng.integers(5, 8))
            r0 = int(rng.integers(2, height - h - 2))
            c0 = int(rng.integers(2, width - w - 2))
            if used[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue
            color = int(rng.choice([1, 2, 3, 4, 6, 7, 8, 9]))
            grid[r0 : r0 + h, c0 : c0 + w] = GREY
            grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = 0
            side = rng.choice(["top", "bottom", "left", "right"])
            count = int(rng.integers(1, 5))
            if side == "top":
                row_choices = list(range(r0 + 1, r0 + 1 + max(1, (h - 2) // 2)))
                col_choices = list(range(c0 + 1, c0 + w - 1))
            elif side == "bottom":
                row_choices = list(range(r0 + h - 1 - max(1, (h - 2) // 2), r0 + h - 1))
                col_choices = list(range(c0 + 1, c0 + w - 1))
            elif side == "left":
                row_choices = list(range(r0 + 1, r0 + h - 1))
                col_choices = list(range(c0 + 1, c0 + 1 + max(1, (w - 2) // 2)))
            else:
                row_choices = list(range(r0 + 1, r0 + h - 1))
                col_choices = list(range(c0 + w - 1 - max(1, (w - 2) // 2), c0 + w - 1))
            cells = [(rr, cc) for rr in row_choices for cc in col_choices]
            count = min(count, len(cells))
            picks = rng.choice(len(cells), size=count, replace=False)
            points = [cells[int(i)] for i in np.atleast_1d(picks)]
            for rr, cc in points:
                grid[rr, cc] = color
            used[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
            break
    return grid
