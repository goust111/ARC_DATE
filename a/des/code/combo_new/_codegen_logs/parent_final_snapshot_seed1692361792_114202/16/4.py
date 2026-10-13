import random
import numpy as np

BLACK = 0
GREY = 5


def detect_arrow(patch):
    colors = [v for v in np.unique(patch) if v not in (BLACK, GREY)]
    if not colors:
        return None
    color = colors[0]
    cells = np.argwhere(patch == color)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    sub = (patch[r0 : r1 + 1, c0 : c1 + 1] == color).astype(int)
    if sub.shape == (5, 3):
        rows = sub.sum(axis=1).tolist()
        if rows[:2] == [1, 1]:
            return color, "up", int(sub.sum())
        if rows[-2:] == [1, 1]:
            return color, "down", int(sub.sum())
    if sub.shape == (3, 5):
        cols = sub.sum(axis=0).tolist()
        if cols[:2] == [1, 1]:
            return color, "left", int(sub.sum())
        if cols[-2:] == [1, 1]:
            return color, "right", int(sub.sum())
    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    grey_rows = [-1] + [r for r in range(h) if np.all(grid[r] == GREY)] + [h]
    grey_cols = [-1] + [c for c in range(w) if np.all(grid[:, c] == GREY)] + [w]
    cells = {}
    claims = {}
    for i in range(len(grey_rows) - 1):
        for j in range(len(grey_cols) - 1):
            r0, r1 = grey_rows[i] + 1, grey_rows[i + 1] - 1
            c0, c1 = grey_cols[j] + 1, grey_cols[j + 1] - 1
            if r0 > r1 or c0 > c1:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            empty = np.all(patch == BLACK)
            arrow = detect_arrow(patch)
            cells[(i, j)] = (r0, r1, c0, c1, empty, arrow)
            if arrow is None:
                continue
            color, direction, area = arrow
            target = {"up": (i - 1, j), "down": (i + 1, j), "left": (i, j - 1), "right": (i, j + 1)}[direction]
            if target not in cells:
                claims.setdefault(target, []).append((area, {"down": 0, "right": 1, "left": 2, "up": 3}[direction], color))
            else:
                claims.setdefault(target, []).append((area, {"down": 0, "right": 1, "left": 2, "up": 3}[direction], color))
    out = grid.copy()
    for target, vals in claims.items():
        if target not in cells:
            continue
        r0, r1, c0, c1, empty, arrow = cells[target]
        if not empty or arrow is not None:
            continue
        area, rank, color = sorted(vals, key=lambda x: (-x[0], x[1]))[0]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def draw_arrow(patch, color, direction):
    h, w = patch.shape
    cr, cc = h // 2, w // 2
    if direction == "up":
        patch[cr - 1 : cr + 2, cc - 1 : cc + 2] = color
        patch[cr - 2, cc] = color
        patch[cr - 3, cc] = color
    elif direction == "down":
        patch[cr - 1 : cr + 2, cc - 1 : cc + 2] = color
        patch[cr + 2, cc] = color
        patch[cr + 3, cc] = color
    elif direction == "left":
        patch[cr - 1 : cr + 2, cc - 1 : cc + 2] = color
        patch[cr, cc - 2] = color
        patch[cr, cc - 3] = color
    else:
        patch[cr - 1 : cr + 2, cc - 1 : cc + 2] = color
        patch[cr, cc + 2] = color
        patch[cr, cc + 3] = color


def generate_input() -> np.ndarray:
    rows = [random.randint(7, 9) for _ in range(random.randint(2, 3))]
    cols = [random.randint(7, 9) for _ in range(random.randint(2, 3))]
    h = sum(rows) + len(rows) - 1
    w = sum(cols) + len(cols) - 1
    grid = np.zeros((h, w), dtype=int)
    rr = 0
    for i, rh in enumerate(rows):
        cc = 0
        for j, cw in enumerate(cols):
            patch = grid[rr : rr + rh, cc : cc + cw]
            if random.random() < 0.55:
                draw_arrow(patch, random.choice([1, 2, 3, 4, 6, 7, 8, 9]), random.choice(["up", "down", "left", "right"]))
            cc += cw + 1
        rr += rh + 1
    r = rows[0]
    for rh in rows[:-1]:
        grid[r, :] = GREY
        r += rh + 1
    c = cols[0]
    for cw in cols[:-1]:
        grid[:, c] = GREY
        c += cw + 1
    return grid
