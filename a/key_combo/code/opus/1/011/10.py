import random

import numpy as np


def _cell(grid, gr, gc, s):
    r, c = gr * (s + 1), gc * (s + 1)
    return grid[r:r + s, c:c + s], r, c


def _cup_bbox(cell):
    pts = np.argwhere(cell == 1)
    if len(pts) == 0:
        return None
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def _draw_cap_and_ray(out, r0, c0, bbox, color, direction):
    a, b, x, y = bbox
    if direction == "up":
        rr = r0 + a - 1
        for c in range(c0 + x, c0 + y + 1):
            if 0 <= rr < out.shape[0]:
                out[rr, c] = color
        r, c = rr - 1, c0 + (x + y) // 2
        dr, dc = -1, 0
    elif direction == "down":
        rr = r0 + b + 1
        for c in range(c0 + x, c0 + y + 1):
            if 0 <= rr < out.shape[0]:
                out[rr, c] = color
        r, c = rr + 1, c0 + (x + y) // 2
        dr, dc = 1, 0
    elif direction == "left":
        cc = c0 + x - 1
        for r in range(r0 + a, r0 + b + 1):
            if 0 <= cc < out.shape[1]:
                out[r, cc] = color
        r, c = r0 + (a + b) // 2, cc - 1
        dr, dc = 0, -1
    else:
        cc = c0 + y + 1
        for r in range(r0 + a, r0 + b + 1):
            if 0 <= cc < out.shape[1]:
                out[r, cc] = color
        r, c = r0 + (a + b) // 2, cc + 1
        dr, dc = 0, 1
    while 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] != 5:
        out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    s = (grid.shape[0] - 3) // 4
    cups = []
    template = None
    for gr in range(4):
        for gc in range(4):
            cell, r0, c0 = _cell(grid, gr, gc, s)
            bbox = _cup_bbox(cell)
            if bbox is None:
                continue
            vals, counts = np.unique(cell[np.isin(cell, [2, 3, 4])], return_counts=True)
            if len(vals) == 0:
                continue
            color = int(vals[np.argmax(counts)])
            info = (gr, gc, r0, c0, bbox, color)
            cups.append(info)
            if counts.max() > 1:
                template = info
    if template is None:
        return out
    tgr, tgc = template[0], template[1]
    for gr, gc, r0, c0, bbox, color in sorted(cups):
        if (gr, gc) == (tgr, tgc):
            continue
        a, b, x, y = bbox
        for r in range(r0 + a + 1, r0 + b):
            for c in range(c0 + x + 1, c0 + y):
                if out[r, c] != 1:
                    out[r, c] = color
        if abs(gr - tgr) >= abs(gc - tgc):
            direction = "up" if gr < tgr else "down"
        else:
            direction = "left" if gc < tgc else "right"
        _draw_cap_and_ray(out, r0, c0, bbox, color, direction)
    return out


def _draw_cup(cell, color, filled=False, cap=None):
    a, b, x, y = 2, 4, 2, 4
    for c in range(x, y + 1):
        cell[a, c] = 1
        cell[b, c] = 1
    for r in range(a, b + 1):
        cell[r, x] = 1
        cell[r, y] = 1
    cell[3, 3] = color
    if filled:
        cell[3, 3] = color
        if cap == "up":
            cell[1, 2:5] = color
        elif cap == "down":
            cell[5, 2:5] = color
        elif cap == "left":
            cell[2:5, 1] = color
        else:
            cell[2:5, 5] = color


def generate_input() -> np.ndarray:
    s = 6
    grid = np.zeros((4 * s + 3, 4 * s + 3), dtype=int)
    for k in range(1, 4):
        grid[k * s + (k - 1), :] = 5
        grid[:, k * s + (k - 1)] = 5
    template_pos = random.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
    positions = [template_pos]
    candidates = [(r, c) for r in range(4) for c in range(4) if (r, c) != template_pos]
    positions += random.sample(candidates, random.randint(3, 5))
    for pos in positions:
        cell, _, _ = _cell(grid, pos[0], pos[1], s)
        color = random.choice([2, 3, 4])
        if pos == template_pos:
            _draw_cup(cell, color, True, random.choice(["up", "down", "left", "right"]))
        else:
            _draw_cup(cell, color, False)
    return grid
