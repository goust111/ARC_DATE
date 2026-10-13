import random
import numpy as np


def _panel_bounds(grid):
    frame = 5
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == frame)]
    return [(cols[i], cols[i + 1]) for i in range(len(cols) - 1)]


def _sprite(panel):
    mask = (panel != 0) & (panel != 4) & (panel != 5)
    pts = [(int(r), int(c), int(panel[r, c])) for r, c in np.argwhere(mask)]
    min_r = min(p[0] for p in pts)
    min_c = min(p[1] for p in pts)
    shape = [(r - min_r, c - min_c) for r, c, _ in pts]
    color = pts[0][2]
    return min_r, min_c, shape, color


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h = grid.shape[0]
    bounds = _panel_bounds(grid)
    panels = [grid[:, left:right + 1] for left, right in bounds[:3]]
    int_h = h - 2
    int_w = panels[0].shape[1] - 2
    sprites = [_sprite(panel[1:-1, 1:-1]) for panel in panels]
    delta_r = sprites[1][0] - sprites[0][0]
    delta_c = sprites[1][1] - sprites[0][1]
    next_r = sprites[2][0] + delta_r
    next_c = sprites[2][1] + delta_c
    shape = sprites[2][2]
    sprite_color = sprites[2][3]

    stripe_cols = []
    for c in range(int_w):
        values = []
        for panel in panels:
            values.extend(panel[1:-1, c + 1].tolist())
        if values.count(4) >= max(1, values.count(0)):
            stripe_cols.append(c)

    out = np.zeros((h, int_w + 2), dtype=grid.dtype)
    out[0, :] = 5
    out[-1, :] = 5
    out[:, 0] = 5
    out[:, -1] = 5
    for c in stripe_cols:
        out[1:-1, c + 1] = 4

    sprite_cells = {(next_r + dr, next_c + dc) for dr, dc in shape}
    collision = all(c in stripe_cols for _, c in sprite_cells)
    if collision:
        for c in stripe_cols:
            if any(sc == c for _, sc in sprite_cells):
                for r in range(int_h):
                    if (r, c) not in sprite_cells and out[r + 1, c + 1] == 4:
                        out[r + 1, c + 1] = sprite_color
        draw_color = 4
    else:
        draw_color = sprite_color
    for r, c in sprite_cells:
        out[r + 1, c + 1] = draw_color
    return out


def _draw_panel(grid, left, int_h, int_w, stripe_cols, sprite_cells, sprite_color):
    right = left + int_w + 1
    grid[0, left:right + 1] = 5
    grid[int_h + 1, left:right + 1] = 5
    grid[:, left] = 5
    grid[:, right] = 5
    for c in stripe_cols:
        grid[1:int_h + 1, left + 1 + c] = 4
    for r, c in sprite_cells:
        grid[1 + r, left + 1 + c] = sprite_color


def generate_input() -> np.ndarray:
    rng = random.Random()
    int_h = rng.choice([5, 6, 7])
    int_w = rng.choice([4, 5, 6])
    h = int_h + 2
    w = 3 * (int_w + 1) + 1
    grid = np.zeros((h, w), dtype=int)
    parity = rng.randint(0, 1)
    stripe_cols = [c for c in range(int_w) if c % 2 == parity]
    if not stripe_cols:
        stripe_cols = [0]
    sprite_color = rng.choice([1, 2, 3, 6, 8, 9])
    shape = rng.choice([
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
    ])
    max_shape_r = max(r for r, _ in shape)
    max_shape_c = max(c for _, c in shape)
    possible_deltas = []
    for delta in (-1, 0, 1):
        low = max(0, -3 * delta)
        high = int_h - 1 - max_shape_r - max(0, 3 * delta)
        if low <= high:
            possible_deltas.append(delta)
    delta = rng.choice(possible_deltas)
    low = max(0, -3 * delta)
    high = int_h - 1 - max_shape_r - max(0, 3 * delta)
    top0 = rng.randint(low, high)
    left0 = rng.randint(0, int_w - 1 - max_shape_c)
    for i in range(3):
        top = top0 + i * delta
        cells = [(top + dr, left0 + dc) for dr, dc in shape]
        _draw_panel(grid, i * (int_w + 1), int_h, int_w, stripe_cols, cells, sprite_color)
    return grid
