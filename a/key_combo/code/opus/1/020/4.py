import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _find_selector(grid):
    for comp in _components(grid, 9):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        if max(rs) - min(rs) == 6 and max(cs) - min(cs) == 6:
            return min(rs), min(cs)
    raise ValueError("selector block not found")


def _sprite_mask(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    mask = np.zeros((3, 3), dtype=bool)
    for r, c in cells:
        rr, cc = r - min_r, c - min_c
        if 0 <= rr < 3 and 0 <= cc < 3:
            mask[rr, cc] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Flip far-side sprites across the gray axis, then assemble them by selector markers.

    Input structure: A complete gray row or column is the axis, four colored
    3x3-bounded sprites sit at distinct distances from it, and a 7x7 maroon
    selector block names the destination quadrant by sprite color.
    Rule: The two sprites closest to the axis keep orientation. The two farther
    sprites are flipped vertically for a horizontal axis or horizontally for a
    vertical axis. The selector's four marker colors choose which processed
    sprite is copied into each 3x3 quadrant of the 7x7 output.
    Key insight: The axis is a classifier for sprite orientation, not a source
    for making mirror copies in place.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    axis_rows = [r for r in range(h) if np.all(grid[r, :] == 5)]
    axis_cols = [c for c in range(w) if np.all(grid[:, c] == 5)]
    horizontal = bool(axis_rows)
    axis = axis_rows[0] if horizontal else axis_cols[0]
    sr, sc = _find_selector(grid)
    selector_centers = {
        (0, 0): (sr + 1, sc + 1),
        (0, 1): (sr + 1, sc + 5),
        (1, 0): (sr + 5, sc + 1),
        (1, 1): (sr + 5, sc + 5),
    }
    marker_colors = {quad: int(grid[pos]) for quad, pos in selector_centers.items()}

    sprites = {}
    distances = []
    selector_area = {(r, c) for r in range(sr, sr + 7) for c in range(sc, sc + 7)}
    for color in set(marker_colors.values()):
        for comp in _components(grid, color):
            if len(comp) <= 1 or set(comp) & selector_area:
                continue
            mask = _sprite_mask(comp)
            min_r, max_r = min(r for r, _ in comp), max(r for r, _ in comp)
            min_c, max_c = min(c for _, c in comp), max(c for _, c in comp)
            center_r = (min_r + max_r) / 2
            center_c = (min_c + max_c) / 2
            dist = abs(center_r - axis) if horizontal else abs(center_c - axis)
            sprites[color] = mask
            distances.append((dist, color))
    far_colors = {color for _, color in sorted(distances)[2:]}
    processed = {}
    for color, mask in sprites.items():
        if color in far_colors:
            processed[color] = np.flipud(mask) if horizontal else np.fliplr(mask)
        else:
            processed[color] = mask

    out = np.zeros((7, 7), dtype=int)
    out[3, :] = 9
    out[:, 3] = 9
    origins = {(0, 0): (0, 0), (0, 1): (0, 4), (1, 0): (4, 0), (1, 1): (4, 4)}
    for quad, color in marker_colors.items():
        mask = processed.get(color)
        if mask is None:
            continue
        rr, cc = origins[quad]
        out[rr:rr + 3, cc:cc + 3][mask] = color
    return out


SHAPES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
]


def _place_mask(grid, top, left, mask, color):
    for r in range(3):
        for c in range(3):
            if mask[r, c]:
                grid[top + r, left + c] = color


def _draw_selector(grid, top, left, colors):
    grid[top:top + 7, left:left + 7] = 9
    for r, c in [(1, 1), (1, 5), (5, 1), (5, 5)]:
        grid[top + r, left + c] = colors.pop()


def generate_input() -> np.ndarray:
    n = 21
    grid = np.zeros((n, n), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8], 4)
    horizontal = random.choice([True, False])
    if horizontal:
        grid[10, :] = 5
        positions = [(7, 1), (5, 5), (12, 1), (15, 5)]
        _draw_selector(grid, 0, 14, colors.copy())
    else:
        grid[:, 10] = 5
        positions = [(1, 7), (5, 5), (1, 12), (5, 15)]
        _draw_selector(grid, 14, 0, colors.copy())
    for (top, left), color, mask in zip(positions, colors, SHAPES):
        _place_mask(grid, top, left, mask, color)
    return grid

