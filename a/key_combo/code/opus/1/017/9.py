import random

import numpy as np


def _tile_bounds():
    starts = [0, 4, 8]
    return [(r, c) for r in starts for c in starts]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    starts = [0, 4, 8]
    marker_tile = None
    marker_local = None
    for ti, r0 in enumerate(starts):
        for tj, c0 in enumerate(starts):
            tile = grid[r0 : r0 + 3, c0 : c0 + 3]
            ys = np.argwhere(tile == 4)
            if len(ys):
                lr, lc = map(int, ys[0])
                marker_tile = (ti, tj)
                marker_local = (lr, lc)
    if marker_tile is None:
        return out
    tr, tc = marker_local
    tr0, tc0 = starts[tr], starts[tc]
    template = grid[tr0 : tr0 + 3, tc0 : tc0 + 3].copy()
    mask = template != 0
    target = None
    for ti, r0 in enumerate(starts):
        for tj, c0 in enumerate(starts):
            if (ti, tj) in (marker_tile, (tr, tc)):
                continue
            tile = grid[r0 : r0 + 3, c0 : c0 + 3]
            if np.array_equal(tile != 0, mask):
                target = (r0, c0)
                break
        if target:
            break
    if target:
        out[target[0] : target[0] + 3, target[1] : target[1] + 3] = template
    mr0, mc0 = starts[marker_tile[0]], starts[marker_tile[1]]
    out[mr0 : mr0 + 3, mc0 : mc0 + 3] = 0
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((11, 11), dtype=int)
    grid[3, :] = 5
    grid[7, :] = 5
    grid[:, 3] = 5
    grid[:, 7] = 5
    starts = [0, 4, 8]
    masks = [
        [(0, 0), (0, 1), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 2)],
    ]
    marker_macro = random.choice([(0, 1), (1, 0), (1, 2), (2, 1)])
    template_macro = random.choice([p for p in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)] if p != marker_macro])
    marker_local = template_macro
    target_macro = random.choice([p for p in [(i, j) for i in range(3) for j in range(3)] if p not in (marker_macro, template_macro)])
    mask = random.choice(masks)
    tcolor, ocolor = random.sample([1, 2, 3, 6, 7, 8, 9], 2)
    for macro, color in [(template_macro, tcolor), (target_macro, ocolor)]:
        r0, c0 = starts[macro[0]], starts[macro[1]]
        for r, c in mask:
            grid[r0 + r, c0 + c] = color
    mr0, mc0 = starts[marker_macro[0]], starts[marker_macro[1]]
    grid[mr0 + marker_local[0], mc0 + marker_local[1]] = 4
    for macro in [(0, 0), (0, 2), (2, 0), (2, 2), (1, 1)]:
        if macro in (marker_macro, template_macro, target_macro):
            continue
        r0, c0 = starts[macro[0]], starts[macro[1]]
        grid[r0 + 1, c0] = random.choice([1, 2, 3, 6, 7, 8, 9])
    return grid

