from __future__ import annotations

import random

import numpy as np


ORANGE = 7
RED = 2
MAGENTA = 6
PALETTE = [0, 1, 3, 4, 5, 8, 9]
GEN_RETRIES = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the control column and cyclically roll the remaining canvas downward.

    Input structure
    - Orange background.
    - The leftmost column is a control column: exactly one red pixel and all
      other cells magenta.
    - The remaining columns form the main canvas.

    Rule
    - Read the row index r of the red control pixel.
    - Delete the control column.
    - Roll the main canvas downward by r rows, wrapping around cyclically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 1 and w >= 2, "grid must include a control column and a canvas"

    control = g[:, 0]
    red_rows = np.flatnonzero(control == RED)
    assert len(red_rows) == 1, "control column must contain exactly one red pixel"
    assert np.all((control == MAGENTA) | (control == RED)), "control column must be magenta except for one red pixel"

    r = int(red_rows[0])
    canvas = g[:, 1:]
    return np.roll(canvas, shift=r, axis=0)


def _paint_rectangle(canvas: np.ndarray, occ: np.ndarray, color: int) -> bool:
    h, w = canvas.shape
    rect_h = random.randint(1, max(1, h // 2))
    rect_w = random.randint(1, max(1, w // 2))
    cells = [(r, c) for r in range(h - rect_h + 1) for c in range(w - rect_w + 1)]
    random.shuffle(cells)
    for r0, c0 in cells:
        if occ[r0 : r0 + rect_h, c0 : c0 + rect_w].any():
            continue
        canvas[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        occ[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        return True
    return False


def _paint_line(canvas: np.ndarray, occ: np.ndarray, color: int) -> bool:
    h, w = canvas.shape
    vertical = random.choice([False, True])
    if vertical:
        length = random.randint(2, h)
        cells = [(r, c) for r in range(h - length + 1) for c in range(w)]
        random.shuffle(cells)
        for r0, c0 in cells:
            if occ[r0 : r0 + length, c0].any():
                continue
            canvas[r0 : r0 + length, c0] = color
            occ[r0 : r0 + length, c0] = True
            return True
    else:
        length = random.randint(2, w)
        cells = [(r, c) for r in range(h) for c in range(w - length + 1)]
        random.shuffle(cells)
        for r0, c0 in cells:
            if occ[r0, c0 : c0 + length].any():
                continue
            canvas[r0, c0 : c0 + length] = color
            occ[r0, c0 : c0 + length] = True
            return True
    return False


def _paint_l(canvas: np.ndarray, occ: np.ndarray, color: int) -> bool:
    h, w = canvas.shape
    arm_h = random.randint(2, max(2, h // 2 + 1))
    arm_w = random.randint(2, max(2, w // 2 + 1))
    cells = [(r, c) for r in range(h - arm_h + 1) for c in range(w - arm_w + 1)]
    random.shuffle(cells)
    for r0, c0 in cells:
        orientation = random.choice([0, 1, 2, 3])
        mask = np.zeros((arm_h, arm_w), dtype=bool)
        if orientation == 0:
            mask[:, 0] = True
            mask[-1, :] = True
        elif orientation == 1:
            mask[:, -1] = True
            mask[-1, :] = True
        elif orientation == 2:
            mask[:, 0] = True
            mask[0, :] = True
        else:
            mask[:, -1] = True
            mask[0, :] = True
        region = occ[r0 : r0 + arm_h, c0 : c0 + arm_w]
        if np.any(region & mask):
            continue
        patch = canvas[r0 : r0 + arm_h, c0 : c0 + arm_w]
        patch[mask] = color
        region[mask] = True
        return True
    return False


def _build_canvas(h: int, w: int) -> np.ndarray | None:
    canvas = np.full((h, w), ORANGE, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    n_objects = random.randint(2, 6)
    painters = [_paint_rectangle, _paint_line, _paint_l]
    colors = random.sample(PALETTE, k=min(n_objects, len(PALETTE)))
    while len(colors) < n_objects:
        colors.append(random.choice(PALETTE))
    random.shuffle(colors)

    painted = 0
    for color in colors:
        order = painters[:]
        random.shuffle(order)
        for painter in order:
            if painter(canvas, occ, color):
                painted += 1
                break

    if painted < 2 or not occ.any():
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(5, 15)
    w = random.randint(5, 15)
    roll = random.randrange(h)

    canvas = _build_canvas(h, w)
    if canvas is None:
        return None
    if roll != 0 and np.array_equal(np.roll(canvas, shift=roll, axis=0), canvas):
        return None

    grid = np.full((h, w + 1), ORANGE, dtype=int)
    grid[:, 0] = MAGENTA
    grid[roll, 0] = RED
    grid[:, 1:] = canvas

    out = np.roll(canvas, shift=roll, axis=0)
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
