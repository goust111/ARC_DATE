import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _find_strips(grid):
    rows, _ = grid.shape
    candidates = [
        r
        for r in range(rows)
        if grid[r].size and np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != BACKGROUND
    ]
    for i, r1 in enumerate(candidates):
        for r2 in candidates[i + 1 :]:
            if r1 == r2 - r1 - 1 == rows - r2 - 1:
                return r1, r2, int(grid[r1, 0])
    gray_rows = [r for r in range(rows) if np.all(grid[r] == SEPARATOR)]
    if len(gray_rows) >= 2:
        return gray_rows[0], gray_rows[1], SEPARATOR
    raise ValueError("expected two full-width horizontal separator rows")


def _positions_by_color(band, separator):
    result = {}
    for color in np.unique(band):
        color = int(color)
        if color not in (BACKGROUND, separator):
            result[color] = np.argwhere(band == color)
    return result


def _ordered_common_colors(top, middle, bottom):
    common = set(top) & set(middle) & set(bottom)
    seen = set()
    ordered = []
    for value in top:
        color = int(value)
        if color in common and color not in seen:
            seen.add(color)
            ordered.append(color)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """Infer motion from two snapshot strips and draw reverse trails in the third strip.

    Input structure:
    The grid has a black background and two full-width gray horizontal separator rows,
    splitting it into three equal-height horizontal strips. Each strip contains one
    copy of every colored object; matching is unambiguous because each object color
    appears once per strip. Copies keep the same strip-relative rows and shape, and
    only their columns change.

    Rule:
    For each color, compare the object's leftmost column in the top strip with its
    leftmost column in the middle strip. This horizontal displacement D determines
    the bottom-strip trail: from every pixel of the bottom copy, paint |D| pixels in
    the opposite horizontal direction, using the same color. Paint only onto black
    background cells, leaving existing objects, separators, and the upper two strips
    unchanged.

    Key insight:
    The top and middle strips are time snapshots; their horizontal offset encodes
    the direction and length of the motion trail that must be drawn backward from
    the final bottom snapshot.
    """
    out = grid.copy()
    sep1, sep2, separator = _find_strips(grid)

    top = grid[:sep1]
    middle = grid[sep1 + 1 : sep2]
    bottom = grid[sep2 + 1 :]

    top_pos = _positions_by_color(top, separator)
    mid_pos = _positions_by_color(middle, separator)
    bot_pos = _positions_by_color(bottom, separator)

    bottom_row_offset = sep2 + 1
    width = grid.shape[1]

    for color in _ordered_common_colors(top.ravel(), mid_pos, bot_pos):
        d = int(mid_pos[color][:, 1].min() - top_pos[color][:, 1].min())
        if d == 0:
            continue
        step_dir = -1 if d > 0 else 1
        length = abs(d)
        for rr, cc in bot_pos[color]:
            global_r = bottom_row_offset + int(rr)
            c = int(cc)
            for step in range(1, length + 1):
                trail_c = c + step_dir * step
                if 0 <= trail_c < width and out[global_r, trail_c] == BACKGROUND:
                    out[global_r, trail_c] = color
    return out


def _random_shape(rng, max_height):
    templates = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 1), (1, 0), (2, 0)],
    ]
    choices = []
    for shape in templates:
        height = max(r for r, _ in shape) + 1
        if height <= max_height:
            choices.append(shape)
    shape = list(rng.choice(choices))
    if rng.random() < 0.5:
        shape = [(r, max(c for _, c in shape) - c) for r, c in shape]
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return sorted((r - min_r, c - min_c) for r, c in shape)


def _cells(shape, row, col):
    return {(row + dr, col + dc) for dr, dc in shape}


def _trail_cells(shape, row, col, displacement):
    direction = -1 if displacement > 0 else 1
    length = abs(displacement)
    cells = set()
    for dr, dc in shape:
        r = row + dr
        c = col + dc
        cells.add((r, c))
        for step in range(1, length + 1):
            cells.add((r, c + direction * step))
    return cells


def _shape_size(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _try_build(rng):
    band_h = rng.randint(5, 9)
    width = rng.randint(14, 30)
    obj_count = rng.randint(2, 5)
    colors = rng.sample(COLORS, obj_count)
    signs = [1, -1]
    while len(signs) < obj_count:
        signs.append(1 if rng.random() < 0.5 else -1)
    rng.shuffle(signs)

    grid = np.zeros((3 * band_h + 2, width), dtype=int)
    sep1 = band_h
    sep2 = 2 * band_h + 1
    grid[sep1, :] = SEPARATOR
    grid[sep2, :] = SEPARATOR

    top_taken = set()
    mid_taken = set()
    bot_taken = set()
    bot_trail_taken = set()
    objects = []

    for color, sign in zip(colors, signs):
        placed = False
        for _ in range(250):
            shape = _random_shape(rng, band_h)
            shape_h, shape_w = _shape_size(shape)
            max_shift = min(6, width - shape_w - 1)
            if max_shift < 1:
                continue
            displacement = sign * rng.randint(1, max_shift)
            shift = abs(displacement)

            top_lo = max(0, -displacement)
            top_hi = min(width - shape_w, width - shape_w - displacement)
            if top_lo > top_hi:
                continue
            if displacement > 0:
                bottom_lo = shift
                bottom_hi = width - shape_w
            else:
                bottom_lo = 0
                bottom_hi = width - shape_w - shift
            if bottom_lo > bottom_hi:
                continue

            row = rng.randint(0, band_h - shape_h)
            top_col = rng.randint(top_lo, top_hi)
            mid_col = top_col + displacement
            bottom_col = rng.randint(bottom_lo, bottom_hi)

            top_cells = _cells(shape, row, top_col)
            mid_cells = _cells(shape, row, mid_col)
            bottom_cells = _cells(shape, row, bottom_col)
            reserved_trail = _trail_cells(shape, row, bottom_col, displacement)

            if top_cells & top_taken or mid_cells & mid_taken:
                continue
            if bottom_cells & bot_taken or reserved_trail & bot_trail_taken:
                continue

            top_taken.update(top_cells)
            mid_taken.update(mid_cells)
            bot_taken.update(bottom_cells)
            bot_trail_taken.update(reserved_trail)
            objects.append((color, shape, row, top_col, mid_col, bottom_col))
            placed = True
            break
        if not placed:
            return None

    mid_offset = sep1 + 1
    bot_offset = sep2 + 1
    for color, shape, row, top_col, mid_col, bottom_col in objects:
        for dr, dc in shape:
            grid[row + dr, top_col + dc] = color
            grid[mid_offset + row + dr, mid_col + dc] = color
            grid[bot_offset + row + dr, bottom_col + dc] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(100):
        rng = random.Random()
        grid = _try_build(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
