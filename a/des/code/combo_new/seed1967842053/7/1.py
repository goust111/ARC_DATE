import random
from typing import Dict, List, Tuple

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_frame(grid: np.ndarray, bg: int) -> Tuple[int, int, int, int, int]:
    best = None
    best_area = -1
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        pos = np.argwhere(grid == color)
        if pos.size == 0:
            continue
        r0, c0 = pos.min(axis=0)
        r1, c1 = pos.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 5 or w < 5:
            continue
        top = grid[r0, c0 : c1 + 1]
        bottom = grid[r1, c0 : c1 + 1]
        left = grid[r0 : r1 + 1, c0]
        right = grid[r0 : r1 + 1, c1]
        if not (
            np.all(top == color)
            and np.all(bottom == color)
            and np.all(left == color)
            and np.all(right == color)
        ):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(interior == color):
            continue
        area = h * w
        if area > best_area:
            best = (int(r0), int(r1), int(c0), int(c1), color)
            best_area = area
    if best is None:
        raise ValueError('no rectangular frame found')
    return best


def _marker_order(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> List[int]:
    region = grid[top + 1 : bottom, left + 1 : right]
    mask = (region != bg) & (region != frame_color)
    local = np.argwhere(mask)
    if local.size == 0:
        return []
    coords = local + np.array([top + 1, left + 1])
    rows = coords[:, 0]
    cols = coords[:, 1]

    if len(np.unique(cols)) == 1:
        near_top = int(rows.min() - (top + 1)) <= int((bottom - 1) - rows.max())
        indices = np.argsort(rows if near_top else -rows)
    elif len(np.unique(rows)) == 1:
        near_left = int(cols.min() - (left + 1)) <= int((right - 1) - cols.max())
        indices = np.argsort(cols if near_left else -cols)
    else:
        distances = {
            'top': int(rows.min() - (top + 1)),
            'bottom': int((bottom - 1) - rows.max()),
            'left': int(cols.min() - (left + 1)),
            'right': int((right - 1) - cols.max()),
        }
        side = min(distances, key=distances.get)
        if side == 'top':
            indices = np.lexsort((cols, rows))
        elif side == 'bottom':
            indices = np.lexsort((cols, -rows))
        elif side == 'left':
            indices = np.lexsort((rows, cols))
        else:
            indices = np.lexsort((rows, -cols))

    order = []
    seen = set()
    for idx in indices:
        color = int(grid[int(coords[idx, 0]), int(coords[idx, 1])])
        if color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _outside_piece_sizes(
    grid: np.ndarray,
    colors: List[int],
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> Dict[int, Tuple[int, int]]:
    rows, cols = grid.shape
    outside = np.ones((rows, cols), dtype=bool)
    outside[top : bottom + 1, left : right + 1] = False
    sizes = {}
    for color in colors:
        coords = np.argwhere((grid == color) & outside)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        sizes[int(color)] = (int(r1 - r0 + 1), int(c1 - c0 + 1))
    return sizes


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild scattered colored rectangles inside the frame using the marker depth order.

    Input structure:
    A dominant background surrounds one single-color rectangular frame. The frame interior is background except for a straight inward run of one-pixel color markers on one side, and matching solid rectangular color blocks are scattered outside the frame.

    Rule:
    Erase the outside blocks and marker pixels, keep the frame border, translate every matching outside rectangle so that its center matches the interior center, and draw the rectangles from the marker nearest the border to the marker nearest the center.

    Key insight:
    The side markers encode depth, not position: their visible inward order is the bottom-to-top painting order for the centered rectangles.
    """
    grid = np.asarray(grid)
    bg = _background_color(grid)
    top, bottom, left, right, frame_color = _find_frame(grid, bg)
    order = _marker_order(grid, bg, frame_color, top, bottom, left, right)
    sizes = _outside_piece_sizes(grid, order, top, bottom, left, right)

    out = np.full_like(grid, bg)
    out[top, left : right + 1] = frame_color
    out[bottom, left : right + 1] = frame_color
    out[top : bottom + 1, left] = frame_color
    out[top : bottom + 1, right] = frame_color

    ir0, ir1 = top + 1, bottom - 1
    ic0, ic1 = left + 1, right - 1
    for color in order:
        if color not in sizes:
            continue
        h, w = sizes[color]
        r0 = (ir0 + ir1 - h + 1) // 2
        c0 = (ic0 + ic1 - w + 1) // 2
        out[r0 : r0 + h, c0 : c0 + w] = color
    return out


def _choices_with_parity(limit: int, parity: int) -> List[int]:
    upper = max(2, min(7, limit))
    vals = [x for x in range(2, upper + 1) if x % 2 == parity]
    if vals:
        return vals
    return [1]


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    rows, cols = occupied.shape
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(rows, r + h + 1)
    c1 = min(cols, c + w + 1)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _place_piece(occupied: np.ndarray, rng: random.Random, h: int, w: int) -> Tuple[int, int]:
    rows, cols = occupied.shape
    for _ in range(800):
        r = rng.randrange(1, rows - h)
        c = rng.randrange(1, cols - w)
        if _can_place(occupied, r, c, h, w):
            occupied[max(0, r - 1) : min(rows, r + h + 1), max(0, c - 1) : min(cols, c + w + 1)] = True
            return r, c
    raise RuntimeError('could not place piece')


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        rows = rng.randint(24, 30)
        cols = rng.randint(24, 30)
        piece_count = rng.randint(3, 6)
        side = rng.randrange(4)

        interior_h = rng.randint(max(8, piece_count + 3), min(14, rows - 8))
        interior_w = rng.randint(max(8, piece_count + 3), min(14, cols - 8))
        frame_h = interior_h + 2
        frame_w = interior_w + 2
        top = rng.randint(3, rows - frame_h - 3)
        left = rng.randint(3, cols - frame_w - 3)
        bottom = top + frame_h - 1
        right = left + frame_w - 1

        palette = list(range(10))
        rng.shuffle(palette)
        bg = palette[0]
        frame_color = palette[1]
        piece_colors = palette[2 : 2 + piece_count]

        grid = np.full((rows, cols), bg, dtype=int)
        grid[top, left : right + 1] = frame_color
        grid[bottom, left : right + 1] = frame_color
        grid[top : bottom + 1, left] = frame_color
        grid[top : bottom + 1, right] = frame_color

        if side == 0:
            marker_col = rng.randint(left + 2, right - 2)
            marker_cells = [(top + 1 + i, marker_col) for i in range(piece_count)]
        elif side == 1:
            marker_col = rng.randint(left + 2, right - 2)
            marker_cells = [(bottom - 1 - i, marker_col) for i in range(piece_count)]
        elif side == 2:
            marker_row = rng.randint(top + 2, bottom - 2)
            marker_cells = [(marker_row, left + 1 + i) for i in range(piece_count)]
        else:
            marker_row = rng.randint(top + 2, bottom - 2)
            marker_cells = [(marker_row, right - 1 - i) for i in range(piece_count)]

        h_options = _choices_with_parity(interior_h, interior_h % 2)
        w_options = _choices_with_parity(interior_w, interior_w % 2)
        sizes = [(rng.choice(h_options), rng.choice(w_options)) for _ in piece_colors]
        if len(set(sizes)) == 1 and len(h_options) + len(w_options) > 2:
            sizes[-1] = (h_options[-1], w_options[0])

        occupied = np.zeros((rows, cols), dtype=bool)
        occupied[max(0, top - 1) : min(rows, bottom + 2), max(0, left - 1) : min(cols, right + 2)] = True
        try:
            placements = [_place_piece(occupied, rng, h, w) for h, w in sizes]
        except RuntimeError:
            continue

        for color, (r, c), (h, w) in zip(piece_colors, placements, sizes):
            grid[r : r + h, c : c + w] = color
        for color, (r, c) in zip(piece_colors, marker_cells):
            grid[r, c] = color
        return grid

    raise RuntimeError('failed to generate a valid puzzle input')
