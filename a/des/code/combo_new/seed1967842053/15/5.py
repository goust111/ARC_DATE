import random
import numpy as np


RED = 2
GREY = 5
BACKGROUND = 0
PATTERN_COLORS = [1, 3, 4, 6, 7, 8, 9]


def _background_color(grid):
    vals, counts = np.unique(grid, return_counts=True)
    best_color = BACKGROUND
    best_count = -1
    for color, count in zip(vals, counts):
        if int(color) in (RED, GREY):
            continue
        if int(count) > best_count:
            best_color = int(color)
            best_count = int(count)
    return best_color


def _segments(size, line_indices):
    lines = sorted(int(x) for x in line_indices)
    cuts = [-1] + lines + [size]
    parts = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a > 1:
            parts.append((a + 1, b))
    return parts


def _arrow_direction(cell):
    pts = {tuple(p) for p in np.argwhere(cell == RED)}
    if len(pts) != 3:
        return None
    patterns = {
        "right": [
            [(0, 0), (0, -1), (-1, -1)],
            [(0, 0), (0, -1), (1, -1)],
        ],
        "left": [
            [(0, 0), (0, 1), (-1, 1)],
            [(0, 0), (0, 1), (1, 1)],
        ],
        "down": [
            [(0, 0), (-1, 0), (-1, -1)],
            [(0, 0), (-1, 0), (-1, 1)],
        ],
        "up": [
            [(0, 0), (1, 0), (1, -1)],
            [(0, 0), (1, 0), (1, 1)],
        ],
    }
    for tip_r, tip_c in pts:
        for direction, variants in patterns.items():
            for variant in variants:
                made = {(tip_r + dr, tip_c + dc) for dr, dc in variant}
                if made == pts:
                    return direction
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Cycle each table row's region patterns in the direction shown by red arrows.

    Input structure:
    A black background grid is divided by one-pixel grey horizontal and vertical
    lines into equal rectangular subregions. Each subregion contains a colored
    pattern using neither red nor grey, plus one three-pixel red arrow marker.

    Rule:
    For each row of subregions, red arrows pointing right mean the row's pattern
    contents move one subregion to the right cyclically; arrows pointing left
    mean the contents move one subregion to the left cyclically. Only non-red,
    non-grey, non-background pattern pixels move. Grey divider lines remain
    fixed, and all red arrow pixels are erased to background in the output.

    Key insight:
    The three red pixels encode an arrow direction, and that direction controls
    a row-wise cyclic shift of whole subregion patterns rather than movement of
    individual pixels inside a subregion.
    """
    grid = np.asarray(grid)
    bg = _background_color(grid)
    h, w = grid.shape

    grey_rows = np.where(np.all(grid == GREY, axis=1))[0]
    grey_cols = np.where(np.all(grid == GREY, axis=0))[0]
    row_parts = _segments(h, grey_rows)
    col_parts = _segments(w, grey_cols)

    out = np.array(grid, copy=True)
    for r0, r1 in row_parts:
        for c0, c1 in col_parts:
            out[r0:r1, c0:c1] = bg

    for r0, r1 in row_parts:
        directions = []
        for c0, c1 in col_parts:
            direction = _arrow_direction(grid[r0:r1, c0:c1])
            if direction is not None:
                directions.append(direction)
        direction = None
        if directions:
            left_count = directions.count("left")
            right_count = directions.count("right")
            if left_count or right_count:
                direction = "left" if left_count >= right_count else "right"

        for src_j, (c0, c1) in enumerate(col_parts):
            if direction == "left":
                dst_j = (src_j - 1) % len(col_parts)
            elif direction == "right":
                dst_j = (src_j + 1) % len(col_parts)
            else:
                dst_j = src_j
            dc0, dc1 = col_parts[dst_j]
            cell = grid[r0:r1, c0:c1]
            mask = (cell != bg) & (cell != RED) & (cell != GREY)
            dest = out[r0:r1, dc0:dc1]
            dest[mask] = cell[mask]

    return out


def _line_positions(count, span):
    return [i * (span + 1) for i in range(count + 1)]


def _arrow_pixels(direction, cell_h, cell_w):
    row = random.randint(1, cell_h - 2)
    side = random.choice([-1, 1])
    if direction == "right":
        col = cell_w - 1
        return {(row, col), (row, col - 1), (row + side, col - 1)}
    col = 0
    return {(row, col), (row, col + 1), (row + side, col + 1)}


def _shape_template():
    templates = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
    ]
    return random.choice(templates)


def _place_pattern(grid, r0, c0, cell_h, cell_w, blocked):
    color = random.choice(PATTERN_COLORS)
    padded_block = set(blocked)
    for rr, cc in list(blocked):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < cell_h and 0 <= nc < cell_w:
                    padded_block.add((nr, nc))

    for _ in range(80):
        shape = _shape_template()
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        base_r = random.randint(0, cell_h - max_r - 1)
        base_c = random.randint(0, cell_w - max_c - 1)
        pts = {(base_r + r, base_c + c) for r, c in shape}
        if pts.isdisjoint(padded_block):
            for rr, cc in pts:
                grid[r0 + rr, c0 + cc] = color
            return

    candidates = [
        (rr, cc)
        for rr in range(cell_h)
        for cc in range(cell_w)
        if (rr, cc) not in blocked
    ]
    rr, cc = random.choice(candidates)
    grid[r0 + rr, c0 + cc] = color


def generate_input() -> np.ndarray:
    rows = random.randint(2, 4)
    cols = random.randint(2, 5)
    max_cell_h = (30 - rows - 1) // rows
    max_cell_w = (30 - cols - 1) // cols
    cell_h = random.randint(4, max_cell_h)
    cell_w = random.randint(4, max_cell_w)

    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1
    grid = np.full((height, width), BACKGROUND, dtype=int)

    for rr in _line_positions(rows, cell_h):
        grid[rr, :] = GREY
    for cc in _line_positions(cols, cell_w):
        grid[:, cc] = GREY

    for row_idx in range(rows):
        direction = random.choice(["left", "right"])
        for col_idx in range(cols):
            r0 = 1 + row_idx * (cell_h + 1)
            c0 = 1 + col_idx * (cell_w + 1)
            arrow = _arrow_pixels(direction, cell_h, cell_w)
            _place_pattern(grid, r0, c0, cell_h, cell_w, arrow)
            for rr, cc in arrow:
                grid[r0 + rr, c0 + cc] = RED

    return grid
