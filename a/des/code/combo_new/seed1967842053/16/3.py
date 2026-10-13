import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _legend_colors(grid):
    h, w = grid.shape
    for r in range(h - 1, -1, -1):
        runs = []
        c = 0
        while c < w:
            color = int(grid[r, c])
            if color not in (BLACK, GREY):
                start = c
                while c < w and int(grid[r, c]) == color:
                    c += 1
                runs.append((start, color))
            else:
                c += 1
        if len(runs) == 4 and len({color for _, color in runs}) == 4:
            runs.sort(key=lambda item: item[0])
            return [color for _, color in runs]
    raise ValueError("no four-color legend row found")


def _grey_frames(grid):
    mask = grid == GREY
    seen = np.zeros(mask.shape, dtype=bool)
    frames = []
    h, w = grid.shape

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != width or height < 3:
            continue

        expected = 2 * height + 2 * width - 4
        if len(cells) != expected:
            continue
        if not all(r in (r0, r1) or c in (c0, c1) for r, c in cells):
            continue
        if not (
            np.all(grid[r0, c0 : c1 + 1] == GREY)
            and np.all(grid[r1, c0 : c1 + 1] == GREY)
            and np.all(grid[r0 : r1 + 1, c0] == GREY)
            and np.all(grid[r0 : r1 + 1, c1] == GREY)
        ):
            continue
        frames.append((r0, r1, c0, c1))

    return frames


def _pointer_direction(grid, frame):
    r0, r1, c0, c1 = frame
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    color = int(grid[cr, cc])
    if color in (BLACK, GREY):
        raise ValueError("frame center is not a pointer color")

    best_direction = None
    best_length = -1
    for dr, dc in DIRECTIONS:
        length = 0
        rr, col = cr + dr, cc + dc
        while r0 < rr < r1 and c0 < col < c1 and int(grid[rr, col]) == color:
            length += 1
            rr += dr
            col += dc
        if length > best_length:
            best_length = length
            best_direction = (dr, dc)

    if best_length <= 0:
        raise ValueError("no pointer arm found")
    return best_direction


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each hollow square according to the direction indicated by its center pointer.

    Input structure:
    The grid has a black background, several same-sized grey hollow square frames, and
    a bottom legend row made of four colored blocks. Each square contains a unique
    non-black, non-grey center color with a short same-colored cardinal arm pointing
    up, right, down, or left.

    Rule:
    Read the four legend colors from left to right as the fill colors for up, right,
    down, and left. For every grey square frame, infer the direction of the same-colored
    arm extending from the center pixel, then fill only the black interior cells of that
    frame with the legend color for that direction. Leave grey borders, pointer pixels,
    and the legend unchanged.

    Key insight:
    The pointer color only identifies the pointer shape; the fill color is obtained by
    converting pointer direction through the spatial order of the bottom legend.
    """
    out = np.array(grid, copy=True)
    legend = _legend_colors(grid)
    color_for_direction = dict(zip(DIRECTIONS, legend))

    for frame in _grey_frames(grid):
        direction = _pointer_direction(grid, frame)
        fill = color_for_direction[direction]
        r0, r1, c0, c1 = frame
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = fill

    return out


def _rects_overlap(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_frames(height, width, size, count):
    placed = []
    for _ in range(count):
        for _attempt in range(300):
            top = random.randint(1, height - size - 2)
            left = random.randint(1, width - size - 1)
            padded = (top - 1, top + size, left - 1, left + size)
            if all(not _rects_overlap(padded, old) for old in placed):
                placed.append(padded)
                break
        else:
            return None
    return [(r0 + 1, c0 + 1) for r0, _, c0, _ in placed]


def generate_input() -> np.ndarray:
    for _ in range(200):
        frame_size = random.choice([7, 9])
        count = random.randint(1, 4)
        height = random.randint(max(14, frame_size + 5), 30)
        width = random.randint(max(16, frame_size + 5), 30)

        positions = _place_frames(height - 1, width, frame_size, count)
        if positions is None:
            continue

        colors = random.sample(PALETTE, 8)
        legend = colors[:4]
        pointer_colors = colors[4 : 4 + count]
        grid = np.zeros((height, width), dtype=int)

        for i, (top, left) in enumerate(positions):
            bottom = top + frame_size - 1
            right = left + frame_size - 1
            grid[top, left : right + 1] = GREY
            grid[bottom, left : right + 1] = GREY
            grid[top : bottom + 1, left] = GREY
            grid[top : bottom + 1, right] = GREY

            cr = top + frame_size // 2
            cc = left + frame_size // 2
            color = pointer_colors[i]
            direction = random.choice(DIRECTIONS)
            grid[cr, cc] = color
            for step in (1, 2):
                grid[cr + direction[0] * step, cc + direction[1] * step] = color

        block_widths = [random.randint(1, 3) for _ in range(4)]
        total_width = sum(block_widths) + 3
        start = random.randint(0, width - total_width)
        c = start
        for color, block_width in zip(legend, block_widths):
            grid[height - 1, c : c + block_width] = color
            c += block_width + 1

        return grid

    raise RuntimeError("could not generate a non-overlapping puzzle")
