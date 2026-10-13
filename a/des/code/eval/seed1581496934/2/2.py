import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _terminal_boxes(grid: np.ndarray):
    h, w = grid.shape
    boxes = []
    occupied = set()
    for r in range(h - 3):
        for c in range(w - 3):
            block = grid[r : r + 4, c : c + 4]
            color = int(block[0, 0])
            if color in (BG, GREY):
                continue
            ring = []
            for rr in range(4):
                for cc in range(4):
                    if rr in (0, 3) or cc in (0, 3):
                        ring.append(int(block[rr, cc]))
            if all(v == color for v in ring) and np.all(block[1:3, 1:3] == BG):
                cells = {
                    (r + rr, c + cc)
                    for rr in range(4)
                    for cc in range(4)
                    if rr in (0, 3) or cc in (0, 3)
                }
                boxes.append(
                    {
                        "top": r,
                        "left": c,
                        "color": color,
                        "ring": cells,
                        "inner": {(r + 1, c + 1), (r + 1, c + 2), (r + 2, c + 1), (r + 2, c + 2)},
                    }
                )
                occupied.update(cells)
    return boxes, occupied


def _neighbors(pos, h, w):
    r, c = pos
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _branch_width(grid: np.ndarray, pos, direction) -> int:
    h, w = grid.shape
    r, c = pos
    dr, dc = direction
    probe = (r + dr, c + dc)
    if 0 <= probe[0] < h and 0 <= probe[1] < w and int(grid[probe]) == GREY:
        r, c = probe
    width = 1
    if dr:
        for side in (-1, 1):
            nc = c + side
            if 0 <= nc < w and int(grid[r, nc]) == GREY:
                width += 1
    else:
        for side in (-1, 1):
            nr = r + side
            if 0 <= nr < h and int(grid[nr, c]) == GREY:
                width += 1
    return min(width, 2)


def _touching_terminal(pos, boxes):
    r, c = pos
    for box in boxes:
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (nr, nc) in box["ring"]:
                return box
    return None


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    boxes, terminal_ring = _terminal_boxes(grid)

    starts = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, GREY) or (r, c) in terminal_ring:
                continue
            grey_neighbors = [(nr, nc) for nr, nc in _neighbors((r, c), h, w) if int(grid[nr, nc]) == GREY]
            if grey_neighbors:
                for nxt in grey_neighbors:
                    starts.append((color, nxt, (r, c)))

    for color, start, prev in starts:
        queue = deque([(start, prev)])
        visited = set()
        while queue:
            cell, came_from = queue.popleft()
            if (cell, color) in visited:
                continue
            visited.add((cell, color))
            if int(grid[cell]) != GREY:
                continue
            out[cell] = color

            box = _touching_terminal(cell, boxes)
            if box is not None:
                if int(box["color"]) == color:
                    for inner in box["inner"]:
                        out[inner] = color
                continue

            candidates = []
            cr, cc = cell
            for nr, nc in _neighbors(cell, h, w):
                if (nr, nc) == came_from or int(grid[nr, nc]) != GREY:
                    continue
                direction = (nr - cr, nc - cc)
                width = _branch_width(grid, (nr, nc), direction)
                candidates.append((width, (nr, nc)))

            if not candidates:
                continue
            if len(candidates) > 1:
                best = max(width for width, _ in candidates)
                candidates = [(width, pos) for width, pos in candidates if width == best]
            for _, nxt in candidates:
                queue.append((nxt, cell))

    return out


def _draw_terminal(grid: np.ndarray, top: int, left: int, color: int):
    grid[top : top + 4, left : left + 4] = color
    grid[top + 1 : top + 3, left + 1 : left + 3] = BG


def _make_one() -> np.ndarray:
    h = random.randint(20, 29)
    w = random.randint(24, 30)
    grid = np.full((h, w), BG, dtype=int)

    row = random.randint(4, max(4, h - 11))
    junction = random.randint(7, 11)
    right_end = w - 7
    down_end = h - 6
    color = random.choice(COLORS)
    other = random.choice([c for c in COLORS if c != color])

    source_col = 1
    grid[row, source_col] = color
    grid[row, source_col + 1 : junction + 1] = GREY

    right_width = random.choice([1, 1, 2])
    down_width = random.choice([1, 2, 2])
    if random.random() < 0.3:
        down_width = right_width

    grid[row, junction + 1 : right_end + 1] = GREY
    if right_width == 2:
        grid[row + 1, junction + 2 : right_end + 1] = GREY

    grid[row + 1 : down_end + 1, junction] = GREY
    if down_width == 2:
        grid[row + 2 : down_end + 1, junction + 1] = GREY

    right_color = color if random.random() < 0.85 else other
    down_color = color if random.random() < 0.85 else other
    _draw_terminal(grid, row - 1, right_end + 1, right_color)
    _draw_terminal(grid, down_end + 1, junction - 1, down_color)

    # Add one disconnected grey spur so untraveled pipe remains visibly grey.
    spur_r = random.randint(1, h - 3)
    spur_c = random.randint(2, w - 9)
    if np.all(grid[spur_r : spur_r + 2, spur_c : spur_c + 6] == BG):
        grid[spur_r, spur_c : spur_c + random.randint(3, 6)] = GREY

    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _make_one()
        out = main(grid)
        if isinstance(out, np.ndarray) and out.shape == grid.shape and np.any(out != grid):
            return grid
    return _make_one()
