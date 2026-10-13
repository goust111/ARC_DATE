import random
import numpy as np


def _marker_positions(grid):
    h, w = grid.shape
    out = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if grid[r, c] != 0:
                    out.append((r, c, int(grid[r, c])))
    return out


def _to_output_coord(pos, in_len, out_len):
    return pos - (in_len - out_len) // 2


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    marker_cells = {(r, c) for r, c, _ in _marker_positions(grid)}
    pieces = {}
    for r, c in np.argwhere(grid != 0):
        if (int(r), int(c)) in marker_cells:
            continue
        color = int(grid[r, c])
        pieces.setdefault(color, []).append((int(r), int(c)))

    rings = []
    for color, pts in pieces.items():
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        side = max(max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
        rings.append((side, color))
    rings.sort(reverse=True)
    if not rings:
        return np.zeros((0, 0), dtype=grid.dtype)

    size = rings[0][0]
    out = np.zeros((size, size), dtype=grid.dtype)
    for side, color in rings:
        top = (size - side) // 2
        left = (size - side) // 2
        bottom = top + side - 1
        right = left + side - 1
        out[top, left:right + 1] = color
        out[bottom, left:right + 1] = color
        out[top:bottom + 1, left] = color
        out[top:bottom + 1, right] = color

    by_side = {"top": [], "right": [], "bottom": [], "left": []}
    for r, c, color in _marker_positions(grid):
        if r == 0:
            by_side["top"].append((_to_output_coord(c, w, size), color))
        elif c == w - 1:
            by_side["right"].append((_to_output_coord(r, h, size), color))
        elif r == h - 1:
            by_side["bottom"].append((_to_output_coord(c, w, size), color))
        elif c == 0:
            by_side["left"].append((_to_output_coord(r, h, size), color))

    for c, color in by_side["top"]:
        if 0 <= c < size:
            for r in range(size):
                if out[r, c] != 0:
                    break
                out[r, c] = color
    for r, color in by_side["right"]:
        if 0 <= r < size:
            for c in range(size - 1, -1, -1):
                if out[r, c] != 0:
                    break
                out[r, c] = color
    for c, color in by_side["bottom"]:
        if 0 <= c < size:
            for r in range(size - 1, -1, -1):
                if out[r, c] != 0:
                    break
                out[r, c] = color
    for r, color in by_side["left"]:
        if 0 <= r < size:
            for c in range(size):
                if out[r, c] != 0:
                    break
                out[r, c] = color
    return out


def _fragment_frame(grid, color, side, center, rng):
    top = center[0] - side // 2
    left = center[1] - side // 2
    cells = []
    for c in range(left, left + side):
        cells.append((top, c))
        cells.append((top + side - 1, c))
    for r in range(top + 1, top + side - 1):
        cells.append((r, left))
        cells.append((r, left + side - 1))
    rng.shuffle(cells)
    for r, c in cells[: max(6, len(cells) * rng.randint(45, 70) // 100)]:
        if 0 < r < grid.shape[0] - 1 and 0 < c < grid.shape[1] - 1:
            grid[r, c] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    max_side = rng.choice([13, 15, 17])
    size = max_side + 4
    grid = np.zeros((size, size), dtype=int)
    available = [1, 3, 4, 6, 8, 9]
    rng.shuffle(available)
    sides = [max_side, max_side - 4, max_side - 8]
    if max_side >= 17 and rng.random() < 0.5:
        sides.append(max_side - 12)
    colors = available[:len(sides)]
    center = (size // 2, size // 2)
    for color, side in zip(colors, sides):
        _fragment_frame(grid, color, side, center, rng)

    marker_colors = [2, 7]
    choices = list(range(3, size - 3))
    top_c = rng.choice(choices)
    right_r = rng.choice(choices)
    bottom_c = rng.choice(choices)
    left_r = rng.choice(choices)
    grid[0, top_c] = rng.choice(marker_colors)
    grid[right_r, size - 1] = rng.choice(marker_colors)
    grid[size - 1, bottom_c] = rng.choice(marker_colors)
    grid[left_r, 0] = rng.choice(marker_colors)
    return grid
