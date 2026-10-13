import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    markers: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in (0, 5):
                markers.setdefault(color, []).append((r, c))

    vertical_paths: dict[tuple[int, int], int] = {}
    vertical_paint: dict[tuple[int, int], int] = {}
    horizontal_paths: dict[tuple[int, int], int] = {}
    horizontal_paint: dict[tuple[int, int], int] = {}

    for color, pts in markers.items():
        if len(pts) != 2:
            continue
        (r1, c1), (r2, c2) = pts
        if r1 == r2:
            a, b = sorted((c1, c2))
            start, stop = a + 2, b - 2
            for c in range(start, stop + 1):
                horizontal_paths[(r1, c)] = color
                if (c - start) % 2 == 0:
                    horizontal_paint[(r1, c)] = color
        elif c1 == c2:
            a, b = sorted((r1, r2))
            start, stop = a + 2, b - 2
            for r in range(start, stop + 1):
                vertical_paths[(r, c1)] = color
                if (r - start) % 2 == 0:
                    vertical_paint[(r, c1)] = color

    out = np.zeros_like(grid)
    for pos, color in vertical_paint.items():
        out[pos] = color
    for pos, hcolor in horizontal_paths.items():
        if pos in vertical_paths:
            out[pos] = hcolor
    for pos, color in horizontal_paint.items():
        out[pos] = color

    for color, pts in markers.items():
        for r, c in pts:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr, cc] = 5
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((17, 17), dtype=int)
    required = [
        ((8, 2), (8, 14)),
        ((2, 8), (14, 8)),
    ]
    extras = [
        ((4, 1), (4, 15)),
        ((1, 4), (15, 4)),
        ((12, 1), (12, 15)),
    ]
    pair_count = random.randint(3, 5)
    colors = random.sample([1, 3, 4, 6, 7], pair_count)
    random.shuffle(extras)
    templates = required + extras[: pair_count - 2]
    random.shuffle(templates)
    for color, (a, b) in zip(colors, templates[:pair_count]):
        grid[a] = color
        grid[b] = color
    return grid
