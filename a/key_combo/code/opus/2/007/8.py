import random
from collections import deque
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    sep = [c for c in range(w) if np.all(grid[:, c] == 5)][0]
    left = grid[:, :sep]
    right = grid[:, sep + 1:]
    sprite_pts = np.argwhere((left != 0) & (left != 9))
    r0, c0 = sprite_pts.min(axis=0)
    template = []
    by_color = {}
    for r, c in sprite_pts:
        item = (int(r - r0), int(c - c0), int(left[r, c]))
        template.append(item)
        by_color.setdefault(int(left[r, c]), []).append((int(r - r0), int(c - c0)))

    canvas = np.zeros_like(right)
    markers = [(int(r), int(c), int(right[r, c])) for r, c in np.argwhere((right != 0) & (right != 9))]
    for r, c, color in sorted(markers, key=lambda x: (x[1], x[0])):
        ar, ac = by_color[color][0]
        top, left_col = r - ar, c - ac
        for dr, dc, pix in template:
            rr, cc = top + dr, left_col + dc
            if 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1]:
                canvas[rr, cc] = pix

    ph, pw = canvas.shape
    rm, cm = (ph - 1) // 2, (pw - 1) // 2
    quads = {
        "tl": (slice(0, rm + 1), slice(0, cm + 1)),
        "tr": (slice(0, rm + 1), slice(cm + 1, pw)),
        "bl": (slice(rm + 1, ph), slice(0, cm + 1)),
        "br": (slice(rm + 1, ph), slice(cm + 1, pw)),
    }
    best = max(quads, key=lambda q: int(np.count_nonzero(canvas[quads[q]])))
    area = canvas[quads[best]]
    area[area == 0] = 5

    out = grid.copy()
    out[:, sep + 1:] = canvas
    return out


def _connected_shape(n):
    cells = {(1, 1)}
    while len(cells) < n:
        r, c = random.choice(list(cells))
        nr, nc = random.choice([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
        if 0 <= nr < 5 and 0 <= nc < 5:
            cells.add((nr, nc))
    return sorted(cells)


def generate_input() -> np.ndarray:
    for _ in range(2000):
        grid = np.zeros((10, 21), dtype=int)
        grid[:, 10] = 5
        cells = _connected_shape(random.randint(9, 13))
        minr, minc = min(r for r, _ in cells), min(c for _, c in cells)
        cells = [(r - minr, c - minc) for r, c in cells]
        off_r, off_c = random.randint(1, 3), random.randint(1, 3)
        colors = [1, 2, 3, 4, 6, 7, 8]
        anchors = random.sample(cells, 3)
        anchor_colors = random.sample(colors, 3)
        filler_colors = [c for c in colors if c not in anchor_colors]
        used = {}
        for cell, color in zip(anchors, anchor_colors):
            used[cell] = color
        for cell in cells:
            if cell not in used:
                used[cell] = random.choice(filler_colors)
        for (r, c), color in used.items():
            grid[off_r + r, off_c + c] = color
        for r, c in random.sample([(r, c) for r in range(10) for c in range(10) if grid[r, c] == 0], random.randint(2, 3)):
            grid[r, c] = 9

        markers = []
        for color in random.sample(anchor_colors, random.randint(2, 3)):
            ar, ac = [p for p, col in used.items() if col == color][0]
            for _j in range(200):
                rr = random.randint(ar, 9 - (max(r for r, _ in cells) - ar))
                cc = random.randint(ac, 9 - (max(c for _, c in cells) - ac))
                if (rr, cc) not in [(a, b) for a, b, _ in markers]:
                    markers.append((rr, cc, color))
                    break
        if len(markers) < 2:
            continue
        for r, c, color in markers:
            grid[r, 11 + c] = color
        for r, c in random.sample([(r, c) for r in range(10) for c in range(10) if grid[r, 11 + c] == 0], random.randint(1, 2)):
            grid[r, 11 + c] = 9
        out = main(grid)
        right = out[:, 11:]
        rm = cm = 4
        counts = [
            np.count_nonzero(right[:rm + 1, :cm + 1] != 0),
            np.count_nonzero(right[:rm + 1, cm + 1:] != 0),
            np.count_nonzero(right[rm + 1:, :cm + 1] != 0),
            np.count_nonzero(right[rm + 1:, cm + 1:] != 0),
        ]
        if counts.count(max(counts)) == 1:
            return grid
    raise RuntimeError("failed to generate input")
