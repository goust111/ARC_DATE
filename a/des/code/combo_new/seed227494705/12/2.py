import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild colored rectangle sizes from fragments and redraw them as legend-ordered nesting.

    Input structure: azure background with colored horizontal/vertical rectangle
    outline fragments and a bottom-right two-row legend whose upper row names
    colors and whose lower row shows each color repeated by its nesting rank.
    Rule: for each legend color, infer its rectangle height and width from the
    bounding box of its non-legend fragments, sort colors by visible rank, and
    draw centered one-pixel rectangular outlines from outer to inner with one
    background margin around the largest outline.
    Key insight: fragment geometry supplies dimensions, but the independent
    legend supplies nesting order.
    """
    bg = 8
    h, w = grid.shape
    top = h - 2
    legend_cols = [c for c in range(w) if grid[top, c] != bg]
    colors = [int(grid[top, c]) for c in legend_cols]
    ranks = {color: int(np.sum(grid[h - 1] == color)) for color in colors}
    sizes = {}
    for color in colors:
        mask = grid == color
        mask[h - 2:h, :] = False
        pts = np.argwhere(mask)
        if len(pts) == 0:
            continue
        sizes[color] = (int(pts[:, 0].max() - pts[:, 0].min() + 1), int(pts[:, 1].max() - pts[:, 1].min() + 1))
    if not sizes:
        return np.full((1, 1), bg, dtype=int)
    max_h = max(v[0] for v in sizes.values())
    max_w = max(v[1] for v in sizes.values())
    out = np.full((max_h + 2, max_w + 2), bg, dtype=int)
    for color in sorted(sizes, key=lambda x: ranks.get(x, 0), reverse=True):
        rh, rw = sizes[color]
        r = (out.shape[0] - rh) // 2
        c = (out.shape[1] - rw) // 2
        out[r, c:c + rw] = color
        out[r + rh - 1, c:c + rw] = color
        out[r:r + rh, c] = color
        out[r:r + rh, c + rw - 1] = color
    return out


def generate_input() -> np.ndarray:
    bg = 8
    n = random.randint(2, 4)
    grid = np.full((24, 28), bg, dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 9], n)
    ranks = list(range(1, n + 1))
    random.shuffle(ranks)
    sizes = {}
    base_h, base_w = 5, 7
    for color, rank in zip(colors, ranks):
        sizes[color] = (base_h + 4 * (rank - 1), base_w + 4 * (rank - 1))
    anchors = [(1, 1), (1, 15), (12, 1), (12, 15)]
    random.shuffle(anchors)
    for color, (r, c) in zip(colors, anchors):
        rh, rw = sizes[color]
        if r + rh >= 22 or c + rw >= 26:
            r, c = 1, 1
        # Draw separated outline fragments whose bbox still exposes the source size.
        grid[r, c:c + min(3, rw)] = color
        grid[r, c + rw - 3:c + rw] = color
        grid[r + rh - 1, c:c + min(4, rw)] = color
        grid[r + rh - 1, c + rw - 3:c + rw] = color
        grid[r:r + min(3, rh), c] = color
        grid[r + rh - 3:r + rh, c] = color
        grid[r:r + min(4, rh), c + rw - 1] = color
        grid[r + rh - 3:r + rh, c + rw - 1] = color
    start = grid.shape[1] - (n * (n + 1) + 1)
    pos = start
    for color, rank in zip(colors, ranks):
        grid[-2, pos] = color
        grid[-1, pos:pos + rank] = color
        pos += n + 1
    return grid
