import numpy as np


BACKGROUND = 0


def _read_legend(top_row):
    cols = np.flatnonzero(top_row != BACKGROUND)
    if len(cols) == 0:
        return []

    runs = []
    start = int(cols[0])
    prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
        else:
            runs.append((start, prev))
            start = col
            prev = col
    runs.append((start, prev))

    start, end = max(runs, key=lambda run: (run[1] - run[0] + 1, -run[0]))
    return [int(x) for x in top_row[start : end + 1]]


def main(grid: np.ndarray) -> np.ndarray:
    """Grow colored concentric square rings around each seed from the legend order.

    Input structure:
    The background is black. The top row contains a contiguous color legend whose
    distinct non-black cells define the visible color order from left to right.
    Below the legend are isolated one-cell seeds; each seed color appears in the
    legend, and the seeds are spaced so their grown squares do not overlap.

    Rule:
    Copy the input, then process every seed below the top row. If the seed color
    is the nth color in the legend, draw n square rings centered on that seed.
    Ring 1 uses the first legend color, ring 2 uses the second legend color, and
    so on until the outer ring uses the seed color. The seed cell itself keeps
    its original color.

    Key insight:
    A seed's visible position in the top-row legend encodes both the number of
    rings to draw and the color of the outermost ring.
    """
    out = np.array(grid, copy=True)
    h, w = grid.shape
    legend = _read_legend(grid[0])
    rank = {color: idx + 1 for idx, color in enumerate(legend)}

    for r in range(1, h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in rank:
                continue

            radius = rank[color]
            for dist in range(1, radius + 1):
                ring_color = legend[dist - 1]
                r0, r1 = r - dist, r + dist
                c0, c1 = c - dist, c + dist
                cs0, cs1 = max(0, c0), min(w, c1 + 1)
                rs0, rs1 = max(0, r0), min(h, r1 + 1)

                if 0 <= r0 < h:
                    out[r0, cs0:cs1] = ring_color
                if 0 <= r1 < h:
                    out[r1, cs0:cs1] = ring_color
                if 0 <= c0 < w:
                    out[rs0:rs1, c0] = ring_color
                if 0 <= c1 < w:
                    out[rs0:rs1, c1] = ring_color

            out[r, c] = color

    return out


def _can_place(occupied, row, col, radius):
    h, w = occupied.shape
    if row - radius < 1 or row + radius >= h:
        return False
    if col - radius < 0 or col + radius >= w:
        return False

    r0 = max(1, row - radius - 1)
    r1 = min(h, row + radius + 2)
    c0 = max(0, col - radius - 1)
    c1 = min(w, col + radius + 2)
    return not occupied[r0:r1, c0:c1].any()


def _mark_place(occupied, row, col, radius):
    h, w = occupied.shape
    r0 = max(1, row - radius - 1)
    r1 = min(h, row + radius + 2)
    c0 = max(0, col - radius - 1)
    c1 = min(w, col + radius + 2)
    occupied[r0:r1, c0:c1] = True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    legend_len = int(rng.integers(3, 7))
    legend = rng.choice(np.arange(1, 10), size=legend_len, replace=False)

    min_size = max(10, 2 * legend_len + 4)
    h = int(rng.integers(min_size, 31))
    w = int(rng.integers(max(min_size, legend_len + 3), 31))

    grid = np.zeros((h, w), dtype=int)
    grid[0, :legend_len] = legend

    occupied = np.zeros((h, w), dtype=bool)
    target_count = int(rng.integers(1, 6))
    seeds = []

    for _ in range(1200):
        if len(seeds) >= target_count:
            break

        radius = int(rng.integers(1, legend_len + 1))
        if h - radius <= radius + 1 or w - radius <= radius:
            continue

        row = int(rng.integers(radius + 1, h - radius))
        col = int(rng.integers(radius, w - radius))
        if not _can_place(occupied, row, col, radius):
            continue

        grid[row, col] = int(legend[radius - 1])
        _mark_place(occupied, row, col, radius)
        seeds.append((row, col))

    if not seeds:
        radius = int(rng.integers(1, legend_len + 1))
        row = int(rng.integers(radius + 1, h - radius))
        col = int(rng.integers(radius, w - radius))
        grid[row, col] = int(legend[radius - 1])

    return grid
