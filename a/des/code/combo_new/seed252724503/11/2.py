import numpy as np


BLACK = 0


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _legend(grid):
    items = []
    for r in range(grid.shape[0]):
        color = grid[r, 0]
        if color != BLACK:
            items.append(color)
    return items


def _segments_for_color(grid, color, skip_col=0):
    mask = (grid == color).copy()
    mask[:, skip_col] = False
    segs = []
    for comp in _components(mask):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        segs.append((r0, r1, c0, c1))
    return segs


def _endpoints(seg):
    r0, r1, c0, c1 = seg
    if r0 == r1:
        return [(r0, c0), (r0, c1)]
    return [(r0, c0), (r1, c0)]


def _clear_path(grid, a, b):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        return np.all(grid[r1, lo + 1 : hi] == BLACK)
    if c1 == c2:
        lo, hi = sorted((r1, r2))
        return np.all(grid[lo + 1 : hi, c1] == BLACK)
    return False


def _draw_line(grid, a, b, color):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        grid[r1, lo : hi + 1] = color
    else:
        lo, hi = sorted((r1, r2))
        grid[lo : hi + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned line-segment endpoints using the lower-priority color from the legend chain.

    Input structure:
    - Black background with colored horizontal or vertical line segments.
    - The left edge contains a vertical legend of singleton colors from highest to lowest priority.
    - Only one endpoint pairing between each adjacent legend colors is aligned in a clear straight line.

    Rule:
    - Read the color hierarchy from top to bottom in the left-edge legend.
    - For each adjacent pair of legend colors, find the unique aligned endpoints from those colors whose connecting path is unobstructed.
    - Draw the connecting segment inclusively in the lower-priority color, so the lower-priority color overwrites the higher-priority endpoint it touches.

    Key insight:
    - The edge legend encodes both connection order and connection color, while the geometry only reveals which endpoints are meant to be chained together.
    """
    out = grid.copy()
    order = _legend(grid)
    for high, low in zip(order, order[1:]):
        candidates = []
        for seg_a in _segments_for_color(grid, high):
            for seg_b in _segments_for_color(grid, low):
                for ep_a in _endpoints(seg_a):
                    for ep_b in _endpoints(seg_b):
                        if _clear_path(grid, ep_a, ep_b):
                            candidates.append((abs(ep_a[0] - ep_b[0]) + abs(ep_a[1] - ep_b[1]), ep_a, ep_b))
        if candidates:
            _, ep_a, ep_b = min(candidates, key=lambda item: item[0])
            _draw_line(out, ep_a, ep_b, low)
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    n_colors = int(rng.integers(3, 5))
    colors = list(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], size=n_colors, replace=False))
    height = 18 + 3 * n_colors
    width = 20 + 2 * n_colors
    grid = np.zeros((height, width), dtype=int)

    legend_rows = list(range(height - (2 * n_colors), height, 2))
    for r, color in zip(legend_rows, colors):
        grid[r, 0] = color

    rows = list(range(2, 2 + 4 * (n_colors - 1), 4))
    for i in range(n_colors - 1):
        high = colors[i]
        low = colors[i + 1]
        col = 4 + 4 * i + int(rng.integers(0, 2))
        top_row = rows[i]
        bot_row = rows[i] + 2
        # Chain segments.
        grid[top_row, col - 2 : col + 1] = high
        grid[bot_row, col : col + 3] = low

    # Spare segments for the ends.
    grid[rows[0] + 1, width - 6 : width - 2] = colors[0]
    grid[rows[-1] + 1, width - 8 : width - 4] = colors[-1]
    return grid
