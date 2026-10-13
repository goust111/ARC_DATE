import random
from collections import deque

import numpy as np


def _background_color(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _template_rows(grid, bg):
    rows = []
    h, w = grid.shape
    for r in range(h):
        row = grid[r]
        if np.all(row != bg) and len(set(map(int, row))) >= 2:
            rows.append(r)
    if len(rows) < 2:
        rows = [
            r
            for r in range(h)
            if np.count_nonzero(grid[r] != bg) == w
            and len(set(map(int, grid[r]))) >= 2
        ]
    if len(rows) != 2:
        raise ValueError("expected exactly two complete template rows")
    return rows


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each object by reading the opposite template row at the same columns.

    Input structure:
    The grid has a dominant black background, exactly two complete horizontal
    template rows, and scattered 4-connected monochrome objects outside those
    rows. Each object color appears in exactly one of the two template rows.

    Rule:
    Keep both template rows unchanged. For every other non-background component,
    use its original color to identify the template row containing that color.
    Then recolor each pixel of the component with the color in the same column
    of the other template row.

    Key insight:
    An object's color is an index key selecting which template row it belongs to;
    the visible output colors come from the opposite template row's column pattern.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    bg = _background_color(grid)
    top, bottom = _template_rows(grid, bg)
    rows = [top, bottom]
    row_values = [set(map(int, grid[r])) for r in rows]

    object_mask = grid != bg
    object_mask[rows, :] = False

    for comp in _components(object_mask):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = colors.pop()
        in_first = color in row_values[0]
        in_second = color in row_values[1]
        if in_first == in_second:
            continue
        source_idx = 0 if in_first else 1
        target_row = rows[1 - source_idx]
        for r, c in comp:
            out[r, c] = grid[target_row, c]
    return out


def _make_template(width, colors):
    seq = []
    block = random.choice([1, 1, 2, 3])
    offset = random.randrange(len(colors))
    for c in range(width):
        seq.append(colors[(c // block + offset) % len(colors)])
    if len(set(seq)) < 2:
        seq[-1] = colors[(offset + 1) % len(colors)]
    return np.array(seq, dtype=int)


def _random_shape(max_cells=10):
    cells = {(0, 0)}
    frontier = [(0, 0)]
    target = random.randint(2, max_cells)
    while len(cells) < target:
        r, c = random.choice(frontier)
        nr, nc = random.choice(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if (nr, nc) not in cells:
            cells.add((nr, nc))
            frontier.append((nr, nc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _can_place(occupied, rows, cols, cells, rr, cc):
    h, w = occupied.shape
    for dr, dc in cells:
        r, c = rr + dr, cc + dc
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c] or r in rows:
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def generate_input() -> np.ndarray:
    h = random.randint(12, 24)
    w = random.randint(12, 28)
    bg = 0
    grid = np.full((h, w), bg, dtype=int)

    palette = list(range(1, 10))
    random.shuffle(palette)
    split = random.randint(3, 5)
    colors_a = palette[:split]
    colors_b = palette[split : split + random.randint(3, 5)]
    if len(colors_b) < 3:
        colors_b = palette[-3:]

    row_a, row_b = sorted(random.sample(range(1, h - 1), 2))
    grid[row_a] = _make_template(w, colors_a)
    grid[row_b] = _make_template(w, colors_b)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[[row_a, row_b], :] = True
    rows = {row_a, row_b}
    count = random.randint(5, 11)
    branch_colors = [colors_a, colors_b]
    branch_quota = [count // 2, count - count // 2]
    random.shuffle(branch_quota)

    for branch, quota in enumerate(branch_quota):
        for _ in range(quota):
            placed = False
            for _attempt in range(100):
                shape = _random_shape(random.randint(4, 9))
                max_r = max(r for r, _ in shape)
                max_c = max(c for _, c in shape)
                if max_r >= h - 2 or max_c >= w - 2:
                    continue
                rr = random.randint(0, h - max_r - 1)
                cc = random.randint(0, w - max_c - 1)
                if _can_place(occupied, rows, w, shape, rr, cc):
                    color = random.choice(branch_colors[branch])
                    for dr, dc in shape:
                        grid[rr + dr, cc + dc] = color
                        occupied[rr + dr, cc + dc] = True
                    placed = True
                    break
            if not placed:
                continue

    return grid
