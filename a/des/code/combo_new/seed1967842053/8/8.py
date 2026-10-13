import numpy as np


BLACK = 0
GREY = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
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
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _value(grid, r, c):
    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        return int(grid[r, c])
    return BLACK


def _same_color_arrow(grid, cells):
    vals = [_value(grid, r, c) for r, c in cells]
    if vals[0] in (BLACK, GREY):
        return None
    if all(v == vals[0] for v in vals):
        return vals[0]
    return None


def _find_source(grid):
    color_mask = (grid != BLACK) & (grid != GREY)
    candidates = []
    for comp in _components(color_mask):
        if len(comp) < 4:
            continue
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area == len(comp) and np.all(color_mask[r0 : r1 + 1, c0 : c1 + 1]):
            candidates.append((area, r0, r1, c0, c1))
    if not candidates:
        raise ValueError("no filled source pattern found")
    _, r0, r1, c0, c1 = max(candidates)
    return r0, r1, c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the source row or column selected by each arrow color into its grey strip.

    Input structure:
    The grid has a black background, one filled rectangular multicolor source pattern,
    and several one-pixel-wide grey horizontal or vertical strips. Each strip has a
    three-pixel colored V-shaped arrow at one end; horizontal strips have the source
    width, vertical strips have the source height, and arrow colors occur in the source.

    Rule:
    For a horizontal strip, find the source row where the arrow color appears most
    often and copy that whole row into the grey strip from left to right. For a vertical
    strip, find the source column where the arrow color appears most often and copy that
    whole column into the grey strip from top to bottom. Clear the three arrow pixels to
    black and leave the source rectangle unchanged.

    Key insight:
    The arrow color is an index key into the source pattern, not the fill color for the
    strip; the selected row or column is determined by where that color is most frequent.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    sr0, sr1, sc0, sc1 = _find_source(arr)
    source = arr[sr0 : sr1 + 1, sc0 : sc1 + 1]
    source_h, source_w = source.shape

    grey_mask = arr == GREY
    for comp in _components(grey_mask):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        if len(set(rows)) == 1:
            r = rows[0]
            c0, c1 = min(cols), max(cols)
            if len(comp) != source_w or c1 - c0 + 1 != source_w:
                continue
            checks = (
                ((r, c0 - 1), ((r, c0 - 1), (r - 1, c0 - 2), (r + 1, c0 - 2))),
                ((r, c1 + 1), ((r, c1 + 1), (r - 1, c1 + 2), (r + 1, c1 + 2))),
            )
            for _, cells in checks:
                color = _same_color_arrow(arr, cells)
                if color is None:
                    continue
                row_counts = np.sum(source == color, axis=1)
                chosen = int(np.argmax(row_counts))
                out[r, c0 : c1 + 1] = source[chosen, :]
                for ar, ac in cells:
                    out[ar, ac] = BLACK
                break

        elif len(set(cols)) == 1:
            c = cols[0]
            r0, r1 = min(rows), max(rows)
            if len(comp) != source_h or r1 - r0 + 1 != source_h:
                continue
            checks = (
                ((r0 - 1, c), ((r0 - 1, c), (r0 - 2, c - 1), (r0 - 2, c + 1))),
                ((r1 + 1, c), ((r1 + 1, c), (r1 + 2, c - 1), (r1 + 2, c + 1))),
            )
            for _, cells in checks:
                color = _same_color_arrow(arr, cells)
                if color is None:
                    continue
                col_counts = np.sum(source == color, axis=0)
                chosen = int(np.argmax(col_counts))
                out[r0 : r1 + 1, c] = source[:, chosen]
                for ar, ac in cells:
                    out[ar, ac] = BLACK
                break

    return out


def _eligible_colors(pattern):
    row_colors = []
    col_colors = []
    for color in np.unique(pattern):
        color = int(color)
        row_counts = np.sum(pattern == color, axis=1)
        col_counts = np.sum(pattern == color, axis=0)
        row_max = int(row_counts.max())
        col_max = int(col_counts.max())
        if row_max >= 2 and np.count_nonzero(row_counts == row_max) == 1:
            row_colors.append(color)
        if col_max >= 2 and np.count_nonzero(col_counts == col_max) == 1:
            col_colors.append(color)
    return row_colors, col_colors


def _make_pattern(rng, height, width):
    colors = rng.choice(np.array([1, 2, 3, 4, 6, 7, 8, 9]), size=int(rng.integers(3, 6)), replace=False)
    for _ in range(500):
        pattern = rng.choice(colors, size=(height, width))
        row_colors, col_colors = _eligible_colors(pattern)
        if row_colors and col_colors:
            return pattern.astype(int), row_colors, col_colors
    pattern = rng.choice(colors, size=(height, width))
    pattern[0, : max(2, width // 2 + 1)] = colors[0]
    pattern[: max(2, height // 2 + 1), 0] = colors[1]
    return pattern.astype(int), *_eligible_colors(pattern)


def _object_cells(direction, r, c, source_h, source_w):
    if direction == "right":
        arrow = [(r, c), (r - 1, c - 1), (r + 1, c - 1)]
        strip = [(r, c + i) for i in range(1, source_w + 1)]
    elif direction == "left":
        arrow = [(r, c), (r - 1, c + 1), (r + 1, c + 1)]
        strip = [(r, c - i) for i in range(1, source_w + 1)]
    elif direction == "down":
        arrow = [(r, c), (r - 1, c - 1), (r - 1, c + 1)]
        strip = [(r + i, c) for i in range(1, source_h + 1)]
    else:
        arrow = [(r, c), (r + 1, c - 1), (r + 1, c + 1)]
        strip = [(r - i, c) for i in range(1, source_h + 1)]
    return arrow, strip


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(200):
        rows = int(rng.integers(19, 31))
        cols = int(rng.integers(19, 31))
        source_h = int(rng.integers(4, 7))
        source_w = int(rng.integers(4, 7))
        pattern, row_colors, col_colors = _make_pattern(rng, source_h, source_w)
        if not row_colors or not col_colors:
            continue

        grid = np.zeros((rows, cols), dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)
        sr = int(rng.integers(1, rows - source_h))
        sc = int(rng.integers(1, cols - source_w))
        grid[sr : sr + source_h, sc : sc + source_w] = pattern
        occupied[sr : sr + source_h, sc : sc + source_w] = True

        count = int(rng.integers(2, 6))
        orientations = ["h", "v"] + [str(rng.choice(["h", "v"])) for _ in range(count - 2)]
        directions = []
        for orient in orientations:
            if orient == "h":
                directions.append(str(rng.choice(["left", "right"])))
            else:
                directions.append(str(rng.choice(["up", "down"])))
        rng.shuffle(directions)

        def can_place(cells):
            if len(set(cells)) != len(cells):
                return False
            for rr, cc in cells:
                if not (0 <= rr < rows and 0 <= cc < cols):
                    return False
                r0, r1 = max(0, rr - 1), min(rows, rr + 2)
                c0, c1 = max(0, cc - 1), min(cols, cc + 2)
                if np.any(occupied[r0:r1, c0:c1]):
                    return False
            return True

        placed = True
        for direction in directions:
            done = False
            for _attempt in range(400):
                r = int(rng.integers(1, rows - 1))
                c = int(rng.integers(1, cols - 1))
                arrow, strip = _object_cells(direction, r, c, source_h, source_w)
                cells = arrow + strip
                if not can_place(cells):
                    continue
                color_pool = row_colors if direction in ("left", "right") else col_colors
                color = int(rng.choice(color_pool))
                for rr, cc in arrow:
                    grid[rr, cc] = color
                    occupied[rr, cc] = True
                for rr, cc in strip:
                    grid[rr, cc] = GREY
                    occupied[rr, cc] = True
                done = True
                break
            if not done:
                placed = False
                break
        if placed:
            return grid

    grid = np.zeros((22, 22), dtype=int)
    pattern = np.array(
        [
            [1, 1, 1, 2, 3],
            [2, 4, 4, 4, 3],
            [6, 2, 6, 6, 7],
            [8, 8, 2, 9, 9],
            [3, 7, 7, 1, 4],
        ],
        dtype=int,
    )
    grid[2:7, 2:7] = pattern
    grid[10, 4:9] = GREY
    grid[10, 3] = grid[9, 2] = grid[11, 2] = 1
    grid[4:9, 15] = GREY
    grid[3, 15] = grid[2, 14] = grid[2, 16] = 4
    return grid
