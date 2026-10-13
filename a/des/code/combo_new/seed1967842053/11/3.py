import random
from collections import deque

import numpy as np


BG = 0


def _find_template(grid):
    rows, cols = grid.shape
    best = None
    best_area = -1
    for r1 in range(rows - 3):
        for r2 in range(r1 + 3, rows):
            for c1 in range(cols - 3):
                for c2 in range(c1 + 3, cols):
                    top = int(grid[r1, c1])
                    bottom = int(grid[r2, c1])
                    if top == BG or bottom == BG or top == bottom:
                        continue
                    if not np.all(grid[r1, c1 : c2 + 1] == top):
                        continue
                    if not np.all(grid[r2, c1 : c2 + 1] == bottom):
                        continue
                    left = int(grid[r1 + 1, c1])
                    right = int(grid[r1 + 1, c2])
                    colors = {top, bottom, left, right}
                    if BG in colors or len(colors) != 4:
                        continue
                    if not np.all(grid[r1 + 1 : r2, c1] == left):
                        continue
                    if not np.all(grid[r1 + 1 : r2, c2] == right):
                        continue
                    if not np.all(grid[r1 + 1 : r2, c1 + 1 : c2] == BG):
                        continue
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if area > best_area:
                        best_area = area
                        best = (r1, r2, c1, c2, top, bottom, left, right)
    return best


def _border_mask(shape, r1, r2, c1, c2):
    mask = np.zeros(shape, dtype=bool)
    mask[r1, c1 : c2 + 1] = True
    mask[r2, c1 : c2 + 1] = True
    mask[r1 + 1 : r2, c1] = True
    mask[r1 + 1 : r2, c2] = True
    return mask


def _components(grid, blocked):
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    comps = []
    for sr in range(rows):
        for sc in range(cols):
            color = int(grid[sr, sc])
            if color == BG or blocked[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if seen[nr, nc] or blocked[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append((color, cells))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Move colored fragments into the matching bordered template.

    Input structure: The grid has a black background, one large hollow
    rectangle with four differently colored one-pixel border sides, and small
    single-color line or L-shaped fragments outside the rectangle. Each
    fragment color is one of the four border colors.
    Rule: Keep the colored template border fixed. For every outside fragment,
    use its color to choose the matching side of the rectangle, then translate
    the fragment into the rectangle without rotation. Top and bottom fragments
    keep their column coordinates and are moved vertically until they touch the
    corresponding inside edge. Left and right fragments keep their row
    coordinates and are moved horizontally until they touch the corresponding
    inside edge. Outside fragments are cleared.
    Key insight: The border color identifies the target side, while the
    fragment's coordinate parallel to that side gives its final offset inside
    the empty template.
    """
    g = np.asarray(grid)
    found = _find_template(g)
    if found is None:
        return g.copy()

    r1, r2, c1, c2, top, bottom, left, right = found
    out = np.zeros_like(g)
    out[r1, c1 : c2 + 1] = top
    out[r2, c1 : c2 + 1] = bottom
    out[r1 + 1 : r2, c1] = left
    out[r1 + 1 : r2, c2] = right

    blocked = _border_mask(g.shape, r1, r2, c1, c2)
    side_for = {top: "top", bottom: "bottom", left: "left", right: "right"}
    for color, cells in _components(g, blocked):
        side = side_for.get(color)
        if side is None:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        if side == "top":
            dr, dc = r1 + 1 - min_r, 0
        elif side == "bottom":
            dr, dc = r2 - 1 - max_r, 0
        elif side == "left":
            dr, dc = 0, c1 + 1 - min_c
        else:
            dr, dc = 0, c2 - 1 - max_c

        for r, c in cells:
            nr, nc = r + dr, c + dc
            if r1 < nr < r2 and c1 < nc < c2:
                out[nr, nc] = color
    return out


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted({(r - min_r, c - min_c) for r, c in cells})


def _random_shape():
    kind = random.choice(("hline", "vline", "corner"))
    if kind == "hline":
        n = random.randint(2, 4)
        return [(0, c) for c in range(n)]
    if kind == "vline":
        n = random.randint(2, 4)
        return [(r, 0) for r in range(n)]

    v_len, h_len = random.choice(((2, 2), (2, 3), (3, 2)))
    v_dir = random.choice((-1, 1))
    h_dir = random.choice((-1, 1))
    cells = [(i * v_dir, 0) for i in range(v_len)]
    cells += [(0, j * h_dir) for j in range(h_len)]
    return _normalize(cells)


def _cells_from(shape, base_r, base_c):
    return [(base_r + r, base_c + c) for r, c in shape]


def _shape_size(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _clear_with_gap(cells, occupied, rows, cols):
    for r, c in cells:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        for nr in range(max(0, r - 1), min(rows, r + 2)):
            for nc in range(max(0, c - 1), min(cols, c + 2)):
                if (nr, nc) in occupied:
                    return False
    return True


def _draw_border(grid, r1, r2, c1, c2, colors):
    top, bottom, left, right = colors
    grid[r1, c1 : c2 + 1] = top
    grid[r2, c1 : c2 + 1] = bottom
    grid[r1 + 1 : r2, c1] = left
    grid[r1 + 1 : r2, c2] = right


def _border_cells(r1, r2, c1, c2):
    cells = set()
    for c in range(c1, c2 + 1):
        cells.add((r1, c))
        cells.add((r2, c))
    for r in range(r1 + 1, r2):
        cells.add((r, c1))
        cells.add((r, c2))
    return cells


def _try_add_piece(grid, side, color, box, input_occ, target_occ):
    rows, cols = grid.shape
    r1, r2, c1, c2 = box
    for _ in range(120):
        shape = _random_shape()
        shape_h, shape_w = _shape_size(shape)

        if side in ("top", "bottom"):
            lo_c, hi_c = c1 + 1, c2 - shape_w
            if lo_c > hi_c:
                continue
            target_c = random.randint(lo_c, hi_c)
            if side == "top":
                target_r = r1 + 1
                hi_source_r = r1 - shape_h - 1
                if hi_source_r < 0:
                    continue
                source_r = random.randint(0, hi_source_r)
            else:
                target_r = r2 - shape_h
                lo_source_r, hi_source_r = r2 + 2, rows - shape_h
                if lo_source_r > hi_source_r:
                    continue
                source_r = random.randint(lo_source_r, hi_source_r)
            source_c = target_c
        else:
            lo_r, hi_r = r1 + 1, r2 - shape_h
            if lo_r > hi_r:
                continue
            target_r = random.randint(lo_r, hi_r)
            if side == "left":
                target_c = c1 + 1
                hi_source_c = c1 - shape_w - 1
                if hi_source_c < 0:
                    continue
                source_c = random.randint(0, hi_source_c)
            else:
                target_c = c2 - shape_w
                lo_source_c, hi_source_c = c2 + 2, cols - shape_w
                if lo_source_c > hi_source_c:
                    continue
                source_c = random.randint(lo_source_c, hi_source_c)
            source_r = target_r

        target_cells = _cells_from(shape, target_r, target_c)
        if any(cell in target_occ for cell in target_cells):
            continue
        source_cells = _cells_from(shape, source_r, source_c)
        if not _clear_with_gap(source_cells, input_occ, rows, cols):
            continue

        for r, c in source_cells:
            grid[r, c] = color
            input_occ.add((r, c))
        target_occ.update(target_cells)
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(200):
        rows = random.randint(22, 30)
        cols = random.randint(22, 30)
        margin = 5
        h = random.randint(8, min(14, rows - 2 * margin))
        w = random.randint(8, min(14, cols - 2 * margin))
        r1 = random.randint(margin, rows - h - margin)
        c1 = random.randint(margin, cols - w - margin)
        r2, c2 = r1 + h - 1, c1 + w - 1

        colors = random.sample(range(1, 10), 4)
        top, bottom, left, right = colors
        grid = np.zeros((rows, cols), dtype=int)
        _draw_border(grid, r1, r2, c1, c2, colors)

        input_occ = _border_cells(r1, r2, c1, c2)
        target_occ = set()
        sides = [
            ("top", top),
            ("bottom", bottom),
            ("left", left),
            ("right", right),
        ]
        random.shuffle(sides)

        ok = True
        for side, color in sides:
            for _ in range(random.randint(1, 2)):
                if not _try_add_piece(
                    grid, side, color, (r1, r2, c1, c2), input_occ, target_occ
                ):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
