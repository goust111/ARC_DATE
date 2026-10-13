import random
from collections import deque

import numpy as np


MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _non_bg_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                    "colors": {int(grid[rr, cc]) for rr, cc in coords},
                }
            )
    return comps


def _find_priority_matrix(grid: np.ndarray, bg: int):
    candidates = []
    for comp in _non_bg_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 4:
            continue
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(np.count_nonzero(block != bg)) != int(comp["area"]):
            continue
        if int(np.count_nonzero(block == bg)) != 1 or int(block[0, 0]) != bg:
            continue
        if np.any(block[0, 1:] == bg) or np.any(block[1:, 0] == bg) or np.any(block[1:, 1:] == bg):
            continue
        row_colors = [int(x) for x in block[1:, 0]]
        col_colors = [int(x) for x in block[0, 1:]]
        if len(set(row_colors)) != side_h - 1 or len(set(col_colors)) != side_h - 1:
            continue
        if set(row_colors) != set(col_colors):
            continue
        inner = np.asarray(block[1:, 1:], dtype=int)
        ok = True
        for i, rc in enumerate(row_colors):
            for j, cc in enumerate(col_colors):
                if int(inner[i, j]) not in (rc, cc):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((c0, -r1, -comp["area"], (r0, c0, r1, c1), row_colors, col_colors, inner))
    assert candidates, "priority matrix not found"
    candidates.sort()
    _, _, _, bbox, row_colors, col_colors, inner = candidates[0]
    return bbox, row_colors, col_colors, inner


def _build_lookup(row_colors, col_colors, inner: np.ndarray):
    lookup = {}
    for i, rc in enumerate(row_colors):
        for j, cc in enumerate(col_colors):
            lookup[(rc, cc)] = int(inner[i, j])
    return lookup


def _scan_first_color(grid: np.ndarray, bg: int, blocked: np.ndarray, r: int, c: int, dr: int, dc: int):
    h, w = grid.shape
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w:
        if blocked[nr, nc]:
            return None
        color = int(grid[nr, nc])
        if color != bg:
            return color
        nr += dr
        nc += dc
    return None


def _find_crossings(grid: np.ndarray, bg: int, blocked: np.ndarray, valid_colors: set[int]):
    h, w = grid.shape
    crossings = []
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or int(grid[r, c]) != bg:
                continue
            left = _scan_first_color(grid, bg, blocked, r, c, 0, -1)
            right = _scan_first_color(grid, bg, blocked, r, c, 0, 1)
            if left is None or left != right or left not in valid_colors:
                continue
            up = _scan_first_color(grid, bg, blocked, r, c, -1, 0)
            down = _scan_first_color(grid, bg, blocked, r, c, 1, 0)
            if up is None or up != down or up not in valid_colors or up == left:
                continue
            crossings.append((r, c, left, up))
    return crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill blank crossings between colored straight paths using the priority matrix in the lower-left.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    bbox, row_colors, col_colors, inner = _find_priority_matrix(g, bg)
    lookup = _build_lookup(row_colors, col_colors, inner)
    path_colors = set(row_colors)

    r0, c0, r1, c1 = bbox
    blocked = np.zeros_like(g, dtype=bool)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    for color in path_colors:
        assert np.any((g == color) & ~blocked), "every matrix color must label a visible path"

    out = g.copy()
    out[blocked] = bg

    for r, c, h_color, v_color in _find_crossings(g, bg, blocked, path_colors):
        winner = lookup.get((h_color, v_color))
        if winner is None:
            winner = lookup.get((v_color, h_color))
        assert winner in (h_color, v_color), "priority lookup must choose one of the two crossing colors"
        out[r, c] = winner

    return out


def _sample_priority(colors):
    winners = {}
    for color in colors:
        winners[frozenset((color, color))] = color
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            winners[frozenset((a, b))] = random.choice((a, b))
    if len(colors) >= 3 and random.random() < 0.5:
        a, b, c = random.sample(colors, 3)
        winners[frozenset((a, b))] = a
        winners[frozenset((b, c))] = b
        winners[frozenset((a, c))] = c
    return winners


def _choose_spaced_positions(start: int, count: int, slots: int):
    base = start + random.randint(0, 1)
    values = [base + 2 * i for i in range(slots)]
    return sorted(random.sample(values, count)), values


def _validate_layout(grid: np.ndarray):
    bg = _get_bg(grid)
    bbox, row_colors, col_colors, inner = _find_priority_matrix(grid, bg)
    assert set(row_colors) == set(col_colors)

    r0, c0, r1, c1 = bbox
    blocked = np.zeros_like(grid, dtype=bool)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    path_colors = set(row_colors)
    path_cells = (grid != bg) & ~blocked
    assert np.any(path_cells)
    for color in path_colors:
        coords = np.argwhere((grid == color) & ~blocked)
        assert len(coords) >= 2
        rows = {int(r) for r, _ in coords}
        cols = {int(c) for _, c in coords}
        assert (len(rows) == 1) ^ (len(cols) == 1)

    crossings = _find_crossings(grid, bg, blocked, path_colors)
    assert crossings, "there must be at least one crossing"

    out = main(grid)
    lookup = _build_lookup(row_colors, col_colors, inner)
    assert np.all(out[blocked] == bg)
    for r, c, h_color, v_color in crossings:
        expected = lookup.get((h_color, v_color), lookup.get((v_color, h_color)))
        assert int(out[r, c]) == int(expected)

    unchanged = np.argwhere((grid != bg) & ~blocked)
    for r, c in unchanged:
        assert int(out[r, c]) == int(grid[r, c])


def _gen_attempt():
    n_paths = random.randint(3, 5)
    h_count = random.randint(1, n_paths - 1)
    v_count = n_paths - h_count
    side = n_paths + 1

    bg = random.randint(0, 9)
    colors = random.sample([c for c in range(10) if c != bg], n_paths)
    winners = _sample_priority(colors)

    matrix_c0 = random.randint(0, 1)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(0, 1)
    right_margin = random.randint(1, 3)

    h_rows, row_slots = _choose_spaced_positions(top_margin + 1, h_count, h_count + random.randint(1, 4))
    row_floor = top_margin
    row_ceiling = row_slots[-1] + 1 + random.randint(0, 2)

    v_cols, col_slots = _choose_spaced_positions(matrix_c0 + side + 2, v_count, v_count + random.randint(1, 4))
    col_left = matrix_c0 + side + 1
    col_right = col_slots[-1] + 1 + random.randint(0, 3)

    matrix_r0 = row_ceiling + 2 + random.randint(0, 1)
    h = matrix_r0 + side + bottom_margin
    w = col_right + 1 + right_margin

    grid = np.full((h, w), bg, dtype=int)

    h_specs = []
    for color, row in zip(colors[:h_count], h_rows):
        start = random.randint(col_left, v_cols[0] - 1)
        end = random.randint(v_cols[-1] + 1, col_right)
        h_specs.append((color, row, start, end))
        grid[row, start : end + 1] = color

    v_specs = []
    for color, col in zip(colors[h_count:], v_cols):
        start = random.randint(row_floor, h_rows[0] - 1)
        end = random.randint(h_rows[-1] + 1, row_ceiling)
        v_specs.append((color, col, start, end))
        grid[start : end + 1, col] = color

    for _, row, _, _ in h_specs:
        for _, col, _, _ in v_specs:
            grid[row, col] = bg

    row_order = random.sample(colors, len(colors))
    col_order = random.sample(colors, len(colors))
    grid[matrix_r0, matrix_c0] = bg
    for i, color in enumerate(row_order, start=1):
        grid[matrix_r0 + i, matrix_c0] = color
    for j, color in enumerate(col_order, start=1):
        grid[matrix_r0, matrix_c0 + j] = color
    for i, r_color in enumerate(row_order, start=1):
        for j, c_color in enumerate(col_order, start=1):
            grid[matrix_r0 + i, matrix_c0 + j] = winners[frozenset((r_color, c_color))]

    _validate_layout(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid crossing-priority input")
