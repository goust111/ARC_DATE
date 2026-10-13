from collections import deque

import numpy as np


BG = 0
WIRE = 3
FRAME = 5
USABLE_COLORS = (1, 2, 4, 6, 7, 8, 9)


def _neighbors4(r, c, h, w):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _component_from(mask, start, seen):
    h, w = mask.shape
    q = deque([start])
    seen[start] = True
    cells = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return np.array(cells, dtype=int)


def _find_table(grid):
    gray = grid == FRAME
    starts = np.argwhere(gray)
    assert len(starts), "expected a gray table frame"

    start = tuple(starts[0])
    seen = np.zeros_like(gray, dtype=bool)
    frame_cells = _component_from(gray, start, seen)
    r0, c0 = frame_cells.min(axis=0)
    r1, c1 = frame_cells.max(axis=0)
    assert c1 - c0 + 1 == 5, "table must have three interior columns"
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "table frame must enclose rows"

    table_mask = np.zeros_like(gray, dtype=bool)
    table_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert np.all(grid[r0, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r1, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c0] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c1] == FRAME)

    mapping = {}
    for r in range(r0 + 1, r1):
        row = grid[r, c0 + 1 : c1]
        if np.all(row == BG):
            continue
        assert np.all(row != BG), "non-separator table rows must contain three colors"
        a, b, out_color = map(int, row)
        assert a != b, "table key colors must be distinct"
        mapping[frozenset((a, b))] = out_color

    assert mapping, "expected at least one table mapping row"
    return mapping, table_mask


def _wire_components(grid, table_mask):
    mask = (grid == WIRE) & ~table_mask
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if not seen[r, c]:
            comps.append(_component_from(mask, (r, c), seen))
    return comps


def _adjacent_endpoint_colors(grid, coords, table_mask):
    h, w = grid.shape
    colors = set()
    for r, c in coords:
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            if table_mask[nr, nc]:
                continue
            color = int(grid[nr, nc])
            if color not in (BG, WIRE, FRAME):
                colors.add(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each green wire by looking up the mixed color of its two endpoint blocks.

    Input structure:
    - The background is black, and a small top-left gray one-pixel rectangular
      frame contains a color table.
    - Each non-separator table row has three pixels: endpoint color A, endpoint
      color B, and the mixed output color C; table rows are separated by black rows.
    - Outside the table, green one-pixel-wide 4-connected wire paths each connect
      two differently colored 2x2 endpoint blocks, and different wires do not cross.

    Rule:
    - Read the table as an order-independent mapping from the color pair {A, B}
      to the output color C.
    - For every green wire component, collect the two endpoint colors adjacent to
      that component, look up their unordered pair in the table, and recolor all
      green pixels in that component to the looked-up mixed color.
    - Leave endpoint blocks, the table, and all background pixels unchanged.

    Key insight:
    - The gray-framed legend must first be decoded as a color-pair lookup table;
      then each wire's two endpoint blocks are traced through the green connected
      component so the correct unordered pair can drive the recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, table_mask = _find_table(g)
    out = g.copy()

    for coords in _wire_components(g, table_mask):
        endpoint_colors = _adjacent_endpoint_colors(g, coords, table_mask)
        assert len(endpoint_colors) == 2, "each wire must touch exactly two endpoint colors"
        key = frozenset(endpoint_colors)
        assert key in mapping, "wire endpoint pair must appear in the table"
        out[tuple(coords.T)] = mapping[key]

    return out


def _draw_table(grid, rows):
    n = len(rows)
    table_h = 2 * n + 1
    grid[:table_h, :5] = BG
    grid[0, :5] = FRAME
    grid[table_h - 1, :5] = FRAME
    grid[:table_h, 0] = FRAME
    grid[:table_h, 4] = FRAME
    for i, row in enumerate(rows):
        grid[1 + 2 * i, 1:4] = row


def _add_segment(cells, start, end):
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            if not cells or cells[-1] != (r0, c):
                cells.append((r0, c))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            if not cells or cells[-1] != (r, c0):
                cells.append((r, c0))


def _dogleg_path(start, end, pivot, orientation):
    cells = []
    sr, sc = start
    er, ec = end
    if orientation == "horizontal":
        p1 = (sr, pivot)
        p2 = (er, pivot)
    else:
        p1 = (pivot, sc)
        p2 = (pivot, ec)
    _add_segment(cells, start, p1)
    _add_segment(cells, p1, p2)
    _add_segment(cells, p2, end)
    return cells


def _composition(total, parts, rng):
    if total <= 0:
        return [0] * parts
    return list(rng.multinomial(int(total), [1.0 / parts] * parts))


def _paint_cells(grid, cells, color):
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _block_cells(r0, c0):
    return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]


def _random_specs(n_wires, rng):
    all_pairs = [
        (USABLE_COLORS[i], USABLE_COLORS[j])
        for i in range(len(USABLE_COLORS))
        for j in range(i + 1, len(USABLE_COLORS))
    ]
    rng.shuffle(all_pairs)
    specs = []
    for a, b in all_pairs[:n_wires]:
        out_choices = [c for c in USABLE_COLORS if c not in (a, b)]
        mixed = int(rng.choice(out_choices))
        specs.append((int(a), int(b), mixed))
    return specs


def _draw_horizontal_wires(grid, specs, table_h, rng):
    h, w = grid.shape
    band_h = 5
    start_r = table_h + 1
    extra = h - (start_r + band_h * len(specs))
    if extra < 0:
        return False
    gaps = _composition(extra, len(specs) + 1, rng)
    band_top = start_r + gaps[0]

    for i, (color_a, color_b, _mixed) in enumerate(specs):
        if i:
            band_top += band_h + gaps[i]

        left_c = int(rng.integers(1, max(2, w // 4 + 1)))
        right_min = max(left_c + 8, (2 * w) // 3)
        if right_min > w - 3:
            return False
        right_c = int(rng.integers(right_min, w - 2))

        left_r = band_top + int(rng.integers(1, 3))
        if rng.random() < 0.35:
            right_r = left_r
            left_attach = int(rng.integers(0, 2))
            right_attach = left_attach
        else:
            right_r = band_top + int(rng.integers(1, 3))
            left_attach = int(rng.integers(0, 2))
            right_attach = int(rng.integers(0, 2))

        start = (left_r + left_attach, left_c + 2)
        end = (right_r + right_attach, right_c - 1)
        if start[1] + 1 >= end[1]:
            return False
        pivot = int(rng.integers(start[1] + 1, end[1]))
        path = _dogleg_path(start, end, pivot, "horizontal")

        _paint_cells(grid, _block_cells(left_r, left_c), color_a)
        _paint_cells(grid, _block_cells(right_r, right_c), color_b)
        _paint_cells(grid, path, WIRE)

    return True


def _draw_vertical_wires(grid, specs, rng):
    h, w = grid.shape
    band_w = 5
    start_c = 6
    extra = w - (start_c + band_w * len(specs))
    if extra < 0:
        return False
    gaps = _composition(extra, len(specs) + 1, rng)
    band_left = start_c + gaps[0]

    for i, (color_a, color_b, _mixed) in enumerate(specs):
        if i:
            band_left += band_w + gaps[i]

        top_r = int(rng.integers(1, max(2, h // 4 + 1)))
        bottom_min = max(top_r + 8, (2 * h) // 3)
        if bottom_min > h - 2:
            return False
        bottom_r = int(rng.integers(bottom_min, h - 1))

        top_c = band_left + int(rng.integers(1, 3))
        if rng.random() < 0.35:
            bottom_c = top_c
            top_attach = int(rng.integers(0, 2))
            bottom_attach = top_attach
        else:
            bottom_c = band_left + int(rng.integers(1, 3))
            top_attach = int(rng.integers(0, 2))
            bottom_attach = int(rng.integers(0, 2))

        start = (top_r + 2, top_c + top_attach)
        end = (bottom_r - 1, bottom_c + bottom_attach)
        if start[0] + 1 >= end[0]:
            return False
        pivot = int(rng.integers(start[0] + 1, end[0]))
        path = _dogleg_path(start, end, pivot, "vertical")

        _paint_cells(grid, _block_cells(top_r, top_c), color_a)
        _paint_cells(grid, _block_cells(bottom_r, bottom_c), color_b)
        _paint_cells(grid, path, WIRE)

    return True


def _valid_generated(grid, expected_wires):
    try:
        mapping, table_mask = _find_table(grid)
        comps = _wire_components(grid, table_mask)
        if len(comps) != expected_wires:
            return False
        for coords in comps:
            colors = _adjacent_endpoint_colors(grid, coords, table_mask)
            if len(colors) != 2 or frozenset(colors) not in mapping:
                return False
        out = main(grid)
        return out.shape == grid.shape and max(out.shape) <= 30
    except AssertionError:
        return False


def _gen_attempt(rng):
    n_wires = int(rng.integers(2, 5))
    table_h = 2 * n_wires + 1
    orientation = str(rng.choice(("horizontal", "vertical")))
    specs = _random_specs(n_wires, rng)
    table_rows = specs.copy()
    rng.shuffle(table_rows)

    if orientation == "horizontal":
        min_h = table_h + 1 + 5 * n_wires
        if min_h > 30:
            return None
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(16, 31))
    else:
        min_w = 6 + 5 * n_wires
        if min_w > 30:
            return None
        h = int(rng.integers(max(16, table_h), 31))
        w = int(rng.integers(min_w, 31))

    grid = np.zeros((h, w), dtype=int)
    _draw_table(grid, table_rows)
    ok = (
        _draw_horizontal_wires(grid, specs, table_h, rng)
        if orientation == "horizontal"
        else _draw_vertical_wires(grid, specs, rng)
    )
    if not ok:
        return None
    if not _valid_generated(grid, n_wires):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wire color-table input")
