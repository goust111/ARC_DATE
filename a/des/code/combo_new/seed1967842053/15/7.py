import numpy as np


ORANGE = 7


def _most_common_color(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_separator_color(grid, bg):
    best_color = None
    best_score = -1
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, ORANGE):
            continue
        rows = np.flatnonzero(np.all(grid == color, axis=1))
        cols = np.flatnonzero(np.all(grid == color, axis=0))
        if len(rows) >= 2 and len(cols) >= 2:
            score = len(rows) + len(cols)
            if score > best_score:
                best_color = color
                best_score = score
    return 5 if best_color is None else best_color


def _bands(indices):
    if len(indices) == 0:
        return []
    result = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            result.append((start, prev + 1))
            start = prev = value
    result.append((start, prev + 1))
    return result


def _cell_ranges(line_bands):
    ranges = []
    for left, right in zip(line_bands, line_bands[1:]):
        start, end = left[1], right[0]
        if start < end:
            ranges.append((start, end))
    return ranges


def _trim(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return mask[:0, :0]
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0) + 1
    return mask[r0:r1, c0:c1]


def _shape_type(mask):
    crop = _trim(mask)
    if crop.size == 0:
        return (False, False)
    return (
        bool(np.array_equal(crop, crop[::-1, :])),
        bool(np.array_equal(crop, crop[:, ::-1])),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each table object with the color keyed by its reflection symmetry.

    Input structure:
    The grid has the most common color as background, full grey separator rows and
    columns forming equal-sized rectangular cells, and one orange object inside
    each cell.  The leftmost cell column is a legend: each legend cell also has a
    single non-background, non-grey, non-orange color marker at its upper-left
    interior corner.

    Rule:
    Read the legend column by classifying each legend orange object as having no
    reflection symmetry, horizontal symmetry only, vertical symmetry only, or
    both horizontal and vertical symmetry.  Associate that symmetry type with the
    marker color in the same legend cell.  For every non-legend cell, classify
    the orange object in the same way and recolor all of its orange pixels with
    the matching legend marker color.  Leave the legend column, separator lines,
    background, and all non-orange pixels unchanged.

    Key insight:
    The leftmost column is a dictionary from shape symmetry type to color; the
    task is to classify new objects by reflection symmetry rather than by exact
    geometry.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    bg = _most_common_color(grid)
    sep = _find_separator_color(grid, bg)

    row_lines = np.flatnonzero(np.all(grid == sep, axis=1))
    col_lines = np.flatnonzero(np.all(grid == sep, axis=0))
    row_ranges = _cell_ranges(_bands(row_lines))
    col_ranges = _cell_ranges(_bands(col_lines))
    if not row_ranges or len(col_ranges) < 2:
        return out

    legend_col = col_ranges[0]
    mapping = {}
    for r0, r1 in row_ranges:
        c0, c1 = legend_col
        cell = grid[r0:r1, c0:c1]
        marker_mask = (cell != bg) & (cell != sep) & (cell != ORANGE)
        marker_pts = np.argwhere(marker_mask)
        orange_mask = cell == ORANGE
        if len(marker_pts) == 0 or not np.any(orange_mask):
            continue
        mr, mc = marker_pts[0]
        mapping[_shape_type(orange_mask)] = int(cell[mr, mc])

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges[1:]:
            cell = grid[r0:r1, c0:c1]
            orange_mask = cell == ORANGE
            if not np.any(orange_mask):
                continue
            color = mapping.get(_shape_type(orange_mask))
            if color is not None:
                out_cell = out[r0:r1, c0:c1]
                out_cell[orange_mask] = color
    return out


def _is_connected(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(int(x) for x in pts[0])
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected(h, w, rng, start=None):
    if start is None:
        start = (int(rng.integers(h)), int(rng.integers(w)))
    target = int(rng.integers(4, min(h * w, 10) + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[start] = True
    while int(mask.sum()) < target:
        candidates = []
        for r, c in np.argwhere(mask):
            r, c = int(r), int(c)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    candidates.append((nr, nc))
        if not candidates:
            break
        nr, nc = candidates[int(rng.integers(len(candidates)))]
        mask[nr, nc] = True
    return mask


def _symmetrize(mask, kind):
    out = mask.copy()
    if kind[0]:
        out |= out[::-1, :]
    if kind[1]:
        out |= out[:, ::-1]
    return out


def _central_start(h, w, kind, rng):
    if kind == (True, True):
        return ((h - 1) // 2, (w - 1) // 2)
    if kind == (True, False):
        return ((h - 1) // 2, int(rng.integers(w)))
    if kind == (False, True):
        return (int(rng.integers(h)), (w - 1) // 2)
    return None


def _make_shape(kind, max_h, max_w, rng):
    max_h = min(int(max_h), 5)
    max_w = min(int(max_w), 5)
    for _ in range(300):
        h = int(rng.integers(3, max_h + 1))
        w = int(rng.integers(3, max_w + 1))
        if kind == (False, False):
            mask = _random_connected(h, w, rng)
        else:
            seed = _random_connected(h, w, rng, _central_start(h, w, kind, rng))
            mask = _symmetrize(seed, kind)
        mask = _trim(mask)
        if _shape_type(mask) == kind and _is_connected(mask):
            return mask
    templates = {
        (False, False): np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        (True, False): np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
        (False, True): np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
        (True, True): np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    }
    return templates[kind].copy()


def _place_shape(cell, mask, rng, reserved=None):
    h, w = mask.shape
    ch, cw = cell.shape
    positions = []
    for r0 in range(ch - h + 1):
        for c0 in range(cw - w + 1):
            if reserved is not None:
                rr, cc = reserved
                mr, mc = rr - r0, cc - c0
                if 0 <= mr < h and 0 <= mc < w and mask[mr, mc]:
                    continue
            positions.append((r0, c0))
    if not positions:
        raise ValueError("shape cannot be placed in this cell")
    r0, c0 = positions[int(rng.integers(len(positions)))]
    cell[r0:r0 + h, c0:c0 + w][mask] = ORANGE


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    kinds = [(False, False), (True, False), (False, True), (True, True)]
    rows = 4
    cols = int(rng.integers(3, 5))
    cell_h = int(rng.integers(5, 7))
    cell_w = int(rng.integers(5, 7))
    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1

    bg = 0
    sep = 5
    grid = np.full((height, width), bg, dtype=int)
    row_lines = [r * (cell_h + 1) for r in range(rows + 1)]
    col_lines = [c * (cell_w + 1) for c in range(cols + 1)]
    for r in row_lines:
        grid[r, :] = sep
    for c in col_lines:
        grid[:, c] = sep

    palette = np.array([1, 2, 3, 4, 6, 8, 9])
    marker_colors = rng.choice(palette, size=4, replace=False)
    color_by_kind = {kind: int(color) for kind, color in zip(kinds, marker_colors)}
    legend_order = [kinds[int(i)] for i in rng.permutation(len(kinds))]

    for row, kind in enumerate(legend_order):
        r0, r1 = row_lines[row] + 1, row_lines[row + 1]
        c0, c1 = col_lines[0] + 1, col_lines[1]
        cell = grid[r0:r1, c0:c1]
        cell[0, 0] = color_by_kind[kind]
        for _ in range(50):
            mask = _make_shape(kind, cell_h, cell_w, rng)
            try:
                _place_shape(cell, mask, rng, reserved=(0, 0))
                break
            except ValueError:
                continue

    for row in range(rows):
        for col in range(1, cols):
            kind = kinds[int(rng.integers(len(kinds)))]
            r0, r1 = row_lines[row] + 1, row_lines[row + 1]
            c0, c1 = col_lines[col] + 1, col_lines[col + 1]
            cell = grid[r0:r1, c0:c1]
            mask = _make_shape(kind, cell_h, cell_w, rng)
            _place_shape(cell, mask, rng)

    return grid
