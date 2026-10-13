import random

import numpy as np


BACKGROUND = 0


def _ordered_pair(a, b):
    if a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1]):
        return a, b
    return b, a


def _walk_segment(a, b):
    r, c = a
    end_r, end_c = b
    dr = (end_r > r) - (end_r < r)
    dc = (end_c > c) - (end_c < c)
    cells = []
    while (r, c) != (end_r, end_c):
        cells.append((r, c))
        r += dr
        c += dc
    cells.append((end_r, end_c))
    return cells


def _l_path(a, b, first_axis):
    start, end = _ordered_pair(a, b)
    if first_axis == "h":
        corner = (start[0], end[1])
    else:
        corner = (end[0], start[1])
    return _walk_segment(start, corner) + _walk_segment(corner, end)[1:]


def _detect_bands(grid):
    h, w = grid.shape
    row_bands = {}
    col_bands = {}

    for r in range(h):
        color = int(grid[r, 0])
        if color != BACKGROUND and np.all(grid[r, :] == color):
            row_bands[r] = color
    for c in range(w):
        color = int(grid[0, c])
        if color != BACKGROUND and np.all(grid[:, c] == color):
            col_bands[c] = color

    changed = True
    while changed:
        changed = False
        free_cols = [c for c in range(w) if c not in col_bands]
        if free_cols:
            for r in range(h):
                if r in row_bands:
                    continue
                vals = grid[r, free_cols]
                color = int(vals[0])
                if color != BACKGROUND and np.all(vals == color):
                    row_bands[r] = color
                    changed = True

        free_rows = [r for r in range(h) if r not in row_bands]
        if free_rows:
            for c in range(w):
                if c in col_bands:
                    continue
                vals = grid[free_rows, c]
                color = int(vals[0])
                if color != BACKGROUND and np.all(vals == color):
                    col_bands[c] = color
                    changed = True

    return row_bands, col_bands


def _band_positions(shape, row_bands, col_bands):
    h, w = shape
    cells = set()
    for r in row_bands:
        for c in range(w):
            cells.add((r, c))
    for c in col_bands:
        for r in range(h):
            cells.add((r, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Connect matched marker pairs with L paths that inherit dye-band colors.

    Input structure:
    The grid has a black background, several pairs of same-colored single-cell
    markers, and one-pixel full-row or full-column dye bands whose colors are
    distinct from all marker colors. Markers are not placed on dye bands, and
    the two markers in each pair are on different rows and different columns.

    Rule:
    For each marker pair, draw the L path from the upper marker to the lower
    marker, trying the horizontal leg before the vertical leg. If that trial
    path would pass through any other marker, use the vertical-first L path
    instead. Path cells begin in the marker color. Whenever the path crosses a
    dye-band cell, that band cell is preserved, and the next path cell onward
    is drawn in the band color until another band is crossed. All original
    markers and dye-band cells remain unchanged.

    Key insight:
    The path geometry is fixed by the paired marker positions, while the final
    drawn colors are segmented by the ordered dye bands encountered along that
    path.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    h, w = grid.shape

    row_bands, col_bands = _detect_bands(grid)
    band_cells = _band_positions((h, w), row_bands, col_bands)

    by_color = {}
    all_marker_cells = set()
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            color = int(grid[r, c])
            if color == BACKGROUND or pos in band_cells:
                continue
            by_color.setdefault(color, []).append(pos)
            all_marker_cells.add(pos)

    pairs = []
    for color, cells in by_color.items():
        if len(cells) != 2:
            continue
        a, b = cells
        if a[0] == b[0] or a[1] == b[1]:
            continue
        lo, hi = _ordered_pair(a, b)
        pairs.append((lo, hi, color))

    pairs.sort(key=lambda item: (item[0], item[1]))

    for a, b, color in pairs:
        endpoints = {a, b}
        other_markers = all_marker_cells - endpoints
        horizontal_path = _l_path(a, b, "h")
        first_axis = "v" if any(p in other_markers for p in horizontal_path[1:-1]) else "h"
        path = _l_path(a, b, first_axis)

        current_color = color
        for pos in path:
            r, c = pos
            if pos in all_marker_cells:
                continue
            if pos in band_cells:
                band_color = int(grid[r, c])
                if band_color != BACKGROUND:
                    current_color = band_color
                continue
            out[r, c] = current_color

    return out


def _is_open_cell(pos, forbidden_rows, forbidden_cols, used):
    r, c = pos
    return r not in forbidden_rows and c not in forbidden_cols and pos not in used


def _random_cell(h, w, forbidden_rows, forbidden_cols, used):
    for _ in range(800):
        pos = (random.randrange(h), random.randrange(w))
        if not _is_open_cell(pos, forbidden_rows, forbidden_cols, used):
            continue
        if any(abs(pos[0] - r) + abs(pos[1] - c) <= 1 for r, c in used):
            continue
        return pos
    return None


def _sample_band_indices(size, count):
    choices = list(range(1, size - 1))
    for _ in range(200):
        picked = sorted(random.sample(choices, count))
        if all(b - a > 1 for a, b in zip(picked, picked[1:])):
            return picked
    return random.sample(choices, count)


def _path_crosses_marker(path, endpoints, markers):
    others = markers - endpoints
    return any(pos in others for pos in path[1:-1])


def _layout_ok(pairs, forbidden_rows, forbidden_cols, require_branch):
    markers = {p for pair in pairs for p in pair}
    if len(markers) != 2 * len(pairs):
        return False

    branch_count = 0
    occupied_path_cells = set()

    for a, b in pairs:
        if a[0] == b[0] or a[1] == b[1]:
            return False

        endpoints = {a, b}
        horizontal_path = _l_path(a, b, "h")
        has_branch = _path_crosses_marker(horizontal_path, endpoints, markers)
        if has_branch:
            branch_count += 1
        chosen_path = _l_path(a, b, "v" if has_branch else "h")

        if _path_crosses_marker(chosen_path, endpoints, markers):
            return False

        for pos in chosen_path[1:-1]:
            if pos in markers:
                continue
            if pos[0] in forbidden_rows or pos[1] in forbidden_cols:
                continue
            if pos in occupied_path_cells:
                return False
            occupied_path_cells.add(pos)

    return branch_count > 0 if require_branch else branch_count == 0


def _build_pairs(h, w, forbidden_rows, forbidden_cols, pair_count, require_branch):
    pairs = []
    used = set()

    if require_branch:
        made_branch = False
        for _ in range(500):
            used.clear()
            a = _random_cell(h, w, forbidden_rows, forbidden_cols, used)
            if a is None:
                continue
            used.add(a)
            b = _random_cell(h, w, forbidden_rows, forbidden_cols, used)
            if b is None or a[0] == b[0] or a[1] == b[1]:
                continue

            trial_path = _l_path(a, b, "h")
            candidates = []
            for pos in trial_path[1:-1]:
                if not _is_open_cell(pos, forbidden_rows, forbidden_cols, {a, b}):
                    continue
                if abs(pos[0] - a[0]) + abs(pos[1] - a[1]) <= 1:
                    continue
                if abs(pos[0] - b[0]) + abs(pos[1] - b[1]) <= 1:
                    continue
                candidates.append(pos)
            if not candidates:
                continue

            blocker = random.choice(candidates)
            used = {a, b, blocker}
            partner = None
            for _ in range(500):
                p = _random_cell(h, w, forbidden_rows, forbidden_cols, used)
                if p is not None and p[0] != blocker[0] and p[1] != blocker[1]:
                    partner = p
                    break
            if partner is None:
                continue

            pairs = [(a, b), (blocker, partner)]
            used.add(partner)
            made_branch = True
            break
        if not made_branch:
            return None

    while len(pairs) < pair_count:
        a = _random_cell(h, w, forbidden_rows, forbidden_cols, used)
        if a is None:
            return None
        used.add(a)

        b = None
        for _ in range(500):
            candidate = _random_cell(h, w, forbidden_rows, forbidden_cols, used)
            if candidate is not None and candidate[0] != a[0] and candidate[1] != a[1]:
                b = candidate
                break
        if b is None:
            return None
        used.add(b)
        pairs.append((a, b))

    if not _layout_ok(pairs, forbidden_rows, forbidden_cols, require_branch):
        return None
    return pairs


def generate_input() -> np.ndarray:
    for _ in range(2000):
        h = random.randint(12, 30)
        w = random.randint(12, 30)

        colors = list(range(1, 10))
        random.shuffle(colors)

        band_count = random.randint(1, 3)
        pair_count = random.randint(2, min(4, 9 - band_count))
        band_colors = colors[:band_count]
        marker_colors = colors[band_count : band_count + pair_count]

        grid = np.zeros((h, w), dtype=int)
        orientation = random.choice(["row", "col"])
        if orientation == "row":
            band_indices = _sample_band_indices(h, band_count)
            forbidden_rows = set(band_indices)
            forbidden_cols = set()
            for r, color in zip(band_indices, band_colors):
                grid[r, :] = color
        else:
            band_indices = _sample_band_indices(w, band_count)
            forbidden_rows = set()
            forbidden_cols = set(band_indices)
            for c, color in zip(band_indices, band_colors):
                grid[:, c] = color

        require_branch = random.choice([False, True])
        pairs = _build_pairs(h, w, forbidden_rows, forbidden_cols, pair_count, require_branch)
        if pairs is None:
            continue

        for color, (a, b) in zip(marker_colors, pairs):
            grid[a] = color
            grid[b] = color

        return grid

    raise RuntimeError("could not generate a valid input")
