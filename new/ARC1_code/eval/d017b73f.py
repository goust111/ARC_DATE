import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste
import random


def _column_groups(g: np.ndarray, bg: int) -> list[np.ndarray]:
    nonempty = np.any(g != bg, axis=0)
    groups = []
    c = 0
    while c < g.shape[1]:
        if not nonempty[c]:
            c += 1
            continue
        c0 = c
        while c < g.shape[1] and nonempty[c]:
            c += 1
        group = g[:, c0:c].copy()
        assert len(set(map(int, group[group != bg]))) == 1
        groups.append(group)
    assert groups
    return groups


def _endpoint_row(group: np.ndarray, bg: int, side: str) -> int:
    h, w = group.shape
    assert side in {"left", "right"}
    c = 0 if side == "left" else w - 1
    mask = group != bg
    rows = []
    for r in range(h):
        if not mask[r, c]:
            continue
        degree = 0
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                degree += 1
        if degree == 1:
            rows.append(r)
    assert len(rows) == 1
    return rows[0]


def _shift_group(group: np.ndarray, bg: int, dr: int) -> np.ndarray:
    h, _ = group.shape
    shifted = np.full_like(group, bg)
    coords = np.argwhere(group != bg)
    for r, c in coords:
        nr = r + dr
        assert 0 <= nr < h
        shifted[nr, c] = group[r, c]
    return shifted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove blank separator columns and slide each vertical object group so adjacent connector endpoints touch.

    Input structure
    A short strip contains monochrome path-like object groups, each occupying a contiguous run of non-background columns; runs are separated by all-background columns. Each group has a unique degree-1 foreground endpoint on both its left and right outer edge.

    Rule
    Delete the all-background separator columns. Keep the first group in its original vertical position, then process the remaining groups from left to right: shift each group vertically, without rotation or recoloring, so its left endpoint is on the same row as the previous placed group's right endpoint, and paste it immediately to the right.

    Key insight
    The intended contacts are the degree-1 edge endpoints, not every possible foreground edge overlap, which makes the packed strip a single connected chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    groups = _column_groups(g, bg)

    h = g.shape[0]
    out = np.full((h, sum(group.shape[1] for group in groups)), bg, dtype=int)
    c = 0

    first = groups[0]
    out[:, : first.shape[1]] = first
    prev_right_row = _endpoint_row(first, bg, "right")
    c += first.shape[1]

    for group in groups[1:]:
        dr = prev_right_row - _endpoint_row(group, bg, "left")
        shifted = _shift_group(group, bg, dr)
        assert out[prev_right_row, c - 1] != bg
        assert shifted[prev_right_row, 0] != bg
        out[:, c : c + group.shape[1]] = shifted
        prev_right_row = _endpoint_row(shifted, bg, "right")
        c += group.shape[1]

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, group structure, input perturbation, and retry budget.
GEN_HEIGHT, BG_COLOR = 3, 0
COLOR_MIN, COLOR_MAX = 1, 9
GROUP_COUNT_CHOICES, GROUP_WIDTH_CHOICES = [3, 4], [2, 3]
GROUP_CELL_CHOICES = [2, 3, 4]
NEIGHBOR_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INPUT_SHIFT_CHOICES = range(-2, 3)
MAX_GENERATION_ATTEMPTS = 300


def _gen_neighbors(r, c, width):
    for dr, dc in NEIGHBOR_DELTAS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GEN_HEIGHT and 0 <= nc < width:
            yield nr, nc


def _gen_degree(cells, r, c, width):
    return sum((nr, nc) in cells for nr, nc in _gen_neighbors(r, c, width))


def _gen_path_is_valid(path, width):
    cells = set(path)
    degrees = [_gen_degree(cells, r, c, width) for r, c in cells]
    if any(deg < 1 or deg > 2 for deg in degrees):
        return False
    if degrees.count(1) != 2:
        return False

    left_endpoints, right_endpoints = [], []
    for r, c in cells:
        deg = _gen_degree(cells, r, c, width)
        if deg == 1 and c == 0:
            left_endpoints.append(r)
        if deg == 1 and c == width - 1:
            right_endpoints.append(r)
    return len(left_endpoints) == 1 and len(right_endpoints) == 1


def _gen_paths(width, n_cells, left_row, right_row):
    start = (left_row, 0)
    goal = (right_row, width - 1)
    paths = []

    def dfs(path):
        if len(path) == n_cells:
            if path[-1] == goal and _gen_path_is_valid(path, width):
                paths.append(tuple(path))
            return

        remaining_steps = n_cells - len(path) - 1
        for nr, nc in _gen_neighbors(*path[-1], width):
            if (nr, nc) in path:
                continue
            if abs(nr - goal[0]) + abs(nc - goal[1]) > remaining_steps:
                continue
            dfs(path + [(nr, nc)])

    dfs([start])
    return paths


def _gen_render_path(width, path, color):
    group = np.zeros((GEN_HEIGHT, width), dtype=int)
    rows, cols = zip(*path)
    group[rows, cols] = color
    return group


def _gen_random_group(left_row, color):
    n_cells = random.choice(GROUP_CELL_CHOICES)
    options = []
    for width in GROUP_WIDTH_CHOICES:
        for right_row in range(GEN_HEIGHT):
            for path in _gen_paths(width, n_cells, left_row, right_row):
                options.append((width, path))

    width, path = random.choice(options)
    return _gen_render_path(width, path, color)


def _gen_endpoint_row(group, side):
    col = 0 if side == "left" else group.shape[1] - 1
    rows = []
    for r in range(GEN_HEIGHT):
        if group[r, col] == BG_COLOR:
            continue
        deg = sum(
            group[nr, nc] != BG_COLOR
            for nr, nc in _gen_neighbors(r, col, group.shape[1])
        )
        if deg == 1:
            rows.append(r)
    assert len(rows) == 1
    return rows[0]


def _gen_shift(group, dr):
    shifted = np.zeros_like(group)
    rows, cols = np.nonzero(group != BG_COLOR)
    new_rows = rows + dr
    if np.any((new_rows < 0) | (new_rows >= GEN_HEIGHT)):
        return None
    shifted[new_rows, cols] = group[rows, cols]
    return shifted


def _gen_render_groups(groups, separator_width):
    width = sum(group.shape[1] for group in groups) + separator_width * (len(groups) - 1)
    canvas = np.zeros((GEN_HEIGHT, width), dtype=int)
    col = 0
    for i, group in enumerate(groups):
        paste(canvas, group, 0, col, transparent=BG_COLOR)
        col += group.shape[1]
        if i < len(groups) - 1:
            col += separator_width
    return canvas


def _gen_build_output_groups(colors):
    groups = []
    left_row = random.randrange(GEN_HEIGHT)
    for color in colors:
        group = _gen_random_group(left_row, color)
        groups.append(group)
        left_row = _gen_endpoint_row(group, "right")
    return groups


def _gen_valid_shifts(group):
    rows = np.nonzero(group != BG_COLOR)[0]
    return [
        dr
        for dr in INPUT_SHIFT_CHOICES
        if rows.min() + dr >= 0 and rows.max() + dr < GEN_HEIGHT
    ]


def _gen_make_input_groups(groups):
    input_groups = [groups[0]]
    for group in groups[1:]:
        shifted = _gen_shift(group, random.choice(_gen_valid_shifts(group)))
        if shifted is None:
            return None
        input_groups.append(shifted)
    return input_groups


def _gen_attempt():
    n_groups = random.choice(GROUP_COUNT_CHOICES)
    colors = [random.randint(COLOR_MIN, COLOR_MAX) for _ in range(n_groups)]
    groups = _gen_build_output_groups(colors)
    out = _gen_render_groups(groups, separator_width=0)

    if len(find_components(out, bg=BG_COLOR, connectivity=4, monochromatic=False)) != 1:
        return None

    input_groups = _gen_make_input_groups(groups)
    if input_groups is None:
        return None
    inp = _gen_render_groups(input_groups, separator_width=1)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d017b73f input")
