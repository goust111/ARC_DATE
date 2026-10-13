from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
PairSpec = tuple[int, Coord, Coord]


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PAIR_COUNT_RANGE = (3, 5)
HEIGHT_MAX = 24
WIDTH_MAX = 24
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 120


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sort_pair(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    return (a, b) if a <= b else (b, a)


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _line_cells(start: Coord, stop: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = stop
    assert r0 == r1 or c0 == c1, "line segment must be horizontal or vertical"

    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]

    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _pair_kind(a_raw: Coord, b_raw: Coord) -> str:
    a, b = _sort_pair(a_raw, b_raw)
    if a[0] == b[0]:
        return "row"
    if a[1] == b[1]:
        return "col"
    return "diag"


def _pair_segments(a_raw: Coord, b_raw: Coord) -> tuple[list[Coord], list[Coord]]:
    a, b = _sort_pair(a_raw, b_raw)
    if a[0] == b[0]:
        return _line_cells(a, b), []
    if a[1] == b[1]:
        return [], _line_cells(a, b)

    elbow = (a[0], b[1])
    return _line_cells(a, elbow), _line_cells(elbow, b)


def _is_isolated(grid: np.ndarray, pos: Coord, color: int) -> bool:
    h, w = grid.shape
    r, c = pos
    for rr, cc in _neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and (rr, cc) != pos and int(grid[rr, cc]) == int(color):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair with a horizontal-then-vertical orthogonal path.

    Input structure:
    - The dominant color is the background.
    - Every foreground color appears exactly twice as two isolated single-pixel endpoint markers.
    - For each color, the two markers may share a row, share a column, or require an L-shaped path.

    Rule:
    - Read each same-colored pair in row-major order.
    - If the pair is not aligned, draw a horizontal segment on the first endpoint's row until the second endpoint's column, then draw a vertical segment on that column down to the second endpoint.
    - If the pair already shares a row or a column, draw the direct horizontal or vertical segment instead.
    - Combine all paths with a fixed layering rule: horizontal segments are painted first, vertical segments are painted afterward, and endpoint pixels are restored last so vertical-over-horizontal crossings keep the vertical color while markers stay unchanged.

    Key insight:
    - The hidden overlap rule is directional rather than color-order-based: every crossing resolves the same way because vertical segments always override horizontal ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground endpoint pair"

    pairs: dict[int, tuple[Coord, Coord]] = {}
    for color in colors:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == int(color))]
        assert len(coords) == 2, "each foreground color must appear exactly twice"
        a, b = _sort_pair(coords[0], coords[1])
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2, "endpoints must stay isolated"
        assert _is_isolated(g, a, color) and _is_isolated(g, b, color), "markers must be isolated pixels"
        pairs[color] = (a, b)

    out = np.full_like(g, bg)
    for color, (a, b) in pairs.items():
        out[a] = color
        out[b] = color

    for color, (a, b) in pairs.items():
        horizontal, _vertical = _pair_segments(a, b)
        for r, c in horizontal:
            out[r, c] = color

    for color, (a, b) in pairs.items():
        _horizontal, vertical = _pair_segments(a, b)
        for r, c in vertical:
            out[r, c] = color

    for color, (a, b) in pairs.items():
        out[a] = color
        out[b] = color

    return out


def _free_rows(height: int, blocked_rows: set[int]) -> list[int]:
    return [r for r in range(1, height - 1) if r not in blocked_rows]


def _free_cols(width: int, blocked_cols: set[int]) -> list[int]:
    return [c for c in range(1, width - 1) if c not in blocked_cols]


def _sample_row_pair(color: int, height: int, width: int, blocked_rows: set[int], blocked_cols: set[int]) -> PairSpec | None:
    rows = _free_rows(height, blocked_rows)
    cols = _free_cols(width, blocked_cols)
    if not rows or len(cols) < 2:
        return None

    for _ in range(PAIR_ATTEMPTS):
        row = random.choice(rows)
        c0, c1 = sorted(random.sample(cols, 2))
        if c1 - c0 < 2:
            continue
        return int(color), (row, c0), (row, c1)
    return None


def _sample_col_pair(color: int, height: int, width: int, blocked_rows: set[int], blocked_cols: set[int]) -> PairSpec | None:
    rows = _free_rows(height, blocked_rows)
    cols = _free_cols(width, blocked_cols)
    if len(rows) < 2 or not cols:
        return None

    for _ in range(PAIR_ATTEMPTS):
        col = random.choice(cols)
        r0, r1 = sorted(random.sample(rows, 2))
        if r1 - r0 < 2:
            continue
        return int(color), (r0, col), (r1, col)
    return None


def _sample_diag_pair(color: int, height: int, width: int, blocked_rows: set[int], blocked_cols: set[int]) -> PairSpec | None:
    rows = _free_rows(height, blocked_rows)
    cols = _free_cols(width, blocked_cols)
    if len(rows) < 2 or len(cols) < 2:
        return None

    for _ in range(PAIR_ATTEMPTS):
        r_top = random.choice(rows)
        lower_rows = [r for r in rows if r >= r_top + 2]
        if not lower_rows:
            continue

        v_col = random.choice(cols)
        left_cols = [c for c in cols if c <= v_col - 2]
        right_cols = [c for c in cols if c >= v_col + 2]
        start_options: list[int] = []
        if left_cols:
            start_options.append(random.choice(left_cols))
        if right_cols:
            start_options.append(random.choice(right_cols))
        if not start_options:
            continue

        r_bottom = random.choice(lower_rows)
        c_start = random.choice(start_options)
        return int(color), (r_top, c_start), (r_bottom, v_col)
    return None


def _render_input(specs: list[PairSpec], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for color, a, b in specs:
        grid[a] = int(color)
        grid[b] = int(color)
    return grid


def _specs_valid(specs: list[PairSpec], height: int, width: int) -> bool:
    if len({int(color) for color, _a, _b in specs}) != len(specs):
        return False

    endpoint_set: set[Coord] = set()
    horizontal_sets: list[set[Coord]] = []
    vertical_sets: list[set[Coord]] = []
    pair_endpoints: list[set[Coord]] = []

    for _color, a_raw, b_raw in specs:
        a, b = _sort_pair(a_raw, b_raw)
        if not (0 <= a[0] < height and 0 <= a[1] < width and 0 <= b[0] < height and 0 <= b[1] < width):
            return False
        if a == b:
            return False
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 2:
            return False
        if a in endpoint_set or b in endpoint_set:
            return False
        endpoint_set.add(a)
        endpoint_set.add(b)

        horizontal, vertical = _pair_segments(a, b)
        horizontal_sets.append(set(horizontal))
        vertical_sets.append(set(vertical))
        pair_endpoints.append({a, b})

    crossing_found = False
    for i in range(len(specs)):
        interior_i = (horizontal_sets[i] | vertical_sets[i]) - pair_endpoints[i]
        if interior_i & (endpoint_set - pair_endpoints[i]):
            return False

        for j in range(i + 1, len(specs)):
            if horizontal_sets[i] & horizontal_sets[j]:
                return False
            if vertical_sets[i] & vertical_sets[j]:
                return False

            hv_cross = ((horizontal_sets[i] - pair_endpoints[i]) & (vertical_sets[j] - pair_endpoints[j])) | (
                (vertical_sets[i] - pair_endpoints[i]) & (horizontal_sets[j] - pair_endpoints[j])
            )
            hv_cross -= endpoint_set
            if hv_cross:
                crossing_found = True

    if not crossing_found:
        return False

    painted: set[Coord] = set()
    for idx in range(len(specs)):
        painted |= horizontal_sets[idx] | vertical_sets[idx]
    if len(painted) > int(0.30 * height * width):
        return False

    return True


def _blocked_axes(specs: list[PairSpec]) -> tuple[set[int], set[int]]:
    rows: set[int] = set()
    cols: set[int] = set()
    for _color, a, b in specs:
        rows.add(int(a[0]))
        rows.add(int(b[0]))
        cols.add(int(a[1]))
        cols.add(int(b[1]))
    return rows, cols


def _sample_crossing_pair_block(height: int, width: int, row_color: int, col_color: int) -> list[PairSpec] | None:
    if height < 8 or width < 8:
        return None

    for _ in range(PAIR_ATTEMPTS):
        cross_r = random.randint(3, height - 4)
        cross_c = random.randint(3, width - 4)

        left_choices = list(range(1, cross_c - 1))
        right_choices = list(range(cross_c + 2, width - 1))
        top_choices = list(range(1, cross_r - 1))
        bottom_choices = list(range(cross_r + 2, height - 1))
        if not left_choices or not right_choices or not top_choices or not bottom_choices:
            continue

        row_pair = (int(row_color), (cross_r, random.choice(left_choices)), (cross_r, random.choice(right_choices)))
        col_pair = (int(col_color), (random.choice(top_choices), cross_c), (random.choice(bottom_choices), cross_c))
        specs = [row_pair, col_pair]
        if _specs_valid(specs, height, width):
            return specs
    return None


def _sample_extra_pair(color: int, height: int, width: int, specs: list[PairSpec]) -> PairSpec | None:
    blocked_rows, blocked_cols = _blocked_axes(specs)
    samplers = [("row", _sample_row_pair), ("col", _sample_col_pair), ("diag", _sample_diag_pair)]

    for _ in range(PAIR_ATTEMPTS):
        _kind, sampler = random.choice(samplers)
        candidate = sampler(color, height, width, blocked_rows, blocked_cols)
        if candidate is None:
            continue
        if _specs_valid(specs + [candidate], height, width):
            return candidate
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    min_side = max(12, 2 * pair_count + 8)
    height = random.randint(min_side, HEIGHT_MAX)
    width = random.randint(min_side, WIDTH_MAX)
    colors = random.sample(COLOR_POOL, pair_count)

    base_specs = _sample_crossing_pair_block(height, width, colors[0], colors[1])
    if base_specs is None:
        return None

    specs = list(base_specs)
    blocked_rows, blocked_cols = _blocked_axes(specs)
    diag_spec = _sample_diag_pair(colors[2], height, width, blocked_rows, blocked_cols)
    if diag_spec is None or not _specs_valid(specs + [diag_spec], height, width):
        return None
    specs.append(diag_spec)

    for color in colors[3:]:
        candidate = _sample_extra_pair(color, height, width, specs)
        if candidate is None:
            return None
        specs.append(candidate)

    kinds = {_pair_kind(a, b) for _color, a, b in specs}
    if kinds != {"row", "col", "diag"}:
        return None

    grid = _render_input(specs, height, width)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != BG)) > int(0.30 * height * width):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input after repeated retries")
