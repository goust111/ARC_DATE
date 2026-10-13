from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))

# Generator configuration.
GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (18, 24)
LEGEND_COLOR_COUNT_OPTIONS = (2, 3)
SAFE_MARGIN = 1
LEFT_RIGHT_RUN_RANGE = (2, 5)
GAP_LENGTH_RANGE = (1, 3)
STUB_LENGTH_RANGE = (0, 4)
EXTRA_PATH_COUNT_RANGE = (0, 2)
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 400
BRANCHES = (("h", True), ("h", False), ("v", True), ("v", False))
NEIGHBOR_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _legend_size(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    n_cols = 0
    c = 1
    while c < w and int(grid[0, c]) != bg:
        n_cols += 1
        c += 1

    n_rows = 0
    r = 1
    while r < h and int(grid[r, 0]) != bg:
        n_rows += 1
        r += 1

    n = min(n_rows, n_cols)
    assert n >= 2, "expected a legend with at least two row/column colors"
    assert bool(np.all(grid[1 : n + 1, 0] != bg))
    assert bool(np.all(grid[0, 1 : n + 1] != bg))
    assert bool(np.all(grid[1 : n + 1, 1 : n + 1] != bg))
    return int(n)


def _read_legend(grid: np.ndarray, bg: int) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    n = _legend_size(grid, bg)
    row_colors = [int(grid[r, 0]) for r in range(1, n + 1)]
    col_colors = [int(grid[0, c]) for c in range(1, n + 1)]

    mapping: dict[tuple[int, int], int] = {}
    for i, row_color in enumerate(row_colors, start=1):
        for j, col_color in enumerate(col_colors, start=1):
            mapping[(row_color, col_color)] = int(grid[i, j])

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[: n + 1, : n + 1] = True
    return mapping, legend_mask


def _fill_horizontal_gaps(
    grid: np.ndarray,
    out: np.ndarray,
    bg: int,
    mapping: dict[tuple[int, int], int],
    legend_mask: np.ndarray,
) -> None:
    h, w = grid.shape
    for r in range(h):
        c = 0
        while c < w:
            if legend_mask[r, c] or int(grid[r, c]) != bg:
                c += 1
                continue
            start = c
            while c < w and (not legend_mask[r, c]) and int(grid[r, c]) == bg:
                c += 1
            end = c
            gap_len = end - start
            if not (1 <= gap_len <= 3):
                continue
            if start < 2 or end + 1 >= w:
                continue
            if legend_mask[r, start - 1] or legend_mask[r, end]:
                continue
            if legend_mask[r, start - 2] or legend_mask[r, end + 1]:
                continue

            left_color = int(grid[r, start - 1])
            right_color = int(grid[r, end])
            if left_color == bg or right_color == bg:
                continue
            if int(grid[r, start - 2]) == bg or int(grid[r, end + 1]) == bg:
                continue
            if (left_color, right_color) not in mapping:
                continue

            clear_sides = True
            for cc in range(start, end):
                if r > 0 and int(grid[r - 1, cc]) != bg:
                    clear_sides = False
                    break
                if r + 1 < h and int(grid[r + 1, cc]) != bg:
                    clear_sides = False
                    break
            if not clear_sides:
                continue

            out[r, start:end] = mapping[(left_color, right_color)]


def _fill_vertical_gaps(
    grid: np.ndarray,
    out: np.ndarray,
    bg: int,
    mapping: dict[tuple[int, int], int],
    legend_mask: np.ndarray,
) -> None:
    h, w = grid.shape
    for c in range(w):
        r = 0
        while r < h:
            if legend_mask[r, c] or int(grid[r, c]) != bg:
                r += 1
                continue
            start = r
            while r < h and (not legend_mask[r, c]) and int(grid[r, c]) == bg:
                r += 1
            end = r
            gap_len = end - start
            if not (1 <= gap_len <= 3):
                continue
            if start < 2 or end + 1 >= h:
                continue
            if legend_mask[start - 1, c] or legend_mask[end, c]:
                continue
            if legend_mask[start - 2, c] or legend_mask[end + 1, c]:
                continue

            top_color = int(grid[start - 1, c])
            bottom_color = int(grid[end, c])
            if top_color == bg or bottom_color == bg:
                continue
            if int(grid[start - 2, c]) == bg or int(grid[end + 1, c]) == bg:
                continue
            if (top_color, bottom_color) not in mapping:
                continue

            clear_sides = True
            for rr in range(start, end):
                if c > 0 and int(grid[rr, c - 1]) != bg:
                    clear_sides = False
                    break
                if c + 1 < w and int(grid[rr, c + 1]) != bg:
                    clear_sides = False
                    break
            if not clear_sides:
                continue

            out[start:end, c] = mapping[(top_color, bottom_color)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair orthogonal path breaks by looking up the bridge color in the top-left legend.

    Input structure:
    - Background is black.
    - The top-left corner contains a legend whose first row and first column list
      path colors, and whose inner cells encode the mixed result color.
    - Outside the legend, one-pixel-wide horizontal and vertical paths may contain
      1-3 cell background gaps.

    Rule:
    - For each horizontal or vertical background gap of length 1-3 whose two
      immediate endpoints are colored path pixels, read the endpoint colors.
    - Look up the corresponding mixed color in the legend.
    - Fill the gap with that mixed color, preserving the legend and all existing
      path pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(g[0, 0])
    mapping, legend_mask = _read_legend(g, bg)

    out = g.copy()
    _fill_horizontal_gaps(g, out, bg, mapping, legend_mask)
    _fill_vertical_gaps(g, out, bg, mapping, legend_mask)
    return out


def _build_legend(color_count: int) -> tuple[list[int], dict[tuple[int, int], int], np.ndarray]:
    base_colors = random.sample(COLOR_POOL, color_count)
    mix_count = color_count * (color_count - 1) // 2
    mix_colors = random.sample([c for c in COLOR_POOL if c not in base_colors], mix_count)

    pair_color: dict[frozenset[int], int] = {}
    mix_idx = 0
    for i in range(color_count):
        for j in range(i + 1, color_count):
            pair_color[frozenset((base_colors[i], base_colors[j]))] = mix_colors[mix_idx]
            mix_idx += 1

    mapping: dict[tuple[int, int], int] = {}
    for a in base_colors:
        for b in base_colors:
            if a == b:
                mapping[(a, b)] = int(a)
            else:
                mapping[(a, b)] = int(pair_color[frozenset((a, b))])

    row_order = random.sample(base_colors, len(base_colors))
    col_order = random.sample(base_colors, len(base_colors))
    legend = np.full((color_count + 1, color_count + 1), BACKGROUND, dtype=int)
    legend[0, 1:] = np.array(col_order, dtype=int)
    legend[1:, 0] = np.array(row_order, dtype=int)
    for i, row_color in enumerate(row_order, start=1):
        for j, col_color in enumerate(col_order, start=1):
            legend[i, j] = mapping[(row_color, col_color)]
    return base_colors, mapping, legend


def _coords_to_set(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _mark_buffer(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr, dc in NEIGHBOR_8:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _choose_endpoint_colors(base_colors: list[int], same_color: bool) -> tuple[int, int]:
    if same_color:
        color = random.choice(base_colors)
        return int(color), int(color)
    a, b = random.sample(base_colors, 2)
    return int(a), int(b)


def _random_stub_lengths() -> tuple[int, int]:
    a = random.randint(*STUB_LENGTH_RANGE)
    b = random.randint(*STUB_LENGTH_RANGE)
    if a == 0 and b == 0 and random.random() < 0.7:
        if random.random() < 0.5:
            a = random.randint(1, STUB_LENGTH_RANGE[1])
        else:
            b = random.randint(1, STUB_LENGTH_RANGE[1])
    return int(a), int(b)


def _place_path(
    grid: np.ndarray,
    blocked: np.ndarray,
    orientation: str,
    same_color: bool,
    base_colors: list[int],
    mapping: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]], int] | None:
    h, w = grid.shape
    left_color, right_color = _choose_endpoint_colors(base_colors, same_color)

    for _ in range(PLACEMENT_RETRIES):
        run_a = random.randint(*LEFT_RIGHT_RUN_RANGE)
        run_b = random.randint(*LEFT_RIGHT_RUN_RANGE)
        gap_len = random.randint(*GAP_LENGTH_RANGE)
        stub_a, stub_b = _random_stub_lengths()

        if orientation == "h":
            dir_a = random.choice((-1, 1)) if stub_a else 0
            dir_b = random.choice((-1, 1)) if stub_b else 0
            up_need = max(stub_a if dir_a == -1 else 0, stub_b if dir_b == -1 else 0)
            down_need = max(stub_a if dir_a == 1 else 0, stub_b if dir_b == 1 else 0)
            row_lo = SAFE_MARGIN + up_need
            row_hi = h - 1 - SAFE_MARGIN - down_need
            total = run_a + gap_len + run_b
            col_lo = SAFE_MARGIN
            col_hi = w - 1 - SAFE_MARGIN - total + 1
            if row_lo > row_hi or col_lo > col_hi:
                continue

            r = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            c1 = c0 + total - 1
            gap_start = c0 + run_a
            gap_end = gap_start + gap_len

            path_cells: list[tuple[int, int, int]] = []
            for c in range(c0, gap_start):
                path_cells.append((r, c, left_color))
            for c in range(gap_end, c1 + 1):
                path_cells.append((r, c, right_color))

            stub_color_a = random.choice(base_colors)
            stub_color_b = random.choice(base_colors)
            for step in range(1, stub_a + 1):
                path_cells.append((r + dir_a * step, c0, int(stub_color_a)))
            for step in range(1, stub_b + 1):
                path_cells.append((r + dir_b * step, c1, int(stub_color_b)))

            gap_cells = [(r, c) for c in range(gap_start, gap_end)]
            side_coords = gap_cells
        else:
            dir_a = random.choice((-1, 1)) if stub_a else 0
            dir_b = random.choice((-1, 1)) if stub_b else 0
            left_need = max(stub_a if dir_a == -1 else 0, stub_b if dir_b == -1 else 0)
            right_need = max(stub_a if dir_a == 1 else 0, stub_b if dir_b == 1 else 0)
            col_lo = SAFE_MARGIN + left_need
            col_hi = w - 1 - SAFE_MARGIN - right_need
            total = run_a + gap_len + run_b
            row_lo = SAFE_MARGIN
            row_hi = h - 1 - SAFE_MARGIN - total + 1
            if row_lo > row_hi or col_lo > col_hi:
                continue

            c = random.randint(col_lo, col_hi)
            r0 = random.randint(row_lo, row_hi)
            r1 = r0 + total - 1
            gap_start = r0 + run_a
            gap_end = gap_start + gap_len

            path_cells = []
            for r in range(r0, gap_start):
                path_cells.append((r, c, left_color))
            for r in range(gap_end, r1 + 1):
                path_cells.append((r, c, right_color))

            stub_color_a = random.choice(base_colors)
            stub_color_b = random.choice(base_colors)
            for step in range(1, stub_a + 1):
                path_cells.append((r0, c + dir_a * step, int(stub_color_a)))
            for step in range(1, stub_b + 1):
                path_cells.append((r1, c + dir_b * step, int(stub_color_b)))

            gap_cells = [(r, c) for r in range(gap_start, gap_end)]
            side_coords = gap_cells

        path_set = _coords_to_set([(r, c) for r, c, _ in path_cells])
        if len(path_set) != len(path_cells):
            continue

        footprint = list(path_set) + gap_cells
        if any(blocked[r, c] for r, c in footprint):
            continue
        if any(int(grid[r, c]) != BACKGROUND for r, c in path_set):
            continue
        if any(int(grid[r, c]) != BACKGROUND for r, c in gap_cells):
            continue

        if orientation == "h":
            for r, c in side_coords:
                if (r > 0 and blocked[r - 1, c]) or (r + 1 < h and blocked[r + 1, c]):
                    break
            else:
                return path_cells, gap_cells, int(mapping[(left_color, right_color)])
        else:
            for r, c in side_coords:
                if (c > 0 and blocked[r, c - 1]) or (c + 1 < w and blocked[r, c + 1]):
                    break
            else:
                return path_cells, gap_cells, int(mapping[(left_color, right_color)])

    return None


def _draw_path(grid: np.ndarray, blocked: np.ndarray, path_cells: list[tuple[int, int, int]], gap_cells: list[tuple[int, int]]) -> None:
    for r, c, color in path_cells:
        grid[r, c] = int(color)
    _mark_buffer(blocked, [(r, c) for r, c, _ in path_cells] + gap_cells)


def _build_case() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    color_count = random.choice(LEGEND_COLOR_COUNT_OPTIONS)
    base_colors, mapping, legend = _build_legend(color_count)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    legend_h, legend_w = legend.shape
    if legend_h + 3 >= h or legend_w + 3 >= w:
        return None
    grid[:legend_h, :legend_w] = legend

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[h - 1, :] = True
    blocked[:, 0] = True
    blocked[:, w - 1] = True
    _mark_buffer(
        blocked,
        [(r, c) for r in range(legend_h) for c in range(legend_w)],
    )

    expected = grid.copy()
    path_specs = list(BRANCHES)
    extra_count = random.randint(*EXTRA_PATH_COUNT_RANGE)
    for _ in range(extra_count):
        path_specs.append((random.choice(("h", "v")), random.choice((False, True))))
    random.shuffle(path_specs)

    for orientation, same_color in path_specs:
        placed = _place_path(grid, blocked, orientation, same_color, base_colors, mapping)
        if placed is None:
            return None
        path_cells, gap_cells, bridge_color = placed
        _draw_path(grid, blocked, path_cells, gap_cells)
        for r, c, color in path_cells:
            expected[r, c] = int(color)
        for r, c in gap_cells:
            expected[r, c] = int(bridge_color)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _build_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid puzzle instance")
