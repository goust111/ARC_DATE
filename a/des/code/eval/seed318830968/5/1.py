import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)


Point = Tuple[int, int]
Cell = Tuple[int, int]
GapEdge = Tuple[str, Cell, Optional[Cell]]


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _l_path_cells(start: Point, end: Point) -> List[Point]:
    r0, c0 = start
    r1, c1 = end
    cells: List[Point] = []

    c_step = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + c_step, c_step):
        cells.append((r0, c))

    if r1 != r0:
        r_step = 1 if r1 >= r0 else -1
        for r in range(r0 + r_step, r1 + r_step, r_step):
            cells.append((r, c1))
    return cells


def _divider_indices(grid: np.ndarray, axis: int) -> List[int]:
    maroon_mask = grid == MAROON
    if axis == 0:
        counts = np.count_nonzero(maroon_mask, axis=1)
        threshold = grid.shape[1] // 2
        size = grid.shape[0]
    else:
        counts = np.count_nonzero(maroon_mask, axis=0)
        threshold = grid.shape[0] // 2
        size = grid.shape[1]

    dividers = [idx for idx, count in enumerate(counts.tolist()) if count > threshold]
    assert len(dividers) >= 2, "expected at least an outer frame"
    assert dividers[0] == 0 and dividers[-1] == size - 1, "outer frame must lie on the boundary"
    assert all(b - a >= 2 for a, b in zip(dividers, dividers[1:])), "regions need positive interior size"
    return dividers


def _region_gap_target(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> Point:
    gaps: List[Point] = []
    targets: List[Point] = []

    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) == BG:
            gaps.append((r0, c))
            targets.append((r0 + 1, c))
        if int(grid[r1, c]) == BG:
            gaps.append((r1, c))
            targets.append((r1 - 1, c))

    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) == BG:
            gaps.append((r, c0))
            targets.append((r, c0 + 1))
        if int(grid[r, c1]) == BG:
            gaps.append((r, c1))
            targets.append((r, c1 - 1))

    assert len(gaps) == 1, "each region must expose exactly one boundary gap"
    return targets[0]


def _nearest_unique_order(seed: Point, posts: Sequence[Point]) -> List[Point]:
    current = seed
    remaining = list(posts)
    order: List[Point] = []
    while remaining:
        remaining.sort(key=lambda p: (_manhattan(current, p), p[0], p[1]))
        order.append(remaining.pop(0))
        current = order[-1]
    return order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_lines = _divider_indices(g, axis=0)
    col_lines = _divider_indices(g, axis=1)

    out = g.copy()
    for i in range(len(row_lines) - 1):
        r0, r1 = row_lines[i], row_lines[i + 1]
        for j in range(len(col_lines) - 1):
            c0, c1 = col_lines[j], col_lines[j + 1]
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            assert interior.size > 0

            seed_local = np.argwhere((interior != BG) & (interior != GREY) & (interior != MAROON))
            assert seed_local.shape[0] == 1, "each region must have exactly one seed"
            seed = (int(seed_local[0, 0]) + r0 + 1, int(seed_local[0, 1]) + c0 + 1)
            seed_color = int(g[seed])

            posts_local = np.argwhere(interior == GREY)
            posts = [(int(r) + r0 + 1, int(c) + c0 + 1) for r, c in posts_local]

            target = _region_gap_target(g, r0, r1, c0, c1)
            current = seed
            for post in _nearest_unique_order(seed, posts):
                for cell in _l_path_cells(current, post):
                    out[cell] = seed_color
                current = post

            for cell in _l_path_cells(current, target):
                out[cell] = seed_color

    return out


LAYOUTS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
CELL_SIZE_RANGE = (4, 7)
GUIDE_COUNT_RANGE = (1, 3)
MAX_MATCHING_ATTEMPTS = 80
MAX_REGION_POINT_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 400


def _line_positions(sizes: Sequence[int]) -> List[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += int(size) + 1
        positions.append(cursor)
    return positions


def _cell_box(row_lines: Sequence[int], col_lines: Sequence[int], cell: Cell) -> Tuple[int, int, int, int]:
    i, j = cell
    return row_lines[i], row_lines[i + 1], col_lines[j], col_lines[j + 1]


def _cell_options(n_rows: int, n_cols: int, cell: Cell) -> List[GapEdge]:
    i, j = cell
    options: List[GapEdge] = []
    if i > 0:
        options.append(("shared_h", cell, (i - 1, j)))
    if i + 1 < n_rows:
        options.append(("shared_h", cell, (i + 1, j)))
    if j > 0:
        options.append(("shared_v", cell, (i, j - 1)))
    if j + 1 < n_cols:
        options.append(("shared_v", cell, (i, j + 1)))
    if i == 0:
        options.append(("top", cell, None))
    if i == n_rows - 1:
        options.append(("bottom", cell, None))
    if j == 0:
        options.append(("left", cell, None))
    if j == n_cols - 1:
        options.append(("right", cell, None))
    return options


def _edge_cells(edge: GapEdge) -> Tuple[Cell, ...]:
    kind, cell, other = edge
    if kind.startswith("shared"):
        assert other is not None
        return (cell, other)
    return (cell,)


def _search_gap_matching(n_rows: int, n_cols: int) -> Optional[List[GapEdge]]:
    cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    option_map = {cell: _cell_options(n_rows, n_cols, cell) for cell in cells}

    def rec(covered: set[Cell], chosen: List[GapEdge]) -> Optional[List[GapEdge]]:
        if len(covered) == len(cells):
            return list(chosen)

        best_cell: Optional[Cell] = None
        best_options: Optional[List[GapEdge]] = None
        for cell in cells:
            if cell in covered:
                continue
            valid = [
                edge
                for edge in option_map[cell]
                if all(member not in covered for member in _edge_cells(edge))
            ]
            if not valid:
                return None
            random.shuffle(valid)
            if best_options is None or len(valid) < len(best_options):
                best_cell = cell
                best_options = valid

        assert best_cell is not None and best_options is not None
        for edge in best_options:
            edge_cover = set(_edge_cells(edge))
            result = rec(covered | edge_cover, chosen + [edge])
            if result is not None:
                return result
        return None

    return rec(set(), [])


def _assign_gaps(row_lines: Sequence[int], col_lines: Sequence[int], edges: Sequence[GapEdge]) -> Dict[Cell, Tuple[Point, Point]]:
    gap_map: Dict[Cell, Tuple[Point, Point]] = {}
    for kind, cell, other in edges:
        i, j = cell
        r0, r1, c0, c1 = _cell_box(row_lines, col_lines, cell)

        if kind == "top":
            gap = (r0, random.randint(c0 + 1, c1 - 1))
            gap_map[cell] = (gap, (r0 + 1, gap[1]))
            continue
        if kind == "bottom":
            gap = (r1, random.randint(c0 + 1, c1 - 1))
            gap_map[cell] = (gap, (r1 - 1, gap[1]))
            continue
        if kind == "left":
            gap = (random.randint(r0 + 1, r1 - 1), c0)
            gap_map[cell] = (gap, (gap[0], c0 + 1))
            continue
        if kind == "right":
            gap = (random.randint(r0 + 1, r1 - 1), c1)
            gap_map[cell] = (gap, (gap[0], c1 - 1))
            continue

        assert other is not None
        oi, oj = other
        if kind == "shared_v":
            left = (i, min(j, oj))
            right = (i, max(j, oj))
            lr0, lr1, lc0, lc1 = _cell_box(row_lines, col_lines, left)
            rr0, rr1, rc0, rc1 = _cell_box(row_lines, col_lines, right)
            assert lr0 == rr0 and lr1 == rr1 and lc1 == rc0
            gap = (random.randint(lr0 + 1, lr1 - 1), lc1)
            gap_map[left] = (gap, (gap[0], lc1 - 1))
            gap_map[right] = (gap, (gap[0], rc0 + 1))
        else:
            top = (min(i, oi), j)
            bottom = (max(i, oi), j)
            tr0, tr1, tc0, tc1 = _cell_box(row_lines, col_lines, top)
            br0, br1, bc0, bc1 = _cell_box(row_lines, col_lines, bottom)
            assert tc0 == bc0 and tc1 == bc1 and tr1 == br0
            gap = (tr1, random.randint(tc0 + 1, tc1 - 1))
            gap_map[top] = (gap, (tr1 - 1, gap[1]))
            gap_map[bottom] = (gap, (br0 + 1, gap[1]))
    return gap_map


def _greedy_route(seed: Point, guides: Sequence[Point], target: Point) -> Optional[List[Point]]:
    remaining = list(guides)
    order: List[Point] = []
    current = seed
    painted = {seed}

    while remaining:
        distances = sorted((_manhattan(current, point), point) for point in remaining)
        if len(distances) >= 2 and distances[0][0] == distances[1][0]:
            return None
        nxt = distances[0][1]
        segment = _l_path_cells(current, nxt)
        body = segment[1:]
        if target in body:
            return None
        if any(point in painted for point in body):
            return None
        if any(point != nxt and point in body for point in remaining):
            return None
        painted.update(body)
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt

    final_segment = _l_path_cells(current, target)
    if any(point in painted for point in final_segment[1:]):
        return None
    return order


def _sample_region_points(
    row_lines: Sequence[int],
    col_lines: Sequence[int],
    cell: Cell,
    target: Point,
    guide_count: int,
) -> Optional[Tuple[Point, List[Point]]]:
    r0, r1, c0, c1 = _cell_box(row_lines, col_lines, cell)
    candidates = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) != target
    ]
    if len(candidates) < guide_count + 1:
        return None

    for _ in range(MAX_REGION_POINT_ATTEMPTS):
        picked = random.sample(candidates, guide_count + 1)
        seed = picked[0]
        guides = picked[1:]
        order = _greedy_route(seed, guides, target)
        if order is not None:
            return seed, order
    return None


def _build_expected(inp: np.ndarray, specs: Sequence[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        current = spec["seed"]
        color = spec["color"]
        for nxt in [*spec["guides"], spec["target"]]:
            for cell in _l_path_cells(current, nxt):
                out[cell] = color
            current = nxt
    return out


def _gen_attempt() -> Optional[np.ndarray]:
    n_rows, n_cols = random.choice(LAYOUTS)
    heights = [random.randint(*CELL_SIZE_RANGE) for _ in range(n_rows)]
    widths = [random.randint(*CELL_SIZE_RANGE) for _ in range(n_cols)]
    row_lines = _line_positions(heights)
    col_lines = _line_positions(widths)

    for _ in range(MAX_MATCHING_ATTEMPTS):
        edges = _search_gap_matching(n_rows, n_cols)
        if edges is None:
            continue
        gap_map = _assign_gaps(row_lines, col_lines, edges)
        if len(gap_map) == n_rows * n_cols:
            break
    else:
        return None

    grid = np.full((row_lines[-1] + 1, col_lines[-1] + 1), BG, dtype=int)
    grid[row_lines, :] = MAROON
    grid[:, col_lines] = MAROON

    for gap, _target in gap_map.values():
        grid[gap] = BG

    cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    colors = random.sample(SEED_COLORS, len(cells))
    specs = []

    for color, cell in zip(colors, cells):
        gap, target = gap_map[cell]
        _ = gap
        guide_count = random.randint(*GUIDE_COUNT_RANGE)
        placement = _sample_region_points(row_lines, col_lines, cell, target, guide_count)
        if placement is None:
            return None
        seed, guides = placement
        grid[seed] = color
        for guide in guides:
            grid[guide] = GREY
        specs.append({"seed": seed, "guides": guides, "target": target, "color": color})

    expected = _build_expected(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
