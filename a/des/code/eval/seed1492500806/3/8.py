from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ARROW_COLOR = 9
SIDE_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
SIZE_OPTIONS = [5, 7, 9]
CELL_GRID_RANGE = {5: (3, 6), 7: (3, 5), 9: (2, 4)}
MAX_GENERATION_ATTEMPTS = 500


def _integral(mask: np.ndarray) -> np.ndarray:
    ii = np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0)
    return ii.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ii: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ii[r1 + 1, c1 + 1] - ii[r0, c1 + 1] - ii[r1 + 1, c0] + ii[r0, c0])


def _ring_coords(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    r1 = int(r0 + size - 1)
    c1 = int(c0 + size - 1)
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _cell_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _cells_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _point_bbox_distance(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = bbox
    dr = 0
    dc = 0
    if r < r0:
        dr = int(r0 - r)
    elif r > r1:
        dr = int(r - r1)
    if c < c0:
        dc = int(c0 - c)
    elif c > c1:
        dc = int(c - c1)
    return dr + dc


def _find_arrow(grid: np.ndarray) -> tuple[int, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, np.ndarray, int]] = []
    for color in map(int, np.unique(g)):
        if color == BG:
            continue
        pts = np.argwhere(g == int(color))
        if int(pts.shape[0]) != 2:
            continue
        pts = pts[np.argsort(pts[:, 0] * g.shape[1] + pts[:, 1])]
        dr = int(pts[1, 0] - pts[0, 0])
        dc = int(pts[1, 1] - pts[0, 1])
        if abs(dr) == 1 and abs(dc) == 1:
            direction = 1 if dc > 0 else -1
            candidates.append((color, pts, direction))
    assert len(candidates) == 1, "expected a unique 2-pixel diagonal arrow"
    return candidates[0]


def _gear_candidate(
    g: np.ndarray,
    ring_ii: np.ndarray,
    black_ii: np.ndarray,
    r0: int,
    c0: int,
    size: int,
) -> dict | None:
    r1 = int(r0 + size - 1)
    c1 = int(c0 + size - 1)
    inner_area = int((size - 2) * (size - 2))
    ring_area = int(size * size - inner_area)

    if _rect_sum(black_ii, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != inner_area:
        return None
    if _rect_sum(ring_ii, r0, c0, r1, c1) - _rect_sum(ring_ii, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != ring_area:
        return None

    top_core = np.unique(g[r0, c0 + 1 : c1])
    right_core = np.unique(g[r0 + 1 : r1, c1])
    bottom_core = np.unique(g[r1, c0 + 1 : c1])
    left_core = np.unique(g[r0 + 1 : r1, c0])
    if any(len(vals) != 1 for vals in (top_core, right_core, bottom_core, left_core)):
        return None

    top = int(top_core[0])
    right = int(right_core[0])
    bottom = int(bottom_core[0])
    left = int(left_core[0])
    sides = (top, right, bottom, left)
    if len(set(sides)) != 4:
        return None
    if BG in sides:
        return None

    tl = int(g[r0, c0])
    tr = int(g[r0, c1])
    br = int(g[r1, c1])
    bl = int(g[r1, c0])
    if tl not in {top, left}:
        return None
    if tr not in {top, right}:
        return None
    if br not in {bottom, right}:
        return None
    if bl not in {bottom, left}:
        return None

    coords = _ring_coords(r0, c0, size)
    border_set = set(coords)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "size": int(size),
        "coords": coords,
        "border_set": border_set,
    }


def _find_gears(grid: np.ndarray, arrow_color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    ring_mask = (g != BG) & (g != int(arrow_color))
    black_mask = g == BG
    ring_ii = _integral(ring_mask)
    black_ii = _integral(black_mask)

    gears: list[dict] = []
    for size in range(5, int(min(h, w)) + 1, 2):
        for r0 in range(0, int(h - size + 1)):
            for c0 in range(0, int(w - size + 1)):
                if int(g[r0, c0]) in (BG, int(arrow_color)):
                    continue
                gear = _gear_candidate(g, ring_ii, black_ii, r0, c0, size)
                if gear is not None:
                    gears.append(gear)
    assert gears, "expected at least one gear"
    return gears


def _gear_graph(gears: list[dict]) -> list[list[int]]:
    graph = [[] for _ in gears]
    for i in range(len(gears)):
        border_i = gears[i]["border_set"]
        for j in range(i + 1, len(gears)):
            border_j = gears[j]["border_set"]
            touching = False
            for r, c in border_i:
                if (
                    (r - 1, c) in border_j
                    or (r + 1, c) in border_j
                    or (r, c - 1) in border_j
                    or (r, c + 1) in border_j
                ):
                    touching = True
                    break
            if touching:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def _find_driver_gear(gears: list[dict], arrow_pts: np.ndarray) -> int:
    pts = [tuple(map(int, pt)) for pt in arrow_pts]
    dists = []
    for gear in gears:
        bbox = gear["bbox"]
        dists.append(min(_point_bbox_distance(pt, bbox) for pt in pts))
    best = min(dists)
    winners = [idx for idx, dist in enumerate(dists) if dist == best]
    assert best <= 2 and len(winners) == 1, "arrow must identify a unique nearby driver gear"
    return int(winners[0])


def _rotation_directions(graph: list[list[int]], start: int, first_dir: int) -> dict[int, int]:
    directions = {int(start): int(first_dir)}
    q: deque[int] = deque([int(start)])
    while q:
        node = int(q.popleft())
        for nxt in graph[node]:
            wanted = -int(directions[node])
            if nxt in directions:
                assert directions[nxt] == wanted, "gear directions must be consistent"
                continue
            directions[int(nxt)] = wanted
            q.append(int(nxt))
    return directions


def _rotate_ring_values(grid: np.ndarray, coords: list[tuple[int, int]], shift: int) -> np.ndarray:
    vals = np.array([int(grid[r, c]) for r, c in coords], dtype=int)
    return np.roll(vals, int(shift))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate a square-ring gear train driven by a 2-pixel arrow.

    Input structure:
    - Black background.
    - Each gear is an odd-sized square ring with a black interior.
    - The four sides of a gear use four distinct colors; corners use one of the two adjacent side colors.
    - Exactly one diagonal 2-pixel arrow marks the driver gear and encodes clockwise or counterclockwise motion.

    Rule:
    - Remove the arrow in the output.
    - Rotate the marked driver gear by 90 degrees.
    - Every meshed neighboring gear rotates by 90 degrees in the opposite direction.
    - Continue this alternating transmission through the whole connected gear component.
    - Unconnected gears stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    arrow_color, arrow_pts, driver_dir = _find_arrow(g)
    gears = _find_gears(g, arrow_color)
    graph = _gear_graph(gears)
    driver = _find_driver_gear(gears, arrow_pts)
    directions = _rotation_directions(graph, driver, driver_dir)

    out = g.copy()
    out[arrow_pts[:, 0], arrow_pts[:, 1]] = BG
    for idx, direction in directions.items():
        gear = gears[idx]
        shift = int(direction * (gear["size"] - 1))
        new_vals = _rotate_ring_values(g, gear["coords"], shift)
        for (r, c), value in zip(gear["coords"], new_vals):
            out[r, c] = int(value)
    return out


def _component_cells(rows: int, cols: int, count: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    cells = {start}
    while len(cells) < count:
        frontier = []
        for cell in cells:
            for nxt in _cell_neighbors(cell):
                if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in cells:
                    frontier.append(nxt)
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            return None
        weights = [1 + sum(nb in cells for nb in _cell_neighbors(cell)) for cell in frontier]
        cells.add(random.choices(frontier, weights=weights, k=1)[0])
    return cells


def _pick_isolated_cells(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    max_count: int,
) -> set[tuple[int, int]]:
    if max_count <= 0:
        return set()
    candidates = []
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if cell in occupied:
                continue
            if any(_cells_adjacent(cell, other) for other in occupied):
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    chosen: set[tuple[int, int]] = set()
    target = random.randint(0, int(max_count))
    for cell in candidates:
        if any(_cells_adjacent(cell, other) for other in chosen):
            continue
        chosen.add(cell)
        if len(chosen) == target:
            break
    if len(chosen) != target:
        return set()
    return chosen


def _assign_side_colors(cells: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int, int, int]] | None:
    order = sorted(
        cells,
        key=lambda cell: (-sum(_cells_adjacent(cell, other) for other in cells if other != cell), cell),
    )
    assigned: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    def fits(cell: tuple[int, int], colors: tuple[int, int, int, int]) -> bool:
        top, right, bottom, left = colors
        for other in _cell_neighbors(cell):
            if other not in assigned:
                continue
            otop, oright, obottom, oleft = assigned[other]
            dr = int(other[0] - cell[0])
            dc = int(other[1] - cell[1])
            if dr == -1 and dc == 0 and top == obottom:
                return False
            if dr == 1 and dc == 0 and bottom == otop:
                return False
            if dr == 0 and dc == -1 and left == oright:
                return False
            if dr == 0 and dc == 1 and right == oleft:
                return False
        return True

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        cell = order[idx]
        seen: set[tuple[int, int, int, int]] = set()
        options: list[tuple[int, int, int, int]] = []
        while len(options) < 80:
            choice = tuple(random.sample(SIDE_COLOR_POOL, 4))
            if choice in seen:
                continue
            seen.add(choice)
            options.append(choice)
        for colors in options:
            if not fits(cell, colors):
                continue
            assigned[cell] = colors
            if backtrack(idx + 1):
                return True
            del assigned[cell]
        return False

    if not backtrack(0):
        return None
    return assigned


def _draw_gear(grid: np.ndarray, r0: int, c0: int, size: int, colors: tuple[int, int, int, int]) -> None:
    top, right, bottom, left = colors
    r1 = int(r0 + size - 1)
    c1 = int(c0 + size - 1)
    grid[r0, c0:c1] = int(top)
    grid[r0 + 1 : r1 + 1, c1] = int(right)
    grid[r1, c0 + 1 : c1 + 1] = int(bottom)
    grid[r0 + 1 : r1, c0] = int(left)
    grid[r0, c0] = int(top)
    grid[r0, c1] = int(right)
    grid[r1, c1] = int(bottom)
    grid[r1, c0] = int(left)


def _cell_bbox(cell: tuple[int, int], size: int, top: int, left: int) -> tuple[int, int, int, int]:
    r0 = int(top + cell[0] * size)
    c0 = int(left + cell[1] * size)
    r1 = int(r0 + size - 1)
    c1 = int(c0 + size - 1)
    return (r0, c0, r1, c1)


def _arrow_pixels(bbox: tuple[int, int, int, int], clockwise: bool) -> np.ndarray:
    r0, c0, _r1, c1 = bbox
    mid = int((c0 + c1) // 2)
    if clockwise:
        pts = np.array([[r0 - 2, mid - 1], [r0 - 1, mid]], dtype=int)
    else:
        pts = np.array([[r0 - 2, mid + 1], [r0 - 1, mid]], dtype=int)
    return pts


def _expected_output(
    grid: np.ndarray,
    gear_cells: set[tuple[int, int]],
    size: int,
    top_margin: int,
    left_margin: int,
    driver_cell: tuple[int, int],
    driver_dir: int,
) -> np.ndarray:
    graph = {cell: [] for cell in gear_cells}
    for cell in gear_cells:
        for other in _cell_neighbors(cell):
            if other in gear_cells:
                graph[cell].append(other)
    directions = {driver_cell: int(driver_dir)}
    q: deque[tuple[int, int]] = deque([driver_cell])
    while q:
        cell = q.popleft()
        for nxt in graph[cell]:
            wanted = -int(directions[cell])
            if nxt in directions:
                continue
            directions[nxt] = wanted
            q.append(nxt)

    arrow_color, arrow_pts, _ = _find_arrow(grid)
    out = np.array(grid, copy=True)
    out[arrow_pts[:, 0], arrow_pts[:, 1]] = BG
    for cell, direction in directions.items():
        r0, c0, _r1, _c1 = _cell_bbox(cell, size, top_margin, left_margin)
        coords = _ring_coords(r0, c0, size)
        rotated = _rotate_ring_values(grid, coords, int(direction * (size - 1)))
        for (r, c), value in zip(coords, rotated):
            out[r, c] = int(value)
    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(SIZE_OPTIONS)
    rows = random.randint(*CELL_GRID_RANGE[size])
    cols = random.randint(*CELL_GRID_RANGE[size])
    gear_count = random.randint(1, min(5, rows * cols))

    component = _component_cells(rows, cols, gear_count)
    if component is None:
        return None

    isolated = _pick_isolated_cells(rows, cols, component, min(2, rows * cols - len(component)))
    gear_cells = set(component) | set(isolated)
    if not gear_cells:
        return None

    colors = _assign_side_colors(sorted(gear_cells))
    if colors is None:
        return None

    driver_candidates = [cell for cell in component if (cell[0] - 1, cell[1]) not in gear_cells]
    if not driver_candidates:
        return None
    driver_cell = random.choice(driver_candidates)
    clockwise = bool(random.getrandbits(1))
    driver_dir = 1 if clockwise else -1

    top_margin = random.randint(2, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    height = int(top_margin + rows * size + bottom_margin)
    width = int(left_margin + cols * size + right_margin)
    grid = np.full((height, width), BG, dtype=int)

    expected_bboxes = set()
    for cell in gear_cells:
        r0, c0, r1, c1 = _cell_bbox(cell, size, top_margin, left_margin)
        _draw_gear(grid, r0, c0, size, colors[cell])
        expected_bboxes.add((r0, c0, r1, c1))

    driver_bbox = _cell_bbox(driver_cell, size, top_margin, left_margin)
    arrow_pts = _arrow_pixels(driver_bbox, clockwise)
    if np.any(arrow_pts[:, 0] < 0) or np.any(arrow_pts[:, 1] < 0):
        return None
    if np.any(arrow_pts[:, 0] >= height) or np.any(arrow_pts[:, 1] >= width):
        return None
    if any(int(grid[r, c]) != BG for r, c in arrow_pts):
        return None
    grid[arrow_pts[:, 0], arrow_pts[:, 1]] = ARROW_COLOR

    try:
        arrow_color, found_arrow, _ = _find_arrow(grid)
        if int(arrow_color) != ARROW_COLOR or not np.array_equal(found_arrow, arrow_pts):
            return None
        gears = _find_gears(grid, arrow_color)
        if {tuple(gear["bbox"]) for gear in gears} != expected_bboxes:
            return None
        driver_idx = _find_driver_gear(gears, arrow_pts)
        if tuple(gears[driver_idx]["bbox"]) != driver_bbox:
            return None
        expected = _expected_output(grid, gear_cells, size, top_margin, left_margin, driver_cell, driver_dir)
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear-train input")
