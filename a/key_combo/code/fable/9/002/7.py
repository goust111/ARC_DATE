import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _path_endpoints(mask: np.ndarray) -> list[tuple[int, int]] | None:
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return None

    endpoints: list[tuple[int, int]] = []
    h, w = mask.shape
    for r, c in coords:
        degree = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                degree += 1
        if degree == 1:
            endpoints.append((int(r), int(c)))
        elif degree != 2:
            return None

    if len(endpoints) != 2:
        return None
    if not all(r in (0, h - 1) or c in (0, w - 1) for r, c in endpoints):
        return None
    return endpoints


def _detect_path_color(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        mask = grid == color
        if len(_component_cells(mask)) != 1:
            continue
        endpoints = _path_endpoints(mask)
        if endpoints is None:
            continue
        candidates.append((int(color), endpoints))
    assert len(candidates) == 1, "expected exactly one simple divider path color"
    return candidates[0]


def _trace_path(mask: np.ndarray, endpoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    start = min(endpoints)
    path: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        path.append(cur)
        rr, cc = cur
        nxt = []
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if prev is None or (nr, nc) != prev:
                    nxt.append((nr, nc))
        if not nxt:
            break
        assert len(nxt) == 1, "simple path traversal must have a unique next step"
        prev, cur = cur, nxt[0]

    assert len(path) == int(mask.sum()), "path traversal missed divider cells"
    return path


def _label_regions(grid: np.ndarray, path_color: int) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    region_ids = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []
    seen = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if grid[r, c] == path_color or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            region_id = len(regions)
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                region_ids[rr, cc] = region_id
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != path_color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            regions.append(cells)
    return region_ids, regions


def _ordered_region_ids(path: list[tuple[int, int]], region_ids: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    h, w = region_ids.shape
    for r, c in path:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                region_id = int(region_ids[nr, nc])
                if region_id >= 0 and region_id not in seen:
                    seen.add(region_id)
                    order.append(region_id)
    return order


def _extract_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    path_color, endpoints = _detect_path_color(g, bg)
    path = _trace_path(g == path_color, endpoints)
    region_ids, regions = _label_regions(g, path_color)
    region_order = _ordered_region_ids(path, region_ids)
    assert len(region_order) == len(regions), "every region must appear along the divider walk"
    return g, bg, path_color, path, region_ids, regions, region_order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the yellow arithmetic progression of regions encountered along the red zigzag divider.

    Input structure:
    - A one-cell-thick red divider path starts on the top border, snakes downward with side-to-side bends,
      and ends on the bottom border.
    - The red path partitions the black field into several 4-connected regions that alternately appear on
      opposite sides of the walk.
    - A few whole regions are already yellow while the remaining regions stay black.

    Rule:
    - Recover the red divider as an ordered simple path from its top endpoint to its bottom endpoint.
    - Treat red as a wall, find the non-red regions, and number those regions by the first moment they touch
      the divider walk.
    - The already-yellow region numbers form the first terms of an arithmetic progression; fill every later
      region whose number belongs to the same progression with yellow.

    Key insight:
    - Region order is defined by walking along the divider, not by sorting bounding boxes, so bends cause the
      next region to alternate between the left and right side of the path.
    """
    g, bg, path_color, _path, _region_ids, regions, region_order = _extract_structure(grid)
    paint_colors = sorted(set(map(int, np.unique(g))) - {int(bg), int(path_color)})
    assert len(paint_colors) == 1, "expected exactly one prefilled region color"
    fill_color = paint_colors[0]

    seeded_positions = []
    for order_idx, region_id in enumerate(region_order):
        if any(int(g[r, c]) == fill_color for r, c in regions[region_id]):
            seeded_positions.append(order_idx)

    assert len(seeded_positions) >= 2, "need at least two yellow seed regions"
    step = seeded_positions[1] - seeded_positions[0]
    assert step > 0, "arithmetic progression step must be positive"
    assert all(b - a == step for a, b in zip(seeded_positions, seeded_positions[1:])), "seeded regions must follow one arithmetic progression"

    out = g.copy()
    for order_idx in range(seeded_positions[0], len(region_order), step):
        for r, c in regions[region_order[order_idx]]:
            out[r, c] = fill_color
    return out


def _build_snake_path(height: int, width: int, start_col: int, turn_rows: list[int], first_side: int) -> list[tuple[int, int]]:
    r, c = 0, start_col
    path = [(r, c)]
    side = first_side
    for turn_row in turn_rows:
        while r < turn_row:
            r += 1
            path.append((r, c))
        target_col = 0 if side == 0 else width - 1
        while c != target_col:
            c += -1 if c > target_col else 1
            path.append((r, c))
        side = 1 - side
    while r < height - 1:
        r += 1
        path.append((r, c))
    return path


def _paint_region(grid: np.ndarray, region: list[tuple[int, int]], color: int) -> None:
    for r, c in region:
        grid[r, c] = color


def _sequence_options(region_count: int) -> list[tuple[int, int, list[int]]]:
    options = []
    for start in range(region_count):
        for step in range(1, region_count):
            terms = list(range(start, region_count, step))
            if 3 <= len(terms) <= 6:
                options.append((start, step, terms))
    return options


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(6, 9)
    interval_heights = [random.randint(2, 3) for _ in range(region_count)]
    height = 1 + sum(interval_heights)
    width = random.randint(10, 20)
    start_col = random.randint(max(2, width // 3), min(width - 3, (2 * width) // 3))
    turn_rows = list(np.cumsum(interval_heights[:-1]))
    first_side = random.randint(0, 1)

    path = _build_snake_path(height, width, start_col, turn_rows, first_side)
    grid = np.full((height, width), BLACK, dtype=int)
    for r, c in path:
        grid[r, c] = RED

    try:
        _, bg, path_color, _ordered_path, _region_ids, regions, region_order = _extract_structure(grid)
    except AssertionError:
        return None

    if bg != BLACK or path_color != RED or len(region_order) != region_count:
        return None

    options = _sequence_options(region_count)
    if not options:
        return None
    start, step, terms = random.choice(options)
    shown_count = random.randint(2, min(4, len(terms) - 1))
    shown_terms = terms[:shown_count]

    inp = grid.copy()
    expected = grid.copy()
    for order_idx in shown_terms:
        _paint_region(inp, regions[region_order[order_idx]], YELLOW)
    for order_idx in terms:
        _paint_region(expected, regions[region_order[order_idx]], YELLOW)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
