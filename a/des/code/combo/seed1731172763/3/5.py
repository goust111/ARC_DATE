import numpy as np
from collections import Counter


RNG = np.random.default_rng()


GEN_BG = 8
GEN_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_GRID_MIN = 16
GEN_GRID_MAX = 25
GEN_REGION_COUNT_CHOICES = (1, 2, 3)
GEN_MAX_ATTEMPTS = 300
GEN_PLACEMENT_TRIES = 80
GEN_MARGIN = 2


def _bg_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append((cells, touches_edge))
    return out


def _boundary_pixels(grid: np.ndarray, bg: int, region):
    h, w = grid.shape
    boundary = set()
    for r, c in region:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                    boundary.add((nr, nc))
    assert boundary, "enclosed region must have a non-empty boundary"
    return boundary


def _cycle_area(order) -> int:
    area = 0
    for (r1, c1), (r2, c2) in zip(order, order[1:] + order[:1]):
        area += c1 * r2 - c2 * r1
    return int(area)


def _trace_clockwise_cycle(boundary):
    boundary = set(boundary)
    adjacency = {}
    for cell in boundary:
        neighbors = [n for n in _neighbors4(*cell) if n in boundary]
        assert len(neighbors) == 2, "boundary must be a simple 4-connected cycle"
        adjacency[cell] = sorted(neighbors)

    start = min(boundary)
    best_order = None
    best_area = None
    for first in adjacency[start]:
        order = [start, first]
        prev, cur = start, first
        ok = True
        while True:
            nexts = [n for n in adjacency[cur] if n != prev]
            if len(nexts) != 1:
                ok = False
                break
            nxt = nexts[0]
            if nxt == start:
                if len(order) != len(boundary):
                    ok = False
                break
            if nxt in order:
                ok = False
                break
            order.append(nxt)
            prev, cur = cur, nxt
        if not ok:
            continue
        area = _cycle_area(order)
        if best_order is None or area > best_area:
            best_order = order
            best_area = area

    assert best_order is not None and best_area is not None
    assert best_area > 0, "failed to orient the boundary clockwise"
    return best_order


def _fill_color_for_region(grid: np.ndarray, bg: int, region) -> int:
    boundary = _boundary_pixels(grid, bg, region)
    order = _trace_clockwise_cycle(boundary)
    counts = Counter(int(grid[r, c]) for r, c in boundary)
    rare_count = min(counts.values())
    rare_colors = {color for color, count in counts.items() if count == rare_count}
    for r, c in order:
        color = int(grid[r, c])
        if color in rare_colors:
            return color
    raise AssertionError("clockwise tie-break failed to select a boundary color")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed background region with the least frequent color on its boundary.

    If multiple boundary colors tie for the minimum count, choose the one that
    appears first when the boundary is read clockwise from its top-leftmost
    boundary pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    for region, touches_edge in _components(g == bg):
        if touches_edge:
            continue
        fill_color = _fill_color_for_region(g, bg, region)
        for r, c in region:
            out[r, c] = fill_color

    return out


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rotate_cells(cells, turns: int):
    cells = set(cells)
    for _ in range(turns % 4):
        height = max(r for r, _ in cells) + 1
        cells = {(c, height - 1 - r) for r, c in cells}
        cells = _normalize(cells)
    return cells


def _rectangle_region(rng: np.random.Generator):
    height = int(rng.integers(2, 6))
    width = int(rng.integers(2, 6))
    return {(r, c) for r in range(height) for c in range(width)}


def _l_region(rng: np.random.Generator):
    height = int(rng.integers(4, 8))
    width = int(rng.integers(4, 8))
    arm_h = int(rng.integers(2, height))
    arm_w = int(rng.integers(2, width))
    cells = {(r, c) for r in range(height) for c in range(width) if r < arm_h or c < arm_w}
    return _rotate_cells(cells, int(rng.integers(4)))


def _region_and_shell(cells):
    region = set(cells)
    shell = set()
    for r, c in region:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                point = (r + dr, c + dc)
                if point not in region:
                    shell.add(point)
    all_cells = region | shell
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    region2 = {(r - min_r, c - min_c) for r, c in region}
    shell2 = {(r - min_r, c - min_c) for r, c in shell}
    return region2, shell2


def _make_object(rng: np.random.Generator):
    if rng.random() < 0.5:
        region = _rectangle_region(rng)
    else:
        region = _l_region(rng)
    region, shell = _region_and_shell(region)
    order = _trace_clockwise_cycle(shell)
    color_map = _assign_boundary_colors(rng, order)
    height = max(r for r, _ in region | shell) + 1
    width = max(c for _, c in region | shell) + 1
    return {
        "region": region,
        "shell": shell,
        "order": order,
        "colors": color_map,
        "height": height,
        "width": width,
    }


def _assign_boundary_colors(rng: np.random.Generator, order):
    length = len(order)
    run_count = 4 if length >= 18 and rng.random() < 0.5 else 3
    tie_branch = bool(rng.integers(2))
    base = [2, 2] + [3] * (run_count - 2) if tie_branch else [2] + [3] * (run_count - 1)
    if sum(base) > length:
        run_count = 3
        base = [2, 2, 3] if tie_branch else [2, 3, 3]
    assert sum(base) <= length

    colors = list(map(int, rng.choice(GEN_COLOR_POOL, size=run_count, replace=False)))
    extras = [0] * run_count
    remaining = length - sum(base)
    movable = list(range(2, run_count)) if tie_branch else list(range(1, run_count))
    while remaining > 0:
        idx = int(movable[rng.integers(len(movable))])
        extras[idx] += 1
        remaining -= 1

    run_lengths = [base_len + extra for base_len, extra in zip(base, extras)]
    perm = list(map(int, rng.permutation(run_count)))
    colors = [colors[i] for i in perm]
    run_lengths = [run_lengths[i] for i in perm]

    offset = int(rng.integers(length))
    rotated = order[offset:] + order[:offset]

    color_map = {}
    pos = 0
    for color, run_length in zip(colors, run_lengths):
        for cell in rotated[pos : pos + run_length]:
            color_map[cell] = color
        pos += run_length
    assert pos == length and len(color_map) == length
    return color_map


def _place_object(grid: np.ndarray, blocked: np.ndarray, obj, rng: np.random.Generator) -> bool:
    h, w = grid.shape
    oh, ow = obj["height"], obj["width"]
    if oh + 2 > h or ow + 2 > w:
        return False

    for _ in range(GEN_PLACEMENT_TRIES):
        top = int(rng.integers(1, h - oh))
        left = int(rng.integers(1, w - ow))
        if blocked[top : top + oh, left : left + ow].any():
            continue
        for (r, c), color in obj["colors"].items():
            grid[top + r, left + c] = color
        r0 = max(0, top - GEN_MARGIN)
        r1 = min(h, top + oh + GEN_MARGIN)
        c0 = max(0, left - GEN_MARGIN)
        c1 = min(w, left + ow + GEN_MARGIN)
        blocked[r0:r1, c0:c1] = True
        return True
    return False


def _enclosed_regions(grid: np.ndarray, bg: int):
    return [cells for cells, touches_edge in _components(grid == bg) if not touches_edge]


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
    w = int(RNG.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
    grid = np.full((h, w), GEN_BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    region_count = int(RNG.choice(GEN_REGION_COUNT_CHOICES))
    objects = [_make_object(RNG) for _ in range(region_count)]
    objects.sort(key=lambda obj: obj["height"] * obj["width"], reverse=True)

    for obj in objects:
        if not _place_object(grid, blocked, obj, RNG):
            return None

    regions = _enclosed_regions(grid, GEN_BG)
    if len(regions) != region_count:
        return None

    out = main(grid)
    if not np.array_equal(out[grid != GEN_BG], grid[grid != GEN_BG]):
        return None
    if np.array_equal(out, grid):
        return None
    for region in regions:
        filled = {int(out[r, c]) for r, c in region}
        if len(filled) != 1 or GEN_BG in filled:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
