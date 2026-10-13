import random
from collections import Counter, deque

import numpy as np


GREEN = 3
RED = 2
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [0, 1, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _divider_spans(indices: list[int], limit: int) -> list[tuple[int, int]]:
    marks = [-1] + sorted(indices) + [limit]
    spans = []
    for left, right in zip(marks, marks[1:]):
        if left + 1 <= right - 1:
            spans.append((left + 1, right - 1))
    return spans


def _find_divider_color(grid: np.ndarray, bg: int) -> int:
    if np.any(grid == GREEN):
        row_hits = sum(bool(np.all(grid[r, :] == GREEN)) for r in range(grid.shape[0]))
        col_hits = sum(bool(np.all(grid[:, c] == GREEN)) for c in range(grid.shape[1]))
        if row_hits and col_hits:
            return GREEN

    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        row_hits = sum(bool(np.all(grid[r, :] == color)) for r in range(grid.shape[0]))
        col_hits = sum(bool(np.all(grid[:, c] == color)) for c in range(grid.shape[1]))
        if row_hits and col_hits:
            score = row_hits + col_hits
            if best is None or score > best[0]:
                best = (score, color)
    if best is None:
        raise ValueError("unable to find divider color")
    return int(best[1])


def _region_sign(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bounds
    red_cells = [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(grid[r, c]) == RED]
    if not red_cells:
        return None
    if len(red_cells) != 1:
        raise ValueError("each region must contain at most one red sign")
    r, c = red_cells[0]
    if r == r0 and c0 < c < c1:
        return (-1, 0)
    if r == r1 and c0 < c < c1:
        return (1, 0)
    if c == c0 and r0 < r < r1:
        return (0, -1)
    if c == c1 and r0 < r < r1:
        return (0, 1)
    raise ValueError("red sign must sit on a region edge midpoint")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = map(int, g.shape)
    bg = _get_bg(g)
    divider = _find_divider_color(g, bg)

    divider_rows = [r for r in range(h) if bool(np.all(g[r, :] == divider))]
    divider_cols = [c for c in range(w) if bool(np.all(g[:, c] == divider))]
    row_spans = _divider_spans(divider_rows, h)
    col_spans = _divider_spans(divider_cols, w)

    regions: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    signs: dict[tuple[int, int], tuple[int, int]] = {}
    patterns = []
    out = g.copy()
    out[out == RED] = bg

    order = 0
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            key = (rr, cc)
            regions[key] = (r0, c0, r1, c1)
            sign = _region_sign(g, (r0, c0, r1, c1))
            if sign is not None:
                signs[key] = sign

            rel_cells = []
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    value = int(g[r, c])
                    if value in (bg, RED):
                        continue
                    rel_cells.append((r - r0, c - c0, value))
                    out[r, c] = bg

            if rel_cells:
                patterns.append((order, key, rel_cells))
                order += 1

    n_rows = len(row_spans)
    n_cols = len(col_spans)
    placements = []
    for source_order, source_region, rel_cells in patterns:
        cur = source_region
        seen = {cur}
        steps = 0
        while cur in signs:
            dr, dc = signs[cur]
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols):
                break
            if nxt in seen:
                break
            cur = nxt
            seen.add(cur)
            steps += 1
        placements.append((steps, source_order, cur, rel_cells))

    placements.sort(key=lambda item: (item[0], item[1]))
    for _steps, _source_order, terminal, rel_cells in placements:
        r0, c0, _r1, _c1 = regions[terminal]
        for dr, dc, value in rel_cells:
            out[r0 + dr, c0 + dc] = value

    return out


def _region_origin(region: tuple[int, int], cell_h: int, cell_w: int) -> tuple[int, int]:
    return region[0] * (cell_h + 1), region[1] * (cell_w + 1)


def _region_bounds(region: tuple[int, int], cell_h: int, cell_w: int) -> tuple[int, int, int, int]:
    r0, c0 = _region_origin(region, cell_h, cell_w)
    return r0, c0, r0 + cell_h - 1, c0 + cell_w - 1


def _neighbors(region: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    out = []
    r, c = region
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n_rows and 0 <= cc < n_cols:
            out.append((rr, cc))
    return out


def _bfs_path(
    src: tuple[int, int],
    dst: tuple[int, int],
    n_rows: int,
    n_cols: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    blocked = set(blocked)
    blocked.discard(src)
    blocked.discard(dst)
    queue = deque([src])
    prev = {src: None}
    while queue:
        cur = queue.popleft()
        if cur == dst:
            break
        for nxt in _neighbors(cur, n_rows, n_cols):
            if nxt in prev or nxt in blocked:
                continue
            prev[nxt] = cur
            queue.append(nxt)
    if dst not in prev:
        return None
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _outward_dirs(region: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    r, c = region
    dirs = []
    if r == 0:
        dirs.append((-1, 0))
    if r == n_rows - 1:
        dirs.append((1, 0))
    if c == 0:
        dirs.append((0, -1))
    if c == n_cols - 1:
        dirs.append((0, 1))
    return dirs


def _sign_position(
    region: tuple[int, int],
    direction: tuple[int, int],
    cell_h: int,
    cell_w: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = _region_bounds(region, cell_h, cell_w)
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == (-1, 0):
        return r0, mid_c
    if direction == (1, 0):
        return r1, mid_c
    if direction == (0, -1):
        return mid_r, c0
    if direction == (0, 1):
        return mid_r, c1
    raise ValueError("invalid direction")


def _random_shape(cell_h: int, cell_w: int, rng: random.Random) -> list[tuple[int, int]]:
    inner = [(r, c) for r in range(1, cell_h - 1) for c in range(1, cell_w - 1)]
    start = rng.choice(inner)
    target = rng.randint(1, min(len(inner), 5 + (cell_h > 5) + (cell_w > 5)))
    cells = {start}
    frontier = set()
    while len(cells) < target:
        frontier.clear()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if 1 <= rr < cell_h - 1 and 1 <= cc < cell_w - 1 and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            break
        cells.add(rng.choice(sorted(frontier)))
    return sorted(cells)


def _paint_grid_lines(grid: np.ndarray, cell_h: int, cell_w: int) -> None:
    n_rows = (grid.shape[0] + 1) // (cell_h + 1)
    n_cols = (grid.shape[1] + 1) // (cell_w + 1)
    for idx in range(1, n_rows):
        grid[idx * (cell_h + 1) - 1, :] = GREEN
    for idx in range(1, n_cols):
        grid[:, idx * (cell_w + 1) - 1] = GREEN


def _region_canvas(bg: int, cell_h: int, cell_w: int, shape: list[tuple[int, int]], color: int) -> np.ndarray:
    region = np.full((cell_h, cell_w), bg, dtype=int)
    for r, c in shape:
        region[r, c] = color
    return region


def _extract_region(grid: np.ndarray, region: tuple[int, int], cell_h: int, cell_w: int) -> np.ndarray:
    r0, c0 = _region_origin(region, cell_h, cell_w)
    return grid[r0 : r0 + cell_h, c0 : c0 + cell_w]


def _add_path_signs(signs: dict[tuple[int, int], tuple[int, int]], path: list[tuple[int, int]]) -> bool:
    for src, dst in zip(path, path[1:]):
        step = (dst[0] - src[0], dst[1] - src[1])
        if src in signs and signs[src] != step:
            return False
        signs[src] = step
    return True


def _choose_dimensions(rng: random.Random) -> tuple[int, int, int, int]:
    n_rows = rng.randint(3, 4)
    n_cols = rng.randint(3, 4)
    cell_h = rng.choice([5, 7, 9] if n_rows == 3 else [5])
    cell_w = rng.choice([5, 7, 9] if n_cols == 3 else [5])
    return n_rows, n_cols, cell_h, cell_w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows, n_cols, cell_h, cell_w = _choose_dimensions(rng)
    all_regions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    interior_regions = [rc for rc in all_regions if 0 < rc[0] < n_rows - 1 and 0 < rc[1] < n_cols - 1]
    edge_regions = [rc for rc in all_regions if rc not in interior_regions]
    if not interior_regions:
        return None

    terminal = rng.choice(interior_regions)

    long_choices = []
    for region in all_regions:
        if region == terminal:
            continue
        path = _bfs_path(region, terminal, n_rows, n_cols, set())
        if path is not None and 2 <= len(path) - 1 <= 3:
            long_choices.append(path)
    if not long_choices:
        return None
    path_long = rng.choice(long_choices)
    used = set(path_long[:-1])

    short_choices = [[neighbor, terminal] for neighbor in _neighbors(terminal, n_rows, n_cols) if neighbor not in used]
    if not short_choices:
        return None
    path_short = rng.choice(short_choices)
    used.add(path_short[0])

    boundary_options = []
    for boundary in edge_regions:
        if boundary in used or boundary == terminal:
            continue
        for source in all_regions:
            if source in used or source in {terminal, boundary}:
                continue
            path = _bfs_path(source, boundary, n_rows, n_cols, used | {terminal})
            if path is None:
                continue
            if 1 <= len(path) - 1 <= 2:
                boundary_options.append((path, boundary))
    if not boundary_options:
        return None
    path_boundary, boundary_terminal = rng.choice(boundary_options)
    used.update(path_boundary)

    stationary_choices = [region for region in all_regions if region not in used and region not in {terminal, boundary_terminal}]
    if not stationary_choices:
        return None
    stationary_region = rng.choice(stationary_choices)
    used.add(stationary_region)

    extra_regions = [region for region in all_regions if region not in used and region not in {terminal, boundary_terminal}]
    extra_count = rng.randint(0, min(2, len(extra_regions)))
    rng.shuffle(extra_regions)
    extra_regions = extra_regions[:extra_count]

    bg = rng.choice(COLOR_POOL)
    colors = [color for color in COLOR_POOL if color != bg]
    if len(colors) < 4 + extra_count:
        return None
    rng.shuffle(colors)

    long_color = colors.pop()
    short_color = colors.pop()
    boundary_color = colors.pop()
    stationary_color = colors.pop()
    extra_colors = [colors.pop() for _ in range(extra_count)]

    shared_shape = _random_shape(cell_h, cell_w, rng)
    boundary_shape = _random_shape(cell_h, cell_w, rng)
    stationary_shape = _random_shape(cell_h, cell_w, rng)
    extra_shapes = [_random_shape(cell_h, cell_w, rng) for _ in range(extra_count)]

    signs: dict[tuple[int, int], tuple[int, int]] = {}
    if not _add_path_signs(signs, path_long):
        return None
    if not _add_path_signs(signs, path_short):
        return None
    if not _add_path_signs(signs, path_boundary):
        return None
    outward_choices = _outward_dirs(boundary_terminal, n_rows, n_cols)
    if not outward_choices:
        return None
    signs[boundary_terminal] = rng.choice(outward_choices)
    if stationary_region in signs or terminal in signs:
        return None
    if any(region in signs for region in extra_regions):
        return None

    h = n_rows * cell_h + (n_rows - 1)
    w = n_cols * cell_w + (n_cols - 1)
    grid = np.full((h, w), bg, dtype=int)
    _paint_grid_lines(grid, cell_h, cell_w)

    objects = [
        (path_long[0], shared_shape, long_color),
        (path_short[0], shared_shape, short_color),
        (path_boundary[0], boundary_shape, boundary_color),
        (stationary_region, stationary_shape, stationary_color),
    ]
    for region, shape, color in zip(extra_regions, extra_shapes, extra_colors):
        objects.append((region, shape, color))

    for region, shape, color in objects:
        r0, c0 = _region_origin(region, cell_h, cell_w)
        for dr, dc in shape:
            grid[r0 + dr, c0 + dc] = color

    for region, direction in signs.items():
        sr, sc = _sign_position(region, direction, cell_h, cell_w)
        if int(grid[sr, sc]) != bg:
            return None
        grid[sr, sc] = RED

    if _get_bg(grid) != bg:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.any(out == RED):
        return None

    empty_region = np.full((cell_h, cell_w), bg, dtype=int)
    expected_stationary = _region_canvas(bg, cell_h, cell_w, stationary_shape, stationary_color)
    expected_collision = _region_canvas(bg, cell_h, cell_w, shared_shape, long_color)
    expected_boundary = _region_canvas(bg, cell_h, cell_w, boundary_shape, boundary_color)

    if not np.array_equal(_extract_region(out, terminal, cell_h, cell_w), expected_collision):
        return None
    if not np.array_equal(_extract_region(out, boundary_terminal, cell_h, cell_w), expected_boundary):
        return None
    if not np.array_equal(_extract_region(out, stationary_region, cell_h, cell_w), expected_stationary):
        return None
    if not np.array_equal(_extract_region(out, path_long[0], cell_h, cell_w), empty_region):
        return None
    if not np.array_equal(_extract_region(out, path_short[0], cell_h, cell_w), empty_region):
        return None
    if not np.array_equal(_extract_region(out, path_boundary[0], cell_h, cell_w), empty_region):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
