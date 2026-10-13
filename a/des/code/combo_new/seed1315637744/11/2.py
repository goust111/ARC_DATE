import random
from collections import Counter

import numpy as np


BG = 0
PATH = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

RUN_COUNT_CHOICES = (3, 4, 5)
ROW_GAP_RANGE = (4, 5)
PAD_RANGE = (3, 4)
LOW_BAND_MAX_RANGE = (2, 5)
HIGH_BAND_GAP_RANGE = (5, 8)
HIGH_BAND_WIDTH_RANGE = (2, 4)
MIN_HORIZONTAL_RUN = 4
MAX_GENERATION_ATTEMPTS = 400

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _path_neighbors(mask: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    neighbors = []
    for dr, dc in DIRS:
        rr, cc = row + dr, col + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            neighbors.append((rr, cc))
    return neighbors


def _direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (int(b[0] - a[0]), int(b[1] - a[1]))


def _is_corner_dirs(dirs: list[tuple[int, int]]) -> bool:
    assert len(dirs) == 2
    dr = dirs[0][0] + dirs[1][0]
    dc = dirs[0][1] + dirs[1][1]
    return (dr, dc) != (0, 0)


def _analyze_path(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    mask = g != bg
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected one foreground path"

    colors = [int(x) for x in np.unique(g[mask])]
    path_color = int(Counter(map(int, g[mask].reshape(-1))).most_common(1)[0][0])

    # The whole non-background structure must be one thin path.
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        row, col = stack.pop()
        for nxt in _path_neighbors(mask, row, col):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert len(seen) == len(coords), "expected a single connected path"

    endpoints = []
    corner_dirs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in seen:
        neighbors = _path_neighbors(mask, row, col)
        degree = len(neighbors)
        assert degree in (1, 2), "path must have no branches"
        if degree == 1:
            endpoints.append((row, col))
        else:
            dirs = [_direction((row, col), nxt) for nxt in neighbors]
            if _is_corner_dirs(dirs):
                corner_dirs[(row, col)] = dirs

    assert len(endpoints) == 2, "path must have exactly two endpoints"

    marked_corners = {
        pos: int(g[pos])
        for pos in corner_dirs
        if int(g[pos]) != path_color
    }
    non_path_color_cells = {
        tuple(map(int, cell))
        for cell in np.argwhere(mask & (g != path_color))
    }
    assert set(marked_corners) == non_path_color_cells, "only corner pixels may carry marker colors"
    assert len(set(marked_corners.values())) == len(marked_corners), "marker colors must be unique"

    return {
        "grid": g,
        "bg": int(bg),
        "path_color": int(path_color),
        "mask": mask,
        "endpoints": sorted(endpoints),
        "corner_dirs": corner_dirs,
        "marked_corners": marked_corners,
        "colors": colors,
    }


def _trace_path(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    order = [start]
    prev = None
    cur = start
    while True:
        neighbors = _path_neighbors(mask, cur[0], cur[1])
        nxts = [nxt for nxt in neighbors if nxt != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "simple path traversal should have one forward step"
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _choose_walk_order(info: dict) -> list[tuple[int, int]]:
    e0, e1 = info["endpoints"]
    order = _trace_path(info["mask"], e0)
    assert order[-1] == e1

    marked = set(info["marked_corners"])
    if not marked:
        return order

    marker_indices = [i for i, pos in enumerate(order) if pos in marked]
    left_prefix = int(marker_indices[0])
    right_prefix = int(len(order) - 1 - marker_indices[-1])
    if left_prefix > right_prefix:
        return order
    if right_prefix > left_prefix:
        return list(reversed(order))
    return order if e0 <= e1 else list(reversed(order))


def _extension_cells(
    corner: tuple[int, int], used_dirs: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    used = set(used_dirs)
    cells = []
    for dr, dc in DIRS:
        if (dr, dc) in used:
            continue
        for step in (1, 2):
            cells.append((corner[0] + dr * step, corner[1] + dc * step))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey path by corner markers and add same-color cross arms at those markers.

    Input structure:
    - The background is black.
    - One connected one-pixel-wide orthogonal path runs from one endpoint to another.
    - Most path pixels are grey; some corner pixels are replaced by unique non-grey marker colors.

    Rule:
    - Recover the path order and compare the two endpoint-side prefixes before the first marker; walk from the endpoint with the longer prefix.
    - Keep both non-corner endpoints grey.
    - As the walk proceeds, each encountered marker color becomes the active color, and every later grey non-endpoint path pixel keeps that color until a new marker is reached.
    - At every colored corner, extend the marker color by two pixels along the two directions not already occupied by the path.

    Key insight:
    - The colored corner pixels act as ordered paint switches on a single polyline, and each marked corner also reveals where the missing two arms of a local cross must be added.
    """
    info = _analyze_path(np.asarray(grid, dtype=int))
    g = info["grid"]
    h, w = g.shape
    out = g.copy()

    order = _choose_walk_order(info)
    endpoints = {order[0], order[-1]}
    active_color = None
    for pos in order:
        if pos in endpoints:
            continue
        if pos in info["marked_corners"]:
            active_color = int(info["marked_corners"][pos])
            continue
        if int(g[pos]) == info["path_color"] and active_color is not None:
            out[pos] = int(active_color)

    for corner, color in info["marked_corners"].items():
        for rr, cc in _extension_cells(corner, info["corner_dirs"][corner]):
            assert 0 <= rr < h and 0 <= cc < w, "cross extension must stay inside the grid"
            out[rr, cc] = int(color)

    return out


def _vertices_to_order(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    order = [vertices[0]]
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for col in range(c0 + step, c1 + step, step):
                order.append((r0, col))
        else:
            step = 1 if r1 > r0 else -1
            for row in range(r0 + step, r1 + step, step):
                order.append((row, c0))
    return order


def _build_local_path() -> tuple[list[tuple[int, int]], list[int]] | None:
    run_count = random.choice(RUN_COUNT_CHOICES)
    rows = [0]
    for _ in range(run_count - 1):
        rows.append(rows[-1] + random.randint(*ROW_GAP_RANGE))

    low_max = random.randint(*LOW_BAND_MAX_RANGE)
    high_min = low_max + random.randint(*HIGH_BAND_GAP_RANGE)
    high_max = high_min + random.randint(*HIGH_BAND_WIDTH_RANGE)
    start_low = random.choice([True, False])

    xs = []
    for i in range(run_count + 1):
        use_low = (i % 2 == 0 and start_low) or (i % 2 == 1 and not start_low)
        if use_low:
            xs.append(random.randint(0, low_max))
        else:
            xs.append(random.randint(high_min, high_max))

    if min(abs(xs[i + 1] - xs[i]) for i in range(run_count)) < MIN_HORIZONTAL_RUN:
        return None

    vertices = [(rows[0], xs[0])]
    for i in range(run_count):
        vertices.append((rows[i], xs[i + 1]))
        if i < run_count - 1:
            vertices.append((rows[i + 1], xs[i + 1]))

    order = _vertices_to_order(vertices)
    if len(order) != len(set(order)):
        return None

    corner_indices = []
    for i in range(1, len(order) - 1):
        prev_pos, cur_pos, next_pos = order[i - 1], order[i], order[i + 1]
        dir1 = _direction(cur_pos, prev_pos)
        dir2 = _direction(cur_pos, next_pos)
        if _is_corner_dirs([dir1, dir2]):
            corner_indices.append(i)

    if len(corner_indices) < 2:
        return None

    return order, corner_indices


def _choose_marker_indices(corner_indices: list[int], path_len: int) -> list[int] | None:
    max_markers = min(5, len(corner_indices))
    want_left_larger = random.choice([True, False])
    schemes = ["ends", "alternate", "random"]

    for _ in range(200):
        scheme = random.choice(schemes)
        count = random.randint(2, max_markers)

        if scheme == "ends":
            first_pool = corner_indices[: min(2, len(corner_indices))]
            last_pool = corner_indices[max(0, len(corner_indices) - 2) :]
            chosen = {random.choice(first_pool), random.choice(last_pool)}
            middle = [idx for idx in corner_indices if idx not in chosen]
            extra = min(len(middle), count - len(chosen))
            if extra > 0:
                chosen.update(random.sample(middle, extra))
            selected = sorted(chosen)
        elif scheme == "alternate":
            parity = random.choice([0, 1])
            pool = corner_indices[parity::2] or corner_indices[1 - parity :: 2]
            if len(pool) < 2:
                continue
            selected = sorted(random.sample(pool, min(count, len(pool))))
        else:
            selected = sorted(random.sample(corner_indices, count))

        left_prefix = selected[0]
        right_prefix = path_len - 1 - selected[-1]
        if left_prefix == right_prefix:
            continue
        if (left_prefix > right_prefix) != want_left_larger:
            continue
        return selected

    return None


def _apply_random_transform(grid: np.ndarray) -> np.ndarray:
    transformed = np.asarray(grid, dtype=int)
    for _ in range(random.randrange(4)):
        transformed = np.rot90(transformed)
    if random.choice([True, False]):
        transformed = np.fliplr(transformed)
    return np.asarray(transformed, dtype=int)


def _validate_generated(grid: np.ndarray) -> bool:
    info = _analyze_path(grid)
    order = _choose_walk_order(info)
    marker_positions = set(info["marked_corners"])
    marker_indices = [i for i, pos in enumerate(order) if pos in marker_positions]
    if len(marker_indices) < 2:
        return False
    if marker_indices[0] == len(order) - 1 - marker_indices[-1]:
        return False

    bg = info["bg"]
    extension_cells = set()
    for corner, dirs in info["corner_dirs"].items():
        if corner not in info["marked_corners"]:
            continue
        for rr, cc in _extension_cells(corner, dirs):
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                return False
            if int(grid[rr, cc]) != bg:
                return False
            if (rr, cc) in extension_cells:
                return False
            extension_cells.add((rr, cc))

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    built = _build_local_path()
    if built is None:
        return None
    order, corner_indices = built

    marker_indices = _choose_marker_indices(corner_indices, len(order))
    if marker_indices is None:
        return None

    rows = [pos[0] for pos in order]
    cols = [pos[1] for pos in order]
    local_h = max(rows) + 1
    local_w = max(cols) + 1
    pad_top = random.randint(*PAD_RANGE)
    pad_bottom = random.randint(*PAD_RANGE)
    pad_left = random.randint(*PAD_RANGE)
    pad_right = random.randint(*PAD_RANGE)
    shape = (local_h + pad_top + pad_bottom, local_w + pad_left + pad_right)
    if max(shape) > 30:
        return None

    grid = np.full(shape, BG, dtype=int)
    for row, col in order:
        grid[row + pad_top, col + pad_left] = PATH

    marker_colors = random.sample(MARKER_COLORS, len(marker_indices))
    for idx, color in zip(marker_indices, marker_colors):
        row, col = order[idx]
        grid[row + pad_top, col + pad_left] = int(color)

    grid = _apply_random_transform(grid)
    if max(grid.shape) > 30:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
