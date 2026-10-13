import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (15, 23)
LEGEND_LEN_RANGE = (3, 5)
SEGMENT_COUNT_RANGE = (3, 6)
SEGMENT_LEN_RANGE = (1, 4)
MAX_PATH_BUILD_ATTEMPTS = 40
MAX_GENERATION_ATTEMPTS = 300

_RNG = random.Random()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _legend_candidate(
    grid: np.ndarray,
    bg: int,
    row_scan: list[int],
    grey_col: int,
    color_col: int,
    inner_col: int | None,
) -> tuple[int, np.ndarray, dict[int, int], int] | None:
    rows: list[int] = []
    colors: list[int] = []
    grey = None
    for r in row_scan:
        gv = int(grid[r, grey_col])
        cv = int(grid[r, color_col])
        if gv == bg or cv == bg or gv == cv:
            break
        if grey is None:
            grey = gv
        elif gv != grey:
            break
        rows.append(r)
        colors.append(cv)

    if grey is None or len(rows) < 2 or len(set(colors)) != len(colors):
        return None

    if inner_col is not None:
        for r in rows:
            if int(grid[r, inner_col]) != bg:
                return None

    if len(rows) < len(row_scan):
        next_r = row_scan[len(rows)]
        if int(grid[next_r, grey_col]) != bg or int(grid[next_r, color_col]) != bg:
            return None

    mask = np.zeros_like(grid, dtype=bool)
    for r in rows:
        mask[r, grey_col] = True
        mask[r, color_col] = True

    ordered_rows = sorted(rows)
    order = {int(grid[r, color_col]): idx for idx, r in enumerate(ordered_rows)}
    return int(grey), mask, order, len(rows)


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray, dict[int, int]]:
    h, w = grid.shape
    assert w >= 2, "legend needs at least two columns"

    specs = [
        (list(range(h)), 0, 1, 2 if w > 2 else None),
        (list(range(h)), w - 2, w - 1, w - 3 if w > 2 else None),
        (list(range(h - 1, -1, -1)), 0, 1, 2 if w > 2 else None),
        (list(range(h - 1, -1, -1)), w - 2, w - 1, w - 3 if w > 2 else None),
    ]

    candidates = []
    for row_scan, grey_col, color_col, inner_col in specs:
        cand = _legend_candidate(grid, bg, row_scan, grey_col, color_col, inner_col)
        if cand is not None:
            candidates.append(cand)

    assert candidates, "expected one corner legend"
    best_len = max(length for _grey, _mask, _order, length in candidates)
    best = [cand for cand in candidates if cand[3] == best_len]
    assert len(best) == 1, "legend detection should be unambiguous"
    grey, mask, order, _ = best[0]
    return grey, mask, order


def _path_endpoints(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    endpoints: list[tuple[int, int]] = []
    for cell in cells:
        deg = sum((nbr in cells) for nbr in _neighbors4(*cell))
        if deg == 1:
            endpoints.append(cell)
        else:
            assert deg == 2, "path components must be non-branching"
    assert len(endpoints) == 2, "each path must have exactly two endpoints"
    return endpoints


def _walk_path(cells: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxt = [nbr for nbr in _neighbors4(*cur) if nbr in cells and nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "simple path traversal should have one forward step"
        prev, cur = cur, nxt[0]
        ordered.append(cur)
    assert len(ordered) == len(cells), "path traversal must cover the full component"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each path by alternating endpoint colors across its color-runs.

    The corner legend provides the visible global ordering of endpoint colors.
    For a path whose endpoint colors are C1 and C2, we traverse from the
    endpoint whose color appears earlier in the legend; odd segments are painted
    with that endpoint color and even segments with the other endpoint color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    grey, legend_mask, legend_order = _detect_legend(g, bg)

    path_mask = (g != bg) & (g != grey) & (~legend_mask)
    out = g.copy()

    comps = _components(path_mask)
    assert comps, "expected at least one path"
    h, w = g.shape

    for comp in comps:
        cells = set(comp)
        endpoints = _path_endpoints(cells)

        endpoint_info = []
        for endpoint in endpoints:
            er, ec = endpoint
            marker_neighbors = []
            for mr, mc in _neighbors4(er, ec):
                if not (0 <= mr < h and 0 <= mc < w):
                    continue
                if legend_mask[mr, mc]:
                    continue
                if int(g[mr, mc]) != grey:
                    continue
                adjacent_path = [(nr, nc) for nr, nc in _neighbors4(mr, mc) if (nr, nc) in cells]
                if adjacent_path == [endpoint]:
                    marker_neighbors.append((mr, mc))
            assert len(marker_neighbors) == 1, "each endpoint must have one grey marker"
            endpoint_info.append((endpoint, int(g[er, ec])))

        ranked = sorted(
            endpoint_info,
            key=lambda item: (legend_order.get(item[1], len(legend_order)), item[0][0], item[0][1]),
        )
        start, start_color = ranked[0]
        ordered = _walk_path(cells, start)
        other_color = int(g[ordered[-1][0], ordered[-1][1]])

        seg_idx = 0
        seg_start = 0
        for i in range(1, len(ordered) + 1):
            if i == len(ordered) or int(g[ordered[i][0], ordered[i][1]]) != int(g[ordered[i - 1][0], ordered[i - 1][1]]):
                paint = start_color if seg_idx % 2 == 0 else other_color
                for j in range(seg_start, i):
                    rr, cc = ordered[j]
                    out[rr, cc] = paint
                seg_idx += 1
                seg_start = i

    return out


def _paint_legend(grid: np.ndarray, colors: list[int], corner: str) -> np.ndarray:
    h, w = grid.shape
    k = len(colors)
    mask = np.zeros_like(grid, dtype=bool)
    rows = list(range(k)) if corner in ("TL", "TR") else list(range(h - k, h))
    grey_col, color_col = (0, 1) if corner in ("TL", "BL") else (w - 2, w - 1)
    for r, color in zip(rows, colors):
        grid[r, grey_col] = GREY
        grid[r, color_col] = color
        mask[r, grey_col] = True
        mask[r, color_col] = True
    return mask


def _block_with_halo(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        blocked[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _valid_step(
    cell: tuple[int, int],
    tail: tuple[int, int],
    path_set: set[tuple[int, int]],
    blocked: np.ndarray,
) -> bool:
    h, w = blocked.shape
    r, c = cell
    if not (2 <= r < h - 2 and 2 <= c < w - 2):
        return False
    if blocked[r, c] or cell in path_set:
        return False
    for nbr in _neighbors4(r, c):
        if nbr in path_set and nbr != tail:
            return False
    return True


def _future_score(
    cell: tuple[int, int],
    tail: tuple[int, int],
    path_set: set[tuple[int, int]],
    blocked: np.ndarray,
) -> int:
    score = 0
    shadow = set(path_set)
    shadow.add(cell)
    for nbr in _neighbors4(*cell):
        if nbr == tail:
            continue
        if _valid_step(nbr, cell, shadow, blocked):
            score += 1
    return score


def _build_walk(length: int, blocked: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    starts = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2) if not blocked[r, c]]
    _RNG.shuffle(starts)
    starts = starts[: min(len(starts), 80)]

    def dfs(path: list[tuple[int, int]], path_set: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            return list(path)

        tail = path[-1]
        candidates = []
        for nbr in _neighbors4(*tail):
            if _valid_step(nbr, tail, path_set, blocked):
                score = _future_score(nbr, tail, path_set, blocked)
                candidates.append((score, _RNG.random(), nbr))
        candidates.sort(reverse=True)

        for _score, _noise, nbr in candidates:
            path.append(nbr)
            path_set.add(nbr)
            result = dfs(path, path_set)
            if result is not None:
                return result
            path.pop()
            path_set.remove(nbr)
        return None

    for start in starts:
        result = dfs([start], {start})
        if result is not None:
            return result
    return None


def _choose_marker(
    endpoint: tuple[int, int],
    path_set: set[tuple[int, int]],
    blocked: np.ndarray,
    reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = blocked.shape
    options = []
    for r, c in _neighbors4(*endpoint):
        if not (0 <= r < h and 0 <= c < w):
            continue
        if blocked[r, c] or (r, c) in path_set or (r, c) in reserved:
            continue
        adjacent_path = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in path_set]
        if adjacent_path == [endpoint]:
            options.append((r, c))
    if not options:
        return None
    _RNG.shuffle(options)
    return options[0]


def _segment_lengths(segment_count: int) -> list[int]:
    return [_RNG.randint(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1]) for _ in range(segment_count)]


def _segment_colors(segment_count: int, start_color: int, end_color: int) -> list[int]:
    colors = [start_color]
    for idx in range(1, segment_count - 1):
        choices = [c for c in PATH_COLORS if c != colors[-1]]
        if idx == segment_count - 2:
            choices = [c for c in choices if c != end_color]
        colors.append(_RNG.choice(choices))
    colors.append(end_color)
    return colors


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    segment_lengths: list[int],
    segment_colors: list[int],
    markers: list[tuple[int, int]],
) -> None:
    idx = 0
    for seg_len, color in zip(segment_lengths, segment_colors):
        for _ in range(seg_len):
            r, c = path[idx]
            grid[r, c] = color
            idx += 1
    for r, c in markers:
        grid[r, c] = GREY


def _place_path(grid: np.ndarray, blocked: np.ndarray, legend_colors: list[int]) -> bool:
    segment_count = _RNG.randint(SEGMENT_COUNT_RANGE[0], SEGMENT_COUNT_RANGE[1])
    segment_lengths = _segment_lengths(segment_count)
    path_len = sum(segment_lengths)

    start_color, end_color = _RNG.sample(legend_colors, 2)
    seg_colors = _segment_colors(segment_count, start_color, end_color)

    for _ in range(MAX_PATH_BUILD_ATTEMPTS):
        path = _build_walk(path_len, blocked)
        if path is None:
            return False
        path_set = set(path)
        marker_a = _choose_marker(path[0], path_set, blocked, set())
        if marker_a is None:
            continue
        marker_b = _choose_marker(path[-1], path_set, blocked, {marker_a})
        if marker_b is None:
            continue

        _paint_path(grid, path, segment_lengths, seg_colors, [marker_a, marker_b])
        _block_with_halo(blocked, path + [marker_a, marker_b])
        return True
    return False


def _max_paths(h: int, w: int) -> int:
    return min(4, max(1, (h * w) // 120 + 1))


def _gen_attempt() -> np.ndarray | None:
    h = _RNG.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = _RNG.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    grid = np.full((h, w), BG, dtype=int)

    legend_len = _RNG.randint(LEGEND_LEN_RANGE[0], LEGEND_LEN_RANGE[1])
    legend_colors = _RNG.sample(list(PATH_COLORS), legend_len)
    legend_corner = _RNG.choice(("TL", "TR", "BL", "BR"))

    legend_mask = _paint_legend(grid, legend_colors, legend_corner)
    blocked = np.zeros((h, w), dtype=bool)
    legend_cells = list(map(tuple, np.argwhere(legend_mask)))
    _block_with_halo(blocked, legend_cells)

    path_count = _RNG.randint(1, _max_paths(h, w))
    for _ in range(path_count):
        if not _place_path(grid, blocked, legend_colors):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
