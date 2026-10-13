import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _endpoint_key(cell: tuple[int, int]) -> tuple[int, int, int]:
    r, c = cell
    return (r * r + c * c, r, c)


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(grid != BG)}
    assert cells, "expected a non-empty foreground path"

    start = next(iter(cells))
    seen = set()
    stack = [start]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in _neighbors4(*cell):
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    assert seen == cells, "foreground must be one 4-connected component"

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints: list[tuple[int, int]] = []
    for cell in cells:
        nbrs = [nbr for nbr in _neighbors4(*cell) if nbr in cells]
        degree = len(nbrs)
        assert degree in (1, 2), "foreground component must be a simple path"
        adjacency[cell] = nbrs
        if degree == 1:
            endpoints.append(cell)
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    ordered: list[tuple[int, int]] = []
    prev = None
    cur = min(endpoints, key=_endpoint_key)
    while True:
        ordered.append(cur)
        nxts = [nbr for nbr in adjacency[cur] if nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must be unambiguous"
        prev, cur = cur, nxts[0]

    assert len(ordered) == len(cells), "traversal must cover the whole path"
    return ordered


def _segment_runs(grid: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    colors = [int(grid[r, c]) for r, c in path]
    assert all(color != BG for color in colors)
    assert len(set(colors)) >= 3, "path must use at least three colors"

    runs: list[tuple[int, int]] = []
    cur_color = colors[0]
    cur_len = 0
    for color in colors:
        if color == cur_color:
            cur_len += 1
        else:
            runs.append((cur_color, cur_len))
            cur_color = color
            cur_len = 1
    runs.append((cur_color, cur_len))
    return runs


def _render_histogram(runs: list[tuple[int, int]]) -> np.ndarray:
    height = max(length for _, length in runs)
    width = len(runs)
    out = np.full((height, width), BG, dtype=int)
    for c, (color, length) in enumerate(runs):
        out[height - length :, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Convert same-color path segments into bottom-aligned histogram bars."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    path = _trace_path(g)
    runs = _segment_runs(g, path)
    return _render_histogram(runs)


def _coarse_turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for a, b, c in zip(path, path[1:], path[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 != d2:
            turns += 1
    return turns


def _dfs_coarse_path(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    rows: int,
    cols: int,
    target_len: int,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        return list(path)

    cur_r, cur_c = path[-1]
    candidates = []
    for nbr in _neighbors4(cur_r, cur_c):
        nr, nc = nbr
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if nbr in used:
            continue
        onward = 0
        for rr, cc in _neighbors4(nr, nc):
            if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in used:
                onward += 1
        candidates.append((-onward, random.random(), nbr))

    candidates.sort()
    for _, _, nbr in candidates:
        path.append(nbr)
        used.add(nbr)
        result = _dfs_coarse_path(path, used, rows, cols, target_len)
        if result is not None:
            return result
        used.remove(nbr)
        path.pop()
    return None


def _random_coarse_path(
    rows: int,
    cols: int,
    target_len: int,
    require_turn: bool,
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        result = _dfs_coarse_path(path, {start}, rows, cols, target_len)
        if result is None:
            continue
        if require_turn and _coarse_turn_count(result) == 0:
            continue
        return result
    return None


def _expand_coarse_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # Doubling node coordinates and inserting midpoints guarantees a 1-cell-wide path.
    expanded = [(2 * path[0][0], 2 * path[0][1])]
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        expanded.append((r0 + r1, c0 + c1))
        expanded.append((2 * r1, 2 * c1))
    return expanded


def _sample_segment_lengths(total: int) -> list[int]:
    n_segments = random.randint(3, min(8, total))
    lengths = [1] * n_segments
    remaining = total - n_segments
    while remaining > 0:
        idx = random.randrange(n_segments)
        add = random.randint(1, min(3, remaining))
        lengths[idx] += add
        remaining -= add
    random.shuffle(lengths)
    return lengths


def _sample_segment_colors(n_segments: int) -> list[int]:
    for _ in range(100):
        palette_size = random.randint(3, min(6, len(COLOR_POOL)))
        palette = random.sample(list(COLOR_POOL), palette_size)
        colors = [random.choice(palette)]
        used = {colors[0]}
        for _ in range(n_segments - 1):
            choices = [color for color in palette if color != colors[-1]]
            nxt = random.choice(choices)
            colors.append(nxt)
            used.add(nxt)
        if len(used) >= 3:
            return colors
    raise RuntimeError("failed to sample enough segment colors")


def _paint_path(
    grid: np.ndarray,
    ordered_path: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
) -> None:
    idx = 0
    for color, length in zip(colors, lengths):
        for _ in range(length):
            r, c = ordered_path[idx]
            grid[r, c] = int(color)
            idx += 1


def _gen_attempt() -> np.ndarray | None:
    coarse_h = random.randint(3, 6)
    coarse_w = random.randint(3, 6)
    max_nodes = min(coarse_h * coarse_w, 10)
    target_nodes = random.randint(4, max_nodes)
    coarse_path = _random_coarse_path(
        coarse_h,
        coarse_w,
        target_nodes,
        require_turn=random.random() < 0.8,
    )
    if coarse_path is None:
        return None

    fine_path = _expand_coarse_path(coarse_path)
    box_h = 2 * coarse_h - 1
    box_w = 2 * coarse_w - 1
    grid_h = box_h + random.randint(0, 5)
    grid_w = box_w + random.randint(0, 5)
    top = random.randint(0, grid_h - box_h)
    left = random.randint(0, grid_w - box_w)
    placed = [(r + top, c + left) for r, c in fine_path]

    if placed[0][0] * placed[0][0] + placed[0][1] * placed[0][1] == placed[-1][0] * placed[-1][0] + placed[-1][1] * placed[-1][1]:
        return None

    if _endpoint_key(placed[-1]) < _endpoint_key(placed[0]):
        placed.reverse()

    lengths = _sample_segment_lengths(len(placed))
    colors = _sample_segment_colors(len(lengths))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _paint_path(grid, placed, lengths, colors)
    expected = _render_histogram(list(zip(colors, lengths, strict=False)))
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
