import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
SEGMENT_COLOR_POOL = (1, 7, 8, 9)

MIN_SEGMENTS = 2
MAX_SEGMENTS = 4
MIN_BODY_LEN = 3
MAX_BODY_LEN = 8
MIN_MARGIN = 1
MAX_MARGIN = 6
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400

RNG = np.random.default_rng()
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # right, down, left, up


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _trace_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    coords = {tuple(map(int, p)) for p in np.argwhere(mask)}
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        neighbor_map[cell] = [nbr for nbr in _neighbors4(cell) if nbr in coords]

    assert len(neighbor_map[start]) == 1
    assert len(neighbor_map[end]) == 1
    for cell, neighbors in neighbor_map.items():
        if cell in (start, end):
            continue
        assert len(neighbors) == 2

    path = [start]
    prev = None
    cur = start
    while cur != end:
        candidates = [nbr for nbr in neighbor_map[cur] if nbr != prev]
        assert len(candidates) == 1
        nxt = candidates[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(coords)
    return path


def _compress_color_runs(path: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    run_start = 0
    prev_color = int(grid[path[0]])
    for idx in range(1, len(path)):
        color = int(grid[path[idx]])
        if color != prev_color:
            runs.append((run_start, idx, prev_color))
            run_start = idx
            prev_color = color
    runs.append((run_start, len(path), prev_color))
    return runs


def _is_straight(points: list[tuple[int, int]]) -> bool:
    rows = {r for r, _ in points}
    cols = {c for _, c in points}
    return len(rows) == 1 or len(cols) == 1


def _turn_color(v1: tuple[int, int], v2: tuple[int, int]) -> int:
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    assert cross != 0
    return YELLOW if cross < 0 else MAGENTA


def main(grid: np.ndarray) -> np.ndarray:
    """
    Join color-relay path segments into one grey path and mark turns by handedness.

    Input structure
    - bg = black.
    - A single simple 4-connected path runs from one green start marker to one red end marker.
    - The path is partitioned into straight colored runs: each run is a segment body plus the
      incoming junction marker whose color equals that segment's body color.

    Rule
    - Trace the path from the green marker to the red marker.
    - Recolor the whole traced path grey.
    - Any internal path pixel where the direction changes becomes yellow for a clockwise turn
      and magenta for a counterclockwise turn.

    Key insight
    - The non-black cells encode one relay chain; segment colors only delimit the straight runs,
      while the output depends on the geometry of the reconstructed full path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    green_pts = np.argwhere(g == GREEN)
    red_pts = np.argwhere(g == RED)
    assert green_pts.shape[0] == 1
    assert red_pts.shape[0] == 1

    start = tuple(map(int, green_pts[0]))
    end = tuple(map(int, red_pts[0]))
    mask = g != BG
    path = _trace_path(mask, start, end)

    runs = _compress_color_runs(path, g)
    assert runs[0][2] == GREEN
    assert runs[-1][2] == RED

    seen_segment_colors: set[int] = set()
    for run_start, run_end, color in runs[1:-1]:
        assert color not in {BG, RED, GREEN}
        assert color not in seen_segment_colors
        seen_segment_colors.add(color)
        points = path[run_start:run_end]
        assert len(points) >= MIN_BODY_LEN
        assert _is_straight(points)

    out = np.full_like(g, BG)
    for r, c in path:
        out[r, c] = GREY

    for idx in range(1, len(path) - 1):
        r0, c0 = path[idx - 1]
        r1, c1 = path[idx]
        r2, c2 = path[idx + 1]
        v1 = (r1 - r0, c1 - c0)
        v2 = (r2 - r1, c2 - c1)
        if v1 != v2:
            out[r1, c1] = _turn_color(v1, v2)

    return out


def _sample_turns(n_segments: int) -> list[int]:
    if n_segments <= 1:
        return []
    if n_segments == 2:
        return [int(RNG.choice(np.array([-1, 0, 1], dtype=int)))]

    base = np.array([-1, 0, 1], dtype=int)
    turns: list[int] = []
    while len(turns) < n_segments - 1:
        perm = list(map(int, RNG.permutation(base)))
        turns.extend(perm)
    return turns[: n_segments - 1]


def _add_segment(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    direction: tuple[int, int],
    body_len: int,
) -> bool:
    cur = path[-1]
    for _ in range(body_len + 1):
        nxt = (cur[0] + direction[0], cur[1] + direction[1])
        if nxt in occupied:
            return False
        occupied_neighbors = [nbr for nbr in _neighbors4(nxt) if nbr in occupied]
        if occupied_neighbors != [cur]:
            return False
        path.append(nxt)
        occupied.add(nxt)
        cur = nxt
    return True


def _build_path(
    body_lengths: list[int],
    turns: list[int],
    start_dir_idx: int,
) -> list[tuple[int, int]] | None:
    dir_idx = int(start_dir_idx)
    path = [(0, 0)]
    occupied = {(0, 0)}
    for seg_idx, body_len in enumerate(body_lengths):
        direction = DIRS[dir_idx]
        if not _add_segment(path, occupied, direction, int(body_len)):
            return None
        if seg_idx < len(turns):
            dir_idx = (dir_idx + int(turns[seg_idx])) % 4
    return path


def _path_indices(body_lengths: list[int]) -> list[int]:
    indices = [0]
    cur = 0
    for body_len in body_lengths:
        cur += int(body_len) + 1
        indices.append(cur)
    return indices


def _paint_path(
    path: list[tuple[int, int]],
    body_lengths: list[int],
    segment_colors: list[int],
    top: int,
    left: int,
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    shifted = [(r + top, c + left) for r, c in path]
    cut_points = _path_indices(body_lengths)

    sr, sc = shifted[0]
    grid[sr, sc] = GREEN
    for seg_idx, color in enumerate(segment_colors):
        start = cut_points[seg_idx]
        end = cut_points[seg_idx + 1]
        for idx in range(start + 1, end):
            r, c = shifted[idx]
            grid[r, c] = int(color)
        jr, jc = shifted[end]
        grid[jr, jc] = RED if seg_idx == len(segment_colors) - 1 else int(segment_colors[seg_idx + 1])

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_segments = int(RNG.integers(MIN_SEGMENTS, MAX_SEGMENTS + 1))
    segment_colors = list(map(int, RNG.choice(np.array(SEGMENT_COLOR_POOL, dtype=int), size=n_segments, replace=False)))
    body_lengths = list(map(int, RNG.integers(MIN_BODY_LEN, MAX_BODY_LEN + 1, size=n_segments)))
    turns = _sample_turns(n_segments)
    start_dir_idx = int(RNG.integers(0, len(DIRS)))

    path = _build_path(body_lengths, turns, start_dir_idx)
    if path is None:
        return None

    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1

    top_margin = int(RNG.integers(MIN_MARGIN, MAX_MARGIN + 1))
    bottom_margin = int(RNG.integers(MIN_MARGIN, MAX_MARGIN + 1))
    left_margin = int(RNG.integers(MIN_MARGIN, MAX_MARGIN + 1))
    right_margin = int(RNG.integers(MIN_MARGIN, MAX_MARGIN + 1))
    height = box_h + top_margin + bottom_margin
    width = box_w + left_margin + right_margin
    if height > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        return None

    top = top_margin - min_r
    left = left_margin - min_c
    grid = _paint_path(path, body_lengths, segment_colors, top, left, (height, width))

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
