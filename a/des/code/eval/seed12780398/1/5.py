import numpy as np
from collections import deque


BLACK = 0
YELLOW = 4
GRAY = 5
COLOR_POOL = [1, 2, 3, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS)}
RNG = np.random.default_rng()

GRID_SHAPES = [(2, 2), (2, 3), (3, 2)]
CELL_SIZES = [4, 5]
MARKER_SPACING = 3
TAIL_LENGTH = 1
MAX_GENERATION_ATTEMPTS = 400

STAMP_TEMPLATES = [
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 0), (1, 1)],
    [(0, 1), (1, 1), (2, 1), (2, 0)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
    [(0, 2), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)],
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_panel(grid: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    gray_pts = np.argwhere(grid == GRAY)
    assert gray_pts.size > 0, "grey panel separators must exist"

    r0, c0 = gray_pts.min(axis=0)
    r1, c1 = gray_pts.max(axis=0)
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = [r for r in range(panel.shape[0]) if np.all(panel[r, :] == GRAY)]
    sep_cols = [c for c in range(panel.shape[1]) if np.all(panel[:, c] == GRAY)]
    assert len(sep_rows) >= 3 and len(sep_cols) >= 3, "panel must be a multi-cell grid"

    row_steps = np.diff(sep_rows)
    col_steps = np.diff(sep_cols)
    assert row_steps.size > 0 and col_steps.size > 0
    assert np.all(row_steps == row_steps[0]) and np.all(col_steps == col_steps[0])
    assert int(row_steps[0]) == int(col_steps[0]), "cells must be square"

    cell_size = int(row_steps[0] - 1)
    assert cell_size >= 2

    stamps: dict[int, np.ndarray] = {}
    for i in range(len(sep_rows) - 1):
        for j in range(len(sep_cols) - 1):
            rr0 = sep_rows[i] + 1
            rr1 = sep_rows[i + 1]
            cc0 = sep_cols[j] + 1
            cc1 = sep_cols[j + 1]
            cell = panel[rr0:rr1, cc0:cc1].copy()
            assert cell.shape == (cell_size, cell_size)
            colors = sorted(set(map(int, np.unique(cell))) - {BLACK})
            assert len(colors) == 1, "each stamp cell must contain exactly one non-black color"
            color = colors[0]
            assert color != YELLOW and color != GRAY
            assert color not in stamps, "stamp colors must be unique"
            stamps[color] = cell

    assert len(stamps) == (len(sep_rows) - 1) * (len(sep_cols) - 1)
    return stamps, cell_size


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = (grid != BLACK) & (grid != GRAY)
    comps = _components(mask)
    assert comps, "path component must exist"
    path = max(comps, key=len)
    path_set = set(path)

    def degree(pt: tuple[int, int]) -> int:
        r, c = pt
        return sum((nr, nc) in path_set for nr, nc in _neighbors4(r, c))

    endpoints = [pt for pt in path if degree(pt) == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    def yellow_neighbor_count(pt: tuple[int, int]) -> int:
        r, c = pt
        total = 0
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == YELLOW:
                total += 1
        return total

    starts = [pt for pt in endpoints if yellow_neighbor_count(pt) == 1]
    assert len(starts) == 1, "start endpoint must be uniquely identifiable"
    start = starts[0]

    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [nb for nb in _neighbors4(*cur) if nb in path_set and nb != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must not branch"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == len(path), "tracing must cover the whole path"
    return ordered


def _turn_delta(path: list[tuple[int, int]], start_idx: int, end_idx: int) -> int:
    dirs = [
        DIR_TO_IDX[(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])]
        for i in range(start_idx, end_idx)
    ]
    deltas: list[int] = []
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i - 1]:
            continue
        diff = (dirs[i] - dirs[i - 1]) % 4
        assert diff in (1, 3), "only quarter turns are allowed"
        deltas.append(1 if diff == 1 else -1)
    if not deltas:
        return 0
    assert len(deltas) == 1, "at most one turn may occur between adjacent markers"
    return deltas[0]


def _compose_output(
    stamps: dict[int, np.ndarray],
    color_order: list[int],
    turn_deltas: list[int],
    cell_size: int,
) -> np.ndarray:
    count = len(color_order)
    out = np.full((cell_size, count * cell_size + max(count - 1, 0)), BLACK, dtype=int)
    orientation = 0
    for i, color in enumerate(color_order):
        if i > 0:
            orientation = (orientation + turn_deltas[i - 1]) % 4
        stamp = np.rot90(stamps[color], -orientation)
        c0 = i * (cell_size + 1)
        out[:, c0 : c0 + cell_size] = stamp
        if i + 1 < count:
            out[:, c0 + cell_size] = GRAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read stamps from the grey-divided panel, trace the marked yellow path, and
    lay the stamps out on a horizontal strip while accumulating 90-degree
    rotations from the path's left/right turns.
    """
    g = np.asarray(grid, dtype=int)
    stamps, cell_size = _parse_panel(g)
    path = _trace_path(g)

    marker_indices: list[int] = []
    color_order: list[int] = []
    for idx, (r, c) in enumerate(path):
        color = int(g[r, c])
        if color == YELLOW:
            continue
        marker_indices.append(idx)
        color_order.append(color)

    assert len(color_order) == len(stamps), "marker count must match stamp count"
    assert set(color_order) == set(stamps), "marker colors must match the stamp colors"

    turn_deltas = [
        _turn_delta(path, marker_indices[i], marker_indices[i + 1])
        for i in range(len(marker_indices) - 1)
    ]
    return _compose_output(stamps, color_order, turn_deltas, cell_size)


def _render_stamp(cell_size: int, color: int, template: list[tuple[int, int]]) -> np.ndarray:
    base = np.zeros((3, 3), dtype=int)
    for r, c in template:
        base[r, c] = color
    stamp = np.rot90(base, int(RNG.integers(4)))
    out = np.zeros((cell_size, cell_size), dtype=int)
    max_off = cell_size - 3
    off_r = int(RNG.integers(max_off + 1))
    off_c = int(RNG.integers(max_off + 1))
    out[off_r : off_r + 3, off_c : off_c + 3][stamp != 0] = color
    return out


def _build_panel(rows: int, cols: int, cell_size: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    panel_h = rows * cell_size + rows + 1
    panel_w = cols * cell_size + cols + 1
    panel = np.full((panel_h, panel_w), BLACK, dtype=int)

    sep_rows = [i * (cell_size + 1) for i in range(rows + 1)]
    sep_cols = [j * (cell_size + 1) for j in range(cols + 1)]
    for r in sep_rows:
        panel[r, :] = GRAY
    for c in sep_cols:
        panel[:, c] = GRAY

    count = rows * cols
    colors = list(RNG.choice(COLOR_POOL, size=count, replace=False))
    template_ids = list(RNG.choice(len(STAMP_TEMPLATES), size=count, replace=False))

    stamps: dict[int, np.ndarray] = {}
    idx = 0
    for i in range(rows):
        for j in range(cols):
            color = int(colors[idx])
            stamp = _render_stamp(cell_size, color, STAMP_TEMPLATES[int(template_ids[idx])])
            rr0 = sep_rows[i] + 1
            cc0 = sep_cols[j] + 1
            panel[rr0 : rr0 + cell_size, cc0 : cc0 + cell_size] = stamp
            stamps[color] = stamp
            idx += 1

    return panel, stamps


def _can_add_step(
    nxt: tuple[int, int],
    prev: tuple[int, int],
    used: set[tuple[int, int]],
) -> bool:
    if nxt in used:
        return False
    for nb in _neighbors4(*nxt):
        if nb in used and nb != prev:
            return False
    return True


def _extend_path(
    coords: list[tuple[int, int]],
    used: set[tuple[int, int]],
    dir_idx: int,
    length: int,
) -> bool:
    for _ in range(length):
        dr, dc = DIRS[dir_idx]
        prev = coords[-1]
        nxt = (prev[0] + dr, prev[1] + dc)
        if not _can_add_step(nxt, prev, used):
            return False
        coords.append(nxt)
        used.add(nxt)
    return True


def _build_path(
    marker_count: int,
) -> tuple[list[tuple[int, int]], list[int], list[int], list[int]] | None:
    for _ in range(200):
        turn_codes = list(RNG.integers(0, 3, size=marker_count - 1))
        dir_idx = int(RNG.integers(4))

        coords = [(0, 0)]
        used = {(0, 0)}
        if not _extend_path(coords, used, dir_idx, MARKER_SPACING):
            continue
        marker_indices = [len(coords) - 1]
        current_dir = dir_idx
        ok = True

        for code in turn_codes:
            if code == 0:
                if not _extend_path(coords, used, current_dir, MARKER_SPACING):
                    ok = False
                    break
                turn_delta = 0
            else:
                first_leg = int(RNG.integers(1, MARKER_SPACING))
                second_leg = MARKER_SPACING - first_leg
                if not _extend_path(coords, used, current_dir, first_leg):
                    ok = False
                    break
                turn_delta = 1 if code == 1 else -1
                current_dir = (current_dir + turn_delta) % 4
                if not _extend_path(coords, used, current_dir, second_leg):
                    ok = False
                    break
            marker_indices.append(len(coords) - 1)

        if not ok or not _extend_path(coords, used, current_dir, TAIL_LENGTH):
            continue

        return coords, marker_indices, turn_codes, [0 if c == 0 else (1 if c == 1 else -1) for c in turn_codes]

    return None


def _normalize_path(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, min_c = min(rows), min(cols)
    norm = [(r - min_r, c - min_c) for r, c in coords]
    return norm, max(rows) - min_r + 1, max(cols) - min_c + 1


def _generate_attempt() -> np.ndarray | None:
    rows, cols = GRID_SHAPES[int(RNG.integers(len(GRID_SHAPES)))]
    cell_size = int(RNG.choice(CELL_SIZES))
    panel, stamps = _build_panel(rows, cols, cell_size)

    built = _build_path(len(stamps))
    if built is None:
        return None
    path_coords, marker_indices, _turn_codes, turn_deltas = built
    color_order = list(RNG.permutation(list(stamps.keys())))

    path_coords, path_h, path_w = _normalize_path(path_coords)

    panel_top = int(RNG.integers(1, 4))
    panel_left = int(RNG.integers(1, 4))
    path_top = int(RNG.integers(1, 4))
    gap = int(RNG.integers(3, 7))
    path_left = panel_left + panel.shape[1] + gap

    grid_h = max(panel_top + panel.shape[0], path_top + path_h) + int(RNG.integers(1, 4))
    grid_w = path_left + path_w + int(RNG.integers(1, 4))
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[panel_top : panel_top + panel.shape[0], panel_left : panel_left + panel.shape[1]] = panel

    for r, c in path_coords:
        grid[path_top + r, path_left + c] = YELLOW
    for idx, color in zip(marker_indices, color_order):
        r, c = path_coords[idx]
        grid[path_top + r, path_left + c] = int(color)

    expected = _compose_output(stamps, color_order, turn_deltas, cell_size)
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
