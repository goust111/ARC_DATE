import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

PATH_COLORS = {RED, GREEN, GRAY}
MARKER_TO_OP = {
    BLUE: lambda patch: np.rot90(patch, -1),
    YELLOW: np.flipud,
    ORANGE: np.fliplr,
}

PATTERN_COLORS = (6, 8, 9)
GEN_RETRIES = 400
MIN_H, MAX_H = 16, 26
MIN_W, MAX_W = 24, 36
MIN_VERTS, MAX_VERTS = 1, 3
MIN_HSEG = 5
MIN_VSEG = 4
START_COL_RANGE = (4, 7)
END_MARGIN_RANGE = (2, 4)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _path_neighbors(grid: np.ndarray, pos: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = pos
    out = []
    for nr, nc in _neighbors4(r, c, h, w):
        if (nr, nc) == end or int(grid[nr, nc]) == GRAY:
            out.append((nr, nc))
    return out


def _find_endpoint(grid: np.ndarray, color: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        gray_neighbors = sum(int(grid[nr, nc]) == GRAY for nr, nc in _neighbors4(r, c, h, w))
        if gray_neighbors == 1:
            candidates.append((r, c))
    assert len(candidates) == 1
    return candidates[0]


def _trace_path(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    prev = None
    cur = start
    while cur != end:
        nxts = [pt for pt in _path_neighbors(grid, cur, end) if pt != prev]
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        path.append(cur)
    return path


def _turn_marker_cell(
    prev: tuple[int, int], cur: tuple[int, int], nxt: tuple[int, int]
) -> tuple[int, int] | None:
    dr_in = cur[0] - prev[0]
    dc_in = cur[1] - prev[1]
    dr_out = nxt[0] - cur[0]
    dc_out = nxt[1] - cur[1]
    if (dr_in, dc_in) == (dr_out, dc_out):
        return None
    return (cur[0] + dr_out - dr_in, cur[1] + dc_out - dc_in)


def _find_start_pattern(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    r, c = start
    candidates = []
    for r0, c0 in ((r - 1, c - 3), (r - 1, c + 1), (r - 3, c - 1), (r + 1, c - 1)):
        if r0 < 0 or c0 < 0 or r0 + 3 > h or c0 + 3 > w:
            continue
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        colored = (patch != BG) & (patch != GRAY)
        if np.any(colored) and not np.any(patch == GRAY):
            candidates.append((int(colored.sum()), patch.copy()))
    assert candidates
    best = max(score for score, _patch in candidates)
    best_patches = [patch for score, patch in candidates if score == best]
    assert len(best_patches) == 1
    return best_patches[0]


def _apply_markers(patch: np.ndarray, marker_colors: list[int]) -> np.ndarray:
    out = patch.copy()
    for color in marker_colors:
        out = MARKER_TO_OP[color](out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the grey path from the red endpoint, transform the adjacent 3x3 pattern at marked turns,
    and stamp the final pattern centered on the green endpoint.
    """
    g = np.asarray(grid, dtype=int)
    start = _find_endpoint(g, RED)
    end = _find_endpoint(g, GREEN)
    pattern = _find_start_pattern(g, start)
    path = _trace_path(g, start, end)

    marker_colors = []
    for i in range(1, len(path) - 1):
        marker_cell = _turn_marker_cell(path[i - 1], path[i], path[i + 1])
        if marker_cell is None:
            continue
        mr, mc = marker_cell
        color = int(g[mr, mc])
        if color in MARKER_TO_OP:
            marker_colors.append(color)

    final_patch = _apply_markers(pattern, marker_colors)
    out = g.copy()
    er, ec = end
    out[er - 1 : er + 2, ec - 1 : ec + 2] = final_patch
    return out


def _random_partition(total: int, parts: int, minimum: int) -> list[int] | None:
    if total < parts * minimum:
        return None
    out = [minimum] * parts
    extra = total - parts * minimum
    for _ in range(extra):
        out[random.randrange(parts)] += 1
    random.shuffle(out)
    return out


def _sample_rows(height: int, count: int) -> list[int] | None:
    lo = 3
    hi = height - 4
    if lo > hi:
        return None
    rows = [random.randint(lo, hi)]
    for _ in range(count):
        choices = [r for r in range(lo, hi + 1) if abs(r - rows[-1]) >= MIN_VSEG]
        if not choices:
            return None
        rows.append(random.choice(choices))
    return rows


def _build_path_cells(rows: list[int], start_col: int, horizontal_lengths: list[int]) -> list[tuple[int, int]]:
    r = rows[0]
    c = start_col
    path = [(r, c)]
    for i, hlen in enumerate(horizontal_lengths):
        for _ in range(hlen):
            c += 1
            path.append((r, c))
        if i == len(horizontal_lengths) - 1:
            break
        target_r = rows[i + 1]
        step = 1 if target_r > r else -1
        while r != target_r:
            r += step
            path.append((r, c))
    return path


def _turn_indices(path: list[tuple[int, int]]) -> list[int]:
    out = []
    for i in range(1, len(path) - 1):
        if _turn_marker_cell(path[i - 1], path[i], path[i + 1]) is not None:
            out.append(i)
    return out


def _choose_markers(turn_count: int) -> list[int | None]:
    marked = [None] * turn_count
    chosen_count = random.randint(1, turn_count)
    for idx in random.sample(range(turn_count), chosen_count):
        marked[idx] = random.choice((BLUE, YELLOW, ORANGE))
    return marked


def _make_pattern(marker_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(200):
        palette = random.sample(PATTERN_COLORS, random.randint(2, len(PATTERN_COLORS)))
        patch = np.array([[random.choice(palette) for _ in range(3)] for _ in range(3)], dtype=int)
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        if np.array_equal(patch, np.rot90(patch, -1)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        final_patch = _apply_markers(patch, marker_colors)
        if np.array_equal(final_patch, patch):
            continue
        return patch, final_patch
    return None


def _pattern_origins(start: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    return [(r - 1, c - 3), (r - 3, c - 1), (r + 1, c - 1)]


def _stamp_pattern(
    grid: np.ndarray, path_cells: set[tuple[int, int]], patch: np.ndarray, start: tuple[int, int]
) -> bool:
    h, w = grid.shape
    origins = _pattern_origins(start)
    random.shuffle(origins)
    for r0, c0 in origins:
        if r0 < 0 or c0 < 0 or r0 + 3 > h or c0 + 3 > w:
            continue
        cells = {(r0 + rr, c0 + cc) for rr in range(3) for cc in range(3)}
        if cells & path_cells:
            continue
        if any(grid[r, c] != BG for r, c in cells):
            continue
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
        return True
    return False


def _symmetry_functions():
    return [
        lambda a: a.copy(),
        lambda a: np.rot90(a, 1).copy(),
        lambda a: np.rot90(a, 2).copy(),
        lambda a: np.rot90(a, 3).copy(),
        lambda a: np.fliplr(a).copy(),
        lambda a: np.flipud(a).copy(),
        lambda a: a.T.copy(),
        lambda a: np.fliplr(a.T).copy(),
    ]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_H, MAX_H)
    width = random.randint(MIN_W, MAX_W)
    vertical_count = random.randint(MIN_VERTS, MAX_VERTS)

    rows = _sample_rows(height, vertical_count)
    if rows is None:
        return None

    start_col = random.randint(*START_COL_RANGE)
    end_margin = random.randint(*END_MARGIN_RANGE)
    end_col = width - 1 - end_margin
    horizontal_total = end_col - start_col
    horizontal_lengths = _random_partition(horizontal_total, vertical_count + 1, MIN_HSEG)
    if horizontal_lengths is None:
        return None

    path = _build_path_cells(rows, start_col, horizontal_lengths)
    path_set = set(path)
    if len(path_set) != len(path):
        return None

    turn_ids = _turn_indices(path)
    if len(turn_ids) < 2:
        return None

    marker_slots = _choose_markers(len(turn_ids))
    marker_colors = [color for color in marker_slots if color is not None]
    assert marker_colors

    built = _make_pattern(marker_colors)
    if built is None:
        return None
    pattern, final_pattern = built

    grid = np.full((height, width), BG, dtype=int)
    for r, c in path[1:-1]:
        grid[r, c] = GRAY
    grid[path[0]] = RED
    grid[path[-1]] = GREEN

    for turn_index, marker_color in zip(turn_ids, marker_slots):
        if marker_color is None:
            continue
        mr, mc = _turn_marker_cell(path[turn_index - 1], path[turn_index], path[turn_index + 1])
        if mr <= 0 or mc <= 0 or mr >= height - 1 or mc >= width - 1:
            return None
        if grid[mr, mc] != BG:
            return None
        grid[mr, mc] = marker_color

    if not _stamp_pattern(grid, path_set, pattern, path[0]):
        return None

    end_r, end_c = path[-1]
    if not (1 <= end_r < height - 1 and 1 <= end_c < width - 1):
        return None

    expected = grid.copy()
    expected[end_r - 1 : end_r + 2, end_c - 1 : end_c + 2] = final_pattern

    transform = random.choice(_symmetry_functions())
    sym_grid = transform(grid)
    sym_expected = transform(expected)

    try:
        if not np.array_equal(main(sym_grid), sym_expected):
            return None
    except Exception:
        return None
    return sym_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for task 3")
