from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
NON_GREY_COLORS = [0, 1, 2, 3, 4, 6, 7, 8, 9]

GRID_ROWS_RANGE = (12, 18)
GRID_COLS_RANGE = (12, 18)
PATH_LEN_RANGE = (10, 22)
TURN_RANGE = (2, 7)
SEGMENT_STEP_RANGE = (1, 5)
RUN_COUNT_RANGE = (3, 6)
RAW_RUN_LEN_RANGE = (1, 5)
BAR_HEIGHT_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _ordered_path(path_coords: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    cells = {tuple(map(int, coord)) for coord in path_coords}
    degrees = {
        cell: sum((nbr in cells) for nbr in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    ordered = []
    prev = None
    cur = endpoints[0]
    while True:
        ordered.append(cur)
        nxt = [nbr for nbr in _neighbors4(*cur) if nbr in cells and nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unique"
        prev, cur = cur, nxt[0]

    assert len(ordered) == len(cells), "path traversal must visit every cell"
    return ordered, tuple(endpoints[0]), tuple(endpoints[1])


def _run_colors(colors: list[int]) -> list[int]:
    runs = [int(colors[0])]
    for color in colors[1:]:
        if int(color) != runs[-1]:
            runs.append(int(color))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewrite the color runs along the marked path into a vertical coded bar of equal-height segments.

    Input structure:
    - The background is the dominant color.
    - All non-background, non-grey pixels form one 4-connected one-pixel-wide path with exactly two endpoints.
    - Grey pixels form exactly two separate components: one isolated grey dot touching one endpoint, and one vertical grey line segment that does not touch the path.

    Rule:
    - Use the grey dot to choose the start endpoint, then walk the path to the opposite endpoint.
    - Compress the visited path colors into consecutive same-color runs, keeping only the run color order and discarding the original run lengths.
    - Let h be the height of the separate vertical grey segment, and output a width-1 grid whose colors follow the run order, each repeated for exactly h cells from top to bottom.

    Key insight:
    - The two grey markers have independent roles: the dot selects which end to read first, while the vertical segment supplies the fixed expansion height for every encoded color run.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg != GREY, "grey markers must be foreground"

    grey_components = _components(g == GREY)
    assert len(grey_components) == 2, "expected one grey dot and one grey line"
    grey_components.sort(key=lambda coords: int(coords.shape[0]))

    dot_coords, line_coords = grey_components
    assert int(dot_coords.shape[0]) == 1, "start marker must be one isolated grey pixel"
    assert int(line_coords.shape[0]) >= 2, "length marker must be a line segment"
    dot = tuple(map(int, dot_coords[0]))

    line_rows = sorted(map(int, line_coords[:, 0]))
    line_cols = sorted(set(map(int, line_coords[:, 1])))
    assert len(line_cols) == 1, "length marker must be vertical"
    assert line_rows == list(range(line_rows[0], line_rows[0] + len(line_rows))), "length marker must be contiguous"
    bar_height = len(line_rows)

    path_mask = (g != bg) & (g != GREY)
    path_components = _components(path_mask)
    assert len(path_components) == 1, "expected a single colored path"
    path, end_a, end_b = _ordered_path(path_components[0])

    endpoint_hits = [endpoint for endpoint in (end_a, end_b) if sum((dot == nbr) for nbr in _neighbors4(*endpoint)) == 1]
    assert len(endpoint_hits) == 1, "grey dot must touch exactly one endpoint"
    start = endpoint_hits[0]
    if path[0] != start:
        path.reverse()

    path_cells = set(path)
    for r, c in line_coords:
        assert all((int(nr), int(nc)) not in path_cells for nr, nc in _neighbors4(int(r), int(c))), "line marker must not touch path"

    colors = [int(g[r, c]) for r, c in path]
    runs = _run_colors(colors)
    assert runs != runs[::-1], "run-color sequence must differ when read backwards"

    out = np.empty((bar_height * len(runs), 1), dtype=int)
    idx = 0
    for color in runs:
        out[idx : idx + bar_height, 0] = int(color)
        idx += bar_height
    return out


def _turn_dirs(prev_dir: tuple[int, int] | None) -> list[tuple[int, int]]:
    if prev_dir is None:
        dirs = list(DIRS)
    else:
        dr, dc = prev_dir
        dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _sample_segment_lengths(path_len: int, turns: int) -> list[int]:
    total_steps = path_len - 1
    n_segments = turns + 1
    min_step, max_step = SEGMENT_STEP_RANGE
    lengths = []
    remaining = total_steps
    for i in range(n_segments):
        slots = n_segments - i - 1
        lo = max(min_step, remaining - max_step * slots)
        hi = min(max_step, remaining - min_step * slots)
        lengths.append(random.randint(lo, hi))
        remaining -= lengths[-1]
    random.shuffle(lengths)
    return lengths


def _extend_segment(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    direction: tuple[int, int],
    steps: int,
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    dr, dc = direction
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]
    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in _neighbors4(nr, nc)):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc
    return new_path, new_used


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        turns += (dr1, dc1) != (dr2, dc2)
    return turns


def _build_path(shape: tuple[int, int], path_len: int, turns: int) -> list[tuple[int, int]] | None:
    segment_lengths = _sample_segment_lengths(path_len, turns)
    h, w = shape
    starts = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(starts)

    def search(
        seg_i: int,
        prev_dir: tuple[int, int] | None,
        path: list[tuple[int, int]],
        used: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        if seg_i == len(segment_lengths):
            return path if _count_turns(path) == turns else None
        for direction in _turn_dirs(prev_dir):
            extended = _extend_segment(path, used, direction, segment_lengths[seg_i], shape)
            if extended is None:
                continue
            found = search(seg_i + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts:
        found = search(0, None, [start], {start})
        if found is not None:
            return found
    return None


def _sample_raw_run_lengths(run_count: int, path_len: int) -> list[int]:
    min_len, max_len = RAW_RUN_LEN_RANGE
    lengths = []
    remaining = path_len
    for i in range(run_count):
        slots = run_count - i - 1
        lo = max(min_len, remaining - max_len * slots)
        hi = min(max_len, remaining - min_len * slots)
        lengths.append(random.randint(lo, hi))
        remaining -= lengths[-1]
    random.shuffle(lengths)
    if max(lengths) == 1:
        return []
    if len(set(lengths)) == 1:
        for i, value in enumerate(lengths):
            if value > min_len:
                j = (i + 1) % len(lengths)
                lengths[i] -= 1
                lengths[j] += 1
                break
    return lengths if len(set(lengths)) >= 2 else []


def _sample_run_colors(run_count: int, palette: list[int]) -> list[int]:
    if random.random() < 0.5 and run_count >= 4:
        colors = random.sample(palette, run_count)
        i, j = sorted(random.sample(range(run_count), 2))
        if j == i + 1:
            j = run_count - 1
            i = 0
        colors[j] = colors[i]
        for k in range(1, run_count):
            if colors[k] == colors[k - 1]:
                for replacement in palette:
                    if replacement != colors[k - 1] and (k + 1 == run_count or replacement != colors[k + 1]):
                        colors[k] = replacement
                        break
    else:
        colors = [random.choice(palette)]
        for _ in range(1, run_count):
            choices = [color for color in palette if color != colors[-1]]
            colors.append(random.choice(choices))
    if colors == colors[::-1]:
        for i in range(run_count - 1, -1, -1):
            for replacement in palette:
                if replacement == colors[i]:
                    continue
                if i > 0 and replacement == colors[i - 1]:
                    continue
                if i + 1 < run_count and replacement == colors[i + 1]:
                    continue
                colors[i] = replacement
                if colors != colors[::-1]:
                    return colors
        return []
    return colors


def _expand_runs(run_colors: list[int], raw_lengths: list[int]) -> list[int]:
    expanded = []
    for color, length in zip(run_colors, raw_lengths):
        expanded.extend([int(color)] * int(length))
    return expanded


def _dot_candidates(endpoint: tuple[int, int], path_cells: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for nr, nc in _neighbors4(*endpoint):
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) in path_cells:
            continue
        if sum((adj in path_cells) for adj in _neighbors4(nr, nc)) == 1:
            out.append((nr, nc))
    random.shuffle(out)
    return out


def _line_candidates(
    grid: np.ndarray,
    bg: int,
    path_cells: set[tuple[int, int]],
    dot: tuple[int, int],
    bar_height: int,
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        for r0 in range(h - bar_height + 1):
            cells = [(r0 + k, c) for k in range(bar_height)]
            if any(grid[r, c0] != bg for r, c0 in cells):
                continue
            blocked = False
            for r, c0 in cells:
                if (r, c0) == dot:
                    blocked = True
                    break
                for nr, nc in _neighbors4(r, c0):
                    if (nr, nc) == dot or (nr, nc) in path_cells:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, raw_run_lengths: list[int], bar_height: int) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if expected.ndim != 2 or max(expected.shape) > 30:
        return False
    if len(set(raw_run_lengths)) < 2:
        return False
    if max(raw_run_lengths) <= 1:
        return False
    if expected.shape[1] != 1 or expected.shape[0] % bar_height != 0:
        return False
    bg = _get_bg(inp)
    grey_components = _components(inp == GREY)
    if len(grey_components) != 2:
        return False
    grey_components.sort(key=lambda coords: int(coords.shape[0]))
    if int(grey_components[0].shape[0]) != 1:
        return False
    line_coords = grey_components[1]
    line_rows = sorted(map(int, line_coords[:, 0]))
    line_cols = sorted(set(map(int, line_coords[:, 1])))
    if len(line_cols) != 1 or len(line_rows) != bar_height:
        return False
    if line_rows != list(range(line_rows[0], line_rows[0] + bar_height)):
        return False
    path_components = _components((inp != bg) & (inp != GREY))
    if len(path_components) != 1:
        return False
    try:
        path, end_a, end_b = _ordered_path(path_components[0])
    except AssertionError:
        return False
    dot = tuple(map(int, grey_components[0][0]))
    endpoint_hits = [endpoint for endpoint in (end_a, end_b) if dot in set(_neighbors4(*endpoint))]
    if len(endpoint_hits) != 1:
        return False
    if path[0] != endpoint_hits[0]:
        path.reverse()
    runs = _run_colors([int(inp[r, c]) for r, c in path])
    if runs == runs[::-1]:
        return False
    rebuilt = np.repeat(np.array(runs, dtype=int), bar_height).reshape(-1, 1)
    if not np.array_equal(rebuilt, expected):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    bg = random.choice([color for color in NON_GREY_COLORS if color != GREY])
    palette = [color for color in NON_GREY_COLORS if color != bg]
    run_count = random.randint(*RUN_COUNT_RANGE)
    bar_height = random.randint(*BAR_HEIGHT_RANGE)
    if run_count * bar_height > 30:
        return None

    min_path_len = max(PATH_LEN_RANGE[0], run_count)
    max_path_len = min(PATH_LEN_RANGE[1], run_count * RAW_RUN_LEN_RANGE[1])
    if min_path_len > max_path_len:
        return None
    path_len = random.randint(min_path_len, max_path_len)

    valid_turns = [
        turns
        for turns in range(TURN_RANGE[0], TURN_RANGE[1] + 1)
        if SEGMENT_STEP_RANGE[0] * (turns + 1) <= path_len - 1 <= SEGMENT_STEP_RANGE[1] * (turns + 1)
    ]
    if not valid_turns:
        return None
    turns = random.choice(valid_turns)

    path = _build_path((rows, cols), path_len, turns)
    if path is None:
        return None
    path_cells = set(path)

    raw_run_lengths = _sample_raw_run_lengths(run_count, path_len)
    if not raw_run_lengths:
        return None
    run_colors = _sample_run_colors(run_count, palette)
    if not run_colors:
        return None

    path_colors = _expand_runs(run_colors, raw_run_lengths)
    if len(path_colors) != len(path):
        return None
    if _run_colors(path_colors) != run_colors:
        return None
    if run_colors == run_colors[::-1]:
        return None

    grid = np.full((rows, cols), bg, dtype=int)
    for (r, c), color in zip(path, path_colors):
        grid[r, c] = int(color)

    ordered, end_a, end_b = _ordered_path(np.array(path, dtype=int))
    assert ordered == path or ordered == path[::-1]
    endpoint_options = []
    for endpoint in (end_a, end_b):
        dots = _dot_candidates(endpoint, path_cells, grid.shape)
        if dots:
            endpoint_options.append((endpoint, dots))
    if not endpoint_options:
        return None

    start, dots = random.choice(endpoint_options)
    dot = dots[0]
    grid[dot] = GREY

    line_options = _line_candidates(grid, bg, path_cells, dot, bar_height)
    if not line_options:
        return None
    for r, c in line_options[0]:
        grid[r, c] = GREY

    if path[0] != start:
        run_order = list(reversed(run_colors))
    else:
        run_order = list(run_colors)

    expected = np.empty((bar_height * len(run_order), 1), dtype=int)
    idx = 0
    for color in run_order:
        expected[idx : idx + bar_height, 0] = int(color)
        idx += bar_height

    if np.sum(grid == bg) <= np.sum(grid != bg):
        return None
    if not _semantic_ok(grid, expected, raw_run_lengths, bar_height):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
