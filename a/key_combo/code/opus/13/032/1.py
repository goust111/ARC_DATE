from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
PANEL_HEIGHT = 3
PANEL_COLORS = (1, 2, 3)
PANEL_TOP_ROWS = (0, 4, 8)
SEPARATOR_ROW = 11

GRID_HEIGHT_RANGE = (16, 22)
GRID_WIDTH_RANGE = (16, 24)
PANEL_WIDTH_RANGE = (4, 8)
UNION_WIDTH_RANGE = (8, 14)

PATH_LEN_RANGE = (9, 20)
PATH_SEGMENT_COUNT_RANGE = (3, 6)
PATH_SEGMENT_STEP_RANGE = (1, 5)
MAX_GENERATION_ATTEMPTS = 400

_SEEN_INPUT_SIGS: set[bytes] = set()
_SEEN_OUTPUT_SIGS: set[bytes] = set()

Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _ordered_path(coords: np.ndarray) -> list[Coord]:
    cells = {tuple(map(int, cell)) for cell in coords}
    degrees = {
        cell: sum((nbr in cells) for nbr in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, deg in degrees.items() if deg == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees.values()), "path must not branch"

    path: list[Coord] = []
    prev: Coord | None = None
    cur = endpoints[0]
    while True:
        path.append(cur)
        nxt = [nbr for nbr in _neighbors4(*cur) if nbr in cells and nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unique"
        prev, cur = cur, nxt[0]
    assert len(path) == len(cells), "path traversal must visit all path cells"
    return path


def _first_occurrence_order(colors: list[int]) -> tuple[int, ...]:
    order: list[int] = []
    seen: set[int] = set()
    for color in colors:
        color = int(color)
        if color not in seen:
            seen.add(color)
            order.append(color)
    return tuple(order)


def _overlay_from_intervals(
    intervals: dict[int, tuple[int, int]], order: tuple[int, int, int]
) -> np.ndarray:
    min_c = min(start for start, _end in intervals.values())
    max_c = max(end for _start, end in intervals.values())
    out = np.zeros((PANEL_HEIGHT, max_c - min_c + 1), dtype=int)
    for color in order:
        start, end = intervals[int(color)]
        out[:, start - min_c : end - min_c + 1] = int(color)
    return out


def _signature(grid: np.ndarray) -> bytes:
    arr = np.asarray(grid, dtype=np.int16)
    return arr.shape[0].to_bytes(2, "little") + arr.shape[1].to_bytes(2, "little") + arr.tobytes()


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    separator_rows = [
        r
        for r in range(h)
        if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != bg
    ]
    assert len(separator_rows) == 1, "expected exactly one full-width separator row"
    sep_r = int(separator_rows[0])
    sep_color = int(g[sep_r, 0])

    upper = g[:sep_r]
    upper_colors = sorted(set(map(int, np.unique(upper))) - {bg, sep_color})
    assert len(upper_colors) == 3, "expected exactly three panel colors above the separator"

    intervals: dict[int, tuple[int, int]] = {}
    row_by_color: dict[int, int] = {}
    for color in upper_colors:
        coords = np.argwhere(upper == color)
        assert coords.size > 0, "panel color must appear in upper area"
        b = _bbox(coords)
        r0, c0, r1, c1 = b
        patch = upper[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape[0] == PANEL_HEIGHT, "each panel must be three rows tall"
        assert np.all(patch == color), "each panel must be a solid rectangle"
        assert len(_components(upper == color)) == 1, "each panel color must form one rectangle"
        intervals[int(color)] = (int(c0), int(c1))
        row_by_color[int(color)] = int(r0)

    spatial_order = tuple(color for color, _row in sorted(row_by_color.items(), key=lambda item: item[1]))

    lower_mask = np.zeros_like(g, dtype=bool)
    lower_mask[sep_r + 1 :, :] = (g[sep_r + 1 :, :] != bg) & (g[sep_r + 1 :, :] != sep_color)
    lower_components = _components(lower_mask)
    assert len(lower_components) == 1, "expected exactly one path below the separator"
    path = _ordered_path(lower_components[0])
    path_colors = [int(g[r, c]) for r, c in path]
    assert set(path_colors) == set(upper_colors), "path must use exactly the panel colors"
    traversal_order = _first_occurrence_order(path_colors)
    reverse_order = _first_occurrence_order(list(reversed(path_colors)))
    assert len(traversal_order) == 3, "all three colors must appear on the path"

    return {
        "bg": bg,
        "separator_row": sep_r,
        "separator_color": sep_color,
        "intervals": intervals,
        "spatial_order": spatial_order,
        "path": path,
        "path_colors": path_colors,
        "traversal_order": traversal_order,
        "reverse_order": reverse_order,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the lower multicolor path to derive the overlay priority of the upper panels, then output the cropped composite strip.

    Input structure:
    - The grid has black background plus one full-width uniform separator row.
    - Above the separator are exactly three solid 3-row rectangular panels in distinct colors, arranged top-to-bottom with empty gaps.
    - Below the separator is a single 4-connected one-pixel-wide non-branching path using exactly those same three colors.

    Rule:
    - Start from the endpoint with smaller row index, breaking ties by smaller column index, and traverse the path to the other endpoint.
    - Record the first time each path color appears; this 3-color sequence gives bottom-to-top panel priority.
    - Project all three panels into one shared 3-row canvas spanning the union of their columns, draw them in that priority order, and keep only the composite canvas.

    Key insight:
    - The lower path encodes layer depth rather than geometry: first appearance order along the traversal decides which panel color wins in overlapping columns.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _overlay_from_intervals(scene["intervals"], scene["traversal_order"])


def _sample_interval_set() -> dict[int, tuple[int, int]] | None:
    union_width = random.randint(*UNION_WIDTH_RANGE)
    widths = {
        color: random.randint(PANEL_WIDTH_RANGE[0], min(PANEL_WIDTH_RANGE[1], union_width))
        for color in PANEL_COLORS
    }
    starts = {
        color: random.randint(0, union_width - widths[color])
        for color in PANEL_COLORS
    }
    intervals = {
        color: (starts[color], starts[color] + widths[color] - 1)
        for color in PANEL_COLORS
    }

    outputs = {
        order: _overlay_from_intervals(intervals, order)
        for order in itertools.permutations(PANEL_COLORS)
    }
    unique = {_signature(out) for out in outputs.values()}
    if len(unique) < 5:
        return None
    return intervals


def _build_u_path(h: int, w: int) -> list[Coord] | None:
    row_min = SEPARATOR_ROW + 2
    row_max = h - 2
    if row_max - row_min + 1 < 3 or w < 8:
        return None
    top = random.randint(row_min, row_max - 2)
    bottom = random.randint(top + 2, row_max)
    left = random.randint(1, w - 7)
    right = random.randint(left + 4, w - 2)

    path: list[Coord] = []
    for r in range(top, bottom + 1):
        path.append((r, left))
    for c in range(left + 1, right):
        path.append((bottom, c))
    for r in range(bottom, top - 1, -1):
        path.append((r, right))
    return path


def _turn_dirs(direction: Coord | None) -> list[Coord]:
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    if direction is None:
        random.shuffle(dirs)
        return dirs
    dr, dc = direction
    turns = [(-dc, dr), (dc, -dr)]
    random.shuffle(turns)
    return turns


def _sample_run_plan() -> list[int]:
    target_len = random.randint(*PATH_LEN_RANGE)
    total_steps = target_len - 1
    segment_count = random.randint(*PATH_SEGMENT_COUNT_RANGE)
    while not (
        PATH_SEGMENT_STEP_RANGE[0] * segment_count
        <= total_steps
        <= PATH_SEGMENT_STEP_RANGE[1] * segment_count
    ):
        segment_count = random.randint(*PATH_SEGMENT_COUNT_RANGE)

    runs: list[int] = []
    remaining = total_steps
    for i in range(segment_count):
        slots_left = segment_count - i - 1
        lo = max(PATH_SEGMENT_STEP_RANGE[0], remaining - PATH_SEGMENT_STEP_RANGE[1] * slots_left)
        hi = min(PATH_SEGMENT_STEP_RANGE[1], remaining - PATH_SEGMENT_STEP_RANGE[0] * slots_left)
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    row_bounds: tuple[int, int],
    col_bounds: tuple[int, int],
) -> tuple[list[Coord], set[Coord]] | None:
    r_min, r_max = row_bounds
    c_min, c_max = col_bounds
    dr, dc = direction
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]

    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (r_min <= nr <= r_max and c_min <= nc <= c_max):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in _neighbors4(nr, nc)):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc

    return new_path, new_used


def _build_walk_path(h: int, w: int) -> list[Coord] | None:
    row_bounds = (SEPARATOR_ROW + 2, h - 2)
    col_bounds = (1, w - 2)
    if row_bounds[0] > row_bounds[1] or col_bounds[0] > col_bounds[1]:
        return None
    starts = [
        (r, c)
        for r in range(row_bounds[0], row_bounds[1] + 1)
        for c in range(col_bounds[0], col_bounds[1] + 1)
    ]
    random.shuffle(starts)
    run_plan = _sample_run_plan()

    def search(
        seg_idx: int,
        prev_dir: Coord | None,
        path: list[Coord],
        used: set[Coord],
    ) -> list[Coord] | None:
        if seg_idx == len(run_plan):
            return path
        for direction in _turn_dirs(prev_dir):
            extended = _extend_segment(path, used, direction, run_plan[seg_idx], row_bounds, col_bounds)
            if extended is None:
                continue
            found = search(seg_idx + 1, direction, extended[0], extended[1])
            if found is not None:
                return found
        return None

    for start in starts:
        found = search(0, None, [start], {start})
        if found is not None:
            return found
    return None


def _build_path(h: int, w: int) -> list[Coord] | None:
    if random.random() < 0.5:
        path = _build_u_path(h, w)
        if path is not None:
            return path
    return _build_walk_path(h, w)


def _build_color_sequence(
    forward_order: tuple[int, int, int],
    reverse_order: tuple[int, int, int],
    length: int,
) -> list[int]:
    assert length >= 6
    intro_lengths = [random.randint(1, 3) for _ in range(3)]
    intro_total = sum(intro_lengths)
    tail = [reverse_order[2], reverse_order[1], reverse_order[0]]
    middle_len = length - intro_total - len(tail)
    if middle_len < 0:
        intro_lengths = [1, 1, 1]
        intro_total = 3
        middle_len = length - intro_total - len(tail)
    assert middle_len >= 0

    seq: list[int] = []
    for color, block_len in zip(forward_order, intro_lengths):
        seq.extend([int(color)] * block_len)
    for _ in range(middle_len):
        seq.append(int(random.choice(forward_order)))
    seq.extend(int(color) for color in tail)
    assert len(seq) == length
    assert _first_occurrence_order(seq) == forward_order
    assert _first_occurrence_order(list(reversed(seq))) == reverse_order
    return seq


def _paint_scene(
    height: int,
    width: int,
    spatial_order: tuple[int, int, int],
    intervals: dict[int, tuple[int, int]],
    path: list[Coord],
    path_colors: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[SEPARATOR_ROW, :] = SEPARATOR_COLOR

    min_c = min(start for start, _end in intervals.values())
    max_c = max(end for _start, end in intervals.values())
    base_c = random.randint(0, width - (max_c - min_c + 1))
    shift = base_c - min_c

    for row0, color in zip(PANEL_TOP_ROWS, spatial_order):
        start, end = intervals[int(color)]
        grid[row0 : row0 + PANEL_HEIGHT, start + shift : end + shift + 1] = int(color)

    for (r, c), color in zip(path, path_colors):
        grid[r, c] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
    except AssertionError:
        return False

    forward_order = tuple(scene["traversal_order"])
    reverse_order = tuple(scene["reverse_order"])
    spatial_order = tuple(scene["spatial_order"])
    intervals = scene["intervals"]

    if forward_order == spatial_order:
        return False
    if reverse_order == forward_order:
        return False

    outputs = {
        order: _overlay_from_intervals(intervals, order)
        for order in itertools.permutations(PANEL_COLORS)
    }
    if len({_signature(out) for out in outputs.values()}) < 5:
        return False

    true_output = outputs[forward_order]
    if outputs[spatial_order].shape == true_output.shape and np.array_equal(outputs[spatial_order], true_output):
        return False
    if outputs[reverse_order].shape == true_output.shape and np.array_equal(outputs[reverse_order], true_output):
        return False
    if set(map(int, np.unique(true_output))) != set(PANEL_COLORS):
        return False
    if not np.array_equal(main(grid), true_output):
        return False

    in_sig = _signature(grid)
    out_sig = _signature(true_output)
    if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    if height <= SEPARATOR_ROW + 3:
        return None

    intervals = _sample_interval_set()
    if intervals is None:
        return None

    spatial_order = tuple(random.sample(PANEL_COLORS, k=3))
    forward_order_choices = [order for order in itertools.permutations(PANEL_COLORS) if order != spatial_order]
    forward_order = random.choice(forward_order_choices)
    reverse_order_choices = [order for order in itertools.permutations(PANEL_COLORS) if order != forward_order]
    reverse_order = random.choice(reverse_order_choices)

    path = _build_path(height, width)
    if path is None:
        return None
    ordered_path = _ordered_path(np.array(path, dtype=int))
    path_colors = _build_color_sequence(forward_order, reverse_order, len(ordered_path))

    grid = _paint_scene(height, width, spatial_order, intervals, ordered_path, path_colors)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            out_sig = _signature(main(grid))
            _SEEN_INPUT_SIGS.add(_signature(grid))
            _SEEN_OUTPUT_SIGS.add(out_sig)
            return grid
    raise RuntimeError("failed to generate a valid input")
