from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds, neighbors4


Coord = tuple[int, int]

GRAY = int(Color.GRAY)
DIRS: tuple[Coord, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))  # left, down, right, up

PATH_LEN_MIN = 9
PATH_LEN_MAX = 20
REGION_MIN = 9
REGION_MAX = 12
GRID_MIN = 18
GRID_MAX = 30
SPIRAL_PATH_PADDING = 2
PATH_BUILD_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 500
SEQUENCE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)


def _ordered_path(path_cells: set[Coord], start: Coord) -> list[Coord]:
    degrees = {
        cell: sum((neighbor in path_cells) for neighbor in neighbors4(*cell))
        for cell in path_cells
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"
    assert start in endpoints, "start must be the endpoint selected by the gray pixel"

    path = []
    prev = None
    cur = start
    while True:
        path.append(cur)
        next_cells = [
            neighbor
            for neighbor in neighbors4(*cur)
            if neighbor in path_cells and neighbor != prev
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(path) == len(path_cells), "path traversal must visit every path cell"
    return path


def _spiral_offsets(length: int) -> list[Coord]:
    offsets: list[Coord] = []
    r = 0
    c = 0
    seg_len = 2
    dir_idx = 0
    while len(offsets) < length:
        dr, dc = DIRS[dir_idx % len(DIRS)]
        for _ in range(seg_len):
            if len(offsets) == length:
                break
            r += dr
            c += dc
            offsets.append((r, c))
        dir_idx += 1
        seg_len += 1
    return offsets


def _spiral_cells(marker: Coord, length: int, shape: tuple[int, int]) -> list[Coord]:
    cells = []
    mr, mc = marker
    for dr, dc in _spiral_offsets(length):
        r = mr + dr
        c = mc + dc
        assert in_bounds(shape, r, c), "spiral must stay inside the canvas"
        cells.append((r, c))
    assert len(cells) == len(set(cells)), "spiral cells must not repeat"
    return cells


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    gray_coords = np.argwhere(g == GRAY)
    assert gray_coords.shape[0] == 1, "expected exactly one gray selector pixel"
    gray = tuple(map(int, gray_coords[0]))

    non_gray_foreground = (g != bg) & (g != GRAY)
    components = components_mask(non_gray_foreground, connectivity=4)
    assert len(components) == 2, "expected a path component and one isolated marker"

    marker_components = [coords for coords in components if coords.shape[0] == 1]
    path_components = [coords for coords in components if coords.shape[0] > 1]
    assert len(marker_components) == 1 and len(path_components) == 1

    marker = tuple(map(int, marker_components[0][0]))
    path_coords = path_components[0]
    path_cells = {tuple(map(int, coord)) for coord in path_coords}

    gray_neighbors = [neighbor for neighbor in neighbors4(*gray) if neighbor in path_cells]
    assert len(gray_neighbors) == 1, "gray selector must touch exactly one path cell"
    ordered = _ordered_path(path_cells, gray_neighbors[0])
    path_colors = [int(g[r, c]) for r, c in ordered]

    marker_color = int(g[marker])
    assert marker_color not in path_colors, "marker color must not appear on the path"
    assert marker not in neighbors4(*gray), "marker must stay separate from the selector"
    assert GRAY not in path_colors, "gray selector color must stay off the path"

    turns = 0
    if len(ordered) >= 3:
        steps = [
            (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
            for i in range(len(ordered) - 1)
        ]
        turns = sum(step_a != step_b for step_a, step_b in zip(steps, steps[1:]))

    return {
        "grid": g,
        "bg": bg,
        "gray": gray,
        "marker": marker,
        "marker_color": marker_color,
        "path": ordered,
        "path_colors": path_colors,
        "turns": int(turns),
    }


def _render_output(shape: tuple[int, int], bg: int, marker: Coord, marker_color: int, sequence: list[int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    out[marker] = int(marker_color)
    for (r, c), color in zip(_spiral_cells(marker, len(sequence), shape), sequence):
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color strip from the gray-selected path end and rewrite it onto an expanding square spiral around the isolated marker.

    Input structure:
    - The background is dominant.
    - One multicolor 4-connected one-pixel-wide path has exactly two endpoints and no branches.
    - A single gray pixel is 4-adjacent to exactly one path endpoint, and one isolated non-gray marker pixel sits elsewhere.

    Rule:
    - Starting from the endpoint touched by the gray pixel, traverse the path and record its color sequence.
    - Erase the original path and the gray selector, but keep the marker pixel.
    - From the marker, walk a spiral with directions left, down, right, up and segment lengths 2, 3, 4, ...; after each move, write the next recorded path color, stopping when the sequence is exhausted.

    Key insight:
    - The bent path is only a storage ribbon for an ordered color sequence; the gray selector chooses which endpoint to read first, while the marker alone fixes the output geometry.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    return _render_output(
        parsed["grid"].shape,
        parsed["bg"],
        parsed["marker"],
        parsed["marker_color"],
        parsed["path_colors"],
    )


def _sample_run_lengths(n_segments: int, total_steps: int, max_run: int = 5) -> list[int]:
    runs = []
    remaining = total_steps
    for i in range(n_segments):
        slots = n_segments - i - 1
        lo = max(1, remaining - max_run * slots)
        hi = min(max_run, remaining - slots)
        if lo > hi:
            raise ValueError("no valid run length split")
        runs.append(random.randint(lo, hi))
        remaining -= runs[-1]
    random.shuffle(runs)
    return runs


def _sample_run_plan(length: int) -> list[int]:
    total_steps = length - 1
    min_segments = max(3, (total_steps + 4) // 5)
    max_segments = min(6, total_steps)
    n_segments = random.randint(min_segments, max_segments)
    return _sample_run_lengths(n_segments, total_steps, max_run=5)


def _turn_dirs(direction: Coord | None) -> list[Coord]:
    if direction is None:
        dirs = list(DIRS)
        random.shuffle(dirs)
        return dirs
    dr, dc = direction
    dirs = [(-dc, dr), (dc, -dr)]
    random.shuffle(dirs)
    return dirs


def _extend_segment(
    path: list[Coord],
    used: set[Coord],
    direction: Coord,
    steps: int,
    height: int,
    width: int,
) -> tuple[list[Coord], set[Coord]] | None:
    dr, dc = direction
    new_path = list(path)
    new_used = set(used)
    r, c = new_path[-1]
    for _ in range(steps):
        nr, nc = r + dr, c + dc
        if not (1 <= nr < height - 1 and 1 <= nc < width - 1):
            return None
        if (nr, nc) in new_used:
            return None
        if any((ar, ac) in new_used and (ar, ac) != (r, c) for ar, ac in neighbors4(nr, nc)):
            return None
        new_path.append((nr, nc))
        new_used.add((nr, nc))
        r, c = nr, nc
    return new_path, new_used


def _build_local_path(height: int, width: int, length: int) -> list[Coord] | None:
    starts = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    for _ in range(PATH_BUILD_ATTEMPTS):
        run_lengths = _sample_run_plan(length)
        random.shuffle(starts)

        def search(
            seg_i: int,
            prev_dir: Coord | None,
            path: list[Coord],
            used: set[Coord],
        ) -> list[Coord] | None:
            if seg_i == len(run_lengths):
                return path
            for direction in _turn_dirs(prev_dir):
                extended = _extend_segment(path, used, direction, run_lengths[seg_i], height, width)
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


def _gray_candidates(endpoint: Coord, path_cells: set[Coord], shape: tuple[int, int]) -> list[Coord]:
    candidates = []
    for nr, nc in neighbors4(*endpoint):
        if not in_bounds(shape, nr, nc) or (nr, nc) in path_cells:
            continue
        touched = [neighbor for neighbor in neighbors4(nr, nc) if neighbor in path_cells]
        if touched == [endpoint]:
            candidates.append((nr, nc))
    return candidates


def _sample_region_shape(length: int) -> tuple[int, int]:
    while True:
        height = random.randint(REGION_MIN, REGION_MAX)
        width = random.randint(REGION_MIN, REGION_MAX)
        if (height - 2) * (width - 2) >= 2 * length:
            return height, width


def _rect_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _pad_rect(rect: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = rect
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _marker_positions(shape: tuple[int, int], spiral_offsets: list[Coord]) -> list[Coord]:
    h, w = shape
    bbox = bbox_coords(np.array(spiral_offsets, dtype=int))
    r0, c0, r1, c1 = bbox
    row_lo = max(1, -r0)
    row_hi = min(h - 2, h - 1 - r1)
    col_lo = max(1, -c0)
    col_hi = min(w - 2, w - 1 - c1)
    if row_lo > row_hi or col_lo > col_hi:
        return []
    return [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]


def _region_positions(shape: tuple[int, int], region_shape: tuple[int, int], forbidden: tuple[int, int, int, int]) -> list[Coord]:
    h, w = shape
    rh, rw = region_shape
    positions = []
    for r0 in range(h - rh + 1):
        for c0 in range(w - rw + 1):
            rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if _rect_disjoint(rect, forbidden):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _color_sequence(length: int, marker_color: int) -> list[int]:
    pool = [color for color in SEQUENCE_COLOR_POOL if color != marker_color]
    palette = random.sample(pool, random.randint(3, min(5, len(pool))))
    sequence = [random.choice(palette) for _ in range(length)]
    for color in palette:
        sequence[random.randrange(length)] = color
    if len(set(sequence)) < 2:
        replacement = next(color for color in palette if color != sequence[0])
        sequence[-1] = replacement
    if sequence == sequence[::-1]:
        replacement = next(color for color in palette if color != sequence[0])
        sequence[-1] = replacement
    assert sequence != sequence[::-1], "gray selector must matter"
    return sequence


def _make_instance() -> tuple[np.ndarray, np.ndarray] | None:
    length = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    region_shape = _sample_region_shape(length)
    local_path = _build_local_path(*region_shape, length)
    if local_path is None:
        return None

    path_set = set(local_path)
    start_gray_options = _gray_candidates(local_path[0], path_set, region_shape)
    end_gray_options = _gray_candidates(local_path[-1], path_set, region_shape)
    if not start_gray_options or not end_gray_options:
        return None

    spiral_offsets = _spiral_offsets(length)
    spiral_bbox = bbox_coords(np.array(spiral_offsets, dtype=int))
    spiral_h = spiral_bbox[2] - spiral_bbox[0] + 1
    spiral_w = spiral_bbox[3] - spiral_bbox[1] + 1

    rh, rw = region_shape
    min_h = max(GRID_MIN, rh + 2, spiral_h + 4)
    min_w = max(GRID_MIN, rw + 2, spiral_w + 4)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)
    marker_choices = _marker_positions((h, w), spiral_offsets)
    if not marker_choices:
        return None
    marker = random.choice(marker_choices)

    mr, mc = marker
    sr0, sc0, sr1, sc1 = spiral_bbox
    spiral_rect = (mr + sr0, mc + sc0, mr + sr1, mc + sc1)
    forbidden = _pad_rect(spiral_rect, SPIRAL_PATH_PADDING, (h, w))
    region_choices = _region_positions((h, w), region_shape, forbidden)
    if not region_choices:
        return None
    region_top_left = random.choice(region_choices)

    marker_color = random.choice(SEQUENCE_COLOR_POOL)
    sequence = _color_sequence(length, marker_color)

    use_start = bool(random.getrandbits(1))
    if use_start:
        gray_local = random.choice(start_gray_options)
        ordered_local = list(local_path)
    else:
        gray_local = random.choice(end_gray_options)
        ordered_local = list(reversed(local_path))

    rr, rc = region_top_left
    ordered_global = [(rr + r, rc + c) for r, c in ordered_local]
    gray = (rr + gray_local[0], rc + gray_local[1])

    inp = np.zeros((h, w), dtype=int)
    inp[marker] = marker_color
    for (r, c), color in zip(ordered_global, sequence):
        inp[r, c] = color
    inp[gray] = GRAY

    out = _render_output(inp.shape, 0, marker, marker_color, sequence)
    return inp, out


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        parsed = _parse_input(inp)
        sequence = parsed["path_colors"]
        assert len(sequence) >= PATH_LEN_MIN
        assert parsed["turns"] >= 2, "generator must use a genuinely bent path"
        assert sequence != sequence[::-1], "gray-selected direction must affect the result"

        spiral_cells = _spiral_cells(parsed["marker"], len(sequence), inp.shape)
        assert len(spiral_cells) == len(sequence)
        assert parsed["gray"] not in spiral_cells, "input elements should stay visually separated"
        assert parsed["marker"] not in parsed["path"], "marker must stay separate from the path"

        reverse_out = _render_output(
            inp.shape,
            parsed["bg"],
            parsed["marker"],
            parsed["marker_color"],
            list(reversed(sequence)),
        )
        assert not np.array_equal(expected, reverse_out), "gray choice must be indispensable"

        solved = main(inp)
        assert np.array_equal(solved, expected)
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        made = _make_instance()
        if made is None:
            continue
        inp, out = made
        if _semantic_check(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid spiral-ribbon input")
