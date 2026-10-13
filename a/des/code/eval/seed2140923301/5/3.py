from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
BG_CHOICES = [1, 2, 3, 4, 6, 7, 8, 9]
STRIPE_MAX = 7
WIDTH_MIN = 9
WIDTH_MAX = 15
BOTTOM_HEIGHT_MAX = 14
POLYLINE_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400

RELATIONS = ("short", "equal", "long")
ENDPOINT_MODES = ("different_row", "same_row")
FIRST_AXES = ("h", "v")

_RNG = np.random.default_rng()
_GEN_COUNTER = 0


Coord = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _valid_top(top: np.ndarray, bg: int, divider: int) -> bool:
    colors: list[int] = []
    for row in top:
        values = np.unique(row)
        if len(values) != 1:
            return False
        color = int(values[0])
        if color in (BLACK, bg, divider):
            return False
        colors.append(color)
    return len(colors) == len(set(colors))


def _valid_mask(mask: np.ndarray, bg: int) -> bool:
    for row in mask:
        values = {int(v) for v in np.unique(row)}
        if not values.issubset({BLACK, bg}):
            return False
        if int(np.count_nonzero(row == bg)) > 1:
            return False
    return True


def _find_regions(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    assert bg != BLACK, "background must differ from black mask cells"
    h = g.shape[0]
    candidates: list[tuple[int, int, int]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        full_rows = [r for r in range(h) if np.all(g[r] == color)]
        for i in range(len(full_rows)):
            for j in range(i + 1, len(full_rows)):
                r1 = full_rows[i]
                r2 = full_rows[j]
                top_h = r1
                if top_h < 2 or r2 != 2 * top_h + 1 or r2 >= h - 1:
                    continue
                top = g[:r1]
                mask = g[r1 + 1 : r2]
                bottom = g[r2 + 1 :]
                if top.shape[0] != mask.shape[0] or bottom.size == 0:
                    continue
                if not _valid_top(top, bg, color):
                    continue
                if not _valid_mask(mask, bg):
                    continue
                if not np.all((bottom == bg) | (bottom == color)):
                    continue
                if not np.any(bottom == color):
                    continue
                candidates.append((color, r1, r2))

    assert len(candidates) == 1, "failed to identify the two divider rows"
    divider, r1, r2 = candidates[0]
    return bg, divider, r1, r2, r2 + 1


def _neighbors(cell: Coord, coords: set[Coord]) -> list[Coord]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIR4 if (r + dr, c + dc) in coords]


def _trace_polyline(mask: np.ndarray) -> list[Coord]:
    coords = {tuple(map(int, pos)) for pos in np.argwhere(mask)}
    assert coords, "expected a grey polyline in the bottom region"

    degree = {cell: _neighbors(cell, coords) for cell in coords}
    endpoints = sorted(cell for cell, nbrs in degree.items() if len(nbrs) == 1)
    assert len(endpoints) == 2, "polyline must have exactly two endpoints"
    assert all(len(nbrs) in (1, 2) for nbrs in degree.values()), "polyline must not branch"

    start = endpoints[0]
    path = [start]
    prev: Coord | None = None
    cur = start
    while True:
        next_cells = [nbr for nbr in degree[cur] if nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "polyline traversal must be unambiguous"
        prev, cur = cur, next_cells[0]
        path.append(cur)

    assert len(path) == len(coords), "polyline must be a single simple path"
    return path


def _segment_ranges(path: list[Coord]) -> list[tuple[int, int]]:
    if len(path) < 2:
        return [(0, len(path))]

    directions = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    starts = [0]
    for i in range(1, len(directions)):
        if directions[i] != directions[i - 1]:
            starts.append(i)
    starts.append(len(path) - 1)

    ranges: list[tuple[int, int]] = []
    for idx, start in enumerate(starts[:-1]):
        stop = starts[idx + 1]
        if idx == len(starts) - 2:
            ranges.append((start, len(path)))
        else:
            ranges.append((start, stop))
    return ranges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sample stripe colors through one-cell mask holes, then repaint the lower
    orthogonal grey polyline segment-by-segment from its topmost/leftmost end.
    """
    g = np.asarray(grid, dtype=int)
    bg, divider, top_h, second_divider, bottom_r0 = _find_regions(g)

    top = g[:top_h]
    mask = g[top_h + 1 : second_divider]
    sampled_colors = []
    for r in range(top_h):
        holes = int(np.count_nonzero(mask[r] == bg))
        assert holes <= 1, "each mask row may expose at most one stripe cell"
        if holes == 1:
            sampled_colors.append(int(top[r, 0]))

    path = _trace_polyline(g[bottom_r0:] == divider)
    ranges = _segment_ranges(path)

    out = g.copy()
    for color, (start, stop) in zip(sampled_colors, ranges):
        for rr, cc in path[start:stop]:
            out[bottom_r0 + rr, cc] = color

    return out


def _step_in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _candidate_segment(
    shape: tuple[int, int],
    occupied: set[Coord],
    start: Coord,
    direction: Coord,
    length: int,
) -> list[Coord] | None:
    segment: list[Coord] = []
    temp = set(occupied)
    prev = start
    for step in range(1, length + 1):
        cell = (start[0] + direction[0] * step, start[1] + direction[1] * step)
        if not _step_in_bounds(shape, cell) or cell in temp:
            return None
        for dr, dc in DIR4:
            nbr = (cell[0] + dr, cell[1] + dc)
            if nbr in temp and nbr != prev:
                return None
        temp.add(cell)
        segment.append(cell)
        prev = cell
    return segment


def _has_future(shape: tuple[int, int], occupied: set[Coord], cell: Coord, axis: str) -> bool:
    directions = [(0, -1), (0, 1)] if axis == "h" else [(-1, 0), (1, 0)]
    for direction in directions:
        if _candidate_segment(shape, occupied, cell, direction, 2) is not None:
            return True
    return False


def _build_polyline(
    height: int,
    width: int,
    segment_count: int,
    endpoint_mode: str,
    first_axis: str,
) -> list[Coord] | None:
    shape = (height, width)

    def dfs(path: list[Coord], occupied: set[Coord], axis: str, used_segments: int) -> list[Coord] | None:
        if used_segments == segment_count:
            same_row = path[0][0] == path[-1][0]
            if endpoint_mode == "same_row" and not same_row:
                return None
            if endpoint_mode == "different_row" and same_row:
                return None
            return list(path)

        next_axis = "v" if axis == "h" else "h"
        directions = [(0, -1), (0, 1)] if axis == "h" else [(-1, 0), (1, 0)]
        lengths = [2, 3, 4, 5]
        _RNG.shuffle(directions)
        _RNG.shuffle(lengths)
        candidates: list[list[Coord]] = []
        for direction in directions:
            for length in lengths:
                segment = _candidate_segment(shape, occupied, path[-1], direction, length)
                if segment is None:
                    continue
                new_occupied = occupied | set(segment)
                if used_segments < segment_count - 1 and not _has_future(shape, new_occupied, segment[-1], next_axis):
                    continue
                candidates.append(segment)

        _RNG.shuffle(candidates)
        for segment in candidates:
            new_path = path + segment
            new_occupied = occupied | set(segment)
            result = dfs(new_path, new_occupied, next_axis, used_segments + 1)
            if result is not None:
                return result
        return None

    for _ in range(POLYLINE_ATTEMPTS):
        start = (int(_RNG.integers(1, height - 1)), int(_RNG.integers(1, width - 1)))
        result = dfs([start], {start}, first_axis, 0)
        if result is not None:
            return result
    return None


def _choose_segment_count(relation: str, endpoint_mode: str, first_axis: str) -> int:
    upper = 5 if relation == "long" else 6
    candidates: list[int] = []
    for count in range(2, upper + 1):
        if endpoint_mode == "same_row":
            vertical_segments = count // 2 if first_axis == "h" else (count + 1) // 2
            if vertical_segments < 2:
                continue
        candidates.append(count)
    return int(_RNG.choice(candidates))


def _choose_generation_modes() -> tuple[str, str, str]:
    global _GEN_COUNTER
    idx = _GEN_COUNTER
    _GEN_COUNTER += 1
    relation = RELATIONS[idx % len(RELATIONS)]
    endpoint_mode = ENDPOINT_MODES[(idx // len(RELATIONS)) % len(ENDPOINT_MODES)]
    first_axis = FIRST_AXES[(idx // (len(RELATIONS) * len(ENDPOINT_MODES))) % len(FIRST_AXES)]
    return relation, endpoint_mode, first_axis


def _generate_mask(top_h: int, width: int, bg: int, sample_count: int) -> np.ndarray:
    mask = np.full((top_h, width), BLACK, dtype=int)
    rows = list(map(int, _RNG.choice(top_h, size=sample_count, replace=False)))
    rows.sort()
    cols = list(map(int, _RNG.integers(0, width, size=sample_count)))
    for r, c in zip(rows, cols):
        mask[r, c] = bg
    return mask


def _dominant_bg(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    pairs = sorted(zip(counts.tolist(), values.tolist()), reverse=True)
    return len(pairs) > 0 and int(pairs[0][1]) == bg and (len(pairs) == 1 or pairs[0][0] > pairs[1][0])


def _gen_attempt(relation: str, endpoint_mode: str, first_axis: str) -> np.ndarray | None:
    bg = int(_RNG.choice(BG_CHOICES))
    segment_count = _choose_segment_count(relation, endpoint_mode, first_axis)

    if relation == "short":
        sample_count = int(_RNG.integers(1, segment_count))
    elif relation == "equal":
        sample_count = segment_count
    else:
        sample_count = int(_RNG.integers(segment_count + 1, min(STRIPE_MAX, segment_count + 2) + 1))

    top_h_min = max(3, sample_count)
    top_h_max = min(STRIPE_MAX, sample_count + 2)
    top_h = int(_RNG.integers(top_h_min, top_h_max + 1))

    width = int(_RNG.integers(max(WIDTH_MIN, segment_count + 4), WIDTH_MAX + 1))
    bottom_h_min = max(8, top_h + 3)
    bottom_h = int(_RNG.integers(bottom_h_min, min(BOTTOM_HEIGHT_MAX, bottom_h_min + 4) + 1))

    polyline = _build_polyline(bottom_h, width, segment_count, endpoint_mode, first_axis)
    if polyline is None:
        return None

    stripe_palette = [color for color in BG_CHOICES if color != bg]
    if top_h > len(stripe_palette):
        return None
    stripe_colors = list(map(int, _RNG.choice(stripe_palette, size=top_h, replace=False)))
    mask = _generate_mask(top_h, width, bg, sample_count)

    grid = np.full((2 * top_h + 2 + bottom_h, width), bg, dtype=int)
    for r, color in enumerate(stripe_colors):
        grid[r, :] = color
    grid[top_h, :] = GREY
    grid[top_h + 1 : 2 * top_h + 1, :] = mask
    grid[2 * top_h + 1, :] = GREY
    for r, c in polyline:
        grid[2 * top_h + 2 + r, c] = GREY

    if not _dominant_bg(grid, bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    relation, endpoint_mode, first_axis = _choose_generation_modes()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(relation, endpoint_mode, first_axis)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
