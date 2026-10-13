import random

import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_BG = 0
_PATH_COLOR = 5

# Generator tuning.
_SEGMENT_COUNT_RANGE = (3, 5)
_LENGTH_POOL = (2, 3, 4, 5, 6)
_MAX_TOTAL_BAND_HEIGHT = 20
_LEGEND_SIDES = ("left", "right", "top", "bottom")
_LEGEND_GAP_OPTIONS = (0, 1)
_OUTER_MARGIN_RANGE = (1, 3)
_PATH_LEGEND_SEP_RANGE = (2, 4)
_GEN_RETRIES = 400
_PATH_BUILD_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _color_coords(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]


def _path_info(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]] | None:
    if len(coords) < 2:
        return None
    pts = set(coords)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg = sum((r + dr, c + dc) in pts for dr, dc in _DIRS4)
        if deg not in (1, 2):
            return None
        degrees[(r, c)] = deg
    endpoints = sorted(pt for pt, deg in degrees.items() if deg == 1)
    if len(endpoints) != 2:
        return None

    seen = set()
    stack = [endpoints[0]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        r, c = cur
        for dr, dc in _DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                stack.append(nxt)
    if len(seen) != len(pts):
        return None
    return endpoints, degrees


def _trace_path(coords: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    pts = set(coords)
    order = [start]
    prev = None
    cur = start
    while True:
        r, c = cur
        nxts = []
        for dr, dc in _DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt != prev:
                nxts.append(nxt)
        if not nxts:
            break
        if len(nxts) != 1:
            raise AssertionError("path trace must have a unique continuation")
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    if len(order) != len(pts):
        raise AssertionError("path trace did not visit every path cell")
    return order


def _segment_lengths(path_order: list[tuple[int, int]]) -> list[int]:
    if len(path_order) < 2:
        raise AssertionError("path must contain at least two cells")
    directions = []
    for (r0, c0), (r1, c1) in zip(path_order, path_order[1:]):
        directions.append((r1 - r0, c1 - c0))

    lengths = []
    run_steps = 1
    for prev_dir, cur_dir in zip(directions, directions[1:]):
        if cur_dir == prev_dir:
            run_steps += 1
        else:
            lengths.append(run_steps + 1)
            run_steps = 1
    lengths.append(run_steps + 1)
    return lengths


def _render_bands(lengths: list[int], legend_colors: list[int]) -> np.ndarray:
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != len(legend_colors):
        raise AssertionError("legend size must match number of distinct segment lengths")
    mapping = {length: color for length, color in zip(unique_lengths, legend_colors)}
    out_h = sum(lengths)
    out_w = max(lengths)
    out = np.zeros((out_h, out_w), dtype=int)
    row = 0
    for length in lengths:
        out[row : row + length, :] = int(mapping[length])
        row += length
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose a single-pixel path into straight runs and encode them as stacked color bands.

    Input structure:
    - The background is black.
    - One non-background color forms a 4-connected, one-pixel-wide path with exactly two endpoints and no branches.
    - Away from the path, a vertical legend of singleton colored pixels lists one color per distinct segment length.

    Rule:
    - Identify the path and start tracing it from the top-left endpoint.
    - Split the traced path whenever its direction changes, and record each straight segment's pixel length.
    - Sort the distinct segment lengths from shortest to longest and match them to the legend colors from top to bottom.
    - Output a rectangle whose width is the longest segment length; for each traced segment, draw one full-width horizontal band whose height equals that segment's length and whose color comes from the length-to-legend mapping.

    Key insight:
    - The path's geometry carries both the ordering and the measurement information: turns define segment boundaries, while segment lengths determine both the output band heights and the legend-based colors.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    bg = _most_common_color(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]

    path_candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    for color in colors:
        coords = _color_coords(g, color)
        info = _path_info(coords)
        if info is not None:
            path_candidates.append((len(coords), color, coords))
    if not path_candidates:
        raise AssertionError("no valid path color found")
    path_candidates.sort(reverse=True)
    _size, path_color, path_coords = path_candidates[0]

    path_endpoints, _degrees = _path_info(path_coords)  # type: ignore[assignment]
    if path_endpoints is None:
        raise AssertionError("path color is not a simple path")
    start = min(path_endpoints)
    ordered_path = _trace_path(path_coords, start)
    lengths = _segment_lengths(ordered_path)

    legend_entries = []
    for color in colors:
        if color == path_color:
            continue
        coords = _color_coords(g, color)
        if len(coords) != 1:
            raise AssertionError("legend colors must appear exactly once")
        r, c = coords[0]
        legend_entries.append((r, c, color))
    legend_entries.sort(key=lambda item: (item[0], item[1]))
    legend_colors = [int(color) for _r, _c, color in legend_entries]

    return _render_bands(lengths, legend_colors)


def _perpendicular_choices(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _extend_path(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]] | None:
    def rollback(added_cells: list[tuple[int, int]]) -> None:
        for _ in added_cells:
            path.pop()
        for cell in reversed(added_cells):
            used.remove(cell)

    current = path[-1]
    added: list[tuple[int, int]] = []
    for _ in range(length - 1):
        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        nxt = (nr, nc)
        if nxt in used:
            rollback(added)
            return None
        for adr, adc in _DIRS4:
            neighbor = (nr + adr, nc + adc)
            if neighbor in used and neighbor != current:
                rollback(added)
                return None
        used.add(nxt)
        path.append(nxt)
        added.append(nxt)
        current = nxt
    return added


def _build_path_from_lengths(lengths: list[int], rng: random.Random) -> list[tuple[int, int]] | None:
    dirs0 = _DIRS4[:]
    rng.shuffle(dirs0)

    def rec(
        seg_idx: int,
        path: list[tuple[int, int]],
        used: set[tuple[int, int]],
        prev_dir: tuple[int, int] | None,
    ) -> list[tuple[int, int]] | None:
        if seg_idx == len(lengths):
            return path.copy()

        if prev_dir is None:
            choices = dirs0[:]
        else:
            choices = _perpendicular_choices(prev_dir)
            rng.shuffle(choices)

        for direction in choices:
            added = _extend_path(path, used, direction, lengths[seg_idx])
            if added is None:
                continue
            result = rec(seg_idx + 1, path, used, direction)
            if result is not None:
                return result
            for _ in added:
                cell = path.pop()
                used.remove(cell)
        return None

    return rec(0, [(0, 0)], {(0, 0)}, None)


def _sample_segment_lengths(rng: random.Random) -> list[int]:
    for _ in range(200):
        segment_count = rng.randint(*_SEGMENT_COUNT_RANGE)
        max_distinct = min(4, segment_count, len(_LENGTH_POOL))
        min_distinct = 2
        if segment_count >= 4 and rng.random() < 0.5:
            distinct_count = rng.randint(min_distinct, max_distinct - 1 if max_distinct > min_distinct else min_distinct)
        else:
            distinct_count = rng.randint(min_distinct, max_distinct)
        distinct_lengths = rng.sample(list(_LENGTH_POOL), distinct_count)
        lengths = distinct_lengths[:]
        while len(lengths) < segment_count:
            lengths.append(rng.choice(distinct_lengths))
        rng.shuffle(lengths)
        if sum(lengths) <= _MAX_TOTAL_BAND_HEIGHT:
            return lengths
    raise RuntimeError("failed to sample valid segment lengths")


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return [(r - min_r, c - min_c) for r, c in coords]


def _place_scene(
    path_coords: list[tuple[int, int]],
    legend_colors: list[int],
    side: str,
    legend_gap: int,
    rng: random.Random,
) -> np.ndarray | None:
    path_coords = _normalize(path_coords)
    path_h = max(r for r, _c in path_coords) + 1
    path_w = max(c for _r, c in path_coords) + 1
    legend_h = len(legend_colors) + legend_gap * (len(legend_colors) - 1)

    top = rng.randint(*_OUTER_MARGIN_RANGE)
    bottom = rng.randint(*_OUTER_MARGIN_RANGE)
    left = rng.randint(*_OUTER_MARGIN_RANGE)
    right = rng.randint(*_OUTER_MARGIN_RANGE)
    sep = rng.randint(*_PATH_LEGEND_SEP_RANGE)

    if side in ("left", "right"):
        inner_h = max(path_h, legend_h)
        inner_w = path_w + sep + 1
        total_h = top + inner_h + bottom
        total_w = left + inner_w + right
        if total_h > 30 or total_w > 30:
            return None
        path_r0 = top + rng.randint(0, inner_h - path_h)
        legend_r0 = top + rng.randint(0, inner_h - legend_h)
        if side == "left":
            legend_c = left
            path_c0 = left + 1 + sep
        else:
            path_c0 = left
            legend_c = left + path_w + sep
    else:
        inner_h = path_h + sep + legend_h
        inner_w = max(path_w, 1)
        total_h = top + inner_h + bottom
        total_w = left + inner_w + right
        if total_h > 30 or total_w > 30:
            return None
        path_c0 = left + rng.randint(0, inner_w - path_w)
        legend_c = left + rng.randint(0, inner_w - 1)
        if side == "top":
            legend_r0 = top
            path_r0 = top + legend_h + sep
        else:
            path_r0 = top
            legend_r0 = top + path_h + sep

    grid = np.full((total_h, total_w), _BG, dtype=int)
    for r, c in path_coords:
        grid[path_r0 + r, path_c0 + c] = _PATH_COLOR
    for idx, color in enumerate(legend_colors):
        rr = legend_r0 + idx * (legend_gap + 1)
        grid[rr, legend_c] = int(color)
    return grid


def _expected_from_construction(
    path_coords: list[tuple[int, int]],
    construction_lengths: list[int],
    legend_colors: list[int],
) -> np.ndarray:
    start, end = path_coords[0], path_coords[-1]
    trace_lengths = construction_lengths if start <= end else list(reversed(construction_lengths))
    return _render_bands(trace_lengths, legend_colors)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    lengths = _sample_segment_lengths(rng)

    path_coords = None
    for _ in range(_PATH_BUILD_RETRIES):
        candidate = _build_path_from_lengths(lengths, rng)
        if candidate is not None:
            path_coords = candidate
            break
    if path_coords is None:
        return None

    distinct_lengths = sorted(set(lengths))
    color_pool = [c for c in range(1, 10) if c != _PATH_COLOR]
    rng.shuffle(color_pool)
    legend_colors = color_pool[: len(distinct_lengths)]
    side = rng.choice(_LEGEND_SIDES)
    legend_gap = rng.choice(_LEGEND_GAP_OPTIONS)

    grid = _place_scene(path_coords, legend_colors, side, legend_gap, rng)
    if grid is None:
        return None

    expected = _expected_from_construction(path_coords, lengths, legend_colors)
    if expected.ndim != 2 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
