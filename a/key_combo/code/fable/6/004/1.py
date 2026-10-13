from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    """Adapted from arc_utils: background is the most common color."""
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen: set[Coord] = set()
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _classify_roles(g: np.ndarray) -> tuple[int, int, int, int]:
    bg = get_bg(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert len(colors) == 3, "expected exactly three foreground colors"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    marker_candidates = [color for color, count in counts.items() if count == 1]
    assert len(marker_candidates) == 1, "expected exactly one single-pixel marker color"
    marker = marker_candidates[0]

    obstacle_candidates: list[int] = []
    dot_candidates: list[int] = []
    for color in colors:
        if color == marker:
            continue
        comps = _components(g == color)
        if len(comps) == 1 and len(comps[0]) > 1:
            obstacle_candidates.append(color)
        else:
            dot_candidates.append(color)

    assert len(obstacle_candidates) == 1, "expected one connected obstacle color"
    assert len(dot_candidates) == 1, "expected one scattered dot color"
    dot_color = dot_candidates[0]
    assert all(len(comp) == 1 for comp in _components(g == dot_color)), "dots must be isolated singles"
    return bg, marker, obstacle_candidates[0], dot_color


def _adjacent_to_color(g: np.ndarray, pos: Coord, color: int) -> bool:
    return any(
        in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == int(color)
        for rr, cc in neighbors4(*pos)
    )


def _sightline_to_obstacle(
    g: np.ndarray,
    marker_pos: Coord,
    obstacle: int,
    bg: int,
) -> tuple[Coord, Coord]:
    hits: list[tuple[Coord, Coord]] = []
    sr, sc = marker_pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = sr + dr, sc + dc
        previous_bg: Coord | None = None
        while in_bounds(g.shape, r, c):
            value = int(g[r, c])
            if value == int(obstacle):
                assert previous_bg is not None, "marker must not touch the obstacle directly"
                hits.append(((dr, dc), previous_bg))
                break
            assert value == int(bg), "the unique sightline must cross only background"
            previous_bg = (r, c)
            r += dr
            c += dc

    assert len(hits) == 1, "expected exactly one cardinal sightline to the obstacle"
    return hits[0]


def _branch_paths(
    g: np.ndarray,
    start: Coord,
    direction: Coord,
    obstacle: int,
    bg: int,
) -> list[tuple[Coord, list[Coord]]]:
    dr, dc = direction
    turns = [(dr, -1), (dr, 1)] if dr else [(-1, dc), (1, dc)]
    paths: list[tuple[Coord, list[Coord]]] = []
    for tr, tc in turns:
        r, c = start
        path: list[Coord] = []
        while True:
            nr, nc = r + tr, c + tc
            if not in_bounds(g.shape, nr, nc):
                break
            if int(g[nr, nc]) != int(bg):
                break
            if not _adjacent_to_color(g, (nr, nc), obstacle):
                break
            path.append((nr, nc))
            r, c = nr, nc
        paths.append(((tr, tc), path))
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the marker by sliding to the obstacle and then walking a contour-adjacent branch for as many steps as there are dots.

    Input structure:
    - A dominant background fills the grid.
    - One color forms a single 4-connected obstacle block.
    - A one-pixel marker of another color has exactly one horizontal or vertical sightline to the obstacle.
    - A third color appears as several isolated dots away from the sightline and contour cells.

    Rule:
    - Slide the marker along its unique sightline to the last background cell before the obstacle.
    - From that contour-start cell, test the two diagonal contour-adjacent turns determined by the sightline direction.
    - Count the isolated dots; exactly one turn has at least that many contour-adjacent background cells.
    - Walk exactly that many cells along the valid turn, place the marker there, and erase all dots.

    Key insight:
    - The dot set is a step budget, not decoration: the marker first locks onto the obstacle by sightline, then uses the dot count as distance measured along the outside contour branch rather than along a straight line.
    """
    g = np.asarray(grid, dtype=int)
    bg, marker, obstacle, dot_color = _classify_roles(g)

    marker_pos = tuple(map(int, np.argwhere(g == marker)[0]))
    direction, contour_start = _sightline_to_obstacle(g, marker_pos, obstacle, bg)
    assert _adjacent_to_color(g, contour_start, obstacle), "contour start must touch the obstacle"

    step_budget = int(np.count_nonzero(g == dot_color))
    assert step_budget >= 1
    branches = _branch_paths(g, contour_start, direction, obstacle, bg)
    valid = [path for _turn, path in branches if len(path) >= step_budget]
    assert len(valid) == 1, "dot count must select exactly one contour branch"
    destination = valid[0][step_budget - 1]

    out = g.copy()
    out[marker_pos] = bg
    out[g == dot_color] = bg
    out[destination] = marker
    return out


GEN_RETRIES = 500
LONG_LEN_RANGE = (3, 7)
MAX_SHORT_LEN = 4
CONTOUR_COL_RANGE = (2, 6)
WIDTH_PADDING_RANGE = (3, 7)
PAD_RANGE = (0, 4)


def _transform(grid: np.ndarray, rotations: int, mirror: bool) -> np.ndarray:
    out = np.rot90(grid, rotations)
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _branch_budget(short_len: int, long_len: int) -> int:
    low = max(2, short_len + 1)
    assert low <= long_len
    if low <= long_len - 1 and random.random() < 0.75:
        return random.randint(low, long_len - 1)
    return random.randint(low, long_len)


def _make_left_boundary(short_len: int, long_len: int, base_left: int) -> list[int]:
    start_row = short_len + 1
    height = short_len + long_len + 3
    lefts = [0] * height

    for i in range(short_len + 1):
        lefts[start_row - i] = base_left + i
    lefts[0] = random.randint(0, lefts[1])

    for i in range(long_len + 1):
        lefts[start_row + i] = base_left + i
    lefts[-1] = random.randint(0, lefts[-2])
    return lefts


def _reserve_safe_dots(
    grid: np.ndarray,
    marker_pos: Coord,
    contour_start: Coord,
    obstacle: int,
    dot_count: int,
    path_cells: set[Coord],
) -> list[Coord] | None:
    candidates: list[Coord] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != get_bg(grid):
                continue
            if r == marker_pos[0] or c == marker_pos[1]:
                continue
            if r == marker_pos[0] and c <= contour_start[1]:
                continue
            if abs(r - marker_pos[0]) + abs(c - marker_pos[1]) <= 1:
                continue
            if (r, c) in path_cells:
                continue
            if min(abs(r - pr) + abs(c - pc) for pr, pc in path_cells) <= 1:
                continue
            if any(in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == int(obstacle) for nr, nc in neighbors4(r, c)):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    chosen: list[Coord] = []
    for cell in candidates:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == dot_count:
                return chosen
    return None


def _pad_canvas(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    top = random.randint(0, min(PAD_RANGE[1], 30 - h))
    bottom = random.randint(0, min(PAD_RANGE[1], 30 - h - top))
    left = random.randint(0, min(PAD_RANGE[1], 30 - w))
    right = random.randint(0, min(PAD_RANGE[1], 30 - w - left))
    out = np.full((h + top + bottom, w + left + right), bg, dtype=int)
    out[top : top + h, left : left + w] = grid
    return out


def _semantic_checks(inp: np.ndarray) -> None:
    bg, marker, obstacle, dot_color = _classify_roles(inp)
    marker_pos = tuple(map(int, np.argwhere(inp == marker)[0]))
    direction, contour_start = _sightline_to_obstacle(inp, marker_pos, obstacle, bg)
    branches = _branch_paths(inp, contour_start, direction, obstacle, bg)
    lengths = sorted(len(path) for _turn, path in branches)
    dot_count = int(np.count_nonzero(inp == dot_color))

    assert dot_count >= 2, "generator should produce a visible dot set"
    assert lengths[1] > lengths[0], "the two contour options must differ"
    assert lengths[0] < dot_count <= lengths[1], "dot count must choose exactly one branch"

    all_path_cells = {contour_start}
    for _turn, path in branches:
        all_path_cells.update(path)
    for r, c in map(tuple, np.argwhere(inp == dot_color)):
        assert min(abs(r - pr) + abs(c - pc) for pr, pc in all_path_cells) > 1
        assert not any(
            in_bounds(inp.shape, nr, nc) and int(inp[nr, nc]) == int(obstacle)
            for nr, nc in neighbors4(r, c)
        )

    out = main(inp)
    assert np.array_equal(inp == obstacle, out == obstacle), "obstacle must stay fixed"
    assert np.count_nonzero(out == dot_color) == 0, "dots must be erased"
    assert np.count_nonzero(out == marker) == 1, "marker must remain unique"
    chosen = [path for _turn, path in branches if len(path) >= dot_count][0]
    expected_dest = chosen[dot_count - 1]
    actual_dest = tuple(map(int, np.argwhere(out == marker)[0]))
    assert actual_dest == expected_dest, "marker must stop at the dot-budget step"


def _generate_attempt() -> np.ndarray | None:
    bg, obstacle, marker, dot_color = random.sample(range(10), 4)
    long_len = random.randint(*LONG_LEN_RANGE)
    short_len = random.randint(0, min(MAX_SHORT_LEN, long_len - 2))
    base_left = random.randint(1, 4)
    lefts = _make_left_boundary(short_len, long_len, base_left)
    start_row = short_len + 1

    contour_col = random.randint(max(CONTOUR_COL_RANGE[0], base_left), CONTOUR_COL_RANGE[1])
    obstacle_c0 = contour_col + 1 - base_left
    width = random.randint(max(lefts) + WIDTH_PADDING_RANGE[0], max(lefts) + WIDTH_PADDING_RANGE[1])

    height = len(lefts) + 2
    width_total = obstacle_c0 + width + 3
    if max(height, width_total) > 30:
        return None

    inp = np.full((height, width_total), bg, dtype=int)
    obstacle_r0 = 1
    for rr, left in enumerate(lefts):
        inp[obstacle_r0 + rr, obstacle_c0 + left : obstacle_c0 + width] = obstacle

    marker_pos = (obstacle_r0 + start_row, 0)
    if int(inp[marker_pos]) != bg:
        return None
    inp[marker_pos] = marker

    direction, contour_start = _sightline_to_obstacle(inp, marker_pos, obstacle, bg)
    assert direction == (0, 1)
    branches = _branch_paths(inp, contour_start, direction, obstacle, bg)
    lengths = sorted(len(path) for _turn, path in branches)
    if lengths[1] <= lengths[0]:
        return None

    dot_count = _branch_budget(lengths[0], lengths[1])
    path_cells = {contour_start}
    for _turn, path in branches:
        path_cells.update(path)
    dots = _reserve_safe_dots(inp, marker_pos, contour_start, obstacle, dot_count, path_cells)
    if dots is None:
        return None
    for r, c in dots:
        inp[r, c] = dot_color

    inp = _transform(inp, random.randrange(4), random.random() < 0.5)
    inp = _pad_canvas(inp, bg)
    if max(inp.shape) > 30:
        return None
    _semantic_checks(inp)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid contour-walk puzzle input")
