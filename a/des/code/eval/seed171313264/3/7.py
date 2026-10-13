from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 400
EXTRA_PATH_RETRIES = 400


def _ordered_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    return (a, b) if a <= b else (b, a)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1, "segment must be straight"
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _l_path(
    start: tuple[int, int],
    end: tuple[int, int],
    horizontal_first: bool,
) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = _ordered_pair(start, end)
    if horizontal_first:
        corner = (r0, c1)
    else:
        corner = (r1, c0)
    first = _segment_cells((r0, c0), corner)
    second = _segment_cells(corner, (r1, c1))[1:]
    return first + second


def _path_clear(path: list[tuple[int, int]], obstacles: np.ndarray) -> bool:
    return not any(obstacles[r, c] for r, c in path)


def _pick_path(
    start: tuple[int, int],
    end: tuple[int, int],
    obstacles: np.ndarray,
) -> tuple[list[tuple[int, int]], bool]:
    horizontal_path = _l_path(start, end, horizontal_first=True)
    if _path_clear(horizontal_path, obstacles):
        return horizontal_path, True
    vertical_path = _l_path(start, end, horizontal_first=False)
    assert _path_clear(vertical_path, obstacles), "expected at least one L path to be open"
    return vertical_path, False


def _find_priority_strip(
    grid: np.ndarray,
    bg: int,
) -> tuple[list[int], set[tuple[int, int]]]:
    counts = Counter(int(v) for v in grid.reshape(-1))
    h, w = grid.shape
    candidates: list[tuple[list[int], set[tuple[int, int]]]] = []

    def consider(values: np.ndarray, coords: list[tuple[int, int]]) -> None:
        non_bg_idx = [i for i, v in enumerate(values) if int(v) != bg]
        if len(non_bg_idx) < 2:
            return
        if non_bg_idx != list(range(non_bg_idx[0], non_bg_idx[-1] + 1)):
            return
        colors = [int(values[i]) for i in non_bg_idx]
        if len(set(colors)) != len(colors):
            return
        if not all(counts[color] == 3 for color in colors):
            return
        strip_positions = {coords[i] for i in non_bg_idx}
        candidates.append((colors, strip_positions))

    consider(grid[0, :], [(0, c) for c in range(w)])
    consider(grid[h - 1, :], [(h - 1, c) for c in range(w)])
    consider(grid[:, 0], [(r, 0) for r in range(h)])
    consider(grid[:, w - 1], [(r, w - 1) for r in range(h)])

    assert len(candidates) == 1, "expected exactly one border priority strip"
    return candidates[0]


def _identify_scene(
    grid: np.ndarray,
) -> tuple[int, list[int], set[tuple[int, int]], dict[int, tuple[tuple[int, int], tuple[int, int]]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.reshape(-1)).argmax())
    order, strip_positions = _find_priority_strip(g, bg)

    strip_by_color = {int(g[r, c]): (int(r), int(c)) for r, c in strip_positions}
    markers: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in order:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        coords = [pt for pt in coords if pt != strip_by_color[color]]
        assert len(coords) == 2, "each strip color must have exactly two markers outside the strip"
        markers[color] = _ordered_pair(coords[0], coords[1])

    obstacles = (g != bg) & (~np.isin(g, np.array(order, dtype=int)))
    return bg, order, strip_positions, markers, obstacles


def _render_paths(
    grid: np.ndarray,
    bg: int,
    order: list[int],
    paths: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for color in order:
        for r, c in paths[color]:
            if out[r, c] == bg:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each color pair with an L path and resolve crossings by the border priority strip.

    The strip order is read left-to-right for rows and top-to-bottom for columns. To make the two possible
    L corners deterministic, the earlier marker in reading order is treated as the start point.
    """
    g = np.asarray(grid, dtype=int)
    bg, order, _strip_positions, markers, obstacles = _identify_scene(g)

    paths: dict[int, list[tuple[int, int]]] = {}
    for color in order:
        paths[color], _ = _pick_path(markers[color][0], markers[color][1], obstacles)

    return _render_paths(g, bg, order, paths)


def _add_crossing_pairs(
    rng: random.Random,
    h: int,
    w: int,
    colors: list[int],
) -> tuple[
    dict[int, tuple[tuple[int, int], tuple[int, int]]],
    dict[int, list[tuple[int, int]]],
    set[tuple[int, int]],
]:
    center_r = rng.randint(3, h - 4)
    center_c = rng.randint(3, w - 4)
    left_c = rng.randint(1, center_c - 2)
    right_c = rng.randint(center_c + 2, w - 2)
    up_r = rng.randint(1, center_r - 2)
    down_r = rng.randint(center_r + 2, h - 2)

    mirror = rng.choice([False, True])
    pair_markers: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    pair_paths: dict[int, list[tuple[int, int]]] = {}
    obstacles: set[tuple[int, int]] = set()

    if not mirror:
        a_start, a_end = (center_r, left_c), (down_r, center_c)
        b_start, b_end = (up_r, center_c), (center_r, right_c)
        a_block_len = rng.randint(1, min(2, down_r - center_r - 1))
        b_block_len = rng.randint(1, min(2, right_c - center_c - 1))
        for d in range(1, a_block_len + 1):
            obstacles.add((center_r + d, left_c))
        for d in range(1, b_block_len + 1):
            obstacles.add((up_r, center_c + d))
    else:
        a_start, a_end = (center_r, right_c), (down_r, center_c)
        b_start, b_end = (up_r, center_c), (center_r, left_c)
        a_block_len = rng.randint(1, min(2, down_r - center_r - 1))
        b_block_len = rng.randint(1, min(2, center_c - left_c - 1))
        for d in range(1, a_block_len + 1):
            obstacles.add((center_r + d, right_c))
        for d in range(1, b_block_len + 1):
            obstacles.add((up_r, center_c - d))

    pair_markers[colors[0]] = _ordered_pair(a_start, a_end)
    pair_markers[colors[1]] = _ordered_pair(b_start, b_end)
    pair_paths[colors[0]] = _l_path(a_start, a_end, horizontal_first=True)
    pair_paths[colors[1]] = _l_path(b_start, b_end, horizontal_first=False)

    assert set(pair_paths[colors[0]]) & set(pair_paths[colors[1]]) == {(center_r, center_c)}
    assert not set(pair_paths[colors[0]]) & obstacles
    assert not set(pair_paths[colors[1]]) & obstacles
    return pair_markers, pair_paths, obstacles


def _sample_extra_pair(
    rng: random.Random,
    h: int,
    w: int,
    occupied_markers: set[tuple[int, int]],
    planned_paths: dict[int, list[tuple[int, int]]],
    obstacles: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]], tuple[int, int]] | None:
    path_union = set().union(*(set(path) for path in planned_paths.values()))
    for _ in range(EXTRA_PATH_RETRIES):
        r0 = rng.randint(1, h - 4)
        r1 = rng.randint(r0 + 2, h - 2)
        c0 = rng.randint(1, w - 2)
        c1 = rng.randint(1, w - 2)
        if c0 == c1:
            continue

        start, end = _ordered_pair((r0, c0), (r1, c1))
        horizontal_first = rng.choice([False, True])
        chosen_path = _l_path(start, end, horizontal_first)
        chosen_set = set(chosen_path)
        if chosen_set & obstacles:
            continue
        if start in occupied_markers or end in occupied_markers:
            continue
        if any(cell in occupied_markers for cell in chosen_set - {start, end}):
            continue
        if chosen_set & path_union:
            continue

        alternative = _l_path(start, end, not horizontal_first)
        blocker_options = [
            cell
            for cell in alternative
            if cell not in chosen_set and cell not in occupied_markers and cell not in obstacles and cell not in path_union
        ]
        if not blocker_options:
            continue

        blocker = rng.choice(blocker_options)
        return (start, end), chosen_path, blocker
    return None


def _place_priority_strip(
    rng: random.Random,
    h: int,
    w: int,
    order: list[int],
    markers: dict[int, tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[str, int]:
    side = rng.choice(["top", "bottom", "left", "right"])
    line_len = w if side in ("top", "bottom") else h
    start_lo = 1 if line_len - len(order) >= 3 else 0
    start_hi = line_len - len(order) - (1 if line_len - len(order) >= 3 else 0)
    starts = list(range(start_lo, start_hi + 1))
    rng.shuffle(starts)

    for start in starts:
        ok = True
        for offset, color in enumerate(order):
            if side == "top":
                pos = (0, start + offset)
            elif side == "bottom":
                pos = (h - 1, start + offset)
            elif side == "left":
                pos = (start + offset, 0)
            else:
                pos = (start + offset, w - 1)

            for marker in markers[color]:
                if abs(pos[0] - marker[0]) + abs(pos[1] - marker[1]) <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return side, start

    raise RuntimeError("failed to place priority strip without touching same-color markers")


def _build_grid(
    h: int,
    w: int,
    order: list[int],
    markers: dict[int, tuple[tuple[int, int], tuple[int, int]]],
    obstacles: set[tuple[int, int]],
    side: str,
    strip_start: int,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in obstacles:
        grid[r, c] = GREY

    for color, pts in markers.items():
        for r, c in pts:
            assert grid[r, c] == BG
            grid[r, c] = color

    for offset, color in enumerate(order):
        if side == "top":
            pos = (0, strip_start + offset)
        elif side == "bottom":
            pos = (h - 1, strip_start + offset)
        elif side == "left":
            pos = (strip_start + offset, 0)
        else:
            pos = (strip_start + offset, w - 1)
        assert grid[pos] == BG
        grid[pos] = color

    return grid


def _validate_generated_scene(
    grid: np.ndarray,
    expected_order: list[int],
    planned_paths: dict[int, list[tuple[int, int]]],
    planned_branches: dict[int, bool],
) -> bool:
    bg, order, _strip_positions, markers, obstacles = _identify_scene(grid)
    if order != expected_order:
        return False

    recomputed_paths: dict[int, list[tuple[int, int]]] = {}
    recomputed_branches: dict[int, bool] = {}
    for color in order:
        path, branch = _pick_path(markers[color][0], markers[color][1], obstacles)
        recomputed_paths[color] = path
        recomputed_branches[color] = branch
        if set(path) != set(planned_paths[color]):
            return False
        if branch != planned_branches[color]:
            return False

    overlaps = Counter(cell for path in recomputed_paths.values() for cell in path)
    if sum(1 for n in overlaps.values() if n > 1) == 0:
        return False
    if {True, False} - set(recomputed_branches.values()):
        return False

    expected = _render_paths(grid, bg, order, recomputed_paths)
    actual = main(grid)
    return np.array_equal(expected, actual) and np.any(actual != grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 18)
    w = rng.randint(12, 18)
    color_count = rng.randint(3, 4)
    colors = rng.sample(PALETTE, color_count)

    markers, planned_paths, obstacles = _add_crossing_pairs(rng, h, w, colors)
    occupied_markers = {pt for pair in markers.values() for pt in pair}

    for color in colors[2:]:
        sample = _sample_extra_pair(rng, h, w, occupied_markers, planned_paths, obstacles)
        if sample is None:
            return None
        pair, chosen_path, blocker = sample
        markers[color] = pair
        planned_paths[color] = chosen_path
        occupied_markers.update(pair)
        obstacles.add(blocker)

    obstacle_mask = np.zeros((h, w), dtype=bool)
    for r, c in obstacles:
        obstacle_mask[r, c] = True

    planned_branches: dict[int, bool] = {}
    for color in colors:
        path, branch = _pick_path(markers[color][0], markers[color][1], obstacle_mask)
        if set(path) != set(planned_paths[color]):
            return None
        planned_branches[color] = branch

    order = colors[:]
    rng.shuffle(order)
    try:
        side, strip_start = _place_priority_strip(rng, h, w, order, markers)
    except RuntimeError:
        return None

    grid = _build_grid(h, w, order, markers, obstacles, side, strip_start)
    if not _validate_generated_scene(grid, order, planned_paths, planned_branches):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
