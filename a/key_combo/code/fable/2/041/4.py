import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
WALLPAPER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

TILE_H_RANGE = (3, 6)
TILE_W_RANGE = (3, 6)
REPEAT_RANGE = (4, 5)
PATH_LEN_RANGE = (6, 18)
GEN_RETRY_LIMIT = 500
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, tile.shape)
    rows = np.arange(shape[0])[:, None] % h
    cols = np.arange(shape[1])[None, :] % w
    return tile[rows, cols]


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    tile = np.asarray(tile, dtype=int)
    h, w = map(int, tile.shape)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if np.array_equal(tile, tile[rows % ph, cols % pw]):
                return ph, pw
    raise AssertionError("unreachable")


def _infer_minimal_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    known = np.argwhere(g != BLACK)

    best_area = None
    best_count = 0
    best_tile = None
    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            tile = np.full((tile_h, tile_w), -1, dtype=int)
            valid = True
            for r, c in known:
                tr = int(r) % tile_h
                tc = int(c) % tile_w
                value = int(g[r, c])
                if tile[tr, tc] == -1:
                    tile[tr, tc] = value
                elif tile[tr, tc] != value:
                    valid = False
                    break
            if not valid or np.any(tile == -1):
                continue

            area = tile_h * tile_w
            if best_area is None or area < best_area:
                best_area = area
                best_count = 1
                best_tile = tile.copy()
            elif area == best_area:
                best_count += 1

    assert best_tile is not None, "expected at least one fully observed repeating tile"
    assert best_count == 1, "expected a unique minimal repeating tile"
    return best_tile


def _path_info(mask: np.ndarray) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    coords = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    assert coords, "expected a black tunnel"

    seen = set()
    stack = [next(iter(coords))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in neighbors4(*cell):
            if nbr in coords and nbr not in seen:
                stack.append(nbr)
    assert seen == coords, "expected a single black tunnel component"

    endpoints = []
    for r, c in sorted(coords):
        degree = sum((nr, nc) in coords for nr, nc in neighbors4(r, c))
        assert degree in (1, 2), "expected a width-1 branchless tunnel"
        if degree == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "expected exactly two tunnel endpoints"
    return coords, endpoints


def _nest_color(tile: np.ndarray) -> int:
    counts = Counter(map(int, tile.ravel()))
    min_count = min(counts.values())
    rare = [color for color, count in counts.items() if count == min_count]
    assert len(rare) == 1 and min_count == 1, "expected one uniquely rare nest color"
    return int(rare[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the black tunnel from the wallpaper period, then mark the nest-colored endpoint in gray.

    Input structure:
    - The grid is a periodic non-black wallpaper with some cells replaced by one 4-connected black tunnel.
    - The tunnel is one cell thick, connected, and has exactly two endpoints with no branches.
    - Ignoring black, the smallest repeating tile is unique, and that tile has one uniquely rare color cell.

    Rule:
    - Ignore black cells and infer the unique minimal wallpaper tile.
    - Restore every black tunnel cell with the wallpaper color at the same row/column phase.
    - Among the two tunnel endpoints, find the one whose restored color is the unique rare nest color and recolor that endpoint gray.

    Key insight:
    - The endpoint choice is invisible in the input itself; only after periodic repair do the two identical tunnel ends separate into a nest phase and a non-nest phase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    tile = _infer_minimal_tile(g)
    wallpaper = _repeat_tile(tile, g.shape)
    tunnel, endpoints = _path_info(g == BLACK)
    nest_color = _nest_color(tile)

    out = wallpaper
    nest_endpoints = [cell for cell in endpoints if int(out[cell]) == nest_color]
    assert len(nest_endpoints) == 1, "expected exactly one endpoint on the nest phase"
    out = out.copy()
    out[nest_endpoints[0]] = GREY
    return out


def _partition_sum(total: int, parts: int, min_each: int) -> list[int]:
    values = [min_each] * parts
    remaining = total - parts * min_each
    for _ in range(remaining):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _build_tile() -> np.ndarray | None:
    tile_h = random.randint(*TILE_H_RANGE)
    tile_w = random.randint(*TILE_W_RANGE)
    area = tile_h * tile_w
    max_colors = min(5, 1 + (area - 1) // 2)
    n_colors = random.randint(3, max_colors)
    chosen = random.sample(WALLPAPER_COLORS, n_colors)
    nest_color = chosen[0]
    other_colors = chosen[1:]
    counts = _partition_sum(area - 1, len(other_colors), 2)

    flat = [nest_color]
    for color, count in zip(other_colors, counts):
        flat.extend([color] * count)
    random.shuffle(flat)
    tile = np.array(flat, dtype=int).reshape(tile_h, tile_w)

    if _minimal_period(tile) != (tile_h, tile_w):
        return None
    if _nest_color(tile) != nest_color:
        return None
    return tile


def _step_ok(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    used: set[tuple[int, int]],
    shape: tuple[int, int],
    margin: int,
) -> bool:
    r, c = nxt
    h, w = shape
    if not (margin <= r < h - margin and margin <= c < w - margin):
        return False
    if nxt in used:
        return False
    for nbr in neighbors4(r, c):
        if nbr in used and nbr != cur:
            return False
    return True


def _continuations(
    cur: tuple[int, int],
    used: set[tuple[int, int]],
    shape: tuple[int, int],
    margin: int,
) -> int:
    count = 0
    for dr, dc in DIRS4:
        nxt = (cur[0] + dr, cur[1] + dc)
        if _step_ok(nxt, cur, used, shape, margin):
            count += 1
    return count


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        prev = (path[i - 1][0] - path[i - 2][0], path[i - 1][1] - path[i - 2][1])
        cur = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        turns += prev != cur
    return turns


def _endpoint_axes_match(path: list[tuple[int, int]]) -> bool:
    head = (abs(path[1][0] - path[0][0]), abs(path[1][1] - path[0][1]))
    tail = (abs(path[-1][0] - path[-2][0]), abs(path[-1][1] - path[-2][1]))
    return head == tail


def _grow_path(
    start: tuple[int, int],
    target_len: int,
    shape: tuple[int, int],
    margin: int,
) -> list[tuple[int, int]] | None:
    path = [start]
    used = {start}

    def search(prev_dir: tuple[int, int] | None) -> list[tuple[int, int]] | None:
        if len(path) == target_len:
            return path.copy()

        directions = DIRS4[:]
        random.shuffle(directions)
        if prev_dir is not None and prev_dir in directions:
            directions.remove(prev_dir)
            if random.random() < 0.55:
                directions.insert(0, prev_dir)
            else:
                directions.append(prev_dir)

        cur = path[-1]
        for dr, dc in directions:
            nxt = (cur[0] + dr, cur[1] + dc)
            if not _step_ok(nxt, cur, used, shape, margin):
                continue
            used.add(nxt)
            path.append(nxt)
            if len(path) == target_len or _continuations(nxt, used, shape, margin) > 0:
                result = search((dr, dc))
                if result is not None:
                    return result
            path.pop()
            used.remove(nxt)
        return None

    return search(None)


def _visible_phase_counts(grid: np.ndarray, tile_shape: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = tile_shape
    counts = np.zeros((tile_h, tile_w), dtype=int)
    for r, c in np.argwhere(grid != BLACK):
        counts[int(r) % tile_h, int(c) % tile_w] += 1
    return counts


def _pick_tunnel(wallpaper: np.ndarray, nest_color: int) -> list[tuple[int, int]] | None:
    h, w = map(int, wallpaper.shape)
    margin = 1
    max_len = min(PATH_LEN_RANGE[1], (h - 2 * margin) * (w - 2 * margin) // 3)
    if max_len < PATH_LEN_RANGE[0]:
        return None

    all_starts = [
        (int(r), int(c))
        for r in range(margin, h - margin)
        for c in range(margin, w - margin)
    ]
    for start_is_nest in random.sample([False, True], k=2):
        starts = [cell for cell in all_starts if (int(wallpaper[cell]) == nest_color) == start_is_nest]
        random.shuffle(starts)
        for _ in range(40):
            for start in starts:
                target_len = random.randint(PATH_LEN_RANGE[0], max_len)
                path = _grow_path(start, target_len, (h, w), margin)
                if path is None:
                    continue
                if _turn_count(path) < 1:
                    continue
                if not _endpoint_axes_match(path):
                    continue
                end = path[-1]
                if (int(wallpaper[end]) == nest_color) == start_is_nest:
                    continue
                if abs(path[0][0] - end[0]) + abs(path[0][1] - end[1]) < 4:
                    continue
                return path
    return None


def _semantic_checks(inp: np.ndarray, wallpaper: np.ndarray) -> tuple[int, int]:
    tile = _infer_minimal_tile(inp)
    assert np.array_equal(_repeat_tile(tile, inp.shape), wallpaper)
    tunnel, endpoints = _path_info(inp == BLACK)
    assert len(tunnel) >= PATH_LEN_RANGE[0]
    assert np.all(_visible_phase_counts(inp, tile.shape) > 0), "every tile phase must stay visible"
    nest_color = _nest_color(tile)
    nest_endpoints = [cell for cell in endpoints if int(wallpaper[cell]) == nest_color]
    assert len(nest_endpoints) == 1
    out = main(inp)
    expected = wallpaper.copy()
    expected[nest_endpoints[0]] = GREY
    assert np.array_equal(out, expected)
    return nest_endpoints[0]


def _gen_attempt() -> np.ndarray | None:
    tile = _build_tile()
    if tile is None:
        return None

    reps_h = random.randint(*REPEAT_RANGE)
    reps_w = random.randint(*REPEAT_RANGE)
    wallpaper = _repeat_tile(tile, (tile.shape[0] * reps_h, tile.shape[1] * reps_w))
    nest_color = _nest_color(tile)
    path = _pick_tunnel(wallpaper, nest_color)
    if path is None:
        return None

    inp = wallpaper.copy()
    for r, c in path:
        inp[r, c] = BLACK

    try:
        _semantic_checks(inp, wallpaper)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid return-to-nest input")
