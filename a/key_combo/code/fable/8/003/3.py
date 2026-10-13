from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PATH = 5
MARBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
GEN_GRID_RANGE = (20, 26)
GEN_MARGIN_RANGE = (4, 6)
GEN_TURN_RANGE = (1, 3)
GEN_MARBLE_COUNT_RANGE = (3, 5)
GEN_MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def mask_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if cells:
        rows, cols = zip(*sorted(cells))
        mask[list(rows), list(cols)] = True
    return mask


def dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = True
        for dr, dc in DIR8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def label_mask(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: set[tuple[int, int]] = set()
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _path_degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        degrees[(r, c)] = sum((nr, nc) in cells for nr, nc in neighbors4(r, c))
    return degrees


def _is_open_simple_path(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    if len(label_mask(mask_from_cells((_bbox(cells)[2] + 1, _bbox(cells)[3] + 1), cells))) != 1:
        return False
    degrees = _path_degrees(cells)
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return False
    return all(deg in (1, 2) for deg in degrees.values())


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_path_color(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        cells = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
        if len(cells) < 4:
            continue
        comps = label_mask(grid == color)
        if len(comps) == 2 and all(len(comp) >= 2 for comp in comps):
            if all(_is_open_simple_path(comp) for comp in comps):
                candidates.append(int(color))
    assert len(candidates) == 1, "expected exactly one two-component path color"
    return candidates[0]


def _extract_channel(path_comps: list[set[tuple[int, int]]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    dilated = [dilate8(mask_from_cells(shape, comp)) for comp in path_comps]
    channel_mask = dilated[0] & dilated[1] & (~(mask_from_cells(shape, path_comps[0]) | mask_from_cells(shape, path_comps[1])))
    channel_comps = label_mask(channel_mask)
    assert len(channel_comps) == 1, "expected a unique channel component"
    channel = channel_comps[0]
    degrees = _path_degrees(channel)
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "expected exactly two channel openings"
    assert all(deg in (1, 2) for deg in degrees.values()), "channel must be a simple corridor"
    return channel


def _channel_endpoints(channel: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [cell for cell, deg in _path_degrees(channel).items() if deg == 1]


def _outward_dir(channel: set[tuple[int, int]], endpoint: tuple[int, int]) -> tuple[int, int]:
    neighbors = [(nr, nc) for nr, nc in neighbors4(*endpoint) if (nr, nc) in channel]
    assert len(neighbors) == 1
    nr, nc = neighbors[0]
    return endpoint[0] - nr, endpoint[1] - nc


def _marbles(grid: np.ndarray, bg: int, path_color: int) -> list[tuple[int, int, int]]:
    marbles = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere((grid != bg) & (grid != path_color))]
    assert marbles, "expected marbles"
    colors = [color for _, _, color in marbles]
    assert len(colors) == len(set(colors)), "marble colors must be distinct"
    return marbles


def _queue_cells(grid: np.ndarray, channel: set[tuple[int, int]], exit_cell: tuple[int, int]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    dr, dc = _outward_dir(channel, exit_cell)
    cells: list[tuple[int, int]] = []
    r = exit_cell[0] + dr
    c = exit_cell[1] + dc
    while in_bounds(grid.shape, r, c) and int(grid[r, c]) == BG:
        cells.append((r, c))
        r += dr
        c += dc
    assert cells, "expected empty run outside the exit"
    return (dr, dc), cells


def _analyze_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    path_color = _find_path_color(g, bg)
    path_comps = label_mask(g == path_color)
    assert len(path_comps) == 2
    channel = _extract_channel(path_comps, g.shape)
    endpoints = _channel_endpoints(channel)
    return {
        "bg": bg,
        "path_color": path_color,
        "channel": channel,
        "endpoints": endpoints,
    }


def _analyze(grid: np.ndarray) -> dict:
    info = _analyze_structure(grid)
    g = np.asarray(grid, dtype=int)
    marbles = _marbles(g, info["bg"], info["path_color"])

    endpoint_profiles = []
    for endpoint in info["endpoints"]:
        ds = sorted(chebyshev((r, c), endpoint) for r, c, _ in marbles)
        endpoint_profiles.append((min(ds), tuple(ds), endpoint))
    endpoint_profiles.sort(key=lambda item: (item[0], item[1]))
    assert endpoint_profiles[0][0] != endpoint_profiles[1][0], "entrance must be uniquely closer to marbles"
    entrance = endpoint_profiles[0][2]
    exit_cell = endpoint_profiles[1][2]

    ranked = sorted(marbles, key=lambda item: chebyshev((item[0], item[1]), entrance))
    ranked_dists = [chebyshev((r, c), entrance) for r, c, _ in ranked]
    assert len(ranked_dists) == len(set(ranked_dists)), "marble distances to entrance must be unique"
    exit_dir, run = _queue_cells(g, info["channel"], exit_cell)
    assert len(run) >= len(ranked), "exit run too short for marble queue"
    queue = list(reversed(run))[: len(ranked)]
    return {
        "bg": info["bg"],
        "path_color": info["path_color"],
        "channel": info["channel"],
        "entrance": entrance,
        "exit": exit_cell,
        "exit_dir": exit_dir,
        "marbles": ranked,
        "queue": queue,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Queue the scattered marbles by entrance distance and eject them into a wall-stopped line outside the corridor.

    Input structure:
    - The background is black.
    - Exactly one foreground color forms two disjoint grey 4-connected one-pixel-wide open paths.
    - The 8-neighborhood dilation overlap of those two paths is a one-cell-wide corridor with two openings, and several uniquely colored singleton marbles lie near one opening.

    Rule:
    - Recover the corridor as the intersection of the two paths' one-step 8-neighborhood dilations, excluding the path cells themselves.
    - The opening nearer to the marbles is the entrance and the other opening is the exit.
    - Sort the marbles by increasing chessboard distance to the entrance, erase their original positions, then place them outside the exit along the exit direction: the first marble slides to the farthest reachable empty cell before the border, and each later marble stops one cell behind the previous one.

    Key insight:
    - The visible grey strokes are only the walls of an implicit transport tube; the answer is a reordered replay of the hidden entrance queue, not a fill of the corridor itself.
    """
    info = _analyze(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r, c, _color in info["marbles"]:
        out[r, c] = info["bg"]
    for (r, c, color), (qr, qc) in zip(info["marbles"], info["queue"]):
        out[qr, qc] = color
    return out


def _append_line(path: list[tuple[int, int]], r1: int, c1: int) -> None:
    r0, c0 = path[-1]
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            path.append((r0, c))
    else:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            path.append((r, c0))


def _partition_total(total: int, count: int, low: int, high: int) -> list[int] | None:
    parts = []
    remaining = total
    for i in range(count - 1):
        rest = count - i - 1
        lo = max(low, remaining - high * rest)
        hi = min(high, remaining - low * rest)
        if lo > hi:
            return None
        value = random.randint(lo, hi)
        parts.append(value)
        remaining -= value
    if not (low <= remaining <= high):
        return None
    parts.append(remaining)
    return parts


def _channel_too_close(path: list[tuple[int, int]]) -> bool:
    return len(set(path)) != len(path)


def _generate_channel_path(inner: int) -> list[tuple[int, int]] | None:
    horizontal_count = random.randint(*GEN_TURN_RANGE)
    vertical_count = horizontal_count + 1
    total_vertical = inner - 1
    vertical_steps = _partition_total(total_vertical, vertical_count, 1, max(2, inner - vertical_count + 1))
    if vertical_steps is None:
        return None

    max_total_horizontal = inner - 3
    if max_total_horizontal < horizontal_count:
        return None
    total_horizontal = random.randint(horizontal_count, max_total_horizontal)
    horizontal_steps = _partition_total(total_horizontal, horizontal_count, 1, max_total_horizontal)
    if horizontal_steps is None:
        return None

    hdir = random.choice([-1, 1])
    if hdir == 1:
        start_c = random.randint(1, inner - 2 - total_horizontal)
    else:
        start_c = random.randint(1 + total_horizontal, inner - 2)

    path = [(0, start_c)]
    row = 0
    col = start_c
    for i, step in enumerate(vertical_steps):
        row += step
        _append_line(path, row, col)
        if i < horizontal_count:
            col += hdir * horizontal_steps[i]
            _append_line(path, row, col)

    if _channel_too_close(path):
        return None
    return path


def _left_normal(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _wall_sets(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    channel = set(path)
    left: set[tuple[int, int]] = set()
    right: set[tuple[int, int]] = set()
    for i, (r, c) in enumerate(path):
        dirs = []
        if i > 0:
            pr, pc = path[i - 1]
            dirs.append((r - pr, c - pc))
        if i + 1 < len(path):
            nr, nc = path[i + 1]
            dirs.append((nr - r, nc - c))
        for dr, dc in dirs:
            lr, lc = _left_normal(dr, dc)
            if (r + lr, c + lc) not in channel:
                left.add((r + lr, c + lc))
            if (r - lr, c - lc) not in channel:
                right.add((r - lr, c - lc))
        if len(dirs) == 2 and dirs[0] != dirs[1]:
            for side in (1, -1):
                offsets = []
                for dr, dc in dirs:
                    lr, lc = _left_normal(dr, dc)
                    offsets.append((side * lr, side * lc))
                er = r + offsets[0][0] + offsets[1][0]
                ec = c + offsets[0][1] + offsets[1][1]
                if (er, ec) not in channel:
                    if side == 1:
                        left.add((er, ec))
                    else:
                        right.add((er, ec))
    return left, right


def _rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k)


def _build_bare_grid() -> np.ndarray | None:
    n = random.randint(*GEN_GRID_RANGE)
    margin = random.randint(*GEN_MARGIN_RANGE)
    inner = n - 2 * margin
    if inner < 8:
        return None
    path = _generate_channel_path(inner)
    if path is None:
        return None
    shifted = [(r + margin, c + margin) for r, c in path]
    left, right = _wall_sets(shifted)
    if not left or not right:
        return None
    all_cells = left | right | set(shifted)
    if any(not in_bounds((n, n), r, c) for r, c in all_cells):
        return None
    if left & right or left & set(shifted) or right & set(shifted):
        return None
    if not _is_open_simple_path(left) or not _is_open_simple_path(right):
        return None

    grid = np.zeros((n, n), dtype=int)
    for r, c in left | right:
        grid[r, c] = PATH

    try:
        channel = _extract_channel(label_mask(grid == PATH), grid.shape)
    except AssertionError:
        return None
    if len(channel) < len(shifted):
        return None

    grid = _rotate_grid(grid, random.randint(0, 3))
    return grid


def _collinear(points: list[tuple[int, int]]) -> bool:
    if len(points) < 3:
        return True
    r0, c0 = points[0]
    r1, c1 = points[1]
    for r2, c2 in points[2:]:
        if (r1 - r0) * (c2 - c0) != (r2 - r0) * (c1 - c0):
            return False
    return True


def _marble_candidates(
    grid: np.ndarray,
    entrance: tuple[int, int],
    exit_cell: tuple[int, int],
    outward: tuple[int, int],
    target_distance: int,
) -> list[tuple[int, int]]:
    dr, dc = outward
    pr, pc = -dc, dr
    candidates = []
    for forward in range(1, target_distance + 1):
        lateral_abs = target_distance if forward < target_distance else None
        lateral_values = (
            [-target_distance, target_distance]
            if lateral_abs is not None
            else list(range(-target_distance, target_distance + 1))
        )
        for lateral in lateral_values:
            r = entrance[0] + forward * dr + lateral * pr
            c = entrance[1] + forward * dc + lateral * pc
            if not in_bounds(grid.shape, r, c):
                continue
            if int(grid[r, c]) != BG:
                continue
            if chebyshev((r, c), entrance) != target_distance:
                continue
            if chebyshev((r, c), entrance) >= chebyshev((r, c), exit_cell):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_marbles(grid: np.ndarray) -> np.ndarray | None:
    info = _analyze_structure(grid)
    endpoints = list(info["endpoints"])
    entrance = random.choice(endpoints)
    exit_cell = endpoints[0] if entrance == endpoints[1] else endpoints[1]
    outward = _outward_dir(info["channel"], entrance)
    exit_dir, exit_run = _queue_cells(grid, info["channel"], exit_cell)
    marble_count = random.randint(GEN_MARBLE_COUNT_RANGE[0], min(GEN_MARBLE_COUNT_RANGE[1], len(exit_run)))
    distances = list(range(1, marble_count + 1))
    placed: list[tuple[int, int]] = []

    def search(i: int) -> bool:
        if i == len(distances):
            return not _collinear(placed)
        distance = distances[i]
        for cell in _marble_candidates(grid, entrance, exit_cell, outward, distance):
            if cell in placed:
                continue
            placed.append(cell)
            if search(i + 1):
                return True
            placed.pop()
        return False

    if not search(0):
        return None

    out = grid.copy()
    colors = random.sample(MARBLE_COLORS, marble_count)
    for (r, c), color in zip(placed, colors):
        out[r, c] = color

    try:
        analyzed = _analyze(out)
    except AssertionError:
        return None
    if analyzed["entrance"] != entrance or analyzed["exit"] != exit_cell:
        return None
    if [chebyshev((r, c), entrance) for r, c, _ in analyzed["marbles"]] != distances:
        return None
    if analyzed["exit_dir"] != exit_dir:
        return None
    return out


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> None:
    info = _analyze(inp)
    moved_colors = [color for _, _, color in info["marbles"]]
    queue = info["queue"]
    dr, dc = info["exit_dir"]
    assert len(queue) == len(moved_colors)
    assert np.array_equal(out == PATH, inp == PATH)
    assert not np.any((out != BG) & (out != PATH) & (inp != BG) & (inp != PATH)), "original marbles must be cleared"
    for (r, c), color in zip(queue, moved_colors):
        assert int(out[r, c]) == color
    for i in range(1, len(queue)):
        pr, pc = queue[i - 1]
        r, c = queue[i]
        assert (pr - r, pc - c) == (dr, dc)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        bare = _build_bare_grid()
        if bare is None:
            continue
        grid = _place_marbles(bare)
        if grid is None:
            continue
        try:
            out = main(grid)
            _semantic_check(grid, out)
        except AssertionError:
            continue
        return grid
    raise AssertionError("failed to generate a valid corridor-marble instance")
