from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


CHANNEL_COLOR = 5
RULER_COLOR = 9
GRID_MIN = 22
GRID_MAX = 30
MAX_RULER = 5
GEN_ATTEMPTS = 400
PLAN_ATTEMPTS = 300

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _channel_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != bg)
    assert counts, "expected non-background pixels"
    repeated = [color for color, count in counts.items() if count > 1]
    if repeated:
        return int(max(repeated, key=lambda color: (counts[color], color)))
    return int(max(counts, key=counts.get))


def _find_seeds(grid: np.ndarray, bg: int, channel: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    seeds: list[tuple[int, int]] = []
    for r, c in np.argwhere((grid != bg) & (grid != channel)):
        rr = int(r)
        cc = int(c)
        if any(int(grid[nr, nc]) == channel for nr, nc in _neighbors4(rr, cc, h, w)):
            seeds.append((rr, cc))
    seeds.sort()
    assert seeds, "expected at least one seed"
    return seeds


def _ruler_length(grid: np.ndarray, seed: tuple[int, int], bg: int, channel: int) -> int:
    h, w = grid.shape
    r, c = seed
    marker = []
    for nr, nc in _neighbors4(r, c, h, w):
        val = int(grid[nr, nc])
        if val != bg and val != channel:
            marker.append((nr, nc, val))
    assert len(marker) == 1, "each seed must touch exactly one ruler"
    nr, nc, ruler_color = marker[0]
    dr = nr - r
    dc = nc - c
    length = 0
    cr = nr
    cc = nc
    while 0 <= cr < h and 0 <= cc < w and int(grid[cr, cc]) == ruler_color:
        length += 1
        cr += dr
        cc += dc
    assert 1 <= length <= MAX_RULER, "ruler length must be in [1, 5]"
    return length


def _bounded_bfs(node_mask: np.ndarray, start: tuple[int, int], max_dist: int) -> set[tuple[int, int]]:
    h, w = node_mask.shape
    q = deque([(start[0], start[1], 0)])
    seen = {start}
    while q:
        r, c, dist = q.popleft()
        if dist == max_dist:
            continue
        for nr, nc in _neighbors4(r, c, h, w):
            nxt = (nr, nc)
            if nxt in seen or not bool(node_mask[nr, nc]):
                continue
            seen.add(nxt)
            q.append((nr, nc, dist + 1))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse each seed color through its channel up to the adjacent ruler length.

    Input structure:
    - The most common color is background.
    - A fixed channel color forms 1-pixel-wide orthogonal paths that may intersect.
    - Several unique seed colors replace single channel pixels.
    - Each seed touches one perpendicular ruler segment in the background.

    Rule:
    - Read the ruler length N next to each seed.
    - Starting from the seed, spread through the channel graph to all channel cells
      within graph distance <= N.
    - Junctions branch naturally because the spread is BFS on the channel graph.
    - Rulers remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    channel = _channel_color(g, bg)
    seeds = _find_seeds(g, bg, channel)

    seed_mask = np.zeros((h, w), dtype=bool)
    for r, c in seeds:
        seed_mask[r, c] = True
    node_mask = (g == channel) | seed_mask

    out = g.copy()
    for seed in seeds:
        length = _ruler_length(g, seed, bg, channel)
        visited = _bounded_bfs(node_mask, seed, length)
        color = int(g[seed])
        for r, c in visited:
            out[r, c] = color
    return out


def _add_h(cells: set[tuple[int, int]], row: int, c0: int, c1: int) -> None:
    lo = min(c0, c1)
    hi = max(c0, c1)
    for c in range(lo, hi + 1):
        cells.add((row, c))


def _add_v(cells: set[tuple[int, int]], col: int, r0: int, r1: int) -> None:
    lo = min(r0, r1)
    hi = max(r0, r1)
    for r in range(lo, hi + 1):
        cells.add((r, col))


def _sample_spaced(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    for _ in range(80):
        values = list(range(lo, hi + 1))
        random.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - other) >= gap for other in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    chosen = []
    for value in range(lo, hi + 1):
        if all(abs(value - other) >= gap for other in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return chosen
    return None


def _build_simple_network(h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if random.random() < 0.5:
        row = random.randint(6, h - 7)
        length = random.randint(15, w - 5)
        c0 = random.randint(2, w - length - 2)
        _add_h(cells, row, c0, c0 + length - 1)
    else:
        col = random.randint(6, w - 7)
        length = random.randint(15, h - 5)
        r0 = random.randint(2, h - length - 2)
        _add_v(cells, col, r0, r0 + length - 1)
    return cells


def _build_cross_network(h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    row = random.randint(6, h - 7)
    c0 = random.randint(2, min(3, w - 19))
    c1 = random.randint(c0 + 16, w - 3)
    _add_h(cells, row, c0, c1)

    branch_count = random.randint(1, 3)
    cols = _sample_spaced(c0 + 4, c1 - 4, branch_count, 4)
    assert cols is not None
    for col in cols:
        mode = random.choice(["up", "down", "both"])
        if mode in ("up", "both"):
            top = random.randint(2, row - 4)
            _add_v(cells, col, top, row)
        if mode in ("down", "both"):
            bottom = random.randint(row + 4, h - 3)
            _add_v(cells, col, row, bottom)
    return cells


def _build_branched_network(h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    row = random.randint(7, h - 8)
    c0 = 2
    c1 = random.randint(c0 + 17, w - 3)
    _add_h(cells, row, c0, c1)

    branch_count = random.randint(2, 3)
    cols = _sample_spaced(c0 + 4, c1 - 4, branch_count, 4)
    assert cols is not None
    anchor = random.choice(cols)
    anchor_top = random.randint(2, row - 5)
    anchor_bottom = random.randint(row + 5, h - 3)
    _add_v(cells, anchor, anchor_top, anchor_bottom)

    for col in cols:
        if col == anchor:
            continue
        mode = random.choice(["up", "down", "both"])
        if mode in ("up", "both"):
            top = random.randint(2, row - 4)
            _add_v(cells, col, top, row)
        if mode in ("down", "both"):
            bottom = random.randint(row + 4, h - 3)
            _add_v(cells, col, row, bottom)

    if random.random() < 0.5 and row - anchor_top >= 6:
        sub_row = random.randint(anchor_top + 2, row - 3)
    else:
        sub_row = random.randint(row + 3, anchor_bottom - 2)
    left = random.randint(2, anchor - 4)
    right = random.randint(anchor + 4, w - 3)
    _add_h(cells, sub_row, left, right)
    return cells


def _channel_degrees(channel: set[tuple[int, int]], h: int, w: int) -> dict[tuple[int, int], int]:
    deg: dict[tuple[int, int], int] = {}
    for r, c in channel:
        deg[(r, c)] = sum((nr, nc) in channel for nr, nc in _neighbors4(r, c, h, w))
    return deg


def _orientation(channel: set[tuple[int, int]], pos: tuple[int, int], h: int, w: int) -> str | None:
    r, c = pos
    nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if (nr, nc) in channel]
    if not nbrs:
        return None
    same_row = all(nr == r for nr, _ in nbrs)
    same_col = all(nc == c for _, nc in nbrs)
    if same_row == same_col:
        return None
    return "h" if same_row else "v"


def _ruler_prefix(
    channel: set[tuple[int, int]],
    seed: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    r, c = seed
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(1, MAX_RULER + 1):
        rr = r + dr * step
        cc = c + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if (rr, cc) in channel:
            break
        ok = True
        for nr, nc in _neighbors4(rr, cc, h, w):
            if (nr, nc) not in channel:
                continue
            if step == 1 and (nr, nc) == seed:
                continue
            ok = False
            break
        if not ok:
            break
        cells.append((rr, cc))
    return cells


def _distances(channel: set[tuple[int, int]], start: tuple[int, int], h: int, w: int) -> dict[tuple[int, int], int]:
    q = deque([start])
    dist = {start: 0}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            nxt = (nr, nc)
            if nxt in dist or nxt not in channel:
                continue
            dist[nxt] = dist[(r, c)] + 1
            q.append(nxt)
    return dist


def _extra_zone(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    zone = set(cells)
    for r, c in cells:
        zone.update(_neighbors4(r, c, h, w))
    return zone


def _seed_options(channel: set[tuple[int, int]], h: int, w: int) -> list[dict[str, object]]:
    deg = _channel_degrees(channel, h, w)
    endpoints = {pos for pos, value in deg.items() if value == 1}
    options: list[dict[str, object]] = []
    for seed in sorted(channel):
        orient = _orientation(channel, seed, h, w)
        if orient is None:
            continue
        directions = [(-1, 0), (1, 0)] if orient == "h" else [(0, -1), (0, 1)]
        dist = _distances(channel, seed, h, w)
        for direction in directions:
            prefix = _ruler_prefix(channel, seed, direction, h, w)
            for length in range(1, len(prefix) + 1):
                visited = {pos for pos, value in dist.items() if value <= length}
                ruler = prefix[:length]
                extra_cells = [seed, *ruler]
                options.append(
                    {
                        "seed": seed,
                        "length": length,
                        "visited": visited,
                        "ruler": ruler,
                        "extra_zone": _extra_zone(extra_cells, h, w),
                        "touch_junction": any(deg[pos] >= 3 and dist[pos] <= length for pos in dist),
                        "hits_end": deg[seed] == 1 or any(pos != seed and pos in endpoints and dist[pos] <= length for pos in dist),
                    }
                )
    return options


def _choose_plan(
    options: list[dict[str, object]],
    want_junction: bool,
    want_end: bool,
    min_seeds: int,
    max_seeds: int,
) -> list[dict[str, object]] | None:
    distinct_seeds = sorted({tuple(opt["seed"]) for opt in options})
    if len(distinct_seeds) < min_seeds:
        return None
    if want_junction and not any(bool(opt["touch_junction"]) for opt in options):
        want_junction = False
    if want_end and not any(bool(opt["hits_end"]) for opt in options):
        want_end = False

    for _ in range(PLAN_ATTEMPTS):
        target = random.randint(min_seeds, min(max_seeds, len(distinct_seeds)))
        chosen: list[dict[str, object]] = []
        used_seeds: set[tuple[int, int]] = set()
        used_visited: set[tuple[int, int]] = set()
        used_zone: set[tuple[int, int]] = set()
        covered_junction = False
        covered_end = False

        for slot in range(target):
            remaining = target - slot
            candidates = []
            for opt in options:
                seed = tuple(opt["seed"])
                visited = set(opt["visited"])
                zone = set(opt["extra_zone"])
                if seed in used_seeds:
                    continue
                if visited & used_visited:
                    continue
                if zone & used_zone:
                    continue
                candidates.append(opt)
            if not candidates:
                break

            if want_junction and not covered_junction:
                need_now = remaining == 1
                prefs = [opt for opt in candidates if bool(opt["touch_junction"])]
                if need_now and not prefs:
                    break
                if prefs and (need_now or random.random() < 0.7):
                    candidates = prefs
            if want_end and not covered_end:
                need_now = remaining == 1 and not (want_junction and not covered_junction)
                prefs = [opt for opt in candidates if bool(opt["hits_end"])]
                if need_now and not prefs:
                    break
                if prefs and (need_now or random.random() < 0.6):
                    candidates = prefs

            opt = random.choice(candidates)
            chosen.append(opt)
            used_seeds.add(tuple(opt["seed"]))
            used_visited |= set(opt["visited"])
            used_zone |= set(opt["extra_zone"])
            covered_junction |= bool(opt["touch_junction"])
            covered_end |= bool(opt["hits_end"])

        if len(chosen) != target:
            continue
        if want_junction and not covered_junction:
            continue
        if want_end and not covered_end:
            continue
        return chosen
    return None


def _transform(grid: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return grid.copy()
    if code == 1:
        return np.flipud(grid).copy()
    if code == 2:
        return np.fliplr(grid).copy()
    if code == 3:
        return np.flipud(np.fliplr(grid)).copy()
    if code == 4:
        return grid.T.copy()
    if code == 5:
        return np.flipud(grid.T).copy()
    if code == 6:
        return np.fliplr(grid.T).copy()
    return np.flipud(np.fliplr(grid.T)).copy()


def _paint_expected(
    grid: np.ndarray,
    plan: list[dict[str, object]],
    colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for opt, color in zip(plan, colors, strict=False):
        for r, c in set(opt["visited"]):
            out[r, c] = color
    return out


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice([color for color in range(10) if color not in (CHANNEL_COLOR, RULER_COLOR)])
    palette = [color for color in range(10) if color not in (bg, CHANNEL_COLOR, RULER_COLOR)]

    network_type = random.choice(["line", "cross", "branched"])
    if network_type == "line":
        channel = _build_simple_network(h, w)
        min_seeds, max_seeds = 2, 3
    elif network_type == "cross":
        channel = _build_cross_network(h, w)
        min_seeds, max_seeds = 2, 4
    else:
        channel = _build_branched_network(h, w)
        min_seeds, max_seeds = 2, 4

    options = _seed_options(channel, h, w)
    if len({tuple(opt["seed"]) for opt in options}) < min_seeds:
        return None

    plan = _choose_plan(
        options,
        want_junction=network_type != "line" and random.random() < 0.75,
        want_end=random.random() < 0.75,
        min_seeds=min_seeds,
        max_seeds=max_seeds,
    )
    if plan is None:
        return None

    random.shuffle(palette)
    colors = palette[: len(plan)]
    grid = np.full((h, w), bg, dtype=int)
    for r, c in channel:
        grid[r, c] = CHANNEL_COLOR

    for opt, color in zip(plan, colors, strict=False):
        seed = tuple(opt["seed"])
        grid[seed] = color
        for r, c in list(opt["ruler"]):
            grid[r, c] = RULER_COLOR

    expected = _paint_expected(grid, plan, colors)
    transform_code = random.randint(0, 7)
    grid = _transform(grid, transform_code)
    expected = _transform(expected, transform_code)

    if _bg_color(grid) != bg:
        return None
    if int((grid == CHANNEL_COLOR).sum()) <= int((grid == RULER_COLOR).sum()):
        return None
    seed_colors = [int(grid[r, c]) for r, c in np.argwhere((grid != bg) & (grid != CHANNEL_COLOR) & (grid != RULER_COLOR))]
    seed_counts = Counter(seed_colors)
    if not seed_counts or any(count != 1 for count in seed_counts.values()):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, _expected = built
        return np.asarray(grid, dtype=int)
    raise RuntimeError("failed to generate a valid puzzle instance")
