from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
PIPE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 300
WALK_ATTEMPTS = 240
SPUR_ATTEMPTS = 20


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_mix_table(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    non_bg = g != bg
    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []

    for comp in _component_coords(non_bg):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if c1 != w - 1:
            continue
        if width != 3:
            continue
        if len(comp) != height * width:
            continue
        if r0 > 2:
            continue
        candidates.append((r0, -height, comp))

    if candidates:
        _, _, comp = min(candidates)
        r0, c0, r1, c1 = _bbox(comp)
        rows = list(range(r0, r1 + 1))
        cols = list(range(c0, c1 + 1))
        table_mask = np.zeros_like(g, dtype=bool)
        table_mask[np.ix_(rows, cols)] = True
    else:
        start = None
        rows: list[int] = []
        c0 = w - 3
        for r in range(h):
            if np.all(g[r, c0:w] != bg):
                start = r
                break
        assert start is not None, "failed to detect the mix table"
        r = start
        while r < h and np.all(g[r, c0:w] != bg):
            rows.append(r)
            r += 1
        table_mask = np.zeros_like(g, dtype=bool)
        table_mask[np.ix_(rows, list(range(c0, w)))] = True

    table_rows = sorted(np.unique(np.argwhere(table_mask)[:, 0]).tolist())
    mix_map: dict[tuple[int, int], int] = {}
    for r in table_rows:
        a, b, mix = [int(x) for x in g[r, np.where(table_mask[r])[0]]]
        key = tuple(sorted((a, b)))
        mix_map[key] = mix

    assert mix_map, "mix table must contain at least one row"
    return table_mask, mix_map


def _bfs_dist(traversable: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    h, w = traversable.shape
    inf = h * w + 1
    dist = np.full((h, w), inf, dtype=int)
    q = deque()
    for r, c in sources:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c, h, w):
            if not bool(traversable[nr, nc]) or nd >= int(dist[nr, nc]):
                continue
            dist[nr, nc] = nd
            q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood each grey pipe network from its colored seeds and mix ties using the top-right lookup table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 6

    bg = _most_common_color(g)
    table_mask, mix_map = _find_mix_table(g, bg)

    outside_table = ~table_mask
    counts = Counter(int(x) for x in g[outside_table] if int(x) != bg)
    assert counts, "expected a pipe network outside the mix table"
    pipe_color = max(counts.items(), key=lambda kv: kv[1])[0]

    traversable = outside_table & (g != bg)
    seed_mask = traversable & (g != pipe_color)
    seed_positions = np.argwhere(seed_mask)
    assert len(seed_positions) >= 2, "expected at least two seed pixels"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in seed_positions:
        by_color.setdefault(int(g[r, c]), []).append((int(r), int(c)))

    dists = {color: _bfs_dist(traversable, coords) for color, coords in by_color.items()}
    inf = g.shape[0] * g.shape[1] + 1

    out = g.copy()
    for r, c in np.argwhere(traversable):
        r = int(r)
        c = int(c)
        if bool(seed_mask[r, c]):
            continue

        nearest_dist = min(int(dist[r, c]) for dist in dists.values())
        assert nearest_dist < inf, "every pipe component must contain a seed"
        nearest_colors = sorted(color for color, dist in dists.items() if int(dist[r, c]) == nearest_dist)
        nearest_colors = sorted(set(nearest_colors))

        if len(nearest_colors) == 1:
            out[r, c] = nearest_colors[0]
        else:
            assert len(nearest_colors) == 2, "generator guarantees no triple-color meetings"
            key = tuple(nearest_colors)
            assert key in mix_map, f"missing mix entry for colors {key}"
            out[r, c] = mix_map[key]

    return out


def _expand_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        assert r0 == r1 or c0 == c1
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        cr, cc = r0, c0
        while (cr, cc) != (r1, c1):
            cr += dr
            cc += dc
            cells.append((cr, cc))
    return cells


def _valid_new_cell(
    cell: tuple[int, int],
    h: int,
    w: int,
    reserved: np.ndarray,
    occupied: set[tuple[int, int]],
    local: set[tuple[int, int]],
    allowed_neighbors: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if bool(reserved[r, c]):
        return False
    if cell in occupied or cell in local:
        return False
    for nr, nc in _neighbors4(r, c, h, w):
        neighbor = (nr, nc)
        if neighbor in occupied:
            return False
        if neighbor in local and neighbor not in allowed_neighbors:
            return False
    return True


def _grow_walk(
    h: int,
    w: int,
    reserved: np.ndarray,
    occupied: set[tuple[int, int]],
    steps: int,
) -> list[tuple[int, int]] | None:
    for _ in range(WALK_ATTEMPTS):
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        if not _valid_new_cell(start, h, w, reserved, occupied, set(), set()):
            continue

        path = [start]
        local = {start}
        prev_dir: tuple[int, int] | None = None

        for _step in range(steps):
            options: list[tuple[tuple[int, int], tuple[int, int], int]] = []
            r, c = path[-1]
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if not _valid_new_cell(nxt, h, w, reserved, occupied, local, {path[-1]}):
                    continue
                weight = 3 if prev_dir == (dr, dc) else 1
                options.append((nxt, (dr, dc), weight))
            if not options:
                break
            cells = [cell for cell, _dir, _weight in options]
            weights = [weight for _cell, _dir, weight in options]
            idx = random.choices(range(len(cells)), weights=weights, k=1)[0]
            nxt, prev_dir = cells[idx], options[idx][1]
            path.append(nxt)
            local.add(nxt)

        if len(path) == steps + 1:
            return path
    return None


def _add_spurs(
    base_path: list[tuple[int, int]],
    h: int,
    w: int,
    reserved: np.ndarray,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    full = list(base_path)
    local = set(full)
    mid = len(base_path) // 2
    target_spurs = random.randint(0, 2)
    root_choices = [idx for idx in range(1, len(base_path) - 1) if idx != mid]
    random.shuffle(root_choices)

    placed = 0
    for root_idx in root_choices:
        if placed >= target_spurs:
            break
        root = base_path[root_idx]
        for _ in range(SPUR_ATTEMPTS):
            dr, dc = random.choice(DIR4)
            length = random.randint(1, 3)
            spur: list[tuple[int, int]] = []
            prev = root
            ok = True
            for step in range(1, length + 1):
                nxt = (root[0] + dr * step, root[1] + dc * step)
                allowed = {prev}
                if not _valid_new_cell(nxt, h, w, reserved, occupied, local, allowed):
                    ok = False
                    break
                spur.append(nxt)
                prev = nxt
            if ok:
                full.extend(spur)
                local.update(spur)
                placed += 1
                break
    return full


def _table_height_and_pairs() -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int, int]]]:
    n_seed_colors = random.choice([2, 3, 3, 4])
    seed_colors = random.sample(PALETTE, n_seed_colors)
    all_pairs = list(combinations(seed_colors, 2))
    max_pairs = min(4, len(all_pairs))
    min_pairs = 1 if max_pairs == 1 else 2
    pair_count = random.randint(min_pairs, max_pairs)
    pairs = random.sample(all_pairs, pair_count)

    remaining = [c for c in PALETTE if c not in seed_colors]
    mix_colors = random.sample(remaining, pair_count)
    rows = [(a, b, mix) for (a, b), mix in zip(pairs, mix_colors)]
    random.shuffle(rows)
    return seed_colors, mix_colors, rows


def _validate_generated(grid: np.ndarray, rows: list[tuple[int, int, int]]) -> bool:
    out = main(grid)
    if np.array_equal(grid, out):
        return False
    if out.shape != grid.shape:
        return False

    table_h = len(rows)
    table_mask = np.zeros_like(grid, dtype=bool)
    table_mask[:table_h, grid.shape[1] - 3 :] = True

    mix_colors = {mix for _a, _b, mix in rows}
    for mix in mix_colors:
        if not np.any((out == mix) & (~table_mask)):
            return False

    pipe_outside = np.sum((grid == PIPE) & (~table_mask))
    if pipe_outside == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(18, 27)
    _seed_colors, _mix_colors, rows = _table_height_and_pairs()
    table_h = len(rows)

    grid = np.full((h, w), BG, dtype=int)
    for r, (a, b, mix) in enumerate(rows):
        grid[r, w - 3] = a
        grid[r, w - 2] = b
        grid[r, w - 1] = mix

    reserved = np.zeros((h, w), dtype=bool)
    reserved[: min(h, table_h + 2), max(0, w - 5) :] = True

    pair_assignments = [(a, b) for a, b, _mix in rows]
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        a, b, _mix = random.choice(rows)
        pair_assignments.append((a, b))
    random.shuffle(pair_assignments)

    occupied: set[tuple[int, int]] = set()
    for a, b in pair_assignments:
        steps = random.choice([4, 6, 8, 10, 12])
        path = _grow_walk(h, w, reserved, occupied, steps)
        if path is None:
            return None
        cells = _add_spurs(path, h, w, reserved, occupied)
        for r, c in cells:
            grid[r, c] = PIPE
        grid[path[0]] = a
        grid[path[-1]] = b
        occupied.update(cells)

    try:
        if not _validate_generated(grid, rows):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
