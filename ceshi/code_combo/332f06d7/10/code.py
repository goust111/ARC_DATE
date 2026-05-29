# Combo variant of 332f06d7 (index 10)
# combo_spec: D4/3+D10/0+D17/2
# Name: 单像素粒子+多体顺序移动+多对多配对
#
# Design analysis
# ===============
# 1) Concrete fused task
#    - Keep the original green background and blue winding river, but reduce every mover/goal to 1 pixel.
#    - Use one simple 1-pixel-wide river path with no branches.
#    - Place 3-4 distinct particle colors on the first k cells of the path.
#    - For each particle color, place one same-colored target pixel later on the same path.
#    - Add one off-river legend pixel per particle color; the legend is drawn in move order.
#      The generator sorts the legend by numeric color value, so it realizes the requested
#      "move by color order" rule, but the solver reads the visible legend instead of relying
#      on hidden color ids. This keeps the rule ARC-observable.
#
# 2) Behavior trace
#    - Original task: one n x n square moves through the river to one target square, or stops at
#      the farthest reachable placement before a chokepoint narrower than n.
#    - Fused task: particles are 1 x 1 and targets are 1 x 1. The river itself stays passable,
#      and target markers are also passable. The only blocking objects are already parked particles.
#    - Example along a path:
#        path order: [srcA, srcC, srcB, ..., tgtB, ..., tgtA, ..., tgtC]
#        legend order: [A, B, C]
#        move A: reaches tgtA and stays there.
#        move B: tgtB lies before A's parked cell, so B still reaches tgtB.
#        move C: tgtC lies beyond B/A blockers, so C stops at the farthest open river cell before
#                the nearest parked particle. Its own same-colored target remains visible downstream.
#    - Difference from the original: the stopping rule is still "farthest reachable place", but now
#      reachability is controlled by earlier movers occupying the single-lane path, not by width<n.
#
# 3) Complexity calibration
#    - Use 3-4 colors, not 1-2, so the many:many pairing is visually undeniable.
#    - Keep the river branch-free so the only hard part is same-color pairing plus sequential blocking,
#      not ambiguous path choice.
#    - Force at least one blocked particle and at least one reached target in every generated grid,
#      so the examples do not collapse into "everyone reaches" or "everyone just queues."
#
# 4) Implicit assumption audit against the original solver
#    - Original assumption "non-bg objects are equal-sized solid squares" is invalid and removed.
#    - Original assumption "sort the two squares by color" is invalid because there are many colors,
#      each used for both source and target, plus a separate legend copy.
#    - Original assumption "one move only" is invalid; later particles must see the updated board.
#    - New solver assumptions:
#      a. The non-bg component containing river cells is a simple path.
#      b. Every active color appears exactly twice on that path: earlier occurrence = source,
#         later occurrence = target.
#      c. The first k path cells are exactly the k source particles, in some color order.
#      d. Each active color has exactly one off-path singleton legend pixel, and all legend pixels
#         lie on one row or one column.
#      e. Target markers are visual goals, not blockers; only current particle positions block motion.
#    - Generator guarantees all of these explicitly.
#
# 5) Placement legality and cleanliness
#    - The path is rendered from a coarse snake walk with spacing >= 3, so it does not self-intersect.
#    - Sources occupy the first k path cells and targets occupy later, well-spaced cells.
#    - Target spacing leaves empty river cells behind the earliest blocker, so blocked particles queue
#      on blue cells instead of overwriting other targets.
#    - Legend pixels are placed on isolated background cells away from the river, so they do not merge
#      into the traversable component.
#
# 6) Why the rule is visually inferable
#    - Same colors visibly pair each source with one downstream target.
#    - The legend makes move order visible from left-to-right or top-to-bottom.
#    - Output examples show some colors reaching their own targets while later colors stop upstream of
#      already parked particles, making the sequential dependency observable in 2-3 examples.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


_BG = 3
_RIVER = 1
_ACTIVE_COLORS = [0, 2, 4, 5, 6, 7, 8, 9]
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_TARGET_RANK_PATTERNS = {
    3: [(1, 2, 0), (2, 0, 1)],
    4: [(1, 3, 0, 2), (2, 0, 3, 1)],
}


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in _DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return seen


def _degree(component: set[tuple[int, int]], cell: tuple[int, int], h: int, w: int) -> int:
    r, c = cell
    return sum((rr, cc) in component for rr, cc in _neighbors4(r, c, h, w))


def _ordered_path(component: set[tuple[int, int]], start_endpoint: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    path = [start_endpoint]
    prev = None
    cur = start_endpoint
    while True:
        nxt = []
        for rr, cc in _neighbors4(cur[0], cur[1], h, w):
            pos = (rr, cc)
            if pos in component and pos != prev:
                nxt.append(pos)
        if not nxt:
            break
        assert len(nxt) == 1
        prev, cur = cur, nxt[0]
        path.append(cur)
    return path


def _compute_offsets(size: int, coarse_dim: int) -> list[int] | None:
    if coarse_dim == 1:
        return [1]
    first = 1
    last = size - 2
    if last < first:
        return None
    step = (last - first) / (coarse_dim - 1)
    offsets = [int(round(first + i * step)) for i in range(coarse_dim)]
    if any(offsets[i + 1] - offsets[i] < 3 for i in range(coarse_dim - 1)):
        return None
    return offsets


def _snake_walk(dim: int) -> list[tuple[int, int]]:
    walk = []
    for r in range(dim):
        cols = range(dim) if r % 2 == 0 else range(dim - 1, -1, -1)
        for c in cols:
            walk.append((r, c))
    return walk


def _transform_walk(walk: list[tuple[int, int]], dim: int) -> list[tuple[int, int]]:
    out = list(walk)
    if random.random() < 0.5:
        out = [(c, r) for r, c in out]
    if random.random() < 0.5:
        out = [(dim - 1 - r, c) for r, c in out]
    if random.random() < 0.5:
        out = [(r, dim - 1 - c) for r, c in out]
    return out


def _extend_to_edge(last_cell: tuple[int, int], coarse_end: tuple[int, int], dim: int, size: int) -> list[tuple[int, int]]:
    r, c = last_cell
    cr, cc = coarse_end
    candidates = []
    if cr == 0:
        candidates.append((-1, 0))
    if cr == dim - 1:
        candidates.append((1, 0))
    if cc == 0:
        candidates.append((0, -1))
    if cc == dim - 1:
        candidates.append((0, 1))
    assert candidates
    dr, dc = random.choice(candidates)
    ext = []
    rr, cc2 = r, c
    while True:
        nr = rr + dr
        nc = cc2 + dc
        if not (0 <= nr < size and 0 <= nc < size):
            break
        ext.append((nr, nc))
        rr, cc2 = nr, nc
    assert ext
    return ext


def _render_path(size: int, walk: list[tuple[int, int]], offsets: list[int]) -> list[tuple[int, int]]:
    centers = [(offsets[r], offsets[c]) for r, c in walk]
    cells = [centers[0]]
    for idx in range(len(centers) - 1):
        r1, c1 = centers[idx]
        r2, c2 = centers[idx + 1]
        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
        dc = 0 if c1 == c2 else (1 if c2 > c1 else -1)
        rr, cc = r1, c1
        while (rr, cc) != (r2, c2):
            rr += dr
            cc += dc
            cells.append((rr, cc))
    cells.extend(_extend_to_edge(cells[-1], walk[-1], len(offsets), size))
    assert len(cells) == len(set(cells))
    return cells


def _degrees_ok(path_cells: list[tuple[int, int]], size: int) -> bool:
    cell_set = set(path_cells)
    degs = [_degree(cell_set, cell, size, size) for cell in cell_set]
    return max(degs) <= 2 and sum(deg == 1 for deg in degs) == 2


def _legend_order(
    g: np.ndarray, path_component: set[tuple[int, int]], active_colors: list[int]
) -> tuple[list[int], dict[int, tuple[int, int]]]:
    legend_cells: list[tuple[int, int, int]] = []
    by_color: dict[int, tuple[int, int]] = {}
    for color in active_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
        on_path = [pos for pos in coords if pos in path_component]
        off_path = [pos for pos in coords if pos not in path_component]
        assert len(on_path) == 2
        assert len(off_path) == 1
        r, c = off_path[0]
        legend_cells.append((r, c, int(color)))
        by_color[int(color)] = (r, c)
    same_row = len({r for r, _c, _col in legend_cells}) == 1
    same_col = len({c for _r, c, _col in legend_cells}) == 1
    assert same_row or same_col
    if same_row:
        legend_cells.sort(key=lambda item: item[1])
    else:
        legend_cells.sort(key=lambda item: item[0])
    return [col for _r, _c, col in legend_cells], by_color


def _roles_for_path(
    g: np.ndarray, path: list[tuple[int, int]], river: int, active_colors: list[int]
) -> dict[int, tuple[int, int]] | None:
    k = len(active_colors)
    if len(path) < 2 * k + 2:
        return None
    prefix_colors = [int(g[path[i]]) for i in range(k)]
    if any(col == int(river) for col in prefix_colors):
        return None
    if sorted(prefix_colors) != sorted(active_colors):
        return None

    occ: dict[int, list[int]] = defaultdict(list)
    for idx, pos in enumerate(path):
        col = int(g[pos])
        if col != int(river):
            occ[col].append(idx)

    if set(occ) != set(active_colors):
        return None

    roles: dict[int, tuple[int, int]] = {}
    first_indices = []
    for color in active_colors:
        idxs = occ[int(color)]
        if len(idxs) != 2:
            return None
        src_idx, tgt_idx = idxs
        if src_idx >= k or tgt_idx <= src_idx:
            return None
        roles[int(color)] = (src_idx, tgt_idx)
        first_indices.append(src_idx)

    if sorted(first_indices) != list(range(k)):
        return None
    return roles


def _infer_setup(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    non_bg_counts = Counter(int(v) for v in g.flatten() if int(v) != int(bg))
    assert non_bg_counts
    river = int(non_bg_counts.most_common(1)[0][0])

    river_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == int(river))]
    assert river_cells
    path_component = _component(g != int(bg), river_cells[0])

    degrees = {pos: _degree(path_component, pos, h, w) for pos in path_component}
    assert all(deg <= 2 for deg in degrees.values())
    endpoints = [pos for pos, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2

    active_colors = sorted({int(g[pos]) for pos in path_component if int(g[pos]) not in (int(bg), int(river))})
    assert 3 <= len(active_colors) <= 4

    order, legend = _legend_order(g, path_component, active_colors)
    path = _ordered_path(path_component, endpoints[0], h, w)
    roles = _roles_for_path(g, path, river, active_colors)
    if roles is None:
        path = list(reversed(path))
        roles = _roles_for_path(g, path, river, active_colors)
    assert roles is not None

    return {
        "grid": g,
        "bg": bg,
        "river": river,
        "path": path,
        "active_colors": active_colors,
        "order": order,
        "roles": roles,
        "legend": legend,
    }


def _simulate_finals(setup: dict, sequential: bool) -> dict[int, int]:
    path_len = len(setup["path"])
    roles = setup["roles"]
    order = setup["order"]
    finals = {color: int(roles[color][1]) for color in roles}
    if not sequential:
        return finals

    positions = {color: int(roles[color][0]) for color in roles}
    occupied = set(positions.values())
    for color in order:
        cur = positions[color]
        occupied.remove(cur)
        blocker = min((idx for idx in occupied if idx > cur), default=path_len)
        max_reach = blocker - 1
        tgt = int(roles[color][1])
        final = tgt if cur <= tgt <= max_reach else max_reach
        positions[color] = final
        finals[color] = final
        occupied.add(final)
    return finals


def _build_output(setup: dict, finals: dict[int, int]) -> np.ndarray:
    g = setup["grid"]
    river = int(setup["river"])
    path = setup["path"]
    roles = setup["roles"]

    out = g.copy()
    for color in setup["active_colors"]:
        src_idx, tgt_idx = roles[color]
        sr, sc = path[src_idx]
        tr, tc = path[tgt_idx]
        out[sr, sc] = river
        out[tr, tc] = river

    for color in setup["active_colors"]:
        tgt_idx = roles[color][1]
        tr, tc = path[tgt_idx]
        out[tr, tc] = int(color)
        fr, fc = path[finals[color]]
        out[fr, fc] = int(color)
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    setup = _infer_setup(grid)
    seq = _simulate_finals(setup, sequential=True)
    ind = _simulate_finals(setup, sequential=False)
    roles = setup["roles"]
    blocked = sum(int(seq[color] != roles[color][1]) for color in setup["order"])
    moved = sum(int(seq[color] != roles[color][0]) for color in setup["order"])
    diff = sum(int(seq[color] != ind[color]) for color in setup["order"])
    later_reached = sum(
        int(idx > 0 and seq[color] == roles[color][1]) for idx, color in enumerate(setup["order"])
    )
    return {
        "particle_count": len(setup["order"]),
        "blocked_count": blocked,
        "moved_count": moved,
        "sequential_diff_count": diff,
        "later_reached_count": later_reached,
    }


def main(grid: np.ndarray) -> np.ndarray:
    setup = _infer_setup(np.array(grid, dtype=int))
    finals = _simulate_finals(setup, sequential=True)
    return _build_output(setup, finals)


def _slot_is_isolated_bg(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != _BG:
        return False
    h, w = grid.shape
    for rr, cc in _neighbors4(r, c, h, w):
        if int(grid[rr, cc]) != _BG:
            return False
    return True


def _place_legend(grid: np.ndarray, colors: list[int]) -> bool:
    h, w = grid.shape
    need = 2 * len(colors) - 1

    for r in range(h):
        for start in range(0, w - need + 1):
            slots = [(r, start + 2 * i) for i in range(len(colors))]
            if all(_slot_is_isolated_bg(grid, rr, cc) for rr, cc in slots):
                for (rr, cc), color in zip(slots, colors):
                    grid[rr, cc] = int(color)
                return True

    for c in range(w):
        for start in range(0, h - need + 1):
            slots = [(start + 2 * i, c) for i in range(len(colors))]
            if all(_slot_is_isolated_bg(grid, rr, cc) for rr, cc in slots):
                for (rr, cc), color in zip(slots, colors):
                    grid[rr, cc] = int(color)
                return True
    return False


def _target_positions(path_len: int, k: int) -> list[int] | None:
    gap = k + 1
    min_first = k + 4
    max_first = path_len - 2 - gap * (k - 1)
    if max_first < min_first:
        return None
    first = random.randint(min_first, max_first)
    return [first + gap * i for i in range(k)]


def _pick_particle_count(path_len: int) -> int | None:
    choices = [k for k in (4, 3) if _target_positions(path_len, k) is not None]
    if not choices:
        return None
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(15, 20)
    coarse_dim = random.randint(4, min(6, max(4, (size - 2) // 3 + 1)))
    offsets = _compute_offsets(size, coarse_dim)
    if offsets is None:
        return None

    walk = _transform_walk(_snake_walk(coarse_dim), coarse_dim)
    path = _render_path(size, walk, offsets)
    if len(path) < 18 or not _degrees_ok(path, size):
        return None

    k = _pick_particle_count(len(path))
    if k is None:
        return None
    target_slots = _target_positions(len(path), k)
    if target_slots is None:
        return None

    colors = random.sample(_ACTIVE_COLORS, k)
    move_order = sorted(colors)
    source_colors = list(colors)
    random.shuffle(source_colors)

    target_rank_pattern = random.choice(_TARGET_RANK_PATTERNS[k])
    target_by_color = {
        color: int(target_slots[target_rank_pattern[idx]]) for idx, color in enumerate(move_order)
    }

    grid = np.full((size, size), _BG, dtype=int)
    for r, c in path:
        grid[r, c] = _RIVER

    for idx, color in enumerate(source_colors):
        r, c = path[idx]
        grid[r, c] = int(color)

    for color in colors:
        r, c = path[target_by_color[color]]
        grid[r, c] = int(color)

    if not _place_legend(grid, move_order):
        return None

    try:
        setup = _infer_setup(grid)
        out = main(grid)
        metrics = _behavior_metrics(grid)
    except Exception:
        return None

    if setup["order"] != move_order:
        return None
    if metrics["blocked_count"] < 1:
        return None
    if metrics["later_reached_count"] < 1:
        return None
    if metrics["moved_count"] < k:
        return None
    if metrics["sequential_diff_count"] < 1:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
