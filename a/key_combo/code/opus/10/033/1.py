from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        count += 1
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        while q:
            cr, cc = q.popleft()
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _shortest_path(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    h, w = map(int, grid.shape)
    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=np.int8)

    q = deque([start])
    dist[start] = 0
    ways[start] = 1

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc) or int(grid[nr, nc]) == BLUE:
                continue
            nd = int(dist[r, c]) + 1
            if int(dist[nr, nc]) == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = ways[r, c]
                q.append((nr, nc))
            elif int(dist[nr, nc]) == nd:
                ways[nr, nc] = min(2, int(ways[nr, nc]) + int(ways[r, c]))

    assert int(dist[goal]) >= 0, "expected the red endpoints to be connected"
    assert int(ways[goal]) == 1, "expected a unique shortest path between the red endpoints"

    path = [goal]
    cur = goal
    while cur != start:
        r, c = cur
        preds = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if _in_bounds((h, w), nr, nc) and int(dist[nr, nc]) == int(dist[r, c]) - 1
        ]
        assert len(preds) == 1, "expected the shortest path reconstruction to be unique"
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path, dist, ways


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            turns += 1
    return turns


def _ray_hit_masks(
    grid: np.ndarray, path: list[tuple[int, int]]
) -> tuple[list[np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    path_set = set(path)
    markers = path[1:-1]
    masks: list[np.ndarray] = []
    counts = np.zeros(g.shape, dtype=int)

    for r, c in markers:
        hit = np.zeros(g.shape, dtype=bool)
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            while _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != BLUE:
                if int(g[nr, nc]) == BLACK and (nr, nc) not in path_set:
                    hit[nr, nc] = True
                nr += dr
                nc += dc
        masks.append(hit)
        counts += hit.astype(int)
    return masks, counts


def _render_output(
    grid: np.ndarray, path: list[tuple[int, int]]
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    hit_masks, hit_counts = _ray_hit_masks(g, path)
    out = g.copy()
    background = g == BLACK
    out[(hit_counts == 1) & background] = YELLOW
    out[(hit_counts >= 2) & background] = ORANGE
    for r, c in path[1:-1]:
        out[r, c] = GREEN
    return out, hit_masks, hit_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique maze path between two red endpoints, then expand its interior pixels into wall-blocked row and column rays.

    Input structure
    - The grid uses black background, blue wall pixels, and exactly two red endpoint pixels.
    - Blue pixels are impassable maze walls; every non-blue cell is corridor space.
    - The two red pixels are joined by one unique shortest 4-neighbor corridor path through the non-blue cells.

    Rule
    - Find the unique shortest 4-neighbor path between the two red endpoints while treating blue as blocked.
    - Recolor the interior pixels of that path green.
    - From every green path pixel, extend rays up, down, left, and right through non-blue cells until a blue wall or the grid edge stops the ray.
    - Other path pixels and the red endpoints do not block the ray, but only background cells reached by rays are recolored.
    - A background cell reached by exactly one path marker becomes yellow; a background cell reached by two or more different path markers becomes orange.
    - Blue walls and red endpoints keep priority over every ray color.

    Key insight
    - The shortest-path pixels are reused as hidden control markers, so the maze walls simultaneously determine the path geometry and the limited visibility of every row and column ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED}, "unexpected input colors"

    reds = np.argwhere(g == RED)
    assert reds.shape == (2, 2), "expected exactly two red endpoints"
    start = tuple(map(int, reds[0]))
    goal = tuple(map(int, reds[1]))

    path, _dist, _ways = _shortest_path(g, start, goal)
    out, _hit_masks, _hit_counts = _render_output(g, path)
    return out


def _interior(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    h, w = map(int, shape)
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _open_neighbor_count(open_set: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    return sum((nr, nc) in open_set for nr, nc in _neighbors4(*cell))


def _add_segment(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    path: list[tuple[int, int]],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    current = path[-1]
    new_cells: list[tuple[int, int]] = []
    combined = set(occupied)
    for _ in range(length):
        nr, nc = current[0] + dr, current[1] + dc
        nxt = (nr, nc)
        if not _interior(shape, nxt) or nxt in combined:
            return None
        if _open_neighbor_count(combined, nxt) != 1:
            return None
        new_cells.append(nxt)
        combined.add(nxt)
        current = nxt
    return new_cells


def _build_polyline_path(
    rng: np.random.Generator, shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    n_segments = int(rng.integers(3, 7))
    min_lengths = [3] + [2] * (n_segments - 2) + [3]
    max_lengths = [6] + [5] * (n_segments - 2) + [6]

    start = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))

    def recurse(
        path: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
        prev_dir: tuple[int, int] | None,
        seg_idx: int,
    ) -> list[tuple[int, int]] | None:
        if seg_idx == n_segments:
            return path

        if prev_dir is None:
            directions = DIRS4.copy()
        else:
            if prev_dir[0] != 0:
                directions = [(0, -1), (0, 1)]
            else:
                directions = [(-1, 0), (1, 0)]
        rng.shuffle(directions)

        for direction in directions:
            lengths = list(range(min_lengths[seg_idx], max_lengths[seg_idx] + 1))
            rng.shuffle(lengths)
            for length in lengths:
                new_cells = _add_segment(shape, occupied, path, direction, int(length))
                if new_cells is None:
                    continue
                result = recurse(
                    path + new_cells,
                    occupied | set(new_cells),
                    direction,
                    seg_idx + 1,
                )
                if result is not None:
                    return result
        return None

    return recurse([start], {start}, None, 0)


def _terminal_direction(path: list[tuple[int, int]], which: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if which == "start":
        return path[0], (path[0][0] - path[1][0], path[0][1] - path[1][1])
    return path[-1], (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])


def _available_tail_cells(
    shape: tuple[int, int], open_set: set[tuple[int, int]], path: list[tuple[int, int]], which: str
) -> list[tuple[int, int]]:
    end, (dr, dc) = _terminal_direction(path, which)
    cells: list[tuple[int, int]] = []
    cur = end
    while True:
        cur = (cur[0] + dr, cur[1] + dc)
        if not _interior(shape, cur) or cur in open_set:
            break
        cells.append(cur)
    return cells


def _straight_runs(path: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    runs: list[list[tuple[int, int]]] = []
    run = [path[0], path[1]]
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(1, len(path) - 1):
        cur_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if cur_dir == prev_dir:
            run.append(path[i + 1])
        else:
            runs.append(run)
            run = [path[i], path[i + 1]]
            prev_dir = cur_dir
    runs.append(run)
    return runs


def _room_is_valid(
    shape: tuple[int, int],
    open_set: set[tuple[int, int]],
    room: set[tuple[int, int]],
    allowed_contacts: set[tuple[int, int]],
) -> bool:
    if not room:
        return False
    for cell in room:
        if not _interior(shape, cell) or cell in open_set:
            return False
    for cell in room:
        for nr, nc in _neighbors4(*cell):
            if (nr, nc) in open_set and (nr, nc) not in allowed_contacts:
                return False
    return True


def _try_attach_side_room(
    rng: np.random.Generator,
    shape: tuple[int, int],
    open_set: set[tuple[int, int]],
    runs: list[list[tuple[int, int]]],
) -> bool:
    run_order = list(range(len(runs)))
    rng.shuffle(run_order)
    for run_idx in run_order:
        run = runs[run_idx]
        if len(run) < 3:
            continue
        horizontal = all(r == run[0][0] for r, _c in run)
        side_choices = [-1, 1]
        rng.shuffle(side_choices)
        span_lengths = list(range(min(8, len(run)), 2, -1))
        for span_len in span_lengths:
            starts = list(range(0, len(run) - span_len + 1))
            rng.shuffle(starts)
            for start in starts:
                sub = run[start : start + span_len]
                allowed_contacts = set(sub)
                depths = list(range(6, 1, -1))
                for side in side_choices:
                    for depth in depths:
                        if horizontal:
                            row = sub[0][0]
                            cols = [c for _r, c in sub]
                            c0, c1 = min(cols), max(cols)
                            if side == -1:
                                room = {(rr, cc) for rr in range(row - depth, row) for cc in range(c0, c1 + 1)}
                            else:
                                room = {
                                    (rr, cc) for rr in range(row + 1, row + depth + 1) for cc in range(c0, c1 + 1)
                                }
                        else:
                            col = sub[0][1]
                            rows = [r for r, _c in sub]
                            r0, r1 = min(rows), max(rows)
                            if side == -1:
                                room = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(col - depth, col)}
                            else:
                                room = {
                                    (rr, cc) for rr in range(r0, r1 + 1) for cc in range(col + 1, col + depth + 1)
                                }
                        if _room_is_valid(shape, open_set, room, allowed_contacts):
                            open_set.update(room)
                            return True
    return False


def _grow_branches(
    rng: np.random.Generator,
    shape: tuple[int, int],
    open_set: set[tuple[int, int]],
    main_path_set: set[tuple[int, int]],
    target_open: int,
) -> bool:
    h, w = map(int, shape)
    while len(open_set) < target_open:
        frontier: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                cell = (r, c)
                if cell in open_set:
                    continue
                parents = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in open_set]
                if len(parents) != 1:
                    continue
                parent = parents[0]
                parent_deg = _open_neighbor_count(open_set, parent)
                weight = 1.0
                if parent in main_path_set:
                    weight += 1.5
                if parent_deg == 1:
                    weight += 2.0
                frontier.append((cell, parent, weight))

        if not frontier:
            return False

        weights = np.array([item[2] for item in frontier], dtype=float)
        weights /= weights.sum()
        idx = int(rng.choice(len(frontier), p=weights))
        open_set.add(frontier[idx][0])
    return True


def _try_attach_room(
    rng: np.random.Generator,
    shape: tuple[int, int],
    open_set: set[tuple[int, int]],
    main_path_set: set[tuple[int, int]],
) -> bool:
    h, w = map(int, shape)
    frontier: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            door = (r, c)
            if door in open_set:
                continue
            parents = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in open_set]
            if len(parents) != 1:
                continue
            parent = parents[0]
            weight = 1.0
            if parent in main_path_set:
                weight += 2.0
            if _open_neighbor_count(open_set, parent) == 1:
                weight += 1.0
            frontier.append((door, parent, weight))

    if not frontier:
        return False

    order = list(range(len(frontier)))
    weights = np.array([frontier[i][2] for i in order], dtype=float)
    weights /= weights.sum()
    sampled = list(rng.choice(order, size=len(order), replace=False, p=weights))

    for idx in sampled:
        door, parent, _weight = frontier[idx]
        dr = door[0] - parent[0]
        dc = door[1] - parent[1]
        depths = list(range(3, 11))
        widths = list(range(3, 11))
        depths.sort(reverse=True)
        widths.sort(reverse=True)

        for depth in depths:
            for width in widths:
                rects: list[set[tuple[int, int]]] = []
                if dr == 1:
                    for left in range(width):
                        rects.append(
                            {
                                (rr, cc)
                                for rr in range(door[0], door[0] + depth)
                                for cc in range(door[1] - left, door[1] - left + width)
                            }
                        )
                elif dr == -1:
                    for left in range(width):
                        rects.append(
                            {
                                (rr, cc)
                                for rr in range(door[0] - depth + 1, door[0] + 1)
                                for cc in range(door[1] - left, door[1] - left + width)
                            }
                        )
                elif dc == 1:
                    for up in range(width):
                        rects.append(
                            {
                                (rr, cc)
                                for rr in range(door[0] - up, door[0] - up + width)
                                for cc in range(door[1], door[1] + depth)
                            }
                        )
                else:
                    for up in range(width):
                        rects.append(
                            {
                                (rr, cc)
                                for rr in range(door[0] - up, door[0] - up + width)
                                for cc in range(door[1] - depth + 1, door[1] + 1)
                            }
                        )
                rng.shuffle(rects)

                for room in rects:
                    if not room or any(not _interior(shape, cell) or cell in open_set for cell in room):
                        continue
                    ok = True
                    for cell in room:
                        for nr, nc in _neighbors4(*cell):
                            if (nr, nc) in open_set and (nr, nc) not in room and cell != door:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    open_set.update(room)
                    return True
    return False


def _fast_attach_room(
    rng: np.random.Generator,
    shape: tuple[int, int],
    open_set: set[tuple[int, int]],
    main_path_set: set[tuple[int, int]],
) -> bool:
    h, w = map(int, shape)
    frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            door = (r, c)
            if door in open_set:
                continue
            parents = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in open_set]
            if len(parents) == 1:
                frontier.append((door, parents[0]))
    if not frontier:
        return False

    order = list(range(len(frontier)))
    rng.shuffle(order)
    for idx in order[:80]:
        door, parent = frontier[idx]
        dr = door[0] - parent[0]
        dc = door[1] - parent[1]
        depth = int(rng.integers(3, 7))
        width = int(rng.integers(3, 7))
        offsets = list(range(width))
        rng.shuffle(offsets)
        for offset in offsets:
            if dr == 1:
                room = {(rr, cc) for rr in range(door[0], door[0] + depth) for cc in range(door[1] - offset, door[1] - offset + width)}
            elif dr == -1:
                room = {(rr, cc) for rr in range(door[0] - depth + 1, door[0] + 1) for cc in range(door[1] - offset, door[1] - offset + width)}
            elif dc == 1:
                room = {(rr, cc) for rr in range(door[0] - offset, door[0] - offset + width) for cc in range(door[1], door[1] + depth)}
            else:
                room = {(rr, cc) for rr in range(door[0] - offset, door[0] - offset + width) for cc in range(door[1] - depth + 1, door[1] + 1)}
            if not room or any(not _interior(shape, cell) or cell in open_set for cell in room):
                continue
            ok = True
            for cell in room:
                for nbr in _neighbors4(*cell):
                    if nbr in open_set and not (cell == door and nbr == parent):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            open_set.update(room)
            return True
    return False


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    cells: list[tuple[int, int]] = []
    if ar == br:
        step = 1 if bc >= ac else -1
        for c in range(ac, bc + step, step):
            cells.append((ar, c))
    else:
        step = 1 if br >= ar else -1
        for r in range(ar, br + step, step):
            cells.append((r, ac))
    return cells


def _build_serpentine_path(rng: np.random.Generator, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    top = int(rng.integers(2, 4))
    mid = h // 2
    bottom = h - int(rng.integers(3, 5))
    rows = sorted({top, mid, bottom})
    if len(rows) < 3 or min(b - a for a, b in zip(rows, rows[1:])) < 3:
        rows = [2, h // 2, h - 3]
    left = int(rng.integers(2, 4))
    right = w - int(rng.integers(3, 5))
    if right - left < 7:
        left, right = 2, w - 3

    points = [(rows[0], left), (rows[0], right), (rows[1], right), (rows[1], left), (rows[2], left), (rows[2], right)]
    if rng.random() < 0.5:
        points = [(r, w - 1 - c) for r, c in points]
    if rng.random() < 0.5:
        points = [(h - 1 - r, c) for r, c in points]

    path: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        segment = _line_cells(a, b)
        if path:
            segment = segment[1:]
        path.extend(segment)
    return path


def _build_grid(
    shape: tuple[int, int], open_set: set[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int]
) -> np.ndarray:
    grid = np.full(shape, BLUE, dtype=int)
    for r, c in open_set:
        grid[r, c] = BLACK
    grid[start] = RED
    grid[goal] = RED
    return grid


def _has_skip_over_case(path: list[tuple[int, int]], hit_masks: list[np.ndarray]) -> bool:
    path_set = set(path)
    for marker, hit in zip(path[1:-1], hit_masks):
        mr, mc = marker
        for rr_raw, cc_raw in np.argwhere(hit):
            rr, cc = int(rr_raw), int(cc_raw)
            if rr == mr:
                step = 1 if cc > mc else -1
                between = [(mr, c) for c in range(mc + step, cc, step)]
            elif cc == mc:
                step = 1 if rr > mr else -1
                between = [(r, mc) for r in range(mr + step, rr, step)]
            else:
                continue
            if any(cell in path_set for cell in between):
                return True
    return False


def _validate_semantics(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) != {BLACK, BLUE, RED}:
        return False

    blue_density = float(np.mean(grid == BLUE))
    if not (0.30 <= blue_density <= 0.49):
        return False

    reds = np.argwhere(grid == RED)
    if reds.shape != (2, 2):
        return False
    start = tuple(map(int, reds[0]))
    goal = tuple(map(int, reds[1]))

    try:
        path, _dist, ways = _shortest_path(grid, start, goal)
    except AssertionError:
        return False

    if int(ways[goal]) != 1:
        return False
    if len(path) - 1 < 5 or _turn_count(path) < 2:
        return False

    out, hit_masks, hit_counts = _render_output(grid, path)
    if np.count_nonzero(out == GREEN) != len(path) - 2:
        return False
    if np.count_nonzero(out == ORANGE) < 1:
        return False
    if np.count_nonzero(out == YELLOW) < 1:
        return False
    if np.count_nonzero(out == BLACK) < 5:
        return False
    if not _has_skip_over_case(path, hit_masks):
        return False

    # Require at least one single-covered leaf and one multi-covered tail/corridor cell.
    if np.count_nonzero(hit_counts == 1) < 2 or np.count_nonzero(hit_counts >= 2) < 1:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(15, 18)),
        int(rng.integers(15, 18)),
    )

    path = _build_serpentine_path(rng, shape)
    if path is None or _turn_count(path) < 2 or len(path) - 1 < 5:
        return None

    open_set = set(path)
    tail_candidates = {
        which: _available_tail_cells(shape, open_set, path, which)
        for which in ("start", "goal")
    }
    usable = [which for which, cells in tail_candidates.items() if len(cells) >= 2]
    if not usable:
        return None

    rng.shuffle(usable)
    n_tails = 1 if len(usable) == 1 or float(rng.random()) < 0.65 else 2
    for which in usable[:n_tails]:
        cells = tail_candidates[which]
        length = int(rng.integers(2, min(5, len(cells)) + 1))
        open_set.update(cells[:length])

    main_path_set = set(path)
    final_target = int(round(np.prod(shape) * float(rng.uniform(0.600, 0.625))))
    runs = _straight_runs(path)
    side_tries = 0
    while len(open_set) < final_target and side_tries < 18:
        if not _try_attach_side_room(rng, shape, open_set, runs):
            break
        side_tries += 1
    if len(open_set) < final_target:
        _grow_branches(rng, shape, open_set, main_path_set, final_target)

    start = path[0]
    goal = path[-1]
    grid = _build_grid(shape, open_set, start, goal)
    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(160):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-ray input")
