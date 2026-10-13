from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid, dtype=int).ravel())
    return int(counts.most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _add(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Coord, b: Coord) -> Coord:
    return a[0] - b[0], a[1] - b[1]


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _turn_left(d: Coord) -> Coord:
    return -d[1], d[0]


def _turn_right(d: Coord) -> Coord:
    return d[1], -d[0]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nr, nc in _neighbors4(cur):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _touches_border(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _trace_component(comp: list[Coord], shape: tuple[int, int]) -> list[Coord]:
    cells = set(comp)
    deg = {cell: sum(nb in cells for nb in _neighbors4(cell)) for cell in cells}
    assert all(v in (1, 2) for v in deg.values()), "path component must be a simple chain"
    ends = [cell for cell, v in deg.items() if v == 1]
    assert len(ends) == 2, "path component must have exactly two endpoints"
    border_ends = [cell for cell in ends if _touches_border(shape, cell)]
    inner_ends = [cell for cell in ends if not _touches_border(shape, cell)]
    assert len(border_ends) == 1 and len(inner_ends) == 1, "each path segment must run from border to gap"

    ordered = [border_ends[0]]
    prev = None
    cur = border_ends[0]
    while True:
        nxts = [nb for nb in _neighbors4(cur) if nb in cells and nb != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path component must not branch"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert ordered[-1] == inner_ends[0], "trace must terminate at the gap endpoint"
    return ordered


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 2, f"expected exactly two foreground colors, got {colors}"

    candidates = []
    for color in colors:
        comps = _components(g == color)
        if len(comps) == 2 and all(any(_touches_border(g.shape, cell) for cell in comp) for comp in comps):
            candidates.append(color)
    assert len(candidates) == 1, f"expected a unique path color, got {candidates}"
    path_color = int(candidates[0])
    blocker_color = int(colors[0] if colors[1] == path_color else colors[1])
    return bg, path_color, blocker_color


def _turn_blockers_for_segment(seg: list[Coord]) -> list[Coord]:
    blockers: list[Coord] = []
    for i in range(1, len(seg) - 1):
        incoming = _sub(seg[i], seg[i - 1])
        outgoing = _sub(seg[i + 1], seg[i])
        if incoming == outgoing:
            continue
        assert abs(incoming[0]) + abs(incoming[1]) == 1
        assert abs(outgoing[0]) + abs(outgoing[1]) == 1
        assert incoming[0] == 0 or outgoing[0] == 0, "turn must be orthogonal"
        blockers.append(_add(seg[i], incoming))
    return blockers


def _full_path_valid(full_path: list[Coord], grid: np.ndarray, blocker_color: int) -> bool:
    if len(full_path) != len(set(full_path)):
        return False
    if not _touches_border(grid.shape, full_path[0]) or not _touches_border(grid.shape, full_path[-1]):
        return False

    for i in range(1, len(full_path)):
        dr = abs(full_path[i][0] - full_path[i - 1][0])
        dc = abs(full_path[i][1] - full_path[i - 1][1])
        if dr + dc != 1:
            return False

    for i in range(1, len(full_path) - 1):
        incoming = _sub(full_path[i], full_path[i - 1])
        outgoing = _sub(full_path[i + 1], full_path[i])
        if incoming == outgoing:
            if _add(full_path[i], incoming) != full_path[i + 1]:
                return False
            continue

        ahead = _add(full_path[i], incoming)
        if not _in_bounds(grid.shape, ahead[0], ahead[1]):
            return False
        if int(grid[ahead]) != int(blocker_color):
            return False

    return True


def _reconstruct_gap(
    grid: np.ndarray,
    path_color: int,
    blocker_color: int,
    first_seg: list[Coord],
    second_seg: list[Coord],
) -> tuple[list[Coord], list[Coord]] | None:
    assert len(first_seg) >= 2 and len(second_seg) >= 2
    start = first_seg[-1]
    start_prev = first_seg[-2]
    goal = second_seg[0]
    goal_next = second_seg[1]
    visible = set(first_seg) | set(second_seg)

    trail = [start]
    used = {start}

    def free_cell(cell: Coord) -> bool:
        r, c = cell
        if not _in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) == int(blocker_color):
            return False
        if cell in visible and cell != goal:
            return False
        if cell in used:
            return False
        return True

    def goal_ok(prev: Coord) -> bool:
        incoming = _sub(goal, prev)
        outgoing = _sub(goal_next, goal)
        if incoming == outgoing:
            return _add(goal, incoming) == goal_next
        ahead = _add(goal, incoming)
        return _in_bounds(grid.shape, ahead[0], ahead[1]) and int(grid[ahead]) == int(blocker_color)

    def dfs(cur: Coord, prev: Coord) -> tuple[list[Coord], list[Coord]] | None:
        if cur == goal:
            if not goal_ok(prev):
                return None
            hidden = trail[1:-1]
            full_path = first_seg[:-1] + trail + second_seg[1:]
            if not _full_path_valid(full_path, grid, blocker_color):
                return None
            return hidden, full_path

        direction = _sub(cur, prev)
        ahead = _add(cur, direction)

        if ahead == goal:
            options = [goal]
        else:
            if not _in_bounds(grid.shape, ahead[0], ahead[1]):
                return None
            if int(grid[ahead]) == int(blocker_color):
                options = []
                for nd in (_turn_left(direction), _turn_right(direction)):
                    nxt = _add(cur, nd)
                    if nxt == goal or free_cell(nxt):
                        options.append(nxt)
                if not options:
                    return None
                options.sort(key=lambda cell: abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]))
            else:
                if not free_cell(ahead):
                    return None
                options = [ahead]

        for nxt in options:
            trail.append(nxt)
            used.add(nxt)
            result = dfs(nxt, cur)
            if result is not None:
                return result
            used.remove(nxt)
            trail.pop()
        return None

    return dfs(start, start_prev)


def _candidate_solution(
    grid: np.ndarray,
    path_color: int,
    blocker_color: int,
    first_seg: list[Coord],
    second_seg: list[Coord],
) -> dict | None:
    out = np.asarray(grid, dtype=int).copy()
    score = 0
    for seg in (first_seg, second_seg):
        for blocker in _turn_blockers_for_segment(seg):
            r, c = blocker
            if not _in_bounds(out.shape, r, c):
                return None
            if int(out[r, c]) == int(path_color):
                return None
            if int(out[r, c]) == int(blocker_color):
                score += 1
            elif int(out[r, c]) == int(_bg_color(out)):
                out[r, c] = blocker_color
            else:
                return None

    rebuilt = _reconstruct_gap(out, path_color, blocker_color, first_seg, second_seg)
    if rebuilt is None:
        return None
    hidden, full_path = rebuilt
    for r, c in hidden:
        out[r, c] = path_color
    return {"score": score, "hidden_len": len(hidden), "grid": out, "path": full_path}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the missing blocker evidence and reconstruct the broken orthogonal path.

    Input structure:
    - A dominant background contains exactly one single-cell-wide path color and one blocker color.
    - The path is split into two visible border-to-gap chains because one contiguous middle segment was erased.
    - Some blocker-color singletons sit in front of path turns, while unrelated blocker singletons may also appear.

    Rule:
    - Identify the path color as the unique foreground color whose two 4-connected components both touch the border.
    - For each visible right-angle turn, add a blocker one step straight ahead if that blocker is missing.
    - Reconstruct the erased segment by following the path direction: keep moving straight unless the straight-ahead cell is a blocker, then turn so the rebuilt segment connects the two visible gap endpoints.

    Key insight:
    - Turn positions and front blockers encode the same control law in opposite directions, so visible turns recover missing blockers and visible blockers recover the missing path.
    """
    g = np.asarray(grid, dtype=int)
    bg, path_color, blocker_color = _infer_colors(g)
    _ = bg  # docstring-facing name; the solver only needs the inferred colors.

    path_components = [_trace_component(comp, g.shape) for comp in _components(g == path_color)]
    assert len(path_components) == 2, "expected exactly two visible path segments"

    candidates = [
        _candidate_solution(g, path_color, blocker_color, path_components[0], list(reversed(path_components[1]))),
        _candidate_solution(g, path_color, blocker_color, path_components[1], list(reversed(path_components[0]))),
    ]
    valid = [cand for cand in candidates if cand is not None]
    assert valid, "failed to reconstruct a valid path"

    valid.sort(
        key=lambda cand: (
            int(cand["score"]),
            -int(cand["hidden_len"]),
            tuple(cand["path"]),
        ),
        reverse=True,
    )
    return np.asarray(valid[0]["grid"], dtype=int)


def _line(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segments must be axis-aligned")


def _build_path(points: list[Coord]) -> list[Coord]:
    out: list[Coord] = []
    for i in range(len(points) - 1):
        seg = _line(points[i], points[i + 1])
        out.extend(seg if i == 0 else seg[1:])
    return out


def _path_turn_indices(path: list[Coord]) -> list[int]:
    turns = []
    for i in range(1, len(path) - 1):
        if _sub(path[i], path[i - 1]) != _sub(path[i + 1], path[i]):
            turns.append(i)
    return turns


def _turn_blockers_with_alternates(path: list[Coord]) -> tuple[list[Coord], set[Coord]]:
    blockers: list[Coord] = []
    reserve: set[Coord] = set()
    for i in _path_turn_indices(path):
        incoming = _sub(path[i], path[i - 1])
        outgoing = _sub(path[i + 1], path[i])
        actual = _add(path[i], incoming)
        alternate = _sub(path[i], outgoing)
        blockers.append(actual)
        reserve.add(actual)
        reserve.add(alternate)
    return blockers, reserve


def _transform_coords(coords: list[Coord], shape: tuple[int, int], rot: int, flip: bool) -> tuple[list[Coord], tuple[int, int]]:
    h, w = shape
    out = list(coords)
    for _ in range(rot):
        out = [(c, h - 1 - r) for r, c in out]
        h, w = w, h
    if flip:
        out = [(r, w - 1 - c) for r, c in out]
    return out, (h, w)


def _choose_rows(h: int) -> list[int] | None:
    rows = list(range(2, h - 2))
    if len(rows) < 6:
        return None
    for _ in range(100):
        choice = random.sample(rows, 3)
        if min(abs(a - b) for i, a in enumerate(choice) for b in choice[i + 1 :]) >= 2:
            return choice
    return None


def _choose_cols(w: int) -> tuple[int, int] | None:
    if w < 14:
        return None
    c1 = random.randint(3, w - 8)
    c2 = random.randint(c1 + 4, w - 4)
    return c1, c2


def _gap_candidates(path: list[Coord], hidden_turns: int) -> list[tuple[int, int]]:
    turn_set = set(_path_turn_indices(path))
    out: list[tuple[int, int]] = []
    n = len(path)
    for a in range(2, n - 3):
        for b in range(a + 1, min(n - 3, a + 11) + 1):
            if sum(a <= idx <= b for idx in turn_set) != hidden_turns:
                continue
            visible_turns = [idx for idx in turn_set if idx < a - 1 or idx > b + 1]
            if len(visible_turns) < 2:
                continue
            out.append((a, b))
    return out


def _render_grid(shape: tuple[int, int], bg: int, path_color: int, blocker_color: int, path: list[Coord], blockers: set[Coord]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for r, c in blockers:
        grid[r, c] = blocker_color
    for r, c in path:
        grid[r, c] = path_color
    return grid


def _distance_from_path(cell: Coord, path_cells: set[Coord]) -> int:
    return min(abs(cell[0] - r) + abs(cell[1] - c) for r, c in path_cells)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(14, 26)
    rows = _choose_rows(h)
    cols = _choose_cols(w)
    if rows is None or cols is None:
        return None

    r0, r1, r2 = rows
    c1, c2 = cols
    points = [(r0, 0), (r0, c1), (r1, c1), (r1, c2), (r2, c2), (r2, w - 1)]
    path = _build_path(points)
    blockers, reserve = _turn_blockers_with_alternates(path)

    hidden_turns = random.choice([0, 1, 2])
    gap_options = _gap_candidates(path, hidden_turns)
    if not gap_options:
        return None
    gap_a, gap_b = random.choice(gap_options)
    turn_indices = set(_path_turn_indices(path))
    visible_turn_indices = [idx for idx in turn_indices if idx < gap_a - 1 or idx > gap_b + 1]
    if len(visible_turn_indices) < 2:
        return None

    blocker_by_turn = {_path_turn_indices(path)[i]: blockers[i] for i in range(len(blockers))}
    remove_count = random.randint(1, len(visible_turn_indices) - 1)
    removed_turns = set(random.sample(visible_turn_indices, remove_count))
    removed_blockers = {blocker_by_turn[idx] for idx in removed_turns}

    rot = random.randint(0, 3)
    flip = random.random() < 0.5
    path, shape = _transform_coords(path, (h, w), rot, flip)
    blockers, _ = _transform_coords(blockers, (h, w), rot, flip)
    removed_blockers_list, _ = _transform_coords(list(removed_blockers), (h, w), rot, flip)
    reserve_cells, _ = _transform_coords(list(reserve), (h, w), rot, flip)

    blockers_set = set(blockers)
    removed_blockers_set = set(removed_blockers_list)
    reserve_set = set(reserve_cells) | set(path)
    gap_cells = set(path[gap_a : gap_b + 1])

    bg, path_color, blocker_color = random.sample(range(10), 3)
    output_grid = _render_grid(shape, bg, path_color, blocker_color, path, blockers_set)
    input_grid = output_grid.copy()
    for r, c in gap_cells:
        input_grid[r, c] = bg
    for r, c in removed_blockers_set:
        input_grid[r, c] = bg

    path_cells = set(path)
    distractor_candidates = [
        (r, c)
        for r in range(shape[0])
        for c in range(shape[1])
        if (r, c) not in reserve_set
        and (r, c) not in blockers_set
        and _distance_from_path((r, c), path_cells) >= 2
    ]
    random.shuffle(distractor_candidates)
    distractor_count = min(len(distractor_candidates), random.randint(0, 4))
    for r, c in distractor_candidates[:distractor_count]:
        input_grid[r, c] = blocker_color
        output_grid[r, c] = blocker_color

    try:
        solved = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(solved, output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid turning-path instance")
