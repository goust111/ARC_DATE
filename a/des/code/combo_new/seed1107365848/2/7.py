from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PASSAGE_COLOR = 0
WALL_COLOR = 5
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

COARSE_MIN = 5
COARSE_MAX = 10
MAX_PAIRS = 3
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 250

STYLE_SPECS = {
    "straight": {"lengths": (3, 6), "min_turns": 0, "max_turns": 0},
    "single": {"lengths": (3, 7), "min_turns": 1, "max_turns": 1},
    "multi": {"lengths": (5, 8), "min_turns": 2, "max_turns": 4},
}


def _neighbors(node: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = node
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        prev_r, prev_c = path[i - 1]
        cur_r, cur_c = path[i]
        next_r, next_c = path[i + 1]
        d1 = (cur_r - prev_r, cur_c - prev_c)
        d2 = (next_r - cur_r, next_c - cur_c)
        if d1 != d2:
            turns += 1
    return turns


def _weighted_choice(options: list[tuple[tuple[int, int], tuple[int, int], int]]) -> tuple[int, int] | None:
    if not options:
        return None
    total = sum(weight for _node, _direction, weight in options)
    pick = random.randint(1, total)
    running = 0
    for node, _direction, weight in options:
        running += weight
        if running >= pick:
            return node
    return options[-1][0]


def _grow_path(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    style: str,
    *,
    start: tuple[int, int] | None = None,
) -> list[tuple[int, int]] | None:
    spec = STYLE_SPECS[style]
    min_len, max_len = spec["lengths"]
    min_turns = spec["min_turns"]
    max_turns = spec["max_turns"]
    all_cells = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if (r, c) not in occupied]
    if start is None and not all_cells:
        return None

    for _ in range(PATH_ATTEMPTS):
        length = random.randint(min_len, max_len)
        if start is None:
            current = random.choice(all_cells)
        else:
            if start in occupied:
                return None
            current = start

        path = [current]
        used = set(occupied)
        used.add(current)
        prev_dir: tuple[int, int] | None = None
        turns = 0

        while len(path) < length:
            cur = path[-1]
            options: list[tuple[tuple[int, int], tuple[int, int], int]] = []
            remaining_after_step = length - len(path) - 1
            turns_left_needed = max(0, min_turns - turns)

            for nxt in _neighbors(cur, shape):
                if nxt in used:
                    continue
                direction = (nxt[0] - cur[0], nxt[1] - cur[1])
                extra_turn = int(prev_dir is not None and direction != prev_dir)
                new_turns = turns + extra_turn
                if new_turns > max_turns:
                    continue

                future_used = used | {nxt}
                future_free = sum(1 for nb in _neighbors(nxt, shape) if nb not in future_used)
                if remaining_after_step > 0 and future_free == 0:
                    continue

                max_future_turns = new_turns + remaining_after_step
                if max_future_turns < min_turns:
                    continue

                weight = 3
                if prev_dir is not None:
                    is_turn = direction != prev_dir
                    if turns_left_needed > 0:
                        weight = 8 if is_turn else 2
                    elif new_turns == max_turns:
                        weight = 8 if not is_turn else 1
                    else:
                        weight = 5 if not is_turn else 4
                options.append((nxt, direction, weight))

            nxt = _weighted_choice(options)
            if nxt is None:
                break

            direction = (nxt[0] - cur[0], nxt[1] - cur[1])
            if prev_dir is not None and direction != prev_dir:
                turns += 1
            prev_dir = direction
            path.append(nxt)
            used.add(nxt)

        if len(path) == length:
            actual_turns = _turn_count(path)
            if min_turns <= actual_turns <= max_turns:
                return path

    return None


def _add_path(
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    path: list[tuple[int, int]],
) -> None:
    for node in path:
        nodes.add(node)
    for a, b in zip(path, path[1:]):
        edges.add(tuple(sorted((a, b))))


def _edge_count(mask: np.ndarray) -> int:
    edges = 0
    edges += int(np.count_nonzero(mask[:, :-1] & mask[:, 1:]))
    edges += int(np.count_nonzero(mask[:-1, :] & mask[1:, :]))
    return edges


def _is_tree(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)

    return len(seen) == len(coords) and _edge_count(mask) == len(coords) - 1


def _marker_colors(grid: np.ndarray) -> list[int]:
    values, counts = np.unique(grid, return_counts=True)
    markers = [int(color) for color, count in zip(values, counts) if int(count) == 2]
    assert markers, "expected at least one marker color appearing exactly twice"
    return markers


def _passage_and_wall_colors(grid: np.ndarray, marker_colors: list[int]) -> tuple[int, int]:
    structural_colors = [int(c) for c in np.unique(grid) if int(c) not in set(marker_colors)]
    assert len(structural_colors) == 2, "expected passage color and wall color"

    candidates: list[tuple[int, int]] = []
    marker_mask = np.isin(grid, marker_colors)
    for color in structural_colors:
        traversable = (grid == color) | marker_mask
        if _is_tree(traversable):
            other = structural_colors[0] if structural_colors[1] == color else structural_colors[1]
            candidates.append((color, other))

    assert len(candidates) == 1, "expected one traversable tree color"
    return candidates[0]


def _positions_by_color(grid: np.ndarray, colors: list[int]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(coords) == 2, f"expected exactly two markers for color {color}"
        out[int(color)] = coords
    return out


def _path_between(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    q = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            nxt = (rr, cc)
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and nxt not in parents:
                parents[nxt] = cur
                q.append(nxt)

    assert goal in parents, "markers must be connected"
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _outer_corner(
    prev: tuple[int, int],
    cur: tuple[int, int],
    nxt: tuple[int, int],
) -> tuple[int, int] | None:
    dr1 = cur[0] - prev[0]
    dc1 = cur[1] - prev[1]
    dr2 = nxt[0] - cur[0]
    dc2 = nxt[1] - cur[1]
    if (dr1, dc1) == (dr2, dc2):
        return None
    return (cur[0] + dr1 - dr2, cur[1] + dc1 - dc2)


def main(grid: np.ndarray) -> np.ndarray:
    """Color the outer wall corners at every turn along each same-color marker path in the acyclic maze.

    Input structure:
    - The grid contains exactly two structural colors: one passage/background color and one grey wall color.
    - Same-color marker pixels appear in pairs on the passage network, and different pairs use different colors.
    - The passage cells together with the markers form a 4-connected tree, so any two markers have a unique route.

    Rule:
    - For each marker color, find the unique 4-connected path between its two pixels through the passage network.
    - Scan that path for turns, meaning consecutive steps where the direction changes between horizontal and vertical.
    - At each turn, locate the diagonal wall pixel on the outside of the bend and recolor that wall pixel to the marker color.
    - Leave the markers themselves and every straight corridor wall unchanged.

    Key insight:
    - The marked result is not the path itself: only the convex wall corner opposite each L-turn on the unique tree path is recolored.
    """
    g = np.asarray(grid, dtype=int)
    marker_colors = _marker_colors(g)
    passage_color, wall_color = _passage_and_wall_colors(g, marker_colors)
    marker_positions = _positions_by_color(g, marker_colors)

    traversable = (g == passage_color) | np.isin(g, marker_colors)
    out = g.copy()

    for color, (start, goal) in marker_positions.items():
        path = _path_between(traversable, start, goal)
        for i in range(1, len(path) - 1):
            corner = _outer_corner(path[i - 1], path[i], path[i + 1])
            if corner is None:
                continue
            rr, cc = corner
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == wall_color:
                out[rr, cc] = int(color)

    return out


def _pair_count_limit(shape: tuple[int, int]) -> int:
    cells = shape[0] * shape[1]
    if cells < 30:
        return 2
    return MAX_PAIRS


def _build_feature_tree(
    shape: tuple[int, int],
) -> tuple[
    list[list[tuple[int, int]]],
    set[tuple[int, int]],
    set[tuple[tuple[int, int], tuple[int, int]]],
] | None:
    pair_count = random.randint(1, _pair_count_limit(shape))
    occupied: set[tuple[int, int]] = set()
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    pair_paths: list[list[tuple[int, int]]] = []
    marker_nodes: set[tuple[int, int]] = set()

    first_style = random.choice(("single", "multi"))
    first_path = _grow_path(shape, occupied, first_style)
    if first_path is None:
        return None
    _add_path(occupied, edges, first_path)
    pair_paths.append(first_path)
    marker_nodes.update((first_path[0], first_path[-1]))

    for _ in range(pair_count - 1):
        style = random.choice(("straight", "single", "multi"))
        candidates = list(occupied)
        random.shuffle(candidates)
        non_marker = [node for node in candidates if node not in marker_nodes]
        if non_marker:
            candidates = non_marker + [node for node in candidates if node in marker_nodes]

        placed = False
        for attach in candidates:
            free_neighbors = [nb for nb in _neighbors(attach, shape) if nb not in occupied]
            random.shuffle(free_neighbors)
            for start in free_neighbors:
                branch = _grow_path(shape, occupied, style, start=start)
                if branch is None:
                    continue
                edges.add(tuple(sorted((attach, start))))
                _add_path(occupied, edges, branch)
                pair_paths.append(branch)
                marker_nodes.update((branch[0], branch[-1]))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    total_edges = len(edges)
    assert total_edges == len(occupied) - 1
    assert any(_turn_count(path) > 0 for path in pair_paths)
    return pair_paths, occupied, edges


def _render_grid(
    shape: tuple[int, int],
    pair_paths: list[list[tuple[int, int]]],
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    h, w = shape
    grid = np.full((2 * h + 1, 2 * w + 1), WALL_COLOR, dtype=int)

    for r, c in nodes:
        grid[2 * r + 1, 2 * c + 1] = PASSAGE_COLOR
    for a, b in edges:
        pr0, pc0 = 2 * a[0] + 1, 2 * a[1] + 1
        pr1, pc1 = 2 * b[0] + 1, 2 * b[1] + 1
        grid[(pr0 + pr1) // 2, (pc0 + pc1) // 2] = PASSAGE_COLOR

    colors = random.sample(MARKER_PALETTE, len(pair_paths))
    for color, path in zip(colors, pair_paths):
        for r, c in (path[0], path[-1]):
            grid[2 * r + 1, 2 * c + 1] = color

    return grid


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    marker_colors = _marker_colors(grid)
    passage_color, _wall_color = _passage_and_wall_colors(grid, marker_colors)
    traversable = (grid == passage_color) | np.isin(grid, marker_colors)
    if not _is_tree(traversable):
        return False
    out = main(grid)
    return bool(np.any(out != grid))


def _generate_case() -> np.ndarray | None:
    coarse_h = random.randint(COARSE_MIN, COARSE_MAX)
    coarse_w = random.randint(COARSE_MIN, COARSE_MAX)
    built = _build_feature_tree((coarse_h, coarse_w))
    if built is None:
        return None
    pair_paths, nodes, edges = built
    grid = _render_grid((coarse_h, coarse_w), pair_paths, nodes, edges)
    try:
        if _validate_input(grid):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze input")
