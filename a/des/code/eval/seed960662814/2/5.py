from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300
WALL = 0


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component_from(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or not bool(mask[sr, sc]):
        return set()
    comp = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in comp:
                comp.add((nr, nc))
                q.append((nr, nc))
    return comp


def _parse_legend(grid: np.ndarray, bg: int):
    assert int(grid[0, 0]) != int(bg), "legend must occupy the top-left corner"
    comp = _component_from(grid != int(bg), (0, 0))
    assert comp, "failed to locate legend component"

    max_r = max(r for r, _ in comp)
    max_c = max(c for _, c in comp)
    legend = np.asarray(grid[: max_r + 1, : max_c + 1], dtype=int)
    assert legend.shape[0] == legend.shape[1] and legend.shape[0] >= 3

    order = [int(v) for v in legend[0, 1:]]
    row_headers = [int(v) for v in legend[1:, 0]]
    assert len(order) == len(row_headers)
    assert set(order) == set(row_headers)
    assert len(set(order)) == len(order)

    mix_map: dict[tuple[int, int], int] = {}
    for i, row_color in enumerate(row_headers, start=1):
        for j, col_color in enumerate(order, start=1):
            mix_map[(int(row_color), int(col_color))] = int(legend[i, j])

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[: legend.shape[0], : legend.shape[1]] = True
    return legend_mask, order, mix_map, int(legend[0, 0])


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    legend_mask, order, mix_map, wall = _parse_legend(g, bg)
    h, w = g.shape

    source_positions: dict[int, list[tuple[int, int]]] = {}
    for color in order:
        pts = [
            (int(r), int(c))
            for r, c in np.argwhere((g == int(color)) & (~legend_mask))
        ]
        assert len(pts) == 1, "each legend color must have exactly one source pixel"
        source_positions[int(color)] = pts

    out = g.copy()
    claimed = np.zeros((h, w), dtype=bool)
    frontier: dict[int, list[tuple[int, int]]] = {}
    for color, pts in source_positions.items():
        frontier[int(color)] = list(pts)
        for r, c in pts:
            claimed[r, c] = True

    while frontier:
        proposals: dict[tuple[int, int], set[int]] = defaultdict(set)
        for color, cells in frontier.items():
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(legend_mask[nr, nc]) or bool(claimed[nr, nc]):
                        continue
                    if int(g[nr, nc]) == int(wall):
                        continue
                    if int(g[nr, nc]) != int(bg):
                        continue
                    proposals[(nr, nc)].add(int(color))

        next_frontier: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (r, c), colors in proposals.items():
            visible_order = [color for color in order if color in colors]
            if len(visible_order) == 1:
                color = int(visible_order[0])
                out[r, c] = color
                claimed[r, c] = True
                next_frontier[color].append((r, c))
            elif len(visible_order) >= 2:
                mix_color = int(mix_map[(visible_order[0], visible_order[1])])
                out[r, c] = mix_color
                claimed[r, c] = True

        frontier = {color: pts for color, pts in next_frontier.items() if pts}

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse multiple colored sources through a black-walled maze and stop fronts at mixed collisions.
    """
    return _transform(grid)


def _in_bounds(rows: int, cols: int, node: tuple[int, int]) -> bool:
    r, c = node
    return 0 <= r < rows and 0 <= c < cols


def _logical_neighbors(rows: int, cols: int, node: tuple[int, int]):
    r, c = node
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _free_degree(rows: int, cols: int, node: tuple[int, int], used: set[tuple[int, int]]) -> int:
    return sum(1 for nb in _logical_neighbors(rows, cols, node) if nb not in used)


def _search_branch(
    rows: int,
    cols: int,
    start: tuple[int, int],
    first_dir: tuple[int, int],
    length: int,
    used: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    first = (start[0] + first_dir[0], start[1] + first_dir[1])
    if not _in_bounds(rows, cols, first) or first in used:
        return None

    branch = [first]
    local_used = set(used)
    local_used.add(first)

    def dfs(cur: tuple[int, int], remain: int) -> bool:
        if remain == 0:
            return True
        options = [nb for nb in _logical_neighbors(rows, cols, cur) if nb not in local_used]
        rng.shuffle(options)
        options.sort(key=lambda nb: _free_degree(rows, cols, nb, local_used | {nb}), reverse=True)
        for nb in options:
            local_used.add(nb)
            branch.append(nb)
            if dfs(nb, remain - 1):
                return True
            branch.pop()
            local_used.remove(nb)
        return False

    if dfs(first, length - 1):
        return branch
    return None


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _build_tree(
    rows: int,
    cols: int,
    source_colors: list[int],
    rng: random.Random,
):
    k = len(source_colors)
    for _ in range(120):
        root = (rng.randint(2, rows - 3), rng.randint(2, cols - 3))
        dirs = list(DIR4)
        rng.shuffle(dirs)
        branch_len = rng.randint(2, 4)
        tail_len = rng.randint(branch_len + (1 if k == 3 else 0), branch_len + 2)

        nodes = {root}
        edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        path_a = _search_branch(rows, cols, root, dirs[0], branch_len, nodes, rng)
        if path_a is None:
            continue
        for prev, cur in zip([root] + path_a[:-1], path_a):
            nodes.add(cur)
            edges.add(_edge_key(prev, cur))

        path_b = _search_branch(rows, cols, root, dirs[1], branch_len, nodes, rng)
        if path_b is None:
            continue
        for prev, cur in zip([root] + path_b[:-1], path_b):
            nodes.add(cur)
            edges.add(_edge_key(prev, cur))

        path_tail = _search_branch(rows, cols, root, dirs[2], tail_len, nodes, rng)
        if path_tail is None:
            continue
        for prev, cur in zip([root] + path_tail[:-1], path_tail):
            nodes.add(cur)
            edges.add(_edge_key(prev, cur))

        target_lo = max(len(nodes) + 2, int(rows * cols * 0.35))
        target_hi = max(target_lo, int(rows * cols * 0.55))
        target_nodes = rng.randint(target_lo, target_hi)
        while len(nodes) < target_nodes:
            candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for parent in nodes:
                if parent == root:
                    continue
                for child in _logical_neighbors(rows, cols, parent):
                    if child not in nodes:
                        candidates.append((parent, child))
            if not candidates:
                break
            parent, child = rng.choice(candidates)
            nodes.add(child)
            edges.add(_edge_key(parent, child))

        source_nodes = {
            path_a[-1]: int(source_colors[0]),
            path_b[-1]: int(source_colors[1]),
        }
        if k == 3:
            source_nodes[path_tail[-1]] = int(source_colors[2])

        return root, nodes, edges, source_nodes
    return None


def _render_maze(
    rows: int,
    cols: int,
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    source_nodes: dict[tuple[int, int], int],
    bg: int,
) -> np.ndarray:
    maze = np.full((2 * rows + 1, 2 * cols + 1), WALL, dtype=int)
    for r, c in nodes:
        maze[2 * r + 1, 2 * c + 1] = int(bg)
    for a, b in edges:
        ar, ac = a
        br, bc = b
        maze[ar + br + 1, ac + bc + 1] = int(bg)
    for (r, c), color in source_nodes.items():
        maze[2 * r + 1, 2 * c + 1] = int(color)
    return maze


def _build_legend(source_colors: list[int], pair_to_mix: dict[tuple[int, int], int]) -> np.ndarray:
    k = len(source_colors)
    legend = np.empty((k + 1, k + 1), dtype=int)
    legend[0, 0] = WALL
    legend[0, 1:] = source_colors
    legend[1:, 0] = source_colors
    for i, row_color in enumerate(source_colors, start=1):
        for j, col_color in enumerate(source_colors, start=1):
            if row_color == col_color:
                legend[i, j] = row_color
            else:
                legend[i, j] = int(pair_to_mix[(row_color, col_color)])
    return legend


def _pair_map(source_colors: list[int], mix_colors: list[int]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    idx = 0
    for i, a in enumerate(source_colors):
        for b in source_colors[i + 1 :]:
            mix = int(mix_colors[idx])
            idx += 1
            out[(int(a), int(b))] = mix
            out[(int(b), int(a))] = mix
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    k = rng.choice([2, 3])
    bg = rng.choice([c for c in range(1, 10) if c != WALL])
    palette = [c for c in range(1, 10) if c != bg]
    rng.shuffle(palette)
    source_colors = palette[:k]
    rng.shuffle(source_colors)
    pair_count = k * (k - 1) // 2
    mix_colors = palette[k : k + pair_count]
    if len(set(source_colors) | set(mix_colors) | {bg, WALL}) != 2 + k + pair_count:
        return None
    pair_to_mix = _pair_map(source_colors, mix_colors)
    legend = _build_legend(source_colors, pair_to_mix)

    logical_rows = rng.randint(6, 8)
    logical_cols = rng.randint(6, 9)
    built = _build_tree(logical_rows, logical_cols, source_colors, rng)
    if built is None:
        return None
    _root, nodes, edges, source_nodes = built
    maze = _render_maze(logical_rows, logical_cols, nodes, edges, source_nodes, bg)

    maze_r0 = legend.shape[0] + rng.randint(2, 4)
    maze_c0 = legend.shape[1] + rng.randint(2, 4)
    bottom_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    h = maze_r0 + maze.shape[0] + bottom_margin
    w = maze_c0 + maze.shape[1] + right_margin
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    grid[: legend.shape[0], : legend.shape[1]] = legend
    grid[maze_r0 : maze_r0 + maze.shape[0], maze_c0 : maze_c0 + maze.shape[1]] = maze

    counts = Counter(map(int, grid.flat))
    if counts.most_common(1)[0][0] != int(bg):
        return None

    out = _transform(grid)
    maze_out = out[maze_r0 : maze_r0 + maze.shape[0], maze_c0 : maze_c0 + maze.shape[1]]
    mixed_in_maze = any(int(color) in set(map(int, maze_out.flat)) for color in mix_colors)
    if not mixed_in_maze:
        return None

    passable_mask = maze != WALL
    remaining_bg = np.any((maze_out == int(bg)) & passable_mask)
    if k == 2 and not remaining_bg:
        return None
    if k == 3 and remaining_bg:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt(rng)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid maze-signal sample")
