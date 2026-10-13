from __future__ import annotations

import random

import numpy as np


BLACK = 0
ODD_SIZES = (3, 5, 7)
MAX_GENERATION_ATTEMPTS = 400


def _candidate_axis_splits(length: int) -> list[tuple[int, int]]:
    """Return all (region_count, region_size) pairs for length = count*size + (count-1)."""
    splits: list[tuple[int, int]] = []
    for region_size in range(3, length + 1, 2):
        step = region_size + 1
        if (length + 1) % step != 0:
            continue
        region_count = (length + 1) // step
        if region_count >= 1:
            splits.append((region_count, region_size))
    return splits


def _chain_info(
    successors: dict[tuple[int, int], tuple[int, int] | None]
) -> tuple[dict[tuple[int, int], tuple[int, int] | None], tuple[int, int], list[tuple[int, int]]] | None:
    indegree = {node: 0 for node in successors}
    predecessors = {node: None for node in successors}
    terminal_nodes = [node for node, succ in successors.items() if succ is None]
    if len(terminal_nodes) != 1:
        return None

    edge_count = 0
    for node, succ in successors.items():
        if succ is None:
            continue
        if succ not in indegree:
            return None
        indegree[succ] += 1
        if indegree[succ] > 1:
            return None
        predecessors[succ] = node
        edge_count += 1

    if edge_count != len(successors) - 1:
        return None

    starts = [node for node, deg in indegree.items() if deg == 0]
    if len(starts) != 1:
        return None
    start = starts[0]
    if any(deg != 1 for node, deg in indegree.items() if node != start):
        return None

    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    cur: tuple[int, int] | None = start
    while cur is not None:
        if cur in seen:
            return None
        seen.add(cur)
        order.append(cur)
        cur = successors[cur]

    if len(seen) != len(successors) or order[-1] != terminal_nodes[0]:
        return None
    return predecessors, start, order


def _validate_layout_candidate(
    grid: np.ndarray, region_rows: int, region_cols: int, cell_h: int, cell_w: int
) -> dict | None:
    height, width = map(int, grid.shape)
    if height != region_rows * cell_h + (region_rows - 1):
        return None
    if width != region_cols * cell_w + (region_cols - 1):
        return None

    for band_r in range(region_rows - 1):
        divider_r = (band_r + 1) * (cell_h + 1) - 1
        if not np.all(grid[divider_r, :] == BLACK):
            return None
    for band_c in range(region_cols - 1):
        divider_c = (band_c + 1) * (cell_w + 1) - 1
        if not np.all(grid[:, divider_c] == BLACK):
            return None

    colors = np.zeros((region_rows, region_cols), dtype=int)
    successors: dict[tuple[int, int], tuple[int, int] | None] = {}
    center_r = cell_h // 2
    center_c = cell_w // 2

    for i in range(region_rows):
        for j in range(region_cols):
            r0 = i * (cell_h + 1)
            c0 = j * (cell_w + 1)
            region = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            marker_coords = np.argwhere(region != BLACK)
            if marker_coords.shape[0] != 1:
                return None
            local_r, local_c = map(int, marker_coords[0])
            dr = local_r - center_r
            dc = local_c - center_c
            if abs(dr) > 1 or abs(dc) > 1:
                return None
            color = int(region[local_r, local_c])
            colors[i, j] = color
            if dr == 0 and dc == 0:
                successors[(i, j)] = None
                continue
            dst = (i + dr, j + dc)
            if not (0 <= dst[0] < region_rows and 0 <= dst[1] < region_cols):
                return None
            successors[(i, j)] = dst

    chain = _chain_info(successors)
    if chain is None:
        return None
    predecessors, start, order = chain
    return {
        "region_rows": region_rows,
        "region_cols": region_cols,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "colors": colors,
        "successors": successors,
        "predecessors": predecessors,
        "start": start,
        "order": order,
    }


def _infer_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    candidates: list[dict] = []
    for region_rows, cell_h in _candidate_axis_splits(int(g.shape[0])):
        for region_cols, cell_w in _candidate_axis_splits(int(g.shape[1])):
            info = _validate_layout_candidate(g, region_rows, region_cols, cell_h, cell_w)
            if info is not None:
                candidates.append(info)

    assert len(candidates) == 1, "expected a unique equal-region partition and chain interpretation"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the partitioned grid into a thumbnail after shifting each region's color forward along a hidden pointer chain.

    Input structure:
    - Full black divider lines split the canvas into an R x C array of equal odd-sized rectangular regions.
    - Each region contains exactly one colored marker pixel, placed at the center or at one of the center's eight neighbors.
    - An off-center marker points to the adjacent region in that direction, while a centered marker is the unique chain endpoint.
    - Exactly one region has indegree zero, and together all regions form one non-branching directed chain.

    Rule:
    - Remove the divider lines and output an R x C thumbnail, one cell per region.
    - For each region, write the color of the region that points into it.
    - The unique chain start has no predecessor, so its output cell is black.

    Key insight:
    - The marker color is payload and the offset from the local center is the routing instruction; the thumbnail is not a direct per-region color compression, but a one-step color relay along the decoded chain.
    """
    info = _infer_layout(np.asarray(grid, dtype=int))
    out = np.zeros((info["region_rows"], info["region_cols"]), dtype=int)
    colors = info["colors"]
    for node, pred in info["predecessors"].items():
        if pred is None:
            continue
        out[node] = colors[pred]
    return out


def _king_neighbors(region_rows: int, region_cols: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(region_rows):
        for c in range(region_cols):
            nbrs: list[tuple[int, int]] = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < region_rows and 0 <= cc < region_cols:
                        nbrs.append((rr, cc))
            neighbors[(r, c)] = nbrs
    return neighbors


def _dfs_hamiltonian(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    total_nodes: int,
) -> bool:
    if len(path) == total_nodes:
        return True

    cur = path[-1]
    candidates = [node for node in neighbors[cur] if node not in used]
    random.shuffle(candidates)
    candidates.sort(key=lambda node: sum(1 for nxt in neighbors[node] if nxt not in used and nxt != cur))

    for node in candidates:
        used.add(node)
        path.append(node)
        if _dfs_hamiltonian(path, used, neighbors, total_nodes):
            return True
        path.pop()
        used.remove(node)
    return False


def _hamiltonian_path(region_rows: int, region_cols: int) -> list[tuple[int, int]] | None:
    neighbors = _king_neighbors(region_rows, region_cols)
    nodes = list(neighbors)
    total_nodes = region_rows * region_cols
    for _ in range(40):
        start = random.choice(nodes)
        path = [start]
        used = {start}
        if _dfs_hamiltonian(path, used, neighbors, total_nodes):
            return path
    return None


def _path_direction_stats(path: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], bool, bool]:
    directions = {(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])}
    has_diagonal = any(abs(dr) == 1 and abs(dc) == 1 for dr, dc in directions)
    has_cardinal = any((dr == 0) ^ (dc == 0) for dr, dc in directions)
    return directions, has_diagonal, has_cardinal


def _color_sequence(length: int) -> list[int]:
    palette_size = random.randint(2, min(6, length))
    palette = random.sample(list(range(1, 10)), k=palette_size)
    colors = [random.choice(palette)]
    for _ in range(1, length):
        choices = [color for color in palette if color != colors[-1]]
        colors.append(random.choice(choices))
    return colors


def _render_grid(
    region_rows: int,
    region_cols: int,
    cell_h: int,
    cell_w: int,
    path: list[tuple[int, int]],
    colors_along_path: list[int],
) -> np.ndarray:
    height = region_rows * cell_h + (region_rows - 1)
    width = region_cols * cell_w + (region_cols - 1)
    grid = np.zeros((height, width), dtype=int)
    color_by_region = {node: color for node, color in zip(path, colors_along_path)}
    center_r = cell_h // 2
    center_c = cell_w // 2

    for index, node in enumerate(path):
        r0 = node[0] * (cell_h + 1)
        c0 = node[1] * (cell_w + 1)
        if index + 1 < len(path):
            nxt = path[index + 1]
            dr = nxt[0] - node[0]
            dc = nxt[1] - node[1]
        else:
            dr = 0
            dc = 0
        grid[r0 + center_r + dr, c0 + center_c + dc] = color_by_region[node]
    return grid


def _self_colors(info: dict) -> np.ndarray:
    return info["colors"].copy()


def _valid_generation_configs() -> list[tuple[int, int, int, int]]:
    configs: list[tuple[int, int, int, int]] = []
    for cell_h in ODD_SIZES:
        for cell_w in ODD_SIZES:
            for region_rows in range(2, 7):
                height = region_rows * cell_h + (region_rows - 1)
                if height > 30:
                    continue
                for region_cols in range(2, 7):
                    width = region_cols * cell_w + (region_cols - 1)
                    region_count = region_rows * region_cols
                    if width <= 30 and 6 <= region_count <= 20:
                        configs.append((region_rows, region_cols, cell_h, cell_w))
    return configs


GENERATION_CONFIGS = _valid_generation_configs()


def _attempt_generate() -> np.ndarray | None:
    region_rows, region_cols, cell_h, cell_w = random.choice(GENERATION_CONFIGS)
    path = _hamiltonian_path(region_rows, region_cols)
    if path is None:
        return None

    directions, has_diagonal, has_cardinal = _path_direction_stats(path)
    if len(directions) < (3 if len(path) >= 8 else 2):
        return None
    if len(path) >= 8 and not has_cardinal:
        return None
    if len(path) >= 10 and not has_diagonal:
        return None

    if random.random() < 0.5:
        path = list(reversed(path))

    colors_along_path = _color_sequence(len(path))
    grid = _render_grid(region_rows, region_cols, cell_h, cell_w, path, colors_along_path)

    try:
        info = _infer_layout(grid)
        out = main(grid)
    except AssertionError:
        return None

    own_colors = _self_colors(info)
    if np.array_equal(out, own_colors):
        return None
    if np.any(out == own_colors):
        return None

    if len(np.unique(own_colors)) < 2:
        return None
    if len(np.unique(out)) < 3:
        return None

    recovered_path = info["order"]
    recovered_colors = [int(info["colors"][node]) for node in recovered_path]
    if any(a == b for a, b in zip(recovered_colors, recovered_colors[1:])):
        return None

    for index, node in enumerate(recovered_path):
        expected = BLACK if index == 0 else recovered_colors[index - 1]
        if int(out[node]) != expected:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a non-degenerate pointer-chain grid")
