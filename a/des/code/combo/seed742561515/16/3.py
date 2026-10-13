from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SEED_COLORS = (1, 2, 3, 6, 7, 8, 9)


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _path_degrees(path_mask: np.ndarray) -> np.ndarray:
    height, width = path_mask.shape
    degrees = np.zeros((height, width), dtype=int)
    for r, c in np.argwhere(path_mask):
        rr = int(r)
        cc = int(c)
        degrees[rr, cc] = sum(
            1 for nr, nc in _neighbors4(rr, cc, height, width) if path_mask[nr, nc]
        )
    return degrees


def _seed_sources(grid: np.ndarray, path_mask: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    seed_coords = np.argwhere((grid != BLACK) & (grid != GREY))
    sources: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for r, c in seed_coords:
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        assert color not in seen_colors, "expected distinct seed colors"
        seen_colors.add(color)

        neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(rr, cc, height, width)
            if path_mask[nr, nc]
        ]
        assert len(neighbors) == 1, "each seed must touch exactly one path endpoint"
        sr, sc = neighbors[0]
        sources.append((int(sr), int(sc), color))

    assert len(sources) >= 2, "expected at least two seeds"
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate simultaneous color waves spreading along the grey path network.

    Input structure:
    - Background is black.
    - Grey pixels form a 1-pixel-wide orthogonal path network.
    - Distinct colored singleton seeds sit next to path endpoints.

    Rule:
    - All seeds spread at the same speed along adjacent grey pixels.
    - A pixel reached first by one color takes that color.
    - A grey pixel reached simultaneously by multiple colors becomes yellow.
    - Tied pixels still pass those tied wavefronts onward, so later branches can
      also become yellow if they are reached in lockstep from both sides.
    - Seed pixels themselves stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    path_mask = g == GREY
    assert bool(path_mask.any()), "expected a grey path network"
    sources = _seed_sources(g, path_mask)

    best_time = np.full((height, width), -1, dtype=int)
    owners: list[list[set[int]]] = [
        [set() for _ in range(width)] for _ in range(height)
    ]
    queue = deque()

    for r, c, color in sources:
        if best_time[r, c] == -1:
            best_time[r, c] = 0
            owners[r][c].add(color)
            queue.append((r, c, color, 0))
        elif best_time[r, c] == 0 and color not in owners[r][c]:
            owners[r][c].add(color)
            queue.append((r, c, color, 0))

    while queue:
        r, c, color, time = queue.popleft()
        if best_time[r, c] != time or color not in owners[r][c]:
            continue

        for nr, nc in _neighbors4(r, c, height, width):
            if not path_mask[nr, nc]:
                continue
            next_time = time + 1
            current = int(best_time[nr, nc])
            if current == -1 or next_time < current:
                best_time[nr, nc] = next_time
                owners[nr][nc] = {color}
                queue.append((nr, nc, color, next_time))
            elif next_time == current and color not in owners[nr][nc]:
                owners[nr][nc].add(color)
                queue.append((nr, nc, color, next_time))

    out = g.copy()
    for r, c in np.argwhere(path_mask):
        rr = int(r)
        cc = int(c)
        assert best_time[rr, cc] >= 0, "all path pixels must be reachable from seeds"
        out[rr, cc] = YELLOW if len(owners[rr][cc]) > 1 else next(iter(owners[rr][cc]))
    return out


def _coarse_neighbors(node: tuple[int, int], rows: int, cols: int):
    r, c = node
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _grow_coarse_tree() -> dict | None:
    rows = random.randint(4, 6)
    cols = random.randint(4, 6)
    target_nodes = random.randint(6, min(rows * cols, 11))

    start = (random.randrange(rows), random.randrange(cols))
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {start: set()}

    while len(adjacency) < target_nodes:
        expandable = [
            node
            for node in adjacency
            if any(nei not in adjacency for nei in _coarse_neighbors(node, rows, cols))
        ]
        if not expandable:
            return None

        weights = [1 + len(adjacency[node]) for node in expandable]
        parent = random.choices(expandable, weights=weights, k=1)[0]
        choices = [
            nei
            for nei in _coarse_neighbors(parent, rows, cols)
            if nei not in adjacency
        ]
        if not choices:
            return None
        child = random.choice(choices)
        adjacency[parent].add(child)
        adjacency[child] = {parent}

    leaves = [node for node, neighbors in adjacency.items() if len(neighbors) == 1]
    hubs = [node for node, neighbors in adjacency.items() if len(neighbors) >= 3]
    if len(leaves) < 3 or not hubs:
        return None

    return {"rows": rows, "cols": cols, "adjacency": adjacency, "leaves": leaves, "hubs": hubs}


def _axis_positions(count: int) -> list[int]:
    positions = [1]
    for _ in range(count - 1):
        positions.append(positions[-1] + random.randint(3, 5))
    return positions


def _weighted_tree(
    adjacency: dict[tuple[int, int], set[tuple[int, int]]],
    row_pos: list[int],
    col_pos: list[int],
) -> dict[tuple[int, int], list[tuple[tuple[int, int], int]]]:
    weighted = {node: [] for node in adjacency}
    for a, neighbors in adjacency.items():
        for b in neighbors:
            if a > b:
                continue
            if a[0] == b[0]:
                weight = abs(col_pos[a[1]] - col_pos[b[1]])
            else:
                weight = abs(row_pos[a[0]] - row_pos[b[0]])
            weighted[a].append((b, int(weight)))
            weighted[b].append((a, int(weight)))
    return weighted


def _branch_leaf_distances(
    weighted: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    hub: tuple[int, int],
) -> dict[tuple[int, int], list[tuple[tuple[int, int], int]]]:
    branches: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {}
    for neighbor, weight in weighted[hub]:
        stack = [(neighbor, hub, int(weight))]
        leaves: list[tuple[tuple[int, int], int]] = []
        while stack:
            node, parent, dist = stack.pop()
            if len(weighted[node]) == 1:
                leaves.append((node, int(dist)))
            for nxt, step in weighted[node]:
                if nxt != parent:
                    stack.append((nxt, node, int(dist + step)))
        branches[neighbor] = leaves
    return branches


def _choose_seed_plan(
    weighted: dict[tuple[int, int], list[tuple[tuple[int, int], int]]]
) -> dict | None:
    leaves = [node for node, neighbors in weighted.items() if len(neighbors) == 1]
    hubs = [node for node, neighbors in weighted.items() if len(neighbors) >= 3]
    mode = random.choices(
        ["hub_tie", "hub_race", "multi_seed"], weights=[4, 4, 2], k=1
    )[0]

    if mode == "multi_seed":
        if len(leaves) < 3:
            return None
        seed_count = random.randint(3, min(5, len(leaves), len(SEED_COLORS)))
        return {"mode": mode, "leaves": random.sample(leaves, seed_count)}

    random.shuffle(hubs)
    for hub in hubs:
        branches = _branch_leaf_distances(weighted, hub)
        branch_items = list(branches.items())
        pairs: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
        for i in range(len(branch_items)):
            branch_a, leaves_a = branch_items[i]
            for j in range(i + 1, len(branch_items)):
                branch_b, leaves_b = branch_items[j]
                for leaf_a, dist_a in leaves_a:
                    for leaf_b, dist_b in leaves_b:
                        if mode == "hub_tie" and dist_a == dist_b:
                            pairs.append((leaf_a, leaf_b, int(dist_a), int(dist_b)))
                        if (
                            mode == "hub_race"
                            and dist_a != dist_b
                            and (dist_a + dist_b) % 2 == 1
                        ):
                            pairs.append((leaf_a, leaf_b, int(dist_a), int(dist_b)))
        if not pairs:
            continue

        leaf_a, leaf_b, dist_a, dist_b = random.choice(pairs)
        if mode == "hub_tie":
            return {"mode": mode, "hub": hub, "leaves": [leaf_a, leaf_b]}

        winner = leaf_a if dist_a < dist_b else leaf_b
        return {
            "mode": mode,
            "hub": hub,
            "leaves": [leaf_a, leaf_b],
            "winner": winner,
        }
    return None


def _draw_tree_grid(
    weighted: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    row_pos: list[int],
    col_pos: list[int],
) -> np.ndarray:
    height = row_pos[-1] + 2
    width = col_pos[-1] + 2
    grid = np.full((height, width), BLACK, dtype=int)

    for node in weighted:
        grid[row_pos[node[0]], col_pos[node[1]]] = GREY

    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for a, neighbors in weighted.items():
        for b, _weight in neighbors:
            edge = tuple(sorted((a, b)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            r0, c0 = row_pos[a[0]], col_pos[a[1]]
            r1, c1 = row_pos[b[0]], col_pos[b[1]]
            if r0 == r1:
                grid[r0, min(c0, c1) : max(c0, c1) + 1] = GREY
            else:
                grid[min(r0, r1) : max(r0, r1) + 1, c0] = GREY

    return grid


def _leaf_seed_cell(
    weighted: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    row_pos: list[int],
    col_pos: list[int],
    leaf: tuple[int, int],
) -> tuple[int, int]:
    neighbor = weighted[leaf][0][0]
    dr = neighbor[0] - leaf[0]
    dc = neighbor[1] - leaf[1]
    return row_pos[leaf[0]] - dr, col_pos[leaf[1]] - dc


def _connected_component_size(mask: np.ndarray, start: tuple[int, int]) -> int:
    height, width = mask.shape
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, height, width):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen)


def _validate_generated_input(grid: np.ndarray, seed_cells: list[tuple[int, int]]) -> bool:
    path_mask = grid == GREY
    if not bool(path_mask.any()):
        return False

    degrees = _path_degrees(path_mask)
    if int(degrees[path_mask].max()) < 3:
        return False

    if int(np.sum(grid == BLACK)) <= int(np.sum(grid != BLACK)):
        return False

    path_points = np.argwhere(path_mask)
    start = tuple(map(int, path_points[0]))
    if _connected_component_size(path_mask, start) != int(path_mask.sum()):
        return False

    if len(seed_cells) < 2:
        return False

    for r, c in seed_cells:
        if int(grid[r, c]) in (BLACK, GREY):
            return False
        neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(int(r), int(c), grid.shape[0], grid.shape[1])
            if path_mask[nr, nc]
        ]
        if len(neighbors) != 1:
            return False
        if int(degrees[neighbors[0]]) != 1:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    coarse = _grow_coarse_tree()
    if coarse is None:
        return None

    row_pos = _axis_positions(int(coarse["rows"]))
    col_pos = _axis_positions(int(coarse["cols"]))
    weighted = _weighted_tree(coarse["adjacency"], row_pos, col_pos)
    plan = _choose_seed_plan(weighted)
    if plan is None:
        return None

    grid = _draw_tree_grid(weighted, row_pos, col_pos)
    chosen_leaves = list(plan["leaves"])
    seed_colors = random.sample(SEED_COLORS, len(chosen_leaves))
    leaf_to_color = dict(zip(chosen_leaves, seed_colors))
    seed_cells: list[tuple[int, int]] = []

    for leaf, color in leaf_to_color.items():
        sr, sc = _leaf_seed_cell(weighted, row_pos, col_pos, leaf)
        if int(grid[sr, sc]) != BLACK:
            return None
        grid[sr, sc] = int(color)
        seed_cells.append((int(sr), int(sc)))

    if not _validate_generated_input(grid, seed_cells):
        return None

    out = main(grid)
    if plan["mode"] == "hub_tie":
        hub_r = row_pos[plan["hub"][0]]
        hub_c = col_pos[plan["hub"][1]]
        if int(out[hub_r, hub_c]) != YELLOW:
            return None
    elif plan["mode"] == "hub_race":
        hub_r = row_pos[plan["hub"][0]]
        hub_c = col_pos[plan["hub"][1]]
        winner_color = int(leaf_to_color[plan["winner"]])
        if int(out[hub_r, hub_c]) != winner_color:
            return None
        if bool(np.any(out == YELLOW)):
            return None
    else:
        path_colors = set(map(int, out[grid == GREY]))
        if len(path_colors) < 2:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid path-wave puzzle")
