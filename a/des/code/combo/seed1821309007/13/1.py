import random
from collections import deque

import numpy as np


BG = 0
FG = 5

RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {(-1, 0): (1, 0), (1, 0): (-1, 0), (0, -1): (0, 1), (0, 1): (0, -1)}
LENGTH_TO_COLOR = {
    1: RED,
    2: BLUE,
    3: GREEN,
    4: YELLOW,
}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _fg_mask(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid, dtype=int) != BG


def _degree_map(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            total = 0
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if _in_bounds(h, w, rr, cc) and mask[rr, cc]:
                    total += 1
            deg[r, c] = total
    return deg


def _path_color(length: int) -> int:
    return LENGTH_TO_COLOR.get(length, ORANGE)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = _fg_mask(g)
    deg = _degree_map(mask)
    nodes = mask & (deg != 2)

    out = np.where(mask, FG, BG).astype(int)
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    h, w = g.shape

    for r, c in np.argwhere(nodes):
        start = (int(r), int(c))
        for dr, dc in DIRS4:
            rr, cc = start[0] + dr, start[1] + dc
            if not _in_bounds(h, w, rr, cc) or not mask[rr, cc]:
                continue
            if nodes[rr, cc]:
                continue

            edge_key = tuple(sorted((start, (int(rr), int(cc)))))
            if edge_key in seen_edges:
                continue

            path = []
            prev = start
            cur = (int(rr), int(cc))

            while True:
                path.append(cur)
                seen_edges.add(tuple(sorted((prev, cur))))
                nbrs = []
                for ddr, ddc in DIRS4:
                    nr, nc = cur[0] + ddr, cur[1] + ddc
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and (nr, nc) != prev:
                        nbrs.append((int(nr), int(nc)))

                if nodes[cur]:
                    break
                if not nbrs:
                    break
                assert len(nbrs) == 1, "ordinary path pixel must have one forward continuation"
                nxt = nbrs[0]
                prev, cur = cur, nxt
                if nodes[cur]:
                    seen_edges.add(tuple(sorted((prev, cur))))
                    break

            if nodes[cur] and path:
                color = _path_color(len(path))
                for pr, pc in path:
                    out[pr, pc] = color

    return out


GRID_MIN = 15
GRID_MAX = 27
EDGE_COUNT_MIN = 4
EDGE_COUNT_MAX = 9
GEN_ATTEMPTS = 400
MAX_EDGE_TRIES = 80
MARGIN = 2
LENGTH_BUCKETS = {
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5, 6, 7],
}


def _choose_length_sequence(edge_count: int) -> list[int]:
    bins = [1, 2, 3, 4, 5]
    chosen_bins = bins.copy()
    while len(chosen_bins) < edge_count:
        chosen_bins.append(random.choice(bins))
    random.shuffle(chosen_bins)
    return [random.choice(LENGTH_BUCKETS[b]) for b in chosen_bins[:edge_count]]


def _add_edge(
    grid: np.ndarray,
    degrees: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
    direction: tuple[int, int],
    ordinary_len: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    dr, dc = direction
    end = (start[0] + dr * (ordinary_len + 1), start[1] + dc * (ordinary_len + 1))

    if not _in_bounds(h, w, end[0], end[1]):
        return None
    if not (MARGIN <= end[0] < h - MARGIN and MARGIN <= end[1] < w - MARGIN):
        return None
    if end in degrees:
        return None

    segment = []
    for step in range(1, ordinary_len + 2):
        rr = start[0] + dr * step
        cc = start[1] + dc * step
        if not _in_bounds(h, w, rr, cc):
            return None
        if grid[rr, cc] != BG:
            return None
        segment.append((rr, cc))

    for rr, cc in segment[:-1]:
        for sdr, sdc in DIRS4:
            nr, nc = rr + sdr, cc + sdc
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) == start or (nr, nc) == end:
                continue
            if grid[nr, nc] != BG:
                return None

    for rr, cc in segment:
        grid[rr, cc] = FG

    degrees.setdefault(start, set()).add(direction)
    degrees[end] = {OPPOSITE[direction]}
    return end


def _realized_edge_count(degrees: dict[tuple[int, int], set[tuple[int, int]]]) -> int:
    return sum(len(v) for v in degrees.values()) // 2


def _intended_node_count(degrees: dict[tuple[int, int], set[tuple[int, int]]]) -> int:
    return sum(1 for dirs in degrees.values() if len(dirs) != 2)


def _grow_tree(grid: np.ndarray, target_edges: int, lengths: list[int]) -> bool:
    h, w = grid.shape
    root = (random.randint(MARGIN + 2, h - MARGIN - 3), random.randint(MARGIN + 2, w - MARGIN - 3))
    grid[root] = FG
    degrees: dict[tuple[int, int], set[tuple[int, int]]] = {root: set()}
    leaves = [root]
    length_queue = deque(lengths)

    root_degree = 3 if target_edges < 5 else random.choice([3, 4])
    root_dirs = random.sample(DIRS4, root_degree)
    for direction in root_dirs:
        if not length_queue:
            return False
        endpoint = _add_edge(grid, degrees, root, direction, length_queue.popleft())
        if endpoint is None:
            return False
        leaves.append(endpoint)

    leaves.remove(root)

    while _realized_edge_count(degrees) < target_edges and leaves:
        expandable = [leaf for leaf in leaves if len(degrees[leaf]) == 1]
        if not expandable:
            break

        node = random.choice(expandable)
        back_dir = next(iter(degrees[node]))
        available = [d for d in DIRS4 if d != back_dir]
        max_new = min(3, len(available), target_edges - _realized_edge_count(degrees))
        if max_new < 2:
            break
        new_count = 2 if max_new == 2 else random.choice([2, 3])

        success = False
        for _ in range(MAX_EDGE_TRIES):
            trial_dirs = random.sample(available, new_count)
            snapshot = grid.copy()
            deg_snapshot = {k: v.copy() for k, v in degrees.items()}
            new_leaves = []
            trial_lengths = []
            if len(length_queue) < new_count:
                break
            for _idx in range(new_count):
                trial_lengths.append(length_queue.popleft())

            ok = True
            for direction, length in zip(trial_dirs, trial_lengths):
                endpoint = _add_edge(grid, degrees, node, direction, length)
                if endpoint is None:
                    ok = False
                    break
                new_leaves.append(endpoint)

            if ok:
                leaves.remove(node)
                leaves.extend(new_leaves)
                success = True
                break

            grid[:, :] = snapshot
            degrees.clear()
            degrees.update({k: v.copy() for k, v in deg_snapshot.items()})
            for length in reversed(trial_lengths):
                length_queue.appendleft(length)

        if not success:
            break

    return _realized_edge_count(degrees) == target_edges and _intended_node_count(degrees) >= 3


def _has_cycle(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not coords:
        return False
    coord_set = set(coords)
    seen = set()
    parent = {}

    for start in coords:
        if start in seen:
            continue
        stack = [start]
        parent[start] = None
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for dr, dc in DIRS4:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in coord_set:
                    continue
                if parent.get(cur) == nxt:
                    continue
                if nxt in seen:
                    return True
                parent[nxt] = cur
                stack.append(nxt)
    return False


def _validate_input(grid: np.ndarray) -> bool:
    mask = grid == FG
    if not np.any(mask):
        return False
    deg = _degree_map(mask)
    fg_deg = deg[mask]
    if np.any(fg_deg == 0):
        return False

    node_mask = mask & (deg != 2)
    ordinary_mask = mask & (deg == 2)
    if np.count_nonzero(node_mask) < 3 or np.count_nonzero(ordinary_mask) == 0:
        return False

    for r, c in np.argwhere(node_mask):
        for dr, dc in DIRS4:
            rr, cc = int(r) + dr, int(c) + dc
            if _in_bounds(*grid.shape, rr, cc) and node_mask[rr, cc]:
                return False

    if _has_cycle(mask):
        return False

    out = main(grid)
    recolored = out[ordinary_mask]
    if np.any(recolored == FG) or np.any(recolored == BG):
        return False

    present_colors = set(map(int, recolored.tolist()))
    if len(present_colors) < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    edge_count = random.randint(EDGE_COUNT_MIN, EDGE_COUNT_MAX)
    lengths = _choose_length_sequence(edge_count)

    grid = np.full((h, w), BG, dtype=int)
    if not _grow_tree(grid, edge_count, lengths):
        return None
    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
