from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SIGNAL_COLORS = {RED, GREEN, BLUE}
PATH_COLORS = {GREY, RED, GREEN, BLUE}
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_path_value(value: int) -> bool:
    return value in PATH_COLORS


def _yellow_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)]
    assert len(pts) == 2, "expected exactly two yellow edge markers"
    h, w = grid.shape
    assert all(r in (0, h - 1) or c in (0, w - 1) for r, c in pts), "yellow markers must sit on the border"
    return pts


def _pick_start_marker(grid: np.ndarray) -> tuple[int, int]:
    pts = _yellow_positions(grid)
    h, w = grid.shape
    upper_left = [(r, c) for r, c in pts if r == 0 or c == 0]
    lower_right = [(r, c) for r, c in pts if r == h - 1 or c == w - 1]
    if len(upper_left) == 1 and len(lower_right) == 1:
        return upper_left[0]
    return min(pts)


def _trace_segment(grid: np.ndarray, origin: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = origin[0] + dr, origin[1] + dc
    removed: list[tuple[int, int]] = []

    while _in_bounds((h, w), r, c) and int(grid[r, c]) == GREY:
        side_branch = False
        for odr, odc in ((dc, dr), (-dc, -dr)):
            nr, nc = r + odr, c + odc
            if _in_bounds((h, w), nr, nc) and _is_path_value(int(grid[nr, nc])):
                side_branch = True
                break
        if side_branch:
            break

        removed.append((r, c))
        nr, nc = r + dr, c + dc
        if not (_in_bounds((h, w), nr, nc) and _is_path_value(int(grid[nr, nc]))):
            break
        r, c = nr, nc

    return removed


def _apply_signal_closures(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    removed: set[tuple[int, int]] = set()

    for r, c in np.argwhere(np.isin(g, [RED, BLUE, GREEN])):
        rr, cc = int(r), int(c)
        color = int(g[rr, cc])
        if color == GREEN:
            continue
        directions = [(0, -1), (0, 1)] if color == RED else [(-1, 0), (1, 0)]
        for dr, dc in directions:
            removed.update(_trace_segment(g, (rr, cc), dr, dc))

    for r, c in removed:
        out[r, c] = BLACK
    return out


def _crop_to_start_component(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    start = _pick_start_marker(g)
    h, w = g.shape

    reachable: set[tuple[int, int]] = set()
    q: deque[tuple[int, int]] = deque()
    for dr, dc in DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if _in_bounds((h, w), nr, nc) and _is_path_value(int(g[nr, nc])):
            q.append((nr, nc))
            reachable.add((nr, nc))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not _in_bounds((h, w), nr, nc):
                continue
            if (nr, nc) in reachable:
                continue
            if not _is_path_value(int(g[nr, nc])):
                continue
            reachable.add((nr, nc))
            q.append((nr, nc))

    mask = np.isin(out, list(PATH_COLORS))
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if (rr, cc) not in reachable:
            out[rr, cc] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Selectively remove path segments according to signal-light colors, then keep
    only the portion connected to the start marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _yellow_positions(g)
    after_lights = _apply_signal_closures(g)
    return _crop_to_start_component(after_lights)


# --------------- generator ---------------

GEN_H_RANGE = (17, 27)
GEN_W_RANGE = (17, 27)
GEN_BRANCH_COUNT_CHOICES = [3, 4]
GEN_BRANCH_LEN_RANGE = (1, 3)
GEN_MAX_ATTEMPTS = 500


def _axis_coords(size: int) -> list[int]:
    coords = list(range(2, size - 2, 4))
    assert len(coords) >= 4
    return coords


def _lattice_neighbors(node: tuple[int, int], nr: int, nc: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < nr:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < nc:
        out.append((r, c + 1))
    return out


def _search_path(
    start: tuple[int, int],
    end: tuple[int, int],
    nr: int,
    nc: int,
    min_len: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    path = [start]
    seen = {start}

    def dfs(cur: tuple[int, int]) -> bool:
        if cur == end:
            return len(path) >= min_len

        choices = [nxt for nxt in _lattice_neighbors(cur, nr, nc) if nxt not in seen]
        rng.shuffle(choices)
        if len(path) < min_len - 1:
            choices.sort(key=lambda p: abs(p[0] - end[0]) + abs(p[1] - end[1]), reverse=True)

        for nxt in choices:
            seen.add(nxt)
            path.append(nxt)
            if dfs(nxt):
                return True
            path.pop()
            seen.remove(nxt)
        return False

    return path.copy() if dfs(start) else None


def _path_to_edges(path: list[tuple[int, int]]) -> set[frozenset[tuple[int, int]]]:
    return {frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)}


def _unused_branch_root_candidates(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    nr: int,
    nc: int,
) -> list[tuple[int, int]]:
    return [
        node
        for node in path[1:-1]
        if any(nbr not in used for nbr in _lattice_neighbors(node, nr, nc))
    ]


def _grow_branch(
    root: tuple[int, int],
    used: set[tuple[int, int]],
    nr: int,
    nc: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    options = [nbr for nbr in _lattice_neighbors(root, nr, nc) if nbr not in used]
    if not options:
        return None

    cur = rng.choice(options)
    branch = [root, cur]
    used.add(cur)
    target_len = rng.randint(GEN_BRANCH_LEN_RANGE[0], GEN_BRANCH_LEN_RANGE[1])

    while len(branch) - 1 < target_len:
        nxt_options = [nbr for nbr in _lattice_neighbors(cur, nr, nc) if nbr not in used]
        if not nxt_options:
            break
        cur = rng.choice(nxt_options)
        branch.append(cur)
        used.add(cur)

    return branch


def _node_to_cell(node: tuple[int, int], row_coords: list[int], col_coords: list[int]) -> tuple[int, int]:
    return row_coords[node[0]], col_coords[node[1]]


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = color
    else:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = color


def _attach_endpoint(
    grid: np.ndarray,
    node: tuple[int, int],
    side: str,
    row_coords: list[int],
    col_coords: list[int],
) -> tuple[int, int]:
    h, w = grid.shape
    nr, nc = _node_to_cell(node, row_coords, col_coords)
    if side == "top":
        endpoint = (1, nc)
        yellow = (0, nc)
    elif side == "bottom":
        endpoint = (h - 2, nc)
        yellow = (h - 1, nc)
    elif side == "left":
        endpoint = (nr, 1)
        yellow = (nr, 0)
    else:
        endpoint = (nr, w - 2)
        yellow = (nr, w - 1)
    _draw_segment(grid, (nr, nc), endpoint, GREY)
    grid[yellow] = YELLOW
    return yellow


def _build_input(
    h: int,
    w: int,
    row_coords: list[int],
    col_coords: list[int],
    edges: set[frozenset[tuple[int, int]]],
    signals: dict[tuple[int, int], int],
    start_node: tuple[int, int],
    start_side: str,
    end_node: tuple[int, int],
    end_side: str,
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for edge in edges:
        a, b = tuple(edge)
        _draw_segment(grid, _node_to_cell(a, row_coords, col_coords), _node_to_cell(b, row_coords, col_coords), GREY)
    _attach_endpoint(grid, start_node, start_side, row_coords, col_coords)
    _attach_endpoint(grid, end_node, end_side, row_coords, col_coords)
    for node, color in signals.items():
        r, c = _node_to_cell(node, row_coords, col_coords)
        grid[r, c] = color
    return grid


def _signal_roots(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    nr: int,
    nc: int,
    root_count: int,
    rng: random.Random,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    node_to_idx = {node: idx for idx, node in enumerate(path)}
    branches: list[list[tuple[int, int]]] = []
    roots: list[tuple[int, int]] = []

    while len(roots) < root_count:
        candidates = _unused_branch_root_candidates(path, used, nr, nc)
        rng.shuffle(candidates)
        picked = None
        for node in candidates:
            idx = node_to_idx[node]
            if any(abs(idx - node_to_idx[root]) < 2 for root in roots):
                continue
            branch = _grow_branch(node, used, nr, nc, rng)
            if branch is None:
                continue
            branches.append(branch)
            roots.append(node)
            picked = node
            break
        if picked is None:
            return None
    return branches, roots


def _endpoint_side_nodes(nr: int, nc: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(0, c) for c in range(1, nc - 1)],
        "bottom": [(nr - 1, c) for c in range(1, nc - 1)],
        "left": [(r, 0) for r in range(1, nr - 1)],
        "right": [(r, nc - 1) for r in range(1, nr - 1)],
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GEN_H_RANGE)
    w = rng.randint(*GEN_W_RANGE)
    row_coords = _axis_coords(h)
    col_coords = _axis_coords(w)
    nr, nc = len(row_coords), len(col_coords)

    sides = _endpoint_side_nodes(nr, nc)
    start_side = rng.choice(["top", "left"])
    end_side = rng.choice(["bottom", "right"])
    start_node = rng.choice(sides[start_side])
    end_node = rng.choice(sides[end_side])

    root_count = rng.choice(GEN_BRANCH_COUNT_CHOICES)
    min_path_len = root_count + 4
    path = _search_path(start_node, end_node, nr, nc, min_path_len, rng)
    if path is None:
        return None

    used = set(path)
    branch_info = _signal_roots(path, used, nr, nc, root_count, rng)
    if branch_info is None:
        return None
    branches, roots = branch_info

    edges = _path_to_edges(path)
    for branch in branches:
        edges.update(_path_to_edges(branch))

    root_colors = [RED, BLUE, GREEN]
    rng.shuffle(root_colors)
    while len(root_colors) < len(roots):
        root_colors.append(rng.choice([RED, BLUE, GREEN]))
    signals = {node: color for node, color in zip(roots, root_colors)}

    grid = _build_input(
        h=h,
        w=w,
        row_coords=row_coords,
        col_coords=col_coords,
        edges=edges,
        signals=signals,
        start_node=start_node,
        start_side=start_side,
        end_node=end_node,
        end_side=end_side,
    )

    if np.count_nonzero(np.isin(grid, [RED, BLUE, GREEN])) < 3:
        return None

    mid = _apply_signal_closures(grid)
    out = _crop_to_start_component(mid)

    path_before = int(np.count_nonzero(np.isin(grid, list(PATH_COLORS))))
    path_mid = int(np.count_nonzero(np.isin(mid, list(PATH_COLORS))))
    path_after = int(np.count_nonzero(np.isin(out, list(PATH_COLORS))))

    if path_after < 10:
        return None
    if path_before - path_mid < 4:
        return None
    if path_mid - path_after < 4:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
