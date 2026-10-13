from __future__ import annotations

from collections import deque
from functools import lru_cache
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
WALL = 6
FLOOR = 7
ENDPOINT_COLORS = (RED, BLUE, GREEN)
STAMP_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    seen = set()
    q = deque([tuple(map(int, coords[0]))])
    seen.add(q[0])
    out = []
    while q:
        cell = q.popleft()
        out.append(cell)
        for nr, nc in _neighbors4(*cell):
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    assert len(seen) == int(mask.sum())
    return np.array(sorted(out), dtype=int)


def _endpoint_info(grid: np.ndarray) -> dict[int, dict]:
    info: dict[int, dict] = {}
    for color in ENDPOINT_COLORS:
        mask = grid == color
        assert int(mask.sum()) == 4, f"endpoint color {color} must occupy exactly four cells"
        coords = _component(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (2, 2), f"endpoint color {color} must be a 2x2 block"
        info[color] = {
            "color": int(color),
            "coords": [tuple(map(int, rc)) for rc in coords],
            "anchor": (int(r0), int(c0)),
        }
    return info


def _bfs(passable: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in sources:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _best_shortest_path(
    passable: np.ndarray,
    start_cells: list[tuple[int, int]],
    end_cells: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], np.ndarray, int]:
    dist_a = _bfs(passable, start_cells)
    dist_b = _bfs(passable, end_cells)
    best_len = min(int(dist_a[r, c]) for r, c in end_cells if dist_a[r, c] >= 0)
    assert best_len >= 0
    shortest_mask = (dist_a >= 0) & (dist_b >= 0) & (dist_a + dist_b == best_len)
    end_set = {cell for cell in end_cells if int(dist_a[cell]) == best_len}
    start_set = {cell for cell in start_cells if shortest_mask[cell]}

    @lru_cache(maxsize=None)
    def best_from(cell: tuple[int, int]):
        r, c = cell
        if cell in end_set:
            return (r, c, (cell,))
        candidates = []
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(passable.shape, nr, nc):
                continue
            if not shortest_mask[nr, nc]:
                continue
            if int(dist_a[nr, nc]) != int(dist_a[r, c]) + 1:
                continue
            child_sum_r, child_sum_c, child_seq = best_from((nr, nc))
            candidates.append((r + child_sum_r, c + child_sum_c, (cell,) + child_seq))
        assert candidates
        return min(candidates, key=lambda item: (item[0], item[1], item[2]))

    scored = [best_from(cell) for cell in sorted(start_set)]
    sum_r, sum_c, seq = min(scored, key=lambda item: (item[0], item[1], item[2]))
    assert len(seq) == best_len + 1
    return list(seq), shortest_mask, best_len


def _solve_details(grid: np.ndarray) -> dict:
    endpoints = _endpoint_info(grid)
    passable = grid != WALL
    pair_specs = [
        (RED, BLUE),
        (RED, GREEN),
        (BLUE, GREEN),
    ]

    pair_data = {}
    for a, b in pair_specs:
        seq, shortest_mask, best_len = _best_shortest_path(
            passable,
            endpoints[a]["coords"],
            endpoints[b]["coords"],
        )
        pair_data[(a, b)] = {
            "colors": (a, b),
            "sequence": seq,
            "sequence_set": set(seq),
            "floor_sequence": [cell for cell in seq if int(grid[cell]) == FLOOR],
            "floor_set": {cell for cell in seq if int(grid[cell]) == FLOOR},
            "start_anchor": endpoints[a]["anchor"],
            "end_anchor": endpoints[b]["anchor"],
            "shortest_mask": shortest_mask,
            "shortest_len": int(best_len),
        }
    return {"endpoints": endpoints, "pairs": pair_data}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint three pairwise maze shortest paths with endpoint-distance colors, then stamp yellow crosses on pairwise crossings.

    Input structure:
    - The grid uses orange floor cells and magenta wall cells to form a 4-connected corridor maze.
    - Exactly three colored endpoint components appear: one red 2x2 block, one blue 2x2 block, and one green 2x2 block.
    - The endpoint blocks sit in different maze regions but remain connected through the floor corridors.

    Rule:
    - For each endpoint pair (red-blue, red-green, blue-green), find a shortest 4-connected path through all non-wall cells.
    - If multiple shortest paths exist for one pair, choose the one with the smallest total path-coordinate bias toward upper-left cells.
    - Recolor each floor cell belonging to exactly one chosen path with the color of the nearer endpoint along that path; exact midpoints use the upper-left endpoint block as the tie-breaker.
    - Any floor cell used by exactly two chosen paths becomes the center of a yellow plus stamp covering the center and its four orthogonal neighbors, but only where those stamped cells are floor in the input.
    - Endpoint blocks keep their original colors, untouched floor stays orange, and walls stay magenta.

    Key insight:
    - The transformation depends on computing all three selected shortest paths first: endpoint-distance coloring is path-specific, while the yellow stamps appear only at overlaps between different pair paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    details = _solve_details(g)
    out = g.copy()

    membership: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pair_key, pair in details["pairs"].items():
        for cell in pair["floor_set"]:
            membership.setdefault(cell, []).append(pair_key)

    for pair_key, pair in details["pairs"].items():
        a, b = pair["colors"]
        anchor_a = pair["start_anchor"]
        anchor_b = pair["end_anchor"]
        seq = pair["sequence"]
        total = len(seq) - 1
        for idx, cell in enumerate(seq):
            if int(g[cell]) != FLOOR:
                continue
            if len(membership[cell]) != 1:
                continue
            dist_a = idx
            dist_b = total - idx
            if dist_a < dist_b:
                color = a
            elif dist_b < dist_a:
                color = b
            else:
                color = a if anchor_a <= anchor_b else b
            out[cell] = int(color)

    centers = [cell for cell, owners in membership.items() if len(owners) == 2]
    for r, c in centers:
        for dr, dc in STAMP_DELTAS:
            rr = r + dr
            cc = c + dc
            if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == FLOOR:
                out[rr, cc] = YELLOW

    return out


def _carve_h(grid: np.ndarray, row: int, c0: int, c1: int) -> None:
    lo, hi = sorted((c0, c1))
    grid[row, lo : hi + 1] = FLOOR


def _carve_v(grid: np.ndarray, col: int, r0: int, r1: int) -> None:
    lo, hi = sorted((r0, r1))
    grid[lo : hi + 1, col] = FLOOR


def _apply_transform(grid: np.ndarray, code: int) -> np.ndarray:
    g = np.array(grid, copy=True)
    rot = code % 4
    flip = code // 4
    if flip:
        g = np.fliplr(g)
    return np.rot90(g, rot).copy()


def _path_supports_tie(pair: dict) -> bool:
    return int(pair["shortest_mask"].sum()) > len(pair["sequence"])


def _build_candidate() -> np.ndarray:
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    trunk_len = random.randint(4, 10)
    branch_dx = random.randint(2, 4)
    up_len = random.randint(branch_dx + 1, branch_dx + 6)
    down_len = random.randint(branch_dx + 1, branch_dx + 6)
    conn_root = random.randint(0, 1)
    conn_top = random.randint(0, 1)
    conn_bottom = random.randint(0, 1)

    top_block_row = top_margin
    top_path_row = top_block_row + conn_top
    junction_row = top_path_row + up_len
    bottom_path_row = junction_row + down_len
    bottom_block_row = bottom_path_row - conn_bottom
    root_block_row = junction_row - conn_root

    root_block_col = left_margin
    junction_col = root_block_col + trunk_len + 1
    direct_col = junction_col + branch_dx
    right_block_col = direct_col + 1

    h = bottom_block_row + 2 + bottom_margin
    w = right_block_col + 2 + right_margin
    if max(h, w) > 30:
        raise ValueError("candidate exceeded ARC bounds")

    grid = np.full((h, w), WALL, dtype=int)

    _carve_h(grid, junction_row, root_block_col + 2, junction_col)
    _carve_v(grid, junction_col, top_path_row, junction_row)
    _carve_v(grid, junction_col, junction_row, bottom_path_row)
    _carve_h(grid, top_path_row, junction_col, direct_col)
    _carve_h(grid, bottom_path_row, junction_col, direct_col)
    _carve_v(grid, direct_col, top_path_row, bottom_path_row)

    ladder_kind = random.choice(("trunk", "top", "bottom", "direct"))
    if ladder_kind == "trunk":
        a = random.randint(root_block_col + 3, junction_col - 2)
        b = random.randint(a + 1, junction_col - 1)
        parallel_row = junction_row + random.choice((-1, 1))
        _carve_h(grid, parallel_row, a, b)
        _carve_v(grid, a, min(junction_row, parallel_row), max(junction_row, parallel_row))
        _carve_v(grid, b, min(junction_row, parallel_row), max(junction_row, parallel_row))
    elif ladder_kind == "top":
        a = random.randint(junction_col + 1, direct_col - 2)
        b = random.randint(a + 1, direct_col - 1)
        parallel_row = top_path_row + random.choice((-1, 1))
        _carve_h(grid, parallel_row, a, b)
        _carve_v(grid, a, min(top_path_row, parallel_row), max(top_path_row, parallel_row))
        _carve_v(grid, b, min(top_path_row, parallel_row), max(top_path_row, parallel_row))
    elif ladder_kind == "bottom":
        a = random.randint(junction_col + 1, direct_col - 2)
        b = random.randint(a + 1, direct_col - 1)
        parallel_row = bottom_path_row + random.choice((-1, 1))
        _carve_h(grid, parallel_row, a, b)
        _carve_v(grid, a, min(bottom_path_row, parallel_row), max(bottom_path_row, parallel_row))
        _carve_v(grid, b, min(bottom_path_row, parallel_row), max(bottom_path_row, parallel_row))
    else:
        a = random.randint(top_path_row + 1, bottom_path_row - 2)
        b = random.randint(a + 1, bottom_path_row - 1)
        parallel_col = direct_col + random.choice((-1, 1))
        _carve_v(grid, parallel_col, a, b)
        _carve_h(grid, a, min(direct_col, parallel_col), max(direct_col, parallel_col))
        _carve_h(grid, b, min(direct_col, parallel_col), max(direct_col, parallel_col))

    # Small dead-end stubs keep some floor cells outside the chosen path union.
    stub_budget = random.randint(1, 3)
    stub_options = []
    if top_path_row - 2 >= 0:
        stub_options.append(("v", junction_col - 1, top_path_row - 2, top_path_row - 1))
    if bottom_path_row + 2 < h:
        stub_options.append(("v", junction_col + 1, bottom_path_row + 1, bottom_path_row + 2))
    if direct_col + 2 < w:
        stub_options.append(("h", top_path_row + 1, direct_col + 1, direct_col + 2))
    if root_block_col + 3 < junction_col:
        stub_options.append(("h", junction_row - 1, root_block_col + 2, root_block_col + 3))
    random.shuffle(stub_options)
    for kind, fixed, a, b in stub_options[:stub_budget]:
        if kind == "v":
            _carve_v(grid, fixed, a, b)
        else:
            _carve_h(grid, fixed, a, b)

    roles = list(ENDPOINT_COLORS)
    random.shuffle(roles)
    root_color, top_color, bottom_color = roles

    grid[root_block_row : root_block_row + 2, root_block_col : root_block_col + 2] = root_color
    grid[top_block_row : top_block_row + 2, right_block_col : right_block_col + 2] = top_color
    grid[bottom_block_row : bottom_block_row + 2, right_block_col : right_block_col + 2] = bottom_color

    transformed = _apply_transform(grid, random.randrange(8))
    return transformed


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        details = _solve_details(grid)
    except Exception:
        return False

    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if min(grid.shape) < 16:
        return False

    floor_mask = grid == FLOOR
    used = np.zeros_like(grid, dtype=int)
    tie_pairs = 0
    for pair in details["pairs"].values():
        tie_pairs += int(_path_supports_tie(pair))
        for cell in pair["floor_set"]:
            used[cell] += 1

    pairwise = int(np.count_nonzero(used == 2))
    triple = int(np.count_nonzero(used >= 3))
    unused_floor = int(np.count_nonzero(floor_mask & (used == 0)))
    single_use = int(np.count_nonzero(used == 1))

    if pairwise < 1 or triple != 0:
        return False
    if tie_pairs < 1:
        return False
    if unused_floor < 2:
        return False
    if single_use < 6:
        return False

    # The shared segment should be substantial but not dominate the whole maze.
    path_sizes = sorted(len(pair["floor_set"]) for pair in details["pairs"].values())
    if path_sizes[0] < 4 or path_sizes[-1] > 40:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _build_candidate()
        except Exception:
            continue
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
