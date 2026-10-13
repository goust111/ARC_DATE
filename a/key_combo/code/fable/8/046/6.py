from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GEN_ATTEMPTS = 500
LOGICAL_RADIUS = 6


def _dominant_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ORTHO_DIRS]


def _neighbors_diag(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIAG_DIRS]


def _single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.shape == (1, 2), f"expected exactly one cell of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _bfs(start: tuple[int, int], nodes: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        cell = queue.popleft()
        base = dist[cell]
        for nxt in _neighbors_diag(cell):
            if nxt in nodes and nxt not in dist:
                dist[nxt] = base + 1
                queue.append(nxt)
    return dist


def _race(
    grid: np.ndarray,
    *,
    ignore_insulators: bool = False,
    keep_ties_grey: bool = True,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_bg(g)
    assert bg == BG, "this puzzle expects a black background"

    allowed = {BG, GREY, RED, BLUE}
    present = set(map(int, g.ravel()))
    assert present <= allowed, f"unexpected colors: {sorted(present - allowed)}"

    red = _single_coord(g, RED)
    blue = _single_coord(g, BLUE)
    grey_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == GREY)}
    occupied = {tuple(map(int, cell)) for cell in np.argwhere(g != bg)}
    assert grey_cells, "expected grey blocks"

    for source in (red, blue):
        assert all(n not in occupied for n in _neighbors4(source)), "source blocks may not share edges"

    insulators = {
        cell
        for cell in grey_cells
        if any(neighbor in occupied for neighbor in _neighbors4(cell))
    }
    conductive = set(grey_cells) if ignore_insulators else (grey_cells - insulators)

    nodes = set(conductive) | {red, blue}
    red_dist = _bfs(red, nodes)
    blue_dist = _bfs(blue, nodes)

    out = g.copy()
    tie_cells: set[tuple[int, int]] = set()
    unreachable_conductive: set[tuple[int, int]] = set()
    recolored_red = 0
    recolored_blue = 0
    inf = 10**9

    for cell in conductive:
        dr = red_dist.get(cell, inf)
        db = blue_dist.get(cell, inf)
        r, c = cell
        if dr == inf and db == inf:
            unreachable_conductive.add(cell)
            continue
        if dr < db:
            out[r, c] = RED
            recolored_red += 1
        elif db < dr:
            out[r, c] = BLUE
            recolored_blue += 1
        else:
            tie_cells.add(cell)
            if not keep_ties_grey:
                out[r, c] = RED
                recolored_red += 1

    return out, {
        "red": red,
        "blue": blue,
        "grey_cells": grey_cells,
        "insulators": insulators,
        "conductive": conductive,
        "red_dist": red_dist,
        "blue_dist": blue_dist,
        "tie_cells": tie_cells,
        "unreachable_conductive": unreachable_conductive,
        "recolored_red": recolored_red,
        "recolored_blue": recolored_blue,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the reachable grey blocks by racing red and blue through diagonal-only contact chains.

    Input structure:
    - The grid has a black background, exactly one red source block, exactly one blue source block, and many grey blocks.
    - Grey blocks may touch other blocks only at corners or may share an edge with another block.
    - The two source blocks themselves do not share an edge with any block.

    Rule:
    - Any grey block that shares an edge with any other block is an insulator: it stays grey and cannot conduct further.
    - Starting simultaneously from the red and blue source blocks, propagate only through diagonal contacts between non-insulating grey blocks.
    - Recolor each reachable non-insulating grey block to the nearer source color by chain length; if both sources are equally near, keep that block grey.
    - Insulators and grey blocks unreachable from both sources remain grey.

    Key insight:
    - Connectivity is inverted: corner-touching blocks are the wires, while edge-touching blocks are the breakers, so the answer comes from a two-source shortest-path race on that diagonal graph.
    """
    out, _meta = _race(grid, ignore_insulators=False, keep_ties_grey=True)
    return out


def _logical_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    u, v = cell
    return [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]


def _logical_to_actual(cell: tuple[int, int]) -> tuple[int, int]:
    u, v = cell
    return u + v, u - v


def _can_add_logical(
    candidate: tuple[int, int],
    parent: tuple[int, int],
    occupied: set[tuple[int, int]],
    local: set[tuple[int, int]],
    radius: int,
) -> bool:
    if max(abs(candidate[0]), abs(candidate[1])) > radius:
        return False
    if candidate in occupied or candidate in local:
        return False
    for neighbor in _logical_neighbors(candidate):
        if neighbor == parent:
            continue
        if neighbor in occupied or neighbor in local:
            return False
    return True


def _extend_logical_path(
    start: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    radius: int,
) -> list[tuple[int, int]] | None:
    path: list[tuple[int, int]] = []
    local: set[tuple[int, int]] = set()
    current = start

    for _ in range(length):
        candidates = []
        for candidate in _logical_neighbors(current):
            if not _can_add_logical(candidate, current, occupied, local, radius):
                continue
            score = random.random() + 0.15 * max(abs(candidate[0]), abs(candidate[1]))
            candidates.append((score, candidate))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_band = [cell for _score, cell in candidates[: min(3, len(candidates))]]
        current = random.choice(top_band)
        path.append(current)
        local.add(current)

    return path


def _sample_reachable_branches(
    occupied: set[tuple[int, int]],
    blocked_sources: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    branches: list[list[tuple[int, int]]] = []
    target_count = random.randint(1, 3)

    for _ in range(target_count):
        anchors = [cell for cell in occupied if cell not in blocked_sources]
        random.shuffle(anchors)
        built = False
        for anchor in anchors:
            for _attempt in range(12):
                length = random.randint(1, 3)
                branch = _extend_logical_path(anchor, length, occupied, LOGICAL_RADIUS)
                if branch is None:
                    continue
                branches.append(branch)
                occupied.update(branch)
                built = True
                break
            if built:
                break
        if not built:
            return None

    return branches


def _sample_breaker_spur(
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    anchors = [cell for cell in occupied if cell not in forbidden]
    random.shuffle(anchors)

    for anchor in anchors:
        for _attempt in range(30):
            branch = _extend_logical_path(anchor, random.randint(2, 3), occupied, LOGICAL_RADIUS)
            if branch is None:
                continue

            gate = branch[0]
            actual_occupied = {_logical_to_actual(cell) for cell in occupied | set(branch)}
            gate_actual = _logical_to_actual(gate)
            breaker_options = []
            for neighbor in _neighbors4(gate_actual):
                if neighbor in actual_occupied:
                    continue
                touches = [other for other in _neighbors4(neighbor) if other in actual_occupied]
                if touches == [gate_actual]:
                    breaker_options.append(neighbor)
            if not breaker_options:
                continue
            return branch, random.choice(breaker_options)

    return None


def _sample_isolated_component(
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    seeds = [
        (u, v)
        for u in range(-LOGICAL_RADIUS, LOGICAL_RADIUS + 1)
        for v in range(-LOGICAL_RADIUS, LOGICAL_RADIUS + 1)
        if (u, v) not in occupied and all(n not in occupied for n in _logical_neighbors((u, v)))
    ]
    random.shuffle(seeds)

    for seed in seeds:
        length = random.randint(2, 4)
        branch = _extend_logical_path(seed, length - 1, occupied | {seed}, LOGICAL_RADIUS)
        if branch is None:
            continue
        return [seed] + branch

    return None


def _shifted_grid(
    regular_cells: set[tuple[int, int]],
    red_actual: tuple[int, int],
    blue_actual: tuple[int, int],
    breakers: set[tuple[int, int]],
) -> np.ndarray | None:
    all_cells = set(regular_cells) | breakers | {red_actual, blue_actual}
    rows = [r for r, _c in all_cells]
    cols = [c for _r, c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    base_h = max_r - min_r + 1
    base_w = max_c - min_c + 1
    if base_h > 28 or base_w > 28:
        return None

    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    if base_h + top + bottom > 30 or base_w + left + right > 30:
        return None

    row_shift = top - min_r
    col_shift = left - min_c
    height = base_h + top + bottom
    width = base_w + left + right
    grid = np.full((height, width), BG, dtype=int)

    for r, c in regular_cells | breakers:
        grid[r + row_shift, c + col_shift] = GREY
    rr, rc = red_actual
    br, bc = blue_actual
    grid[rr + row_shift, rc + col_shift] = RED
    grid[br + row_shift, bc + col_shift] = BLUE
    return grid


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.rot90(out, random.randint(0, 3)).copy()


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, meta = _race(grid, ignore_insulators=False, keep_ties_grey=True)
        naive_out, _naive_meta = _race(grid, ignore_insulators=True, keep_ties_grey=True)
        tie_break_out, _tie_meta = _race(grid, ignore_insulators=False, keep_ties_grey=False)
    except AssertionError:
        return False

    grey_mask = grid == GREY
    changed_to_color = grey_mask & ((out == RED) | (out == BLUE))
    blocked_difference = grey_mask & (out == GREY) & ((naive_out == RED) | (naive_out == BLUE))

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, naive_out):
        return False
    if np.array_equal(out, tie_break_out):
        return False
    if int(np.count_nonzero(grid == GREY)) < 12:
        return False
    if meta["recolored_red"] < 3 or meta["recolored_blue"] < 3:
        return False
    if len(meta["tie_cells"]) < 1:
        return False
    if len(meta["insulators"]) < 2:
        return False
    if len(meta["unreachable_conductive"]) < 2:
        return False
    if int(np.count_nonzero(changed_to_color)) < 8:
        return False
    if int(np.count_nonzero(blocked_difference)) < 2:
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    core: set[tuple[int, int]] = {(0, 0)}
    branch_len = random.randint(3, 5)

    red_path = _extend_logical_path((0, 0), branch_len, core, LOGICAL_RADIUS)
    if red_path is None:
        return None
    core.update(red_path)

    blue_path = None
    for _attempt in range(40):
        trial = _extend_logical_path((0, 0), branch_len, core - {(0, 0)}, LOGICAL_RADIUS)
        if trial is None:
            continue
        blue_path = trial
        break
    if blue_path is None:
        return None
    core.update(blue_path)

    red_source = red_path[-1]
    blue_source = blue_path[-1]
    source_cells = {red_source, blue_source}

    branches = _sample_reachable_branches(core, source_cells)
    if branches is None:
        return None

    breaker_result = _sample_breaker_spur(core, source_cells | {(0, 0)})
    if breaker_result is None:
        return None
    breaker_branch, breaker_cell = breaker_result
    core.update(breaker_branch)

    isolated = _sample_isolated_component(core)
    if isolated is None:
        return None
    core.update(isolated)

    actual_regular = {_logical_to_actual(cell) for cell in core}
    red_actual = _logical_to_actual(red_source)
    blue_actual = _logical_to_actual(blue_source)
    grid = _shifted_grid(actual_regular, red_actual, blue_actual, {breaker_cell})
    if grid is None:
        return None

    grid = _random_symmetry(grid)
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-wire puzzle input")
