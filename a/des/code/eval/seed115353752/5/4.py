from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
SOURCE_COL = 1

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SOURCE_COLOR_POOL = [1, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if _inside(shape, nr, nc):
            out.append((nr, nc))
    return out


def _adjacent_to_color(grid: np.ndarray, coord: tuple[int, int], color: int) -> bool:
    r, c = coord
    return any(int(grid[nr, nc]) == int(color) for nr, nc in _neighbors(grid.shape, r, c))


def _collect_sources(grid: np.ndarray) -> list[tuple[int, int, int]]:
    special = {BG, RED, GREEN, GREY}
    sources: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(~np.isin(grid, list(special))):
        rr, cc = int(r), int(c)
        if _adjacent_to_color(grid, (rr, cc), GREY):
            sources.append((rr, cc, int(grid[rr, cc])))
    sources.sort(key=lambda item: (item[0], item[1], item[2]))
    assert sources, "expected at least one colored source pixel"
    return sources


def _collect_bulbs(grid: np.ndarray) -> list[tuple[int, int]]:
    bulbs: list[tuple[int, int]] = []
    for r, c in np.argwhere(grid == GREEN):
        rr, cc = int(r), int(c)
        if _adjacent_to_color(grid, (rr, cc), GREY):
            bulbs.append((rr, cc))
    bulbs.sort()
    assert bulbs, "expected at least one connected bulb"
    return bulbs


def _switch_edges(grid: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    edges: dict[tuple[int, int], set[tuple[int, int]]] = {}
    h, w = grid.shape
    for r, c in np.argwhere(grid == RED):
        rr, cc = int(r), int(c)
        grey_neighbors = {(nr, nc) for nr, nc in _neighbors((h, w), rr, cc) if int(grid[nr, nc]) == GREY}
        assert len(grey_neighbors) == 2, "each switch center must touch exactly two grey endpoints"

        horizontal = {(rr, cc - 1), (rr, cc + 1)}
        vertical = {(rr - 1, cc), (rr + 1, cc)}
        if grey_neighbors == horizontal:
            left, right = (rr, cc - 1), (rr, cc + 1)
            edges.setdefault(left, set()).add(right)
            edges.setdefault(right, set()).add(left)
        else:
            assert grey_neighbors == vertical, "switches must be straight 3-cell bars"
    return edges


def _reachable_bulbs(
    grid: np.ndarray,
    source: tuple[int, int, int],
    bulbs: set[tuple[int, int]],
    switch_edges: dict[tuple[int, int], set[tuple[int, int]]],
) -> set[tuple[int, int]]:
    start = (int(source[0]), int(source[1]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    reached: set[tuple[int, int]] = set()

    while q:
        r, c = q.popleft()
        cell = int(grid[r, c])
        if (r, c) in bulbs:
            reached.add((r, c))
            continue

        nxt: list[tuple[int, int]] = []
        if cell == GREY:
            for nr, nc in _neighbors(grid.shape, r, c):
                value = int(grid[nr, nc])
                if value in {GREY, GREEN}:
                    nxt.append((nr, nc))
            nxt.extend(sorted(switch_edges.get((r, c), ())))
        else:
            assert cell not in {BG, RED, GREEN, GREY}, "search may only start on source pixels"
            for nr, nc in _neighbors(grid.shape, r, c):
                if int(grid[nr, nc]) == GREY:
                    nxt.append((nr, nc))

        for node in nxt:
            if node not in seen:
                seen.add(node)
                q.append(node)

    return reached


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow colored source signals through a grey wire network.

    Horizontal red-centered switches bridge their two grey endpoints, while
    vertical red-centered switches block propagation. Reached green bulbs are
    recolored by the highest source that can reach them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sources = _collect_sources(g)
    bulbs = _collect_bulbs(g)
    switch_edges = _switch_edges(g)
    bulb_set = set(bulbs)

    out = g.copy()
    colored: set[tuple[int, int]] = set()
    for source in sources:
        for bulb in sorted(_reachable_bulbs(g, source, bulb_set, switch_edges)):
            if bulb not in colored:
                out[bulb] = int(source[2])
                colored.add(bulb)
    return out


def _make_groups(n_sources: int, shared_mode: bool, rng: random.Random) -> list[list[int]]:
    if not shared_mode:
        return [[i] for i in range(n_sources)]

    shared_size = 3 if n_sources >= 5 and rng.random() < 0.35 else 2
    shared_size = min(shared_size, n_sources - 1)
    start = rng.randint(0, n_sources - shared_size)

    groups: list[list[int]] = []
    i = 0
    while i < n_sources:
        if i == start:
            groups.append(list(range(i, i + shared_size)))
            i += shared_size
        else:
            groups.append([i])
            i += 1
    return groups


def _sample_states(groups: list[list[int]], rng: random.Random) -> list[bool]:
    n_sources = sum(len(group) for group in groups)
    shared_group = max(groups, key=len)
    shared_mode = len(shared_group) > 1

    if not shared_mode:
        while True:
            states = [rng.random() < 0.5 for _ in range(n_sources)]
            if any(states) and any(not state for state in states):
                return states

    states = [rng.random() < 0.5 for _ in range(n_sources)]
    for idx in shared_group[:2]:
        states[idx] = True

    outside = [i for i in range(n_sources) if i not in shared_group]
    assert outside, "shared mode requires at least one source outside the shared group"
    states[rng.choice(outside)] = False
    return states


def _group_rows(count: int, start: int, gap_choices: tuple[int, int], rng: random.Random) -> list[int]:
    rows = [start]
    for _ in range(count - 1):
        rows.append(rows[-1] + rng.randint(*gap_choices))
    return rows


def _expected_output(
    grid: np.ndarray,
    groups: list[list[int]],
    bulb_rows: list[int],
    bulb_col: int,
    source_colors: list[int],
    states: list[bool],
) -> np.ndarray:
    out = grid.copy()
    for group_idx, group in enumerate(groups):
        winners = [idx for idx in group if states[idx]]
        if winners:
            out[bulb_rows[group_idx], bulb_col] = int(source_colors[winners[0]])
    return out


def _draw_route(
    grid: np.ndarray,
    src_row: int,
    vcol: int,
    bulb_row: int,
    bulb_col: int,
) -> None:
    grid[src_row, 2 : vcol + 1] = GREY
    grid[src_row : bulb_row + 1, vcol] = GREY
    grid[bulb_row, vcol:bulb_col] = GREY


def _place_horizontal_switch(grid: np.ndarray, row: int, vcol: int, rng: random.Random) -> None:
    center = rng.randint(3, vcol - 1)
    grid[row, center] = RED


def _place_vertical_switch(grid: np.ndarray, src_row: int, bulb_row: int, vcol: int, rng: random.Random) -> None:
    center = rng.randint(src_row + 1, bulb_row - 1)
    grid[center, vcol] = RED


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_sources = rng.randint(2, 5)
    shared_mode = n_sources >= 3 and rng.random() < 0.5
    groups = _make_groups(n_sources, shared_mode, rng)
    states = _sample_states(groups, rng)

    source_colors = rng.sample(SOURCE_COLOR_POOL, n_sources)
    src_rows = _group_rows(n_sources, rng.randint(1, 2), (2, 3), rng)
    bulb_rows = _group_rows(len(groups), src_rows[-1] + rng.randint(4, 6), (2, 3), rng)

    min_vcol = rng.randint(4, 6)
    step = rng.randint(2, 3)
    vcols = [min_vcol + step * (n_sources - 1 - i) for i in range(n_sources)]
    bulb_col = vcols[0] + rng.randint(6, 10)

    height = bulb_rows[-1] + rng.randint(2, 4)
    width = bulb_col + 2
    grid = np.full((height, width), BG, dtype=int)

    source_to_group: dict[int, int] = {}
    for group_idx, group in enumerate(groups):
        for source_idx in group:
            source_to_group[source_idx] = group_idx

    for group_idx, bulb_row in enumerate(bulb_rows):
        grid[bulb_row, bulb_col] = GREEN

    for idx in range(n_sources):
        src_row = src_rows[idx]
        vcol = vcols[idx]
        group_idx = source_to_group[idx]
        bulb_row = bulb_rows[group_idx]
        grid[src_row, SOURCE_COL] = source_colors[idx]
        _draw_route(grid, src_row, vcol, bulb_row, bulb_col)

    for idx in range(n_sources):
        src_row = src_rows[idx]
        vcol = vcols[idx]
        bulb_row = bulb_rows[source_to_group[idx]]
        if states[idx]:
            _place_horizontal_switch(grid, src_row, vcol, rng)
        else:
            if rng.random() < 0.5:
                _place_horizontal_switch(grid, src_row, vcol, rng)
            _place_vertical_switch(grid, src_row, bulb_row, vcol, rng)

    expected = _expected_output(grid, groups, bulb_rows, bulb_col, source_colors, states)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid circuit-switch puzzle")
