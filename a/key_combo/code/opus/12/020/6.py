from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
MAROON = 9

GRID_CELLS = 4
CELL_SIZE = 4
STEP = CELL_SIZE + 1
GRID_SIZE = GRID_CELLS * CELL_SIZE + (GRID_CELLS - 1)

MARK_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
)

_GEN_CALL_COUNT = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r, c - 1
    yield r, c + 1
    yield r + 1, c


def _cell_spans(dividers: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for div in dividers:
        spans.append((start, div - 1))
        start = div + 1
    spans.append((start, limit - 1))
    return spans


def _infer_layout(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_divs = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    col_divs = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(row_divs) >= 1 and len(col_divs) >= 1, "missing separator lines"
    assert len(row_divs) == len(col_divs), "separator grid must be square"

    row_gaps = [b - a for a, b in zip(row_divs, row_divs[1:])]
    col_gaps = [b - a for a, b in zip(col_divs, col_divs[1:])]
    assert len(set(row_gaps)) <= 1 and len(set(col_gaps)) <= 1, "separator spacing must be regular"

    row_spans = _cell_spans(row_divs, h)
    col_spans = _cell_spans(col_divs, w)
    assert len(row_spans) == len(col_spans), "cell grid must be square"
    assert all(r1 - r0 + 1 == len(row_spans) for r0, r1 in row_spans), "self-similar row size required"
    assert all(c1 - c0 + 1 == len(col_spans) for c0, c1 in col_spans), "self-similar col size required"
    return row_divs, col_divs, row_spans, col_spans


def _cell_view(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], pos: tuple[int, int]) -> np.ndarray:
    cr, cc = pos
    r0, r1 = row_spans[cr]
    c0, c1 = col_spans[cc]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _minority_positions(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pos: list[tuple[int, int]] = []
    for cr in range(len(row_spans)):
        for cc in range(len(col_spans)):
            cell = _cell_view(grid, row_spans, col_spans, (cr, cc))
            if np.any(cell != BLACK):
                pos.append((cr, cc))
    return pos


def _extract_masks(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    maroon_mask = cell == MAROON
    mark_mask = (cell != BLACK) & (cell != MAROON)
    colors = sorted(set(int(v) for v in cell[mark_mask]))
    assert colors in ([BLUE], [RED]), "minority mark must be monochrome red or blue"
    return mark_mask, maroon_mask, colors[0]


def _canonical_signature(cell: np.ndarray) -> tuple[tuple[int, ...], ...]:
    normalized = cell.copy()
    normalized[(normalized == BLUE) | (normalized == RED)] = 7
    return tuple(tuple(int(v) for v in row) for row in normalized)


def _build_cell(mark_mask: np.ndarray, maroon_mask: np.ndarray, color: int) -> np.ndarray:
    cell = np.zeros(mark_mask.shape, dtype=int)
    cell[maroon_mask] = MAROON
    cell[mark_mask] = int(color)
    return cell


def _stamp_cell(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], pos: tuple[int, int], cell: np.ndarray) -> None:
    view = _cell_view(grid, row_spans, col_spans, pos)
    view[:, :] = cell


def _ports_for_cell(pos: tuple[int, int], row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = pos
    h, w = shape
    r0, r1 = row_spans[cr]
    c0, c1 = col_spans[cc]
    ports: set[tuple[int, int]] = set()
    if cr > 0:
        ports.update((r0 - 1, c) for c in range(c0, c1 + 1))
    if cr + 1 < len(row_spans):
        ports.update((r1 + 1, c) for c in range(c0, c1 + 1))
    if cc > 0:
        ports.update((r, c0 - 1) for r in range(r0, r1 + 1))
    if cc + 1 < len(col_spans):
        ports.update((r, c1 + 1) for r in range(r0, r1 + 1))
    return {(r, c) for r, c in ports if 0 <= r < h and 0 <= c < w}


def _search_route(
    grid: np.ndarray,
    starts: set[tuple[int, int]],
    ends: set[tuple[int, int]],
    color: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], int]:
    walkable = {GREY, int(color)}
    starts = {p for p in starts if int(grid[p]) in walkable}
    ends = {p for p in ends if int(grid[p]) in walkable}
    assert starts and ends, "connector needs accessible channel pixels"

    common = sorted(starts & ends)
    if common:
        return [common[0]], (0, 0), 1

    pq: list[tuple[int, int, int, int, int]] = []
    best: dict[tuple[int, int, int], tuple[int, int]] = {}
    prev: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}

    for r, c in sorted(starts):
        state = (r, c, 2)
        best[state] = (0, 0)
        prev[state] = None
        heappush(pq, (0, 0, r, c, 2))

    best_end_cost: tuple[int, int] | None = None
    end_states: list[tuple[int, int, int]] = []

    while pq:
        dist, turns, r, c, last_axis = heappop(pq)
        state = (r, c, last_axis)
        if best.get(state) != (dist, turns):
            continue
        if best_end_cost is not None and (dist, turns) > best_end_cost:
            break
        if (r, c) in ends:
            best_end_cost = (dist, turns)
            end_states.append(state)
            continue
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if int(grid[nr, nc]) not in walkable:
                continue
            axis = 0 if nr == r else 1
            nturns = turns + int(last_axis != 2 and last_axis != axis)
            ndist = dist + 1
            nxt = (nr, nc, axis)
            old = best.get(nxt)
            if old is None or (ndist, nturns) < old:
                best[nxt] = (ndist, nturns)
                prev[nxt] = state
                heappush(pq, (ndist, nturns, nr, nc, axis))

    assert end_states, "no separator route between same-color cells"
    end_state = min(end_states, key=lambda state: (state[0], state[1], state[2]))

    path: list[tuple[int, int]] = []
    cur: tuple[int, int, int] | None = end_state
    while cur is not None:
        path.append((cur[0], cur[1]))
        cur = prev[cur]
    path.reverse()

    ordered_states = sorted(best, key=lambda state: (best[state][0], best[state][1], state[0], state[1], state[2]))
    counts = {state: 0 for state in best}
    for state in ordered_states:
        if prev[state] is None:
            counts[state] = 1
        else:
            parent = prev[state]
            counts[state] = counts[parent]
        r, c, last_axis = state
        cost = best[state]
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if int(grid[nr, nc]) not in walkable:
                continue
            axis = 0 if nr == r else 1
            nxt = (nr, nc, axis)
            if nxt not in best:
                continue
            ncost = (cost[0] + 1, cost[1] + int(last_axis != 2 and last_axis != axis))
            if best[nxt] == ncost and prev[nxt] != state:
                counts[nxt] = min(2, counts[nxt] + counts[state])

    total_ways = 0
    for state in end_states:
        if best[state] == best_end_cost:
            total_ways = min(2, total_ways + counts[state])
    return path, best_end_cost, total_ways


def _route_pair(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    src: tuple[int, int],
    dst: tuple[int, int],
    color: int,
    *,
    require_unique: bool = False,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    starts = _ports_for_cell(src, row_spans, col_spans, grid.shape)
    ends = _ports_for_cell(dst, row_spans, col_spans, grid.shape)
    path, cost, ways = _search_route(grid, starts, ends, color)
    if require_unique:
        assert ways == 1, "generator expects a unique optimal path"
    return path, cost


def _paint_chain(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    positions: list[tuple[int, int]],
    color: int,
    *,
    require_unique: bool = False,
) -> list[list[tuple[int, int]]]:
    paths: list[list[tuple[int, int]]] = []
    for src, dst in zip(positions, positions[1:]):
        path, _ = _route_pair(
            grid,
            row_spans,
            col_spans,
            src,
            dst,
            color,
            require_unique=require_unique,
        )
        for r, c in path:
            assert int(grid[r, c]) in (GREY, color), "connector collision"
            grid[r, c] = int(color)
        paths.append(path)
    return paths


def _base_grid() -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for k in range(1, GRID_CELLS):
        idx = k * STEP - 1
        grid[idx, :] = GREY
        grid[:, idx] = GREY
    return grid


def _direct_output(
    blueprint_positions: list[tuple[int, int]],
    mark_mask: np.ndarray,
    maroon_mask: np.ndarray,
) -> np.ndarray:
    grid = _base_grid()
    row_spans = [(r * STEP, r * STEP + CELL_SIZE - 1) for r in range(GRID_CELLS)]
    col_spans = [(c * STEP, c * STEP + CELL_SIZE - 1) for c in range(GRID_CELLS)]
    for idx, pos in enumerate(blueprint_positions):
        color = RED if idx % 2 == 0 else BLUE
        _stamp_cell(grid, row_spans, col_spans, pos, _build_cell(mark_mask, maroon_mask, color))
    red_positions = [pos for idx, pos in enumerate(blueprint_positions) if idx % 2 == 0]
    blue_positions = [pos for idx, pos in enumerate(blueprint_positions) if idx % 2 == 1]
    _paint_chain(grid, row_spans, col_spans, red_positions, RED, require_unique=False)
    _paint_chain(grid, row_spans, col_spans, blue_positions, BLUE, require_unique=False)
    return grid


def _choose_mark_mask(maroon_mask: np.ndarray) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for shape in MARK_SHAPES:
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        for r0 in range(CELL_SIZE - height + 1):
            for c0 in range(CELL_SIZE - width + 1):
                mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
                coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if any(maroon_mask[r, c] for r, c in coords):
                    continue
                for r, c in coords:
                    mask[r, c] = True
                candidates.append(mask)
    assert candidates, "need a non-overlapping mark shape"
    return random.choice(candidates)


def _subset_for_input(positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    indexed = list(enumerate(positions))
    target_size = 2 if len(positions) == 4 else random.choice([2, 3])
    for _ in range(100):
        chosen = sorted(random.sample(indexed, target_size))
        colors = {RED if idx % 2 == 0 else BLUE for idx, _ in chosen}
        if colors != {RED, BLUE}:
            continue
        missing = [idx for idx, _ in indexed if idx not in {i for i, _ in chosen}]
        missing_colors = {RED if idx % 2 == 0 else BLUE for idx in missing}
        if len(missing_colors) == 2 or len(positions) == 4:
            return [pos for _, pos in chosen]
    return [positions[0], positions[1 if len(positions) > 2 else -1]]


def _fixed_spans() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = [(r * STEP, r * STEP + CELL_SIZE - 1) for r in range(GRID_CELLS)]
    col_spans = [(c * STEP, c * STEP + CELL_SIZE - 1) for c in range(GRID_CELLS)]
    return row_spans, col_spans


def _classify_blueprint(positions: list[tuple[int, int]]) -> tuple[bool, int] | None:
    row_spans, col_spans = _fixed_spans()
    red_positions = [pos for idx, pos in enumerate(positions) if idx % 2 == 0]
    blue_positions = [pos for idx, pos in enumerate(positions) if idx % 2 == 1]
    if len(red_positions) < 2 or len(blue_positions) < 2:
        return None

    try:
        blue_clean_grid = _base_grid()
        blue_clean_paths = []
        for src, dst in zip(blue_positions, blue_positions[1:]):
            path, _ = _route_pair(blue_clean_grid, row_spans, col_spans, src, dst, BLUE, require_unique=False)
            for r, c in path:
                blue_clean_grid[r, c] = BLUE
            blue_clean_paths.append(path)

        routed = _base_grid()
        red_paths = _paint_chain(routed, row_spans, col_spans, red_positions, RED, require_unique=False)
        blue_blocked_paths = []
        for src, dst in zip(blue_positions, blue_positions[1:]):
            path, _ = _route_pair(routed, row_spans, col_spans, src, dst, BLUE, require_unique=False)
            for r, c in path:
                routed[r, c] = BLUE
            blue_blocked_paths.append(path)
    except AssertionError:
        return None

    conflict = blue_clean_paths != blue_blocked_paths
    total_added = sum(len(path) for path in red_paths + blue_blocked_paths)
    if total_added < 6:
        return None
    rows = {r for r, _ in positions}
    cols = {c for _, c in positions}
    if len(rows) == 1 or len(cols) == 1:
        return None
    return conflict, total_added


def _build_blueprint_pools() -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    conflict_pool: list[list[tuple[int, int]]] = []
    calm_pool: list[list[tuple[int, int]]] = []
    cells = [(r, c) for r in range(GRID_CELLS) for c in range(GRID_CELLS)]
    for count in (4, 5, 6):
        for combo in combinations(cells, count):
            positions = sorted(combo)
            classified = _classify_blueprint(positions)
            if classified is None:
                continue
            conflict, total_added = classified
            if total_added > 18:
                continue
            if conflict:
                conflict_pool.append(positions)
            else:
                calm_pool.append(positions)
    assert conflict_pool and calm_pool, "need both blueprint families"
    return conflict_pool, calm_pool


_CONFLICT_BLUEPRINTS, _CALM_BLUEPRINTS = _build_blueprint_pools()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the self-similar minority cells and connect same-colored copies through the separator channels.

    Input structure:
    - A square separator grid splits the canvas into equal interior cells, with grey channel pixels between them.
    - Most cells are all black; a small subset are minority cells containing maroon blueprint pixels and a red or blue mark shape.
    - All minority cells share the same internal layout except for the mark color.

    Rule:
    - Read the maroon pixels inside a minority cell as a macro blueprint: each maroon coordinate names a cell position in the cell array.
    - Fill every blueprint position with that same minority layout, assigning mark colors in row-major order as red, blue, red, blue, ...
    - Then, within each color group, connect consecutive minority cells in row-major order by repainting a shortest separator-channel path in that color.
    - Red connectors are painted first; blue connectors must detour around any red-painted separator pixels.

    Key insight:
    - The minority cell is a miniature map of its own large-scale placement, and that completed placement determines which separator corridors must be chained together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, _, row_spans, col_spans = _infer_layout(g)
    minority = _minority_positions(g, row_spans, col_spans)
    assert minority, "need at least one minority cell"

    sample = _cell_view(g, row_spans, col_spans, minority[0]).copy()
    mark_mask, maroon_mask, _ = _extract_masks(sample)
    signature = _canonical_signature(sample)

    for pos in minority[1:]:
        cell = _cell_view(g, row_spans, col_spans, pos)
        assert _canonical_signature(cell) == signature, "minority cells must share one layout"

    blueprint_positions = sorted((int(r), int(c)) for r, c in np.argwhere(maroon_mask))
    assert blueprint_positions, "minority cell must contain blueprint pixels"
    assert set(minority).issubset(set(blueprint_positions)), "given minority cells must agree with the blueprint"

    out = g.copy()
    for idx, pos in enumerate(blueprint_positions):
        color = RED if idx % 2 == 0 else BLUE
        _stamp_cell(out, row_spans, col_spans, pos, _build_cell(mark_mask, maroon_mask, color))

    red_positions = [pos for idx, pos in enumerate(blueprint_positions) if idx % 2 == 0]
    blue_positions = [pos for idx, pos in enumerate(blueprint_positions) if idx % 2 == 1]
    _paint_chain(out, row_spans, col_spans, red_positions, RED, require_unique=False)
    _paint_chain(out, row_spans, col_spans, blue_positions, BLUE, require_unique=False)
    return out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    target_conflict = (_GEN_CALL_COUNT % 2 == 0)
    pool = _CONFLICT_BLUEPRINTS if target_conflict else _CALM_BLUEPRINTS

    for _ in range(300):
        blueprint_positions = list(random.choice(pool))
        maroon_mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        for r, c in blueprint_positions:
            maroon_mask[r, c] = True
        mark_mask = _choose_mark_mask(maroon_mask)
        given_positions = _subset_for_input(blueprint_positions)

        grid = _base_grid()
        row_spans, col_spans = _fixed_spans()
        for pos in given_positions:
            idx = blueprint_positions.index(pos)
            color = RED if idx % 2 == 0 else BLUE
            _stamp_cell(grid, row_spans, col_spans, pos, _build_cell(mark_mask, maroon_mask, color))

        expected = _direct_output(blueprint_positions, mark_mask, maroon_mask)
        produced = main(grid)
        if not np.array_equal(produced, expected):
            continue

        minority = _minority_positions(grid, row_spans, col_spans)
        if len(minority) not in (2, 3):
            continue
        if len(set(minority)) != len(minority):
            continue
        final_minority = _minority_positions(expected, row_spans, col_spans)
        if set(final_minority) != set(blueprint_positions):
            continue
        if len(blueprint_positions) - len(given_positions) < 1:
            continue

        if target_conflict:
            classified = _classify_blueprint(blueprint_positions)
            if classified is None or not classified[0]:
                continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
