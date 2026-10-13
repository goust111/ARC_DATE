from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PIPE = 5
VALVE = 7
MIX = 9
SOURCE_PALETTE = (1, 2, 3, 4, 6, 8)

GRID_HEIGHT_RANGE = (14, 26)
GRID_WIDTH_RANGE = (14, 28)
MIN_SOURCE_GAP = 3
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _source_records(grid: np.ndarray) -> list[tuple[int, int]]:
    sources = [
        (c, int(grid[0, c]))
        for c in range(grid.shape[1])
        if int(grid[0, c]) not in (BG, PIPE, VALVE)
    ]
    assert 2 <= len(sources) <= 4, "expected 2 to 4 top-row sources"
    return sources


def _exit_pos(grid: np.ndarray) -> tuple[int, int]:
    cols = [c for c in range(grid.shape[1]) if int(grid[-1, c]) == PIPE]
    assert len(cols) == 1, "expected exactly one grey exit on the bottom row"
    return int(grid.shape[0] - 1), int(cols[0])


def _reachable_masks(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources = _source_records(g)
    exit_pos = _exit_pos(g)
    passable = g == PIPE

    masks: list[np.ndarray] = []
    for src_col, _color in sources:
        seen = np.zeros((h, w), dtype=bool)
        q: deque[tuple[int, int]] = deque()
        for rr, cc in _neighbors4(0, int(src_col), h, w):
            if passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
        while q:
            r, c = q.popleft()
            for rr, cc in _neighbors4(r, c, h, w):
                if passable[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        masks.append(seen)

    return sources, exit_pos, masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace color flow through a pipe network with shutoff valves and recolor the open pipes and exit.

    Input structure:
    - The background is black.
    - The top row contains 2 to 4 single-pixel colored source points in distinct columns.
    - Grey pixels form a 4-connected pipe network below the sources and converge to one grey exit pixel on the bottom row.
    - Some pipe pixels may be replaced by orange valve pixels that still mark the pipe visually but block flow.

    Rule:
    - From each top-row source, its color spreads through 4-connected grey pipe pixels only.
    - Orange valve pixels cannot be crossed, so they cut any path that would pass through them.
    - Every grey pipe pixel reached by exactly one source takes that source color; if reached by multiple sources, it becomes maroon.
    - Grey pipe pixels reached by no source stay grey, while the bottom exit becomes that single reaching color, maroon for multiple colors, or black if no source reaches it.
    - The top-row source pixels and all orange valves stay unchanged.

    Key insight:
    - Solve the task by independent reachability from each source on the open grey subgraph, then decide each pipe pixel and the exit by how many distinct source colors can reach them.
    """
    g = np.asarray(grid, dtype=int)
    sources, exit_pos, masks = _reachable_masks(g)

    out = g.copy()
    pipe_mask = g == PIPE
    stack = np.stack(masks, axis=0)
    counts = stack.sum(axis=0)

    out[pipe_mask & (counts >= 2)] = MIX
    single_mask = pipe_mask & (counts == 1)
    for i, (_src_col, color) in enumerate(sources):
        out[single_mask & stack[i]] = int(color)

    er, ec = exit_pos
    exit_colors = [
        int(color)
        for (_src_col, color), reached in zip(sources, masks, strict=False)
        if bool(reached[er, ec])
    ]
    if len(exit_colors) == 0:
        out[er, ec] = BG
    elif len(exit_colors) == 1:
        out[er, ec] = int(exit_colors[0])
    else:
        out[er, ec] = MIX

    return out


def _sample_spaced_columns(width: int, count: int) -> list[int] | None:
    choices = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(choices, count))
        if all(cols[i + 1] - cols[i] >= MIN_SOURCE_GAP for i in range(count - 1)):
            return cols
    return None


def _sample_merge_rows(height: int, count: int) -> list[int] | None:
    choices = list(range(4, height - 3))
    if len(choices) < count:
        return None
    for _ in range(200):
        rows = sorted(random.sample(choices, count))
        if all(rows[i + 1] - rows[i] >= 2 for i in range(count - 1)):
            return rows
    return None


def _edge_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells: list[tuple[int, int]] = []
    step_r = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step_r, step_r):
        cells.append((int(r), int(c0)))
    if c0 != c1:
        step_c = 1 if c1 > c0 else -1
        for c in range(c0 + step_c, c1 + step_c, step_c):
            cells.append((int(r1), int(c)))
    return cells


def _network_component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _reachable_source_ids(grid: np.ndarray) -> set[int]:
    sources, exit_pos, masks = _reachable_masks(grid)
    er, ec = exit_pos
    return {
        i
        for i, _source in enumerate(sources)
        if bool(masks[i][er, ec])
    }


def _build_open_network(height: int, width: int, source_cols: list[int]) -> tuple[set[tuple[int, int]], dict[int, list[tuple[int, int]]], list[tuple[int, int]]] | None:
    merge_rows = _sample_merge_rows(height, len(source_cols) - 1)
    if merge_rows is None:
        return None

    active = [
        {"row": 1, "col": int(col), "leaves": (i,)}
        for i, col in enumerate(source_cols)
    ]
    pipe_cells: set[tuple[int, int]] = {(1, int(col)) for col in source_cols}
    branch_valve_options: dict[int, list[tuple[int, int]]] = {}

    for merge_row in merge_rows:
        pair_idx = random.randrange(len(active) - 1)
        left = active[pair_idx]
        right = active[pair_idx + 1]

        lo = int(min(left["col"], right["col"]))
        hi = int(max(left["col"], right["col"]))
        if random.random() < 0.35:
            merge_col = int(random.choice([left["col"], right["col"]]))
        else:
            merge_col = int(random.randint(lo, hi))

        for child in (left, right):
            child_end = (int(merge_row), int(merge_col))
            pipe_cells.update(_edge_cells((int(child["row"]), int(child["col"])), child_end))
            if len(child["leaves"]) == 1:
                leaf_id = int(child["leaves"][0])
                if leaf_id not in branch_valve_options:
                    max_row = int(merge_row) if int(child["col"]) != int(merge_col) else int(merge_row) - 1
                    candidates = [(rr, int(child["col"])) for rr in range(2, max_row + 1)]
                    if not candidates:
                        return None
                    branch_valve_options[leaf_id] = candidates

        parent = {
            "row": int(merge_row),
            "col": int(merge_col),
            "leaves": tuple(sorted(left["leaves"] + right["leaves"])),
        }
        active[pair_idx : pair_idx + 2] = [parent]

    root = active[0]
    root_row = int(root["row"])
    root_col = int(root["col"])
    pipe_cells.update(_edge_cells((root_row, root_col), (height - 1, root_col)))
    trunk_valve_options = [(rr, root_col) for rr in range(root_row + 1, height - 1)]
    if not trunk_valve_options:
        return None

    if len(branch_valve_options) != len(source_cols):
        return None
    return pipe_cells, branch_valve_options, trunk_valve_options


def _pick_case(source_count: int) -> tuple[str, set[int]]:
    case_choices = ("none", "single") if source_count == 2 else ("none", "single", "multi")
    case = random.choice(case_choices)
    if case == "none":
        return case, set()
    if case == "single":
        return case, {random.randrange(source_count)}
    keep_count = random.randint(2, source_count - 1)
    return case, set(random.sample(range(source_count), keep_count))


def _choose_valves(
    source_count: int,
    reachable_ids: set[int],
    branch_valve_options: dict[int, list[tuple[int, int]]],
    trunk_valve_options: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    valves: set[tuple[int, int]] = set()

    if len(reachable_ids) == 0:
        valves.add(random.choice(trunk_valve_options))
        for leaf_id in range(source_count):
            if random.random() < 0.45:
                valves.add(random.choice(branch_valve_options[leaf_id]))
        return valves

    blocked = [leaf_id for leaf_id in range(source_count) if leaf_id not in reachable_ids]
    for leaf_id in blocked:
        candidates = list(branch_valve_options[leaf_id])
        first = random.choice(candidates)
        valves.add(first)
        spare = [cell for cell in candidates if cell != first]
        if spare and random.random() < 0.35:
            valves.add(random.choice(spare))
    return valves


def _render_grid(
    height: int,
    width: int,
    source_cols: list[int],
    source_colors: list[int],
    pipe_cells: set[tuple[int, int]],
    valves: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r, c in pipe_cells:
        grid[int(r), int(c)] = PIPE
    for r, c in valves:
        grid[int(r), int(c)] = VALVE
    for col, color in zip(source_cols, source_colors, strict=False):
        grid[0, int(col)] = int(color)
    return grid


def _validate_generated(grid: np.ndarray, expected_exit_count: int) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    h, w = g.shape
    try:
        sources = _source_records(g)
        exit_pos = _exit_pos(g)
    except AssertionError:
        return False

    source_cols = [int(col) for col, _color in sources]
    if len(set(source_cols)) != len(source_cols):
        return False
    if any(source_cols[i + 1] - source_cols[i] < MIN_SOURCE_GAP for i in range(len(source_cols) - 1)):
        return False
    if any(int(g[1, col]) != PIPE for col in source_cols):
        return False

    top = g[0].copy()
    top[source_cols] = BG
    if not bool(np.all(top == BG)):
        return False

    reserved = set(source_cols)
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if r == 0 and c in reserved:
                continue
            if v not in (BG, PIPE, VALVE):
                return False

    if int((g == VALVE).sum()) < 1:
        return False

    network_mask = (g == PIPE) | (g == VALVE)
    if int(network_mask.sum()) == 0 or _network_component_count(network_mask) != 1:
        return False
    if not bool(np.any(network_mask[:, 1:] & network_mask[:, :-1])):
        return False

    reachable = _reachable_source_ids(g)
    if len(reachable) != expected_exit_count:
        return False

    out = main(g)
    er, ec = exit_pos
    if not bool(np.array_equal(out[g == VALVE], g[g == VALVE])):
        return False
    if not bool(np.array_equal(out[0, source_cols], g[0, source_cols])):
        return False
    if len(reachable) == 0 and int(out[er, ec]) != BG:
        return False
    if len(reachable) == 1 and int(out[er, ec]) == MIX:
        return False
    if len(reachable) >= 2 and int(out[er, ec]) != MIX:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    source_count = random.randint(2, 4)
    if source_count == 2 and random.random() < 0.35:
        source_count = 3

    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    source_cols = _sample_spaced_columns(width, source_count)
    if source_cols is None:
        return None

    source_colors = random.sample(SOURCE_PALETTE, source_count)
    built = _build_open_network(height, width, source_cols)
    if built is None:
        return None
    pipe_cells, branch_valve_options, trunk_valve_options = built

    case, reachable_ids = _pick_case(source_count)
    if case == "multi" and source_count < 3:
        return None

    valves = _choose_valves(source_count, reachable_ids, branch_valve_options, trunk_valve_options)
    grid = _render_grid(height, width, source_cols, source_colors, pipe_cells, valves)
    if _validate_generated(grid, len(reachable_ids)):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid valve-pipe ARC input")
