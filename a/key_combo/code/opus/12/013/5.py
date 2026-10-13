import numpy as np


BACKGROUND = 0
GREY = 5
OBSTACLE = 6
PALETTE = (1, 2, 3, 4, 7, 8, 9)
GRID_SIZE = 15
MACRO = 3
MACRO_COUNT = GRID_SIZE // MACRO
RETRY_LIMIT = 500

_GEN_CALL_COUNT = 0

_GEN_MODES = (
    {
        "source_count": 3,
        "start_choices": (6,),
        "obstacle_rows": (1, 3),
        "endpoint_count": 0,
        "require_tie_reroute": True,
        "require_second_band_reroute": True,
        "require_offcol_endpoint": False,
    },
    {
        "source_count": 4,
        "start_choices": (4, 5),
        "obstacle_rows": (1,),
        "endpoint_count": 2,
        "require_tie_reroute": False,
        "require_second_band_reroute": False,
        "require_offcol_endpoint": True,
    },
    {
        "source_count": 3,
        "start_choices": (5, 6),
        "obstacle_rows": (1, 3),
        "endpoint_count": 3,
        "require_tie_reroute": False,
        "require_second_band_reroute": True,
        "require_offcol_endpoint": True,
    },
    {
        "source_count": 4,
        "start_choices": (3, 4, 5),
        "obstacle_rows": (3,),
        "endpoint_count": 1,
        "require_tie_reroute": False,
        "require_second_band_reroute": False,
        "require_offcol_endpoint": True,
    },
)


def _top_sources(grid: np.ndarray) -> list[tuple[int, int]]:
    top_cols = np.flatnonzero(grid[0] != BACKGROUND)
    assert 3 <= top_cols.size <= 4
    assert np.all(np.diff(top_cols) == 1)
    sources = [(int(c), int(grid[0, c])) for c in top_cols]
    colors = [color for _, color in sources]
    assert len(colors) == len(set(colors))
    return sources


def _marked_blocks(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    assert h % MACRO == 0 and w % MACRO == 0
    marked = []
    for br in range(h // MACRO):
        for bc in range(w // MACRO):
            cr = br * MACRO + 1
            cc = bc * MACRO + 1
            if int(grid[cr, cc]) == GREY:
                marked.append((br, bc))
    return marked


def _obstacle_mask(shape: tuple[int, int], blocks: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for br, bc in blocks:
        r0 = br * MACRO
        c0 = bc * MACRO
        mask[r0 : r0 + MACRO, c0 : c0 + MACRO] = True
    return mask


def _endpoints(grid: np.ndarray, sources: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for src_col, color in sources:
        coords = np.argwhere(grid == color)
        others = [
            (int(r), int(c))
            for r, c in coords
            if not (int(r) == 0 and int(c) == src_col)
        ]
        assert len(others) <= 1
        if others:
            out[color] = others[0]
    return out


def _nearest_free_column(blocked_row: np.ndarray, current_col: int) -> int:
    w = blocked_row.shape[0]
    for dist in range(w):
        right = current_col + dist
        if right < w and not bool(blocked_row[right]):
            return int(right)
        if dist == 0:
            continue
        left = current_col - dist
        if left >= 0 and not bool(blocked_row[left]):
            return int(left)
    raise AssertionError("no free column available for reroute")


def _column_at_row(obstacle_mask: np.ndarray, src_col: int, target_row: int) -> int:
    r, c = 0, int(src_col)
    while r < target_row:
        if obstacle_mask[r + 1, c]:
            c = _nearest_free_column(obstacle_mask[r + 1], c)
            continue
        r += 1
    return c


def _paint_horizontal(
    out: np.ndarray,
    row: int,
    c0: int,
    c1: int,
    color: int,
    obstacle_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    step = 1 if c1 >= c0 else -1
    cells = []
    for c in range(c0, c1 + step, step):
        if obstacle_mask is not None:
            assert not bool(obstacle_mask[row, c]), "horizontal segment cannot cross an obstacle"
        out[row, c] = color
        cells.append((row, c))
    return cells


def _trace_path(
    obstacle_mask: np.ndarray,
    src_col: int,
    color: int,
    endpoint: tuple[int, int] | None,
    shape: tuple[int, int],
    out: np.ndarray | None = None,
) -> dict:
    h, _ = shape
    r, c = 0, int(src_col)
    if out is not None:
        out[r, c] = color

    target_row = endpoint[0] if endpoint is not None else h - 1
    visited = [(r, c)]
    reroute_rows = []
    reroute_meta = []
    tie_reroutes = 0

    while r < target_row:
        if obstacle_mask[r + 1, c]:
            blocked_row = obstacle_mask[r + 1]
            left_dist = None
            right_dist = None
            for delta in range(1, blocked_row.shape[0] + 1):
                if right_dist is None and c + delta < blocked_row.shape[0] and not bool(blocked_row[c + delta]):
                    right_dist = delta
                if left_dist is None and c - delta >= 0 and not bool(blocked_row[c - delta]):
                    left_dist = delta
                if left_dist is not None and right_dist is not None:
                    break
            new_c = _nearest_free_column(blocked_row, c)
            if left_dist is not None and right_dist is not None and left_dist == right_dist:
                tie_reroutes += 1
            if out is not None:
                visited.extend(_paint_horizontal(out, r, c, new_c, color, obstacle_mask)[1:])
            else:
                step = 1 if new_c >= c else -1
                for cc in range(c + step, new_c + step, step):
                    assert not bool(obstacle_mask[r, cc])
                    visited.append((r, cc))
            reroute_rows.append(r)
            reroute_meta.append((r, c, new_c))
            c = new_c
            continue

        r += 1
        if out is not None:
            assert not bool(obstacle_mask[r, c])
            out[r, c] = color
        visited.append((r, c))

    horizontal_turn = 0
    if endpoint is not None and c != endpoint[1]:
        horizontal_turn = abs(endpoint[1] - c)
        if out is not None:
            visited.extend(_paint_horizontal(out, r, c, endpoint[1], color, obstacle_mask)[1:])
        else:
            step = 1 if endpoint[1] >= c else -1
            for cc in range(c + step, endpoint[1] + step, step):
                assert not bool(obstacle_mask[r, cc])
                visited.append((r, cc))
        c = endpoint[1]

    return {
        "color": color,
        "cells": visited,
        "endpoint": endpoint,
        "reroute_count": len(reroute_rows),
        "reroute_rows": tuple(reroute_rows),
        "reroute_meta": tuple(reroute_meta),
        "tie_reroutes": tie_reroutes,
        "horizontal_turn": horizontal_turn,
        "final_col": c,
    }


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] % MACRO == 0 and g.shape[1] % MACRO == 0
    assert set(np.unique(g)).issubset(set(PALETTE) | {BACKGROUND, GREY})

    sources = _top_sources(g)
    endpoint_map = _endpoints(g, sources)
    marked = _marked_blocks(g)
    obstacle_mask = _obstacle_mask(g.shape, marked)

    out = g.copy()
    out[obstacle_mask] = OBSTACLE

    traces = []
    for src_col, color in sources:
        trace = _trace_path(obstacle_mask, src_col, color, endpoint_map.get(color), g.shape, out=out)
        traces.append(trace)

    return out, traces, obstacle_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill center-marked 3x3 macro-blocks into obstacles, then reroute each top-edge source path around them on the way to its matching endpoint or the bottom edge.

    Input structure:
    - A black background grid is partitioned into 3x3 macro-blocks.
    - The top row contains one contiguous run of 3 or 4 distinct source colors.
    - Some non-top macro-blocks contain a single gray center marker, and some source colors may also have one matching endpoint pixel elsewhere.

    Rule:
    - Every 3x3 macro-block whose center is gray is fully filled with magenta and becomes an obstacle.
    - For each top-row source, draw its path downward; whenever the next step would enter an obstacle, stop above it, shift horizontally to the nearest free column (preferring the right on ties), and continue downward.
    - If that color has a matching endpoint, stop descending at the endpoint row, draw horizontally to the endpoint column, and then stop; otherwise continue to the bottom edge.

    Key insight:
    - The gray center markers do not produce the final answer directly: they first create solid obstacle macro-blocks, and those obstacles force later source paths to detour before any endpoint turn can be resolved.
    """
    out, _, _ = _solve_with_trace(grid)
    return out


def _macro_cols_for_sources(source_cols: list[int]) -> list[int]:
    return sorted({c // MACRO for c in source_cols})


def _add_obstacles(
    grid: np.ndarray,
    source_cols: list[int],
    mode: dict,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    source_macro_cols = _macro_cols_for_sources(source_cols)
    placed: set[tuple[int, int]] = set()
    required_rows = tuple(mode["obstacle_rows"])

    if 1 in required_rows:
        if mode["require_tie_reroute"]:
            first_col = source_cols[len(source_cols) // 2] // MACRO
        else:
            first_col = int(rng.choice(source_macro_cols))
        placed.add((1, int(first_col)))
        if rng.random() < 0.55:
            neighbor_choices = [first_col]
            if first_col > 0:
                neighbor_choices.append(first_col - 1)
            if first_col < MACRO_COUNT - 1:
                neighbor_choices.append(first_col + 1)
            placed.add((1, int(rng.choice(neighbor_choices))))

    if 3 in required_rows:
        partial_mask = _obstacle_mask(grid.shape, sorted(placed))
        stop_cols = [_column_at_row(partial_mask, src_col, 8) for src_col in source_cols]
        anchor_cols = sorted({c // MACRO for c in stop_cols})
        second_col = int(rng.choice(anchor_cols))
        placed.add((3, second_col))
        if mode["require_second_band_reroute"] or rng.random() < 0.45:
            neighbor_choices = [second_col]
            if second_col > 0:
                neighbor_choices.append(second_col - 1)
            if second_col < MACRO_COUNT - 1:
                neighbor_choices.append(second_col + 1)
            placed.add((3, int(rng.choice(neighbor_choices))))

    desired_total = max(2, min(4, len(placed) + int(rng.integers(0, 2))))
    while len(placed) < desired_total:
        macro_row = int(rng.choice(required_rows))
        if macro_row == 1 and (1, source_cols[len(source_cols) // 2] // MACRO) not in placed and mode["require_tie_reroute"]:
            macro_col = source_cols[len(source_cols) // 2] // MACRO
        else:
            current_cols = source_macro_cols
            if macro_row == 3:
                partial_mask = _obstacle_mask(grid.shape, sorted(placed))
                current_cols = sorted({_column_at_row(partial_mask, src_col, 8) // MACRO for src_col in source_cols})
            macro_col = int(rng.choice(current_cols))
            if rng.random() < 0.5:
                shift = int(rng.choice((-1, 0, 1)))
                macro_col = min(MACRO_COUNT - 1, max(0, macro_col + shift))
        placed.add((macro_row, macro_col))

    for br, bc in placed:
        grid[br * MACRO + 1, bc * MACRO + 1] = GREY
    return sorted(placed)


def _safe_endpoint_rows(mode: dict, rng: np.random.Generator, count: int) -> list[int]:
    rows = []
    for idx in range(count):
        if 3 in mode["obstacle_rows"] and (mode["require_second_band_reroute"] or idx == 0):
            pool = (12, 13, 14)
        elif 1 in mode["obstacle_rows"]:
            pool = (6, 7, 8, 12, 13, 14)
        else:
            pool = (10, 11, 12, 13, 14)
        rows.append(int(rng.choice(pool)))
    return rows


def _place_endpoints(
    grid: np.ndarray,
    sources: list[tuple[int, int]],
    obstacle_blocks: list[tuple[int, int]],
    mode: dict,
    rng: np.random.Generator,
) -> dict[int, tuple[int, int]]:
    endpoint_count = min(mode["endpoint_count"], len(sources))
    if endpoint_count == 0:
        return {}

    obstacle_mask = _obstacle_mask(grid.shape, obstacle_blocks)
    chosen = list(rng.choice(len(sources), size=endpoint_count, replace=False))
    rows = _safe_endpoint_rows(mode, rng, endpoint_count)
    endpoints: dict[int, tuple[int, int]] = {}

    for idx, src_index in enumerate(chosen):
        src_col, color = sources[int(src_index)]
        row = rows[idx]
        used_cols = {c for _, c in endpoints.values()}
        candidates = [
            c
            for c in range(GRID_SIZE)
            if not obstacle_mask[row, c]
            and c not in used_cols
            and not (row == 0 and c == src_col)
        ]
        if mode["require_offcol_endpoint"] or idx > 0:
            offcol = [c for c in candidates if c != src_col]
            if offcol:
                candidates = offcol
        edge_bias = [c for c in candidates if c <= 2 or c >= GRID_SIZE - 3]
        if edge_bias and rng.random() < 0.75:
            candidates = edge_bias
        col = int(rng.choice(candidates))
        grid[row, col] = color
        endpoints[color] = (row, col)
    return endpoints


def _input_validity_checks(grid: np.ndarray, mode: dict) -> bool:
    try:
        out, traces, obstacle_mask = _solve_with_trace(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    top_sources = _top_sources(grid)
    endpoints = _endpoints(grid, top_sources)
    if len(endpoints) != min(mode["endpoint_count"], len(top_sources)):
        return False

    marked = _marked_blocks(grid)
    if not (2 <= len(marked) <= 4):
        return False

    if any(int(grid[r, c]) == OBSTACLE for r, c in np.argwhere(obstacle_mask)):
        return False

    if any(trace["reroute_count"] == 0 for trace in traces) and sum(t["reroute_count"] for t in traces) == 0:
        return False
    if sum(t["reroute_count"] for t in traces) < 1:
        return False
    if mode["require_second_band_reroute"] and not any(8 in trace["reroute_rows"] for trace in traces):
        return False
    if mode["require_tie_reroute"] and sum(t["tie_reroutes"] for t in traces) < 1:
        return False
    if mode["endpoint_count"] > 0 and not any(t["endpoint"] is not None and t["horizontal_turn"] > 0 for t in traces):
        return False

    all_cells = []
    for trace in traces:
        all_cells.extend(trace["cells"])
    overlap_count = len(all_cells) - len(set(all_cells))
    if overlap_count > 18:
        return False

    if np.count_nonzero(out == OBSTACLE) != 9 * len(marked):
        return False

    return True


def _gen_attempt(mode: dict, rng: np.random.Generator) -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    source_count = int(mode["source_count"])
    start_col = int(rng.choice(mode["start_choices"]))
    source_cols = list(range(start_col, start_col + source_count))
    if source_cols[-1] >= GRID_SIZE:
        return None

    colors = list(rng.choice(PALETTE, size=source_count, replace=False))
    rng.shuffle(colors)
    for c, color in zip(source_cols, colors):
        grid[0, c] = int(color)
    sources = list(zip(source_cols, colors))

    obstacle_blocks = _add_obstacles(grid, source_cols, mode, rng)
    _place_endpoints(grid, sources, obstacle_blocks, mode, rng)

    if not _input_validity_checks(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _GEN_MODES[_GEN_CALL_COUNT % len(_GEN_MODES)]
    _GEN_CALL_COUNT += 1

    for _ in range(RETRY_LIMIT):
        rng = np.random.default_rng()
        grid = _gen_attempt(mode, rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
