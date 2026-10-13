from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND_COLOR = 0
SOURCE_COLOR = 4
SEPARATOR_COLOR = 8
RELAY_PALETTE = [1, 2, 3, 6, 7, 9]
GENERATION_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _infer_structure(grid: np.ndarray) -> tuple[int, int, tuple[int, int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    source_positions = [tuple(map(int, rc)) for rc in np.argwhere(g == SOURCE_COLOR)]
    assert len(source_positions) == 1, "expected exactly one yellow source pixel"
    source = source_positions[0]

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert SOURCE_COLOR in colors, "source color must be present"

    separator_candidates = [color for color in colors if color != SOURCE_COLOR]
    assert separator_candidates, "expected a separator color"
    separator = max(separator_candidates, key=lambda color: int(np.count_nonzero(g == int(color))))

    row_strength = max(int(np.count_nonzero(g[r, :] == separator)) for r in range(g.shape[0]))
    col_strength = max(int(np.count_nonzero(g[:, c] == separator)) for c in range(g.shape[1]))
    assert row_strength >= 2 and col_strength >= 2, "separator must form both horizontal and vertical lines"

    relay_positions = {
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (g != SOURCE_COLOR) & (g != separator))
    }
    assert relay_positions, "expected at least one relay pixel"

    return bg, int(separator), source, relay_positions


def _propagate(
    grid: np.ndarray,
    *,
    separator: int,
    bg: int,
    source: tuple[int, int],
    relay_positions: set[tuple[int, int]],
    initial_active_relays: set[tuple[int, int]] | None = None,
    allow_activation: bool = True,
    respect_walls: bool = True,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    initial = set() if initial_active_relays is None else {tuple(map(int, rc)) for rc in initial_active_relays}
    queue = deque([source, *sorted(initial)])
    seen_emitters = {source, *initial}
    activated_relays = set(initial)
    touched_relays: set[tuple[int, int]] = set()

    for rr, cc in initial:
        assert (rr, cc) in relay_positions

    for origin_r, origin_c in queue:
        assert in_bounds(g.shape, origin_r, origin_c)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r0, c0 = queue.popleft()
        for dr, dc in directions:
            r = int(r0)
            c = int(c0)
            while True:
                r += dr
                c += dc
                if not in_bounds(g.shape, r, c):
                    break

                value = int(g[r, c])
                if value == separator:
                    if respect_walls:
                        break
                    continue

                if value == bg:
                    out[r, c] = SOURCE_COLOR
                    continue

                if (r, c) in relay_positions:
                    touched_relays.add((r, c))
                    if allow_activation and (r, c) not in seen_emitters:
                        seen_emitters.add((r, c))
                        activated_relays.add((r, c))
                        queue.append((r, c))
                    continue

                assert value == SOURCE_COLOR, "unexpected non-passable pixel"

    return out, activated_relays, touched_relays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade yellow light through a separator grid so only gate-aligned rays and lit relays extend the illumination.

    Input structure:
    - Equal rectangular cells are partitioned by one-cell-thick separator lines, with some separator cells replaced by background as single-cell gates.
    - Exactly one yellow source pixel sits inside a cell, and several other singleton relay pixels of other colors are scattered in distinct cells.
    - Relay pixels lie on ordinary cell backgrounds; separator cells are the only opaque walls.

    Rule:
    - Emit yellow rays from the source along its row and column in all four orthogonal directions.
    - Rays travel straight through background and relay cells, but stop at separator cells unless that exact separator cell is a background gate.
    - Every background cell crossed by light becomes yellow; relay cells keep their original colors.
    - Any relay touched by a ray becomes a new emitter that repeats the same row-and-column propagation, and the process continues until no new relay is activated.

    Key insight:
    - The separator grid is a wall system with point gates, so the answer is determined by exact row or column alignment through those gates plus the chain reaction from only the relays that actually get illuminated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, separator, source, relay_positions = _infer_structure(g)
    out, _activated_relays, _touched_relays = _propagate(
        g,
        separator=separator,
        bg=bg,
        source=source,
        relay_positions=relay_positions,
    )
    return out


def _grid_shape(cell_rows: int, cell_cols: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return cell_rows * cell_h + (cell_rows - 1), cell_cols * cell_w + (cell_cols - 1)


def _cell_origin(cell_r: int, cell_c: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return cell_r * (cell_h + 1), cell_c * (cell_w + 1)


def _cell_coord(cell_r: int, cell_c: int, local_r: int, local_c: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    r0, c0 = _cell_origin(cell_r, cell_c, cell_h, cell_w)
    return int(r0 + local_r), int(c0 + local_c)


def _separator_row(upper_cell_r: int, cell_h: int) -> int:
    return upper_cell_r * (cell_h + 1) + cell_h


def _separator_col(left_cell_c: int, cell_w: int) -> int:
    return left_cell_c * (cell_w + 1) + cell_w


def _build_separator_grid(cell_rows: int, cell_cols: int, cell_h: int, cell_w: int) -> np.ndarray:
    height, width = _grid_shape(cell_rows, cell_cols, cell_h, cell_w)
    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    for rr in range(cell_rows - 1):
        grid[_separator_row(rr, cell_h), :] = SEPARATOR_COLOR
    for cc in range(cell_cols - 1):
        grid[:, _separator_col(cc, cell_w)] = SEPARATOR_COLOR
    return grid


def _open_vertical_gate(grid: np.ndarray, *, cell_r: int, left_cell_c: int, local_r: int, cell_h: int, cell_w: int) -> None:
    gate_r, gate_c = _cell_coord(cell_r, left_cell_c, local_r, cell_w, cell_h, cell_w)
    assert int(grid[gate_r, gate_c]) == SEPARATOR_COLOR
    grid[gate_r, gate_c] = BACKGROUND_COLOR


def _open_horizontal_gate(grid: np.ndarray, *, upper_cell_r: int, cell_c: int, local_c: int, cell_h: int, cell_w: int) -> None:
    gate_r, gate_c = _cell_coord(upper_cell_r, cell_c, cell_h, local_c, cell_h, cell_w)
    assert int(grid[gate_r, gate_c]) == SEPARATOR_COLOR
    grid[gate_r, gate_c] = BACKGROUND_COLOR


def _available_cells(cell_rows: int, cell_cols: int, used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (rr, cc)
        for rr in range(cell_rows)
        for cc in range(cell_cols)
        if (rr, cc) not in used
    ]


def _apply_random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=int, copy=True)


def _sample_local_pair(limit: int) -> tuple[int, int]:
    a, b = random.sample(range(limit), 2)
    return int(a), int(b)


def _generate_attempt() -> np.ndarray | None:
    cell_rows = random.randint(3, 4)
    cell_cols = random.randint(3, 4)
    cell_h = random.randint(3, 5)
    cell_w = random.randint(3, 5)

    grid = _build_separator_grid(cell_rows, cell_cols, cell_h, cell_w)

    anchor_r = random.randint(0, cell_rows - 3)
    anchor_c = random.randint(0, cell_cols - 3)
    turn_up = random.choice([False, True])
    branch = random.choice([False, True])

    source_cell = (anchor_r + 1, anchor_c)
    first_relay_cell = (anchor_r + 1, anchor_c + 1)
    second_relay_cell = (anchor_r if turn_up else anchor_r + 2, anchor_c + 1)
    blocked_relay_cell = (anchor_r + 1, anchor_c + 2)
    branch_relay_cell = (second_relay_cell[0], anchor_c + 2) if branch else None

    shared_row, turn_row = _sample_local_pair(cell_h)
    source_col, turn_col = _sample_local_pair(cell_w)
    blocked_col = random.randrange(cell_w)
    branch_col = random.randrange(cell_w) if branch else None

    relay_count = 3 + int(branch)
    relay_colors = random.sample(RELAY_PALETTE, relay_count)

    placements = [
        (source_cell, (shared_row, source_col), SOURCE_COLOR),
        (first_relay_cell, (shared_row, turn_col), relay_colors[0]),
        (second_relay_cell, (turn_row, turn_col), relay_colors[1]),
        (blocked_relay_cell, (shared_row, blocked_col), relay_colors[2]),
    ]
    if branch_relay_cell is not None and branch_col is not None:
        placements.append((branch_relay_cell, (turn_row, branch_col), relay_colors[3]))

    for (cell_r, cell_c), (local_r, local_c), color in placements:
        rr, cc = _cell_coord(cell_r, cell_c, local_r, local_c, cell_h, cell_w)
        assert int(grid[rr, cc]) == BACKGROUND_COLOR
        grid[rr, cc] = int(color)

    _open_vertical_gate(
        grid,
        cell_r=source_cell[0],
        left_cell_c=source_cell[1],
        local_r=shared_row,
        cell_h=cell_h,
        cell_w=cell_w,
    )
    _open_horizontal_gate(
        grid,
        upper_cell_r=min(first_relay_cell[0], second_relay_cell[0]),
        cell_c=first_relay_cell[1],
        local_c=turn_col,
        cell_h=cell_h,
        cell_w=cell_w,
    )

    if branch_relay_cell is not None:
        _open_vertical_gate(
            grid,
            cell_r=branch_relay_cell[0],
            left_cell_c=first_relay_cell[1],
            local_r=turn_row,
            cell_h=cell_h,
            cell_w=cell_w,
        )
    else:
        decoy_row = random.choice([row for row in range(cell_h) if row != shared_row])
        _open_vertical_gate(
            grid,
            cell_r=first_relay_cell[0],
            left_cell_c=first_relay_cell[1],
            local_r=decoy_row,
            cell_h=cell_h,
            cell_w=cell_w,
        )

    used_cells = {cell for cell, _local, _color in placements}
    extra_relays = random.choice([0, 0, 1])
    if extra_relays:
        spare = _available_cells(cell_rows, cell_cols, used_cells)
        if spare:
            extra_cell = random.choice(spare)
            extra_row = random.randrange(cell_h)
            extra_col = random.randrange(cell_w)
            rr, cc = _cell_coord(extra_cell[0], extra_cell[1], extra_row, extra_col, cell_h, cell_w)
            grid[rr, cc] = int(random.choice([color for color in RELAY_PALETTE if color not in relay_colors]))

    transformed = _apply_random_transform(grid)

    correct = main(transformed)
    bg, separator, source, relay_positions = _infer_structure(transformed)
    normal_out, normal_activated, normal_touched = _propagate(
        transformed,
        separator=separator,
        bg=bg,
        source=source,
        relay_positions=relay_positions,
    )
    source_only_out, _source_only_activated, source_only_touched = _propagate(
        transformed,
        separator=separator,
        bg=bg,
        source=source,
        relay_positions=relay_positions,
        allow_activation=False,
    )
    all_active_out, _all_active_activated, _all_active_touched = _propagate(
        transformed,
        separator=separator,
        bg=bg,
        source=source,
        relay_positions=relay_positions,
        initial_active_relays=relay_positions,
    )
    ignore_walls_out, _ignore_walls_activated, _ignore_walls_touched = _propagate(
        transformed,
        separator=separator,
        bg=bg,
        source=source,
        relay_positions=relay_positions,
        respect_walls=False,
    )

    if not np.array_equal(correct, normal_out):
        return None
    if len(normal_activated) < 2:
        return None
    if not (normal_activated - source_only_touched):
        return None
    if not (relay_positions - normal_activated):
        return None
    if np.array_equal(normal_out, source_only_out):
        return None
    if np.array_equal(normal_out, all_active_out):
        return None
    if np.array_equal(normal_out, ignore_walls_out):
        return None

    painted = int(np.count_nonzero((transformed == bg) & (normal_out == SOURCE_COLOR)))
    if painted < 8:
        return None
    if painted > transformed.size // 2:
        return None
    if len(normal_touched) < 2:
        return None

    return transformed


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sample = _generate_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid gate-light cascade puzzle")
