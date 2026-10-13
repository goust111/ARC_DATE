from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3


UP, RIGHT, DOWN, LEFT = range(4)
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

LEFT_TURN_COLOR = Color.BLUE
RIGHT_TURN_COLOR = Color.GREEN

MIN_BLOCKS = 3
MAX_BLOCKS = 6
MIN_MARGIN = 1
MAX_MARGIN = 4
MIN_TURNS = 3
MIN_STEPS = 10
GEN_RETRIES = 400


@dataclass(frozen=True)
class LatticeInfo:
    start: tuple[int, int]
    start_dir: int
    block_colors: tuple[int, int]
    row_gutters: np.ndarray
    col_gutters: np.ndarray


@dataclass(frozen=True)
class TraceResult:
    success: bool
    output: np.ndarray | None
    steps: int
    turns: tuple[str, ...]
    reason: str


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _true_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    i = 0
    n = int(mask.shape[0])
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask[j + 1]):
            j += 1
        intervals.append((i, j))
        i = j + 1
    return intervals


def _boundary_start(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], int]:
    edge_mask = np.zeros(grid.shape, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    coords = np.argwhere(edge_mask & (grid != bg))
    assert coords.shape == (1, 2), "expected a unique non-background boundary start pixel"
    r, c = map(int, coords[0])
    assert int(grid[r, c]) == Color.RED, "the boundary start pixel must be red"
    if r == 0:
        return (r, c), DOWN
    if r == grid.shape[0] - 1:
        return (r, c), UP
    if c == 0:
        return (r, c), RIGHT
    return (r, c), LEFT


def _parse_lattice(grid: np.ndarray) -> LatticeInfo:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    start, start_dir = _boundary_start(g, bg)

    block_mask = g != bg
    block_mask[start] = False
    assert bool(block_mask.any()), "expected a block lattice"

    block_colors = tuple(sorted(int(v) for v in np.unique(g[block_mask])))
    assert len(block_colors) == 2, "expected exactly two block colors"

    row_bands = _true_intervals(block_mask.any(axis=1))
    col_bands = _true_intervals(block_mask.any(axis=0))
    assert len(row_bands) >= 2 and len(col_bands) >= 2
    assert all(e - s + 1 == 2 for s, e in row_bands)
    assert all(e - s + 1 == 2 for s, e in col_bands)
    assert all(nxt[0] == prev[1] + 2 for prev, nxt in zip(row_bands, row_bands[1:]))
    assert all(nxt[0] == prev[1] + 2 for prev, nxt in zip(col_bands, col_bands[1:]))

    row_band_mask = np.zeros(g.shape[0], dtype=bool)
    col_band_mask = np.zeros(g.shape[1], dtype=bool)
    for s, e in row_bands:
        row_band_mask[s : e + 1] = True
    for s, e in col_bands:
        col_band_mask[s : e + 1] = True
    expected_blocks = row_band_mask[:, None] & col_band_mask[None, :]
    assert np.array_equal(block_mask, expected_blocks), "expected a complete rectangular array of 2x2 blocks"

    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            vals = np.unique(patch)
            assert vals.size == 1 and int(vals[0]) in block_colors

    row_gutters = np.zeros(g.shape[0], dtype=bool)
    col_gutters = np.zeros(g.shape[1], dtype=bool)
    for prev, nxt in zip(row_bands, row_bands[1:]):
        row_gutters[prev[1] + 1] = True
    for prev, nxt in zip(col_bands, col_bands[1:]):
        col_gutters[prev[1] + 1] = True

    sr, sc = start
    assert bool(row_gutters[sr] or col_gutters[sc]), "start must lie on a gutter extension"

    return LatticeInfo(
        start=start,
        start_dir=start_dir,
        block_colors=block_colors,
        row_gutters=row_gutters,
        col_gutters=col_gutters,
    )


def _trace_path(grid: np.ndarray, info: LatticeInfo, left_turn_color: int) -> TraceResult:
    if left_turn_color not in info.block_colors:
        return TraceResult(False, None, 0, (), "unknown-left-color")

    walkable = info.row_gutters[:, None] | info.col_gutters[None, :]
    out = np.asarray(grid, dtype=int).copy()
    r, c = info.start
    direction = info.start_dir
    steps = 0
    turns: list[str] = []
    seen: set[tuple[int, int, int]] = set()

    while True:
        state = (r, c, direction)
        if state in seen:
            return TraceResult(False, None, steps, tuple(turns), "loop")
        seen.add(state)
        out[r, c] = Color.RED

        if bool(info.row_gutters[r] and info.col_gutters[c]):
            left_dir = (direction - 1) % 4
            fr_r = r + DIRS[direction][0] + DIRS[left_dir][0]
            fr_c = c + DIRS[direction][1] + DIRS[left_dir][1]
            if not (0 <= fr_r < grid.shape[0] and 0 <= fr_c < grid.shape[1]):
                return TraceResult(False, None, steps, tuple(turns), "front-left-oob")
            front_left_color = int(grid[fr_r, fr_c])
            if front_left_color == left_turn_color:
                direction = left_dir
                turns.append("L")
            elif front_left_color in info.block_colors:
                direction = (direction + 1) % 4
                turns.append("R")
            else:
                return TraceResult(False, None, steps, tuple(turns), "front-left-not-block")

        nr = r + DIRS[direction][0]
        nc = c + DIRS[direction][1]
        steps += 1
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            return TraceResult(True, out, steps, tuple(turns), "exit")
        if not bool(walkable[nr, nc]):
            return TraceResult(False, None, steps, tuple(turns), "left-gutter")
        r, c = nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the routed red path through a gutter lattice of 2x2 colored blocks.

    Input structure:
    - The grid contains a complete rectangular array of separated 2x2 blocks in
      exactly two colors, with one-cell-wide gutters between neighboring block rows
      and columns.
    - A single red start pixel sits on the outer boundary at the entrance of one
      gutter line, facing into the lattice.

    Rule:
    - Move the red path straight along the current gutter and paint every visited
      gutter cell red.
    - Whenever the path reaches a gutter crossroads, inspect the block in the
      front-left diagonal position relative to the current heading: one block color
      means turn left and the other means turn right.
    - Continue until the path exits the grid, leaving the blocks unchanged.

    Key insight:
    - The two block colors are route signs consumed one junction at a time, and the
      lookup frame is relative to the traveler heading rather than absolute north,
      south, east, or west.
    """
    g = np.asarray(grid, dtype=int)
    info = _parse_lattice(g)

    traces = {
        color: _trace_path(g, info, color)
        for color in info.block_colors
    }
    successful = [color for color, trace in traces.items() if trace.success]

    if len(successful) == 1:
        return traces[successful[0]].output

    if LEFT_TURN_COLOR in successful and RIGHT_TURN_COLOR in info.block_colors:
        return traces[LEFT_TURN_COLOR].output

    if len(successful) == 2:
        first, second = successful
        if np.array_equal(traces[first].output, traces[second].output):
            return traces[first].output
        preferred = max(successful, key=lambda color: (len(traces[color].turns), traces[color].steps))
        return traces[preferred].output

    raise AssertionError(f"no valid routing interpretation found: {traces}")


def _candidate_starts(n_block_rows: int, n_block_cols: int) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    for side in (UP, DOWN):
        for lane in range(n_block_cols - 1):
            starts.append((side, lane))
    for side in (LEFT, RIGHT):
        for lane in range(n_block_rows - 1):
            starts.append((side, lane))
    random.shuffle(starts)
    return starts


def _render_input(layout: np.ndarray, side: int, lane: int, margins: tuple[int, int, int, int]) -> np.ndarray:
    n_block_rows, n_block_cols = map(int, layout.shape)
    top, bottom, left, right = map(int, margins)
    h = top + (3 * n_block_rows - 1) + bottom
    w = left + (3 * n_block_cols - 1) + right
    grid = np.full((h, w), Color.BLACK, dtype=int)

    for br in range(n_block_rows):
        for bc in range(n_block_cols):
            r0 = top + 3 * br
            c0 = left + 3 * bc
            grid[r0 : r0 + 2, c0 : c0 + 2] = int(layout[br, bc])

    if side == UP:
        grid[0, left + 3 * lane + 2] = Color.RED
    elif side == DOWN:
        grid[h - 1, left + 3 * lane + 2] = Color.RED
    elif side == LEFT:
        grid[top + 3 * lane + 2, 0] = Color.RED
    else:
        grid[top + 3 * lane + 2, w - 1] = Color.RED

    return grid


def _sample_layout() -> np.ndarray:
    n_block_rows = random.randint(MIN_BLOCKS, MAX_BLOCKS)
    n_block_cols = random.randint(MIN_BLOCKS, MAX_BLOCKS)
    layout = np.random.choice([LEFT_TURN_COLOR, RIGHT_TURN_COLOR], size=(n_block_rows, n_block_cols))
    if np.all(layout == LEFT_TURN_COLOR) or np.all(layout == RIGHT_TURN_COLOR):
        layout[random.randrange(n_block_rows), random.randrange(n_block_cols)] = (
            RIGHT_TURN_COLOR if int(layout[0, 0]) == LEFT_TURN_COLOR else LEFT_TURN_COLOR
        )
    return layout.astype(int)


def _random_margins() -> tuple[int, int, int, int]:
    return tuple(random.randint(MIN_MARGIN, MAX_MARGIN) for _ in range(4))


def _valid_training_case(layout: np.ndarray, side: int, lane: int) -> bool:
    probe_grid = _render_input(layout, side, lane, (1, 1, 1, 1))
    info = _parse_lattice(probe_grid)
    true_trace = _trace_path(probe_grid, info, LEFT_TURN_COLOR)
    false_trace = _trace_path(probe_grid, info, RIGHT_TURN_COLOR)

    if not true_trace.success or false_trace.success:
        return False
    if true_trace.steps < MIN_STEPS or len(true_trace.turns) < MIN_TURNS:
        return False
    if "L" not in true_trace.turns or "R" not in true_trace.turns:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        layout = _sample_layout()
        for side, lane in _candidate_starts(*layout.shape):
            if not _valid_training_case(layout, side, lane):
                continue

            grid = _render_input(layout, side, lane, _random_margins())
            info = _parse_lattice(grid)
            expected = _trace_path(grid, info, LEFT_TURN_COLOR)
            reversed_trace = _trace_path(grid, info, RIGHT_TURN_COLOR)
            if not expected.success or reversed_trace.success:
                continue

            out = main(grid)
            if np.array_equal(out, expected.output):
                return grid

    raise RuntimeError("failed to generate a valid gutter-routing input")
