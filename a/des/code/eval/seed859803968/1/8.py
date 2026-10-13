import random
from typing import Dict, List, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
MAROON = 9

SIGNAL_COLORS = (BLUE, RED)
GATE_COLORS = (GREEN, ORANGE)
GEN_ATTEMPTS = 300


def _find_gates(grid: np.ndarray) -> Dict[Tuple[int, int], int]:
    gates: Dict[Tuple[int, int], int] = {}
    height, width = grid.shape
    for r in range(height - 1):
        for c in range(width - 1):
            color = int(grid[r, c])
            if color not in GATE_COLORS:
                continue
            block = grid[r : r + 2, c : c + 2]
            if np.all(block == color):
                gates[(r, c)] = color
    return gates


def _validate_layout(grid: np.ndarray, gate_rows: List[int], gate_cols: List[int], gates: Dict[Tuple[int, int], int]) -> None:
    height, width = grid.shape
    assert gate_rows and gate_cols
    assert len(gates) == len(gate_rows) * len(gate_cols)

    for r in gate_rows:
        assert int(grid[r, 0]) in SIGNAL_COLORS
        assert int(grid[r, width - 1]) == MAROON
        for c in range(1, width - 1):
            if any(gc <= c <= gc + 1 for gc in gate_cols):
                continue
            assert int(grid[r, c]) == GREY

    for c in gate_cols:
        assert int(grid[0, c]) in SIGNAL_COLORS
        assert int(grid[height - 1, c]) == MAROON
        for r in range(1, height - 1):
            if any(gr <= r <= gr + 1 for gr in gate_rows):
                continue
            assert int(grid[r, c]) == GREY

    for r in gate_rows:
        for c in gate_cols:
            assert (r, c) in gates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate colored signals moving along grey wires through 2x2 swap/keep gates.

    The supported layout is a rectangular gate matrix:
    - left-edge red/blue pixels feed horizontal wires,
    - top-edge red/blue pixels feed vertical wires,
    - every gate-row/gate-column intersection contains a 2x2 orange or green gate,
    - right/bottom maroon lamps are replaced by the arriving signal color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height >= 8 and width >= 8

    gates = _find_gates(g)
    assert gates
    gate_rows = sorted({r for r, _ in gates})
    gate_cols = sorted({c for _, c in gates})
    _validate_layout(g, gate_rows, gate_cols, gates)

    out = g.copy()
    down_colors = [int(g[0, c]) for c in gate_cols]
    vertical_starts = [0] * len(gate_cols)

    for r in gate_rows:
        right_color = int(g[r, 0])
        horiz_start = 0

        for j, c in enumerate(gate_cols):
            top_color = down_colors[j]
            out[r, horiz_start:c] = right_color
            out[vertical_starts[j]:r, c] = top_color

            gate_type = int(gates[(r, c)])
            if gate_type == ORANGE:
                right_color, down_colors[j] = top_color, right_color
            else:
                assert gate_type == GREEN
                down_colors[j] = top_color

            horiz_start = c + 2
            vertical_starts[j] = r + 2

        out[r, horiz_start:width] = right_color

    for j, c in enumerate(gate_cols):
        out[vertical_starts[j]:height, c] = down_colors[j]

    return out


def _simulate_gate_matrix(
    row_inputs: List[int], col_inputs: List[int], gate_types: List[List[int]]
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], List[int], List[int]]:
    down = list(col_inputs)
    gate_inputs: Dict[Tuple[int, int], Tuple[int, int]] = {}
    row_outputs: List[int] = []

    for i, left_seed in enumerate(row_inputs):
        right = left_seed
        for j in range(len(col_inputs)):
            top = down[j]
            gate_inputs[(i, j)] = (right, top)
            if gate_types[i][j] == ORANGE:
                right, down[j] = top, right
            else:
                down[j] = top
        row_outputs.append(right)

    return gate_inputs, row_outputs, down


def _sample_signal_colors(count: int) -> List[int]:
    colors = [random.choice(SIGNAL_COLORS) for _ in range(count)]
    if count >= 2 and len(set(colors)) == 1:
        flip = random.randrange(count)
        colors[flip] = RED if colors[flip] == BLUE else BLUE
    return colors


def _sample_gate_types(rows: int, cols: int) -> List[List[int]]:
    total = rows * cols
    flat = [random.choice(GATE_COLORS) for _ in range(total)]
    if total >= 2 and len(set(flat)) == 1:
        flip = random.randrange(total)
        flat[flip] = GREEN if flat[flip] == ORANGE else ORANGE
    return [flat[i * cols : (i + 1) * cols] for i in range(rows)]


def _choose_positions(count: int, start_range: Tuple[int, int], gap_range: Tuple[int, int]) -> List[int]:
    positions = [random.randint(*start_range)]
    for _ in range(1, count):
        positions.append(positions[-1] + random.randint(*gap_range))
    return positions


def _build_input(
    gate_rows: List[int],
    gate_cols: List[int],
    row_inputs: List[int],
    col_inputs: List[int],
    gate_types: List[List[int]],
) -> np.ndarray:
    height = gate_rows[-1] + 3 + random.randint(1, 3)
    width = gate_cols[-1] + 3 + random.randint(1, 3)
    grid = np.full((height, width), BLACK, dtype=int)

    for r in gate_rows:
        grid[r, 1 : width - 1] = GREY
    for c in gate_cols:
        grid[1 : height - 1, c] = GREY

    for r, color in zip(gate_rows, row_inputs):
        grid[r, 0] = color
        grid[r, width - 1] = MAROON

    for c, color in zip(gate_cols, col_inputs):
        grid[0, c] = color
        grid[height - 1, c] = MAROON

    for i, r in enumerate(gate_rows):
        for j, c in enumerate(gate_cols):
            grid[r : r + 2, c : c + 2] = gate_types[i][j]

    return grid


def _is_interesting(gate_inputs: Dict[Tuple[int, int], Tuple[int, int]], gate_types: List[List[int]]) -> bool:
    seen_swap = False
    seen_keep = False
    for (i, j), (left, top) in gate_inputs.items():
        if left == top:
            continue
        if gate_types[i][j] == ORANGE:
            seen_swap = True
        else:
            seen_keep = True
    return seen_swap and seen_keep


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(2, 4)
    col_count = random.randint(2, 4)

    gate_rows = _choose_positions(row_count, (3, 5), (4, 6))
    gate_cols = _choose_positions(col_count, (3, 5), (4, 6))
    row_inputs = _sample_signal_colors(row_count)
    col_inputs = _sample_signal_colors(col_count)
    gate_types = _sample_gate_types(row_count, col_count)

    if set(row_inputs + col_inputs) != set(SIGNAL_COLORS):
        return None

    gate_inputs, row_outputs, col_outputs = _simulate_gate_matrix(row_inputs, col_inputs, gate_types)
    if not _is_interesting(gate_inputs, gate_types):
        return None
    if set(row_outputs + col_outputs) != set(SIGNAL_COLORS):
        return None

    grid = _build_input(gate_rows, gate_cols, row_inputs, col_inputs, gate_types)
    out = main(grid)

    if np.array_equal(out, grid):
        return None
    if not np.any(out == RED) or not np.any(out == BLUE):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
