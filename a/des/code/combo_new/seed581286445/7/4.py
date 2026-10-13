from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
GATE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

WIRE_COUNT_RANGE = (2, 5)
HEIGHT_CAP = 30
WIDTH_CAP = 30
WIRE_GAP = 3
MIN_GATE_GAP = 2
GATE_COUNT_RANGE = (2, 6)
GENERATION_ATTEMPTS = 400


def _cancel_adjacent_pairs(colors: list[int]) -> list[int]:
    """Repeatedly delete adjacent equal pairs via the equivalent stack process."""
    stack: list[int] = []
    for color in colors:
        if stack and stack[-1] == color:
            stack.pop()
        else:
            stack.append(color)
    return stack


def _extract_wire_sequences(grid: np.ndarray) -> dict[int, list[int]]:
    """Map each wire row to its left-to-right gate-color sequence."""
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    wires: dict[int, list[tuple[int, int]]] = {}
    for row in range(1, height - 1):
        if int(g[row, 0]) != GREY or int(g[row, width - 1]) != GREY:
            continue
        gates: list[tuple[int, int]] = []
        for col in range(width):
            color = int(g[row, col])
            if color in (BLACK, GREY):
                continue
            if int(g[row - 1, col]) == color and int(g[row + 1, col]) == color:
                gates.append((col, color))
        if gates:
            gates.sort()
            wires[row] = [color for _col, color in gates]
    return wires


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine each wire terminal color by canceling adjacent equal gate colors along the wire.

    Input structure:
    - The background is black.
    - One-pixel-thick grey horizontal wires run from the left edge to the right edge, with at least two blank rows between wires.
    - Each gate is a vertical 3x1 monochrome block in a non-black, non-grey color, and its middle cell lies on exactly one wire.
    - The rightmost cell of every wire is a grey terminal pixel.

    Rule:
    - For each wire, read the colors of its crossed gates from left to right.
    - Repeatedly remove any adjacent pair of equal colors from that sequence until no such pair remains.
    - If colors remain, recolor the wire's rightmost terminal cell with the last remaining color.
    - If the whole sequence cancels away, keep the terminal grey and leave every other cell unchanged.

    Key insight:
    - The spatial order of gates encodes a sequence, and the final terminal color is the rightmost survivor after stack-like pair cancellation rather than simply the last gate seen.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for row, colors in _extract_wire_sequences(g).items():
        reduced = _cancel_adjacent_pairs(colors)
        if reduced:
            out[row, -1] = int(reduced[-1])
    return out


def _sample_spaced_positions(start: int, end: int, count: int, min_gap: int) -> list[int] | None:
    """Sample sorted inclusive positions with pairwise distance at least min_gap."""
    assert count >= 0
    if count == 0:
        return []
    base_end = end - (min_gap - 1) * (count - 1)
    if base_end < start:
        return None
    base_positions = sorted(random.sample(range(start, base_end + 1), count))
    return [base + (min_gap - 1) * idx for idx, base in enumerate(base_positions)]


def _choose_palette() -> list[int]:
    color_count = random.randint(3, 6)
    return random.sample(list(GATE_COLORS), color_count)


def _reduced_sequence(length: int, palette: list[int]) -> list[int]:
    seq: list[int] = []
    for _ in range(length):
        choices = [color for color in palette if not seq or color != seq[-1]]
        seq.append(random.choice(choices))
    return seq


def _insert_canceling_pair(seq: list[int], palette: list[int]) -> list[int]:
    pos = random.randint(0, len(seq))
    left = seq[pos - 1] if pos > 0 else None
    right = seq[pos] if pos < len(seq) else None
    choices = [color for color in palette if color != left and color != right]
    if not choices:
        choices = list(palette)
    pair_color = random.choice(choices)
    return seq[:pos] + [pair_color, pair_color] + seq[pos:]


def _build_gate_sequence(reduces_to_empty: bool, palette: list[int]) -> list[int]:
    if reduces_to_empty:
        reduced: list[int] = []
        target_len = random.choice((2, 4, 6))
    else:
        reduced_len = random.randint(1, 4)
        reduced = _reduced_sequence(reduced_len, palette)
        target_options = [
            total_len
            for total_len in range(max(2, reduced_len), GATE_COUNT_RANGE[1] + 1)
            if total_len > reduced_len and total_len % 2 == reduced_len % 2
        ]
        target_len = random.choice(target_options)

    seq = list(reduced)
    while len(seq) < target_len:
        seq = _insert_canceling_pair(seq, palette)
    assert GATE_COUNT_RANGE[0] <= len(seq) <= GATE_COUNT_RANGE[1]
    assert _cancel_adjacent_pairs(seq) == reduced
    return seq


def _expected_output(grid: np.ndarray, wire_rows: list[int], sequences: list[list[int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for row, seq in zip(wire_rows, sequences):
        reduced = _cancel_adjacent_pairs(seq)
        if reduced:
            out[row, -1] = int(reduced[-1])
    return out


def _render_case(height: int, width: int, wire_rows: list[int], sequences: list[list[int]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for row, seq in zip(wire_rows, sequences):
        grid[row, :] = GREY
        gate_cols = _sample_spaced_positions(2, width - 3, len(seq), MIN_GATE_GAP)
        assert gate_cols is not None
        for col, color in zip(gate_cols, seq):
            grid[row - 1 : row + 2, col] = int(color)
    return grid


def _make_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    wire_count = random.randint(*WIRE_COUNT_RANGE)
    max_height = min(HEIGHT_CAP, 3 * wire_count + 12)
    min_height = max(3 * wire_count, 6)
    if min_height > max_height:
        return None

    branch_flags = [True] * random.randint(1, wire_count - 1) + [False] * wire_count
    branch_flags = branch_flags[:wire_count]
    random.shuffle(branch_flags)
    if all(branch_flags) or not any(branch_flags):
        return None

    palette = _choose_palette()
    sequences = [_build_gate_sequence(flag, palette) for flag in branch_flags]

    max_gates = max(len(seq) for seq in sequences)
    min_width = max(2 * max_gates + 3, 10)
    max_width = min(WIDTH_CAP, min_width + 12)
    if min_width > max_width:
        return None

    height = random.randint(min_height, max_height)
    width = random.randint(min_width, max_width)
    wire_rows = _sample_spaced_positions(1, height - 2, wire_count, WIRE_GAP)
    if wire_rows is None:
        return None

    grid = _render_case(height, width, wire_rows, sequences)
    expected = _expected_output(grid, wire_rows, sequences)

    reduced_lengths = [_cancel_adjacent_pairs(seq) for seq in sequences]
    if not any(len(seq) == 0 for seq in reduced_lengths):
        return None
    if not any(len(seq) > 0 for seq in reduced_lengths):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        attempt = _make_attempt()
        if attempt is None:
            continue
        grid, expected = attempt
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GENERATION_ATTEMPTS} attempts")
