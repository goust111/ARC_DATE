from __future__ import annotations

import random

import numpy as np


HOLE = 0
PALETTE = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (5, 10)
GRID_WIDTH_RANGE = (6, 12)
PERIOD_RANGE = (4, 9)
HOLE_COUNT_RANGE = (3, 7)
GENERATION_RETRIES = 500


def _spiral_layers(height: int, width: int) -> list[list[tuple[int, int]]]:
    layers: list[list[tuple[int, int]]] = []
    top, left = 0, 0
    bottom, right = height - 1, width - 1
    while top <= bottom and left <= right:
        coords: list[tuple[int, int]] = []
        for c in range(left, right + 1):
            coords.append((top, c))
        for r in range(top + 1, bottom + 1):
            coords.append((r, right))
        if bottom > top:
            for c in range(right - 1, left - 1, -1):
                coords.append((bottom, c))
        if right > left:
            for r in range(bottom - 1, top, -1):
                coords.append((r, left))
        layers.append(coords)
        top += 1
        left += 1
        bottom -= 1
        right -= 1
    return layers


def _spiral_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for layer in _spiral_layers(height, width):
        coords.extend(layer)
    return coords


def _full_sequence_period(sequence: list[int]) -> int:
    n = len(sequence)
    for period in range(1, n + 1):
        ok = True
        for i, value in enumerate(sequence):
            if value != sequence[i % period]:
                ok = False
                break
        if ok:
            return period
    raise AssertionError("unreachable")


def _infer_period_from_partial(sequence: list[int]) -> int:
    n = len(sequence)
    holes = [i for i, value in enumerate(sequence) if value == HOLE]
    for period in range(1, n):
        phase_to_color: dict[int, int] = {}
        ok = True
        for i, value in enumerate(sequence):
            if value == HOLE:
                continue
            phase = i % period
            prev = phase_to_color.get(phase)
            if prev is None:
                phase_to_color[phase] = int(value)
            elif prev != int(value):
                ok = False
                break
        if not ok:
            continue
        if all((i % period) in phase_to_color for i in holes):
            return period
    raise ValueError("could not infer a valid spiral period")


def _sequence_to_grid(
    shape: tuple[int, int], coords: list[tuple[int, int]], sequence: list[int]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for (r, c), value in zip(coords, sequence):
        out[r, c] = int(value)
    return out


def _periodic_candidates(values: list[int], target_index: int) -> set[int]:
    n = len(values)
    candidates: set[int] = set()
    for period in range(1, n):
        phase_to_color: dict[int, int] = {}
        ok = True
        for i, value in enumerate(values):
            if i == target_index or value == HOLE:
                continue
            phase = i % period
            prev = phase_to_color.get(phase)
            if prev is None:
                phase_to_color[phase] = int(value)
            elif prev != int(value):
                ok = False
                break
        if ok and (target_index % period) in phase_to_color:
            candidates.add(int(phase_to_color[target_index % period]))
    return candidates


def _ring_lookup(layers: list[list[tuple[int, int]]]) -> dict[tuple[int, int], tuple[int, int]]:
    lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for ring_id, layer in enumerate(layers):
        for idx, coord in enumerate(layer):
            lookup[coord] = (ring_id, idx)
    return lookup


def _choose_dimensions() -> tuple[int, int]:
    for _ in range(100):
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        if height != width and max(height, width) <= 30:
            return height, width
    raise RuntimeError("failed to choose non-square dimensions")


def _choose_period(height: int, width: int, ring_lengths: list[int], n: int) -> int | None:
    candidates = []
    low, high = PERIOD_RANGE
    for period in range(low, min(high, n // 3) + 1):
        if period in (height, width):
            continue
        if any(ring_length == period or ring_length % period == 0 for ring_length in ring_lengths):
            continue
        if n // period < 3:
            continue
        candidates.append(period)
    if not candidates:
        return None
    return random.choice(candidates)


def _random_pattern(period: int) -> list[int]:
    for _ in range(200):
        color_count = random.randint(3, min(5, period))
        palette = random.sample(PALETTE, color_count)
        pattern = [random.choice(palette) for _ in range(period)]
        if len(set(pattern)) < 3:
            continue
        if _full_sequence_period(pattern) != period:
            continue
        return pattern
    raise RuntimeError("failed to build a primitive period pattern")


def _choose_holes(
    sequence_length: int, period: int, ring_by_index: list[int]
) -> list[int] | None:
    phase_positions = {
        phase: list(range(phase, sequence_length, period))
        for phase in range(period)
    }
    usable_phases = [phase for phase, positions in phase_positions.items() if len(positions) >= 2]
    if len(usable_phases) < HOLE_COUNT_RANGE[0]:
        return None

    hole_count = random.randint(HOLE_COUNT_RANGE[0], min(HOLE_COUNT_RANGE[1], len(usable_phases)))
    chosen_phases = random.sample(usable_phases, hole_count)
    holes: list[int] = []

    outer_candidates = []
    inner_candidates = []
    for phase in chosen_phases:
        positions = phase_positions[phase]
        pick = random.choice(positions)
        holes.append(pick)
        if ring_by_index[pick] == 0:
            outer_candidates.append(pick)
        else:
            inner_candidates.append(pick)

    if not outer_candidates or not inner_candidates:
        return None
    return sorted(holes)


def _validate_instance(
    inp: np.ndarray, out: np.ndarray, expected_period: int | None = None
) -> None:
    assert isinstance(inp, np.ndarray)
    assert isinstance(out, np.ndarray)
    assert inp.shape == out.shape
    assert inp.ndim == 2
    assert inp.shape[0] != inp.shape[1]
    assert max(inp.shape) <= 30

    layers = _spiral_layers(*inp.shape)
    coords = [coord for layer in layers for coord in layer]
    ring_lookup = _ring_lookup(layers)
    ring_by_index = [ring_lookup[coord][0] for coord in coords]
    inp_sequence = [int(inp[r, c]) for r, c in coords]
    out_sequence = [int(out[r, c]) for r, c in coords]

    holes = [i for i, value in enumerate(inp_sequence) if value == HOLE]
    assert len(holes) >= HOLE_COUNT_RANGE[0]
    assert all(value != HOLE for value in out_sequence)
    assert _full_sequence_period(out_sequence) < len(out_sequence)

    period = _infer_period_from_partial(inp_sequence)
    assert _full_sequence_period(out_sequence) == period
    if expected_period is not None:
        assert period == expected_period

    ring_lengths = [len(layer) for layer in layers]
    assert period not in (inp.shape[0], inp.shape[1])
    assert all(ring_length != period and ring_length % period != 0 for ring_length in ring_lengths)

    row_fail = False
    col_fail = False
    ring_fail = False
    triple_fail = False

    for index in holes:
        r, c = coords[index]
        true_color = int(out_sequence[index])
        phase = index % period
        same_phase_visible = []
        if index - period >= 0 and inp_sequence[index - period] != HOLE:
            same_phase_visible.append(int(inp_sequence[index - period]))
        if index + period < len(inp_sequence) and inp_sequence[index + period] != HOLE:
            same_phase_visible.append(int(inp_sequence[index + period]))
        assert same_phase_visible
        assert all(color == true_color for color in same_phase_visible)

        row_candidates = _periodic_candidates([int(v) for v in inp[r, :]], c)
        col_candidates = _periodic_candidates([int(v) for v in inp[:, c]], r)

        ring_id, ring_index = ring_lookup[(r, c)]
        ring_values = [int(inp[rr, cc]) for rr, cc in layers[ring_id]]
        ring_candidates = _periodic_candidates(ring_values, ring_index)

        row_bad = row_candidates != {true_color}
        col_bad = col_candidates != {true_color}
        ring_bad = ring_candidates != {true_color}

        row_fail |= row_bad
        col_fail |= col_bad
        ring_fail |= ring_bad
        triple_fail |= row_bad and col_bad and ring_bad

    assert row_fail and col_fail and ring_fail and triple_fail


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair black holes by recovering the shortest color cycle along the outer-to-inner clockwise spiral.

    Input structure:
    - A non-square rectangular grid is fully colored except for several black holes (color 0).
    - Reading the cells in clockwise spiral order from the top-left outer corner produces a 1D sequence.
    - The visible cells on that spiral follow one shortest repeating period, while row, column, and ring views are misleading.

    Rule:
    - Convert the grid into its clockwise outer-to-inner spiral sequence.
    - Infer the shortest period whose phase classes are consistent with all visible colors.
    - Fill each black hole with the visible color from the same spiral phase, then write the repaired sequence back into the grid.

    Key insight:
    - The only stable coordinate system is the spiral index itself: the color cycle crosses ring boundaries, so the grid must be unrolled before the period can be recovered.
    """
    g = np.asarray(grid, dtype=int)
    coords = _spiral_coords(*g.shape)
    sequence = [int(g[r, c]) for r, c in coords]
    period = _infer_period_from_partial(sequence)

    phase_to_color: dict[int, int] = {}
    for i, value in enumerate(sequence):
        if value == HOLE:
            continue
        phase = i % period
        prev = phase_to_color.get(phase)
        if prev is None:
            phase_to_color[phase] = int(value)
        elif prev != int(value):
            raise ValueError("inconsistent visible colors for the inferred spiral period")

    repaired = list(sequence)
    for i, value in enumerate(repaired):
        if value == HOLE:
            repaired[i] = int(phase_to_color[i % period])
    return _sequence_to_grid(g.shape, coords, repaired)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        height, width = _choose_dimensions()
        layers = _spiral_layers(height, width)
        ring_lengths = [len(layer) for layer in layers]
        coords = [coord for layer in layers for coord in layer]
        n = len(coords)

        period = _choose_period(height, width, ring_lengths, n)
        if period is None:
            continue

        pattern = _random_pattern(period)
        full_sequence = [int(pattern[i % period]) for i in range(n)]

        ring_lookup = _ring_lookup(layers)
        ring_by_index = [ring_lookup[coord][0] for coord in coords]
        holes = _choose_holes(n, period, ring_by_index)
        if holes is None:
            continue

        partial_sequence = list(full_sequence)
        for index in holes:
            partial_sequence[index] = HOLE

        inp = _sequence_to_grid((height, width), coords, partial_sequence)
        out = _sequence_to_grid((height, width), coords, full_sequence)

        try:
            if not np.array_equal(main(inp), out):
                continue
            _validate_instance(inp, out, expected_period=period)
        except (AssertionError, ValueError):
            continue
        return inp

    raise RuntimeError("failed to generate a valid spiral-period repair instance")
