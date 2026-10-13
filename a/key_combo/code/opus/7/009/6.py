from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
PINK = 6

CELL_SIZE = 2
SIGNATURE_BAG = [
    (True, True, True),
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (False, True, False),
    (False, False, True),
]
MIN_CELL_COUNT = 5
MAX_CELL_COUNT = 8
MIN_SOURCES = 2
MAX_SOURCES = 4

_CONFIG_POOLS: dict[tuple[bool, bool, bool], list[tuple[int, tuple[int, ...], tuple[int, ...]]]] | None = None


def _cell_bands_from_walls(walls: list[int], limit: int) -> list[tuple[int, int]]:
    assert walls == sorted(walls)
    assert walls[0] == 0 and walls[-1] == limit - 1
    bands: list[tuple[int, int]] = []
    for left, right in zip(walls, walls[1:]):
        assert right - left >= 2
        bands.append((left + 1, right - 1))
    return bands


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    matches = [idx for idx, (lo, hi) in enumerate(bands) if lo <= pos <= hi]
    assert len(matches) == 1
    return matches[0]


def _regular_bands(cell_count: int) -> list[tuple[int, int]]:
    return [(1 + 3 * idx, 2 + 3 * idx) for idx in range(cell_count)]


def _simulate_flows(
    row_count: int,
    col_count: int,
    red_cols: tuple[int, ...],
    green_rows: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, int]]:
    state = np.zeros((row_count, col_count), dtype=int)
    red_heads = {(0, int(c)) for c in red_cols}
    green_heads = {(int(r), 0) for r in green_rows}

    for r, c in red_heads:
        assert state[r, c] == BLACK
        state[r, c] = RED
    for r, c in green_heads:
        assert state[r, c] == BLACK
        state[r, c] = GREEN

    summary = {"mixes": 0, "red_deflections": 0, "green_deflections": 0, "terminations": 0}
    step_limit = row_count * col_count * 2
    for _ in range(step_limit):
        previous = state.copy()
        arrivals: dict[tuple[int, int], int] = {}

        for r, c in red_heads:
            direct = (r + 1, c)
            if direct[0] >= row_count:
                summary["terminations"] += 1
                continue
            target = direct
            if int(previous[direct]) in (GREEN, PINK):
                summary["red_deflections"] += 1
                target = (r + 1, c + 1)
                if target[0] >= row_count or target[1] >= col_count or int(previous[target]) != BLACK:
                    summary["terminations"] += 1
                    continue
            arrivals[target] = arrivals.get(target, 0) | 1

        for r, c in green_heads:
            direct = (r, c + 1)
            if direct[1] >= col_count:
                summary["terminations"] += 1
                continue
            target = direct
            if int(previous[direct]) in (RED, PINK):
                summary["green_deflections"] += 1
                target = (r + 1, c + 1)
                if target[0] >= row_count or target[1] >= col_count or int(previous[target]) != BLACK:
                    summary["terminations"] += 1
                    continue
            arrivals[target] = arrivals.get(target, 0) | 2

        if not arrivals:
            break

        new_red_heads: set[tuple[int, int]] = set()
        new_green_heads: set[tuple[int, int]] = set()
        for (r, c), mask in arrivals.items():
            if mask == 3:
                summary["mixes"] += 1
                state[r, c] = PINK
                new_red_heads.add((r, c))
                new_green_heads.add((r, c))
            elif mask == 1:
                if int(state[r, c]) == BLACK:
                    state[r, c] = RED
                new_red_heads.add((r, c))
            else:
                if int(state[r, c]) == BLACK:
                    state[r, c] = GREEN
                new_green_heads.add((r, c))

        red_heads = new_red_heads
        green_heads = new_green_heads
        if not red_heads and not green_heads:
            break
    else:
        raise AssertionError("flow simulation exceeded step limit")

    return state, summary


def _timed_straight_state(
    row_count: int,
    col_count: int,
    red_cols: tuple[int, ...],
    green_rows: tuple[int, ...],
) -> np.ndarray:
    state = np.zeros((row_count, col_count), dtype=int)
    red_col_set = set(red_cols)
    green_row_set = set(green_rows)
    for r in range(row_count):
        for c in range(col_count):
            red_time = r if c in red_col_set else None
            green_time = c if r in green_row_set else None
            if red_time is None and green_time is None:
                continue
            if red_time is None:
                state[r, c] = GREEN
            elif green_time is None:
                state[r, c] = RED
            elif red_time == green_time:
                state[r, c] = PINK
            elif red_time < green_time:
                state[r, c] = RED
            else:
                state[r, c] = GREEN
    return state


def _geometric_union_state(
    row_count: int,
    col_count: int,
    red_cols: tuple[int, ...],
    green_rows: tuple[int, ...],
) -> np.ndarray:
    state = np.zeros((row_count, col_count), dtype=int)
    red_col_set = set(red_cols)
    green_row_set = set(green_rows)
    for r in range(row_count):
        for c in range(col_count):
            hit_red = c in red_col_set
            hit_green = r in green_row_set
            if hit_red and hit_green:
                state[r, c] = PINK
            elif hit_red:
                state[r, c] = RED
            elif hit_green:
                state[r, c] = GREEN
    return state


def _render_input_from_sources(
    cell_count: int,
    red_cols: tuple[int, ...],
    green_rows: tuple[int, ...],
) -> np.ndarray:
    size = 3 * cell_count + 1
    grid = np.full((size, size), BLACK, dtype=int)
    wall_indices = list(range(0, size, 3))
    grid[wall_indices, :] = GREY
    grid[:, wall_indices] = GREY

    bands = _regular_bands(cell_count)
    for col_idx in red_cols:
        r0, _r1 = bands[0]
        c0, _c1 = bands[col_idx]
        grid[r0, c0] = RED
    for row_idx in green_rows:
        r0, _r1 = bands[row_idx]
        c0, _c1 = bands[0]
        assert grid[r0, c0] == BLACK
        grid[r0, c0] = GREEN
    return grid


def _render_output_from_state(
    input_grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    cell_state: np.ndarray,
) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    mask = out != GREY
    out[mask] = BLACK
    for row_idx, (r0, r1) in enumerate(row_bands):
        for col_idx, (c0, c1) in enumerate(col_bands):
            color = int(cell_state[row_idx, col_idx])
            if color != BLACK:
                out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _parse_lattice(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, ...], tuple[int, ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREEN, GREY}

    height, width = g.shape
    wall_rows = [r for r in range(height) if np.all(g[r, :] == GREY)]
    wall_cols = [c for c in range(width) if np.all(g[:, c] == GREY)]
    row_bands = _cell_bands_from_walls(wall_rows, height)
    col_bands = _cell_bands_from_walls(wall_cols, width)
    assert row_bands and col_bands
    assert all((r1 - r0 + 1) == CELL_SIZE for r0, r1 in row_bands)
    assert all((c1 - c0 + 1) == CELL_SIZE for c0, c1 in col_bands)

    red_sources: list[int] = []
    green_sources: list[int] = []
    for col_idx, (c0, c1) in enumerate(col_bands):
        cell = g[row_bands[0][0] : row_bands[0][1] + 1, c0 : c1 + 1]
        if np.any(cell == RED):
            red_sources.append(col_idx)
        if col_idx != 0:
            assert not np.any(cell == GREEN)
    for row_idx, (r0, r1) in enumerate(row_bands):
        cell = g[r0 : r1 + 1, col_bands[0][0] : col_bands[0][1] + 1]
        if np.any(cell == GREEN):
            green_sources.append(row_idx)
        if row_idx != 0:
            assert not np.any(cell == RED)

    for r_raw, c_raw in np.argwhere((g == RED) | (g == GREEN)):
        r = int(r_raw)
        c = int(c_raw)
        row_idx = _band_index(r, row_bands)
        col_idx = _band_index(c, col_bands)
        if int(g[r, c]) == RED:
            assert row_idx == 0
            assert col_idx in red_sources
        else:
            assert col_idx == 0
            assert row_idx in green_sources

    assert MIN_SOURCES <= len(red_sources) <= MAX_SOURCES
    assert MIN_SOURCES <= len(green_sources) <= MAX_SOURCES
    assert not (0 in red_sources and 0 in green_sources)
    return row_bands, col_bands, tuple(red_sources), tuple(green_sources)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate top-down red flows and left-right green flows on a grey lattice, mixing simultaneous arrivals and deflecting late arrivals.

    Input structure:
    - A grey lattice with outer borders and evenly spaced internal walls divides the canvas into 2x2 black cells.
    - Some top-row cells contain a red marker pixel, which marks a downward red flow source.
    - Some left-column cells contain a green marker pixel, which marks a rightward green flow source.

    Rule:
    - Treat each lattice cell as one coarse grid unit and keep the grey walls unchanged.
    - Red heads start in their top-row source cells and move one cell downward per step; green heads start in their left-column source cells and move one cell rightward per step.
    - If a red and a green head enter the same cell in the same step, fill that whole cell pink and let both continue from there.
    - If a red head would enter a cell already occupied from an earlier step by green or pink, it deflects to the down-right neighboring cell; green does the symmetric down-right deflection when its direct target was already red or pink.
    - A deflected head stops if the deflected target is outside the lattice or was already occupied earlier; otherwise every visited cell is filled with the moving head color.

    Key insight:
    - The visible output is a real-time interaction on the lattice-cell grid: the answer is not a simple overlay of straight red columns and straight green rows because timing controls pink mixing and prior occupancy controls later deflections.
    """
    g = np.asarray(grid, dtype=int)
    row_bands, col_bands, red_cols, green_rows = _parse_lattice(g)
    cell_state, _summary = _simulate_flows(len(row_bands), len(col_bands), red_cols, green_rows)
    return _render_output_from_state(g, row_bands, col_bands, cell_state)


def _is_valid_config(cell_count: int, red_cols: tuple[int, ...], green_rows: tuple[int, ...]) -> bool:
    if len(red_cols) * len(green_rows) < 2:
        return False
    cell_state, summary = _simulate_flows(cell_count, cell_count, red_cols, green_rows)
    if np.array_equal(cell_state, _timed_straight_state(cell_count, cell_count, red_cols, green_rows)):
        return False
    if np.array_equal(cell_state, _geometric_union_state(cell_count, cell_count, red_cols, green_rows)):
        return False

    occupied = int(np.count_nonzero(cell_state))
    if occupied < max(len(red_cols), len(green_rows)) + 3:
        return False
    if occupied * 10 > cell_count * cell_count * 6:
        return False

    visible_colors = set(map(int, np.unique(cell_state))) - {BLACK}
    if len(visible_colors) < 2:
        return False

    signature = (
        summary["mixes"] > 0,
        summary["red_deflections"] > 0,
        summary["green_deflections"] > 0,
    )
    return signature in set(SIGNATURE_BAG)


def _build_config_pools() -> dict[tuple[bool, bool, bool], list[tuple[int, tuple[int, ...], tuple[int, ...]]]]:
    pools: dict[tuple[bool, bool, bool], list[tuple[int, tuple[int, ...], tuple[int, ...]]]] = {
        signature: [] for signature in set(SIGNATURE_BAG)
    }
    seen_outputs: set[tuple[int, bytes]] = set()

    for cell_count in range(MIN_CELL_COUNT, MAX_CELL_COUNT + 1):
        choices = [
            combo
            for count in range(MIN_SOURCES, min(MAX_SOURCES, cell_count) + 1)
            for combo in combinations(range(cell_count), count)
        ]
        for red_cols in choices:
            for green_rows in choices:
                if 0 in red_cols and 0 in green_rows:
                    continue
                if not _is_valid_config(cell_count, red_cols, green_rows):
                    continue
                cell_state, summary = _simulate_flows(cell_count, cell_count, red_cols, green_rows)
                signature = (
                    summary["mixes"] > 0,
                    summary["red_deflections"] > 0,
                    summary["green_deflections"] > 0,
                )
                output_key = (cell_count, cell_state.tobytes())
                if output_key in seen_outputs:
                    continue
                seen_outputs.add(output_key)
                pools[signature].append((cell_count, red_cols, green_rows))

    for signature, configs in pools.items():
        assert configs, f"missing generator configs for signature {signature}"
    return pools


def _get_config_pools() -> dict[tuple[bool, bool, bool], list[tuple[int, tuple[int, ...], tuple[int, ...]]]]:
    global _CONFIG_POOLS
    if _CONFIG_POOLS is None:
        _CONFIG_POOLS = _build_config_pools()
    return _CONFIG_POOLS


def generate_input() -> np.ndarray:
    pools = _get_config_pools()
    signature = random.choice(SIGNATURE_BAG)
    cell_count, red_cols, green_rows = random.choice(pools[signature])
    grid = _render_input_from_sources(cell_count, red_cols, green_rows)

    expected_state, summary = _simulate_flows(cell_count, cell_count, red_cols, green_rows)
    expected = _render_output_from_state(grid, _regular_bands(cell_count), _regular_bands(cell_count), expected_state)
    assert np.array_equal(main(grid), expected)
    assert (
        summary["mixes"] > 0,
        summary["red_deflections"] > 0,
        summary["green_deflections"] > 0,
    ) == signature
    assert not np.array_equal(expected_state, _timed_straight_state(cell_count, cell_count, red_cols, green_rows))
    assert not np.array_equal(expected_state, _geometric_union_state(cell_count, cell_count, red_cols, green_rows))
    return grid
