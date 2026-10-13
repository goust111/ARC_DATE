from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7

GEN_HEIGHT_RANGE = (14, 19)
GEN_WIDTH_RANGE = (16, 22)
GEN_DIVIDER_RANGE = (5, 7)
GEN_MARKER_HEIGHTS = (2, 3)
GEN_MAX_ATTEMPTS = 400


def _find_divider_row(grid: np.ndarray) -> int:
    divider_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert divider_rows.size == 1, "expected exactly one full-width grey divider"
    return int(divider_rows[0])


def _find_marker_rows(grid: np.ndarray, divider_row: int) -> list[int]:
    marker_rows = [r for r in range(divider_row) if int(grid[r, 0]) == GREY]
    assert marker_rows, "expected grey marker rows in the leftmost column"
    start = marker_rows[0]
    expected = list(range(start, start + len(marker_rows)))
    assert marker_rows == expected, "marker rows must form one contiguous run"
    assert all(r < divider_row for r in marker_rows), "marker rows must stay above the divider"
    return marker_rows


def _extract_motif(grid: np.ndarray, divider_row: int) -> tuple[list[int], np.ndarray]:
    marker_rows = _find_marker_rows(grid, divider_row)
    motif = grid[marker_rows].copy()
    motif[:, 0] = BLACK
    assert motif.shape[0] >= 2, "motif height must be at least 2"
    assert np.any(motif == ORANGE), "motif must contain orange foreground"
    return marker_rows, motif


def _lower_ray_lengths_from_input(grid: np.ndarray, divider_row: int) -> dict[int, int]:
    red_above = np.argwhere(grid[:divider_row] == RED)
    lengths: dict[int, int] = {}
    for r, c in red_above:
        rr = int(r)
        cc = int(c)
        assert cc not in lengths, "each column may contain at most one red source pixel"
        lengths[cc] = divider_row - rr - 1
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    """Use each red pixel's distance to the grey divider as the per-column phase offset for cyclic motif filling below the divider.

    Input structure:
    - The grid has a black background and one full-width horizontal grey divider row.
    - Above the divider, a contiguous grey run in the leftmost column marks 2-3 motif rows containing orange pixels.
    - Additional isolated red pixels appear above the divider, outside the marked motif rows, with at most one red source per column.

    Rule:
    - Read the marked motif rows and replace their leftmost grey marker cells with black.
    - For each red source pixel, draw a red vertical ray down to the divider and continue below the divider for the same length, without overwriting the grey divider.
    - For each column independently, count the number of red pixels below the divider in that column and start cyclically copying that motif column only after those red cells, keeping black motif cells black and preserving all existing pixels elsewhere.

    Key insight:
    - The red rays are not decorative: each ray length becomes that column's cyclic-fill phase offset, so the lower repeated motif starts on different rows in different columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_row = _find_divider_row(g)
    marker_rows, motif = _extract_motif(g, divider_row)

    out = g.copy()
    lower_height = g.shape[0] - divider_row - 1

    for c, upper_len in _lower_ray_lengths_from_input(g, divider_row).items():
        source_rows = np.flatnonzero(g[:divider_row, c] == RED)
        assert source_rows.size == 1
        source_row = int(source_rows[0])
        assert source_row not in marker_rows, "red sources must stay outside motif rows"
        assert upper_len >= 0
        assert upper_len <= lower_height, "continued ray must fit below the divider"

        if upper_len > 0:
            out[source_row + 1 : divider_row, c] = RED
            out[divider_row + 1 : divider_row + 1 + upper_len, c] = RED

    motif_h = motif.shape[0]
    for c in range(g.shape[1]):
        lower_red = int(np.count_nonzero(out[divider_row + 1 :, c] == RED))
        start_row = divider_row + 1 + lower_red
        for r in range(start_row, g.shape[0]):
            value = int(motif[(r - start_row) % motif_h, c])
            if value != BLACK and out[r, c] != RED:
                out[r, c] = value

    return out


def _build_motif(width: int, marker_height: int) -> np.ndarray:
    motif = np.zeros((marker_height, width), dtype=int)
    motif[:, 0] = GREY

    active_cols = sorted(random.sample(range(2, width - 1), random.randint(5, min(8, width - 3))))
    patterns = [
        bits
        for bits in range(1, 1 << marker_height)
        if bits != (1 << marker_height) - 1 or marker_height == 2
    ]
    random.shuffle(patterns)

    chosen_cols: list[int] = []
    varied_cols: list[int] = []
    for idx, c in enumerate(active_cols):
        pattern = patterns[idx % len(patterns)]
        for r in range(marker_height):
            if (pattern >> r) & 1:
                motif[r, c] = ORANGE
        chosen_cols.append(c)
        col_values = motif[:, c]
        if not np.all(col_values == col_values[0]):
            varied_cols.append(c)

    for r in range(marker_height):
        if not np.any(motif[r] == ORANGE):
            c = random.choice(active_cols)
            motif[r, c] = ORANGE

    if not varied_cols:
        c = active_cols[0]
        motif[:, c] = BLACK
        motif[0, c] = ORANGE
        motif[-1, c] = BLACK
        varied_cols.append(c)

    assert np.any(np.all(motif[:, 1:] == BLACK, axis=0)), "need at least one all-black motif column"
    return motif


def _columns_with_phase_signal(motif: np.ndarray) -> list[int]:
    cols = []
    for c in range(1, motif.shape[1]):
        col = motif[:, c]
        if not np.all(col == col[0]):
            cols.append(c)
    return cols


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    divider_row = random.randint(*GEN_DIVIDER_RANGE)
    if divider_row >= height - divider_row - 1:
        return None

    marker_height = random.choice(GEN_MARKER_HEIGHTS)
    marker_start = random.randint(0, max(0, divider_row - marker_height - 2))
    marker_rows = list(range(marker_start, marker_start + marker_height))

    motif = _build_motif(width, marker_height)
    phase_cols = [c for c in _columns_with_phase_signal(motif) if c != 0]
    if len(phase_cols) < 2:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[divider_row, :] = GREY
    for offset, r in enumerate(marker_rows):
        grid[r] = motif[offset]

    available_rows = [r for r in range(divider_row - 1) if r not in marker_rows]
    if len(available_rows) < 2:
        return None

    usable_red_rows = [r for r in available_rows if divider_row - r - 1 >= 1]
    if len(usable_red_rows) < 2:
        return None

    unique_lengths = {}
    for r in usable_red_rows:
        unique_lengths.setdefault(divider_row - r - 1, r)
    if len(unique_lengths) < 2:
        return None

    red_count = random.randint(2, min(4, len(unique_lengths), width - 1))
    chosen_lengths = random.sample(sorted(unique_lengths), red_count)

    red_columns = random.sample(phase_cols, min(2, red_count))
    remaining_pool = [c for c in range(1, width) if c not in red_columns]
    random.shuffle(remaining_pool)
    while len(red_columns) < red_count:
        red_columns.append(remaining_pool.pop())

    random.shuffle(red_columns)
    for c, ray_len in zip(red_columns, chosen_lengths):
        source_row = unique_lengths[ray_len]
        grid[source_row, c] = RED

    if not np.any((grid[:divider_row] == RED)):
        return None

    return grid


def _decorative_ray_variant(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    divider_row = _find_divider_row(g)
    _, motif = _extract_motif(g, divider_row)
    out = g.copy()

    for c, upper_len in _lower_ray_lengths_from_input(g, divider_row).items():
        source_row = int(np.flatnonzero(g[:divider_row, c] == RED)[0])
        if upper_len > 0:
            out[source_row + 1 : divider_row, c] = RED
            out[divider_row + 1 : divider_row + 1 + upper_len, c] = RED

    motif_h = motif.shape[0]
    start_row = divider_row + 1
    for c in range(g.shape[1]):
        for r in range(start_row, g.shape[0]):
            if out[r, c] == RED:
                continue
            value = int(motif[(r - start_row) % motif_h, c])
            if value != BLACK:
                out[r, c] = value
    return out


def _validate_semantics(grid: np.ndarray) -> bool:
    divider_row = _find_divider_row(grid)
    marker_rows, motif = _extract_motif(grid, divider_row)

    assert np.all(grid[divider_row] == GREY)
    assert np.count_nonzero(grid[divider_row + 1 :] != BLACK) == 0, "lower half must start empty"
    assert np.all(grid[marker_rows, 0] == GREY)
    assert np.count_nonzero(grid[:divider_row, 0] == GREY) == len(marker_rows), "only the motif markers may use grey above the divider"
    assert not np.any(grid[marker_rows] == RED), "red sources must avoid motif rows"

    red_positions = np.argwhere(grid[:divider_row] == RED)
    assert 2 <= len(red_positions) <= 4, "need multiple red source columns"
    red_cols = [int(c) for _r, c in red_positions]
    assert len(red_cols) == len(set(red_cols))

    ray_lengths = sorted(divider_row - int(r) - 1 for r, _c in red_positions)
    assert len(set(ray_lengths)) >= 2, "need multiple phase offsets"
    assert all(length >= 1 for length in ray_lengths), "reserve phase 0 for columns without red sources"
    assert len(set(range(grid.shape[1])) - set(red_cols) - {0}) >= 2, "need columns with zero phase"

    signal_cols = _columns_with_phase_signal(motif)
    assert any(c in red_cols for c in signal_cols), "at least one red column must carry a visible phase change"
    assert np.any(np.all(motif[:, 1:] == BLACK, axis=0)), "need all-black motif columns"
    assert len(signal_cols) >= 2, "need multiple motif columns whose phase matters"

    solved = main(grid)
    decorative = _decorative_ray_variant(grid)
    assert not np.array_equal(solved, decorative), "ray length must affect the lower cyclic fill"
    assert np.count_nonzero(solved[divider_row + 1 :] == RED) > 0, "continued rays must appear below the divider"

    lower = solved[divider_row + 1 :]
    assert np.any(lower == ORANGE), "lower region must contain cyclic motif paint"
    assert np.any(lower == BLACK), "some lower cells should stay black"
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate()
        if grid is None:
            continue
        try:
            if _validate_semantics(grid):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate valid input for puzzle 3")
