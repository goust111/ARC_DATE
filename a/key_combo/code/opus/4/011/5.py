from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
BAR_COLORS = {BLUE, RED}

MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _candidate_starts(ref_row: np.ndarray, pair: tuple[int, int]) -> list[int]:
    return [
        c
        for c in range(ref_row.shape[0] - 1)
        if (int(ref_row[c]), int(ref_row[c + 1])) == pair
    ]


def _bar_rows(grid: np.ndarray) -> list[tuple[int, int, tuple[int, int]]]:
    bars: list[tuple[int, int, tuple[int, int]]] = []
    for r in range(1, grid.shape[0]):
        cols = np.flatnonzero(np.isin(grid[r], [BLUE, RED]))
        if cols.size == 0:
            continue
        assert cols.size == 2, "each bar row must contain exactly one two-pixel bar"
        assert int(cols[1]) == int(cols[0]) + 1, "bar cells must be adjacent"
        pair = (int(grid[r, cols[0]]), int(grid[r, cols[1]]))
        assert set(pair) == BAR_COLORS, "each bar must use blue and red"
        bars.append((r, int(cols[0]), pair))
    return bars


def _choose_target_start(
    ref_row: np.ndarray,
    pair: tuple[int, int],
    start: int,
) -> int:
    candidates = _candidate_starts(ref_row, pair)
    assert candidates, "reference row must provide a matching color pair"
    if start in candidates:
        return start

    blue_offset = 0 if pair[0] == BLUE else 1
    blue_col = start + blue_offset
    ranked = sorted(
        candidates,
        key=lambda c: (abs((c + blue_offset) - blue_col), abs(c - start), c),
    )
    return int(ranked[0])


def _align_bars(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    ref_row = out[0].copy()

    for r, start, pair in _bar_rows(g):
        target = _choose_target_start(ref_row, pair, start)
        if target == start:
            continue
        out[r, start : start + 2] = BG
        out[r, target : target + 2] = pair

    return out


def _zero_components(grid: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[np.ndarray, bool]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BG:
                continue
            stack = [(r, c)]
            cells = []
            touches_edge = False
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((np.array(cells, dtype=int), touches_edge))
    return comps


def _fill_closed_regions(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for coords, touches_edge in _zero_components(out):
        if touches_edge:
            continue
        out[coords[:, 0], coords[:, 1]] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align scattered two-color bars to the reference row, then fill newly enclosed background with the marker color.

    Input structure:
    - The top row is a full-width blue/red reference strip that defines valid adjacent pair slots.
    - Lower rows contain isolated two-pixel horizontal blue/red bars plus 1-3 green marker pixels on black background.
    - Markers stay separate from the bar rows and do not overlap any moving bar.

    Rule:
    - For each lower two-pixel bar, compare its color order to the reference row and find the nearest matching adjacent reference slot for that same ordered pair.
    - If the bar is already on such a slot, leave it in place; otherwise shift the whole two-pixel bar horizontally to that slot and clear the old cells to black.
    - After all bars are aligned, find every 4-connected black region and recolor each region that does not touch the grid edge to green.
    - Preserve the original green markers.

    Key insight:
    - The horizontal alignment step is what changes topology: bars that merely slide sideways can plug a background passage and create a new enclosed region that must then be filled.
    """
    aligned = _align_bars(np.asarray(grid, dtype=int))
    return _fill_closed_regions(aligned)


def _reference_row(width: int, left_red: int) -> np.ndarray:
    blue_len = width // 2
    right_red = width - left_red - blue_len
    assert blue_len >= 8 and left_red >= 4 and right_red >= 4
    row = np.full(width, RED, dtype=int)
    row[left_red : left_red + blue_len] = BLUE
    return row


def _pair_targets(width: int, left_red: int) -> dict[tuple[int, int], int]:
    blue_len = width // 2
    return {
        (RED, BLUE): left_red - 1,
        (BLUE, RED): left_red + blue_len - 1,
    }


def _misaligned_start_ranges(width: int, left_red: int) -> dict[tuple[int, int], list[int]]:
    blue_len = width // 2
    right_red_start = left_red + blue_len
    return {
        (RED, BLUE): list(range(0, left_red - 3)),
        (BLUE, RED): list(range(right_red_start + 2, width - 1)),
    }


def _paint_bars(
    grid: np.ndarray,
    rows: list[tuple[int, int, tuple[int, int]]],
) -> None:
    for r, start, pair in rows:
        grid[r, start : start + 2] = pair


def _touches_bar(inp: np.ndarray, r: int, c: int) -> bool:
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and int(inp[nr, nc]) in BAR_COLORS:
            return True
    return False


def _closed_component_sets(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    result = []
    for coords, touches_edge in _zero_components(grid):
        if not touches_edge:
            result.append({(int(r), int(c)) for r, c in coords})
    return result


def _valid_extra_marker_cells(inp: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = []
    for r in range(2, inp.shape[0]):
        for c in range(inp.shape[1]):
            if int(inp[r, c]) != BG or (r, c) in forbidden or _touches_bar(inp, r, c):
                continue
            cells.append((r, c))
    return cells


def _build_attempt() -> np.ndarray | None:
    width = random.choice([16, 18, 20, 22, 24])
    blue_len = width // 2
    left_red = random.randint(4, blue_len - 4)
    ref = _reference_row(width, left_red)
    targets = _pair_targets(width, left_red)
    source_ranges = _misaligned_start_ranges(width, left_red)

    height = random.randint(10, 15)
    bar_count = random.randint(4, 7)
    if bar_count > height - 2:
        return None

    extra_rows = sorted(random.sample(range(4, height), bar_count - 2))
    cavity_pair = random.choice([(RED, BLUE), (BLUE, RED)])
    cavity_target = targets[cavity_pair]
    cavity_cells = {(2, cavity_target), (2, cavity_target + 1)}
    side_markers = [(2, cavity_target - 1), (2, cavity_target + 2)]

    bar_specs: list[dict[str, object]] = [
        {
            "row": 1,
            "pair": cavity_pair,
            "target": cavity_target,
            "input": cavity_target,
        },
        {
            "row": 3,
            "pair": cavity_pair,
            "target": cavity_target,
            "input": random.choice(source_ranges[cavity_pair]),
        },
    ]

    opposite_pair = (BLUE, RED) if cavity_pair == (RED, BLUE) else (RED, BLUE)
    aligned_other_added = False
    misaligned_other_added = False

    for idx, row in enumerate(extra_rows):
        if idx == 0:
            pair = opposite_pair
            aligned = True
        else:
            pair = random.choice([(RED, BLUE), (BLUE, RED)])
            aligned = random.random() < 0.45
        if aligned:
            input_start = targets[pair]
            aligned_other_added = True
        else:
            input_start = random.choice(source_ranges[pair])
            misaligned_other_added = True
        bar_specs.append(
            {
                "row": row,
                "pair": pair,
                "target": targets[pair],
                "input": input_start,
            }
        )

    if not aligned_other_added or not misaligned_other_added:
        return None

    inp = np.zeros((height, width), dtype=int)
    inp[0] = ref
    aligned_grid = inp.copy()
    aligned_grid[0] = ref

    input_bars = [(int(spec["row"]), int(spec["input"]), spec["pair"]) for spec in bar_specs]
    aligned_bars = [(int(spec["row"]), int(spec["target"]), spec["pair"]) for spec in bar_specs]
    _paint_bars(inp, input_bars)
    _paint_bars(aligned_grid, aligned_bars)

    for r, c in side_markers:
        if _touches_bar(inp, r, c):
            return None
        inp[r, c] = GREEN
        aligned_grid[r, c] = GREEN

    marker_total = random.choice([2, 2, 3])
    forbidden = set(side_markers)
    if marker_total == 3:
        extra_choices = _valid_extra_marker_cells(inp, forbidden)
        if not extra_choices:
            return None
        r, c = random.choice(extra_choices)
        inp[r, c] = GREEN
        aligned_grid[r, c] = GREEN

    if any(_touches_bar(inp, r, c) for r, c in np.argwhere(inp == GREEN)):
        return None

    before_closed = _closed_component_sets(inp)
    after_closed = _closed_component_sets(aligned_grid)
    cavity_set = set(cavity_cells)
    if before_closed:
        if any(cavity_set <= comp for comp in before_closed):
            return None
        if len(before_closed) != 0:
            return None
    if not any(cavity_set <= comp for comp in after_closed):
        return None

    expected = _fill_closed_regions(aligned_grid)
    if not np.all(expected[2, cavity_target : cavity_target + 2] == GREEN):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, actual):
        return None

    bars = _bar_rows(inp)
    if len(bars) != bar_count:
        return None
    if {pair for _, _, pair in bars} != {(RED, BLUE), (BLUE, RED)}:
        return None
    if sum(start != targets[pair] for _, start, pair in bars) < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid new ARC input")
