from __future__ import annotations

import math
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

CONTROL_COLORS = (BLUE, RED, YELLOW)
PAYLOAD_COLORS = (GREEN, GREY)

ODD_HEIGHTS = (11, 13, 15, 17)
EVEN_HEIGHTS = (12, 14, 16)
WIDTH_RANGE = (11, 17)  # upper bound exclusive
BAND_HEIGHT_RANGE = (5, 8)
ROW_WIDTH_RANGE = (4, 8)
GREEN_FRACTION_RANGE = (0.30, 0.50)
TIE_SAMPLE_PROB = 0.55
THREE_CONTROL_PROB = 0.7
MAX_GENERATION_ATTEMPTS = 500

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    out = g.copy()

    gray_mask = g == GREY
    if np.any(gray_mask):
        gray_rows = np.nonzero(gray_mask)[0]
        out[gray_mask] = g[gray_rows, 0]

    green_mask = g == GREEN
    if np.any(green_mask):
        green_rows = np.nonzero(green_mask)[0]
        top_dist = green_rows
        bottom_dist = (h - 1) - green_rows
        out[green_mask] = np.where(top_dist <= bottom_dist, ORANGE, MAROON)

    out[0, :] = BLACK
    out[-1, :] = BLACK
    out[1:-1, 0] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor row-keyed grey placeholders and edge-keyed green markers, then erase the keys and edge lines.

    Input structure:
    - A black-background grid has a full orange top edge line and a full maroon bottom edge line.
    - Interior payload cells use only grey and green, and every payload row has one control pixel in column 0.
    - Control pixels are blue, red, or yellow and label only the rows that contain payload cells.

    Rule:
    - Recolor every grey cell to the color of its own row's column-0 control pixel.
    - Recolor every green cell by the nearer horizontal edge line: orange if the top edge is closer or tied, maroon if the bottom edge is closer.
    - Erase all column-0 control pixels, the entire top orange edge line, and the entire bottom maroon edge line to black.
    - Leave all remaining black background cells unchanged.

    Key insight:
    - Grey and green cells can be interwoven inside the same connected shape, so the rule choice is determined by each pixel's original color rather than by position, component membership, or row alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 3 <= h <= 30 and 3 <= w <= 30, "grid must fit ARC bounds"
    assert np.all(g[0, :] == ORANGE), "top row must be the orange edge line"
    assert np.all(g[-1, :] == MAROON), "bottom row must be the maroon edge line"

    allowed = {BLACK, BLUE, RED, GREEN, YELLOW, GREY, ORANGE, MAROON}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected colors present"

    active_rows: list[int] = []
    for r in range(1, h - 1):
        row_payload = g[r, 1:]
        assert np.all(np.isin(row_payload, (BLACK, GREEN, GREY))), "interior rows may only contain black, green, or grey"
        has_payload = bool(np.any(np.isin(row_payload, PAYLOAD_COLORS)))
        ctrl = int(g[r, 0])
        if has_payload:
            assert ctrl in CONTROL_COLORS, "each payload row needs a valid control color"
            active_rows.append(r)
        else:
            assert ctrl == BLACK, "rows without payload must not have control pixels"
    assert active_rows, "expected at least one payload row"

    out = g.copy()
    for r in active_rows:
        ctrl = int(g[r, 0])
        gray_cols = np.where(g[r, :] == GREY)[0]
        if gray_cols.size:
            out[r, gray_cols] = ctrl

    for r, c in np.argwhere(g == GREEN):
        top_dist = int(r)
        bottom_dist = int(h - 1 - r)
        out[r, c] = ORANGE if top_dist <= bottom_dist else MAROON

    out[0, :] = BLACK
    out[-1, :] = BLACK
    out[1:-1, 0] = BLACK
    return out


def _sample_height(require_tie: bool) -> int:
    if require_tie:
        return int(_RNG.choice(ODD_HEIGHTS))
    choices = ODD_HEIGHTS + EVEN_HEIGHTS
    return int(_RNG.choice(choices))


def _sample_active_rows(h: int, require_tie: bool) -> list[int] | None:
    max_band = min(BAND_HEIGHT_RANGE[1], h - 2)
    band_sizes = np.arange(BAND_HEIGHT_RANGE[0], max_band + 1, dtype=int)
    if band_sizes.size == 0:
        return None

    mid = (h - 1) / 2.0
    tie_row = (h - 1) // 2
    for band_h in _RNG.permutation(band_sizes):
        valid_tops = []
        last_top = h - 1 - int(band_h)
        for top in range(1, last_top + 1):
            rows = list(range(top, top + int(band_h)))
            if not any(r < mid for r in rows):
                continue
            if not any(r > mid for r in rows):
                continue
            if require_tie and tie_row not in rows:
                continue
            valid_tops.append(top)
        if valid_tops:
            top = int(_RNG.choice(valid_tops))
            return list(range(top, top + int(band_h)))
    return None


def _sample_intervals(w: int, rows: list[int]) -> dict[int, tuple[int, int]] | None:
    left_min = 2
    left_max = w - 2
    max_width = min(ROW_WIDTH_RANGE[1], left_max - left_min + 1)
    if max_width < ROW_WIDTH_RANGE[0]:
        return None

    intervals: dict[int, tuple[int, int]] = {}
    prev_left = None
    prev_width = None
    for r in rows:
        width = int(_RNG.integers(ROW_WIDTH_RANGE[0], max_width + 1))
        max_left_start = left_max - width + 1
        if prev_left is None:
            left = int(_RNG.integers(left_min, max_left_start + 1))
        else:
            overlap = 2 if min(int(prev_width), width) >= 4 else 1
            left_low = max(left_min, int(prev_left) - width + overlap)
            left_high = min(max_left_start, int(prev_left) + int(prev_width) - overlap)
            if left_low > left_high:
                return None
            left = int(_RNG.integers(left_low, left_high + 1))
        intervals[r] = (left, width)
        prev_left, prev_width = left, width
    return intervals


def _sample_control_sequence(n_rows: int) -> list[int] | None:
    for _ in range(100):
        use_three = n_rows >= 5 and _RNG.random() < THREE_CONTROL_PROB
        color_count = 3 if use_three else 2
        palette = [int(v) for v in _RNG.choice(CONTROL_COLORS, size=color_count, replace=False)]
        seq = [palette[int(_RNG.integers(0, color_count))] for _ in range(n_rows)]
        if len(set(seq)) != color_count:
            continue
        if all(seq[i] == seq[i + 1] for i in range(n_rows - 1)):
            continue
        if not any(seq[i] != seq[i + 1] for i in range(n_rows - 1)):
            continue
        if not any(seq[i] == seq[j] for i in range(n_rows) for j in range(i + 2, n_rows)):
            continue
        return seq
    return None


def _build_shape_cells(intervals: dict[int, tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, list[tuple[int, int]]]]:
    cells: list[tuple[int, int]] = []
    row_cells: dict[int, list[tuple[int, int]]] = {}
    for r in sorted(intervals):
        left, width = intervals[r]
        row = [(r, c) for c in range(left, left + width)]
        row_cells[r] = row
        cells.extend(row)
    return cells, row_cells


def _sample_green_cells(
    row_cells: dict[int, list[tuple[int, int]]],
    h: int,
    require_tie: bool,
) -> set[tuple[int, int]] | None:
    cells = [cell for r in sorted(row_cells) for cell in row_cells[r]]
    total = len(cells)
    min_green = math.ceil(GREEN_FRACTION_RANGE[0] * total)
    max_green = math.floor(GREEN_FRACTION_RANGE[1] * total)
    if min_green > max_green:
        return None

    p_green = float(_RNG.uniform(0.33, 0.47))
    green = {cell for cell in cells if _RNG.random() < p_green}
    row_green_counts = {r: sum(cell in green for cell in row) for r, row in row_cells.items()}
    required_green: set[tuple[int, int]] = set()

    mid = (h - 1) / 2.0
    rows_above = [r for r in row_cells if r < mid]
    rows_below = [r for r in row_cells if r > mid]
    if not rows_above or not rows_below:
        return None

    required_rows = [int(_RNG.choice(rows_above)), int(_RNG.choice(rows_below))]
    if require_tie:
        tie_row = (h - 1) // 2
        if tie_row not in row_cells:
            return None
        required_rows.append(tie_row)

    for r, row in row_cells.items():
        if row_green_counts[r] == 0:
            chosen = row[int(_RNG.integers(0, len(row)))]
            green.add(chosen)
            row_green_counts[r] += 1
        elif row_green_counts[r] == len(row):
            keep_gray = row[int(_RNG.integers(0, len(row)))]
            green.remove(keep_gray)
            row_green_counts[r] -= 1

    for r in required_rows:
        row = row_cells[r]
        if row_green_counts[r] == 0:
            chosen = row[int(_RNG.integers(0, len(row)))]
            green.add(chosen)
            row_green_counts[r] += 1
        required_green.add(row[int(_RNG.integers(0, len(row)))])

    for cell in list(required_green):
        if cell not in green:
            green.add(cell)
            row_green_counts[cell[0]] += 1

    while len(green) < min_green:
        candidates = [
            cell
            for cell in cells
            if cell not in green and row_green_counts[cell[0]] < len(row_cells[cell[0]]) - 1
        ]
        if not candidates:
            return None
        chosen = candidates[int(_RNG.integers(0, len(candidates)))]
        green.add(chosen)
        row_green_counts[chosen[0]] += 1

    while len(green) > max_green:
        candidates = [
            cell
            for cell in green
            if cell not in required_green and row_green_counts[cell[0]] > 1
        ]
        if not candidates:
            return None
        chosen = candidates[int(_RNG.integers(0, len(candidates)))]
        green.remove(chosen)
        row_green_counts[chosen[0]] -= 1

    if not (min_green <= len(green) <= max_green):
        return None
    if any(row_green_counts[r] == 0 or row_green_counts[r] == len(row_cells[r]) for r in row_cells):
        return None
    return green


def _render_candidate(
    h: int,
    w: int,
    rows: list[int],
    row_cells: dict[int, list[tuple[int, int]]],
    controls: list[int],
    green_cells: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = ORANGE
    grid[-1, :] = MAROON

    for r, ctrl in zip(rows, controls):
        grid[r, 0] = ctrl
        for cell in row_cells[r]:
            grid[cell] = GREEN if cell in green_cells else GREY
    return grid


def _mixed_edge_counts(grid: np.ndarray) -> tuple[int, int]:
    horizontal = 0
    vertical = 0
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w):
            if grid[r, c] not in PAYLOAD_COLORS:
                continue
            if c + 1 < w and grid[r, c + 1] in PAYLOAD_COLORS and grid[r, c] != grid[r, c + 1]:
                horizontal += 1
            if r + 1 < h - 1 and grid[r + 1, c] in PAYLOAD_COLORS and grid[r, c] != grid[r + 1, c]:
                vertical += 1
    return horizontal, vertical


def _semantic_checks(grid: np.ndarray, require_tie: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    h, w = g.shape
    if h < 3 or w < 5:
        return False
    if not np.all(g[0, :] == ORANGE) or not np.all(g[-1, :] == MAROON):
        return False
    if not np.all(np.isin(g[1:-1, 1:], (BLACK, GREEN, GREY))):
        return False

    active_rows = [
        r
        for r in range(1, h - 1)
        if np.any(np.isin(g[r, 1:], PAYLOAD_COLORS))
    ]
    if len(active_rows) < BAND_HEIGHT_RANGE[0]:
        return False
    if not all(int(g[r, 0]) in CONTROL_COLORS for r in active_rows):
        return False
    if any(int(g[r, 0]) != BLACK for r in range(1, h - 1) if r not in active_rows):
        return False
    if len(set(int(g[r, 0]) for r in active_rows)) < 2:
        return False
    if not any(int(g[active_rows[i], 0]) == int(g[active_rows[j], 0]) for i in range(len(active_rows)) for j in range(i + 2, len(active_rows))):
        return False

    payload_mask = np.isin(g, PAYLOAD_COLORS)
    comps = _components(payload_mask)
    if len(comps) != 1:
        return False

    green_count = int(np.count_nonzero(g == GREEN))
    grey_count = int(np.count_nonzero(g == GREY))
    if green_count == 0 or grey_count == 0:
        return False
    green_fraction = green_count / (green_count + grey_count)
    if not (GREEN_FRACTION_RANGE[0] <= green_fraction <= GREEN_FRACTION_RANGE[1]):
        return False

    mixed_rows = 0
    for r in active_rows:
        row = g[r, 1:]
        has_green = bool(np.any(row == GREEN))
        has_grey = bool(np.any(row == GREY))
        if not (has_green and has_grey):
            return False
        mixed_rows += 1
    if mixed_rows < 3:
        return False

    mixed_cols = 0
    for c in range(1, w):
        col = g[1:-1, c]
        if np.any(col == GREEN) and np.any(col == GREY):
            mixed_cols += 1
    if mixed_cols < 2:
        return False

    horizontal_mixed, vertical_mixed = _mixed_edge_counts(g)
    if horizontal_mixed < 2 or vertical_mixed < 2:
        return False

    mid = (h - 1) / 2.0
    green_rows = np.nonzero(np.any(g == GREEN, axis=1))[0].tolist()
    if not any(r < mid for r in green_rows):
        return False
    if not any(r > mid for r in green_rows):
        return False
    if require_tie:
        tie_row = (h - 1) // 2
        if h % 2 == 0 or not np.any(g[tie_row, :] == GREEN):
            return False

    expected = _oracle_output(g)
    if not np.array_equal(expected[0, :], np.zeros(w, dtype=int)):
        return False
    if not np.array_equal(expected[-1, :], np.zeros(w, dtype=int)):
        return False
    if not any(np.any(expected[r, 1:] == int(g[r, 0])) for r in active_rows):
        return False
    if not np.any(expected == ORANGE) or not np.any(expected == MAROON):
        return False
    if not any(
        np.any(expected[r, 1:] == int(g[r, 0])) and np.any(expected[r, 1:] == ORANGE)
        for r in active_rows
    ):
        return False
    if not any(
        np.any(expected[r, 1:] == int(g[r, 0])) and np.any(expected[r, 1:] == MAROON)
        for r in active_rows
    ):
        return False

    try:
        actual = main(g)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> tuple[np.ndarray, bool] | None:
    require_tie = bool(_RNG.random() < TIE_SAMPLE_PROB)
    h = _sample_height(require_tie)
    w = int(_RNG.integers(*WIDTH_RANGE))

    rows = _sample_active_rows(h, require_tie)
    if rows is None:
        return None

    intervals = _sample_intervals(w, rows)
    if intervals is None:
        return None

    _cells, row_cells = _build_shape_cells(intervals)
    controls = _sample_control_sequence(len(rows))
    if controls is None:
        return None

    green_cells = _sample_green_cells(row_cells, h, require_tie)
    if green_cells is None:
        return None

    grid = _render_candidate(h, w, rows, row_cells, controls, green_cells)
    return grid, require_tie


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        grid, require_tie = attempt
        if _semantic_checks(grid, require_tie):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
