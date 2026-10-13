from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg, in_bounds


LINE_COLOR = Color.RED
MARKER_COLOR = Color.PINK
PATTERN_COLORS = (Color.BLUE, Color.GREEN)
UNKNOWN_COLORS = {LINE_COLOR, MARKER_COLOR}

GEN_ATTEMPTS = 500
GRID_MIN = 14
GRID_MAX = 18
PERIOD_MIN = 2
PERIOD_MAX = 5


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    runs = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    return runs


def _consistent_tile(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    for rr in range(row_period):
        for cc in range(col_period):
            values = {
                int(grid[r, c])
                for r in range(rr, grid.shape[0], row_period)
                for c in range(cc, grid.shape[1], col_period)
                if int(grid[r, c]) not in UNKNOWN_COLORS
            }
            if not values or len(values) > 1:
                return None
            tile[rr, cc] = int(next(iter(values)))
    return tile


def _infer_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    visible_colors = sorted(
        int(v) for v in np.unique(g) if int(v) not in UNKNOWN_COLORS
    )
    assert len(visible_colors) == 2, "expected exactly two visible pattern colors"

    candidates: list[tuple[int, int, np.ndarray]] = []
    for row_period in range(1, g.shape[0] + 1):
        for col_period in range(1, g.shape[1] + 1):
            tile = _consistent_tile(g, row_period, col_period)
            if tile is not None:
                candidates.append((row_period, col_period, tile))

    assert candidates, "visible cells must define a periodic tile"
    min_area = min(row_period * col_period for row_period, col_period, _tile in candidates)
    minimal = [
        (row_period, col_period, tile)
        for row_period, col_period, tile in candidates
        if row_period * col_period == min_area
    ]
    assert len(minimal) == 1, "minimum-area periodic tile must be unique"
    row_period, col_period, tile = minimal[0]
    assert set(map(int, tile.ravel())) == set(visible_colors)
    return int(row_period), int(col_period), tile


def _restore_markers(grid: np.ndarray, row_period: int, col_period: int, tile: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(out == MARKER_COLOR):
        out[int(r), int(c)] = int(tile[int(r) % row_period, int(c) % col_period])
    return out


def _fill_short_gaps(restored: np.ndarray, red_mask: np.ndarray, gap_limit: int) -> np.ndarray:
    out = np.asarray(restored, dtype=int).copy()
    fill_mask = np.zeros_like(red_mask, dtype=bool)
    h, w = red_mask.shape

    for r in range(h):
        runs = _runs(np.where(red_mask[r])[0])
        for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
            left_len = a1 - a0 + 1
            right_len = b1 - b0 + 1
            gap_len = b0 - a1 - 1
            if left_len >= 2 and right_len >= 2 and gap_len <= gap_limit:
                fill_mask[r, a1 + 1 : b0] = True

    for c in range(w):
        runs = _runs(np.where(red_mask[:, c])[0])
        for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
            top_len = a1 - a0 + 1
            bottom_len = b1 - b0 + 1
            gap_len = b0 - a1 - 1
            if top_len >= 2 and bottom_len >= 2 and gap_len <= gap_limit:
                fill_mask[a1 + 1 : b0, c] = True

    out[fill_mask] = LINE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the background tile period, restore pink occlusions, and repair only red line gaps up to the period-derived threshold.

    Input structure:
    - The grid contains a full blue/green periodic background, but some non-red cells are replaced by pink markers.
    - Red horizontal and vertical line segments are drawn over the periodic background; some segments are split by non-red gaps.
    - Red and pink cells must be treated as unknown when recovering the background tile.

    Rule:
    - Infer the unique minimum-area rectangular period from the visible blue/green cells.
    - Let G be the smaller of the row and column periods.
    - Replace every pink cell with the inferred tile color at the same coordinate residue.
    - For each horizontal or vertical red line, fill any internal non-red gap whose length is at most G, and leave longer gaps unchanged.

    Key insight:
    - The same periodic tile explains both tasks: it reconstructs the masked background and its smaller period dimension becomes the red-gap repair threshold.
    """
    g = np.asarray(grid, dtype=int)
    row_period, col_period, tile = _infer_tile(g)
    restored = _restore_markers(g, row_period, col_period, tile)
    return _fill_short_gaps(restored, g == LINE_COLOR, min(row_period, col_period))


def _tile_periods(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    row_period = h
    for cand in range(1, h + 1):
        if all(np.array_equal(tile[r], tile[r % cand]) for r in range(h)):
            row_period = cand
            break

    col_period = w
    for cand in range(1, w + 1):
        ok = True
        for c in range(w):
            if not np.array_equal(tile[:, c], tile[:, c % cand]):
                ok = False
                break
        if ok:
            col_period = cand
            break
    return int(row_period), int(col_period)


def _make_tile(rng: random.Random, row_period: int, col_period: int) -> np.ndarray:
    while True:
        tile = rng.choice(PATTERN_COLORS) * np.ones((row_period, col_period), dtype=int)
        mask = np.array(
            [rng.random() < 0.5 for _ in range(row_period * col_period)], dtype=bool
        ).reshape(row_period, col_period)
        tile[mask] = PATTERN_COLORS[1]
        tile[~mask] = PATTERN_COLORS[0]
        counts = {color: int(np.count_nonzero(tile == color)) for color in PATTERN_COLORS}
        if min(counts.values()) < max(1, tile.size // 4):
            continue
        if _tile_periods(tile) != (row_period, col_period):
            continue
        return tile


def _tile_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % tile.shape[0]
    cc = np.arange(width)[None, :] % tile.shape[1]
    return tile[rr, cc].astype(int)


def _line_cells(
    orientation: str,
    fixed: int,
    start: int,
    segment_lengths: list[int],
    gap_lengths: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[tuple[int, int]]]]:
    corridor = []
    red_cells = []
    gap_groups: list[list[tuple[int, int]]] = []
    pos = int(start)
    for idx, seg_len in enumerate(segment_lengths):
        for offset in range(seg_len):
            if orientation == "H":
                cell = (fixed, pos + offset)
            else:
                cell = (pos + offset, fixed)
            corridor.append(cell)
            red_cells.append(cell)
        pos += seg_len
        if idx < len(gap_lengths):
            gap_cells = []
            for offset in range(gap_lengths[idx]):
                if orientation == "H":
                    cell = (fixed, pos + offset)
                else:
                    cell = (pos + offset, fixed)
                corridor.append(cell)
                gap_cells.append(cell)
            gap_groups.append(gap_cells)
            pos += gap_lengths[idx]
    return corridor, red_cells, gap_groups


def _buffer_clear(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if in_bounds((h, w), rr, cc) and blocked[rr, cc]:
                    return False
    return True


def _mark_buffer(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if in_bounds((h, w), rr, cc):
                    blocked[rr, cc] = True


def _sample_segments(
    rng: random.Random,
    span_limit: int,
    gap_lengths: list[int] | None,
    gap_cap: int,
) -> tuple[list[int], list[int]] | None:
    for _ in range(200):
        if gap_lengths is None:
            gap_count = rng.choices([0, 1, 2], weights=[2, 4, 3], k=1)[0]
            if gap_count == 0:
                segment_lengths = [rng.randint(3, min(7, span_limit))]
                chosen_gaps: list[int] = []
            else:
                chosen_gaps = [rng.randint(1, gap_cap) for _ in range(gap_count)]
                segment_lengths = [rng.randint(2, 4) for _ in range(gap_count + 1)]
        else:
            chosen_gaps = list(gap_lengths)
            segment_lengths = [rng.randint(2, 4) for _ in range(len(chosen_gaps) + 1)]

        total = sum(segment_lengths) + sum(chosen_gaps)
        if total <= span_limit:
            return segment_lengths, chosen_gaps
    return None


def _place_line(
    rng: random.Random,
    blocked: np.ndarray,
    orientation: str,
    segment_lengths: list[int],
    gap_lengths: list[int],
) -> dict | None:
    h, w = blocked.shape
    span = sum(segment_lengths) + sum(gap_lengths)
    choices = []
    if orientation == "H":
        for row in range(h):
            for start in range(w - span + 1):
                choices.append((row, start))
    else:
        for col in range(w):
            for start in range(h - span + 1):
                choices.append((col, start))
    rng.shuffle(choices)

    for fixed, start in choices:
        corridor, red_cells, gap_groups = _line_cells(
            orientation, fixed, start, segment_lengths, gap_lengths
        )
        if _buffer_clear(blocked, corridor):
            _mark_buffer(blocked, corridor)
            return {
                "orientation": orientation,
                "segment_lengths": list(segment_lengths),
                "gap_lengths": list(gap_lengths),
                "red_cells": list(red_cells),
                "gap_groups": [list(group) for group in gap_groups],
            }
    return None


def _count_visible_by_residue(
    height: int,
    width: int,
    row_period: int,
    col_period: int,
    red_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {
        (rr, cc): [] for rr in range(row_period) for cc in range(col_period)
    }
    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if cell in red_cells or cell in marker_cells:
                continue
            buckets[(r % row_period, c % col_period)].append(cell)
    return buckets


def _expected_output_from_meta(
    tile_grid: np.ndarray,
    lines: list[dict],
    threshold: int,
) -> np.ndarray:
    out = np.array(tile_grid, dtype=int, copy=True)
    for line in lines:
        for r, c in line["red_cells"]:
            out[r, c] = LINE_COLOR
        for gap_len, gap_cells in zip(line["gap_lengths"], line["gap_groups"]):
            if gap_len <= threshold:
                for r, c in gap_cells:
                    out[r, c] = LINE_COLOR
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    tile: np.ndarray,
    row_period: int,
    col_period: int,
    lines: list[dict],
    long_gap_group: list[tuple[int, int]],
) -> bool:
    inferred_row, inferred_col, inferred_tile = _infer_tile(inp)
    if (inferred_row, inferred_col) != (row_period, col_period):
        return False
    if not np.array_equal(inferred_tile, tile):
        return False

    if not np.array_equal(main(inp), expected):
        return False

    threshold = min(row_period, col_period)
    alt_low = _expected_output_from_meta(_restore_markers(inp, row_period, col_period, tile), lines, threshold - 1)
    alt_high = _expected_output_from_meta(_restore_markers(inp, row_period, col_period, tile), lines, threshold + 1)
    if np.array_equal(expected, alt_low) or np.array_equal(expected, alt_high):
        return False

    if not any(int(inp[r, c]) == MARKER_COLOR and int(expected[r, c]) != MARKER_COLOR for r, c in np.argwhere(inp == MARKER_COLOR)):
        return False

    if not all(int(expected[r, c]) in PATTERN_COLORS for r, c in long_gap_group):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_period = rng.randint(PERIOD_MIN, PERIOD_MAX)
    col_period = rng.randint(PERIOD_MIN, PERIOD_MAX)
    threshold = min(row_period, col_period)

    tile = _make_tile(rng, row_period, col_period)
    height = rng.randint(max(GRID_MIN, threshold + 8), GRID_MAX)
    width = rng.randint(max(GRID_MIN, threshold + 8), GRID_MAX)
    tile_grid = _tile_grid(tile, height, width)

    blocked = np.zeros((height, width), dtype=bool)
    lines: list[dict] = []

    mandatory = [("H", [threshold]), ("V", [threshold + 1])]
    if rng.random() < 0.5:
        mandatory = [("V", [threshold]), ("H", [threshold + 1])]

    for orientation, gaps in mandatory:
        span_limit = width if orientation == "H" else height
        sampled = _sample_segments(rng, span_limit, gaps, threshold + 1)
        if sampled is None:
            return None
        segment_lengths, gap_lengths = sampled
        line = _place_line(rng, blocked, orientation, segment_lengths, gap_lengths)
        if line is None:
            return None
        lines.append(line)

    extra_count = rng.randint(1, 3)
    for _ in range(extra_count):
        orientation = rng.choice(["H", "V"])
        span_limit = width if orientation == "H" else height
        sampled = _sample_segments(rng, span_limit, None, min(4, threshold + 1))
        if sampled is None:
            return None
        segment_lengths, gap_lengths = sampled
        line = _place_line(rng, blocked, orientation, segment_lengths, gap_lengths)
        if line is None:
            return None
        lines.append(line)

    red_cells = {cell for line in lines for cell in line["red_cells"]}
    if not any(line["orientation"] == "H" for line in lines):
        return None
    if not any(line["orientation"] == "V" for line in lines):
        return None

    short_gap_groups = [
        group
        for line in lines
        for gap_len, group in zip(line["gap_lengths"], line["gap_groups"])
        if gap_len == threshold
    ]
    long_gap_groups = [
        group
        for line in lines
        for gap_len, group in zip(line["gap_lengths"], line["gap_groups"])
        if gap_len == threshold + 1
    ]
    if not short_gap_groups or not long_gap_groups:
        return None

    long_gap_group = rng.choice(long_gap_groups)
    marker_cells: set[tuple[int, int]] = set()

    long_gap_marker_choices = [cell for cell in long_gap_group if cell not in red_cells]
    if not long_gap_marker_choices:
        return None
    marker_cells.add(rng.choice(long_gap_marker_choices))

    for color in PATTERN_COLORS:
        color_choices = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in red_cells and tile_grid[r, c] == color
        ]
        rng.shuffle(color_choices)
        placed = False
        for cell in color_choices:
            trial = marker_cells | {cell}
            buckets = _count_visible_by_residue(
                height, width, row_period, col_period, red_cells, trial
            )
            if all(buckets[key] for key in buckets):
                marker_cells = trial
                placed = True
                break
        if not placed:
            return None

    buckets = _count_visible_by_residue(height, width, row_period, col_period, red_cells, marker_cells)
    if not all(buckets[key] for key in buckets):
        return None

    preserve = {rng.choice(cells) for cells in buckets.values()}
    non_red_cells = [(r, c) for r in range(height) for c in range(width) if (r, c) not in red_cells]
    max_markers = max(len(marker_cells), int(0.15 * len(non_red_cells)))
    target_markers = rng.randint(max(3, len(marker_cells)), min(max_markers, len(non_red_cells) - len(preserve)))
    pool = [cell for cell in non_red_cells if cell not in preserve and cell not in marker_cells]
    rng.shuffle(pool)
    for cell in pool:
        if len(marker_cells) >= target_markers:
            break
        trial = marker_cells | {cell}
        trial_buckets = _count_visible_by_residue(
            height, width, row_period, col_period, red_cells, trial
        )
        if all(trial_buckets[key] for key in trial_buckets):
            marker_cells = trial

    visible_after = _count_visible_by_residue(
        height, width, row_period, col_period, red_cells, marker_cells
    )
    if not all(visible_after[key] for key in visible_after):
        return None

    inp = np.array(tile_grid, copy=True)
    for r, c in red_cells:
        inp[r, c] = LINE_COLOR
    for r, c in marker_cells:
        inp[r, c] = MARKER_COLOR

    expected = _expected_output_from_meta(tile_grid, lines, threshold)
    if not _semantic_checks(inp, expected, tile, row_period, col_period, lines, long_gap_group):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-threshold repair grid")
