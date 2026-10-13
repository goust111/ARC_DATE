from __future__ import annotations

import random

import numpy as np


BG = 0
BAR = 1
SEPARATOR = 2
EDGE = 3
MARK = 4

GEN_HEIGHT_RANGE = (14, 20)
GEN_WIDTH_RANGE = (18, 28)
GEN_RETRY_LIMIT = 400
MIN_BAR_GAP = 2


def _edge_columns(row: np.ndarray) -> list[int]:
    cols = [c for c, value in enumerate(row) if int(value) == EDGE]
    assert cols, "expected one green edge bar"
    assert set(map(int, row)).issubset({BG, EDGE}), "edge row may only contain background and the green bar"
    assert cols == list(range(cols[0], cols[-1] + 1)), "edge bar must be contiguous"
    return cols


def _separator_row(g: np.ndarray) -> int:
    rows = [r for r in range(1, g.shape[0] - 1) if np.all(g[r] == SEPARATOR)]
    assert len(rows) == 1, "expected exactly one full-width red separator row"
    return rows[0]


def _scan_bar(g: np.ndarray, sep_r: int, col: int) -> dict | None:
    rows = [r for r in range(1, sep_r) if int(g[r, col]) in {BAR, MARK}]
    if not rows:
        return None

    assert rows == list(range(rows[0], sep_r)), "vertical bars must be contiguous and bottom-aligned"
    assert rows[-1] == sep_r - 1

    values = [int(g[r, col]) for r in rows]
    height = len(rows)
    assert height % 2 == 0, "every bar height must be even"

    yellow_rows = [r for r in rows if int(g[r, col]) == MARK]
    if yellow_rows:
        half = height // 2
        assert yellow_rows == list(range(rows[0], rows[0] + half)), "marker must occupy the top half"
        assert values[half:] == [BAR] * half, "marked bar lower half must stay blue"
        marked = True
    else:
        assert all(v == BAR for v in values), "unmarked bars must be fully blue"
        marked = False

    return {
        "col": int(col),
        "top": int(rows[0]),
        "rows": np.array(rows, dtype=int),
        "height": int(height),
        "marked": marked,
    }


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    sep_r = _separator_row(g)
    top_cols = _edge_columns(g[0])
    bottom_cols = _edge_columns(g[-1])
    assert len(top_cols) != len(bottom_cols), "one edge bar must be longer"

    overlap = sorted(set(top_cols) & set(bottom_cols))
    assert overlap, "top and bottom edge bars must overlap"

    fill_side = "top" if len(top_cols) > len(bottom_cols) else "bottom"
    fill_rows = range(1, sep_r) if fill_side == "top" else range(sep_r + 1, h - 1)

    bars = []
    for c in range(w):
        bar = _scan_bar(g, sep_r, c)
        if bar is not None:
            bars.append(bar)

    assert bars, "expected at least one internal vertical bar"
    heights = [bar["height"] for bar in bars]
    assert len(set(heights)) == len(heights), "bar heights must be pairwise distinct"

    marked = [bar for bar in bars if bar["marked"]]
    assert len(marked) == 1, "expected exactly one marked bar"
    source = marked[0]

    active_cols = set(overlap)
    assert source["col"] in active_cols, "the marked bar must lie inside the overlap columns"

    target_height = source["height"] + 2
    targets = [bar for bar in bars if bar["col"] in active_cols and bar["height"] == target_height]
    assert len(targets) == 1, "expected exactly one overlap bar with height source+2"
    target = targets[0]
    assert not target["marked"]

    outside = [bar for bar in bars if bar["col"] not in active_cols]
    assert outside, "at least one bar must lie outside the overlap region"

    fill_cells = [(r, c) for r in fill_rows for c in overlap if int(g[r, c]) == BG]
    assert fill_cells, "fill step must add visible yellow cells"

    return {
        "grid": g,
        "sep_r": int(sep_r),
        "overlap": overlap,
        "active_cols": active_cols,
        "bars": bars,
        "outside": outside,
        "source": source,
        "target": target,
        "fill_side": fill_side,
        "fill_cells": fill_cells,
    }


def _apply_transform(
    grid: np.ndarray,
    info: dict,
    *,
    do_fill: bool = True,
    do_erase: bool = True,
    do_move: bool = True,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)

    if do_fill:
        for r, c in info["fill_cells"]:
            out[r, c] = MARK

    if do_erase:
        for bar in info["outside"]:
            out[bar["rows"], bar["col"]] = BG

    if do_move:
        source = info["source"]
        target = info["target"]
        out[source["rows"], source["col"]] = BAR
        half = target["height"] // 2
        out[target["top"] : target["top"] + half, target["col"]] = MARK

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter vertical bars by the overlapping edge-strip columns, then advance the top-half marker to the next taller surviving bar.

    Input structure:
    - The grid uses black background with a green bar on the top edge, a green bar on the bottom edge, and one full-width red separator row inside the grid.
    - Above the separator sit several one-cell-wide vertical bars of distinct even heights, all bottom-aligned to the separator; one of them has its top half recolored yellow as the current marker.

    Rule:
    - Intersect the top and bottom green-bar column sets to get the active overlap columns.
    - On the side whose green edge bar is longer, fill yellow through the overlap columns between that edge and the separator without overwriting existing bar pixels.
    - Erase every internal vertical bar whose column lies outside the overlap columns.
    - Restore the marked source bar to solid blue, then recolor the top half of the unique remaining bar whose height is exactly two pixels larger.

    Key insight:
    - The edge-bar overlap acts as a spatial eligibility filter: it both defines where the yellow fill appears and decides which vertical bars are allowed to participate in the height-based marker progression.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    return _apply_transform(grid, info)


def _sample_edge_segments(width: int, top_longer: bool) -> tuple[list[int], list[int], list[int]] | None:
    max_len = min(width - 2, 12)
    for _ in range(200):
        long_len = random.randint(7, max_len)
        short_len = random.randint(5, long_len - 1)
        top_len, bottom_len = (long_len, short_len) if top_longer else (short_len, long_len)

        top_start = random.randint(0, width - top_len)
        bottom_start = random.randint(0, width - bottom_len)
        top_cols = list(range(top_start, top_start + top_len))
        bottom_cols = list(range(bottom_start, bottom_start + bottom_len))
        overlap = sorted(set(top_cols) & set(bottom_cols))
        if 5 <= len(overlap) < min(top_len, bottom_len):
            return top_cols, bottom_cols, overlap
    return None


def _sample_spaced(pool: list[int], count: int, *, forbidden: set[int] | None = None) -> list[int] | None:
    forbidden = forbidden or set()
    candidates = [c for c in sorted(pool) if c not in forbidden]
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(200):
        choice = sorted(random.sample(candidates, count))
        if all(choice[i + 1] - choice[i] >= MIN_BAR_GAP for i in range(len(choice) - 1)):
            return choice
    return None


def _render_input(
    h: int,
    w: int,
    sep_r: int,
    top_cols: list[int],
    bottom_cols: list[int],
    bar_specs: list[tuple[int, int, bool]],
) -> np.ndarray:
    g = np.full((h, w), BG, dtype=int)
    g[0, top_cols] = EDGE
    g[-1, bottom_cols] = EDGE
    g[sep_r, :] = SEPARATOR

    for col, height, marked in bar_specs:
        top = sep_r - height
        g[top:sep_r, col] = BAR
        if marked:
            g[top : top + height // 2, col] = MARK
    return g


def _generator_checks(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    full = _apply_transform(grid, info)
    if np.array_equal(full, grid):
        return False
    if np.array_equal(full, _apply_transform(grid, info, do_fill=False)):
        return False
    if np.array_equal(full, _apply_transform(grid, info, do_erase=False)):
        return False
    if np.array_equal(full, _apply_transform(grid, info, do_move=False)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    sep_r = random.randint(9, h - 5)
    top_longer = bool(random.getrandbits(1))

    edge_data = _sample_edge_segments(w, top_longer)
    if edge_data is None:
        return None
    top_cols, bottom_cols, overlap = edge_data
    active_cols = set(overlap)

    top_space = sep_r - 1
    heights = [height for height in range(2, top_space + 1, 2)]
    if len(heights) < 4:
        return None

    max_inside = 4 if _sample_spaced(overlap, 4) is not None else 3
    non_overlap = [c for c in range(w) if c not in active_cols]
    max_outside = 2 if _sample_spaced(non_overlap, 2) is not None else 1

    total_bars_max = min(5, len(heights), max_inside + max_outside)
    if total_bars_max < 4:
        return None
    total_bars = random.randint(4, total_bars_max)

    inside_lo = max(3, total_bars - max_outside)
    inside_hi = min(max_inside, total_bars - 1)
    if inside_lo > inside_hi:
        return None
    inside_count = random.randint(inside_lo, inside_hi)
    outside_count = total_bars - inside_count

    source_choices = [height for height in heights if height >= 4 and height + 2 in heights]
    if not source_choices:
        return None
    source_h = random.choice(source_choices)
    target_h = source_h + 2

    remaining = [height for height in heights if height not in {source_h, target_h}]
    if len(remaining) < total_bars - 2:
        return None
    extras = random.sample(remaining, total_bars - 2)
    inside_heights = [source_h, target_h] + extras[: inside_count - 2]
    outside_heights = extras[inside_count - 2 :]

    inside_cols = _sample_spaced(overlap, inside_count)
    if inside_cols is None:
        return None
    source_col, target_col = random.sample(inside_cols, 2)

    forbidden = {
        c2
        for c in inside_cols
        for c2 in range(c - (MIN_BAR_GAP - 1), c + MIN_BAR_GAP)
    }
    outside_cols = _sample_spaced(non_overlap, outside_count, forbidden=forbidden)
    if outside_cols is None:
        return None

    remaining_inside_cols = [c for c in inside_cols if c not in {source_col, target_col}]
    random.shuffle(remaining_inside_cols)
    random.shuffle(outside_cols)

    bar_specs = [(source_col, source_h, True), (target_col, target_h, False)]
    for col, height in zip(remaining_inside_cols, inside_heights[2:]):
        bar_specs.append((col, height, False))
    for col, height in zip(outside_cols, outside_heights):
        bar_specs.append((col, height, False))

    grid = _render_input(h, w, sep_r, top_cols, bottom_cols, bar_specs)
    return grid if _generator_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
