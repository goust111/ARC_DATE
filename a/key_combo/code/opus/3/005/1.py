from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
TEAL = 8

GEN_PIXEL_COLORS = (BLUE, GREEN, YELLOW)
DIR_DOWN = "down"
DIR_RIGHT = "right"
ORIENT_NWSE = "nwse"
ORIENT_NESW = "nesw"


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _halo(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                out.add((rr, cc))
    return out


def _segment_cells(
    start_r: int, start_c: int, length: int, orientation: str
) -> list[tuple[int, int]]:
    if orientation == ORIENT_NWSE:
        return [(start_r + k, start_c + k) for k in range(length)]
    return [(start_r + k, start_c - k) for k in range(length)]


def _classify_gray_pixels(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    nwse: list[tuple[int, int]] = []
    nesw: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == GRAY):
        r = int(r)
        c = int(c)
        has_nwse = (
            (_in_bounds(g.shape, r - 1, c - 1) and g[r - 1, c - 1] == GRAY)
            or (_in_bounds(g.shape, r + 1, c + 1) and g[r + 1, c + 1] == GRAY)
        )
        has_nesw = (
            (_in_bounds(g.shape, r - 1, c + 1) and g[r - 1, c + 1] == GRAY)
            or (_in_bounds(g.shape, r + 1, c - 1) and g[r + 1, c - 1] == GRAY)
        )
        assert has_nwse != has_nesw, "every gray pixel must belong to exactly one diagonal orientation"
        if has_nwse:
            nwse.append((r, c))
        else:
            nesw.append((r, c))
    return nwse, nesw


def _bar_maps(
    grid: np.ndarray,
) -> tuple[dict[int, list[int]], dict[int, list[int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_bars: dict[int, list[int]] = {}
    col_bars: dict[int, list[int]] = {}
    bar_mask = np.zeros((h, w), dtype=bool)

    for r in range(h):
        row = [int(v) for v in g[r, :]]
        if any(v in (BG, GRAY) for v in row):
            continue
        colors = sorted(set(row))
        if len(colors) > 2:
            continue
        best = max(colors, key=row.count)
        if row.count(best) >= w - 1:
            row_bars.setdefault(best, []).append(r)
            bar_mask[r, :] = True

    for c in range(w):
        col = [int(v) for v in g[:, c]]
        if any(v in (BG, GRAY) for v in col):
            continue
        colors = sorted(set(col))
        if len(colors) > 2:
            continue
        best = max(colors, key=col.count)
        if col.count(best) >= h - 1:
            col_bars.setdefault(best, []).append(c)
            bar_mask[:, c] = True

    for rows in row_bars.values():
        rows.sort()
    for cols in col_bars.values():
        cols.sort()
    return row_bars, col_bars, bar_mask


def _extract_loose_pixels(grid: np.ndarray, bar_mask: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    loose_mask = (g != BG) & (g != GRAY) & ~bar_mask
    coords = np.argwhere(loose_mask)
    out: list[tuple[int, int, int]] = []
    for r, c in coords:
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (0, 1)):
            rr = r + dr
            cc = c + dc
            if _in_bounds(g.shape, rr, cc) and loose_mask[rr, cc]:
                raise AssertionError("loose colored cells must be isolated single pixels")
        out.append((r, c, int(g[r, c])))
    return out


def _settle_loose_pixels(
    loose_pixels: list[tuple[int, int, int]],
    row_bars: dict[int, list[int]],
    col_bars: dict[int, list[int]],
    direction: str,
    *,
    use_blocks: bool,
) -> list[tuple[int, int, int]]:
    if direction == DIR_DOWN:
        ordered = sorted(loose_pixels, key=lambda item: (-item[0], item[1], item[2]))
    else:
        ordered = sorted(loose_pixels, key=lambda item: (-item[1], item[0], item[2]))

    settled: list[tuple[int, int, int]] = []
    for r, c, color in ordered:
        if direction == DIR_DOWN:
            next_bar = next((rr for rr in row_bars.get(color, []) if rr > r), None)
            next_block = None
            if use_blocks:
                next_block = min((rr for rr, cc, _ in settled if cc == c and rr > r), default=None)
            if next_block is not None and (next_bar is None or next_block < next_bar):
                settled.append((next_block - 1, c, color))
            elif next_bar is not None:
                settled.append((next_bar - 1, c, color))
        else:
            next_bar = next((cc for cc in col_bars.get(color, []) if cc > c), None)
            next_block = None
            if use_blocks:
                next_block = min((cc for rr, cc, _ in settled if rr == r and cc > c), default=None)
            if next_block is not None and (next_bar is None or next_block < next_bar):
                settled.append((r, next_block - 1, color))
            elif next_bar is not None:
                settled.append((r, next_bar - 1, color))
    return settled


def _solve(grid: np.ndarray, force_direction: str | None = None, *, use_blocks: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_bars, col_bars, bar_mask = _bar_maps(g)
    nwse, nesw = _classify_gray_pixels(g)
    assert nwse and nesw and len(nwse) != len(nesw), "diagonal pixel orientations must both appear with unequal counts"

    majority = nwse if len(nwse) > len(nesw) else nesw
    minority = nesw if len(nwse) > len(nesw) else nwse
    direction = force_direction
    if direction is None:
        direction = DIR_DOWN if len(nwse) > len(nesw) else DIR_RIGHT

    loose_pixels = _extract_loose_pixels(g, bar_mask)
    out = np.zeros_like(g)
    out[bar_mask] = g[bar_mask]
    for r, c in majority:
        out[r, c] = TEAL
    for r, c in minority:
        out[r, c] = RED
    for r, c, color in _settle_loose_pixels(loose_pixels, row_bars, col_bars, direction, use_blocks=use_blocks):
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor gray diagonal segments by orientation vote, then slide loose pixels along the vote-chosen axis.

    Input structure:
    - Black background contains gray single-cell diagonal segments, full-width or full-height colored bars,
      and isolated colored single pixels away from those bars.
    - Every gray segment runs along exactly one diagonal orientation: NW-SE or NE-SW.
    - Full bars span an entire row or an entire column and keep their original color.

    Rule:
    - Classify each gray pixel by which diagonal neighbor direction it connects through, then count gray pixels
      in the NW-SE group and in the NE-SW group.
    - Recolor the majority-orientation gray pixels to teal (8) and the minority-orientation gray pixels to red (2).
    - If NW-SE is the majority, settle isolated pixels downward; if NE-SW is the majority, settle them rightward.
    - Settle loose pixels nearest the motion front first so stacking is deterministic. Each loose pixel moves until
      it reaches the near side of a same-color full bar, or until it would hit an already settled loose pixel first.
      Gray diagonal pixels do not block this motion.
    - A loose pixel that never finds a same-color bar and never gets blocked by a settled loose pixel is deleted.

    Key insight:
    - The gray diagonals are not just recolored decoration: their orientation vote selects the global motion axis
      for a different object class, so the diagonal majority and the loose-pixel/bar geometry must be combined.
    """
    return _solve(grid)


def _reserve_bar_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}


def _place_diagonal_segments(
    rng: random.Random,
    grid: np.ndarray,
    reserved_cells: set[tuple[int, int]],
    blocked_cells: set[tuple[int, int]],
    major_orientation: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    if major_orientation == ORIENT_NWSE:
        major_lengths = [4, 3]
        minor_lengths = [2, 2, 2]
        major_name = ORIENT_NWSE
        minor_name = ORIENT_NESW
    else:
        major_lengths = [4, 3]
        minor_lengths = [2, 2, 2]
        major_name = ORIENT_NESW
        minor_name = ORIENT_NWSE

    specs = [(major_name, length, True) for length in major_lengths] + [
        (minor_name, length, False) for length in minor_lengths
    ]
    specs.sort(key=lambda item: item[1], reverse=True)

    occupied = set(reserved_cells)
    protected = set(blocked_cells)
    major_cells: list[tuple[int, int]] = []
    minor_cells: list[tuple[int, int]] = []

    for orientation, length, is_major in specs:
        candidates: list[list[tuple[int, int]]] = []
        if orientation == ORIENT_NWSE:
            row_range = range(h - length + 1)
            col_range = range(w - length + 1)
        else:
            row_range = range(h - length + 1)
            col_range = range(length - 1, w)

        for r in row_range:
            for c in col_range:
                cells = _segment_cells(r, c, length, orientation)
                if any(cell in occupied for cell in cells):
                    continue
                if any(cell in protected for cell in cells):
                    continue
                candidates.append(cells)

        if not candidates:
            raise ValueError("failed to place diagonal segments")

        cells = rng.choice(candidates)
        for r, c in cells:
            grid[r, c] = GRAY
        occupied.update(cells)
        protected.update(_halo(cells, grid.shape))
        if is_major:
            major_cells.extend(cells)
        else:
            minor_cells.extend(cells)

    return major_cells, minor_cells


def _build_down_case(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, dict]:
    color_a, color_b, color_c = rng.sample(list(GEN_PIXEL_COLORS), 3)
    grid = np.zeros((h, w), dtype=int)

    row_a = rng.randint(6, h - 5)
    row_b = rng.randint(row_a + 3, h - 2)
    col_a = rng.randint(4, w - 3)
    shared_col = rng.randint(1, col_a - 2)

    src_a_row = rng.randint(3, row_a - 3)
    src_b_row = rng.randint(1, src_a_row - 2)

    delete_cols = [c for c in range(w) if c not in {shared_col, col_a}]
    delete_rows = [r for r in range(1, h - 1) if r not in {row_a, row_b, src_a_row, src_b_row}]
    if not delete_cols or not delete_rows:
        raise ValueError("no deletion slot")
    src_c_row = rng.choice(delete_rows)
    src_c_col = rng.choice(delete_cols)

    grid[row_a, :] = color_a
    grid[:, col_a] = color_a
    grid[row_b, :] = color_b

    src_a = (src_a_row, shared_col)
    src_b = (src_b_row, shared_col)
    src_c = (src_c_row, src_c_col)
    grid[src_a] = color_a
    grid[src_b] = color_b
    grid[src_c] = color_c

    a_final = (row_a - 1, shared_col)
    b_final = (row_a - 2, shared_col)
    protected = _halo([src_a, src_b, src_c, a_final, b_final], grid.shape)
    reserved = _reserve_bar_cells(grid) | {a_final, b_final}
    major_cells, minor_cells = _place_diagonal_segments(rng, grid, reserved, protected, ORIENT_NWSE)

    meta = {
        "direction": DIR_DOWN,
        "colors": {"a": color_a, "b": color_b, "c": color_c},
        "source_a": src_a,
        "source_b": src_b,
        "source_c": src_c,
        "a_final": a_final,
        "b_final": b_final,
        "row_a": row_a,
        "row_b": row_b,
        "col_a": col_a,
        "major_cells": major_cells,
        "minor_cells": minor_cells,
    }
    return grid, meta


def _build_right_case(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, dict]:
    color_a, color_b, color_c = rng.sample(list(GEN_PIXEL_COLORS), 3)
    grid = np.zeros((h, w), dtype=int)

    row_a = rng.randint(4, h - 3)
    col_a = rng.randint(6, w - 5)
    col_b = rng.randint(col_a + 3, w - 2)
    shared_row = rng.randint(1, row_a - 2)

    src_a_col = rng.randint(3, col_a - 3)
    src_b_col = rng.randint(1, src_a_col - 2)

    delete_rows = [r for r in range(h) if r not in {shared_row, row_a}]
    delete_cols = [c for c in range(1, w - 1) if c not in {src_a_col, src_b_col, col_a, col_b}]
    if not delete_rows or not delete_cols:
        raise ValueError("no deletion slot")
    src_c_row = rng.choice(delete_rows)
    src_c_col = rng.choice(delete_cols)

    grid[row_a, :] = color_a
    grid[:, col_a] = color_a
    grid[:, col_b] = color_b

    src_a = (shared_row, src_a_col)
    src_b = (shared_row, src_b_col)
    src_c = (src_c_row, src_c_col)
    grid[src_a] = color_a
    grid[src_b] = color_b
    grid[src_c] = color_c

    a_final = (shared_row, col_a - 1)
    b_final = (shared_row, col_a - 2)
    protected = _halo([src_a, src_b, src_c, a_final, b_final], grid.shape)
    reserved = _reserve_bar_cells(grid) | {a_final, b_final}
    major_cells, minor_cells = _place_diagonal_segments(rng, grid, reserved, protected, ORIENT_NESW)

    meta = {
        "direction": DIR_RIGHT,
        "colors": {"a": color_a, "b": color_b, "c": color_c},
        "source_a": src_a,
        "source_b": src_b,
        "source_c": src_c,
        "a_final": a_final,
        "b_final": b_final,
        "row_a": row_a,
        "col_a": col_a,
        "col_b": col_b,
        "major_cells": major_cells,
        "minor_cells": minor_cells,
    }
    return grid, meta


def _validate_case(grid: np.ndarray, meta: dict) -> None:
    row_bars, col_bars, bar_mask = _bar_maps(grid)
    loose_pixels = _extract_loose_pixels(grid, bar_mask)
    assert len(loose_pixels) == 3

    nwse, nesw = _classify_gray_pixels(grid)
    if meta["direction"] == DIR_DOWN:
        assert len(nwse) > len(nesw)
    else:
        assert len(nesw) > len(nwse)
    assert len(meta["major_cells"]) == 7 and len(meta["minor_cells"]) == 6

    actual = _solve(grid)
    flipped = _solve(grid, DIR_RIGHT if meta["direction"] == DIR_DOWN else DIR_DOWN)
    no_blocks = _solve(grid, meta["direction"], use_blocks=False)

    assert not np.array_equal(actual, flipped), "motion axis must matter"
    assert not np.array_equal(actual, no_blocks), "settled-pixel blocking must matter"
    assert np.array_equal(actual[bar_mask], grid[bar_mask]), "bars must stay unchanged"

    for cell in meta["major_cells"]:
        assert actual[cell] == TEAL
    for cell in meta["minor_cells"]:
        assert actual[cell] == RED

    color_a = meta["colors"]["a"]
    color_b = meta["colors"]["b"]
    color_c = meta["colors"]["c"]
    a_final = meta["a_final"]
    b_final = meta["b_final"]
    assert actual[a_final] == color_a
    assert actual[b_final] == color_b
    assert not np.any(actual == color_c), "the unmatched loose pixel must be deleted"

    if meta["direction"] == DIR_DOWN:
        assert a_final[0] + 1 == meta["row_a"]
        assert b_final == (a_final[0] - 1, a_final[1])
        assert meta["row_b"] > meta["row_a"] + 1
        assert no_blocks[meta["row_b"] - 1, b_final[1]] == color_b
    else:
        assert a_final[1] + 1 == meta["col_a"]
        assert b_final == (a_final[0], a_final[1] - 1)
        assert meta["col_b"] > meta["col_a"] + 1
        assert no_blocks[b_final[0], meta["col_b"] - 1] == color_b


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        h = rng.randint(13, 15)
        w = rng.randint(13, 15)
        try:
            if rng.random() < 0.5:
                grid, meta = _build_down_case(rng, h, w)
            else:
                grid, meta = _build_right_case(rng, h, w)
            _validate_case(grid, meta)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
