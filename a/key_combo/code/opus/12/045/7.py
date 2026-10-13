from __future__ import annotations

import itertools
import random

import numpy as np


BG_COLOR = 0
PIVOT_COLOR = 5
BAR_COLOR = 7
PALETTE = (1, 2, 3, 4, 6, 8, 9)

MIN_SIZE = 10
MAX_SIZE = 16
MIN_PIXELS = 3
MAX_PIXELS = 5
GEN_RETRIES = 500
OVERLAP_PROB = 0.55


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_bar_row(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == BAR_COLOR)]
    assert len(rows) == 1, "expected exactly one full-width orange bar"
    return int(rows[0])


def _find_pivot(g: np.ndarray) -> tuple[int, int]:
    pivots = np.argwhere(g == PIVOT_COLOR)
    assert len(pivots) == 1, "expected exactly one grey pivot"
    return tuple(map(int, pivots[0]))


def _source_pixels(g: np.ndarray) -> list[tuple[int, int, int]]:
    mask = (g != BG_COLOR) & (g != PIVOT_COLOR) & (g != BAR_COLOR)
    coords = np.argwhere(mask)
    return [(int(r), int(c), int(g[r, c])) for r, c in coords]


def _rotate_about_pivot(pr: int, pc: int, r: int, c: int) -> tuple[int, int]:
    return 2 * pr - r, 2 * pc - c


def _ray_trace(shape: tuple[int, int], start: tuple[int, int], dc: int) -> list[tuple[int, int]]:
    h, w = shape
    rr = start[0] - 1
    cc = start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr -= 1
        cc += dc
    return cells


def _render_output(
    shape: tuple[int, int],
    bar_r: int,
    pivot: tuple[int, int],
    rotated_pixels: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full(shape, BG_COLOR, dtype=int)
    out[bar_r, :] = BAR_COLOR
    out[pivot] = PIVOT_COLOR

    occupied = {pivot}
    for r, c, color in rotated_pixels:
        assert out[r, c] == BG_COLOR, "rotated pixels must start on background"
        out[r, c] = color
        occupied.add((r, c))

    triggers = sorted(
        [(r, c, color) for r, c, color in rotated_pixels if r == bar_r - 1],
        key=lambda item: item[1],
    )
    for r, c, color in triggers:
        for dc in (-1, 1):
            for rr, cc in _ray_trace(shape, (r, c), dc):
                if (rr, cc) == pivot:
                    continue
                out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored pixel 180 degrees around the grey pivot, then let the rotated pixels that land just above the orange bar spray upward diagonals.

    Input structure
    - The grid has a black background, exactly one full-width orange horizontal bar, and exactly one grey pivot pixel above that bar.
    - Three to five other colored single pixels lie around the pivot area and never occupy the bar.
    - The grey pivot and orange bar are fixed landmarks; every other non-black pixel is a movable source.

    Rule
    - Replace each colored source pixel at (r, c) by its 180-degree rotation about the grey pivot at (pr, pc), namely (2*pr-r, 2*pc-c), preserving color.
    - Keep the grey pivot in place and keep the orange horizontal bar unchanged.
    - Any rotated pixel whose landing row is exactly one row above the bar becomes a trigger and emits one up-left and one up-right diagonal ray in its own color until the grid edge.
    - Rays overwrite other painted cells if needed, but they never overwrite the grey pivot; if multiple trigger rays overlap, triggers are drawn from smaller landing column to larger landing column, so larger-column triggers win on shared cells.

    Key insight
    - Triggering is not decided from the input positions themselves: you must first perform the half-turn around the pivot, then test which rotated landings end up adjacent to the bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bar_r = _find_bar_row(g)
    pr, pc = _find_pivot(g)
    assert pr < bar_r - 1, "pivot must lie above the trigger row"

    sources = _source_pixels(g)
    assert sources, "expected colored source pixels"
    assert all(r != bar_r for r, _c, _color in sources), "sources cannot lie on the bar"

    out = np.full_like(g, BG_COLOR)
    out[bar_r, :] = BAR_COLOR
    out[pr, pc] = PIVOT_COLOR

    rotated_pixels: list[tuple[int, int, int]] = []
    occupied = {(pr, pc)}
    for r, c, color in sources:
        rr, cc = _rotate_about_pivot(pr, pc, r, c)
        assert _in_bounds(g.shape, rr, cc), "rotated pixel left the grid"
        assert rr < bar_r, "rotated pixels must stay above the bar"
        assert (rr, cc) not in occupied, "rotated pixels must not collide"
        occupied.add((rr, cc))
        out[rr, cc] = color
        rotated_pixels.append((rr, cc, color))

    triggers = sorted(
        [(r, c, color) for r, c, color in rotated_pixels if r == bar_r - 1],
        key=lambda item: item[1],
    )
    for r, c, color in triggers:
        for dc in (-1, 1):
            for rr, cc in _ray_trace(g.shape, (r, c), dc):
                if (rr, cc) == (pr, pc):
                    continue
                out[rr, cc] = color

    return out


def _trigger_combo_info(
    shape: tuple[int, int],
    bar_r: int,
    pivot: tuple[int, int],
    cols: tuple[int, ...],
) -> tuple[dict[int, tuple[tuple[int, int], ...]], dict[tuple[int, int], tuple[int, ...]]] | None:
    per_col: dict[int, tuple[tuple[int, int], ...]] = {}
    touched_by: dict[tuple[int, int], list[int]] = {}
    for col in cols:
        cells = tuple(
            itertools.chain(
                _ray_trace(shape, (bar_r - 1, col), -1),
                _ray_trace(shape, (bar_r - 1, col), 1),
            )
        )
        if pivot in cells:
            return None
        per_col[col] = cells
        for cell in cells:
            touched_by.setdefault(cell, []).append(col)
    overlaps = {
        cell: tuple(sorted(cols_here))
        for cell, cols_here in touched_by.items()
        if len(cols_here) > 1
    }
    return per_col, overlaps


def _pick_trigger_columns(
    shape: tuple[int, int],
    bar_r: int,
    pivot: tuple[int, int],
    trigger_count: int,
    want_overlap: bool,
) -> tuple[tuple[int, ...], dict[int, tuple[tuple[int, int], ...]], dict[tuple[int, int], tuple[int, ...]]] | None:
    h, w = shape
    pr, pc = pivot
    trigger_src_r = 2 * pr - (bar_r - 1)
    valid_cols: list[int] = []
    for c in range(1, w - 1):
        sc = 2 * pc - c
        if not _in_bounds(shape, trigger_src_r, sc):
            continue
        if abs(c - pc) == bar_r - 1 - pr:
            continue
        valid_cols.append(c)

    candidates = []
    for cols in itertools.combinations(valid_cols, trigger_count):
        info = _trigger_combo_info(shape, bar_r, pivot, cols)
        if info is None:
            continue
        per_col, overlaps = info
        has_overlap = bool(overlaps)
        if has_overlap != want_overlap:
            continue
        if len(overlaps) > max(2, trigger_count):
            continue
        candidates.append((cols, per_col, overlaps))

    if not candidates:
        return None
    return random.choice(candidates)


def _pick_nontrigger_outputs(
    shape: tuple[int, int],
    bar_r: int,
    pivot: tuple[int, int],
    trigger_cols: tuple[int, ...],
    count: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    pr, pc = pivot
    trigger_outputs = [(bar_r - 1, col) for col in trigger_cols]
    candidates = []
    for r in range(pr, bar_r - 1):
        for c in range(shape[1]):
            if (r, c) == pivot or (r, c) in blocked:
                continue
            sr, sc = _rotate_about_pivot(pr, pc, r, c)
            if not _in_bounds(shape, sr, sc):
                continue
            if (sr, sc) == pivot or sr >= bar_r:
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    occupied = list(trigger_outputs)
    for cand in candidates:
        if any(max(abs(cand[0] - r), abs(cand[1] - c)) <= 1 for r, c in occupied):
            continue
        chosen.append(cand)
        occupied.append(cand)
        if len(chosen) == count:
            return chosen
    return None


def _validate_input_structure(g: np.ndarray) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], tuple[int, int, int]]]:
    bar_r = _find_bar_row(g)
    pr, pc = _find_pivot(g)
    sources = _source_pixels(g)

    assert MIN_PIXELS <= len(sources) <= MAX_PIXELS, "generator must create 3-5 sources"
    assert len({color for _r, _c, color in sources}) == len(sources), "source colors should be distinct"
    assert pr <= bar_r - 3, "pivot should not be adjacent to the bar"

    rotated_by_source: dict[tuple[int, int], tuple[int, int, int]] = {}
    for r, c, color in sources:
        rr, cc = _rotate_about_pivot(pr, pc, r, c)
        assert _in_bounds(g.shape, rr, cc), "rotated position left the grid"
        assert rr < bar_r, "rotated position must stay above the bar"
        rotated_by_source[(r, c)] = (rr, cc, color)

    rotated_pixels = list(rotated_by_source.values())
    assert len({(r, c) for r, c, _color in rotated_pixels}) == len(rotated_pixels), "rotated positions must be unique"

    trigger_sources = [
        (src, dst)
        for src, dst in rotated_by_source.items()
        if dst[0] == bar_r - 1
    ]
    nontrigger_sources = [
        (src, dst)
        for src, dst in rotated_by_source.items()
        if dst[0] < bar_r - 1
    ]
    assert trigger_sources, "need at least one triggered pixel"
    assert nontrigger_sources, "need at least one non-triggered pixel"
    assert all(src[0] <= pr for src, _dst in trigger_sources), "triggering sources should sit on or above the pivot band"
    assert all(abs(src[0] - (bar_r - 1)) >= 2 for src, _dst in trigger_sources), "trigger must depend on rotation, not pre-existing adjacency"

    return rotated_pixels, rotated_by_source


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    bar_r = _find_bar_row(inp)
    pivot = _find_pivot(inp)
    rotated_pixels, _rotated_by_source = _validate_input_structure(inp)

    expected = _render_output(inp.shape, bar_r, pivot, rotated_pixels)
    assert np.array_equal(out, expected), "main output disagrees with the constructed scene"
    assert np.all(out[bar_r, :] == BAR_COLOR), "bar must remain unchanged"
    assert out[pivot] == PIVOT_COLOR, "pivot must remain unchanged"

    trigger_cols = sorted(c for r, c, _color in rotated_pixels if r == bar_r - 1)
    trigger_color_by_col = {
        c: color for r, c, color in rotated_pixels if r == bar_r - 1
    }
    info = _trigger_combo_info(inp.shape, bar_r, pivot, tuple(trigger_cols))
    assert info is not None, "generated trigger columns should stay valid"
    _per_col, overlaps = info
    for cell, cols in overlaps.items():
        winner = max(cols)
        assert out[cell] == trigger_color_by_col[winner], "rightmost trigger should win on overlapping rays"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)

    bar_lo = max(7, h // 2 + 2)
    bar_hi = h - 3
    if bar_lo > bar_hi:
        return None
    bar_r = random.randint(bar_lo, bar_hi)

    pivot_lo = (bar_r + 1) // 2
    pivot_hi = bar_r - 4
    if pivot_lo > pivot_hi:
        return None
    pr = random.randint(pivot_lo, pivot_hi)

    if w < 8:
        return None
    pc = random.randint(3, w - 4)
    pivot = (pr, pc)

    total = random.randint(MIN_PIXELS, MAX_PIXELS)
    max_triggers = min(3, total - 1)
    trigger_choices = [count for count in (1, 2, 3) if count <= max_triggers]
    weights = {1: 3, 2: 4, 3: 2}
    trigger_count = random.choices(trigger_choices, weights=[weights[c] for c in trigger_choices], k=1)[0]
    want_overlap = trigger_count >= 2 and (trigger_count == 3 or random.random() < OVERLAP_PROB)

    trigger_pick = _pick_trigger_columns((h, w), bar_r, pivot, trigger_count, want_overlap)
    if trigger_pick is None:
        return None
    trigger_cols, per_col, overlaps = trigger_pick

    blocked = {(bar_r - 1, c) for c in trigger_cols}
    for cells in per_col.values():
        blocked.update(cells)

    nontrigger_outputs = _pick_nontrigger_outputs(
        (h, w),
        bar_r,
        pivot,
        trigger_cols,
        total - trigger_count,
        blocked,
    )
    if nontrigger_outputs is None:
        return None

    output_positions = [(bar_r - 1, c) for c in trigger_cols] + nontrigger_outputs
    colors = random.sample(PALETTE, len(output_positions))
    random.shuffle(output_positions)
    rotated_pixels = [
        (r, c, color) for (r, c), color in zip(output_positions, colors)
    ]

    inp = np.full((h, w), BG_COLOR, dtype=int)
    inp[bar_r, :] = BAR_COLOR
    inp[pivot] = PIVOT_COLOR
    for rr, cc, color in rotated_pixels:
        sr, sc = _rotate_about_pivot(pr, pc, rr, cc)
        if inp[sr, sc] != BG_COLOR:
            return None
        inp[sr, sc] = color

    try:
        built_rotated, mapping = _validate_input_structure(inp)
        if want_overlap:
            actual_trigger_cols = tuple(sorted(c for r, c, _color in built_rotated if r == bar_r - 1))
            actual_info = _trigger_combo_info(inp.shape, bar_r, pivot, actual_trigger_cols)
            assert actual_info is not None and actual_info[1], "overlap branch was requested but not realized"
        else:
            actual_trigger_cols = tuple(sorted(c for r, c, _color in built_rotated if r == bar_r - 1))
            actual_info = _trigger_combo_info(inp.shape, bar_r, pivot, actual_trigger_cols)
            assert actual_info is not None and not actual_info[1], "unexpected ray overlap in non-overlap branch"

        for (src_r, _src_c), (dst_r, _dst_c, _color) in mapping.items():
            if dst_r == bar_r - 1:
                assert src_r != dst_r, "a trigger must be created by rotation, not already sit on the trigger row"

        out = main(inp)
        _validate_output_semantics(inp, out)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
