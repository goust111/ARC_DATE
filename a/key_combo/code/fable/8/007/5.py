from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
GREY = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_BAR_LENGTH = 5
GEN_HEIGHT_RANGE = (12, 19)
GEN_WIDTH_RANGE = (9, 18)
GEN_MAX_ATTEMPTS = 400
GEN_REQUIRED_KINDS = ("unit_rest", "staged_rest", "consume_bottom", "board_only")


def _parse_column(grid: np.ndarray, c: int) -> tuple[int, int, list[int]]:
    """Return (bar_length, bar_color, blocker_rows) for one column."""
    col = np.asarray(grid[:, c], dtype=int)
    h = int(col.shape[0])

    length = 0
    color = BLACK
    if int(col[0]) not in (BLACK, GREY):
        color = int(col[0])
        assert color in BAR_COLORS, "bar colors must exclude black and grey"
        while length < h and int(col[length]) == color:
            length += 1
        assert 1 <= length <= MAX_BAR_LENGTH, "bar length must be 1..5"
        assert not np.any((col[length:] != BLACK) & (col[length:] != GREY)), "only blockers may appear below a bar"

    blocker_rows = [int(r) for r in np.flatnonzero(col == GREY)]
    if length > 0:
        assert all(r >= length for r in blocker_rows), "blockers must be below the initial bar"
    assert all(b - a >= 2 for a, b in zip(blocker_rows, blocker_rows[1:])), "blockers need gaps between them"
    assert not np.any((col != BLACK) & (col != GREY) & (col != color)), "each column may have at most one bar color"
    return length, color, blocker_rows


def _simulate_column(height: int, length: int, blocker_rows: list[int]) -> tuple[int, int | None, list[int]]:
    """Step-by-step per-column simulation used by main()."""
    if length == 0:
        return 0, None, list(blocker_rows)

    remaining = set(map(int, blocker_rows))
    current_length = int(length)
    bottom = current_length - 1

    while True:
        if bottom == height - 1:
            break
        next_row = bottom + 1
        if next_row in remaining:
            if current_length == 1:
                break
            remaining.remove(next_row)
            current_length -= 1
        else:
            bottom = next_row

    top = bottom - current_length + 1
    return current_length, int(top), sorted(remaining)


def _oracle_column(height: int, length: int, blocker_rows: list[int]) -> tuple[int, int | None, list[int]]:
    """
    Closed-form oracle for generator validation.

    It reasons only over ordered blockers:
    each encountered blocker consumes one bar cell until the bar reaches length 1,
    and the next blocker (if any) becomes the resting support.
    """
    if length == 0:
        return 0, None, list(blocker_rows)

    current_length = int(length)
    for idx, row in enumerate(blocker_rows):
        if current_length == 1:
            return 1, int(row - 1), blocker_rows[idx:]
        current_length -= 1

    top = height - current_length
    return current_length, int(top), []


def _render_from_columns(grid: np.ndarray, columns: list[tuple[int, int, list[int]]], use_oracle: bool) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h, w), BLACK, dtype=int)
    simulator = _oracle_column if use_oracle else _simulate_column

    for c, (length, color, blocker_rows) in enumerate(columns):
        final_length, top, survivors = simulator(h, length, blocker_rows)
        for row in survivors:
            out[row, c] = GREY
        if final_length > 0:
            assert top is not None
            out[top : top + final_length, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each hanging vertical bar through its own blocker stack, wearing it down one cell per smashed grey plate.

    Input structure:
    - The grid uses black background.
    - Some columns contain one top-attached monochrome vertical bar of length 1 to 5.
    - Grey single-cell blockers may appear lower in any column, with at least one blank row between blockers.
    - Some columns contain only blockers and no bar.

    Rule:
    - Process each column independently.
    - A bar falls downward in its own column.
    - When its bottom reaches a grey blocker: if the bar length is greater than 1, the blocker and the bottom bar cell both disappear and the shortened bar keeps falling; if the bar length is exactly 1, the bar stops on the cell above that blocker.
    - Any remaining bar that reaches the bottom row stops there.
    - Output the surviving bar cells and the untouched blockers; destroyed blockers disappear.

    Key insight:
    - Every destroyed blocker removes exactly one bar cell, so depth gained by a bar is tied to its remaining length, and length-1 bars are the only ones that rest on blockers instead of smashing through them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    columns = [_parse_column(g, c) for c in range(g.shape[1])]
    return _render_from_columns(g, columns, use_oracle=False)


def _max_nonadjacent_count(start_row: int, height: int) -> int:
    rows = height - start_row
    if rows <= 0:
        return 0
    return (rows + 1) // 2


def _sample_nonadjacent_rows(rng: np.random.Generator, start_row: int, height: int, count: int) -> list[int] | None:
    rows = list(range(start_row, height))
    options = [
        combo
        for combo in itertools.combinations(rows, count)
        if all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]
    if not options:
        return None
    choice = options[int(rng.integers(0, len(options)))]
    return [int(v) for v in choice]


def _sample_plan(kind: str, height: int, color: int, rng: np.random.Generator) -> dict | None:
    if kind == "board_only":
        max_count = min(4, _max_nonadjacent_count(2, height))
        if max_count < 1:
            return None
        count = int(rng.integers(1, max_count + 1))
        rows = _sample_nonadjacent_rows(rng, 2, height, count)
        if rows is None:
            return None
        return {"kind": kind, "length": 0, "color": BLACK, "blockers": rows}

    if kind == "unit_rest":
        length = 1
        max_count = min(3, _max_nonadjacent_count(length, height))
        if max_count < 1:
            return None
        count = int(rng.integers(1, max_count + 1))
        rows = _sample_nonadjacent_rows(rng, length, height, count)
        if rows is None:
            return None
        return {"kind": kind, "length": length, "color": color, "blockers": rows}

    if kind == "consume_bottom":
        feasible_lengths = [
            length
            for length in range(2, MAX_BAR_LENGTH + 1)
            if _max_nonadjacent_count(length, height) >= 1
        ]
        if not feasible_lengths:
            return None
        length = int(rng.choice(feasible_lengths))
        max_count = min(length - 1, _max_nonadjacent_count(length, height))
        if max_count < 1:
            return None
        count = int(rng.integers(1, max_count + 1))
        rows = _sample_nonadjacent_rows(rng, length, height, count)
        if rows is None:
            return None
        return {"kind": kind, "length": length, "color": color, "blockers": rows}

    if kind == "staged_rest":
        feasible_lengths = [
            length
            for length in range(3, MAX_BAR_LENGTH + 1)
            if _max_nonadjacent_count(length, height) >= length
        ]
        if not feasible_lengths:
            return None
        length = int(rng.choice(feasible_lengths))
        max_count = min(length + 2, _max_nonadjacent_count(length, height))
        if max_count < length:
            return None
        count = int(rng.integers(length, max_count + 1))
        rows = _sample_nonadjacent_rows(rng, length, height, count)
        if rows is None:
            return None
        return {"kind": kind, "length": length, "color": color, "blockers": rows}

    if kind == "no_blocker":
        length = int(rng.integers(1, MAX_BAR_LENGTH + 1))
        return {"kind": kind, "length": length, "color": color, "blockers": []}

    raise ValueError(f"unknown plan kind: {kind}")


def _shortcut_stop_at_first_blocker(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BLACK)
    for c in range(w):
        length, color, blockers = _parse_column(g, c)
        for row in blockers:
            out[row, c] = GREY
        if length == 0:
            continue
        if blockers:
            top = blockers[0] - length
        else:
            top = h - length
        out[top : top + length, c] = color
    return out


def _shortcut_erase_then_bottom(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BLACK)
    for c in range(w):
        length, color, blockers = _parse_column(g, c)
        if length == 0:
            for row in blockers:
                out[row, c] = GREY
            continue
        top = h - length
        out[top : top + length, c] = color
    return out


def _validate_input(grid: np.ndarray, plans: list[dict], columns: list[int]) -> bool:
    h, w = grid.shape
    if not (grid.ndim == 2 and h <= 30 and w <= 30):
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, GREY, *BAR_COLORS}):
        return False

    active_cols = set(columns)
    parsed = []
    bar_count = 0
    board_only_count = 0
    shortened_count = 0
    bottom_count = 0
    unit_rest_count = 0
    staged_count = 0

    for c in range(w):
        try:
            length, color, blockers = _parse_column(grid, c)
        except AssertionError:
            return False
        parsed.append((length, color, blockers))

        if c in active_cols:
            if length > 0:
                bar_count += 1
            elif blockers:
                board_only_count += 1

    oracle = _render_from_columns(grid, parsed, use_oracle=True)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, oracle):
        return False
    if np.array_equal(actual, grid):
        return False
    if np.array_equal(actual, _shortcut_stop_at_first_blocker(grid)):
        return False
    if np.array_equal(actual, _shortcut_erase_then_bottom(grid)):
        return False

    for c in columns:
        plan = plans[columns.index(c)]
        length, _color, blockers = parsed[c]
        final_length, top, survivors = _oracle_column(h, length, blockers)
        if plan["kind"] == "board_only":
            if blockers != survivors or final_length != 0:
                return False
        if final_length < length:
            shortened_count += 1
        if final_length > 0 and top is not None and top + final_length - 1 == h - 1:
            bottom_count += 1
        if plan["kind"] == "unit_rest":
            if not survivors or final_length != 1 or top != survivors[0] - 1:
                return False
            unit_rest_count += 1
        if plan["kind"] == "staged_rest":
            if final_length != 1 or not survivors:
                return False
            if survivors[0] != top + 1:
                return False
            staged_count += 1

    return (
        bar_count >= 3
        and board_only_count >= 1
        and shortened_count >= 2
        and bottom_count >= 1
        and unit_rest_count >= 1
        and staged_count >= 1
    )


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GEN_HEIGHT_RANGE))
    width = int(rng.integers(*GEN_WIDTH_RANGE))
    candidate_cols = list(range(1, width, 2))
    if len(candidate_cols) < len(GEN_REQUIRED_KINDS):
        return None

    active_count = int(rng.integers(len(GEN_REQUIRED_KINDS), min(len(candidate_cols), 6) + 1))
    active_cols = sorted(rng.choice(candidate_cols, size=active_count, replace=False).tolist())

    kinds = list(GEN_REQUIRED_KINDS)
    extra_choices = ("unit_rest", "staged_rest", "consume_bottom", "board_only", "no_blocker")
    while len(kinds) < active_count:
        kinds.append(str(rng.choice(extra_choices)))
    rng.shuffle(kinds)

    color_pool = list(rng.choice(np.array(BAR_COLORS, dtype=int), size=active_count, replace=False))
    plans: list[dict] = []
    for kind, color in zip(kinds, color_pool):
        plan = _sample_plan(kind, height, int(color), rng)
        if plan is None:
            return None
        plans.append(plan)

    grid = np.full((height, width), BLACK, dtype=int)
    for col, plan in zip(active_cols, plans):
        length = int(plan["length"])
        if length > 0:
            grid[:length, col] = int(plan["color"])
        for row in plan["blockers"]:
            grid[int(row), col] = GREY

    if not _validate_input(grid, plans, active_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid worn-fall input")
