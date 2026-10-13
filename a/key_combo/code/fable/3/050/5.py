from __future__ import annotations

from collections import Counter
from fractions import Fraction
import random

import numpy as np


Run = dict[str, int]


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _row_runs(grid: np.ndarray, bg: int) -> dict[int, list[Run]]:
    h, w = grid.shape
    by_row: dict[int, list[Run]] = {}
    for r in range(h):
        runs: list[Run] = []
        c = 0
        while c < w:
            if int(grid[r, c]) == int(bg):
                c += 1
                continue
            color = int(grid[r, c])
            c0 = c
            while c < w and int(grid[r, c]) == color:
                c += 1
            runs.append(
                {
                    "r": int(r),
                    "c0": int(c0),
                    "c1": int(c - 1),
                    "length": int(c - c0),
                    "color": int(color),
                }
            )
        if runs:
            by_row[int(r)] = runs
    return by_row


def _split_runs(by_row: dict[int, list[Run]], width: int) -> tuple[list[Run], list[Run], dict[int, Run]]:
    moving: list[Run] = []
    settled: list[Run] = []
    settled_by_row: dict[int, Run] = {}

    for r, runs in by_row.items():
        assert 1 <= len(runs) <= 2, "each active row must contain one or two horizontal runs"
        assert len({run["color"] for run in runs}) == 1, "runs in a row must share one color"

        row_moving = [run for run in runs if run["c1"] == width - 1]
        row_settled = [run for run in runs if run["c1"] != width - 1]
        assert len(row_moving) <= 1 and len(row_settled) <= 1, "expected at most one moving and one settled run per row"

        if row_moving:
            moving.append(row_moving[0])
        if row_settled:
            still = row_settled[0]
            settled.append(still)
            settled_by_row[int(r)] = still

        if row_moving and row_settled:
            assert row_settled[0]["c1"] + 1 < row_moving[0]["c0"], "input runs must be separated before sliding"

    assert moving, "expected at least one right-edge moving run"
    assert len(settled) >= 2, "expected at least two settled runs to determine the barrier"
    return moving, settled, settled_by_row


def _infer_barrier(settled: list[Run]) -> tuple[int, int, int]:
    points = sorted((int(run["r"]), int(run["c0"])) for run in settled)
    rows = [r for r, _c in points]
    assert len(set(rows)) == len(rows), "settled left endpoints must occupy distinct rows"

    r0, c0 = points[0]
    r1, c1 = points[1]
    slope = Fraction(c1 - c0, r1 - r0)
    assert slope.denominator == 1, "barrier must shift by a fixed integer number of columns per row"
    step = int(slope)

    for r, c in points:
        assert int(c) == int(c0 + step * (r - r0)), "settled left endpoints must be collinear"
    return int(r0), int(c0), int(step)


def _line_col(anchor_r: int, anchor_c: int, slope: int, r: int) -> int:
    return int(anchor_c + slope * (int(r) - int(anchor_r)))


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each right-edge horizontal run leftward to the same-row blocker or to the inferred invisible barrier.

    Input structure:
    - A uniform background contains several active rows, each with one or two same-color horizontal runs.
    - Runs whose right end touches the grid's right edge are still moving; runs whose right end does not
      touch the right edge are already settled.
    - The left endpoints of all settled runs lie on one vertical line or on a diagonal that shifts by a
      fixed integer number of columns per row.
    - Some rows have both a settled run and a moving run, while other rows have only a moving run; there
      are no visible obstacle pixels.

    Rule:
    - Keep every settled run unchanged.
    - Infer the invisible barrier from the settled runs' left endpoints and extend it to every row.
    - For each moving run, preserve its color and length while sliding it left until its next step would
      overlap the same-row settled run, or until its left endpoint reaches the barrier column for that row,
      whichever stopping column is encountered first.

    Key insight:
    - Right-edge contact is the only cue for motion.  Rows with visible settled blockers look like ordinary
      collision stops, but moving-only rows reveal the unseen stop surface obtained by extrapolating the
      collinear settled left endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must be at most 30x30"

    bg = _background(g)
    by_row = _row_runs(g, bg)
    assert by_row, "expected horizontal runs"
    moving, settled, settled_by_row = _split_runs(by_row, w)
    anchor_r, anchor_c, slope = _infer_barrier(settled)

    out = g.copy()
    for run in moving:
        out[run["r"], run["c0"] : run["c1"] + 1] = int(bg)

    for run in moving:
        r = int(run["r"])
        barrier_col = _line_col(anchor_r, anchor_c, slope, r)
        assert 0 <= barrier_col < w, "extended barrier must pass through the moving row"

        blocker_col = -10**9
        if r in settled_by_row:
            blocker_col = int(settled_by_row[r]["c1"] + 1)

        dest_c = max(int(barrier_col), int(blocker_col))
        length = int(run["length"])
        assert 0 <= dest_c <= run["c0"], "moving run may only slide left inside the grid"
        assert dest_c + length <= w, "slid run must stay inside the grid"
        assert np.all(out[r, dest_c : dest_c + length] == int(bg)), "slid run must not overlap foreground"
        out[r, dest_c : dest_c + length] = int(run["color"])

    return out


# ---------------- generator ----------------

COLORS = tuple(range(10))
GEN_GRID_H = (12, 24)
GEN_GRID_W = (18, 30)
GEN_ACTIVE_ROWS = (6, 9)
GEN_RETRIES = 600
SLOPE_CYCLE = (0, 1, -1, 0, 1, -1)

SETTLED_LEN = (2, 4)
MOVING_LEN = (2, 5)
LINE_LEFT_MARGIN = 1
LINE_RIGHT_RESERVE = 10

_GEN_CALL_COUNT = 0


def _sample_spaced_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(1, height - 1))
    for _ in range(80):
        random.shuffle(candidates)
        chosen: list[int] = []
        for r in candidates:
            if all(abs(int(r) - int(old)) >= 2 for old in chosen):
                chosen.append(int(r))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _choose_roles(rows: list[int]) -> dict[int, str] | None:
    if len(rows) < 6:
        return None
    roles: dict[int, str] = {int(rows[0]): "moving", int(rows[-1]): "moving"}
    middle = rows[1:-1]
    if len(middle) < 3:
        return None

    both_count = random.randint(3, min(4, len(middle)))
    both_rows = set(random.sample(middle, both_count))
    for r in both_rows:
        roles[int(r)] = "both"

    remaining = [r for r in middle if r not in both_rows]
    random.shuffle(remaining)
    settled_only_count = random.randint(0, min(2, len(remaining)))
    for r in remaining[:settled_only_count]:
        roles[int(r)] = "settled"
    for r in remaining[settled_only_count:]:
        roles[int(r)] = "moving"
    return roles


def _line_anchor_range(rows: list[int], width: int, slope: int, anchor_r: int) -> tuple[int, int] | None:
    offsets = [int(slope) * (int(r) - int(anchor_r)) for r in rows]
    lo = LINE_LEFT_MARGIN - min(offsets)
    hi = width - LINE_RIGHT_RESERVE - max(offsets)
    if lo > hi:
        return None
    return int(lo), int(hi)


def _render_from_plan(
    height: int,
    width: int,
    bg: int,
    rows: list[int],
    roles: dict[int, str],
    anchor_r: int,
    anchor_c: int,
    slope: int,
    row_colors: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]] | None:
    inp = np.full((height, width), int(bg), dtype=int)
    expected = np.full((height, width), int(bg), dtype=int)
    records: list[dict[str, int]] = []

    for r in rows:
        role = roles[int(r)]
        color = int(row_colors[int(r)])
        barrier = _line_col(anchor_r, anchor_c, slope, int(r))
        if not (0 <= barrier < width - LINE_RIGHT_RESERVE + 1):
            return None

        settled_len = 0
        if role in {"both", "settled"}:
            settled_len = random.randint(*SETTLED_LEN)
            if barrier + settled_len >= width:
                return None
            inp[r, barrier : barrier + settled_len] = color
            expected[r, barrier : barrier + settled_len] = color

        moving_len = 0
        if role in {"both", "moving"}:
            moving_len = random.randint(*MOVING_LEN)
            start = width - moving_len
            stop = barrier if role == "moving" else barrier + settled_len
            if not (0 <= stop < start):
                return None
            if stop + moving_len > width:
                return None
            if role == "both" and barrier + settled_len >= start:
                return None
            inp[r, start:width] = color
            expected[r, stop : stop + moving_len] = color
            records.append(
                {
                    "r": int(r),
                    "role": 1 if role == "moving" else 2,
                    "barrier": int(barrier),
                    "input_left": int(start),
                    "output_left": int(stop),
                    "moving_len": int(moving_len),
                    "settled_len": int(settled_len),
                    "color": int(color),
                }
            )

    return inp, expected, records


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    rows: list[int],
    roles: dict[int, str],
    anchor_r: int,
    anchor_c: int,
    slope: int,
    records: list[dict[str, int]],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    settled_rows = [r for r in rows if roles[int(r)] in {"both", "settled"}]
    moving_only_rows = [r for r in rows if roles[int(r)] == "moving"]
    both_rows = [r for r in rows if roles[int(r)] == "both"]
    if len(settled_rows) < 3 or len(moving_only_rows) < 2 or len(both_rows) < 2:
        return False
    if not any(r < min(settled_rows) for r in moving_only_rows):
        return False
    if not any(r > max(settled_rows) for r in moving_only_rows):
        return False

    barrier_cols = {r: _line_col(anchor_r, anchor_c, slope, int(r)) for r in rows}
    if any(c <= 0 for c in barrier_cols.values()):
        return False
    if slope != 0 and len({barrier_cols[r] for r in rows}) < 3:
        return False

    blocked = 0
    hidden = 0
    moved = 0
    for record in records:
        if record["input_left"] <= record["output_left"]:
            return False
        moved += 1
        if record["role"] == 1:
            hidden += 1
            if record["output_left"] != record["barrier"]:
                return False
        else:
            blocked += 1
            if record["output_left"] != record["barrier"] + record["settled_len"]:
                return False
            if record["output_left"] <= record["barrier"]:
                return False
    if blocked < 2 or hidden < 2 or moved != len(records):
        return False

    # The invisible barrier is not reducible to a visible blocker: a moving-only row
    # stops away from the left edge, and a same-row blocker stops strictly to its right.
    if not any(record["role"] == 1 and record["output_left"] > 0 for record in records):
        return False
    if not any(record["role"] == 2 and record["output_left"] > record["barrier"] for record in records):
        return False

    return True


def _gen_attempt(forced_slope: int) -> np.ndarray | None:
    height = random.randint(*GEN_GRID_H)
    width = random.randint(*GEN_GRID_W)
    max_active = min(GEN_ACTIVE_ROWS[1], (height - 1) // 2)
    if max_active < GEN_ACTIVE_ROWS[0]:
        return None
    active_count = random.randint(GEN_ACTIVE_ROWS[0], max_active)
    rows = _sample_spaced_rows(height, active_count)
    if rows is None:
        return None

    roles = _choose_roles(rows)
    if roles is None:
        return None

    slope = int(forced_slope)
    settled_rows = [r for r in rows if roles[int(r)] in {"both", "settled"}]
    anchor_r = int(random.choice(settled_rows))
    anchor_range = _line_anchor_range(rows, width, slope, anchor_r)
    if anchor_range is None:
        return None
    anchor_c = random.randint(*anchor_range)

    bg = 0
    palette = [color for color in COLORS if color != bg]
    random.shuffle(palette)
    row_colors = {int(r): int(random.choice(palette)) for r in rows}

    rendered = _render_from_plan(height, width, bg, rows, roles, anchor_r, anchor_c, slope, row_colors)
    if rendered is None:
        return None
    inp, expected, records = rendered

    if _background(inp) != bg:
        return None
    if not _semantic_checks(inp, expected, rows, roles, anchor_r, anchor_c, slope, records):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    slope = SLOPE_CYCLE[_GEN_CALL_COUNT % len(SLOPE_CYCLE)]
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(slope)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid invisible-barrier sliding input")
