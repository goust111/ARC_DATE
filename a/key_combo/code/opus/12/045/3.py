from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
TEAL = 8

LANE_COLORS = {BLUE, TEAL}
ALLOWED_COLORS = {BLACK, BLUE, RED, GREEN, YELLOW, TEAL}
GEN_ATTEMPTS = 400


def _nonblack_rows(grid: np.ndarray) -> list[int]:
    rows = [r for r in range(grid.shape[0]) if np.any(grid[r] != BLACK)]
    assert rows, "expected at least one colored carrier row"
    for r0, r1 in zip(rows, rows[1:], strict=False):
        assert r1 - r0 >= 2, "carrier rows must be separated by at least one black row"
    return rows


def _parse_lane(row: np.ndarray) -> dict[str, object]:
    unique = set(map(int, np.unique(row)))
    assert unique <= {BLACK, BLUE, RED, GREEN, TEAL}, "unexpected colors in input"
    assert BLACK not in unique, "carrier rows must span the full width"

    carriers = sorted(unique & LANE_COLORS)
    assert len(carriers) == 1, "each carrier row must use exactly one lane color"
    carrier = carriers[0]

    red_cols = np.flatnonzero(row == RED)
    assert 1 <= red_cols.size <= 3, "red segment must be 1..3 cells wide"
    assert int(red_cols[-1] - red_cols[0] + 1) == int(red_cols.size), "red segment must be contiguous"
    red_start = int(red_cols[0])
    red_end = int(red_cols[-1])
    red_len = int(red_cols.size)

    green_cols = np.flatnonzero(row == GREEN)
    assert green_cols.size == 2, "expected exactly two green markers per carrier row"
    period = abs(int(green_cols[1]) - int(green_cols[0]))
    assert 2 <= period <= 5, "green marker spacing must encode a period in 2..5"
    assert np.all((green_cols < red_start - 1) | (green_cols > red_end + 1)), "green markers cannot touch the red segment"

    mask = (row != RED) & (row != GREEN)
    assert np.all(row[mask] == carrier), "all remaining cells in the lane must be the carrier color"

    width = int(row.shape[0])
    if carrier == BLUE:
        dst_start = width - red_len
        track_left = red_start
        track_right = dst_start - 1
    else:
        dst_start = 0
        track_left = red_len
        track_right = red_end

    assert track_left <= track_right, "the red segment must actually slide"
    assert not np.any((green_cols >= dst_start) & (green_cols < dst_start + red_len)), "green markers must stay outside the final red position"

    return {
        "carrier": carrier,
        "red_start": red_start,
        "red_len": red_len,
        "period": period,
        "greens": tuple(int(c) for c in green_cols.tolist()),
        "dst_start": dst_start,
        "track_left": track_left,
        "track_right": track_right,
    }


def _fill_positions(spec: dict[str, object]) -> list[int]:
    period = int(spec["period"])
    left = int(spec["track_left"])
    right = int(spec["track_right"])
    carrier = int(spec["carrier"])
    if carrier == BLUE:
        return list(range(left, right + 1, period))
    return list(range(right, left - 1, -period))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each red segment to the direction encoded by its carrier row, then stripe the exposed track periodically.

    Input structure
    - The grid has a black background with 3-4 full-width horizontal carrier rows separated by black rows.
    - Each carrier row is entirely blue or teal except for one contiguous red segment and exactly two green marker cells.
    - The green markers are not adjacent to the red segment, and their column spacing encodes a period.

    Rule
    - In a blue carrier row, slide the red segment right until it is flush with the grid edge; in a teal carrier row, slide it left until it is flush with the grid edge.
    - Restore the red segment's old cells back to the carrier color, and treat the swept interval from the old position to just before the new position as the track.
    - On that track, place yellow cells every p columns, where p is the absolute column difference between the two green markers; blue rows count from the track's left end, teal rows from the track's right end.
    - If a periodic yellow position lands on a green marker, leave that cell green; all other cells stay unchanged except for the moved red segment.

    Key insight
    - Three independent signals must be combined: the carrier color sets the slide direction, the slide itself defines the fillable track, and the green-marker gap sets the stripe period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN, TEAL}, "unexpected palette in input"

    out = g.copy()
    for r in _nonblack_rows(g):
        spec = _parse_lane(g[r])
        red_start = int(spec["red_start"])
        red_len = int(spec["red_len"])
        dst_start = int(spec["dst_start"])
        carrier = int(spec["carrier"])
        greens = set(spec["greens"])

        out[r, red_start : red_start + red_len] = carrier
        out[r, dst_start : dst_start + red_len] = RED

        for c in _fill_positions(spec):
            if c in greens:
                continue
            out[r, c] = YELLOW

    return out


def _lane_rows_for_height(height: int, lane_count: int, rng: np.random.Generator) -> list[int]:
    picks = sorted(rng.choice(height - (lane_count - 1), size=lane_count, replace=False).tolist())
    return [int(p + i) for i, p in enumerate(picks)]


def _carrier_sequence(lane_count: int, rng: np.random.Generator) -> list[int]:
    colors = [BLUE if bool(rng.integers(0, 2)) else TEAL for _ in range(lane_count)]
    if all(color == BLUE for color in colors):
        colors[int(rng.integers(0, lane_count))] = TEAL
    if all(color == TEAL for color in colors):
        colors[int(rng.integers(0, lane_count))] = BLUE
    return colors


def _track_and_beats(width: int, carrier: int, red_start: int, red_len: int, period: int) -> tuple[tuple[int, int], set[int]]:
    if carrier == BLUE:
        left = red_start
        right = width - red_len - 1
        beats = set(range(left, right + 1, period))
    else:
        left = red_len
        right = red_start + red_len - 1
        beats = set(range(right, left - 1, -period))
    return (left, right), beats


def _marker_pairs(width: int, carrier: int, red_start: int, red_len: int, period: int) -> list[tuple[int, int]]:
    banned = set(range(red_start, red_start + red_len))
    if red_start - 1 >= 0:
        banned.add(red_start - 1)
    if red_start + red_len < width:
        banned.add(red_start + red_len)
    if carrier == BLUE:
        banned.update(range(width - red_len, width))
    else:
        banned.update(range(red_len))

    pairs = []
    for c0 in range(width - period):
        c1 = c0 + period
        if c0 in banned or c1 in banned:
            continue
        pairs.append((c0, c1))
    return pairs


def _build_lane_spec(
    width: int,
    carrier: int,
    need_skip: bool,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    periods = list(range(2, min(5, width - 1) + 1))
    rng.shuffle(periods)

    for _ in range(120):
        red_len = int(rng.integers(1, 4))
        if carrier == BLUE:
            max_start = width - red_len - 4
            if max_start < 0:
                continue
            red_start = int(rng.integers(0, max_start + 1))
        else:
            min_start = 4
            max_start = width - red_len
            if min_start > max_start:
                continue
            red_start = int(rng.integers(min_start, max_start + 1))

        for period in periods:
            (track_left, track_right), beats = _track_and_beats(width, carrier, red_start, red_len, period)
            pairs = _marker_pairs(width, carrier, red_start, red_len, period)
            desired = []
            for greens in pairs:
                on_beats = any(c in beats for c in greens)
                if on_beats != need_skip:
                    continue
                paint_positions = [c for c in beats if c not in greens]
                if not paint_positions:
                    continue
                desired.append((greens, paint_positions))
            if not desired:
                continue

            greens, paint_positions = desired[int(rng.integers(0, len(desired)))]
            return {
                "carrier": carrier,
                "red_start": red_start,
                "red_len": red_len,
                "period": period,
                "greens": greens,
                "track_left": track_left,
                "track_right": track_right,
                "paint_positions": tuple(sorted(paint_positions)),
            }
    return None


def _oracle_output(grid: np.ndarray, row_specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    width = grid.shape[1]
    for spec in row_specs:
        r = int(spec["row"])
        carrier = int(spec["carrier"])
        red_start = int(spec["red_start"])
        red_len = int(spec["red_len"])
        greens = set(int(c) for c in spec["greens"])

        out[r, red_start : red_start + red_len] = carrier
        if carrier == BLUE:
            dst_start = width - red_len
        else:
            dst_start = 0
        out[r, dst_start : dst_start + red_len] = RED

        for c in _fill_positions(
            {
                "carrier": carrier,
                "track_left": int(spec["track_left"]),
                "track_right": int(spec["track_right"]),
                "period": int(spec["period"]),
            }
        ):
            if c not in greens:
                out[r, c] = YELLOW
    return out


def _render_input(height: int, width: int, row_specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in row_specs:
        r = int(spec["row"])
        carrier = int(spec["carrier"])
        red_start = int(spec["red_start"])
        red_len = int(spec["red_len"])
        greens = tuple(int(c) for c in spec["greens"])
        grid[r, :] = carrier
        grid[r, greens[0]] = GREEN
        grid[r, greens[1]] = GREEN
        grid[r, red_start : red_start + red_len] = RED
    return grid


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    lane_count = int(rng.integers(3, 5))
    height = int(rng.integers(max(7, 2 * lane_count - 1), 12))
    width = int(rng.integers(13, 21))
    lane_rows = _lane_rows_for_height(height, lane_count, rng)
    carriers = _carrier_sequence(lane_count, rng)

    skip_flags = [bool(rng.integers(0, 2)) for _ in range(lane_count)]
    skip_flags[int(rng.integers(0, lane_count))] = True
    if all(skip_flags):
        skip_flags[int(rng.integers(0, lane_count))] = False
    if not any(flag is False for flag in skip_flags):
        skip_flags[0] = False
        skip_flags[-1] = True

    row_specs: list[dict[str, object]] = []
    for r, carrier, need_skip in zip(lane_rows, carriers, skip_flags, strict=False):
        spec = _build_lane_spec(width, carrier, need_skip, rng)
        if spec is None:
            return None
        spec["row"] = r
        spec["need_skip"] = need_skip
        row_specs.append(spec)

    periods = {int(spec["period"]) for spec in row_specs}
    if len(periods) < 2:
        return None

    grid = _render_input(height, width, row_specs)
    expected = _oracle_output(grid, row_specs)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not np.any(actual == YELLOW):
        return None
    if not any(bool(spec["need_skip"]) for spec in row_specs):
        return None
    if not any(not bool(spec["need_skip"]) for spec in row_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(np.random.default_rng())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
