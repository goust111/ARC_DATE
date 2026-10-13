from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

SLOPE_CHOICES = (-3, -2, -1, 0, 1, 2, 3)
SIDE_BEHAVIORS = ("left_exit", "right_clip", "stay_inside")

GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (15, 25)
BAR_LEN_RANGE = (3, 7)
MARKER_COUNT_RANGE = (2, 4)
SIDE_SEARCH_ATTEMPTS = 300
GENERATION_ATTEMPTS = 600


def _sorted_side_points(grid: np.ndarray, color: int, bar_row: int, direction: int) -> list[tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    assert pts, "each side must have at least one marker"

    rows = [r for r, _c in pts]
    assert len(rows) == len(set(rows)), "marker rows must be unique"
    if direction == -1:
        assert all(r < bar_row for r, _c in pts), "upper markers must lie above the red segment"
        pts.sort(key=lambda rc: rc[0], reverse=True)
    else:
        assert all(r > bar_row for r, _c in pts), "lower markers must lie below the red segment"
        pts.sort(key=lambda rc: rc[0])
    return pts


def _trace_side(
    height: int,
    start: tuple[int, int],
    points: list[tuple[int, int]],
    direction: int,
    continue_slope: int | None = None,
) -> tuple[dict[int, int], list[int]]:
    boundary: dict[int, int] = {}
    slopes: list[int] = []

    cur_r, cur_c = map(int, start)
    for nxt_r, nxt_c in points:
        row_gap = direction * (nxt_r - cur_r)
        assert row_gap > 0, "points must advance strictly away from the red segment"
        delta_c = nxt_c - cur_c
        assert delta_c % row_gap == 0, "every segment must have an integer per-row column step"
        slope = delta_c // row_gap
        slopes.append(int(slope))
        for step in range(1, row_gap + 1):
            r = cur_r + direction * step
            c = cur_c + slope * step
            boundary[int(r)] = int(c)
        cur_r, cur_c = int(nxt_r), int(nxt_c)

    assert slopes, "each side must contain at least one segment"
    slope = slopes[-1] if continue_slope is None else int(continue_slope)
    while True:
        nxt_r = cur_r + direction
        if not (0 <= nxt_r < height):
            break
        cur_r = nxt_r
        cur_c += slope
        boundary[int(cur_r)] = int(cur_c)
    return boundary, slopes


def _side_mask(
    shape: tuple[int, int],
    bar_row: int,
    bar_len: int,
    points: list[tuple[int, int]],
    direction: int,
    continue_slope: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    height, width = map(int, shape)
    mask = np.zeros((height, width), dtype=bool)
    boundary, slopes = _trace_side(height, (bar_row, bar_len - 1), points, direction, continue_slope)
    if direction == -1:
        rows = range(bar_row - 1, -1, -1)
    else:
        rows = range(bar_row + 1, height)

    for r in rows:
        col = boundary[int(r)]
        if col < 0:
            break
        mask[int(r), : min(int(col), width - 1) + 1] = True
    return mask, slopes


def _render_output(
    shape: tuple[int, int],
    bar_row: int,
    bar_len: int,
    top_points: list[tuple[int, int]],
    bottom_points: list[tuple[int, int]],
    top_continue_slope: int | None = None,
    bottom_continue_slope: int | None = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    out = np.zeros(shape, dtype=int)
    top_mask, top_slopes = _side_mask(shape, bar_row, bar_len, top_points, -1, top_continue_slope)
    bottom_mask, bottom_slopes = _side_mask(shape, bar_row, bar_len, bottom_points, 1, bottom_continue_slope)
    out[top_mask] = GREEN
    out[bottom_mask] = BLUE
    out[int(bar_row), : int(bar_len)] = RED
    return out, top_slopes, bottom_slopes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the two hidden broken-line borders from their residue pixels and fill the wedges on each side of the red bar.

    Input structure:
    - A black background contains one red horizontal segment anchored at the left edge.
    - Several isolated green pixels lie above the red segment, and several isolated blue pixels lie below it.
    - On each side, those pixels are the successive turning points of an unpainted border polyline that starts at the red segment's right endpoint.

    Rule:
    - Sort the upper markers by row from near the red segment toward the top, and sort the lower markers by row from near the red segment toward the bottom.
    - Connect the red endpoint to every marker in that row order; between consecutive turning points, the border moves by a fixed integer number of columns per row.
    - After the last marker on a side, keep extending with that side's last segment slope to the grid edge; clip fills at the right edge, and stop a side once its border exits left of column 0.
    - Fill every row above the red segment green from the left edge through the upper border column, and every row below it blue from the left edge through the lower border column; keep the red segment.

    Key insight:
    - The visible pixels are not the border itself but only its breakpoints, so the solution must recover a piecewise integer-slope polyline whose final segment also controls the extrapolated rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN}, "unexpected colors"

    red = np.argwhere(g == RED)
    assert red.size > 0, "missing red segment"
    red_rows = np.unique(red[:, 0])
    assert len(red_rows) == 1, "red segment must occupy exactly one row"
    bar_row = int(red_rows[0])
    bar_cols = np.sort(red[:, 1])
    assert np.array_equal(bar_cols, np.arange(bar_cols[-1] + 1)), "red segment must be contiguous and left-anchored"
    bar_len = int(bar_cols[-1] + 1)
    assert np.count_nonzero(g[bar_row] == RED) == bar_len, "red row must contain only the segment"

    top_points = _sorted_side_points(g, GREEN, bar_row, -1)
    bottom_points = _sorted_side_points(g, BLUE, bar_row, 1)
    out, _top_slopes, _bottom_slopes = _render_output(g.shape, bar_row, bar_len, top_points, bottom_points)
    return out


def _partition_with_min(total: int, parts: int, min_part: int) -> list[int]:
    values = [int(min_part)] * int(parts)
    extra = int(total) - int(parts) * int(min_part)
    assert extra >= 0
    for _ in range(extra):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _sample_bar_len(width: int) -> int:
    lo, hi = BAR_LEN_RANGE
    hi = min(hi, width - 6)
    return random.randint(lo, hi)


def _last_slope_candidates(behavior: str) -> list[int]:
    if behavior == "left_exit":
        return [-3, -2, -1]
    if behavior == "right_clip":
        return [1, 2, 3]
    return [-2, -1, 0, 1, 2]


def _build_side(
    total_rows: int,
    bar_row: int,
    bar_len: int,
    width: int,
    direction: int,
) -> dict[str, object] | None:
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    min_rows_needed = 2 * marker_count + 2
    if total_rows < min_rows_needed:
        return None

    max_extrap_rows = total_rows - 2 * marker_count
    extrap_rows = random.randint(2, min(6, max_extrap_rows))
    gaps = _partition_with_min(total_rows - extrap_rows, marker_count, 2)
    behavior = random.choice(SIDE_BEHAVIORS)
    start_col = bar_len - 1

    for _ in range(SIDE_SEARCH_ATTEMPTS):
        slopes: list[int] = []
        prev_slope: int | None = None
        for idx in range(marker_count):
            candidates = _last_slope_candidates(behavior) if idx == marker_count - 1 else list(SLOPE_CHOICES)
            if prev_slope is not None:
                candidates = [s for s in candidates if s != prev_slope]
            if not candidates:
                break
            slope = random.choice(candidates)
            slopes.append(int(slope))
            prev_slope = int(slope)
        if len(slopes) != marker_count:
            continue

        cols: list[int] = []
        cur_col = start_col
        ok = True
        for gap, slope in zip(gaps, slopes):
            cur_col += gap * slope
            if not (0 <= cur_col < width):
                ok = False
                break
            cols.append(int(cur_col))
        if not ok:
            continue

        last_col = cols[-1]
        last_slope = slopes[-1]
        final_col = last_col + extrap_rows * last_slope
        if behavior == "left_exit":
            if not (last_slope < 0 and last_col + last_slope >= 0 and final_col < 0):
                continue
        elif behavior == "right_clip":
            if not (last_slope > 0 and final_col > width - 1):
                continue
        else:
            if not (0 <= final_col <= width - 1):
                continue

        points: list[tuple[int, int]] = []
        cur_row = bar_row
        for gap, col in zip(gaps, cols):
            cur_row += direction * gap
            points.append((int(cur_row), int(col)))
        return {
            "points": points,
            "gaps": gaps,
            "slopes": slopes,
            "behavior": behavior,
            "extrap_rows": extrap_rows,
        }
    return None


def _marker_isolation_ok(grid: np.ndarray, markers: list[tuple[int, int]]) -> bool:
    height, width = grid.shape
    for r, c in markers:
        if grid[r, c] == BLACK:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if grid[nr, nc] != BLACK:
                    return False
    return True


def _side_semantics_ok(
    shape: tuple[int, int],
    bar_row: int,
    bar_len: int,
    points: list[tuple[int, int]],
    direction: int,
) -> bool:
    mask, slopes = _side_mask(shape, bar_row, bar_len, points, direction)
    if len(points) < 2 or len(slopes) < 2:
        return False
    if any(a == b for a, b in zip(slopes, slopes[1:])):
        return False

    last_row = points[-1][0]
    beyond = mask[:last_row, :] if direction == -1 else mask[last_row + 1 :, :]
    if not beyond.any():
        return False

    start_col = bar_len - 1
    total_gap = abs(points[-1][0] - bar_row)
    delta_c = points[-1][1] - start_col
    if delta_c % total_gap == 0:
        straight_mask, _ = _side_mask(shape, bar_row, bar_len, [points[-1]], direction)
        if np.array_equal(mask, straight_mask):
            return False

    alt_mask, _ = _side_mask(shape, bar_row, bar_len, points, direction, continue_slope=slopes[-2])
    if np.array_equal(mask, alt_mask):
        return False
    return True


def _case_is_semantic(
    inp: np.ndarray,
    expected: np.ndarray,
    bar_row: int,
    bar_len: int,
    top_points: list[tuple[int, int]],
    bottom_points: list[tuple[int, int]],
) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not _marker_isolation_ok(inp, top_points + bottom_points):
        return False
    if not _side_semantics_ok(inp.shape, bar_row, bar_len, top_points, -1):
        return False
    if not _side_semantics_ok(inp.shape, bar_row, bar_len, bottom_points, 1):
        return False
    if not np.any(expected == GREEN) or not np.any(expected == BLUE):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bar_len = _sample_bar_len(width)

    top_need = 2 * MARKER_COUNT_RANGE[1] + 2
    bottom_need = 2 * MARKER_COUNT_RANGE[1] + 2
    if height - 1 < top_need + bottom_need:
        return None

    bar_row = random.randint(top_need, height - 1 - bottom_need)
    top_side = _build_side(bar_row, bar_row, bar_len, width, -1)
    bottom_side = _build_side(height - 1 - bar_row, bar_row, bar_len, width, 1)
    if top_side is None or bottom_side is None:
        return None

    top_points = list(top_side["points"])
    bottom_points = list(bottom_side["points"])
    inp = np.zeros((height, width), dtype=int)
    inp[bar_row, :bar_len] = RED
    for r, c in top_points:
        inp[r, c] = GREEN
    for r, c in bottom_points:
        inp[r, c] = BLUE

    expected, _top_slopes, _bottom_slopes = _render_output(inp.shape, bar_row, bar_len, top_points, bottom_points)
    if not _case_is_semantic(inp, expected, bar_row, bar_len, top_points, bottom_points):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid folded-wedge input")
