from __future__ import annotations

import random

import numpy as np


ORANGE = 7
RED = 2
GREY = 5
PINK = 6
SIGNAL_COLORS = (RED, GREY)
MAX_ATTEMPTS = 400


def _flip_band(band: np.ndarray) -> np.ndarray:
    out = np.asarray(band, dtype=int).copy()
    out[out == RED] = -1
    out[out == GREY] = RED
    out[out == -1] = GREY
    return out


def _top_span(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    row0 = np.asarray(grid[0], dtype=int)
    cols = np.flatnonzero(np.isin(row0, SIGNAL_COLORS))
    assert cols.size >= 5 and cols.size % 2 == 1, "expected an odd top span of width 5-11"
    left = int(cols[0])
    right = int(cols[-1])
    assert np.array_equal(cols, np.arange(left, right + 1)), "top span must be contiguous"
    span = row0[left : right + 1].copy()
    assert span.size <= 11
    for i in range(1, span.size):
        assert int(span[i]) != int(span[i - 1]), "top span must alternate red/grey"
    assert np.all(np.isin(row0[:left], [ORANGE])) and np.all(np.isin(row0[right + 1 :], [ORANGE]))
    return left, right, span


def _gap_segments(row: np.ndarray) -> list[tuple[int, int]]:
    row = np.asarray(row, dtype=int)
    assert set(map(int, row)).issubset({ORANGE, PINK}), "obstacle rows may only contain orange and pink"
    assert np.any(row == PINK), "obstacle row must contain pink pixels"
    gaps: list[tuple[int, int]] = []
    c = 0
    while c < row.size:
        if int(row[c]) != ORANGE:
            c += 1
            continue
        start = c
        while c < row.size and int(row[c]) == ORANGE:
            c += 1
        gaps.append((int(start), int(c - 1)))
    assert 1 <= len(gaps) <= 2, "each obstacle row must have one or two gaps"
    return gaps


def _obstacles(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    obstacle_rows: dict[int, list[tuple[int, int]]] = {}
    for r in range(1, int(grid.shape[0])):
        if np.any(grid[r] == PINK):
            obstacle_rows[int(r)] = _gap_segments(grid[r])
        else:
            assert set(map(int, grid[r])).issubset({ORANGE}), "non-obstacle rows must be plain orange"
    assert 1 <= len(obstacle_rows) <= 2, "expected one or two obstacle rows"
    return obstacle_rows


def _trace_wedges(
    height: int,
    left: int,
    right: int,
    top_band: np.ndarray,
    obstacle_rows: dict[int, list[tuple[int, int]]],
) -> tuple[list[tuple[int, int, int, np.ndarray]], list[dict]]:
    paints: list[tuple[int, int, int, np.ndarray]] = []
    nodes: list[dict] = []
    stack = [
        {
            "row": 0,
            "left": int(left),
            "right": int(right),
            "band": np.asarray(top_band, dtype=int).copy(),
            "seeded": True,
            "parent": None,
            "from_obstacle": None,
        }
    ]

    while stack:
        state = stack.pop()
        row = int(state["row"])
        cur_left = int(state["left"])
        cur_right = int(state["right"])
        band = np.asarray(state["band"], dtype=int)
        seeded = bool(state["seeded"])
        assert band.size == cur_right - cur_left + 1 and band.size >= 1
        assert row < height

        node_id = len(nodes)
        nodes.append(
            {
                "id": node_id,
                "row": row,
                "left": cur_left,
                "right": cur_right,
                "width": int(band.size),
                "parent": state["parent"],
                "from_obstacle": state["from_obstacle"],
                "seeded": seeded,
            }
        )

        if not seeded:
            paints.append((row, cur_left, cur_right, band.copy()))

        next_row = row + 1
        if next_row >= height:
            continue

        if next_row in obstacle_rows:
            child_row = next_row + 1
            if child_row >= height:
                continue
            gaps = obstacle_rows[next_row]
            for gap_left, gap_right in reversed(gaps):
                overlap_left = max(cur_left, int(gap_left))
                overlap_right = min(cur_right, int(gap_right))
                if overlap_left > overlap_right:
                    continue
                offset_left = overlap_left - cur_left
                offset_right = overlap_right - cur_left + 1
                child_band = band[offset_left:offset_right].copy()
                stack.append(
                    {
                        "row": child_row,
                        "left": overlap_left,
                        "right": overlap_right,
                        "band": child_band,
                        "seeded": False,
                        "parent": node_id,
                        "from_obstacle": next_row,
                    }
                )
            continue

        if band.size < 3:
            continue
        stack.append(
            {
                "row": next_row,
                "left": cur_left + 1,
                "right": cur_right - 1,
                "band": _flip_band(band[:-2]),
                "seeded": False,
                "parent": node_id,
                "from_obstacle": None,
            }
        )

    return paints, nodes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a shrinking red/grey wedge downward and let obstacle gaps filter it into independent child wedges.

    Input structure:
    - The background is orange.
    - Row 0 contains exactly one centered odd-length red/grey alternating span.
    - Lower rows are either plain orange or pink obstacle bars with one or two orange gap segments.

    Rule:
    - Copy the input.
    - Treat the top span as a color band; each normal row below shrinks the active band by one cell per side and flips red<->grey.
    - If the next row is a pink obstacle, do not draw on that row; instead, each overlapping obstacle gap cuts out a contiguous slice of the current band, and that slice reappears one row lower as an independent child wedge.
    - Each child wedge keeps propagating by the same shrink-and-flip rule and may be filtered again by a later obstacle.

    Key insight:
    - The obstacle is not a stopper but a beam splitter: it preserves only the visible gap slices of the current wedge, so gap positions and widths determine every downstream branch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({ORANGE, RED, GREY, PINK}), "unexpected colors"

    left, right, top_band = _top_span(g)
    obstacle_rows = _obstacles(g)

    out = g.copy()
    paints, _nodes = _trace_wedges(int(g.shape[0]), left, right, top_band, obstacle_rows)
    for row, cur_left, cur_right, band in paints:
        assert row not in obstacle_rows, "cannot paint over an obstacle row"
        existing = out[row, cur_left : cur_right + 1]
        assert np.all((existing == ORANGE) | (existing == band)), "branches must not conflict"
        out[row, cur_left : cur_right + 1] = band
    return out


def _alternating_span(width: int) -> np.ndarray:
    start = random.choice(SIGNAL_COLORS)
    other = GREY if start == RED else RED
    return np.array([start if i % 2 == 0 else other for i in range(width)], dtype=int)


def _choose_centered_width() -> tuple[int, int]:
    span_width = random.choice([5, 7, 9, 11])
    pad = random.randint(2, 5)
    width = span_width + 2 * pad
    return width, span_width


def _active_range(left: int, right: int, row: int) -> tuple[int, int]:
    return left + row, right - row


def _place_gap_group(range_left: int, range_right: int, widths: list[int]) -> list[tuple[int, int]]:
    total_width = range_right - range_left + 1
    min_sep_total = len(widths) - 1
    need = sum(widths) + min_sep_total
    assert need <= total_width
    slack = total_width - need
    buckets = [0] * (len(widths) + 1)
    for _ in range(slack):
        buckets[random.randrange(len(buckets))] += 1

    gaps: list[tuple[int, int]] = []
    cursor = range_left + buckets[0]
    for i, width in enumerate(widths):
        gaps.append((int(cursor), int(cursor + width - 1)))
        cursor += width
        if i < len(widths) - 1:
            cursor += 1 + buckets[i + 1]
    return gaps


def _scenario_single_split(left: int, right: int) -> dict[int, list[tuple[int, int]]]:
    options = []
    for row in (2, 3):
        cur_left, cur_right = _active_range(left, right, row - 1)
        width = cur_right - cur_left + 1
        if width >= 7:
            options.append((row, cur_left, cur_right, width))
    obstacle_row, cur_left, cur_right, width = random.choice(options)
    widths = [3, 3] if width == 7 else random.choice([[3, 3], [5, 3], [3, 5]])
    return {int(obstacle_row): _place_gap_group(cur_left, cur_right, widths)}


def _scenario_filter_then_split(left: int, right: int) -> dict[int, list[tuple[int, int]]]:
    cur_left, cur_right = _active_range(left, right, 1)
    first_gap = random.choice(_place_gap_group(cur_left, cur_right, [7]))
    child_left, child_right = first_gap
    second_gaps = _place_gap_group(child_left, child_right, [3, 3])
    return {2: [first_gap], 4: second_gaps}


def _scenario_split_then_filter(left: int, right: int) -> dict[int, list[tuple[int, int]]]:
    cur_left, cur_right = _active_range(left, right, 1)
    first_gaps = _place_gap_group(cur_left, cur_right, [5, 3])
    wide_gap = max(first_gaps, key=lambda seg: seg[1] - seg[0])
    filter_gap = random.choice(_place_gap_group(wide_gap[0], wide_gap[1], [3]))
    return {2: first_gaps, 4: [filter_gap]}


def _make_grid(height: int, width: int, left: int, right: int, obstacles: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    grid = np.full((height, width), ORANGE, dtype=int)
    grid[0, left : right + 1] = _alternating_span(right - left + 1)
    for row, gaps in obstacles.items():
        grid[row, :] = PINK
        for gap_left, gap_right in gaps:
            grid[row, gap_left : gap_right + 1] = ORANGE
    return grid


def _validate_generated(grid: np.ndarray, scenario: str) -> bool:
    left, right, top_band = _top_span(grid)
    obstacle_rows = _obstacles(grid)
    paints, nodes = _trace_wedges(int(grid.shape[0]), left, right, top_band, obstacle_rows)
    out = main(grid)

    if not paints:
        return False
    if np.array_equal(out, grid):
        return False
    if not np.any(np.isin(out[1:], SIGNAL_COLORS)):
        return False

    child_counts: dict[int, int] = {row: 0 for row in obstacle_rows}
    for node in nodes:
        if node["from_obstacle"] is not None:
            child_counts[int(node["from_obstacle"])] += 1

    if scenario == "single_split" and max(child_counts.values()) < 2:
        return False
    if scenario == "filter_then_split":
        if list(sorted(obstacle_rows)) != [2, 4]:
            return False
        if child_counts[2] != 1 or child_counts[4] < 2:
            return False
    if scenario == "split_then_filter":
        if list(sorted(obstacle_rows)) != [2, 4]:
            return False
        if child_counts[2] < 2 or child_counts[4] != 1:
            return False

    obstacle_pixels = grid == PINK
    if not np.array_equal(out[obstacle_pixels], grid[obstacle_pixels]):
        return False
    if np.count_nonzero(np.isin(out, SIGNAL_COLORS)) <= np.count_nonzero(np.isin(grid, SIGNAL_COLORS)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width, span_width = _choose_centered_width()
    left = (width - span_width) // 2
    right = left + span_width - 1

    scenario = random.choices(
        ["single_split", "filter_then_split", "split_then_filter"],
        weights=[4, 3, 3],
        k=1,
    )[0]

    if scenario == "single_split" and span_width < 9:
        return None
    if scenario == "filter_then_split" and span_width < 11:
        return None
    if scenario == "split_then_filter" and span_width < 11:
        return None

    if scenario == "single_split":
        obstacles = _scenario_single_split(left, right)
        min_height = max(obstacles) + 4
    elif scenario == "filter_then_split":
        obstacles = _scenario_filter_then_split(left, right)
        min_height = 8
    else:
        obstacles = _scenario_split_then_filter(left, right)
        min_height = 8

    height = random.randint(min_height, min(min_height + 5, 20))
    grid = _make_grid(height, width, left, right, obstacles)
    if not _validate_generated(grid, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid branching wedge puzzle")
