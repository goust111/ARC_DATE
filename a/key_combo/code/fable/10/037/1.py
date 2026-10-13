from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FLOOR = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (12, 19)
GRID_W_RANGE = (11, 19)
BAR_COUNT_RANGE = (4, 5)
MIN_BAR_LENGTH = 3
MAX_ATTEMPTS = 600


def _most_common_value(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _extract_bars(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    floor_color = int(g[h - 1, 0])
    assert floor_color != BG
    assert np.all(g[h - 1, :] == floor_color), "expected a solid floor on the last row"

    bg = _most_common_value(g[:-1, :]) if h > 1 else _most_common_value(g)
    colors = [int(color) for color in np.unique(g) if color not in (bg, floor_color)]
    assert colors, "expected at least one floating bar"

    bars: list[dict[str, int]] = []
    seen_lengths: set[int] = set()
    for color in colors:
        coords = np.argwhere(g == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        assert rows.size == 1, "each bar must be exactly one cell high"
        row = int(rows[0])
        assert row < h - 1, "bars cannot occupy the floor row"
        left = int(cols.min())
        right = int(cols.max())
        length = right - left + 1
        assert cols.size == length, "each bar must be a solid horizontal segment"
        assert length not in seen_lengths, "bar lengths must be distinct"
        seen_lengths.add(length)
        bars.append(
            {
                "color": int(color),
                "row": row,
                "left": left,
                "right": right,
                "length": length,
            }
        )

    return bg, floor_color, bars


def _simulate(
    shape: tuple[int, int],
    bars: list[dict[str, int]],
    bg: int,
    floor_color: int,
    order: list[dict[str, int]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    out[h - 1, :] = floor_color

    support_top = np.full(w, h - 1, dtype=int)
    placements: list[dict[str, object]] = []

    for bar in order:
        left = int(bar["left"])
        right = int(bar["right"])
        support_slice = support_top[left : right + 1].copy()
        final_row = int(support_slice.min()) - 1
        assert 0 <= final_row < h - 1, "bar must land above the floor"

        out[final_row, left : right + 1] = int(bar["color"])
        support_top[left : right + 1] = np.minimum(support_top[left : right + 1], final_row)
        placements.append(
            {
                "color": int(bar["color"]),
                "source_row": int(bar["row"]),
                "left": left,
                "right": right,
                "length": int(bar["length"]),
                "final_row": final_row,
                "support_levels": tuple(int(x) for x in support_slice),
            }
        )

    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """Drop the floating horizontal bars one by one in descending length order until they stack on the floor and on each other.

    Input structure:
    - The background is black and the last row is a solid grey floor.
    - Above the floor are several single-row horizontal bars, each in its own color.
    - Every bar is a contiguous segment, all bar lengths are distinct, and the bars start at mixed heights.

    Rule:
    - Ignore the initial vertical ordering and sort the bars by length from longest to shortest.
    - Process them in that order; each bar keeps its original column interval and falls straight down.
    - A falling bar stops as soon as any column of the bar would hit the floor or a previously settled bar, so it comes to rest one row above the highest blocking support under its span.
    - The floor stays unchanged and all bars finish in their settled positions.

    Key insight:
    - Length ranking is the time axis: earlier landed longer bars become obstacles, so later shorter bars must be dropped against the already-built stack rather than against the original airborne layout.
    """
    bg, floor_color, bars = _extract_bars(grid)
    order = sorted(bars, key=lambda bar: int(bar["length"]), reverse=True)
    out, _placements = _simulate(np.asarray(grid).shape, bars, bg, floor_color, order)
    return out


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _overlap_graph_connected(intervals: list[tuple[int, int]]) -> bool:
    n = len(intervals)
    seen = {0}
    queue: deque[int] = deque([0])
    while queue:
        i = queue.popleft()
        for j in range(n):
            if j in seen:
                continue
            if _intervals_overlap(intervals[i], intervals[j]):
                seen.add(j)
                queue.append(j)
    return len(seen) == n


def _candidate_intervals(width: int, anchor_a: int, anchor_b: int, role: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for left in range(width):
        for right in range(left + MIN_BAR_LENGTH - 1, width):
            if role == "bridge":
                ok = left <= anchor_a < anchor_b <= right
            elif role == "left":
                ok = left <= anchor_a <= right < anchor_b
            else:
                assert role == "right"
                ok = anchor_a < left <= anchor_b <= right
            if ok:
                candidates.append((left, right))
    random.shuffle(candidates)
    return candidates


def _choose_intervals(width: int, bar_count: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        anchor_a = random.randint(1, width - 5)
        anchor_b = random.randint(anchor_a + 2, width - 2)
        roles = ["bridge", "left", "right"]
        while len(roles) < bar_count:
            roles.append(random.choice(("bridge", "left", "right")))
        random.shuffle(roles)

        pools = {role: _candidate_intervals(width, anchor_a, anchor_b, role) for role in ("bridge", "left", "right")}
        chosen: list[tuple[int, int]] = []
        used_lengths: set[int] = set()

        def backtrack(i: int) -> bool:
            if i == len(roles):
                return True
            role = roles[i]
            for left, right in pools[role]:
                length = right - left + 1
                if length in used_lengths or (left, right) in chosen:
                    continue
                chosen.append((left, right))
                used_lengths.add(length)
                if backtrack(i + 1):
                    return True
                used_lengths.remove(length)
                chosen.pop()
            return False

        if not backtrack(0):
            continue

        max_left = max(left for left, _right in chosen)
        min_right = min(right for _left, right in chosen)
        intervals = list(chosen)
        if max_left <= min_right:
            continue
        if not _overlap_graph_connected(intervals):
            continue
        return intervals
    return None


def _make_wrong_output(
    shape: tuple[int, int],
    bars: list[dict[str, int]],
    bg: int,
    floor_color: int,
    mode: str,
) -> np.ndarray:
    if mode == "length_asc":
        order = sorted(bars, key=lambda bar: int(bar["length"]))
    elif mode == "row_low_first":
        order = sorted(bars, key=lambda bar: int(bar["row"]), reverse=True)
    elif mode == "row_high_first":
        order = sorted(bars, key=lambda bar: int(bar["row"]))
    else:
        raise ValueError(mode)
    out, _ = _simulate(shape, bars, bg, floor_color, order)
    return out


def _bars_from_layout(
    colors: list[int],
    rows: list[int],
    intervals: list[tuple[int, int]],
) -> list[dict[str, int]]:
    bars: list[dict[str, int]] = []
    for color, row, (left, right) in zip(colors, rows, intervals):
        bars.append(
            {
                "color": int(color),
                "row": int(row),
                "left": int(left),
                "right": int(right),
                "length": int(right - left + 1),
            }
        )
    return bars


def _render_input(shape: tuple[int, int], bars: list[dict[str, int]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    grid[h - 1, :] = FLOOR
    for bar in bars:
        row = int(bar["row"])
        left = int(bar["left"])
        right = int(bar["right"])
        grid[row, left : right + 1] = int(bar["color"])
    return grid


def _valid_example(
    shape: tuple[int, int],
    bars: list[dict[str, int]],
) -> bool:
    correct_order = sorted(bars, key=lambda bar: int(bar["length"]), reverse=True)
    correct_out, placements = _simulate(shape, bars, BG, FLOOR, correct_order)

    if any(int(info["source_row"]) >= int(info["final_row"]) for info in placements):
        return False

    landed_on_floor = 0
    landed_on_bar = 0
    multi_level_support = 0
    for info in placements:
        support_levels = tuple(int(x) for x in info["support_levels"])
        if min(support_levels) == shape[0] - 1:
            landed_on_floor += 1
        else:
            landed_on_bar += 1
            if len(set(support_levels)) >= 2:
                multi_level_support += 1

    if landed_on_floor < 1 or landed_on_bar < 2 or multi_level_support < 1:
        return False

    length_order = [int(bar["length"]) for bar in correct_order]
    rows_low_first = [int(bar["length"]) for bar in sorted(bars, key=lambda bar: int(bar["row"]), reverse=True)]
    rows_high_first = [int(bar["length"]) for bar in sorted(bars, key=lambda bar: int(bar["row"]))]
    if rows_low_first == length_order or rows_high_first == length_order:
        return False

    for mode in ("length_asc", "row_low_first", "row_high_first"):
        if np.array_equal(correct_out, _make_wrong_output(shape, bars, BG, FLOOR, mode)):
            return False

    if np.array_equal(correct_out, _render_input(shape, bars)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bar_count = random.randint(*BAR_COUNT_RANGE)

    intervals = _choose_intervals(w, bar_count)
    if intervals is None:
        return None

    max_source_row = h - bar_count - 2
    if max_source_row < 2 or max_source_row < bar_count:
        return None
    rows = random.sample(range(1, max_source_row + 1), bar_count)
    random.shuffle(rows)
    colors = random.sample(BAR_COLORS, bar_count)
    bars = _bars_from_layout(colors, rows, intervals)

    if not _valid_example((h, w), bars):
        return None

    grid = _render_input((h, w), bars)
    if not np.array_equal(main(grid), _simulate((h, w), bars, BG, FLOOR, sorted(bars, key=lambda bar: int(bar["length"]), reverse=True))[0]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate falling-bars input")
