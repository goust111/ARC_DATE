from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

NON_MAROON_SHELVES = (GREEN, YELLOW, ORANGE, GRAY, MAGENTA)
SHELF_COLORS = NON_MAROON_SHELVES + (MAROON,)
GEN_RETRIES = 500


def _find_shelf_span(grid: np.ndarray, row: int, col: int) -> tuple[int, int, int]:
    color = int(grid[row, col])
    c0 = int(col)
    c1 = int(col)
    while c0 - 1 >= 0 and int(grid[row, c0 - 1]) == color:
        c0 -= 1
    while c1 + 1 < grid.shape[1] and int(grid[row, c1 + 1]) == color:
        c1 += 1
    return c0, c1, color


def _source_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in np.flatnonzero(grid[0] == BLUE)]


def _simulate(grid: np.ndarray, collect_stats: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    out = g.copy()
    inf = 1 << 30
    water_time = np.full((h, w), inf, dtype=int)
    claim_time = np.full((h, w), inf, dtype=int)
    triggered_shelves: set[tuple[int, int, int]] = set()

    stats = {
        "sources": 0,
        "splits": 0,
        "merges": 0,
        "absorbed": 0,
        "bottoms": 0,
    }

    queue: list[tuple[int, int, int, int, int]] = []
    order = 0
    for col in _source_columns(g):
        water_time[0, col] = 0
        stats["sources"] += 1
        heapq.heappush(queue, (0, order, 0, col, BLUE))
        order += 1

    while queue:
        time, _seq, row, col, color = heapq.heappop(queue)
        if not (0 <= row < h and 0 <= col < w):
            continue
        if int(g[row, col]) not in (BLACK, BLUE):
            continue
        if int(water_time[row, col]) < time:
            stats["merges"] += 1
            continue
        if int(water_time[row, col]) > time:
            water_time[row, col] = time
            out[row, col] = int(color)

        if row == h - 1:
            stats["bottoms"] += 1
            continue

        below = int(g[row + 1, col])
        if below in SHELF_COLORS:
            if below == MAROON:
                stats["absorbed"] += 1
                continue

            c0, c1, shelf_color = _find_shelf_span(g, row + 1, col)
            shelf_key = (row + 1, c0, c1)
            if shelf_key in triggered_shelves:
                continue
            triggered_shelves.add(shelf_key)
            stats["splits"] += 1

            left = max(0, c0 - 1)
            right = min(w - 1, c1 + 1)
            for cc in range(left, right + 1):
                if int(g[row, cc]) in (BLACK, BLUE) and int(water_time[row, cc]) >= time:
                    out[row, cc] = int(shelf_color)
                    if int(water_time[row, cc]) > time:
                        water_time[row, cc] = time

            if c0 - 1 >= 0:
                heapq.heappush(queue, (time, order, row, c0 - 1, shelf_color))
                order += 1
            if c1 + 1 < w:
                heapq.heappush(queue, (time, order, row, c1 + 1, shelf_color))
                order += 1
            continue

        next_row = row + 1
        if int(water_time[next_row, col]) != inf or int(claim_time[next_row, col]) <= time + 1:
            stats["merges"] += 1
            continue
        claim_time[next_row, col] = time + 1
        heapq.heappush(queue, (time + 1, order, next_row, col, color))
        order += 1

    if collect_stats:
        return out, stats
    return out, {}


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, collect_stats=False)[0]


def _sample_row(
    rng: np.random.Generator,
    low: int,
    high: int,
    used_rows: Iterable[int],
) -> int | None:
    choices = [r for r in range(int(low), int(high) + 1) if all(abs(r - u) >= 3 for u in used_rows)]
    if not choices:
        return None
    return int(rng.choice(np.array(choices, dtype=int)))


def _sample_covering_span(
    rng: np.random.Generator,
    hit_col: int,
    width: int,
    forbidden_in_span: Iterable[int] = (),
    forbidden_children: Iterable[int] = (),
) -> tuple[int, int] | None:
    span_forbid = tuple(int(x) for x in forbidden_in_span)
    child_forbid = {int(x) for x in forbidden_children}
    options: list[tuple[int, int]] = []
    for length in range(3, 9):
        start_lo = max(2, int(hit_col) - length + 1)
        start_hi = min(int(hit_col), int(width) - length - 2)
        for start in range(start_lo, start_hi + 1):
            end = start + length - 1
            if any(start <= col <= end for col in span_forbid):
                continue
            if start - 1 in child_forbid or end + 1 in child_forbid:
                continue
            options.append((start, end))
    if not options:
        return None
    return options[int(rng.integers(0, len(options)))]


def _sample_span_with_child(
    rng: np.random.Generator,
    hit_col: int,
    child_col: int,
    width: int,
    child_side: str,
    forbidden_in_span: Iterable[int] = (),
    forbidden_children: Iterable[int] = (),
) -> tuple[int, int] | None:
    span_forbid = tuple(int(x) for x in forbidden_in_span)
    child_forbid = {int(x) for x in forbidden_children}
    options: list[tuple[int, int]] = []
    if child_side == "left":
        start = int(child_col) + 1
        for length in range(3, 9):
            end = start + length - 1
            if end > int(width) - 3:
                continue
            if not (start <= int(hit_col) <= end):
                continue
            if any(start <= col <= end for col in span_forbid):
                continue
            other_child = end + 1
            if other_child in child_forbid:
                continue
            options.append((start, end))
    else:
        end = int(child_col) - 1
        for length in range(3, 9):
            start = end - length + 1
            if start < 2:
                continue
            if not (start <= int(hit_col) <= end):
                continue
            if any(start <= col <= end for col in span_forbid):
                continue
            other_child = start - 1
            if other_child in child_forbid:
                continue
            options.append((start, end))
    if not options:
        return None
    return options[int(rng.integers(0, len(options)))]


def _choose_non_maroon(rng: np.random.Generator, exclude: Iterable[int] = ()) -> int:
    banned = {int(x) for x in exclude}
    palette = [int(c) for c in NON_MAROON_SHELVES if int(c) not in banned]
    if not palette:
        palette = [int(c) for c in NON_MAROON_SHELVES]
    return int(rng.choice(np.array(palette, dtype=int)))


def _place_shelf(grid: np.ndarray, row: int, span: tuple[int, int], color: int) -> tuple[int, int, int]:
    c0, c1 = map(int, span)
    grid[int(row), c0 : c1 + 1] = int(color)
    return int(row), c0, c1


def _build_single_source(rng: np.random.Generator, want_maroon: bool) -> np.ndarray | None:
    h = int(rng.integers(18, 25))
    w = int(rng.integers(22, 29))
    grid = np.full((h, w), BLACK, dtype=int)

    src = int(rng.integers(5, w - 5))
    grid[0, src] = BLUE

    used_rows: list[int] = []
    top_row = _sample_row(rng, 4, min(7, h - 8), used_rows)
    if top_row is None:
        return None
    top_color = _choose_non_maroon(rng)
    top_span = _sample_covering_span(rng, src, w)
    if top_span is None:
        return None
    used_rows.append(int(top_row))
    _, top_c0, top_c1 = _place_shelf(grid, top_row, top_span, top_color)
    left_col = top_c0 - 1
    right_col = top_c1 + 1

    lower_row = _sample_row(rng, top_row + 4, h - 4, used_rows)
    if lower_row is None:
        return None

    if want_maroon:
        absorb_left = bool(rng.integers(0, 2) == 0)
        absorb_col = left_col if absorb_left else right_col
        free_col = right_col if absorb_left else left_col
        maroon_span = _sample_covering_span(
            rng,
            absorb_col,
            w,
            forbidden_in_span=(free_col,),
        )
        if maroon_span is None:
            return None
        used_rows.append(int(lower_row))
        _place_shelf(grid, lower_row, maroon_span, MAROON)

        if rng.random() < 0.6:
            extra_row = _sample_row(rng, lower_row + 4, h - 3, used_rows)
            if extra_row is None:
                return None
            extra_span = _sample_covering_span(
                rng,
                free_col,
                w,
                forbidden_in_span=(absorb_col,),
                forbidden_children=(absorb_col,),
            )
            if extra_span is None:
                return None
            extra_color = _choose_non_maroon(rng, exclude=(top_color,))
            used_rows.append(int(extra_row))
            _place_shelf(grid, extra_row, extra_span, extra_color)
    else:
        second_span = _sample_covering_span(
            rng,
            right_col if bool(rng.integers(0, 2)) else left_col,
            w,
            forbidden_in_span=(left_col, right_col),
            forbidden_children=(left_col, right_col),
        )
        second_hit = right_col if second_span is not None and right_col >= second_span[0] and right_col <= second_span[1] else left_col
        if second_span is None:
            second_hit = left_col
            second_span = _sample_covering_span(
                rng,
                second_hit,
                w,
                forbidden_in_span=(right_col,),
                forbidden_children=(right_col,),
            )
        if second_span is None:
            return None
        used_rows.append(int(lower_row))
        second_color = _choose_non_maroon(rng, exclude=(top_color,))
        _place_shelf(grid, lower_row, second_span, second_color)

    return grid


def _build_two_source_merge(rng: np.random.Generator, want_maroon: bool) -> np.ndarray | None:
    h = int(rng.integers(18, 25))
    w = int(rng.integers(22, 29))
    grid = np.full((h, w), BLACK, dtype=int)

    s1 = int(rng.integers(4, max(5, w // 2 - 3)))
    s2_low = max(s1 + 8, w // 2 + 2)
    if s2_low >= w - 4:
        return None
    s2 = int(rng.integers(s2_low, w - 4))
    grid[0, s1] = BLUE
    grid[0, s2] = BLUE

    shared_options = [m for m in range(s1 + 2, s2 - 1) if 2 < m < w - 3]
    if not shared_options:
        return None
    shared = int(rng.choice(np.array(shared_options, dtype=int)))

    color = _choose_non_maroon(rng)
    left_row = int(rng.integers(4, min(7, h - 9) + 1))
    right_row = int(rng.integers(left_row + 3, min(left_row + 7, h - 5) + 1))

    left_span = _sample_span_with_child(rng, s1, shared, w, "right")
    right_span = _sample_span_with_child(rng, s2, shared, w, "left")
    if left_span is None or right_span is None:
        return None

    _, lc0, lc1 = _place_shelf(grid, left_row, left_span, color)
    _, rc0, rc1 = _place_shelf(grid, right_row, right_span, color)
    outer_left = lc0 - 1
    outer_right = rc1 + 1

    lower_row = _sample_row(rng, right_row + 4, h - 4, (left_row, right_row))
    if lower_row is None:
        return None

    if want_maroon:
        absorb_left = bool(rng.integers(0, 2) == 0)
        absorb_col = outer_left if absorb_left else outer_right
        free_col = outer_right if absorb_left else outer_left
        maroon_span = _sample_covering_span(
            rng,
            absorb_col,
            w,
            forbidden_in_span=(shared, free_col),
            forbidden_children=(shared,),
        )
        if maroon_span is None:
            return None
        _place_shelf(grid, lower_row, maroon_span, MAROON)
    elif rng.random() < 0.7:
        extra_col = outer_left if bool(rng.integers(0, 2)) else outer_right
        other_outer = outer_right if extra_col == outer_left else outer_left
        extra_span = _sample_covering_span(
            rng,
            extra_col,
            w,
            forbidden_in_span=(shared, other_outer),
            forbidden_children=(shared, other_outer),
        )
        if extra_span is None:
            return None
        extra_color = _choose_non_maroon(rng, exclude=(color,))
        _place_shelf(grid, lower_row, extra_span, extra_color)

    return grid


def _iter_shelves(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    shelves: list[tuple[int, int, int, int]] = []
    for row in range(1, h):
        col = 0
        while col < w:
            value = int(g[row, col])
            if value in SHELF_COLORS:
                c0 = col
                while col + 1 < w and int(g[row, col + 1]) == value:
                    col += 1
                shelves.append((row, c0, col, value))
            col += 1
    return shelves


def _scene_valid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False
    h, w = map(int, g.shape)
    if h < 12 or w < 12:
        return False
    if np.any(g[1:] == BLUE):
        return False

    allowed = np.isin(g, np.array((BLACK, BLUE) + SHELF_COLORS, dtype=int))
    if not bool(np.all(allowed)):
        return False

    sources = _source_columns(g)
    if len(sources) not in (1, 2):
        return False
    if len(sources) == 2 and sources[1] - sources[0] < 7:
        return False

    shelves = _iter_shelves(g)
    if not shelves:
        return False
    shelf_mask = np.isin(g, np.array(SHELF_COLORS, dtype=int))
    for row, c0, c1, _color in shelves:
        if row <= 0 or c0 < 2 or c1 > w - 3:
            return False
        if not (3 <= c1 - c0 + 1 <= 8):
            return False
        rr0 = max(1, row - 1)
        rr1 = min(h - 1, row + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w - 1, c1 + 1)
        neighborhood = shelf_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        neighborhood[row - rr0, c0 - cc0 : c1 - cc0 + 1] = False
        if bool(np.any(neighborhood)):
            return False

    non_bg_mid = (g[1:] != BLACK) & ~np.isin(g[1:], np.array(SHELF_COLORS, dtype=int))
    if bool(np.any(non_bg_mid)):
        return False
    return True


def _scenario_matches(grid: np.ndarray, stats: dict[str, int], scenario: int) -> bool:
    out = main(grid)
    painted = int(np.sum((grid == BLACK) & (out != BLACK)))
    if painted < 10 or stats["splits"] < 1 or stats["bottoms"] < 1:
        return False

    if scenario == 0:
        return stats["sources"] == 1 and stats["absorbed"] >= 1 and stats["merges"] == 0
    if scenario == 1:
        return stats["sources"] == 1 and stats["absorbed"] == 0 and stats["merges"] == 0
    if scenario == 2:
        return stats["sources"] == 2 and stats["absorbed"] >= 1 and stats["merges"] >= 1
    return stats["sources"] == 2 and stats["absorbed"] == 0 and stats["merges"] >= 1


def _gen_attempt(rng: np.random.Generator, scenario: int) -> np.ndarray | None:
    if scenario == 0:
        grid = _build_single_source(rng, want_maroon=True)
    elif scenario == 1:
        grid = _build_single_source(rng, want_maroon=False)
    elif scenario == 2:
        grid = _build_two_source_merge(rng, want_maroon=True)
    else:
        grid = _build_two_source_merge(rng, want_maroon=False)

    if grid is None or not _scene_valid(grid):
        return None

    out, stats = _simulate(grid, collect_stats=True)
    if not np.any(out != grid):
        return None
    if not _scenario_matches(grid, stats, scenario):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    scenario = int(rng.integers(0, 4))
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng, scenario)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
