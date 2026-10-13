from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PLATFORM = 5
START_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator ranges.
GEN_HEIGHT_RANGE = (14, 22)
GEN_WIDTH_RANGE = (14, 24)
GEN_START_COUNT_RANGE = (2, 5)
GEN_RETRIES = 500
GEN_MIN_ROW_GAP = 2
GEN_MAX_EXTRA_ROWS = 4


def _extract_starts(grid: np.ndarray) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    for c, value in enumerate(grid[-1]):
        color = int(value)
        if color not in (BACKGROUND, PLATFORM):
            starts.append((int(c), color))
    return starts


def _search_gap(
    grid: np.ndarray,
    occupied: np.ndarray,
    start_mask: np.ndarray,
    r: int,
    c: int,
    dc: int,
    start_col: int,
    *,
    dynamic: bool,
) -> int | None:
    """Return the distance to the nearest reachable opening in one direction."""
    _h, w = grid.shape
    cc = int(c)
    dist = 0
    while True:
        cc += int(dc)
        dist += 1
        if not (0 <= cc < w):
            return None
        if int(grid[r, cc]) == PLATFORM:
            return None
        if dynamic and bool(occupied[r, cc]):
            return None
        if bool(start_mask[r, cc]) and cc != int(start_col):
            return None

        blocked_above = int(grid[r - 1, cc]) == PLATFORM
        if dynamic and bool(occupied[r - 1, cc]):
            blocked_above = True
        if not blocked_above:
            return int(dist)


def _choose_direction(left_dist: int | None, right_dist: int | None) -> tuple[int, int] | None:
    if left_dist is None and right_dist is None:
        return None
    if right_dist is None or (left_dist is not None and left_dist <= right_dist):
        return -1, int(left_dist)
    return 1, int(right_dist)


def _simulate_paths(grid: np.ndarray, *, collect_stats: bool = False) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    starts = _extract_starts(g)
    assert starts, "expected at least one colored start on the bottom row"

    out = g.copy()
    occupied = np.zeros((h, w), dtype=bool)
    start_mask = np.zeros((h, w), dtype=bool)
    for c, _color in starts:
        start_mask[h - 1, c] = True

    stats: dict[str, object] = {
        "reached_all_top": True,
        "detours": 0,
        "tie_count": 0,
        "directions": set(),
        "blocking_shift_count": 0,
    }

    for start_col, color in starts:
        r = h - 1
        c = int(start_col)
        path_cells = {(r, c)}

        while r > 0:
            if int(g[r - 1, c]) != PLATFORM and not bool(occupied[r - 1, c]):
                r -= 1
                out[r, c] = int(color)
                path_cells.add((r, c))
                continue

            left_dist = _search_gap(g, occupied, start_mask, r, c, -1, start_col, dynamic=True)
            right_dist = _search_gap(g, occupied, start_mask, r, c, 1, start_col, dynamic=True)
            choice = _choose_direction(left_dist, right_dist)
            if choice is None:
                stats["reached_all_top"] = False
                break

            if collect_stats:
                static_left = _search_gap(g, occupied, start_mask, r, c, -1, start_col, dynamic=False)
                static_right = _search_gap(g, occupied, start_mask, r, c, 1, start_col, dynamic=False)
                if left_dist is not None and right_dist is not None and left_dist == right_dist:
                    stats["tie_count"] = int(stats["tie_count"]) + 1
                if (left_dist, right_dist) != (static_left, static_right):
                    stats["blocking_shift_count"] = int(stats["blocking_shift_count"]) + 1

            dc, dist = choice
            stats["detours"] = int(stats["detours"]) + 1
            cast_dirs = stats["directions"]
            assert isinstance(cast_dirs, set)
            cast_dirs.add(int(dc))

            for _ in range(dist):
                c += dc
                out[r, c] = int(color)
                path_cells.add((r, c))

        if r != 0:
            stats["reached_all_top"] = False
        for rr, cc in path_cells:
            occupied[rr, cc] = True

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each bottom start upward, detouring toward the nearer opening under grey platforms.

    Input structure:
    - The grid has a black background, several differently colored start pixels on the bottom row,
      and one-cell-thick horizontal grey platform bars higher up in the grid.
    - Platform rows may contain one or more openings, and earlier generated path pixels later act
      as new blockers.

    Rule:
    - Process the start pixels from left to right.
    - For each path, move upward one cell whenever the cell above is free.
    - If the cell above is blocked by a grey platform or an already drawn path, scan left and right
      for the nearest reachable column whose above cell is open, move horizontally to that opening,
      and then continue climbing; ties choose the left side.
    - Stop once the path reaches the top row, drawing every new path cell in the start color.

    Key insight:
    - The output depends on a local nearest-opening rule plus the hidden generation order: once an
      earlier path occupies an opening or corridor, later paths can be forced to take a different detour.
    """
    out, _stats = _simulate_paths(grid, collect_stats=False)
    return out


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(int(lo), int(hi) + 1))
    if count > len(values):
        return None
    for _ in range(500):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return [int(v) for v in picks]
    return None


def _paint_platform_row(grid: np.ndarray, r: int, left: int, right: int, gaps: list[tuple[int, int]]) -> None:
    for c in range(int(left), int(right) + 1):
        if any(g0 <= c <= g1 for g0, g1 in gaps):
            continue
        grid[int(r), c] = PLATFORM


def _random_platform_spec(width: int) -> tuple[int, int, list[tuple[int, int]]]:
    left = random.randint(0, max(0, width // 6))
    right = random.randint(max(left + 6, (3 * width) // 5), width - 1)
    candidates = list(range(left + 1, right))
    random.shuffle(candidates)

    centers: list[int] = []
    gap_count = random.choice([1, 2, 2, 3])
    for cand in candidates:
        if all(abs(cand - other) >= 3 for other in centers):
            centers.append(int(cand))
            if len(centers) >= gap_count:
                break
    centers.sort()
    if not centers:
        centers = [int((left + right) // 2)]

    gaps: list[tuple[int, int]] = []
    for center in centers:
        gap_width = random.choice([1, 1, 2])
        g0 = max(left + 1, center - gap_width // 2)
        g1 = min(right - 1, g0 + gap_width - 1)
        gaps.append((int(g0), int(g1)))
    return int(left), int(right), gaps


def _place_special_row(grid: np.ndarray, starts: list[int], mode: str, r: int) -> bool:
    width = grid.shape[1]

    if mode == "sequential":
        pairs = [(a, b) for a, b in zip(starts, starts[1:]) if 2 <= b - a <= 3 and b + 2 < width - 1]
        if not pairs:
            return False
        a, b = random.choice(pairs)
        shared_gap = a + 1
        alt_gap = b + 2
        left = max(0, a - 3)
        right = min(width - 1, alt_gap + 2)
        _paint_platform_row(grid, r, left, right, [(shared_gap, shared_gap), (alt_gap, alt_gap)])
        return True

    if mode == "tie":
        candidates = [c for c in starts if c - 2 >= 1 and c + 2 <= width - 2]
        if not candidates:
            return False
        center = random.choice(candidates)
        dist = random.choice([1, 2])
        if not (1 <= center - dist and center + dist <= width - 2):
            return False
        left = max(0, center - dist - 2)
        right = min(width - 1, center + dist + 2)
        _paint_platform_row(grid, r, left, right, [(center - dist, center - dist), (center + dist, center + dist)])
        return True

    left, right, gaps = _random_platform_spec(width)
    _paint_platform_row(grid, r, left, right, gaps)
    return True


def _extra_rows(height: int, special_row: int) -> list[int]:
    candidates = list(range(2, special_row - 1))
    random.shuffle(candidates)
    rows: list[int] = []
    target = random.randint(1, min(GEN_MAX_EXTRA_ROWS, max(1, len(candidates))))
    for r in candidates:
        if all(abs(r - other) >= GEN_MIN_ROW_GAP for other in rows):
            rows.append(int(r))
            if len(rows) >= target:
                break
    rows.sort()
    return rows


def _build_candidate(mode: str) -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    max_starts = min(GEN_START_COUNT_RANGE[1], len(START_COLORS), (width - 2) // 2)
    if max_starts < GEN_START_COUNT_RANGE[0]:
        return None

    start_count = random.randint(GEN_START_COUNT_RANGE[0], max_starts)
    min_gap = 2 if mode == "sequential" else random.choice([2, 3])
    starts = _sample_spaced_positions(1, width - 2, start_count, min_gap)
    if starts is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for c, color in zip(starts, random.sample(START_COLORS, start_count)):
        grid[height - 1, c] = int(color)

    special_row = random.randint(height - 6, height - 4)
    if not _place_special_row(grid, starts, mode, special_row):
        return None

    for row in _extra_rows(height, special_row):
        left, right, gaps = _random_platform_spec(width)
        _paint_platform_row(grid, row, left, right, gaps)

    return grid


def _accept_candidate(inp: np.ndarray, out: np.ndarray, stats: dict[str, object], mode: str) -> bool:
    if not bool(stats["reached_all_top"]):
        return False
    if int(stats["detours"]) < 1:
        return False

    directions = stats["directions"]
    assert isinstance(directions, set)

    if mode == "sequential" and int(stats["blocking_shift_count"]) < 1:
        return False
    if mode == "tie" and int(stats["tie_count"]) < 1:
        return False
    if mode == "balanced" and not ({-1, 1} <= directions):
        return False

    grey_ratio = float(np.count_nonzero(inp == PLATFORM)) / float(inp.size)
    filled_ratio = float(np.count_nonzero(out != BACKGROUND)) / float(out.size)
    if grey_ratio > 0.28 or filled_ratio > 0.62:
        return False
    return True


def generate_input() -> np.ndarray:
    modes = ("sequential", "tie", "balanced", "mixed")
    weights = (3, 3, 2, 2)
    for _ in range(GEN_RETRIES):
        mode = random.choices(modes, weights=weights, k=1)[0]
        candidate = _build_candidate(mode)
        if candidate is None:
            continue
        output, stats = _simulate_paths(candidate, collect_stats=True)
        if _accept_candidate(candidate, output, stats, mode):
            return candidate
    raise RuntimeError("failed to generate a valid climbing-platform input")
