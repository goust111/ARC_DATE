from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
SOURCE_COLORS = (BLUE, RED, GREEN)

GEN_RETRY_LIMIT = 500
SOURCE_RETRY_LIMIT = 400
OBSTACLE_RETRY_LIMIT = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _iter_sources(grid: np.ndarray) -> list[tuple[str, int, int]]:
    h, w = grid.shape
    sources: list[tuple[str, int, int]] = []

    for c in range(1, w - 1):
        if int(grid[0, c]) in SOURCE_COLORS:
            sources.append(("T", 0, c))
        if int(grid[h - 1, c]) in SOURCE_COLORS:
            sources.append(("B", h - 1, c))

    for r in range(1, h - 1):
        if int(grid[r, 0]) in SOURCE_COLORS:
            sources.append(("L", r, 0))
        if int(grid[r, w - 1]) in SOURCE_COLORS:
            sources.append(("R", r, w - 1))

    return sources


def _ray_delta(side: str) -> tuple[int, int]:
    if side == "T":
        return 1, 0
    if side == "B":
        return -1, 0
    if side == "L":
        return 0, 1
    if side == "R":
        return 0, -1
    raise ValueError(f"unknown side {side}")


def _trace_counts(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    g = np.asarray(grid, dtype=int)
    counts = np.zeros_like(g, dtype=int)
    sources = _iter_sources(g)

    for side, r, c in sources:
        dr, dc = _ray_delta(side)
        rr, cc = r + dr, c + dc
        while _in_bounds(g.shape, rr, cc):
            if int(g[rr, cc]) == GREY:
                break
            counts[rr, cc] += 1
            rr += dr
            cc += dc

    return counts, sources


def _encode_counts(counts: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(counts, dtype=int)
    out[obstacle_mask] = GREY
    out[counts == 1] = BLUE
    out[counts == 2] = RED
    out[counts >= 3] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode each background cell by how many edge-fired orthogonal rays cross it before hitting gray obstacles.

    Input structure:
    - A black background grid contains colored source pixels on the outer border and gray 2x2 interior blocker squares.
    - Source pixels use blue/red/green only to mark where rays start; source color itself has no output meaning.
    - Gray cells are solid obstacles that stop rays before the obstacle cell, and all other cells are black.

    Rule:
    - Fire one orthogonal ray from every border source, using the inward normal of its edge.
    - Advance cell by cell; the source pixel is excluded, and the ray stops immediately before the first gray cell it would enter.
    - For each traversed cell, count how many rays pass through it, then encode k=1 as blue, k=2 as red, and k>=3 as green; untouched cells stay black.
    - Gray obstacle cells remain gray, and all source pixels are erased to black in the output.

    Key insight:
    - The result is determined jointly by obstacle-truncated ray geometry and by aggregating all ray visits at each cell, not by any individual source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must stay within 30x30"

    counts, sources = _trace_counts(g)
    assert 3 <= len(sources), "expected at least three border sources"

    source_mask = np.zeros_like(g, dtype=bool)
    for side, r, c in sources:
        source_mask[r, c] = True
        assert int(g[r, c]) in SOURCE_COLORS, "unexpected source color"
        if side in {"T", "B"}:
            assert 0 < c < w - 1, "corner sources are not supported"
        else:
            assert 0 < r < h - 1, "corner sources are not supported"

    obstacle_mask = g == GREY
    assert not np.any(obstacle_mask & source_mask), "a source cannot also be an obstacle"
    assert np.all((g == BG) | obstacle_mask | source_mask), "unexpected non-rule colors in input"

    return _encode_counts(counts, obstacle_mask)


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 - 1 or br1 + 1 < ar0 - 1 or ac1 + 1 < bc0 - 1 or bc1 + 1 < ac0 - 1


def _place_obstacles(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    if h < 10 or w < 10:
        return None

    row_lo, row_hi = 3, h - 6
    col_lo, col_hi = 3, w - 6
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(OBSTACLE_RETRY_LIMIT):
        count = random.randint(2, 4)
        rects: list[tuple[int, int, int, int]] = []
        tries = 0
        while len(rects) < count and tries < 80:
            tries += 1
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            rect = (r0, c0, r0 + 1, c0 + 1)
            if all(_rect_separated(rect, other) for other in rects):
                rects.append(rect)
        rows = {r for r0, _c0, r1, _c1 in rects for r in range(r0, r1 + 1)}
        cols = {c for _r0, c0, _r1, c1 in rects for c in range(c0, c1 + 1)}
        if len(rects) == count and len(rows) >= 2 and len(cols) >= 2:
            return rects
    return None


def _rows_with_obstacles(rects: Iterable[tuple[int, int, int, int]]) -> list[int]:
    return sorted({r for r0, _c0, r1, _c1 in rects for r in range(r0, r1 + 1)})


def _cols_with_obstacles(rects: Iterable[tuple[int, int, int, int]]) -> list[int]:
    return sorted({c for _r0, c0, _r1, c1 in rects for c in range(c0, c1 + 1)})


def _stop_column_from_left(row: int, rects: Iterable[tuple[int, int, int, int]]) -> tuple[int, int]:
    hits = [(c0, idx) for idx, (r0, c0, r1, _c1) in enumerate(rects) if r0 <= row <= r1]
    stop, idx = min(hits)
    return stop, idx


def _stop_column_from_right(row: int, rects: Iterable[tuple[int, int, int, int]]) -> tuple[int, int]:
    hits = [(c1, idx) for idx, (r0, _c0, r1, c1) in enumerate(rects) if r0 <= row <= r1]
    stop, idx = max(hits)
    return stop, idx


def _stop_row_from_top(col: int, rects: Iterable[tuple[int, int, int, int]]) -> tuple[int, int]:
    hits = [(r0, idx) for idx, (r0, c0, _r1, c1) in enumerate(rects) if c0 <= col <= c1]
    stop, idx = min(hits)
    return stop, idx


def _stop_row_from_bottom(col: int, rects: Iterable[tuple[int, int, int, int]]) -> tuple[int, int]:
    hits = [(r1, idx) for idx, (_r0, c0, r1, c1) in enumerate(rects) if c0 <= col <= c1]
    stop, idx = max(hits)
    return stop, idx


def _source_path_length(h: int, w: int, side: str, axis: int, rects: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    if side == "L":
        stop, idx = _stop_column_from_left(axis, rects)
        return stop - 1, idx
    if side == "R":
        stop, idx = _stop_column_from_right(axis, rects)
        return (w - 2) - (stop + 1) + 1, idx
    if side == "T":
        stop, idx = _stop_row_from_top(axis, rects)
        return stop - 1, idx
    stop, idx = _stop_row_from_bottom(axis, rects)
    return (h - 2) - (stop + 1) + 1, idx


def _source_color(side: str, axis: int) -> int:
    return (axis + {"T": 0, "R": 1, "B": 2, "L": 3}[side]) % 3 + 1


def _render_input(
    h: int,
    w: int,
    rects: list[tuple[int, int, int, int]],
    sources: list[tuple[str, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for side, axis in sources:
        color = _source_color(side, axis)
        if side == "T":
            grid[0, axis] = color
        elif side == "B":
            grid[h - 1, axis] = color
        elif side == "L":
            grid[axis, 0] = color
        else:
            grid[axis, w - 1] = color
    return grid


def _choose_sources(
    h: int,
    w: int,
    rects: list[tuple[int, int, int, int]],
) -> list[tuple[str, int]] | None:
    rows = _rows_with_obstacles(rects)
    cols = _cols_with_obstacles(rects)

    for _ in range(SOURCE_RETRY_LIMIT):
        total = random.choice((3, 4, 4, 5, 5, 6))
        feasible_splits = [
            (h_count, total - h_count)
            for h_count in range(1, min(len(rows), total - 1) + 1)
            if 1 <= total - h_count <= len(cols)
        ]
        if not feasible_splits:
            continue

        h_count, v_count = random.choice(feasible_splits)
        chosen_rows = sorted(random.sample(rows, h_count))
        chosen_cols = sorted(random.sample(cols, v_count))

        sources: list[tuple[str, int]] = []
        used_obstacles: set[int] = set()

        good = True
        for row in chosen_rows:
            side = random.choice(("L", "R"))
            path_len, obs_idx = _source_path_length(h, w, side, row, rects)
            if path_len < 2:
                good = False
                break
            used_obstacles.add(obs_idx)
            sources.append((side, row))
        if not good:
            continue

        for col in chosen_cols:
            side = random.choice(("T", "B"))
            path_len, obs_idx = _source_path_length(h, w, side, col, rects)
            if path_len < 2:
                good = False
                break
            used_obstacles.add(obs_idx)
            sources.append((side, col))
        if not good or len(used_obstacles) < 2:
            continue

        grid = _render_input(h, w, rects, sources)
        out = main(grid)
        counts, detected_sources = _trace_counts(grid)
        if len(detected_sources) != len(sources):
            continue

        positive_counts = counts[counts > 0]
        if positive_counts.size == 0:
            continue

        distinct_k = set(map(int, np.unique(positive_counts)))
        if not ({1, 2} <= distinct_k):
            continue

        covered = int(np.count_nonzero(positive_counts))
        if covered < 10 or covered > (h * w) // 2:
            continue

        if np.count_nonzero(counts == 2) < 2:
            continue

        if np.any(out[grid == GREY] != GREY):
            continue
        if np.any(out[0, :] == GREY) or np.any(out[-1, :] == GREY) or np.any(out[:, 0] == GREY) or np.any(out[:, -1] == GREY):
            continue

        # Under the rule-consistent generator distribution, overlap depth tops out at 2;
        # main still supports k>=3 for completeness.
        return sources

    return None


def _validate_instance(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    sources: list[tuple[str, int]],
) -> bool:
    h, w = grid.shape
    if not (13 <= h <= 18 and 13 <= w <= 18):
        return False

    if not (2 <= len(rects) <= 4 and 3 <= len(sources) <= 6):
        return False

    for i, rect in enumerate(rects):
        r0, c0, r1, c1 = rect
        if r1 - r0 != 1 or c1 - c0 != 1:
            return False
        if grid[r0 : r1 + 1, c0 : c1 + 1].shape != (2, 2):
            return False
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            return False
        for other in rects[i + 1 :]:
            if not _rect_separated(rect, other):
                return False

    h_rows = [axis for side, axis in sources if side in {"L", "R"}]
    v_cols = [axis for side, axis in sources if side in {"T", "B"}]
    if len(h_rows) != len(set(h_rows)) or len(v_cols) != len(set(v_cols)):
        return False

    if not h_rows or not v_cols:
        return False

    counts, detected_sources = _trace_counts(grid)
    if len(detected_sources) != len(sources):
        return False

    positive_counts = counts[counts > 0]
    if not ({1, 2} <= set(map(int, np.unique(positive_counts)))):
        return False

    for side, axis in sources:
        path_len, _ = _source_path_length(h, w, side, axis, rects)
        if path_len < 2:
            return False

    oracle = _encode_counts(counts, grid == GREY)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, oracle):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        h = random.randint(13, 18)
        w = random.randint(13, 18)
        rects = _place_obstacles(h, w)
        if rects is None:
            continue
        sources = _choose_sources(h, w, rects)
        if sources is None:
            continue
        grid = _render_input(h, w, rects, sources)
        if _validate_instance(grid, rects, sources):
            return grid
    raise RuntimeError("failed to generate a valid orthogonal-ray-depth input")
