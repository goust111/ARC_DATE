from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MIN_SIZE = 12
MAX_SIZE = 30
MIN_COUNT = 3
MAX_COUNT = 5
MAX_GENERATION_ATTEMPTS = 500


def _unique_nearest(value: int, choices: list[int]) -> int:
    distances = [abs(int(value) - int(choice)) for choice in choices]
    best = min(distances)
    winners = [int(choice) for choice, dist in zip(choices, distances) if dist == best]
    assert len(winners) == 1, "nearest spine point must be unique"
    return winners[0]


def _paint_horizontal_ray(grid: np.ndarray, r: int, c: int) -> None:
    _, width = grid.shape
    for dc in (-1, 1):
        cc = c + dc
        while 0 <= cc < width:
            if grid[r, cc] != BLACK:
                break
            grid[r, cc] = GREEN
            cc += dc


def _paint_vertical_ray(grid: np.ndarray, r: int, c: int) -> None:
    height, _ = grid.shape
    for dr in (-1, 1):
        rr = r + dr
        while 0 <= rr < height:
            value = int(grid[rr, c])
            if value == BLACK:
                grid[rr, c] = GREEN
            elif value == GREEN:
                grid[rr, c] = YELLOW
            else:
                break
            rr += dr


def _extract_sources(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    reds = [tuple(map(int, coord)) for coord in np.argwhere(grid == RED)]
    blues = [tuple(map(int, coord)) for coord in np.argwhere(grid == BLUE)]
    assert reds and blues, "input must contain red and blue control pixels"
    return reds, blues


def _spine_from_rays(grid: np.ndarray) -> np.ndarray:
    work = np.asarray(grid, dtype=int).copy()
    reds, blues = _extract_sources(work)

    for r, c in sorted(reds, key=lambda rc: (rc[0], rc[1])):
        _paint_horizontal_ray(work, r, c)
    for r, c in sorted(blues, key=lambda rc: (rc[1], rc[0])):
        _paint_vertical_ray(work, r, c)

    return work == YELLOW


def _apply_connections(
    base: np.ndarray,
    reds: list[tuple[int, int]],
    blues: list[tuple[int, int]],
    spine: np.ndarray,
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    yellow_rows: dict[int, list[int]] = {}
    yellow_cols: dict[int, list[int]] = {}

    for r, c in np.argwhere(spine):
        rr = int(r)
        cc = int(c)
        yellow_rows.setdefault(rr, []).append(cc)
        yellow_cols.setdefault(cc, []).append(rr)

    for values in yellow_rows.values():
        values.sort()
    for values in yellow_cols.values():
        values.sort()

    for r, c in reds:
        candidates = yellow_rows.get(int(r), [])
        if not candidates:
            continue
        target_c = _unique_nearest(c, candidates)
        step = 1 if target_c > c else -1
        for cc in range(c + step, target_c, step):
            if out[r, cc] == BLACK:
                out[r, cc] = RED

    for r, c in blues:
        candidates = yellow_cols.get(int(c), [])
        if not candidates:
            continue
        target_r = _unique_nearest(r, candidates)
        step = 1 if target_r > r else -1
        for rr in range(r + step, target_r, step):
            if out[rr, c] == BLACK:
                out[rr, c] = BLUE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emerge a yellow spine from red horizontal rays and blue vertical rays, then orthogonally link the controls to that spine.

    Input structure:
    - A black background contains 3-5 red single pixels and 3-5 blue single pixels.
    - Red pixels have distinct row coordinates; blue pixels have distinct column coordinates.
    - No red pixel shares a row or column with any blue pixel, so ray crossings happen only on blank cells.

    Rule:
    - Extend green rays left and right from each red pixel until the boundary or a non-background cell.
    - Then extend green rays up and down from each blue pixel; when a vertical ray passes through green, recolor that cell yellow as a crossing, otherwise stop at non-background cells.
    - Delete every non-crossing green cell, keep only the yellow crossings as the emerged spine, and connect each original red or blue pixel straight to its uniquely nearest yellow spine point on the same row or column.

    Key insight:
    - The yellow structure is not given in the input: it is the set of ray intersections, and the final colored links depend on that emergent spine rather than on any pre-marked target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    reds, blues = _extract_sources(g)
    spine = _spine_from_rays(g)

    out = np.full(g.shape, BLACK, dtype=int)
    out[spine] = YELLOW
    for r, c in reds:
        out[r, c] = RED
    for r, c in blues:
        out[r, c] = BLUE

    return _apply_connections(out, reds, blues, spine)


def _chebyshev_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def _build_ref_axis(count: int) -> list[int]:
    refs = [random.randint(2, 3)]
    for _ in range(count - 1):
        refs.append(refs[-1] + random.randint(5, 6))
    return refs


def _axis_candidate_pool(refs: list[int]) -> list[dict[str, int | str]]:
    pool = [
        {"pos": refs[0] - 2, "target": refs[0], "relation": "outer", "direction": "high"},
        {"pos": refs[-1] + 2, "target": refs[-1], "relation": "outer", "direction": "low"},
    ]
    for left, right in zip(refs, refs[1:]):
        pool.append({"pos": left + 2, "target": left, "relation": "between", "direction": "low"})
        pool.append({"pos": right - 2, "target": right, "relation": "between", "direction": "high"})
    return pool


def _build_other_axis(refs: list[int], count: int) -> list[int]:
    pool = _axis_candidate_pool(refs)
    between = [item for item in pool if str(item["relation"]) == "between"]
    chosen = [pool[0], pool[1], random.choice(between)]

    extras = pool[2:]
    random.shuffle(extras)
    for item in extras:
        if len(chosen) >= count:
            break
        chosen.append(item)
    while len(chosen) < count:
        chosen.append(random.choice(pool))

    random.shuffle(chosen)
    return [int(item["pos"]) for item in chosen]


def _finalize_axis(refs: list[int], others: list[int]) -> tuple[int, list[int], list[int]]:
    span_max = max(refs + others)
    size_lo = max(MIN_SIZE, span_max + 1)
    size_hi = min(MAX_SIZE, span_max + 3)
    size = random.randint(size_lo, size_hi)
    offset = random.randint(0, size - span_max - 1)
    return (
        size,
        [int(value + offset) for value in refs],
        [int(value + offset) for value in others],
    )


def _expected_from_sources(
    shape: tuple[int, int],
    reds: list[tuple[int, int]],
    blues: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    red_rows = sorted({int(r) for r, _ in reds})
    blue_cols = sorted({int(c) for _, c in blues})

    for r in red_rows:
        for c in blue_cols:
            out[r, c] = YELLOW

    for r, c in reds:
        target_c = _unique_nearest(c, blue_cols)
        step = 1 if target_c > c else -1
        for cc in range(c + step, target_c, step):
            out[r, cc] = RED
    for r, c in blues:
        target_r = _unique_nearest(r, red_rows)
        step = 1 if target_r > r else -1
        for rr in range(r + step, target_r, step):
            assert out[rr, c] == BLACK, "connection interiors should not overlap"
            out[rr, c] = BLUE

    for r, c in reds:
        out[r, c] = RED
    for r, c in blues:
        out[r, c] = BLUE
    for r in red_rows:
        for c in blue_cols:
            out[r, c] = YELLOW

    return out


def _validate_generated_case(grid: np.ndarray) -> bool:
    reds, blues = _extract_sources(grid)
    if not (MIN_COUNT <= len(reds) <= MAX_COUNT and MIN_COUNT <= len(blues) <= MAX_COUNT):
        return False

    red_rows = [int(r) for r, _ in reds]
    blue_rows = [int(r) for r, _ in blues]
    red_cols = [int(c) for _, c in reds]
    blue_cols = [int(c) for _, c in blues]
    red_rows_sorted = sorted(red_rows)
    blue_cols_sorted = sorted(blue_cols)

    if len(set(red_rows)) != len(red_rows):
        return False
    if len(set(blue_cols)) != len(blue_cols):
        return False
    if set(red_rows) & set(blue_rows):
        return False
    if set(red_cols) & set(blue_cols):
        return False
    if any(_chebyshev_adjacent(a, b) for i, a in enumerate(reds + blues) for b in (reds + blues)[i + 1 :]):
        return False
    if len(reds) * len(blues) < 2:
        return False

    try:
        red_targets = [_unique_nearest(c, blue_cols_sorted) for _, c in reds]
        blue_targets = [_unique_nearest(r, red_rows_sorted) for r, _ in blues]
    except AssertionError:
        return False

    if min(abs(c - t) for (_, c), t in zip(reds, red_targets)) < 2:
        return False
    if min(abs(r - t) for (r, _), t in zip(blues, blue_targets)) < 2:
        return False
    if len(set(red_targets)) < 2 or len(set(blue_targets)) < 2:
        return False
    if len({("left" if t < c else "right") for (_, c), t in zip(reds, red_targets)}) < 2:
        return False
    if len({("up" if t < r else "down") for (r, _), t in zip(blues, blue_targets)}) < 2:
        return False
    if not any(blue_cols_sorted[0] < c < blue_cols_sorted[-1] for _, c in reds):
        return False
    if not any(red_rows_sorted[0] < r < red_rows_sorted[-1] for r, _ in blues):
        return False

    expected = _expected_from_sources(grid.shape, reds, blues)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_case() -> np.ndarray | None:
    red_count = random.randint(MIN_COUNT, MAX_COUNT)
    blue_count = random.randint(MIN_COUNT, MAX_COUNT)

    red_rows_local = _build_ref_axis(red_count)
    blue_cols_local = _build_ref_axis(blue_count)
    blue_rows_local = _build_other_axis(red_rows_local, blue_count)
    red_cols_local = _build_other_axis(blue_cols_local, red_count)

    height, red_rows, blue_rows = _finalize_axis(red_rows_local, blue_rows_local)
    width, blue_cols, red_cols = _finalize_axis(blue_cols_local, red_cols_local)

    red_cols = red_cols[:]
    blue_rows = blue_rows[:]
    random.shuffle(red_cols)
    random.shuffle(blue_rows)
    reds = list(zip(red_rows, red_cols))
    blues = list(zip(blue_rows, blue_cols))

    grid = np.full((height, width), BLACK, dtype=int)
    for r, c in reds:
        grid[r, c] = RED
    for r, c in blues:
        grid[r, c] = BLUE

    if not _validate_generated_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
