import random
from typing import Any

import numpy as np


BG = 0
MOVABLE_COLORS = (1, 2, 4, 6)
BARRIER_COLORS = (3, 7)

MIN_HEIGHT = 13
MAX_HEIGHT = 18
MIN_WIDTH = 18
MAX_WIDTH = 24
MAX_GENERATION_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return True iff (r,c) is within a grid of shape (H,W)."""
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _make_spec(
    width: int,
    row: int,
    run_color: int,
    barrier_color: int,
    run_len: int,
    terminal_col: int,
) -> dict[str, int]:
    run_left = int(width - run_len)
    distance = int(run_left - terminal_col)
    return {
        "row": int(row),
        "run_color": int(run_color),
        "barrier_color": int(barrier_color),
        "run_len": int(run_len),
        "run_left": run_left,
        "barrier_col": int(terminal_col - 1),
        "terminal_col": int(terminal_col),
        "distance": distance,
    }


def _extract_specs(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    specs: list[dict[str, int]] = []

    for r in range(h):
        barrier_cols = np.flatnonzero(np.isin(g[r], BARRIER_COLORS))
        if int(g[r, w - 1]) not in MOVABLE_COLORS:
            continue
        run_color = int(g[r, w - 1])
        c0 = w - 1
        while c0 > 0 and int(g[r, c0 - 1]) == run_color:
            c0 -= 1
        run_len = w - c0

        if barrier_cols.size != 1:
            continue
        barrier_col = int(barrier_cols[0])
        barrier_color = int(g[r, barrier_col])
        if barrier_col >= c0:
            continue
        if np.any(g[r, barrier_col + 1 : c0] != BG):
            continue

        specs.append(
            {
                "row": int(r),
                "run_color": run_color,
                "barrier_color": barrier_color,
                "run_len": int(run_len),
                "run_left": int(c0),
                "barrier_col": barrier_col,
                "terminal_col": int(barrier_col + 1),
                "distance": int(c0 - barrier_col - 1),
            }
        )

    return specs


def _apply_specs(
    grid: np.ndarray,
    specs: list[dict[str, int]],
    *,
    with_stats: bool = False,
) -> Any:
    out = np.asarray(grid, dtype=int).copy()
    h, _w = out.shape
    painted = np.zeros_like(out, dtype=bool)
    stats = {
        "clipped_rows": set(),
        "overlap_rows": set(),
        "overlap_cells": 0,
        "barrier_blocks": 0,
        "erased_painted": 0,
    }

    for spec in sorted(specs, key=lambda item: item["row"]):
        r = int(spec["row"])
        left = int(spec["run_left"])
        right = left + int(spec["run_len"])
        stats["erased_painted"] += int(np.count_nonzero(painted[r, left:right]))
        painted[r, left:right] = False
        out[r, left:right] = BG

        step = -1 if int(spec["barrier_color"]) == 3 else 1
        col = int(spec["terminal_col"])
        distance = int(spec["distance"])
        color = int(spec["run_color"])

        for k in range(distance):
            rr = r + step * k
            if not in_bounds(out.shape, rr, col):
                stats["clipped_rows"].add(r)
                continue
            if painted[rr, col]:
                stats["overlap_rows"].add(r)
                stats["overlap_cells"] += 1
                continue
            if int(out[rr, col]) != BG:
                stats["barrier_blocks"] += 1
                continue
            out[rr, col] = color
            painted[rr, col] = True

    if with_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each right-edge segment against its row barrier, then turn the slide distance into a vertical bar.

    Input structure:
    - The grid is mostly black (0).
    - Some rows contain one fixed barrier pixel colored green (3) or orange (7).
    - Some barrier rows also contain one horizontal run touching the right border, colored blue (1), red (2), yellow (4), or magenta (6), with black cells between the barrier and the run.

    Rule:
    - Process rows independently from top to bottom.
    - For each right-edge run, slide it left until its leftmost cell would be immediately to the right of that row's barrier; let d be the leftward slide distance.
    - Erase the run from its original position.
    - At the run's terminal left column, draw a vertical bar of the same color and length d, starting on the run row.
    - A green barrier makes the bar grow upward; an orange barrier makes it grow downward; off-grid cells are clipped.
    - Barrier pixels stay unchanged, and if a later vertical bar reaches a cell already painted by an earlier one, the earlier color stays.

    Key insight:
    - The horizontal slide is only an intermediate measurement: its displacement is reused as the vertical bar height, while the barrier color simultaneously chooses the growth direction.
    """
    g = np.asarray(grid, dtype=int)
    return _apply_specs(g, _extract_specs(g))


def _sample_rows(height: int, count: int) -> list[int] | None:
    for _ in range(200):
        rows = sorted(random.sample(range(height), count))
        if all(rows[i + 1] - rows[i] >= 2 for i in range(count - 1)):
            return rows
    return None


def _seed_barrier_colors(count: int) -> list[int]:
    colors = [3, 7]
    while len(colors) < count:
        colors.append(random.choice(BARRIER_COLORS))
    random.shuffle(colors)
    return colors[:count]


def _plan_specs(
    height: int,
    width: int,
    rows: list[int],
    run_colors: list[int],
    run_lengths: list[int],
    want_overlap: bool,
    want_clip: bool,
) -> tuple[list[dict[str, int]], bool, bool] | None:
    count = len(rows)
    barrier_colors = _seed_barrier_colors(count)
    terminals: list[int | None] = [None] * count
    overlap_pair: tuple[int, int] | None = None

    if want_overlap:
        overlap_candidates: list[tuple[int, int, int]] = []
        for i in range(count):
            for j in range(i + 1, count):
                if rows[j] - rows[i] < 3:
                    continue
                for terminal_col in range(1, width - 5):
                    d1 = width - run_lengths[i] - terminal_col
                    d2 = width - run_lengths[j] - terminal_col
                    if d1 <= 0 or d2 <= 0:
                        continue
                    if rows[i] + d1 - 1 >= height:
                        continue
                    if rows[j] - d2 + 1 < 0:
                        continue
                    overlap_len = d1 + d2 - (rows[j] - rows[i]) - 1
                    if 1 <= overlap_len <= 2:
                        overlap_candidates.append((i, j, terminal_col))
        if overlap_candidates:
            i, j, terminal_col = random.choice(overlap_candidates)
            overlap_pair = (i, j)
            terminals[i] = terminal_col
            terminals[j] = terminal_col
            barrier_colors[i] = 7
            barrier_colors[j] = 3
        else:
            want_overlap = False

    if want_clip and count == 2 and want_overlap:
        want_clip = False

    if want_clip:
        clip_candidates: list[tuple[int, int, int]] = []
        for idx in range(count):
            if overlap_pair is not None and idx in overlap_pair:
                continue
            for terminal_col in range(1, width - 5):
                d = width - run_lengths[idx] - terminal_col
                if d <= 0:
                    continue
                if d > rows[idx] + 1:
                    clip_candidates.append((idx, 3, terminal_col))
                if d > height - rows[idx]:
                    clip_candidates.append((idx, 7, terminal_col))
        if clip_candidates:
            idx, barrier_color, terminal_col = random.choice(clip_candidates)
            terminals[idx] = terminal_col
            barrier_colors[idx] = barrier_color
        else:
            want_clip = False

    for idx in range(count):
        if terminals[idx] is not None:
            continue
        options = []
        for terminal_col in range(1, width - 5):
            if terminal_col >= width - run_lengths[idx]:
                continue
            ok = True
            for other_idx, other_terminal in enumerate(terminals):
                if other_terminal is None or other_idx == idx:
                    continue
                if overlap_pair is not None and {idx, other_idx} == set(overlap_pair):
                    continue
                if abs(terminal_col - other_terminal) < 2:
                    ok = False
                    break
            if ok:
                options.append(terminal_col)
        if not options:
            return None
        terminals[idx] = random.choice(options)

    specs = [
        _make_spec(width, rows[idx], run_colors[idx], barrier_colors[idx], run_lengths[idx], int(terminals[idx]))
        for idx in range(count)
    ]
    return specs, want_overlap, want_clip


def _sample_extra_barriers(height: int, width: int, specs: list[dict[str, int]]) -> list[tuple[int, int, int]]:
    movable_rows = {int(spec["row"]) for spec in specs}
    forbidden_cols = {int(spec["terminal_col"]) for spec in specs}
    free_rows = [r for r in range(height) if r not in movable_rows]
    random.shuffle(free_rows)

    max_extras = min(3, len(free_rows))
    if max_extras == 0:
        return []
    count = random.randint(0, max_extras)
    candidate_cols = [c for c in range(0, width - 5) if c not in forbidden_cols]
    if not candidate_cols:
        return []

    extras: list[tuple[int, int, int]] = []
    for row in free_rows:
        if len(extras) >= count or not candidate_cols:
            break
        col = random.choice(candidate_cols)
        candidate_cols.remove(col)
        extras.append((row, col, random.choice(BARRIER_COLORS)))
    return extras


def _render_input(height: int, width: int, specs: list[dict[str, int]], extras: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for row, col, color in extras:
        grid[int(row), int(col)] = int(color)
    for spec in specs:
        r = int(spec["row"])
        grid[r, int(spec["barrier_col"])] = int(spec["barrier_color"])
        left = int(spec["run_left"])
        grid[r, left : left + int(spec["run_len"])] = int(spec["run_color"])
    return grid


def _canonical_specs(specs: list[dict[str, int]]) -> list[tuple[int, ...]]:
    return sorted(
        (
            int(spec["row"]),
            int(spec["barrier_col"]),
            int(spec["barrier_color"]),
            int(spec["run_color"]),
            int(spec["run_left"]),
            int(spec["run_len"]),
            int(spec["terminal_col"]),
            int(spec["distance"]),
        )
        for spec in specs
    )


def _is_valid_example(
    grid: np.ndarray,
    specs: list[dict[str, int]],
    want_overlap: bool,
    want_clip: bool,
) -> bool:
    parsed = _extract_specs(grid)
    if _canonical_specs(parsed) != _canonical_specs(specs):
        return False

    distances = [int(spec["distance"]) for spec in specs]
    if any(distance <= 0 for distance in distances):
        return False
    if len(set(distances)) < 2:
        return False
    if {int(spec["barrier_color"]) for spec in specs} != {3, 7}:
        return False

    expected, stats = _apply_specs(grid, specs, with_stats=True)
    if stats["barrier_blocks"] != 0 or stats["erased_painted"] != 0:
        return False
    if want_overlap:
        if not (1 <= stats["overlap_cells"] <= 2):
            return False
    elif stats["overlap_cells"] != 0:
        return False
    if want_clip:
        if not stats["clipped_rows"]:
            return False
    elif stats["clipped_rows"]:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    # Check semantics directly from the generated spec list rather than only from the solver output.
    for spec in sorted(specs, key=lambda item: item["row"]):
        row = int(spec["row"])
        col = int(spec["terminal_col"])
        step = -1 if int(spec["barrier_color"]) == 3 else 1
        visible = 0
        for k in range(int(spec["distance"])):
            rr = row + step * k
            if not in_bounds(actual.shape, rr, col):
                continue
            if int(actual[rr, col]) == int(spec["run_color"]):
                visible += 1
        if visible == 0:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    max_segments = min(4, (height + 1) // 2)
    segment_count = random.randint(2, max_segments)

    rows = _sample_rows(height, segment_count)
    if rows is None:
        return None

    run_colors = random.sample(list(MOVABLE_COLORS), segment_count)
    run_lengths = [random.randint(2, 4) for _ in range(segment_count)]

    want_overlap = segment_count >= 2 and random.random() < 0.4
    want_clip = random.random() < 0.55

    plan = _plan_specs(height, width, rows, run_colors, run_lengths, want_overlap, want_clip)
    if plan is None:
        return None
    specs, want_overlap, want_clip = plan

    extras = _sample_extra_barriers(height, width, specs)
    grid = _render_input(height, width, specs, extras)

    if not _is_valid_example(grid, specs, want_overlap, want_clip):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-distance ARC input")
