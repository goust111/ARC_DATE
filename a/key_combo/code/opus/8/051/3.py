from __future__ import annotations

import bisect
import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

BAR_MARGIN = 3
BAR_GAP_MIN = 5
COL_MARGIN = 2
GEN_ATTEMPTS = 400


def _bar_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]


def _control_positions(grid: np.ndarray, bar_rows: list[int]) -> list[tuple[int, int]]:
    bar_set = set(bar_rows)
    controls = []
    for r, c in np.argwhere(grid == RED):
        rr = int(r)
        cc = int(c)
        if rr not in bar_set:
            controls.append((rr, cc))
    controls.sort()
    return controls


def _adjacent_bars(bar_rows: list[int], row: int) -> tuple[int | None, int | None]:
    idx = bisect.bisect_left(bar_rows, int(row))
    above = None if idx == 0 else int(bar_rows[idx - 1])
    below = None if idx == len(bar_rows) else int(bar_rows[idx])
    return above, below


def _analyze_scene(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int]], list[list[tuple[int, int]]], dict[tuple[int, int], set[int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors.issubset({BLACK, RED, GREY}), "input may only contain black, red, and grey"

    bar_rows = _bar_rows(g)
    assert 2 <= len(bar_rows) <= 4, "expected 2-4 full grey bars"
    assert all(bar_rows[i + 1] - bar_rows[i] >= BAR_GAP_MIN for i in range(len(bar_rows) - 1))

    controls = _control_positions(g, bar_rows)
    assert 4 <= len(controls) <= 6, "expected 4-6 red control pixels"

    control_hits: list[list[tuple[int, int]]] = []
    hits: dict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, (r, c) in enumerate(controls):
        above, below = _adjacent_bars(bar_rows, r)
        local_hits: list[tuple[int, int]] = []
        if above is not None:
            local_hits.append((above, c))
            hits[(above, c)].add(idx)
        if below is not None:
            local_hits.append((below, c))
            hits[(below, c)].add(idx)
        assert local_hits, "every control must reach at least one bar"
        control_hits.append(local_hits)
    assert hits, "expected at least one hit point"
    return bar_rows, controls, control_hits, hits


def main(grid: np.ndarray) -> np.ndarray:
    """Cast vertical rays from red controls to grey bars, then size each contact square by how many controls share that exact hit.

    Input structure:
    - Black background with 2-4 full grey horizontal bars spanning the whole width.
    - Red isolated control pixels lie off the bars, above, below, or between them.
    - Controls may share a column, so different controls can hit the same bar cell.

    Rule:
    - For each red control, cast one vertical ray upward and one downward; each ray stops at the first grey bar in that direction and paints its traversed cells red.
    - At every hit point on a grey bar, count how many different controls hit that exact bar-row/column cell.
    - Draw a solid yellow square centered on the hit whose side length is 2c+1, processing hits from top to bottom and left to right, then restore that square's center cell to red.

    Key insight:
    - The yellow stamp size is not encoded by any single control: it depends on global convergence, namely how many distinct control rays meet at the same bar cell.
    """
    g = np.asarray(grid, dtype=int)
    bar_rows, controls, _control_hits, hits = _analyze_scene(g)

    out = g.copy()
    for r, c in controls:
        above, below = _adjacent_bars(bar_rows, r)
        for hit_r in (above, below):
            if hit_r is None:
                continue
            step = 1 if hit_r > r else -1
            out[r : hit_r + 1 : step, c] = RED

    for hit_r, hit_c in sorted(hits):
        radius = len(hits[(hit_r, hit_c)])
        r0 = max(0, hit_r - radius)
        r1 = min(out.shape[0] - 1, hit_r + radius)
        c0 = max(0, hit_c - radius)
        c1 = min(out.shape[1] - 1, hit_c + radius)
        out[r0 : r1 + 1, c0 : c1 + 1] = YELLOW
        out[hit_r, hit_c] = RED

    return out


def _strip_rows(height: int, bar_rows: list[int]) -> list[list[int]]:
    strips: list[list[int]] = []
    prev = None
    for nxt in bar_rows + [None]:
        lo = 1 if prev is None else prev + 2
        hi = height - 2 if nxt is None else nxt - 2
        rows = list(range(lo, hi + 1))
        assert rows, "every strip must allow at least one control row"
        strips.append(rows)
        prev = nxt
    return strips


def _sample_bar_rows(rng: random.Random, height: int, n_bars: int) -> list[int] | None:
    candidates = list(range(BAR_MARGIN, height - BAR_MARGIN))
    for _ in range(120):
        rows = sorted(rng.sample(candidates, n_bars))
        if all(rows[i + 1] - rows[i] >= BAR_GAP_MIN for i in range(n_bars - 1)):
            return rows
    return None


def _choose_column(
    rng: random.Random,
    width: int,
    used: set[int],
    *,
    min_col: int = COL_MARGIN,
    max_col: int | None = None,
    close_to: int | None = None,
    max_gap: int | None = None,
    far_from: set[int] | None = None,
    min_gap_from_far: int = 0,
) -> int | None:
    hi = width - COL_MARGIN - 1 if max_col is None else max_col
    cols = []
    for col in range(min_col, hi + 1):
        if col in used:
            continue
        if close_to is not None and max_gap is not None and abs(col - close_to) > max_gap:
            continue
        if far_from is not None and any(abs(col - other) < min_gap_from_far for other in far_from):
            continue
        cols.append(col)
    if not cols:
        return None
    return int(rng.choice(cols))


def _place_unique_control(
    rng: random.Random,
    strips: list[list[int]],
    width: int,
    used_cols: set[int],
    placements: list[tuple[int, int]],
    *,
    preferred_strips: list[int] | None = None,
    far_from: set[int] | None = None,
    min_gap_from_far: int = 0,
) -> bool:
    strip_ids = list(range(len(strips))) if preferred_strips is None else list(preferred_strips)
    rng.shuffle(strip_ids)
    col = _choose_column(
        rng,
        width,
        used_cols,
        far_from=far_from,
        min_gap_from_far=min_gap_from_far,
    )
    if col is None:
        return False
    for strip_id in strip_ids:
        row = int(rng.choice(strips[strip_id]))
        placements.append((row, col))
        used_cols.add(col)
        return True
    return False


def _scene_has_overlay_dependency(hits: dict[tuple[int, int], set[int]]) -> bool:
    ordered = sorted(hits)
    for i, earlier in enumerate(ordered):
        r0, c0 = earlier
        for later in ordered[i + 1 :]:
            r1, c1 = later
            radius = len(hits[later])
            if abs(r1 - r0) <= radius and abs(c1 - c0) <= radius:
                return True
    return False


def _validate_generated_scene(grid: np.ndarray) -> bool:
    try:
        bar_rows, controls, control_hits, hits = _analyze_scene(grid)
    except AssertionError:
        return False

    counts = {pt: len(srcs) for pt, srcs in hits.items()}
    if 1 not in counts.values() or 2 not in counts.values():
        return False
    if max(counts.values()) > 2:
        return False
    if not any(len(paths) == 1 for paths in control_hits):
        return False
    if not any(len(paths) == 2 for paths in control_hits):
        return False

    by_col: dict[int, list[int]] = defaultdict(list)
    for (r, c), count in counts.items():
        by_col[c].append(count)
        if r < 2 or r > grid.shape[0] - 3 or c < 2 or c > grid.shape[1] - 3:
            return False
    if not any(len(values) >= 2 and len(set(values)) >= 2 for values in by_col.values()):
        return False
    if not _scene_has_overlay_dependency(hits):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    yellow = int(np.count_nonzero(out == YELLOW))
    if yellow == 0 or yellow > int(grid.size * 0.45):
        return False
    if not np.any((grid == GREY) & (out == GREY)):
        return False
    if len(set(controls)) != len(controls):
        return False
    if bar_rows[0] < BAR_MARGIN or bar_rows[-1] > grid.shape[0] - BAR_MARGIN - 1:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_bars = rng.choices([2, 3, 4], weights=[3, 5, 2])[0]
    min_height = 2 * BAR_MARGIN + 1 + (n_bars - 1) * BAR_GAP_MIN
    height = rng.randint(min_height, min(24, min_height + 4))
    width = rng.randint(17, 24)

    bar_rows = _sample_bar_rows(rng, height, n_bars)
    if bar_rows is None:
        return None
    strips = _strip_rows(height, bar_rows)

    target_controls = rng.randint(4, 6)
    placements: list[tuple[int, int]] = []
    used_cols: set[int] = set()

    # Mandatory pattern:
    # - shared column on adjacent strips -> one bar gets count 2
    # - nearby single column to its left -> overlapping 3x3 and 5x5 stamps, so draw order matters
    target_bar = rng.randint(0, n_bars - 2)
    shared_col = _choose_column(rng, width, used_cols, min_col=4)
    if shared_col is None:
        return None
    left_col = shared_col - rng.choice([1, 2])
    if left_col < COL_MARGIN or left_col in used_cols:
        return None

    placements.append((int(rng.choice(strips[target_bar])), shared_col))
    placements.append((int(rng.choice(strips[target_bar + 1])), shared_col))
    used_cols.add(shared_col)

    left_strip = rng.choice([target_bar, target_bar + 1])
    placements.append((int(rng.choice(strips[left_strip])), left_col))
    used_cols.add(left_col)

    far_bar_choices = list(range(len(strips)))
    rng.shuffle(far_bar_choices)
    if not _place_unique_control(
        rng,
        strips,
        width,
        used_cols,
        placements,
        preferred_strips=far_bar_choices,
        far_from={shared_col, left_col},
        min_gap_from_far=3,
    ):
        return None

    while len(placements) < target_controls:
        remaining = target_controls - len(placements)
        made_pair = False
        if remaining >= 2 and rng.random() < 0.45:
            pair_bar = rng.randint(0, n_bars - 1)
            pair_col = _choose_column(
                rng,
                width,
                used_cols,
                far_from={shared_col, left_col},
                min_gap_from_far=3,
            )
            if pair_col is not None:
                placements.append((int(rng.choice(strips[pair_bar])), pair_col))
                placements.append((int(rng.choice(strips[pair_bar + 1])), pair_col))
                used_cols.add(pair_col)
                made_pair = True
        if made_pair:
            continue
        if not _place_unique_control(
            rng,
            strips,
            width,
            used_cols,
            placements,
            far_from={shared_col, left_col},
            min_gap_from_far=2,
        ):
            return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[bar_rows, :] = GREY
    for r, c in placements:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = RED

    if not _validate_generated_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
