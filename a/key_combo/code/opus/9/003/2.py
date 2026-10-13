from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present


BG = Color.BLACK
TOP_COLORS = {Color.BLUE, Color.ORANGE}
GUIDE = Color.GREY
MARK = Color.PINK
BAD = Color.RED
GOOD = Color.GREEN

GEN_RETRIES = 400


def _guide_row_candidates(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        row = grid[r]
        vals = set(map(int, np.unique(row)))
        if not vals <= {BG, GUIDE, MARK}:
            continue
        grey_count = int(np.count_nonzero(row == GUIDE))
        mark_count = int(np.count_nonzero(row == MARK))
        if grey_count < w // 2 or mark_count < 2:
            continue
        candidates.append(r)
    return candidates


def _find_guide_row(grid: np.ndarray) -> int:
    candidates = _guide_row_candidates(grid)
    assert candidates, "expected a guide row"
    scores = []
    for r in candidates:
        row = grid[r]
        grey_count = int(np.count_nonzero(row == GUIDE))
        mark_count = int(np.count_nonzero(row == MARK))
        scores.append((grey_count + mark_count, grey_count, -abs(r - grid.shape[0] // 2), r))
    scores.sort()
    best = scores[-1][3]
    assert sum(item[:3] == scores[-1][:3] for item in scores) == 1, "guide row must be unique"
    return best


def _decode_period(row: np.ndarray) -> tuple[int, int]:
    magenta_cols = sorted(int(c) for c in np.where(row == MARK)[0])
    assert len(magenta_cols) >= 2, "expected magenta period markers"
    if len(magenta_cols) == 2:
        step = magenta_cols[1] - magenta_cols[0]
        assert step > 0
        return step, magenta_cols[0] % step

    diffs = Counter(
        magenta_cols[i + 1] - magenta_cols[i]
        for i in range(len(magenta_cols) - 1)
        if magenta_cols[i + 1] > magenta_cols[i]
    )
    step, count = max(diffs.items(), key=lambda item: (item[1], item[0]))
    assert step > 0 and count >= 1
    starts = [magenta_cols[i] for i in range(len(magenta_cols) - 1) if magenta_cols[i + 1] - magenta_cols[i] == step]
    assert starts
    return step, min(starts) % step


def _project_column(grid: np.ndarray, guide_r: int, col: int, side: str, choose: str) -> int:
    if side == "top":
        rows = np.where(np.isin(grid[:guide_r, col], list(TOP_COLORS)))[0]
        if rows.size == 0:
            return BG
        src_r = int(rows[-1] if choose == "nearest" else rows[0])
    else:
        rows = np.where(np.isin(grid[guide_r + 1 :, col], list(TOP_COLORS)))[0]
        if rows.size == 0:
            return BG
        src_r = int(rows[0] if choose == "nearest" else rows[-1]) + guide_r + 1
    return int(grid[src_r, col])


def _solve(
    grid: np.ndarray,
    *,
    apply_periodic: bool = True,
    apply_gap: bool = True,
    choose: str = "nearest",
    apply_recolor: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    guide_r = _find_guide_row(g)
    assert 0 < guide_r < h - 1, "guide row must have room above and below"
    guide_row = g[guide_r]
    step, phase = _decode_period(guide_row)

    out = np.zeros_like(g)
    out[guide_r] = guide_row

    top_row = np.zeros(w, dtype=int)
    bottom_row = np.zeros(w, dtype=int)
    for c in range(w):
        top_row[c] = _project_column(g, guide_r, c, "top", choose)
        bottom_row[c] = _project_column(g, guide_r, c, "bottom", choose)

    if apply_periodic:
        keep = np.array([(c % step) == phase for c in range(w)], dtype=bool)
        top_row[~keep] = BG
        bottom_row[~keep] = BG

    if apply_gap:
        gaps = guide_row == BG
        top_row[gaps] = BG
        bottom_row[gaps] = BG

    if apply_recolor:
        for c in range(w):
            a = int(top_row[c])
            b = int(bottom_row[c])
            if a == BG or b == BG:
                continue
            if a == b:
                top_row[c] = GOOD
                bottom_row[c] = GOOD
            else:
                top_row[c] = BAD
                bottom_row[c] = BAD

    out[guide_r - 1] = top_row
    out[guide_r + 1] = bottom_row
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress scattered pixels toward a gapped guide bar, filter them by a period marker, then recolor columnwise matches.

    Input structure:
    - The grid has a black background with one horizontal grey guide row near the middle; the guide may contain black single-cell gaps.
    - The same row also contains magenta marker pixels whose spacing encodes a period.
    - Above and below the guide row, blue and orange single pixels are scattered in various columns.

    Rule:
    - Find the grey guide row and decode the period step from the magenta markers on that row.
    - For each column, compress the upper pixels downward to the row just above the guide and compress the lower pixels upward to the row just below it, keeping only the pixel nearest to the guide on each side.
    - Remove any compressed pixel whose column fails the period test or lands at a guide-row gap.
    - Compare the surviving upper and lower pixels column by column: recolor both to green when both sides are present with the same color, recolor both to red when both sides are present with different colors, and leave single-sided survivors unchanged.
    - Output only the guide row, the magenta markers already embedded in it, and the final rows immediately above and below the guide; all other cells become black.

    Key insight:
    - The magenta spacing does not mark target columns directly; it defines a residue class that filters the compressed projection before the upper/lower symmetry verdict is applied.
    """
    return _solve(grid)


def _choose_distinct(rng: random.Random, items: list[int], count: int) -> list[int]:
    assert count <= len(items)
    chosen = rng.sample(items, count)
    chosen.sort()
    return chosen


def _pick_row(rng: random.Random, rows: list[int], avoid: set[int]) -> int:
    pool = [r for r in rows if r not in avoid]
    assert pool
    return int(rng.choice(pool))


def _paint_side(
    grid: np.ndarray,
    guide_r: int,
    col: int,
    side: str,
    near_color: int,
    *,
    rng: random.Random,
    force_far: bool = False,
) -> int:
    if side == "top":
        available = list(range(0, guide_r))
        near_pool = [r for r in range(max(0, guide_r - 3), guide_r)]
        far_pool = [r for r in range(0, max(0, guide_r - 3))]
    else:
        available = list(range(guide_r + 1, grid.shape[0]))
        near_pool = [r for r in range(guide_r + 1, min(grid.shape[0], guide_r + 4))]
        far_pool = [r for r in range(min(grid.shape[0], guide_r + 4), grid.shape[0])]

    used: set[int] = set()
    near_row = _pick_row(rng, near_pool or available, used)
    grid[near_row, col] = near_color
    used.add(near_row)

    far_written = 0
    if force_far or (far_pool and rng.random() < 0.45):
        far_color = Color.ORANGE if near_color == Color.BLUE else Color.BLUE
        far_row = _pick_row(rng, far_pool or available, used)
        grid[far_row, col] = far_color
        far_written = 1
    return 1 + far_written


def _build_input(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(17, 24)
    w = rng.randint(20, 26)
    guide_r = rng.randint(5, h - 6)

    step = rng.randint(3, 5)
    left_a = rng.randint(1, 3)
    left_b = left_a + step
    phase = left_a % step
    right_candidates = list(range(left_b + 4, w - step - 1))
    if not right_candidates:
        return None
    off_phase = [c for c in right_candidates if c % step != phase]
    right_a = rng.choice((off_phase or right_candidates)[-4:])
    right_b = right_a + step

    all_cols = list(range(w))
    marker_cols = {left_a, left_b, right_a, right_b}
    usable = [c for c in all_cols if c not in marker_cols]
    aligned = [c for c in usable if c % step == phase]
    misaligned = [c for c in usable if c % step != phase]
    if len(aligned) < 5 or len(misaligned) < 4:
        return None

    aligned_gap = rng.choice(aligned)
    extra_gap_count = rng.randint(1, min(3, len(misaligned)))
    gap_cols = {aligned_gap, *rng.sample(misaligned, extra_gap_count)}
    aligned_live = [c for c in aligned if c not in gap_cols]
    misaligned_live = [c for c in misaligned if c not in gap_cols]
    if len(aligned_live) < 4 or len(misaligned_live) < 3:
        return None

    same_count = 1
    diff_count = 1
    top_only_count = 1
    bottom_only_count = 1
    for bucket in rng.sample(["same", "diff", "top", "bottom"], k=rng.randint(0, min(2, len(aligned_live) - 4))):
        if bucket == "same":
            same_count += 1
        elif bucket == "diff":
            diff_count += 1
        elif bucket == "top":
            top_only_count += 1
        else:
            bottom_only_count += 1
    needed = same_count + diff_count + top_only_count + bottom_only_count
    if needed > len(aligned_live):
        return None

    chosen = _choose_distinct(rng, aligned_live, needed)
    idx = 0
    same_cols = chosen[idx : idx + same_count]
    idx += same_count
    diff_cols = chosen[idx : idx + diff_count]
    idx += diff_count
    top_only_cols = chosen[idx : idx + top_only_count]
    idx += top_only_count
    bottom_only_cols = chosen[idx : idx + bottom_only_count]

    periodic_cols = _choose_distinct(rng, misaligned_live, rng.randint(2, min(4, len(misaligned_live))))
    gap_action_cols = [aligned_gap]

    grid = np.zeros((h, w), dtype=int)
    guide_row = np.full(w, GUIDE, dtype=int)
    for c in gap_cols:
        guide_row[c] = BG
    for c in marker_cols:
        guide_row[c] = MARK
    grid[guide_r] = guide_row

    top_pixels = 0
    bottom_pixels = 0

    nearest_probe_col = rng.choice(same_cols + diff_cols + top_only_cols + bottom_only_cols)
    nearest_probe_side = rng.choice(["top", "bottom"])

    for c in same_cols:
        color = rng.choice([Color.BLUE, Color.ORANGE])
        top_pixels += _paint_side(
            grid,
            guide_r,
            c,
            "top",
            color,
            rng=rng,
            force_far=(c == nearest_probe_col and nearest_probe_side == "top"),
        )
        bottom_pixels += _paint_side(
            grid,
            guide_r,
            c,
            "bottom",
            color,
            rng=rng,
            force_far=(c == nearest_probe_col and nearest_probe_side == "bottom"),
        )

    for c in diff_cols:
        top_color = rng.choice([Color.BLUE, Color.ORANGE])
        bottom_color = Color.ORANGE if top_color == Color.BLUE else Color.BLUE
        top_pixels += _paint_side(
            grid,
            guide_r,
            c,
            "top",
            top_color,
            rng=rng,
            force_far=(c == nearest_probe_col and nearest_probe_side == "top"),
        )
        bottom_pixels += _paint_side(
            grid,
            guide_r,
            c,
            "bottom",
            bottom_color,
            rng=rng,
            force_far=(c == nearest_probe_col and nearest_probe_side == "bottom"),
        )

    for c in top_only_cols:
        top_pixels += _paint_side(
            grid,
            guide_r,
            c,
            "top",
            rng.choice([Color.BLUE, Color.ORANGE]),
            rng=rng,
            force_far=(c == nearest_probe_col and nearest_probe_side == "top"),
        )

    for c in bottom_only_cols:
        bottom_pixels += _paint_side(
            grid,
            guide_r,
            c,
            "bottom",
            rng.choice([Color.BLUE, Color.ORANGE]),
            rng=rng,
            force_far=(c == nearest_probe_col and nearest_probe_side == "bottom"),
        )

    for c in periodic_cols:
        if rng.random() < 0.7:
            top_pixels += _paint_side(grid, guide_r, c, "top", rng.choice([Color.BLUE, Color.ORANGE]), rng=rng)
        if rng.random() < 0.7:
            bottom_pixels += _paint_side(grid, guide_r, c, "bottom", rng.choice([Color.BLUE, Color.ORANGE]), rng=rng)

    for c in gap_action_cols:
        if rng.random() < 0.5:
            top_pixels += _paint_side(grid, guide_r, c, "top", rng.choice([Color.BLUE, Color.ORANGE]), rng=rng)
        else:
            bottom_pixels += _paint_side(grid, guide_r, c, "bottom", rng.choice([Color.BLUE, Color.ORANGE]), rng=rng)

    if top_pixels < 4 or bottom_pixels < 4:
        return None
    return grid


def _semantic_check(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    guide_candidates = _guide_row_candidates(g)
    assert len(guide_candidates) == 1, "generator must create a unique guide row"
    guide_r = guide_candidates[0]
    assert 0 < guide_r < h - 1

    non_guide_colors = set(colors_present(g, exclude={BG, GUIDE, MARK}))
    assert non_guide_colors <= TOP_COLORS

    above_count = int(np.count_nonzero(np.isin(g[:guide_r], list(TOP_COLORS))))
    below_count = int(np.count_nonzero(np.isin(g[guide_r + 1 :], list(TOP_COLORS))))
    assert above_count >= 4 and below_count >= 4

    out = _solve(g)
    top = out[guide_r - 1]
    bottom = out[guide_r + 1]

    keep_rows = {guide_r - 1, guide_r, guide_r + 1}
    for r in range(h):
        if r in keep_rows:
            continue
        assert np.all(out[r] == BG)

    assert np.any(top == GOOD) and np.any(bottom == GOOD), "need same-color branch"
    assert np.any(top == BAD) and np.any(bottom == BAD), "need different-color branch"
    single_survivor = False
    for c in range(w):
        a = int(top[c])
        b = int(bottom[c])
        if (a in TOP_COLORS and b == BG) or (b in TOP_COLORS and a == BG):
            single_survivor = True
            break
    assert single_survivor, "need single-sided branch"

    assert not np.array_equal(out, _solve(g, apply_periodic=False)), "period filter must matter"
    assert not np.array_equal(out, _solve(g, apply_gap=False)), "gap filter must matter"
    assert not np.array_equal(out, _solve(g, choose="farthest")), "nearest-selection must matter"
    assert not np.array_equal(out, _solve(g, apply_recolor=False)), "symmetry recolor must matter"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _build_input(rng)
        if grid is None:
            continue
        try:
            _semantic_check(grid)
            main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid periodic-guide puzzle input")
