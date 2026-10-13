from __future__ import annotations

from collections import deque
import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_HEIGHT_RANGE = (11, 17)
GEN_WIDTH_RANGE = (11, 17)
GEN_SEED_COUNT_RANGE = (4, 6)
GEN_MAX_PROFILE_STEP_CHOICES = (2, 3)
GEN_PROFILE_SPAN_MIN = 4
GEN_ATTEMPTS = 800

_GEN_CALL_COUNT = 0
_GEN_SEEN_INPUT_SIGS: set[str] = set()
_GEN_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _grid_sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _terrain_starts(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 4
    assert np.all(g[0, :] == GREY), "row 0 must be the separator row"
    assert np.all(g[-1, :] == GREEN), "bottom row must be solid terrain"

    starts: list[int] = []
    for c in range(g.shape[1]):
        terrain_rows = np.flatnonzero(g[:, c] == GREEN)
        assert terrain_rows.size > 0, "every column must contain terrain"
        start = int(terrain_rows[0])
        assert np.all(g[start:, c] == GREEN), "terrain must be a contiguous suffix"
        if start > 1:
            upper = g[1:start, c]
            assert np.all((upper == BLACK) | (upper == BLUE)), "sky may only contain black or blue"
        starts.append(start)
    return np.array(starts, dtype=int)


def _extreme_columns(starts: np.ndarray) -> tuple[int, int]:
    deep_cols = np.flatnonzero(starts == int(starts.max()))
    assert deep_cols.size == 1, "deepest column must be unique"
    deep_col = int(deep_cols[0])
    shallow_col = int(np.flatnonzero(starts == int(starts.min()))[-1])
    assert deep_col != shallow_col, "deep and shallow columns must differ"
    return deep_col, shallow_col


def _competitive_distances(base: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    dist = np.full(base.shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < base.shape[0] and 0 <= cc < base.shape[1]:
                if base[rr, cc] == BLACK and dist[rr, cc] == -1:
                    dist[rr, cc] = dist[r, c] + 1
                    q.append((rr, cc))
    return dist


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    starts = _terrain_starts(g)
    deep_col, shallow_col = _extreme_columns(starts)

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    assert seeds, "expected blue seeds"

    deep_seeds = [(r, c) for r, c in seeds if c == deep_col]
    shallow_seeds = [(r, c) for r, c in seeds if c == shallow_col]
    inactive = [(r, c) for r, c in seeds if c not in (deep_col, shallow_col)]
    assert deep_seeds, "deep column must contain at least one seed"
    assert shallow_seeds, "shallow column must contain at least one seed"

    base = g.copy()
    for r, c in inactive:
        base[r, c] = BLACK

    deep_dist = _competitive_distances(base, deep_seeds)
    shallow_dist = _competitive_distances(base, shallow_seeds)

    out = base.copy()
    black_cells = np.argwhere(base == BLACK)
    for r0, c0 in black_cells:
        r = int(r0)
        c = int(c0)
        d_deep = int(deep_dist[r, c])
        d_shallow = int(shallow_dist[r, c])
        if d_deep < 0 and d_shallow < 0:
            continue
        if d_shallow < 0 or (d_deep >= 0 and d_deep <= d_shallow):
            out[r, c] = MAROON
        else:
            out[r, c] = YELLOW

    info = {
        "starts": starts,
        "deep_col": deep_col,
        "shallow_col": shallow_col,
        "deep_seeds": deep_seeds,
        "shallow_seeds": shallow_seeds,
        "inactive_seeds": inactive,
        "base": base,
        "deep_dist": deep_dist,
        "shallow_dist": shallow_dist,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """Use terrain-profile extreme columns to activate seeds, then spread competing floods through the sky.

    Input structure:
    - Row 0 is a solid grey separator.
    - In every lower column, black sky appears above a contiguous green terrain suffix, with the bottom row fully green.
    - Blue seed pixels are scattered only in the sky cells.

    Rule:
    - Find the unique deepest column by the latest first green cell, and the shallowest column by the earliest first green cell with rightmost tie-break.
    - Keep only the blue seeds in those two extreme columns; erase every other seed to black.
    - From the deepest-column seeds and the shallowest-column seeds, expand 4-neighbor floods through black cells; deep-side cells become maroon, shallow-side cells become yellow, and equal-distance conflicts go to maroon.
    - Preserve the separator, terrain, and all surviving blue seeds.

    Key insight:
    - The terrain profile has a double role: it selects which seed columns are active and simultaneously shapes how the two floods compete for the same black sky region.
    """
    out, _ = _solve_core(grid)
    return out


def _weighted_choice(values: list[int], weights: list[int]) -> int:
    return int(random.choices(values, weights=weights, k=1)[0])


def _sample_segment(start: int, end: int, steps: int, lo: int, hi: int, max_step: int) -> list[int] | None:
    if steps == 0:
        return [int(start)] if start == end else None

    vals = [int(start)]
    cur = int(start)
    end = int(end)
    for pos in range(1, steps):
        rem = steps - pos
        candidates: list[int] = []
        weights: list[int] = []
        for nxt in range(lo, hi + 1):
            delta = abs(nxt - cur)
            if delta > max_step:
                continue
            if abs(end - nxt) > max_step * rem:
                continue
            candidates.append(nxt)
            weights.append({0: 5, 1: 4, 2: 2}.get(delta, 1))
        if not candidates:
            return None
        cur = _weighted_choice(candidates, weights)
        vals.append(cur)

    if abs(end - cur) > max_step:
        return None
    vals.append(end)
    return vals


def _sample_profile(height: int, width: int, want_min_tie: bool) -> tuple[np.ndarray, int, int] | None:
    for _ in range(300):
        max_step = random.choice(GEN_MAX_PROFILE_STEP_CHOICES)
        min_start = random.randint(2, min(4, height - 2))
        max_start_lo = min_start + GEN_PROFILE_SPAN_MIN
        if max_start_lo > height - 1:
            continue
        max_start = random.randint(max_start_lo, height - 1)

        tie_len = random.randint(2, min(4, width - 2)) if want_min_tie else 1
        shallow_col = random.randint(tie_len - 1, width - 2)
        shallow_cols = list(range(shallow_col - tie_len + 1, shallow_col + 1))

        need_gap = (max_start - min_start + max_step - 1) // max_step
        deep_candidates = [
            c
            for c in range(width)
            if c not in shallow_cols and min(abs(c - s) for s in shallow_cols) >= need_gap
        ]
        if not deep_candidates:
            continue
        deep_col = random.choice(deep_candidates)

        anchors: dict[int, int] = {c: min_start for c in shallow_cols}
        anchors[deep_col] = max_start
        ok = True
        for edge in (0, width - 1):
            if edge in anchors:
                continue
            nearest = min(anchors, key=lambda c: abs(c - edge))
            dist = abs(nearest - edge)
            candidates = [
                v
                for v in range(min_start + 1, max_start)
                if abs(v - anchors[nearest]) <= max_step * dist
            ]
            if not candidates:
                ok = False
                break
            anchors[edge] = random.choice(candidates)
        if not ok:
            continue

        cols = sorted(anchors)
        profile: list[int] = []
        for left, right in zip(cols, cols[1:]):
            seg = _sample_segment(
                anchors[left],
                anchors[right],
                right - left,
                min_start + 1,
                max_start - 1,
                max_step,
            )
            if seg is None:
                ok = False
                break
            if profile:
                profile.extend(seg[1:])
            else:
                profile.extend(seg)
        if not ok:
            continue

        starts = np.array(profile, dtype=int)
        if starts.shape != (width,):
            continue
        if np.any(np.abs(np.diff(starts)) > max_step):
            continue
        if int(starts.max()) != max_start or np.count_nonzero(starts == max_start) != 1:
            continue
        if int(starts[deep_col]) != max_start:
            continue
        min_cols = np.flatnonzero(starts == min_start)
        if min_cols.size != tie_len or int(min_cols[-1]) != shallow_col:
            continue
        if deep_col == shallow_col:
            continue
        return starts, deep_col, shallow_col
    return None


def _render_grid(height: int, width: int, starts: np.ndarray) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = GREY
    for c, start in enumerate(starts):
        grid[int(start) :, c] = GREEN
    return grid


def _candidate_seed_rows(start: int, prefer_low: bool) -> list[int]:
    rows = list(range(1, start))
    if not rows:
        return []
    if prefer_low:
        rows.sort(key=lambda r: (-r, random.random()))
    else:
        rows.sort(key=lambda r: (r, random.random()))
    return rows


def _place_seeds(
    grid: np.ndarray,
    starts: np.ndarray,
    deep_col: int,
    shallow_col: int,
) -> np.ndarray | None:
    out = grid.copy()
    width = out.shape[1]
    seed_count = random.randint(*GEN_SEED_COUNT_RANGE)
    other_cols = [c for c in range(width) if c not in (deep_col, shallow_col)]
    if len(other_cols) < seed_count - 2:
        return None

    seed_cols = [deep_col, shallow_col] + random.sample(other_cols, seed_count - 2)
    deep_low = random.random() < 0.5
    shallow_low = not deep_low if random.random() < 0.7 else random.random() < 0.5

    chosen: dict[int, int] = {}
    for col in seed_cols:
        if col == deep_col:
            rows = _candidate_seed_rows(int(starts[col]), prefer_low=deep_low)
        elif col == shallow_col:
            rows = _candidate_seed_rows(int(starts[col]), prefer_low=shallow_low)
        else:
            rows = _candidate_seed_rows(int(starts[col]), prefer_low=random.random() < 0.5)
        if not rows:
            return None
        chosen[col] = int(rows[0])

    for col, row in chosen.items():
        out[row, col] = BLUE
    return out


def _validate_example(grid: np.ndarray, out: np.ndarray, want_tie: bool) -> bool:
    try:
        expected, info = _solve_core(grid)
    except AssertionError:
        return False
    if not np.array_equal(out, expected):
        return False

    seeds = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    if not (GEN_SEED_COUNT_RANGE[0] <= len(seeds) <= GEN_SEED_COUNT_RANGE[1]):
        return False
    cols = [c for _, c in seeds]
    if len(set(cols)) != len(cols):
        return False

    deep_col = int(info["deep_col"])
    shallow_col = int(info["shallow_col"])
    inactive = list(info["inactive_seeds"])
    if sum(c == deep_col for _, c in seeds) < 1 or sum(c == shallow_col for _, c in seeds) < 1:
        return False
    if len(inactive) < 2:
        return False

    if np.count_nonzero(out == MAROON) < 5 or np.count_nonzero(out == YELLOW) < 5:
        return False

    deep_dist = np.asarray(info["deep_dist"], dtype=int)
    shallow_dist = np.asarray(info["shallow_dist"], dtype=int)
    tie_mask = (deep_dist >= 0) & (deep_dist == shallow_dist) & (np.asarray(info["base"]) == BLACK)
    if want_tie and not np.any(tie_mask):
        return False
    if want_tie and not np.any(out[tie_mask] == MAROON):
        return False

    inactive_colors = {int(out[r, c]) for r, c in inactive}
    if BLUE in inactive_colors:
        return False
    if not ({MAROON, YELLOW} & inactive_colors):
        return False
    if len(inactive_colors) < 2:
        return False

    return True


def _gen_attempt(mode: int) -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randrange(*GEN_HEIGHT_RANGE)
    width = random.randrange(*GEN_WIDTH_RANGE)
    want_min_tie = mode in (0, 2)
    want_distance_tie = mode in (0, 1)

    prof = _sample_profile(height, width, want_min_tie=want_min_tie)
    if prof is None:
        return None
    starts, deep_col, shallow_col = prof

    grid = _render_grid(height, width, starts)
    grid = _place_seeds(grid, starts, deep_col, shallow_col)
    if grid is None:
        return None

    out, _ = _solve_core(grid)
    if not _validate_example(grid, out, want_tie=want_distance_tie):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _GEN_CALL_COUNT % 4
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt(mode)
        if sample is None:
            continue
        grid, out = sample
        in_sig = _grid_sig(grid)
        out_sig = _grid_sig(out)
        if in_sig in _GEN_SEEN_INPUT_SIGS or out_sig in _GEN_SEEN_OUTPUT_SIGS:
            continue
        _GEN_SEEN_INPUT_SIGS.add(in_sig)
        _GEN_SEEN_OUTPUT_SIGS.add(out_sig)
        _GEN_CALL_COUNT += 1
        return grid
    raise RuntimeError("failed to generate a valid terrain-gated flood input")
