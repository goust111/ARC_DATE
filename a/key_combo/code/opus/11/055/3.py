from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
OBSTACLE = 2
FILL = 5
SOURCE = 8

GEN_ATTEMPTS = 600


def _contiguous_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if len(cols) == 0:
        return runs
    start = prev = int(cols[0])
    for value in cols[1:]:
        col = int(value)
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))
    return runs


def _nearest_side(col: int, left: int, right: int, width: int) -> int:
    candidates: list[tuple[int, int, int]] = []
    if left - 1 >= 0:
        candidates.append((col - (left - 1), 0, left - 1))
    if right + 1 < width:
        candidates.append(((right + 1) - col, 1, right + 1))
    assert candidates
    return min(candidates, key=lambda item: (item[0], -item[1]))[2]


def _obstacle_runs_by_row(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    rows: dict[int, list[tuple[int, int]]] = {}
    for r in np.flatnonzero(np.any(grid == OBSTACLE, axis=1)):
        cols = np.flatnonzero(grid[int(r)] == OBSTACLE)
        rows[int(r)] = _contiguous_runs(cols)
    return rows


def _run_covering_col(runs: list[tuple[int, int]], col: int) -> tuple[int, int] | None:
    for left, right in runs:
        if left <= col <= right:
            return left, right
    return None


def _mark_owner(owner: np.ndarray | None, r: int, c: int, owner_id: int) -> bool:
    if owner is None:
        return True
    prev = int(owner[r, c])
    if prev not in (-1, owner_id):
        return False
    owner[r, c] = owner_id
    return True


def _trace_single_path(
    out: np.ndarray,
    obstacle_runs: dict[int, list[tuple[int, int]]],
    start_col: int,
    owner: np.ndarray | None = None,
    owner_id: int = 0,
) -> tuple[list[tuple[int, int, int, int]], bool]:
    h, w = out.shape
    hits: list[tuple[int, int, int, int]] = []
    r = 0
    col = int(start_col)
    if not _mark_owner(owner, r, col, owner_id):
        return hits, False
    out[r, col] = SOURCE

    guard = 0
    while r < h - 1:
        guard += 1
        if guard > h * (w + 1):
            return hits, False

        next_r = r + 1
        run = _run_covering_col(obstacle_runs.get(next_r, []), col)
        if run is None:
            r = next_r
            if not _mark_owner(owner, r, col, owner_id):
                return hits, False
            out[r, col] = SOURCE
            continue

        left, right = run
        side_col = _nearest_side(col, left, right, w)
        c0, c1 = sorted((col, side_col))
        for cc in range(c0, c1 + 1):
            if not _mark_owner(owner, r, cc, owner_id):
                return hits, False
            out[r, cc] = SOURCE
        hits.append((next_r, left, right, side_col))
        col = side_col

    return hits, True


def _trace_paths(grid: np.ndarray, *, check_conflicts: bool) -> tuple[np.ndarray, list[dict[str, object]], bool]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    owner = np.full(g.shape, -1, dtype=int) if check_conflicts else None
    obstacle_runs = _obstacle_runs_by_row(g)

    infos: list[dict[str, object]] = []
    for owner_id, src_col in enumerate(np.flatnonzero(g[0] == SOURCE)):
        hits, ok = _trace_single_path(out, obstacle_runs, int(src_col), owner, owner_id)
        if not ok:
            return out, infos, False
        infos.append(
            {
                "source_col": int(src_col),
                "hits": hits,
                "tie_hits": sum(1 for _, left, right, _ in hits if int(src_col) * 2 == left + right),
            }
        )

    out[g == OBSTACLE] = OBSTACLE
    return out, infos, True


def _enumerate_black_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    rects: list[tuple[int, int, int, int]] = []
    for r0 in range(h):
        valid = np.ones(w, dtype=bool)
        for r1 in range(r0, h):
            valid &= grid[r1] == BG
            if r1 - r0 + 1 < 2:
                continue
            c = 0
            while c < w:
                while c < w and not valid[c]:
                    c += 1
                start = c
                while c < w and valid[c]:
                    c += 1
                end = c - 1
                if end - start + 1 >= 2:
                    rects.append((r0, start, r1, end))
    return rects


def _select_largest_black_rectangle(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    best: tuple[int, int, int, int] | None = None
    best_key: tuple[int, int, int, int, int] | None = None
    for rect in _enumerate_black_rectangles(grid):
        r0, c0, r1, c1 = rect
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        key = (area, -r0, -c0, -r1, -c1)
        if best_key is None or key > best_key:
            best = rect
            best_key = key
    return best


def _fill_rectangle(grid: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = FILL
    return out


def _paint_straight_columns(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    source_cols = np.flatnonzero(out[0] == SOURCE)
    if len(source_cols) > 0:
        out[:, source_cols] = SOURCE
    out[grid == OBSTACLE] = OBSTACLE
    return out


def _black_component_signature(grid: np.ndarray, seed: tuple[int, int]) -> tuple[int, tuple[int, int, int, int]]:
    h, w = grid.shape
    q = deque([seed])
    seen = {seed}
    cells: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return len(cells), (min(rows), min(cols), max(rows), max(cols))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the top teal sources around red obstacles, then fill the largest black rectangle shaped by the resulting paths.

    Input structure:
    - The grid is mostly black, with teal source pixels on row 0 and 2-3 red horizontal obstacle segments below them.
    - Each source represents one falling path column; obstacle segments occupy contiguous cells on distinct rows.
    - Remaining cells are plain black background.

    Rule:
    - Copy the input and extend each teal source downward one row at a time.
    - When a falling teal path would enter a red segment, move horizontally on the row just above that segment to the nearest outside column; exact midpoint ties detour to the right, then continue dropping from that outside column.
    - After all teal paths are painted, find every all-black axis-aligned rectangle with height and width at least 2, choose the maximum-area one, and break equal-area ties by the smallest top row then the smallest left column.
    - Fill that chosen rectangle grey while leaving the red obstacles unchanged.

    Key insight:
    - The red segments matter only because they reroute the falling teal paths, and that rerouting reshapes which empty black rectangle becomes the largest fillable target.
    """
    g = np.asarray(grid, dtype=int)
    path_grid, _, ok = _trace_paths(g, check_conflicts=False)
    assert ok, "invalid path layout"
    rect = _select_largest_black_rectangle(path_grid)
    assert rect is not None, "expected at least one 2x2 black rectangle"
    return _fill_rectangle(path_grid, rect)


def _sample_source_cols(rng: random.Random, width: int, count: int) -> list[int] | None:
    choices = list(range(1, width - 1))
    for _ in range(80):
        cols = sorted(rng.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _sample_rows(rng: random.Random, height: int, count: int) -> list[int] | None:
    choices = list(range(3, height - 2))
    for _ in range(80):
        rows = sorted(rng.sample(choices, count))
        if all(b - a >= 3 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _segment_candidates(
    width: int,
    source_cols: list[int],
    current_cols: list[int],
    target_idx: int,
) -> list[dict[str, int | str]]:
    x = int(current_cols[target_idx])
    others = [int(col) for idx, col in enumerate(current_cols) if idx != target_idx]
    candidates: list[dict[str, int | str]] = []
    for length in range(3, 7):
        for left in range(1, width - length):
            right = left + length - 1
            if not (left <= x <= right):
                continue
            if any(left <= other <= right for other in others):
                continue
            if sum(left <= src <= right for src in source_cols) > 1:
                continue
            dest = _nearest_side(x, left, right, width)
            if dest in others:
                continue
            c0, c1 = sorted((x, dest))
            if any(c0 <= other <= c1 for other in others):
                continue
            mode = "tie" if x * 2 == left + right else ("left" if dest < x else "right")
            gap = min(abs(dest - other) for other in others) if others else width
            candidates.append(
                {
                    "left": left,
                    "right": right,
                    "dest": dest,
                    "mode": mode,
                    "margin": min(dest, width - 1 - dest),
                    "gap": gap,
                }
            )
    return candidates


def _choose_candidate(
    rng: random.Random,
    candidates: list[dict[str, int | str]],
    desired_mode: str | None,
    future_hits: int,
) -> dict[str, int | str] | None:
    if not candidates:
        return None
    pool = [cand for cand in candidates if desired_mode is not None and cand["mode"] == desired_mode]
    if not pool:
        pool = candidates

    def score(cand: dict[str, int | str]) -> tuple[int, int, int]:
        margin = int(cand["margin"])
        gap = int(cand["gap"])
        preferred = 1 if cand["mode"] != "tie" else 0
        if future_hits > 0:
            return (margin + gap, margin, preferred)
        return (gap + preferred, margin, rng.randrange(1_000_000))

    ranked = sorted(pool, key=score, reverse=True)
    top = ranked[: min(6, len(ranked))]
    return rng.choice(top)


def _plan_hit_order(rng: random.Random, source_count: int, obstacle_count: int) -> list[int]:
    primary = rng.randrange(source_count)
    others = [idx for idx in range(source_count) if idx != primary]
    secondary = rng.choice(others)
    if obstacle_count == 2:
        patterns = [
            [primary, secondary],
            [primary, primary],
            [secondary, primary],
        ]
        return list(rng.choice(patterns))

    tertiary_pool = [idx for idx in others if idx != secondary] or others
    tertiary = rng.choice(tertiary_pool)
    patterns = [
        [primary, primary, secondary],
        [primary, secondary, primary],
        [secondary, primary, primary],
        [primary, secondary, tertiary],
    ]
    pattern = list(rng.choice(patterns))
    if len(set(pattern)) == source_count and source_count == obstacle_count:
        pattern[-1] = primary
    return pattern


def _build_candidate_input(rng: random.Random) -> tuple[np.ndarray, dict[str, object]] | None:
    height = rng.randint(14, 19)
    width = rng.randint(14, 19)
    source_count = rng.randint(3, 5)
    obstacle_count = rng.choice([2, 3])

    source_cols = _sample_source_cols(rng, width, source_count)
    obstacle_rows = _sample_rows(rng, height, obstacle_count)
    if source_cols is None or obstacle_rows is None:
        return None

    hit_order = _plan_hit_order(rng, source_count, obstacle_count)
    current_cols = list(source_cols)
    segments: list[tuple[int, int, int]] = []
    need_tie = rng.random() < 0.45
    tie_index = rng.randrange(obstacle_count) if need_tie else -1

    for order_idx, row in enumerate(obstacle_rows):
        target_idx = hit_order[order_idx]
        future_hits = hit_order[order_idx + 1 :].count(target_idx)
        desired_mode = "tie" if order_idx == tie_index else rng.choice([None, "left", "right"])
        candidates = _segment_candidates(width, source_cols, current_cols, target_idx)
        chosen = _choose_candidate(rng, candidates, desired_mode, future_hits)
        if chosen is None:
            return None
        left = int(chosen["left"])
        right = int(chosen["right"])
        dest = int(chosen["dest"])
        segments.append((row, left, right))
        current_cols[target_idx] = dest

    grid = np.zeros((height, width), dtype=int)
    grid[0, source_cols] = SOURCE
    for row, left, right in segments:
        grid[row, left : right + 1] = OBSTACLE

    meta = {
        "source_cols": source_cols,
        "obstacle_rows": obstacle_rows,
        "segments": segments,
        "hit_order": hit_order,
        "need_tie": need_tie,
    }
    return grid, meta


def _rectangle_area(rect: tuple[int, int, int, int] | None) -> int:
    if rect is None:
        return 0
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _validate_candidate(grid: np.ndarray, meta: dict[str, object]) -> bool:
    path_grid, infos, ok = _trace_paths(grid, check_conflicts=True)
    if not ok:
        return False

    source_cols = list(meta["source_cols"])
    obstacle_rows = list(meta["obstacle_rows"])
    if not all(np.sum(grid[row] == OBSTACLE) >= 3 for row in obstacle_rows):
        return False
    if any(np.sum((grid[row] == OBSTACLE) & np.isin(np.arange(grid.shape[1]), source_cols)) > 1 for row in obstacle_rows):
        return False

    total_hits = sum(len(info["hits"]) for info in infos)
    if total_hits != len(obstacle_rows):
        return False
    if not any(len(info["hits"]) == 0 for info in infos):
        return False
    if not any(len(info["hits"]) >= 1 for info in infos):
        return False
    if max(len(info["hits"]) for info in infos) < 2 and Counter(meta["hit_order"]).most_common(1)[0][1] >= 2:
        return False

    tie_hits = sum(
        1
        for info in infos
        for _, left, right, side_col in info["hits"]
        if int(info["source_col"]) * 2 == left + right and side_col == right + 1
    )
    if bool(meta["need_tie"]) and tie_hits == 0:
        return False

    rect = _select_largest_black_rectangle(path_grid)
    if rect is None or _rectangle_area(rect) < 6:
        return False

    input_rect = _select_largest_black_rectangle(grid)
    if input_rect == rect:
        return False

    straight_rect = _select_largest_black_rectangle(_paint_straight_columns(grid))
    if straight_rect == rect:
        return False

    r0, c0, r1, c1 = rect
    if path_grid[r0, c0] != BG:
        return False
    comp_size, comp_bbox = _black_component_signature(path_grid, (r0, c0))
    if comp_size == _rectangle_area(rect) and comp_bbox == rect:
        return False

    output = main(grid)
    if np.array_equal(output, path_grid):
        return False
    if not np.all(output[grid == OBSTACLE] == OBSTACLE):
        return False
    if not np.all(output[r0 : r1 + 1, c0 : c1 + 1] == FILL):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _build_candidate_input(rng)
        if result is None:
            continue
        grid, meta = result
        if _validate_candidate(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
