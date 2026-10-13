from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != YELLOW or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == YELLOW:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in pts]
            cols = [cc for _, cc in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if len(pts) != (r1 - r0 + 1) * (c1 - c0 + 1):
                raise AssertionError("yellow components must be filled rectangles")
            rects.append((r0, c0, r1, c1))
    return sorted(rects)


def _find_runs(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    runs: list[dict[str, int]] = []
    for r in range(h):
        for color, edge in ((BLUE, "left"), (RED, "right")):
            cols = np.flatnonzero(grid[r] == color)
            if len(cols) == 0:
                continue
            start = int(cols[0])
            end = int(cols[-1])
            if len(cols) != end - start + 1:
                raise AssertionError("moving pixels must form a contiguous horizontal run")
            if edge == "left" and start != 0:
                raise AssertionError("blue runs must start at the left edge")
            if edge == "right" and end != w - 1:
                raise AssertionError("red runs must end at the right edge")
            runs.append(
                {
                    "row": r,
                    "color": color,
                    "start": start,
                    "end": end,
                    "length": len(cols),
                }
            )
    rows = [run["row"] for run in runs]
    if len(rows) != len(set(rows)):
        raise AssertionError("each moving row must contain exactly one run")
    return sorted(runs, key=lambda run: run["row"])


def _reveal_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    runs = _find_runs(g)
    run_by_row = {run["row"]: run for run in runs}

    for r0, c0, r1, c1 in _find_rectangles(g):
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        for r in range(r0, r1 + 1):
            run = run_by_row.get(r)
            if run is None:
                continue
            if run["color"] == RED and c1 < run["start"]:
                out[r, c1] = GREEN
            if run["color"] == BLUE and c0 > run["end"]:
                out[r, c0] = GREEN
    return out, runs


def _slide_runs(base: np.ndarray, runs: list[dict[str, int]]) -> np.ndarray:
    h, w = base.shape
    out = base.copy()
    for run in runs:
        r = run["row"]
        start = run["start"]
        end = run["end"]
        length = run["length"]
        out[r, start : end + 1] = BG
        obstacles = np.flatnonzero(base[r] == GREEN)
        if run["color"] == RED:
            blockers = obstacles[obstacles < start]
            target_start = int(blockers[-1]) + 1 if len(blockers) else 0
            target_end = target_start + length
            if target_end > w:
                raise AssertionError("red run does not fit after sliding")
            out[r, target_start:target_end] = RED
        else:
            blockers = obstacles[obstacles > end]
            target_end = (int(blockers[0]) - 1) if len(blockers) else (w - 1)
            target_start = target_end - length + 1
            if target_start < 0:
                raise AssertionError("blue run does not fit after sliding")
            out[r, target_start : target_end + 1] = BLUE
    return out


def _visible_only_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    g[g == YELLOW] = BG
    return _slide_runs(g, _find_runs(grid))


def _targets_after_transform(grid: np.ndarray, *, reveal_hidden: bool) -> dict[int, tuple[int, int]]:
    base = np.asarray(grid, dtype=int).copy()
    runs = _find_runs(base)
    if reveal_hidden:
        base, _ = _reveal_grid(base)
    else:
        base[base == YELLOW] = BG
    targets: dict[int, tuple[int, int]] = {}
    w = base.shape[1]
    for run in runs:
        r = run["row"]
        start = run["start"]
        end = run["end"]
        length = run["length"]
        obstacles = np.flatnonzero(base[r] == GREEN)
        if run["color"] == RED:
            blockers = obstacles[obstacles < start]
            target_start = int(blockers[-1]) + 1 if len(blockers) else 0
            targets[r] = (target_start, target_start + length - 1)
        else:
            blockers = obstacles[obstacles > end]
            target_end = int(blockers[0]) - 1 if len(blockers) else w - 1
            targets[r] = (target_end - length + 1, target_end)
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase directional mask rectangles, reveal the hidden obstacle stoppers they cover, and slide each edge run toward its nearest stopper.

    Input structure:
    - The grid has a black background with isolated green obstacle pixels, blue runs touching the left edge, and red runs touching the right edge.
    - One or two yellow 4-connected solid rectangles act as masks: they never cover run pixels, and they may hide obstacle stoppers on the rows used by moving runs.
    - Each moving row contains exactly one contiguous horizontal run, and different runs occupy different rows.

    Rule:
    - Detect every yellow solid rectangle and erase it to black.
    - In each rectangle row that contains a moving run, reveal the hidden obstacle stopper on the edge that the run moves toward: the rectangle's right edge for a red run and its left edge for a blue run.
    - Then slide every red run left until its left end is immediately to the right of the nearest green obstacle in that row, or to column 0 if none exists.
    - Slide every blue run right until its right end is immediately to the left of the nearest green obstacle in that row, or to the right border if none exists.
    - Clear each run's original cells, preserve run length and color, and keep all visible or revealed green obstacles fixed.

    Key insight:
    - The yellow rectangles must be removed before sliding, because some decisive row-wise stoppers are hidden under those masks rather than visible in the raw input.
    """
    revealed, runs = _reveal_grid(np.asarray(grid, dtype=int))
    return _slide_runs(revealed, runs)


def _is_isolated(pos: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = pos
    return (
        (r - 1, c) not in occupied
        and (r + 1, c) not in occupied
        and (r, c - 1) not in occupied
        and (r, c + 1) not in occupied
    )


def _build_row_obstacles(
    row: int,
    candidates: list[int],
    occupied: set[tuple[int, int]],
    count_lo: int,
    count_hi: int,
) -> list[int] | None:
    valid = [c for c in candidates if _is_isolated((row, c), occupied)]
    if len(valid) < count_lo:
        return None
    count = random.randint(count_lo, min(count_hi, len(valid)))
    chosen: list[int] = []
    pool = valid[:]
    random.shuffle(pool)
    for c in pool:
        if any(abs(c - prev) == 1 for prev in chosen):
            continue
        if not _is_isolated((row, c), occupied | {(row, cc) for cc in chosen}):
            continue
        chosen.append(c)
        if len(chosen) == count:
            break
    return chosen if len(chosen) >= count_lo else None


def _choose_hidden_rows(rows: list[int]) -> list[int] | None:
    options = [[r] for r in rows]
    for a, b in combinations(rows, 2):
        if 2 <= b - a <= 3:
            options.append([a, b])
    if not options:
        return None
    random.shuffle(options)
    return options[0]


def _rect_candidates(
    h: int,
    w: int,
    hidden_rows: list[int],
    moving_rows: set[int],
    run_endpoints: dict[int, tuple[int, int]],
    color: int,
    taken: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    row_min = min(hidden_rows)
    row_max = max(hidden_rows)
    out: list[tuple[int, int, int, int]] = []

    for rect_h in range(max(2, row_max - row_min + 1), 5):
        for r0 in range(max(0, row_max - rect_h + 1), min(row_min, h - rect_h) + 1):
            r1 = r0 + rect_h - 1
            band_rows = set(range(r0, r1 + 1))
            if any(row in moving_rows and row not in hidden_rows for row in band_rows):
                continue

            for rect_w in range(2, 5):
                if color == RED:
                    max_c1 = min(run_endpoints[row][0] - 1 for row in hidden_rows)
                    for c1 in range(rect_w - 1, max_c1 + 1):
                        c0 = c1 - rect_w + 1
                        rect = (r0, c0, r1, c1)
                        if any(_rects_touch_or_overlap(rect, other) for other in taken):
                            continue
                        out.append(rect)
                else:
                    min_c0 = max(run_endpoints[row][1] + 1 for row in hidden_rows)
                    for c0 in range(min_c0, w - rect_w + 1):
                        c1 = c0 + rect_w - 1
                        rect = (r0, c0, r1, c1)
                        if any(_rects_touch_or_overlap(rect, other) for other in taken):
                            continue
                        out.append(rect)
    random.shuffle(out)
    return out


def _rects_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(11, 15)
    w = random.randint(13, 18)

    red_count = random.randint(2, 4)
    blue_count = random.randint(2, 4)
    total = red_count + blue_count
    if total > h - 2:
        return None

    all_rows = random.sample(range(h), total)
    red_rows = sorted(all_rows[:red_count])
    blue_rows = sorted(all_rows[red_count:])
    if set(red_rows) & set(blue_rows):
        return None

    run_lengths: dict[int, int] = {}
    run_endpoints: dict[int, tuple[int, int]] = {}
    for row in red_rows:
        length = random.randint(2, 5)
        run_lengths[row] = length
        run_endpoints[row] = (w - length, w - 1)
    for row in blue_rows:
        length = random.randint(2, 5)
        run_lengths[row] = length
        run_endpoints[row] = (0, length - 1)

    n_rects = random.randint(1, 2)
    moving_rows = set(red_rows) | set(blue_rows)
    rects: list[tuple[int, int, int, int]] = []
    hidden_rows_by_rect: list[tuple[int, list[int], tuple[int, int, int, int]]] = []

    if n_rects == 1:
        color = random.choice([RED, BLUE])
        rows = red_rows if color == RED else blue_rows
        hidden_rows = _choose_hidden_rows(rows)
        if hidden_rows is None:
            return None
        cands = _rect_candidates(h, w, hidden_rows, moving_rows, run_endpoints, color, rects)
        if not cands:
            return None
        rect = cands[0]
        rects.append(rect)
        hidden_rows_by_rect.append((color, hidden_rows, rect))
    else:
        for color, rows in ((RED, red_rows), (BLUE, blue_rows)):
            hidden_rows = _choose_hidden_rows(rows)
            if hidden_rows is None:
                return None
            cands = _rect_candidates(h, w, hidden_rows, moving_rows, run_endpoints, color, rects)
            if not cands:
                return None
            rect = cands[0]
            rects.append(rect)
            hidden_rows_by_rect.append((color, hidden_rows, rect))

    hidden_positions: set[tuple[int, int]] = set()
    hidden_rows = set()
    for color, rows, (_r0, c0, _r1, c1) in hidden_rows_by_rect:
        for row in rows:
            hidden_rows.add(row)
            pos = (row, c1 if color == RED else c0)
            if not _is_isolated(pos, hidden_positions):
                return None
            hidden_positions.add(pos)

    visible_positions: set[tuple[int, int]] = set()
    all_obstacles = set(hidden_positions)

    for row in red_rows:
        start, _end = run_endpoints[row]
        if row in hidden_rows:
            nearest = max(c for rr, c in hidden_positions if rr == row)
            extra_pool = list(range(0, max(0, nearest - 1)))
            extras = _build_row_obstacles(row, extra_pool, all_obstacles | visible_positions, 0, 1)
            if extras is None:
                extras = []
        else:
            nearest = random.randint(0, start - 1)
            extra_pool = list(range(0, max(0, nearest - 1)))
            extras = _build_row_obstacles(row, extra_pool, all_obstacles | visible_positions, 0, 1)
            if extras is None:
                return None
            if not _is_isolated((row, nearest), all_obstacles | visible_positions | {(row, c) for c in extras}):
                return None
            extras.append(nearest)
        for c in extras:
            visible_positions.add((row, c))
            all_obstacles.add((row, c))

    for row in blue_rows:
        _start, end = run_endpoints[row]
        if row in hidden_rows:
            nearest = min(c for rr, c in hidden_positions if rr == row)
            extra_pool = list(range(nearest + 2, w))
            extras = _build_row_obstacles(row, extra_pool, all_obstacles | visible_positions, 0, 1)
            if extras is None:
                extras = []
        else:
            nearest = random.randint(end + 1, w - 1)
            extra_pool = list(range(nearest + 2, w))
            extras = _build_row_obstacles(row, extra_pool, all_obstacles | visible_positions, 0, 1)
            if extras is None:
                return None
            if not _is_isolated((row, nearest), all_obstacles | visible_positions | {(row, c) for c in extras}):
                return None
            extras.append(nearest)
        for c in extras:
            visible_positions.add((row, c))
            all_obstacles.add((row, c))

    non_moving_rows = [row for row in range(h) if row not in moving_rows]
    random.shuffle(non_moving_rows)
    for row in non_moving_rows:
        if random.random() < 0.55:
            count_lo = 1 if random.random() < 0.35 else 0
            cols = list(range(w))
            extras = _build_row_obstacles(row, cols, all_obstacles | visible_positions, count_lo, 2)
            if extras is None:
                if count_lo == 0:
                    extras = []
                else:
                    return None
            for c in extras:
                visible_positions.add((row, c))
                all_obstacles.add((row, c))

    grid = np.full((h, w), BG, dtype=int)
    for r, c in visible_positions:
        grid[r, c] = GREEN
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = YELLOW
    for row in red_rows:
        length = run_lengths[row]
        grid[row, w - length : w] = RED
    for row in blue_rows:
        length = run_lengths[row]
        grid[row, :length] = BLUE

    if any(grid[r, c] != YELLOW for r, c in hidden_positions):
        return None
    if np.any((grid == RED) & (grid == YELLOW)) or np.any((grid == BLUE) & (grid == YELLOW)):
        return None

    out = main(grid)
    visible_only = _visible_only_output(grid)
    real_targets = _targets_after_transform(grid, reveal_hidden=True)
    naive_targets = _targets_after_transform(grid, reveal_hidden=False)

    if all(real_targets[row] == naive_targets[row] for row in real_targets):
        return None
    if not any(real_targets[row] != naive_targets[row] for row in hidden_rows):
        return None
    if len(_find_rectangles(grid)) != len(rects):
        return None
    if not set(np.unique(grid)).issubset({BG, BLUE, RED, GREEN, YELLOW}):
        return None
    if not set(np.unique(out)).issubset({BG, BLUE, RED, GREEN}):
        return None
    if np.array_equal(out, visible_only):
        return None
    if np.any(out == YELLOW):
        return None
    if len(hidden_positions) < 1:
        return None
    for row in moving_rows:
        if not np.any(out[row] == GREEN):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid directional-mask sliding puzzle")
