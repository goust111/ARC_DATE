# Combo variant of 2b83f449 (index 11)
# combo_spec: D3/1+D12/1
# Name: 水平留痕+深度编码颜色

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask, paste

"""
Design analysis for the fused variant.

1. Concrete behavior on-grid
   - Original task: each token sits on a pipe-row edge, drops straight through the
     unique opening of its current segment, and finally packs at the far end of the
     destination segment. Tokens keep the same color.
   - Fused task: before each drop, a token first slides horizontally inside its
     current pipe segment until it is directly above the unique exit. Only that
     horizontal motion leaves a trace, and the trace stays in the original token
     color. After the token finishes all drops and packs, the token itself is
     recolored by its drop depth. Example sketch:
       start at c=0 in row r, exit below at c=4
       original: token effectively teleports to c=4, then falls
       fused: cells (0..4) on row r become trace-color; the settled token color is
       depth-color(depth), not trace-color.
   - This creates two visible signals in the output: horizontal green trails and a
     non-green settled token whose color reflects how many wall layers it crossed.

2. Complexity calibration / observability
   - The generator rejects degenerate cases with no real horizontal motion or with
     too little depth variation. This keeps both D3 and D12 visible.
   - The depth palette uses a small, fixed set of colors for exact depths 0..6:
     0->5, 1->1, 2->2, 3->4, 4->9, 5->7, 6->6. That is enough variation to infer a
     counting rule without introducing too many colors.
   - Trace color stays equal to the input token color (3), so humans can separate
     "where the token moved" from "what depth it finally achieved".

3. Implicit assumptions checked against the original solver
   - Gate inference still relies on the only 1x3 horizontal connected components in
     the input being gates. The generator preserves that.
   - Segment logic still assumes <=1 exit below and <=1 entry above per pipe segment.
     The original hole-repair logic is retained so the solver can keep using that.
   - Original packing only counted indistinguishable tokens. Here tokens can end in
     different colors, so packing must retain token identity/order within a bucket.
     The solver now groups records by destination bucket and places them in stable
     source order.
   - Depth recoloring introduces a new assumption: the generator must keep maximum
     depth inside the explicit palette. It therefore restricts wall count to 5-6,
     so all reachable depths stay in 0..6.

4. Placement legality / cleanliness
   - Gates remain interior 1x3 bars; holes stay inside pipe rows and never at edges.
   - Traces are drawn only on existing pipe cells between the token column and the
     exit column, so they never overlap walls or leave the grid.
   - Final tokens overwrite traces at their resting cells, which keeps the terminal
     token readable instead of visually merged with the path.

5. Why this is learnable from a few examples
   - Across generated examples, several tokens move sideways before dropping, so the
     green trails are obvious and repeated.
   - Multiple positive depths appear in each accepted sample, so the output contains
     repeated settled colors tied to different drop counts.
   - The two cues reinforce each other: the viewer can follow a token's trace to its
     route, count the vertical drops, and verify that the final color depends on
     that count rather than on side, row parity, or gate color.
"""

WALL = 0
TRACE = 3
DEPTH_0 = 5
DEPTH_1 = 1
DEPTH_2 = 2
DEPTH_3 = 4
DEPTH_4 = 9
DEPTH_5 = 7
DEPTH_6 = 6
OPEN = 6
GATE = 7
BG = 8

DEPTH_COLORS = [DEPTH_0, DEPTH_1, DEPTH_2, DEPTH_3, DEPTH_4, DEPTH_5, DEPTH_6]


def _depth_color(depth: int) -> int:
    return int(DEPTH_COLORS[min(max(depth, 0), len(DEPTH_COLORS) - 1)])


def _infer_input_colors(grid: np.ndarray) -> tuple[int, int, int]:
    colors = set(map(int, grid.flatten()))
    gate = None
    for col in sorted(colors):
        if col == WALL:
            continue
        lab, n = label_mask(grid == col, connectivity=4)
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == k)
            if int(pts.shape[0]) != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                gate = int(col)
                break
        if gate is not None:
            break
    assert gate is not None

    counts = Counter(map(int, grid.flatten()))
    candidates = [c for c in colors if c not in (WALL, gate)]
    assert candidates
    token = int(min(candidates, key=lambda c: (int(counts[c]), int(c))))
    bg_candidates = [c for c in colors if c not in (WALL, gate, token)]
    assert bg_candidates
    bg = int(max(bg_candidates, key=lambda c: (int(counts[c]), -int(c))))
    return gate, token, bg


def _open_gates(grid: np.ndarray, gate: int, bg: int) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int)
    height = out.shape[0]
    open_col = OPEN if gate == GATE else (gate - 1 if gate > 0 else OPEN)
    lab, n = label_mask(out == gate, connectivity=4)
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == k)
        if int(pts.shape[0]) != 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        r = int(r0)
        left = int(c0)
        center = left + 1
        out[r, left : left + 3] = bg
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < height:
                out[rr, center] = open_col
    return out, int(open_col)


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != bg:
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == bg:
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _segment_exit(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> int | None:
    if r + 2 >= grid.shape[0]:
        return None
    a, b = seg
    drops = [c for c in range(a, b + 1) if int(grid[r + 1, c]) == bg and int(grid[r + 2, c]) == bg]
    assert len(drops) <= 1, f"segment ({a},{b}) at row {r} has {len(drops)} exits: {drops}"
    return int(drops[0]) if drops else None


def _entry_from_above(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> int | None:
    if r < 2:
        return None
    a, b = seg
    entries = [c for c in range(a, b + 1) if int(grid[r - 1, c]) == bg and int(grid[r - 2, c]) == bg]
    assert len(entries) <= 1, f"segment ({a},{b}) at row {r} has {len(entries)} entries from above: {entries}"
    return int(entries[0]) if entries else None


def _trace_and_drop(
    opened: np.ndarray,
    token_points: list[tuple[int, int]],
    bg: int,
    trace_color: int | None,
) -> tuple[np.ndarray, list[dict[str, int | tuple[int, int]]]]:
    out = opened.copy()
    records: list[dict[str, int | tuple[int, int]]] = []
    for order, (start_r, start_c) in enumerate(sorted(token_points)):
        r = int(start_r)
        c = int(start_c)
        depth = 0
        horiz_total = 0
        horiz_steps = 0
        while True:
            seg = _find_segment(opened, r, c, bg)
            if seg is None:
                break
            exit_col = _segment_exit(opened, r, seg, bg)
            if exit_col is None:
                break
            if exit_col != c and trace_color is not None:
                lo, hi = sorted((c, exit_col))
                out[r, lo : hi + 1] = int(trace_color)
                horiz_total += hi - lo
                horiz_steps += 1
            r += 2
            c = int(exit_col)
            depth += 1
            if r >= opened.shape[0]:
                break
        seg = _find_segment(opened, r, c, bg)
        records.append(
            {
                "order": order,
                "start_r": int(start_r),
                "start_c": int(start_c),
                "r": int(r),
                "c": int(c),
                "depth": int(depth),
                "horiz_total": int(horiz_total),
                "horiz_steps": int(horiz_steps),
                "seg_a": int(seg[0]) if seg is not None else int(c),
                "seg_b": int(seg[1]) if seg is not None else int(c),
            }
        )
    return out, records


def _pack_records(
    out: np.ndarray,
    opened: np.ndarray,
    records: list[dict[str, int | tuple[int, int]]],
    bg: int,
    fixed_color: int | None = None,
) -> None:
    buckets: dict[tuple[int, int, int, int], list[dict[str, int | tuple[int, int]]]] = defaultdict(list)
    for rec in records:
        r = int(rec["r"])
        c = int(rec["c"])
        seg = (int(rec["seg_a"]), int(rec["seg_b"]))
        a, b = seg
        entry = _entry_from_above(opened, r, seg, bg)
        if entry is not None:
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if c <= a else b
        buckets[(r, a, b, pack_end)].append(rec)

    for (r, a, b, pack_end), bucket in buckets.items():
        ordered = sorted(bucket, key=lambda rec: int(rec["order"]))
        if pack_end == a:
            positions = [a + idx for idx in range(len(ordered)) if a + idx <= b]
        else:
            positions = [b - idx for idx in range(len(ordered)) if b - idx >= a]
        for pos, rec in zip(positions, ordered):
            out[r, pos] = _depth_color(int(rec["depth"])) if fixed_color is None else int(fixed_color)


def _simulate_combo_output(opened: np.ndarray, token_points: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, list[dict[str, int | tuple[int, int]]], dict[str, int]]:
    traced, records = _trace_and_drop(opened, token_points, bg, TRACE)
    out = traced.copy()
    _pack_records(out, opened, records, bg)
    stats = {
        "trace_cells": int(np.count_nonzero((traced == TRACE) & (opened == bg))),
        "max_depth": max((int(rec["depth"]) for rec in records), default=0),
        "min_depth": min((int(rec["depth"]) for rec in records), default=0),
        "distinct_depths": len({int(rec["depth"]) for rec in records}),
        "tokens_with_horizontal": sum(1 for rec in records if int(rec["horiz_total"]) > 0),
        "max_horizontal": max((int(rec["horiz_total"]) for rec in records), default=0),
        "distinct_final_colors": len({int(_depth_color(int(rec["depth"]))) for rec in records}),
    }
    return out, records, stats


def _simulate_original_output(opened: np.ndarray, token_points: list[tuple[int, int]], bg: int, token_color: int) -> np.ndarray:
    out = opened.copy()
    _, records = _trace_and_drop(opened, token_points, bg, None)
    _pack_records(out, opened, records, bg, fixed_color=token_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gate, token, bg = _infer_input_colors(g)
    opened, _open_col = _open_gates(g, gate, bg)
    token_points = [(int(r), int(c)) for r, c in np.argwhere(opened == token)]
    opened[opened == token] = bg
    out, _records, _stats = _simulate_combo_output(opened, token_points, bg)
    return out


def _build_opened_grid(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        r = 2 * wall_idx + 1
        for c in centers:
            grid[r, c - 1] = BG
            grid[r, c] = OPEN
            grid[r, c + 1] = BG
            grid[r - 1, c] = OPEN
            grid[r + 1, c] = OPEN

    return grid


def _build_input(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    gate_patch = np.full((1, 3), GATE, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        r = 2 * wall_idx + 1
        for c in centers:
            paste(grid, gate_patch, r, c - 1)

    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TRACE
        grid[r, width - 1] = TRACE

    return grid


def _find_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r, BG):
            a, b = seg
            above = []
            below = []
            if r >= 2:
                above = [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]
            if r + 2 < grid.shape[0]:
                below = [c for c in range(a, b + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]
            if len(above) > 1 or len(below) > 1:
                issues.append((r, seg, above, below))
    return issues


def _separator_column(grid: np.ndarray, r: int, left: int, right: int) -> int | None:
    choices = [c for c in range(left + 1, right) if int(grid[r, c]) == BG]
    if not choices:
        return None
    return int(choices[len(choices) // 2])


def _repair_holes(gates: list[list[int]], holes: list[list[int]], width: int) -> list[list[int]] | None:
    hole_sets = [set(row) for row in holes]

    while True:
        grid = _build_opened_grid(gates, [sorted(row) for row in hole_sets], width)
        issues = _find_issues(grid)
        if not issues:
            return [sorted(row) for row in hole_sets]

        r, _seg, above, below = issues[0]
        pipe_idx = r // 2
        if len(hole_sets[pipe_idx]) >= 2:
            return None

        events = sorted([(c, "up") for c in above] + [(c, "down") for c in below])
        seen_up = False
        seen_down = False
        prev_col: int | None = None
        new_cols: list[int] = []

        for col, kind in events:
            duplicate = (kind == "up" and seen_up) or (kind == "down" and seen_down)
            if duplicate:
                assert prev_col is not None
                sep = _separator_column(grid, r, prev_col, col)
                if sep is None:
                    return None
                new_cols.append(sep)
                seen_up = False
                seen_down = False

            if kind == "up":
                seen_up = True
            else:
                seen_down = True
            prev_col = col

        if not new_cols:
            return None

        for c in new_cols:
            hole_sets[pipe_idx].add(c)
            if len(hole_sets[pipe_idx]) > 2:
                return None

        if sum(len(row) for row in hole_sets) > 4:
            return None


def _candidate_centers(gates: list[list[int]], row: int, width: int) -> list[int]:
    candidates: list[int] = []
    for c in range(2, width - 2):
        if c in gates[row]:
            continue
        if any(abs(c - other) < 5 for other in gates[row]):
            continue

        ok = True
        for adj in (row - 1, row + 1):
            if 0 <= adj < len(gates) and any(abs(c - other) < 2 for other in gates[adj]):
                ok = False
                break
        if ok:
            candidates.append(c)
    return candidates


def _source_points(width: int, n_walls: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        points.append((r, 0))
        points.append((r, width - 1))
    return points


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(5, 7))
    width = int(rng.integers(16, 20))

    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    spine_len = int(rng.integers(4, min(6, n_walls) + 1))
    spine_start = int(rng.integers(0, n_walls - spine_len + 1))

    prev_center: int | None = None
    go_left = bool(rng.integers(2))
    for row in range(spine_start, spine_start + spine_len):
        if prev_center is None:
            options = list(range(4, width - 4))
        elif go_left:
            options = [c for c in range(3, prev_center - 2)]
        else:
            options = [c for c in range(prev_center + 3, width - 2)]
        if not options:
            return None
        center = int(rng.choice(options))
        gates[row] = [center]
        prev_center = center
        go_left = not go_left

    holes = _repair_holes(gates, holes, width)
    if holes is None:
        return None

    total_gates = sum(len(row) for row in gates)
    target_total = int(rng.integers(max(7, total_gates), 13))

    while total_gates < target_total:
        candidates: list[tuple[int, int]] = []
        for row in range(n_walls):
            if len(gates[row]) >= 3:
                continue
            for center in _candidate_centers(gates, row, width):
                candidates.append((row, center))

        if not candidates:
            break

        order = rng.permutation(len(candidates))
        added = False
        for idx in order:
            row, center = candidates[int(idx)]
            new_gates = [list(cols) for cols in gates]
            new_gates[row] = sorted(new_gates[row] + [center])
            new_holes = _repair_holes(new_gates, holes, width)
            if new_holes is None:
                continue
            gates = new_gates
            holes = new_holes
            total_gates += 1
            added = True
            break

        if not added:
            break

    if not (6 <= total_gates <= 12):
        return None

    opened = _build_opened_grid(gates, holes, width)
    if _find_issues(opened):
        return None

    token_points = _source_points(width, n_walls)
    combo_out, records, stats = _simulate_combo_output(opened, token_points, BG)
    depths = [int(rec["depth"]) for rec in records]
    horiz = [int(rec["horiz_total"]) for rec in records]

    if max(depths) < 3:
        return None
    if len(set(depths)) < 3:
        return None
    if sum(1 for value in horiz if value > 0) < max(4, len(records) // 4):
        return None
    if max(horiz, default=0) < 2:
        return None
    if stats["trace_cells"] < max(6, width // 2):
        return None
    passable_cells = int(np.count_nonzero(opened == BG))
    if stats["trace_cells"] > passable_cells // 2:
        return None

    final_color_counts = Counter(_depth_color(depth) for depth in depths)
    repeated_colors = sum(1 for count in final_color_counts.values() if count >= 2)
    if repeated_colors < 2:
        return None

    base_out = _simulate_original_output(opened, token_points, BG, TRACE)
    if np.array_equal(combo_out, base_out):
        return None

    inp = _build_input(gates, holes, width)
    if not np.array_equal(main(inp), combo_out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo input for task 2b83f449")
