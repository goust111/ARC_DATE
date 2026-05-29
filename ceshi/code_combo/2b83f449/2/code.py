# Combo variant of 2b83f449 (index 2)
# combo_spec: D1/0+D4/0
# Name: 门消失+近端堆积
#
# Design analysis
# - Concrete behavior:
#   Input still shows horizontal 1x3 orange gates in wall rows. In the original task,
#   a gate at columns c-1..c+1 opens into a pink vertical post at column c, so tokens
#   fall through the flanks and later stack at the far end of their destination pipe.
#   In this combo, the gate vanishes and leaves a 1-cell hole at c instead: the wall
#   row becomes 0,8,0 across those three cells, with no vertical post anywhere. A token
#   that previously would have dropped beside the post now drops through the center hole,
#   and once it stops, it packs toward the endpoint on the same side as the entry column.
# - Complexity calibration:
#   Erasing the post removes the most salient original divider, so the generator keeps the
#   single-entry/single-exit invariant by inserting pipe-row holes when needed. This keeps
#   the puzzle inferable from 2-4 examples: the visible differences are structural
#   (no pink posts, only wall holes) and behavioral (settling flips to the entry side).
# - Hidden-assumption audit:
#   `main()` assumes gates are 1x3 bars of color 7 on odd rows, pipe rows are even rows,
#   holes are wall-colored cells inside pipe rows, and every reachable pipe segment has at
#   most one opening above and below. The generator enforces spacing, bounds, and repairs
#   holes until those entry/exit counts are <= 1, so the solver never needs an arbitrary
#   multi-exit tie-breaker.
# - Placement legality:
#   Gate centers stay in [2, W-3], same-row centers are spaced by >= 5, adjacent wall rows
#   differ by >= 2 when both contain gates, and added pipe holes stay interior and capped
#   at the original observed limits.
# - Visual observability / cleanliness:
#   Output uses only wall/background/token colors. Former gate locations become single wall
#   holes, so the absence of color 6 is obvious. Near-end packing is also easy to see,
#   because tokens line up beside the incoming hole instead of collecting at the opposite end.
# - Anti-degeneracy:
#   The generator rejects samples unless at least one token drops 3+ levels and the near-pack
#   output differs from the same opened network under the original far-pack rule.

from __future__ import annotations

from collections import defaultdict

import numpy as np

WALL = 0
TOKEN = 3
GATE = 7
BG = 8


def _parse_gate_centers(grid: np.ndarray) -> list[list[int]]:
    height, width = grid.shape
    n_walls = (height - 1) // 2
    gates: list[list[int]] = [[] for _ in range(n_walls)]
    for wall_idx in range(n_walls):
        wr = 2 * wall_idx + 1
        for c in range(1, width - 1):
            if (
                int(grid[wr, c - 1]) == GATE
                and int(grid[wr, c]) == GATE
                and int(grid[wr, c + 1]) == GATE
            ):
                gates[wall_idx].append(c)
    return gates


def _build_opened_from_input(grid: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    g = np.array(grid, dtype=int)
    opened = g.copy()
    gates = _parse_gate_centers(g)
    opened[opened == TOKEN] = BG
    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for c in centers:
            opened[wr, c - 1 : c + 2] = WALL
            opened[wr, c] = BG
    return opened, gates


def _segments_in_row(grid: np.ndarray, r: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != BG:
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == BG:
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]


def _trace_records(opened: np.ndarray) -> tuple[list[dict[str, int | tuple[int, int] | None]], list[int]]:
    height, width = opened.shape
    n_walls = (height - 1) // 2
    records: list[dict[str, int | tuple[int, int] | None]] = []
    drop_counts: list[int] = []

    for pipe_idx in range(n_walls):
        for start_c in (0, width - 1):
            r = 2 * pipe_idx
            c = start_c
            drops = 0
            while True:
                seg = _find_segment(opened, r, c)
                if seg is None:
                    break
                exits = _exits_below(opened, r, seg)
                assert len(exits) <= 1, f"row {r} segment {seg} has multiple exits {exits}"
                if not exits:
                    break
                r += 2
                c = exits[0]
                drops += 1

            seg = _find_segment(opened, r, c)
            if seg is None:
                continue
            entries = _entries_above(opened, r, seg)
            assert len(entries) <= 1, f"row {r} segment {seg} has multiple entries {entries}"
            records.append(
                {
                    "row": r,
                    "seg_a": seg[0],
                    "seg_b": seg[1],
                    "final_c": c,
                    "start_c": start_c,
                    "entry": entries[0] if entries else -1,
                }
            )
            drop_counts.append(drops)

    return records, drop_counts


def _pack_output(
    opened: np.ndarray,
    records: list[dict[str, int | tuple[int, int] | None]],
    *,
    near: bool,
) -> np.ndarray:
    out = opened.copy()
    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)

    for rec in records:
        r = int(rec["row"])
        seg = (int(rec["seg_a"]), int(rec["seg_b"]))
        a, b = seg
        entry = int(rec["entry"])
        if entry >= 0:
            near_end = a if (entry - a) <= (b - entry) else b
            pack_end = near_end if near else (b if near_end == a else a)
        else:
            start_c = int(rec["start_c"])
            pack_end = a if start_c <= a else b
        buckets[(r, seg, pack_end)] += 1

    for (r, (a, b), pack_end), count in buckets.items():
        if pack_end == a:
            for i in range(count):
                if a + i <= b:
                    out[r, a + i] = TOKEN
        else:
            for i in range(count):
                if b - i >= a:
                    out[r, b - i] = TOKEN

    return out


def _simulate_output(opened: np.ndarray, *, near: bool) -> tuple[np.ndarray, list[int], list[dict[str, int | tuple[int, int] | None]]]:
    records, drop_counts = _trace_records(opened)
    return _pack_output(opened, records, near=near), drop_counts, records


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
        wr = 2 * wall_idx + 1
        for c in centers:
            grid[wr, c] = BG

    return grid


def _build_input(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for c in centers:
            grid[wr, c - 1 : c + 2] = GATE

    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _find_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r):
            above = _entries_above(grid, r, seg)
            below = _exits_below(grid, r, seg)
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


def _behavior_metrics_from_parts(gates: list[list[int]], holes: list[list[int]], width: int) -> dict[str, int]:
    opened = _build_opened_grid(gates, holes, width)
    near_out, drop_counts, records = _simulate_output(opened, near=True)
    far_out, _, _ = _simulate_output(opened, near=False)
    return {
        "gate_count": sum(len(row) for row in gates),
        "entry_settles": sum(1 for rec in records if int(rec["entry"]) >= 0),
        "max_drop": max(drop_counts, default=0),
        "packing_delta": int(np.count_nonzero((near_out == TOKEN) != (far_out == TOKEN))),
    }


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    opened, gates = _build_opened_from_input(np.array(grid, dtype=int))
    holes = []
    for r in range(0, opened.shape[0], 2):
        holes.append([int(c) for c in np.where(opened[r] == WALL)[0]])
    near_out, drop_counts, records = _simulate_output(opened, near=True)
    far_out, _, _ = _simulate_output(opened, near=False)
    return {
        "gate_count": sum(len(row) for row in gates),
        "entry_settles": sum(1 for rec in records if int(rec["entry"]) >= 0),
        "max_drop": max(drop_counts, default=0),
        "packing_delta": int(np.count_nonzero((near_out == TOKEN) != (far_out == TOKEN))),
        "color6_count": int(np.count_nonzero(near_out == 6)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(5, 10))
    width = int(rng.integers(16, 20))

    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    spine_len = int(rng.integers(3, min(6, n_walls) + 1))
    spine_start = int(rng.integers(0, n_walls - spine_len + 1))

    prev_center: int | None = None
    go_left = bool(rng.integers(2))
    for row in range(spine_start, spine_start + spine_len):
        if prev_center is None:
            options = list(range(3, width - 3))
        elif go_left:
            options = [c for c in range(3, prev_center - 1)]
        else:
            options = [c for c in range(prev_center + 2, width - 2)]
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
    target_total = int(rng.integers(max(6, total_gates), 13))

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

    output, _, _ = _simulate_output(opened, near=True)
    metrics = _behavior_metrics_from_parts(gates, holes, width)
    if metrics["max_drop"] < 3:
        return None
    if metrics["entry_settles"] < 4:
        return None
    if metrics["packing_delta"] < 2:
        return None

    inp = _build_input(gates, holes, width)
    if not np.array_equal(main(inp), output):
        return None

    return inp


def main(grid: np.ndarray) -> np.ndarray:
    opened, _gates = _build_opened_from_input(np.array(grid, dtype=int))
    out, _drop_counts, _records = _simulate_output(opened, near=True)
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/2")
