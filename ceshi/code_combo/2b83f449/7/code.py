# Combo variant of 2b83f449 (index 7)
# combo_spec: D3/1+D4/1
# Name: 水平移动留痕+到达顺序排列

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

WALL = 0
TRACE = 1
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior on the ARC grid:
#    The base task alternates even pipe rows and odd wall rows. Each 1x3 gate bar opens into a
#    3-tall vertical post, leaving exactly one downward exit per pipe segment. In the base task
#    a token at an edge effectively teleports to that exit and drops. Here that hidden motion is
#    made visible: on every visited pipe row the token first slides horizontally to the unique
#    exit column, painting TRACE on the traversed bg cells, then it drops by 2 rows.
#    Example sketch on one segment:
#      input row:   3 8 8 8 8 8
#      exit above:        ^
#      base output idea:  token simply appears below after the drop
#      combo output: 1 1 1 1 1 8 on that row segment before any final token overwrite.
#    Once all drops stop, the base task packs a destination bucket to the far end from entry.
#    This combo instead lays the whole bucket from the left edge of the destination segment.
#    Inside that left-to-right run, token order follows original source rows from top to bottom;
#    this matches the D4/1 reference rule "按原始出发行号从高到低对应从左到右".
# b) Complexity calibration:
#    TRACE is a single extra color and the packing rule is a single extra ordering rule, so the
#    change is visible but not noisy. The generator rejects samples with weak traces or with no
#    multi-token destination bucket, so the fusion does not collapse back to a near-base puzzle.
# c) Implicit assumptions audited against main():
#    - Gate detection still assumes a unique horizontal 1x3 gate color in the input.
#    - Routing assumes each bg segment has <=1 exit below and <=1 entry above after opening.
#    - TRACE must not affect routing, so all path reasoning is done on a separate opened grid
#      that never contains TRACE.
#    - Ordered placement assumes the destination segment is long enough; generation checks that
#      all 2*(n_walls) tokens survive final placement.
# d) Placement legality:
#    Gates stay interior, holes stay interior, and traces are only drawn over pipe-row bg cells.
#    No object is placed outside the grid and final tokens overwrite traces only on the cells
#    where they physically settle.
# e) Visual observability:
#    Human solvers can see both components directly: horizontal blue streaks reveal the source
#    route, and left-anchored final token runs differ from the original far-end packing.
# f) Visual cleanliness:
#    The output has only three active non-bg structures in pipe rows: black separators, blue
#    traces, and green tokens. The post color remains the original open-post color, so layers are
#    easy to distinguish.
# g) Generator-side anti-degeneration metric:
#    `visible_trace_pixels` measures D3/1 directly on the rendered output, while `d4_diff_pixels`
#    compares the fused output against a D3-only rendering on the same opened network. Requiring
#    both to be positive guarantees both halves of the combo are actually active.


def _segments_in_row(grid: np.ndarray, r: int, bg: int = BG) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != int(bg):
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == int(bg):
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int = BG) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _seg_exit(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int = BG) -> int | None:
    if r + 2 >= grid.shape[0]:
        return None
    a, b = seg
    drops = _exits_below_unchecked(grid, r, seg, bg)
    assert len(drops) <= 1, f"segment ({a},{b}) at row {r} has {len(drops)} exits: {drops}"
    return drops[0] if drops else None


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int = BG) -> list[int]:
    entries = _entries_above_unchecked(grid, r, seg, bg)
    a, b = seg
    assert len(entries) <= 1, f"segment ({a},{b}) at row {r} has {len(entries)} entries: {entries}"
    return entries


def _exits_below_unchecked(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int = BG) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == int(bg) and int(grid[r + 2, c]) == int(bg)]


def _entries_above_unchecked(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int = BG) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == int(bg) and int(grid[r - 2, c]) == int(bg)]


def _default_sources(height: int, width: int) -> list[dict[str, int]]:
    sources: list[dict[str, int]] = []
    order = 0
    for pipe_idx in range((height - 1) // 2):
        r = 2 * pipe_idx
        for c in (0, width - 1):
            sources.append({"r": r, "c": c, "index": order})
            order += 1
    return sources


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.array(grid, dtype=int)
    colors = sorted(set(map(int, g.flatten())))

    gate = None
    for col in colors:
        if col == WALL:
            continue
        found = False
        for r in range(g.shape[0]):
            for c in range(g.shape[1] - 2):
                if int(g[r, c]) == col and int(g[r, c + 1]) == col and int(g[r, c + 2]) == col:
                    found = True
                    break
            if found:
                break
        if found:
            gate = int(col)
            break
    assert gate is not None

    counts = Counter(map(int, g.flatten()))
    token_candidates = [c for c in colors if c not in (WALL, gate)]
    assert token_candidates
    token = int(min(token_candidates, key=lambda c: (int(counts[c]), int(c))))
    bg_candidates = [c for c in colors if c not in (WALL, gate, token)]
    assert bg_candidates
    bg = int(max(bg_candidates, key=lambda c: (int(counts[c]), -int(c))))
    open_col = 6 if gate == 7 and 6 not in colors else (gate - 1 if gate > 0 else 6)
    return gate, token, bg, open_col


def _opened_from_input(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]], dict[str, int]]:
    gate, token, bg, open_col = _infer_colors(grid)
    g = np.array(grid, dtype=int)
    opened = g.copy()
    height, width = g.shape

    for r in range(height):
        c = 0
        while c <= width - 3:
            if int(g[r, c]) == gate and int(g[r, c + 1]) == gate and int(g[r, c + 2]) == gate:
                center = c + 1
                opened[r, c : c + 3] = int(bg)
                for rr in (r - 1, r, r + 1):
                    if 0 <= rr < height:
                        opened[rr, center] = int(open_col)
                c += 3
            else:
                c += 1

    sources: list[dict[str, int]] = []
    for idx, (r, c) in enumerate(np.argwhere(g == int(token))):
        sources.append({"r": int(r), "c": int(c), "index": idx})
        opened[int(r), int(c)] = int(bg)

    meta = {"gate": int(gate), "token": int(token), "bg": int(bg), "open": int(open_col)}
    return opened, sources, meta


def _simulate_records(opened: np.ndarray, sources: list[dict[str, int]]) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    trace_cells: set[tuple[int, int]] = set()
    records: list[dict[str, object]] = []

    for source in sources:
        cur_r = int(source["r"])
        cur_c = int(source["c"])
        depth = 0
        horizontal_steps = 0

        while True:
            seg = _find_segment(opened, cur_r, cur_c)
            if seg is None:
                break
            exit_col = _seg_exit(opened, cur_r, seg)
            if exit_col is None:
                break
            if cur_c != exit_col:
                lo, hi = sorted((cur_c, exit_col))
                for cc in range(lo, hi + 1):
                    trace_cells.add((cur_r, cc))
                horizontal_steps += hi - lo
            cur_r += 2
            cur_c = exit_col
            depth += 1
            if cur_r >= opened.shape[0]:
                break

        final_seg = _find_segment(opened, cur_r, cur_c)
        if final_seg is None:
            final_seg = (cur_c, cur_c)
        records.append(
            {
                "r": cur_r,
                "seg": final_seg,
                "final_c": cur_c,
                "depth": depth,
                "horizontal_steps": horizontal_steps,
                "source_r": int(source["r"]),
                "source_c": int(source["c"]),
                "source_index": int(source["index"]),
            }
        )

    return records, trace_cells


def _arrival_sort_key(rec: dict[str, object]) -> tuple[int, int, int]:
    return (int(rec["source_r"]), int(rec["source_c"]), int(rec["source_index"]))


def _paint_traces(opened: np.ndarray, trace: int, trace_cells: set[tuple[int, int]]) -> np.ndarray:
    out = opened.copy()
    for r, c in trace_cells:
        if int(out[r, c]) == BG:
            out[r, c] = int(trace)
    return out


def _render_combo(opened: np.ndarray, token: int, trace: int, records: list[dict[str, object]], trace_cells: set[tuple[int, int]]) -> np.ndarray:
    out = _paint_traces(opened, trace, trace_cells)

    buckets: dict[tuple[int, tuple[int, int]], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        buckets[(int(rec["r"]), rec["seg"])].append(rec)

    for (r, seg), recs in buckets.items():
        a, b = seg
        ordered = sorted(recs, key=_arrival_sort_key)
        for idx, _rec in enumerate(ordered):
            c = a + idx
            if c <= b:
                out[r, c] = int(token)

    return out


def _render_base(opened: np.ndarray, token: int, records: list[dict[str, object]], trace: int | None = None, trace_cells: set[tuple[int, int]] | None = None) -> np.ndarray:
    out = opened.copy() if trace is None or trace_cells is None else _paint_traces(opened, trace, trace_cells)
    buckets: dict[tuple[int, tuple[int, int], int], list[dict[str, object]]] = defaultdict(list)

    for rec in records:
        r = int(rec["r"])
        seg = rec["seg"]
        a, b = seg
        entries = _entries_above(opened, r, seg)
        if entries:
            entry = entries[0]
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if int(rec["final_c"]) <= a else b
        buckets[(r, seg, pack_end)].append(rec)

    for (r, seg, pack_end), recs in buckets.items():
        a, b = seg
        ordered = sorted(recs, key=lambda rec: int(rec["source_index"]))
        if pack_end == a:
            for idx, _rec in enumerate(ordered):
                c = a + idx
                if c <= b:
                    out[r, c] = int(token)
        else:
            for idx, _rec in enumerate(ordered):
                c = b - idx
                if c >= a:
                    out[r, c] = int(token)
    return out


def _combo_stats(opened: np.ndarray, records: list[dict[str, object]], trace_cells: set[tuple[int, int]], combo_out: np.ndarray, base_out: np.ndarray) -> dict[str, int]:
    buckets: dict[tuple[int, tuple[int, int]], list[dict[str, object]]] = defaultdict(list)
    right_pack_multi = 0
    for rec in records:
        buckets[(int(rec["r"]), rec["seg"])].append(rec)

    for (r, seg), recs in buckets.items():
        if len(recs) < 2:
            continue
        a, b = seg
        entries = _entries_above(opened, r, seg)
        if entries:
            entry = entries[0]
            pack_end = b if (entry - a) <= (b - entry) else a
            if pack_end == b:
                right_pack_multi += 1

    return {
        "trace_cells": len(trace_cells),
        "visible_trace_pixels": int(np.count_nonzero(combo_out == TRACE)),
        "tokens_with_horizontal_move": sum(1 for rec in records if int(rec["horizontal_steps"]) > 0),
        "max_horizontal_steps": max((int(rec["horizontal_steps"]) for rec in records), default=0),
        "max_drop_depth": max((int(rec["depth"]) for rec in records), default=0),
        "multi_bucket_count": sum(1 for recs in buckets.values() if len(recs) >= 2),
        "right_pack_multi_bucket_count": right_pack_multi,
        "base_diff_pixels": int(np.count_nonzero(combo_out != base_out)),
        "placed_tokens": int(np.count_nonzero(combo_out == TOKEN)),
    }


def _solve_opened(opened: np.ndarray, sources: list[dict[str, int]], token: int = TOKEN, trace: int = TRACE) -> tuple[np.ndarray, dict[str, int]]:
    records, trace_cells = _simulate_records(opened, sources)
    combo_out = _render_combo(opened, token, trace, records, trace_cells)
    base_out = _render_base(opened, token, records)
    d3_only_out = _render_base(opened, token, records, trace=trace, trace_cells=trace_cells)
    stats = _combo_stats(opened, records, trace_cells, combo_out, base_out)
    stats["d4_diff_pixels"] = int(np.count_nonzero(combo_out != d3_only_out))
    return combo_out, stats


def main(grid: np.ndarray) -> np.ndarray:
    opened, sources, meta = _opened_from_input(np.array(grid, dtype=int))
    out, _stats = _solve_opened(opened, sources, token=int(meta["token"]), trace=TRACE)
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

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        r = 2 * wall_idx + 1
        for c in centers:
            grid[r, c - 1 : c + 2] = GATE

    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _find_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r):
            above = _entries_above_unchecked(grid, r, seg)
            below = _exits_below_unchecked(grid, r, seg)
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


def _opened_combo_output_and_stats(gates: list[list[int]], holes: list[list[int]], width: int) -> tuple[np.ndarray, dict[str, int]]:
    opened = _build_opened_grid(gates, holes, width)
    sources = _default_sources(opened.shape[0], opened.shape[1])
    return _solve_opened(opened, sources, token=TOKEN, trace=TRACE)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(5, 10))
    width = int(rng.integers(16, 20))

    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    spine_len = int(rng.integers(4, min(7, n_walls) + 1))
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

    repaired = _repair_holes(gates, holes, width)
    if repaired is None:
        return None
    holes = repaired

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

    output, stats = _opened_combo_output_and_stats(gates, holes, width)
    if stats["max_drop_depth"] < 3:
        return None
    if stats["tokens_with_horizontal_move"] < 4 or stats["visible_trace_pixels"] < 8:
        return None
    if stats["multi_bucket_count"] < 1:
        return None
    if stats["right_pack_multi_bucket_count"] < 1:
        return None
    if stats["d4_diff_pixels"] < 2:
        return None
    if stats["placed_tokens"] != 2 * n_walls:
        return None

    inp = _build_input(gates, holes, width)
    solved = main(inp)
    if not np.array_equal(solved, output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/7")


def _behavior_metrics_from_input(grid: np.ndarray) -> dict[str, int]:
    opened, sources, meta = _opened_from_input(np.array(grid, dtype=int))
    out, stats = _solve_opened(opened, sources, token=int(meta["token"]), trace=TRACE)
    del out
    return stats
