# Combo variant of 2b83f449 (index 26)
# combo_spec: D3/1+D4/1+D9/1
# Name: 留痕+顺序排列+逐个下落
#
# Design analysis
# - Concrete behavior:
#   Gates still open exactly as in the original task. After opening, tokens are
#   processed one by one in reading order. For each token:
#   1) in its current pipe segment, move horizontally to the unique exit column and
#      paint the whole horizontal run with TRACE;
#   2) drop through the wall gap, painting the traversed wall/pipe cells;
#   3) repeat until no exit remains;
#   4) settle immediately at the leftmost free cell of the current reachable segment.
#   Because earlier settled tokens stay in the occupancy grid, later tokens can lose
#   exits and stop sooner. A small example is the two-token basin used by the
#   templates: the earlier token claims the left slot, the later token lands beside
#   it, and the output keeps both travel traces.
# - Complexity calibration:
#   Only one extra color is introduced, TRACE=1, so the new rule is visible without
#   becoming a color-lookup puzzle. Generator accepts a sample only if all three
#   fused effects are active at once:
#   visible trace, at least one 2-token left-to-right basin, and a nonzero difference
#   from a non-sequential control simulation.
# - Assumption audit:
#   1) The original solver assumed tokens do not block one another. Here every source
#      token is inserted into a dynamic occupancy grid before the simulation starts;
#      the current token is removed from that grid only while it is moving.
#   2) The original solver assumed settling direction depends on the entry side.
#      Here the settling rule is replaced by a simpler invariant: first arrival takes
#      the leftmost free cell, second arrival takes the next one, and so on.
#   3) Trace color must not affect physics. The code therefore writes TRACE only to
#      the rendered output; the motion state is tracked separately in `occupied`.
#   4) The opened network must still satisfy the original <=1-exit structure. The
#      fixed templates preserve that invariant exactly.
# - Placement legality and observability:
#   Generator uses a small family of hand-checked 5-wall templates and applies only
#   horizontal shift and mirror transforms, so objects never overlap or leave the
#   grid. Source tokens are intentionally internal rather than edge-only: this gives
#   repeated, human-visible examples where one token blocks another in the next step.
#   Walls, opened posts, tokens, and traces remain visually distinct because trace is
#   only painted onto background cells and final tokens overwrite their own endpoint.

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import paste

WALL = 0
TRACE = 1
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _segments_in_row(grid: np.ndarray, r: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segments: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != BG:
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == BG:
            c += 1
        segments.append((start, c))
        c += 1
    return segments


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


def _mark_trace_cell(out: np.ndarray, r: int, c: int) -> None:
    if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(out[r, c]) == BG:
        out[r, c] = TRACE


def _mark_horizontal_trace(out: np.ndarray, r: int, c0: int, c1: int) -> None:
    lo, hi = sorted((c0, c1))
    for c in range(lo, hi + 1):
        _mark_trace_cell(out, r, c)


def _queue_entry_column(occupied: np.ndarray, r: int, start_c: int) -> int | None:
    width = occupied.shape[1]
    if 0 <= start_c < width and int(occupied[r, start_c]) == BG:
        return start_c
    if start_c == 0:
        c = 0
        while c < width and int(occupied[r, c]) == TOKEN:
            c += 1
        return c if c < width and int(occupied[r, c]) == BG else None
    if start_c == width - 1:
        c = width - 1
        while c >= 0 and int(occupied[r, c]) == TOKEN:
            c -= 1
        return c if c >= 0 and int(occupied[r, c]) == BG else None
    return start_c if int(occupied[r, start_c]) == BG else None


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[list[int]]]:
    g = np.array(grid, dtype=int)
    height, width = g.shape
    n_walls = (height - 1) // 2
    opened = g.copy()
    gate_centers: list[list[int]] = [[] for _ in range(n_walls)]

    sources = [(int(r), int(c)) for r, c in np.argwhere(g == TOKEN)]
    sources.sort()
    opened[opened == TOKEN] = BG

    for wall_idx in range(n_walls):
        wr = 2 * wall_idx + 1
        for c in range(1, width - 1):
            if int(g[wr, c - 1]) == GATE and int(g[wr, c]) == GATE and int(g[wr, c + 1]) == GATE:
                gate_centers[wall_idx].append(c)
        for center in gate_centers[wall_idx]:
            opened[wr, center - 1 : center + 2] = BG
            opened[wr - 1, center] = OPEN
            opened[wr, center] = OPEN
            opened[wr + 1, center] = OPEN

    return opened, sources, gate_centers


def _simulate_sequential(opened: np.ndarray, sources: list[tuple[int, int]]) -> tuple[np.ndarray, dict[str, int]]:
    out = opened.copy()
    occupied = opened.copy()
    for r, c in sources:
        occupied[r, c] = TOKEN
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    max_horizontal = 0
    max_drops = 0
    blocked_stops = 0

    for start_r, start_c in sorted(sources):
        r = start_r
        c = start_c
        occupied[r, c] = BG
        drops = 0
        while True:
            seg = _find_segment(occupied, r, c)
            assert seg is not None, f"source/step {(r, c)} left passable space"
            exits = _exits_below(occupied, r, seg)
            assert len(exits) <= 1
            if len(exits) != 1:
                bare_seg = _find_segment(opened, r, c)
                if bare_seg is not None and len(_exits_below(opened, r, bare_seg)) == 1:
                    blocked_stops += 1
                break
            exit_c = exits[0]
            max_horizontal = max(max_horizontal, abs(exit_c - c))
            _mark_horizontal_trace(out, r, c, exit_c)
            _mark_trace_cell(out, r + 1, exit_c)
            _mark_trace_cell(out, r + 2, exit_c)
            r += 2
            c = exit_c
            drops += 1

        seg = _find_segment(occupied, r, c)
        assert seg is not None, f"stop cell {(r, c)} is not inside an occupied segment"
        final_c = seg[0]
        max_horizontal = max(max_horizontal, abs(final_c - c))
        _mark_horizontal_trace(out, r, c, final_c)
        occupied[r, final_c] = TOKEN
        out[r, final_c] = TOKEN
        bare_seg = _find_segment(opened, r, final_c)
        assert bare_seg is not None
        counts[(r, bare_seg[0], bare_seg[1])] += 1
        max_drops = max(max_drops, drops)

    metrics = {
        "trace_cells": int(np.count_nonzero(out == TRACE)),
        "max_horizontal": int(max_horizontal),
        "max_bucket": int(max(counts.values(), default=0)),
        "blocked_stops": int(blocked_stops),
        "max_drops": int(max_drops),
    }
    return out, metrics


def _simulate_nonsequential(opened: np.ndarray, sources: list[tuple[int, int]]) -> tuple[np.ndarray, dict[str, int]]:
    visual = opened.copy()
    records: list[dict[str, int]] = []
    max_horizontal = 0
    max_drops = 0

    for order, (start_r, start_c) in enumerate(sorted(sources)):
        r, c = start_r, start_c
        drops = 0
        while True:
            seg = _find_segment(opened, r, c)
            if seg is None:
                break
            exits = _exits_below(opened, r, seg)
            if len(exits) != 1:
                break
            exit_c = exits[0]
            max_horizontal = max(max_horizontal, abs(exit_c - c))
            _mark_horizontal_trace(visual, r, c, exit_c)
            _mark_trace_cell(visual, r + 1, exit_c)
            _mark_trace_cell(visual, r + 2, exit_c)
            r += 2
            c = exit_c
            drops += 1

        bare_seg = _find_segment(opened, r, c)
        assert bare_seg is not None
        records.append({"order": order, "r": r, "stop_c": c, "seg_a": bare_seg[0], "seg_b": bare_seg[1]})
        max_drops = max(max_drops, drops)

    out = visual.copy()
    buckets: dict[tuple[int, int, int], list[dict[str, int]]] = defaultdict(list)
    for rec in records:
        buckets[(rec["r"], rec["seg_a"], rec["seg_b"])].append(rec)

    for (r, a, b), recs in buckets.items():
        ordered = sorted(recs, key=lambda rec: rec["order"])
        for idx, rec in enumerate(ordered):
            final_c = a + idx
            assert final_c <= b
            max_horizontal = max(max_horizontal, abs(final_c - rec["stop_c"]))
            _mark_horizontal_trace(out, r, rec["stop_c"], final_c)
            out[r, final_c] = TOKEN

    metrics = {
        "trace_cells": int(np.count_nonzero(out == TRACE)),
        "max_horizontal": int(max_horizontal),
        "max_bucket": int(max((len(v) for v in buckets.values()), default=0)),
        "blocked_stops": 0,
        "max_drops": int(max_drops),
    }
    return out, metrics


def _choose_far_endpoint(grid: np.ndarray, r: int, seg: tuple[int, int], c: int) -> int:
    a, b = seg
    entries = _entries_above(grid, r, seg)
    assert len(entries) <= 1
    if len(entries) == 1:
        entry = entries[0]
        return b if (entry - a) <= (b - entry) else a
    return a if c <= (a + b) // 2 else b


def _simulate_sequential_far(opened: np.ndarray, sources: list[tuple[int, int]]) -> np.ndarray:
    out = opened.copy()
    occupied = opened.copy()
    for r, c in sources:
        occupied[r, c] = TOKEN

    for start_r, start_c in sorted(sources):
        r = start_r
        c = start_c
        occupied[r, c] = BG
        while True:
            seg = _find_segment(occupied, r, c)
            assert seg is not None
            exits = _exits_below(occupied, r, seg)
            assert len(exits) <= 1
            if len(exits) != 1:
                break
            exit_c = exits[0]
            _mark_horizontal_trace(out, r, c, exit_c)
            _mark_trace_cell(out, r + 1, exit_c)
            _mark_trace_cell(out, r + 2, exit_c)
            r += 2
            c = exit_c

        seg = _find_segment(occupied, r, c)
        assert seg is not None
        final_c = _choose_far_endpoint(occupied, r, seg, c)
        _mark_horizontal_trace(out, r, c, final_c)
        occupied[r, final_c] = TOKEN
        out[r, final_c] = TOKEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    opened, sources, _gate_centers = _parse_input(np.array(grid, dtype=int))
    out, _metrics = _simulate_sequential(opened, sources)
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


def _build_input(gates: list[list[int]], holes: list[list[int]], width: int, sources: list[tuple[int, int]]) -> np.ndarray:
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

    for r, c in sources:
        grid[r, c] = TOKEN

    return grid


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]


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


def _source_positions(height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(0, height - 1, 2) for c in (0, width - 1)]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    width = int(rng.integers(16, 20))
    n_walls = 5
    templates = [
        {
            "gates": [[7], [4], [2], [9], [4]],
            "sources": [(2, 1), (2, 6), (4, 5), (4, 10)],
        },
        {
            "gates": [[8], [5], [3], [10], [5]],
            "sources": [(2, 4), (2, 6), (4, 6), (4, 7)],
        },
        {
            "gates": [[10], [7], [4], [7], [10]],
            "sources": [(0, 1), (0, 7), (2, 1), (4, 10)],
        },
    ]
    template = templates[int(rng.integers(len(templates)))]
    base_width = 13
    shift = int(rng.integers(0, width - base_width + 1))
    mirror = bool(rng.integers(2))

    def remap_col(c: int) -> int:
        mapped = c + shift
        if mirror:
            return width - 1 - mapped
        return mapped

    gates = [[remap_col(c) for c in row] for row in template["gates"]]
    holes = [[] for _ in range(n_walls + 1)]
    sources = sorted((r, remap_col(c)) for r, c in template["sources"])

    opened = _build_opened_grid(gates, holes, width)
    if _find_issues(opened):
        return None

    seq_out, seq_metrics = _simulate_sequential(opened, sources)
    nonseq_out, _nonseq_metrics = _simulate_nonsequential(opened, sources)
    far_out = _simulate_sequential_far(opened, sources)
    divergence = int(np.count_nonzero(seq_out != nonseq_out))
    packing_delta = int(np.count_nonzero((seq_out == TOKEN) != (far_out == TOKEN)))

    if seq_metrics["max_drops"] < 3:
        return None
    if seq_metrics["max_horizontal"] < 2:
        return None
    if seq_metrics["max_bucket"] < 2:
        return None
    if seq_metrics["trace_cells"] < 8:
        return None
    if seq_metrics["blocked_stops"] < 1:
        return None
    if divergence < 3:
        return None
    if packing_delta < 2:
        return None

    inp = _build_input(gates, holes, width, sources)
    if not np.array_equal(main(inp), seq_out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/26")
