# Combo variant of 2b83f449 (index 6)
# combo_spec: D3/0+D4/0
# Name: 令牌上升+近端堆积
#
# Design analysis
# - Concrete behavior:
#   The base task has alternating pipe rows and wall rows. Horizontal 1x3 gates open into
#   vertical blockers, and tokens originally start on every pipe row except the bottom one,
#   then fall downward and settle at horizontal segment ends. This combo changes two linked
#   steps together:
#   1. Sources move to every pipe row except the top one, so tokens rise upward through the
#      unique opening above their current segment.
#   2. Once a token reaches its terminal pipe segment, it no longer packs to a horizontal
#      left/right end. "Near the entry" is reinterpreted using the upward motion: the entry
#      into the final segment is the 3-cell doorway directly below that segment
#      (lower pipe cell, wall cell, target pipe cell), so tokens stack bottom-to-top inside
#      that doorway. The interaction is visible because some output tokens land on wall rows.
#   A small mental example: if a token rises from row 8 into row 6 via column 4 and can rise
#   no further, the original task would place it somewhere on row 6 near one horizontal end.
#   This combo instead uses the vertical entry portal at column 4 and fills (8,4), then
#   (7,4), then (6,4) as more tokens share that same terminal segment/entry.
# - Complexity calibration:
#   The core "single opening per segment" structure is preserved, so path tracing remains
#   ARC-scale and inferable. At the same time, the output is not a trivial flip: the source
#   rows are mirrored and the settling geometry changes from horizontal line ends to short
#   vertical stacks, which is visually distinct in 2-4 examples.
# - Hidden-assumption audit:
#   `main()` assumes odd rows are walls, even rows are pipe rows, gate bars remain 1x3 on
#   input, and every reachable pipe segment has <=1 opening above and <=1 opening below after
#   opening the gates. It also assumes the fixed task palette from the spec (0/3/6/7/8), and
#   that generated source tokens never start on the top pipe row: otherwise an upward-moving
#   token could already sit in a terminal segment whose "near" side is ambiguous. Finally, any
#   terminal segment used for near-end stacking must have at most three tokens sharing one
#   lower entry portal, because that portal contributes only three passable cells. The
#   generator enforces all of these conditions, preserves the single-opening invariant, and
#   rejects samples unless every portal bucket fits without collision.
# - Placement legality:
#   Gate centers stay strictly interior, same-row gates are spaced apart, adjacent wall rows
#   keep a minimum center offset, and repaired holes stay inside pipe rows. Portal stacking
#   only writes onto passable cells already present in the opened grid, so there is no
#   out-of-bounds or object overlap in valid generated samples.
# - Visual observability / cleanliness:
#   Inputs still show the same wall/pipeline scaffold and clearly place tokens on lower rows,
#   making the upward direction observable. Outputs retain the pink gate-post blockers and add
#   compact vertical token piles at final entry portals; this is much easier to distinguish by
#   eye than a subtle tie-breaker or color-based rule.
# - Anti-degeneracy:
#   The generator rejects samples unless:
#   1. at least one token rises 3+ wall levels,
#   2. at least one terminal portal receives >=2 tokens, producing a genuine vertical stack,
#   3. at least one output token lands on a wall row,
#   4. the combo output differs from an "upward + horizontal near-end" ablation on the same
#      opened network.

from __future__ import annotations

from collections import defaultdict

import numpy as np

WALL = 0
TOKEN = 3
OPEN = 6
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
            opened[wr, c - 1 : c + 2] = BG
            opened[wr - 1, c] = OPEN
            opened[wr, c] = OPEN
            opened[wr + 1, c] = OPEN
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


def _openings_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]


def _openings_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]


def _source_points(height: int, width: int) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    for r in range(2, height, 2):
        starts.append((r, 0))
        starts.append((r, width - 1))
    return starts


def _trace_upward(
    opened: np.ndarray,
    starts: list[tuple[int, int]],
) -> tuple[list[dict[str, int]], list[int]]:
    records: list[dict[str, int]] = []
    rise_counts: list[int] = []

    for start_r, start_c in starts:
        r = int(start_r)
        c = int(start_c)
        rises = 0
        if int(opened[r, c]) != BG:
            continue

        while True:
            seg = _find_segment(opened, r, c)
            if seg is None:
                break
            above = _openings_above(opened, r, seg)
            assert len(above) <= 1, f"row {r} segment {seg} has multiple openings above {above}"
            if not above:
                break
            r -= 2
            c = above[0]
            rises += 1

        seg = _find_segment(opened, r, c)
        if seg is None:
            continue
        below = _openings_below(opened, r, seg)
        assert len(below) <= 1, f"row {r} segment {seg} has multiple openings below {below}"
        records.append(
            {
                "row": r,
                "seg_a": seg[0],
                "seg_b": seg[1],
                "final_c": c,
                "start_r": start_r,
                "start_c": start_c,
                "entry_below": below[0] if below else -1,
            }
        )
        rise_counts.append(rises)

    return records, rise_counts


def _pack_combo_vertical(opened: np.ndarray, records: list[dict[str, int]]) -> tuple[np.ndarray, dict[str, int]] | None:
    out = opened.copy()
    occupied: set[tuple[int, int]] = set()
    portal_buckets: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)

    for rec in records:
        entry = int(rec["entry_below"])
        if entry >= 0:
            portal_buckets[(int(rec["row"]), entry)].append(rec)

    max_stack = 0
    wall_token_count = 0

    for (r, entry), bucket in sorted(portal_buckets.items()):
        slots = [(r + 2, entry), (r + 1, entry), (r, entry)]
        if len(bucket) > len(slots):
            return None
        max_stack = max(max_stack, len(bucket))
        for idx, cell in enumerate(slots[: len(bucket)]):
            rr, cc = cell
            if rr >= opened.shape[0] or int(opened[rr, cc]) != BG or cell in occupied:
                return None
            out[rr, cc] = TOKEN
            occupied.add(cell)
            if rr % 2 == 1:
                wall_token_count += 1

    for rec in records:
        if int(rec["entry_below"]) >= 0:
            continue
        cell = (int(rec["row"]), int(rec["final_c"]))
        rr, cc = cell
        if int(opened[rr, cc]) != BG or cell in occupied:
            return None
        out[rr, cc] = TOKEN
        occupied.add(cell)

    return out, {"max_stack": max_stack, "wall_token_count": wall_token_count}


def _pack_horizontal_near(opened: np.ndarray, records: list[dict[str, int]]) -> np.ndarray:
    out = opened.copy()
    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)

    for rec in records:
        r = int(rec["row"])
        seg = (int(rec["seg_a"]), int(rec["seg_b"]))
        a, b = seg
        entry = int(rec["entry_below"])
        if entry >= 0:
            pack_end = a if (entry - a) <= (b - entry) else b
        else:
            pack_end = a if int(rec["final_c"]) <= (a + b) // 2 else b
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


def _simulate_combo_output(opened: np.ndarray) -> tuple[np.ndarray, list[int], dict[str, int]] | None:
    records, rise_counts = _trace_upward(opened, _source_points(*opened.shape))
    packed = _pack_combo_vertical(opened, records)
    if packed is None:
        return None
    out, metrics = packed
    return out, rise_counts, metrics


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
            grid[wr, c - 1] = BG
            grid[wr, c] = OPEN
            grid[wr, c + 1] = BG
            grid[wr - 1, c] = OPEN
            grid[wr + 1, c] = OPEN

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

    for r, c in _source_points(height, width):
        grid[r, c] = TOKEN

    return grid


def _find_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r):
            above = _openings_above(grid, r, seg)
            below = _openings_below(grid, r, seg)
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

    combo = _simulate_combo_output(opened)
    if combo is None:
        return None
    output, rise_counts, metrics = combo

    records, _ = _trace_upward(opened, _source_points(*opened.shape))
    ablation = _pack_horizontal_near(opened, records)

    if max(rise_counts, default=0) < 3:
        return None
    if metrics["max_stack"] < 2:
        return None
    if metrics["wall_token_count"] <= 0:
        return None
    if np.array_equal(output, ablation):
        return None

    inp = _build_input(gates, holes, width)
    if not np.array_equal(main(inp), output):
        return None

    return inp


def main(grid: np.ndarray) -> np.ndarray:
    opened, _gates = _build_opened_from_input(np.array(grid, dtype=int))
    starts = [(int(r), int(c)) for r, c in np.argwhere(np.array(grid, dtype=int) == TOKEN)]
    records, _rise_counts = _trace_upward(opened, starts)
    packed = _pack_combo_vertical(opened, records)
    if packed is None:
        raise RuntimeError("invalid upward portal packing for this grid")
    out, _metrics = packed
    return out


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo 2b83f449/6")
