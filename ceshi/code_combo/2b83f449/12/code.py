# Combo variant of 2b83f449 (index 12)
# combo_spec: D5/0+D6/0
# Name: 90度旋转网格+单侧令牌

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask

"""
Design analysis for the fused variant.

1. Concrete fused behavior on the pixel grid
   - Canonical view: keep the seed task's original geometry in mind:
     horizontal pipe rows, horizontal 1x3 gates in wall rows, and tokens entering
     from the left edge only. Tokens drop through unique exits and finally pack at
     the far end of the destination segment.
   - Displayed fused view: rotate the whole canonical scene 90 degrees clockwise.
     Then:
       * pipe rows become vertical pipe columns,
       * wall rows become horizontal wall columns,
       * input gates become vertical 3x1 bars,
       * opened gates become horizontal 1x3 posts,
       * left-edge source tokens become top-edge source tokens.
   - Small traceable example:
       canonical input row token at (r, 0) -> after CW rotation it is on row 0 of
       some pipe column c. If it drops three wall layers canonically, the displayed
       token travels three columns to the right through three horizontal walls.
       When it reaches its final vertical pipe segment, canonical left/right end
       packing becomes displayed top/bottom end packing in that column.
   - This is a real fusion rather than two independent edits: the D5 source-side
     change is expressed only after the D6 rotation, so the visible rule is
     "top-only sources in vertical pipes", not merely "left-only sources".

2. Complexity calibration / observability
   - The generator enforces real horizontal travel in displayed space by requiring
     at least one token to cross >=3 wall layers in canonical space.
   - The generator also rejects samples where the combo output is too close to the
     rotated two-sided baseline, so the single-sided-source change has a visible
     effect and does not collapse into "just rotate the original task".
   - No extra colors or legends are introduced. The only cues are structural:
     vertical pipes, top-only tokens, and output tokens packed at vertical ends.

3. Implicit assumptions in the original solver, checked one by one
   - Gate inference assumed gates are 1x3 horizontal bars. That no longer holds in
     displayed space, so `main()` first rotates the grid back to canonical space
     before inferring gates.
   - Segment traversal assumed pipe segments are horizontal rows with <=1 exit
     below and <=1 entry above. That assumption still holds in canonical space, and
     the generator preserves it before the final display rotation.
   - Token simulation assumed tokens are already on passable pipe cells. The new
     generator guarantees top-edge displayed tokens rotate back to left-edge
     canonical tokens on bg cells.
   - Packing assumed "opposite end from entry above" for dropped tokens and "stay
     at the edge" for tokens that never enter from above. After rotation this
     becomes "opposite vertical end from left-entry" / "stay at the top edge", and
     the rotate-solve-rotate approach preserves that exactly.

4. Placement legality
   - Canonical generation keeps gate centers interior, with the same spacing and
     hole-repair logic as the seed task, so no overlaps or out-of-bounds writes are
     introduced.
   - After the final CW rotation, every vertical gate bar still lies strictly
     inside the displayed grid, and top-edge tokens remain disjoint from gates.

5. Visual cleanliness
   - Input uses only four colors: wall 0, token 3, gate 7, bg 8.
   - Output uses wall 0, token 3, opened posts 6, bg 8.
   - There is no multi-layer annotation. Tokens overwrite only their final resting
     cells, while doors/posts remain clean 3-cell shapes.

6. Learnability from a few examples
   - Humans can see that all sources sit on the top border of vertical columns,
     never on the bottom.
   - They can also see that gates change orientation between input and output:
     vertical 3x1 bars become horizontal 1x3 posts.
   - Repeated examples with nontrivial rightward travel make it clear that the
     rotation is not cosmetic: tokens advance across wall layers and then stack at
     top or bottom ends of vertical pipe segments.

7. New solver assumptions introduced by the fusion
   - `main()` now assumes the displayed task is exactly the CW-rotated canonical
     geometry. The generator enforces that by construction.
   - Because the solver relies on that orientation recovery, the generator must not
     emit arbitrary rotations or mirrored versions of the scene. It does not.
"""

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _infer_gate_token_bg(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))

    gate = None
    for col in sorted(colors):
        if col == WALL:
            continue
        lab, n = label_mask(g == col, connectivity=4)
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

    counts = Counter(map(int, g.flatten()))
    token_candidates = [c for c in colors if c not in (WALL, gate)]
    assert token_candidates
    token = int(min(token_candidates, key=lambda c: (int(counts[int(c)]), int(c))))

    bg_candidates = [c for c in colors if c not in (WALL, gate, token)]
    assert bg_candidates
    bg = int(max(bg_candidates, key=lambda c: (int(counts[int(c)]), -int(c))))
    return int(gate), int(token), int(bg)


def _open_gates(grid: np.ndarray, gate: int, bg: int) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int, copy=True)
    height = out.shape[0]
    open_col = OPEN if gate == GATE and OPEN not in set(map(int, out.flatten())) else (gate - 1 if gate > 0 else OPEN)

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
        out[r, left : left + 3] = int(bg)
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < height:
                out[rr, center] = int(open_col)
    return out, int(open_col)


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
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
        segs.append((int(start), int(c)))
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
    drops = [c for c in range(a, b + 1) if int(grid[r + 1, c]) == int(bg) and int(grid[r + 2, c]) == int(bg)]
    assert len(drops) <= 1, f"segment ({a},{b}) at row {r} has {len(drops)} exits: {drops}"
    return int(drops[0]) if drops else None


def _entry_from_above(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> int | None:
    if r < 2:
        return None
    a, b = seg
    entries = [c for c in range(a, b + 1) if int(grid[r - 1, c]) == int(bg) and int(grid[r - 2, c]) == int(bg)]
    assert len(entries) <= 1, f"segment ({a},{b}) at row {r} has {len(entries)} entries from above: {entries}"
    return int(entries[0]) if entries else None


def _simulate_opened(
    opened: np.ndarray,
    token_points: list[tuple[int, int]],
    bg: int,
    token: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    out = np.array(opened, dtype=int, copy=True)
    for r, c in token_points:
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(out[r, c]) == int(token):
            out[r, c] = int(bg)

    records: list[dict[str, int]] = []
    for order, (start_r, start_c) in enumerate(sorted((int(r), int(c)) for r, c in token_points)):
        r = int(start_r)
        c = int(start_c)
        depth = 0
        if int(out[r, c]) != int(bg):
            records.append({"order": order, "r": r, "c": c, "depth": depth, "seg_a": c, "seg_b": c})
            continue
        while True:
            seg = _find_segment(out, r, c, bg)
            if seg is None:
                break
            exit_col = _segment_exit(out, r, seg, bg)
            if exit_col is None:
                break
            r += 2
            c = int(exit_col)
            depth += 1
            if r >= out.shape[0]:
                break
        final_seg = _find_segment(out, r, c, bg)
        if final_seg is None:
            final_seg = (int(c), int(c))
        records.append(
            {
                "order": int(order),
                "r": int(r),
                "c": int(c),
                "depth": int(depth),
                "seg_a": int(final_seg[0]),
                "seg_b": int(final_seg[1]),
            }
        )

    buckets: dict[tuple[int, int, int, int], list[dict[str, int]]] = defaultdict(list)
    for rec in records:
        r = int(rec["r"])
        c = int(rec["c"])
        seg = (int(rec["seg_a"]), int(rec["seg_b"]))
        a, b = seg
        entry = _entry_from_above(out, r, seg, bg)
        if entry is not None:
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if c <= a else b
        buckets[(r, a, b, int(pack_end))].append(rec)

    for (r, a, b, pack_end), bucket in buckets.items():
        ordered = sorted(bucket, key=lambda rec: int(rec["order"]))
        if pack_end == a:
            for idx, _rec in enumerate(ordered):
                if a + idx <= b:
                    out[r, a + idx] = int(token)
        else:
            for idx, _rec in enumerate(ordered):
                if b - idx >= a:
                    out[r, b - idx] = int(token)
    return out, records


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]]]:
    gate, token, bg = _infer_gate_token_bg(grid)
    opened, _open_col = _open_gates(grid, gate, bg)
    token_points = [(int(r), int(c)) for r, c in np.argwhere(np.array(grid, dtype=int) == int(token))]
    return _simulate_opened(opened, token_points, bg, token)


def main(grid: np.ndarray) -> np.ndarray:
    displayed = np.array(grid, dtype=int)
    canonical = np.rot90(displayed, 1)
    solved, _records = _solve_core(canonical)
    return np.rot90(solved, -1)


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


def _build_input_left(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
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
        grid[2 * pipe_idx, 0] = TOKEN

    return grid


def _build_input_both(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    grid = _build_input_left(gates, holes, width)
    n_walls = len(gates)
    for pipe_idx in range(n_walls):
        grid[2 * pipe_idx, width - 1] = TOKEN
    return grid


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    height = grid.shape[0]
    if r + 2 >= height:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]


def _find_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r, BG):
            above = _entries_above(grid, r, seg)
            below = _exits_below(grid, r, seg)
            if len(above) > 1 or len(below) > 1:
                issues.append((int(r), seg, above, below))
    return issues


def _separator_column(grid: np.ndarray, r: int, left: int, right: int) -> int | None:
    choices = [c for c in range(left + 1, right) if int(grid[r, c]) == BG]
    if not choices:
        return None
    return int(choices[len(choices) // 2])


def _repair_holes(
    gates: list[list[int]],
    holes: list[list[int]],
    width: int,
) -> list[list[int]] | None:
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
                new_cols.append(int(sep))
                seen_up = False
                seen_down = False

            if kind == "up":
                seen_up = True
            else:
                seen_down = True
            prev_col = int(col)

        if not new_cols:
            return None

        for c in new_cols:
            hole_sets[pipe_idx].add(int(c))
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
            candidates.append(int(c))
    return candidates


def _rotated_token_rows(grid: np.ndarray) -> list[int]:
    return sorted(int(r) for r, _c in np.argwhere(grid == TOKEN))


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
                candidates.append((int(row), int(center)))

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

    inp_base = _build_input_left(gates, holes, width)
    out_base, records = _solve_core(inp_base)
    drop_counts = [int(rec["depth"]) for rec in records]
    if max(drop_counts, default=0) < 3:
        return None
    if sum(depth > 0 for depth in drop_counts) < 2:
        return None

    inp = np.rot90(inp_base, -1)
    out = np.rot90(out_base, -1)
    if not np.array_equal(main(inp), out):
        return None

    token_rows_in = _rotated_token_rows(inp)
    if not token_rows_in or set(token_rows_in) != {0}:
        return None

    token_rows_out = _rotated_token_rows(out)
    if len(set(token_rows_out)) < 2:
        return None

    both_base = _build_input_both(gates, holes, width)
    both_out, _both_records = _solve_core(both_base)
    both_rot = np.rot90(both_out, -1)
    diff_to_two_sided = int(np.count_nonzero(out != both_rot))
    if diff_to_two_sided < max(3, n_walls // 2):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/12")
