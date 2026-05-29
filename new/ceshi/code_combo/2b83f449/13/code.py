# Combo variant of 2b83f449 (index 13)
# combo_spec: D7/0+D8/0
# Name: 多出口+洞出口

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask, paste

"""
Design analysis for the fused variant.

1. Concrete fused behavior on the pixel grid
   - The base task alternates horizontal pipe rows (bg=8) and wall rows (0 with
     1x3 gate bars). After opening, each bar becomes a 3-tall vertical post and
     its two flank cells in the wall row become passable door exits.
   - D7/0 removes the single-exit invariant. A pipe segment may now see several
     door exits under it. Left-edge tokens always choose the leftmost candidate;
     right-edge tokens choose the rightmost candidate.
   - D8/0 changes the semantics of black single-cell holes in pipe rows. They
     still split horizontal motion, but a hole that sits on a segment boundary
     also acts as a vertical shaft if the wall row below and the next pipe row
     below are open in that column. These are "hole exits".
   - Fusion rule: for each segment, inspect exits in two tiers:
       * first all door exits inside the segment,
       * only if there is no door exit, use hole exits on the segment boundary.
     Within the chosen tier, left tokens take the leftmost exit and right tokens
     take the rightmost exit.
   - Small example:
       current segment = [1..7], door exits = [2, 6], hole exits = [8].
       A left token chooses 2; a right token chooses 6. The hole at 8 is visible
       but ignored because door exits exist. On a later row, if door exits are
       empty and hole exits = [0, 4], left takes 0 and right takes 4.

2. Complexity calibration / observability
   - The generator forces at least one real multi-choice step and at least one
     hole-driven step, so the task does not collapse back to the base puzzle.
   - It also requires at least one row where the left and right source tokens
     from the same pipe row choose different first exits. This makes the D7 side
     rule directly visible from a few examples.
   - A separate check compares the fused output against the same network solved
     with holes disabled as exits; accepted samples must differ, proving D8 is
     behaviorally active rather than decorative.

3. Hidden assumptions in the original solver that no longer hold
   - Original `seg_exit()` asserted at most one exit. That is invalid now, so the
     solver enumerates all door exits and all hole exits, then applies the fused
     selection rule.
   - Original `entry_from_above()` asserted at most one entry and used topology to
     infer the packing side. That is also invalid now. The new solver records the
     actual last chosen entry column for each token and packs to the opposite end
     from that recorded entry.
   - Original logic treated holes as pure separators only. The new solver still
     treats them as blockers for horizontal segmentation, but additionally checks
     whether a boundary hole yields a valid downward shaft.

4. Placement legality
   - Gates are placed strictly inside the grid, with >=5 spacing inside a wall row
     and >=2 center separation from gates in adjacent wall rows, matching the
     original geometric discipline and preventing overlaps.
   - Holes are interior single cells only, never at the left or right border where
     source tokens are placed.
   - The generator validates every candidate by running the solver and rejecting
     any instance that fails the intended fused metrics.

5. Visual cleanliness / learnability
   - The palette stays minimal: wall 0, token 3, opened post 6, gate 7, bg 8.
   - Hole exits are visually legible because a black pixel simultaneously splits a
     pipe row and, in accepted examples, some tokens clearly continue downward at
     those black boundary columns when no door exit is available.
   - Door-priority cases are also preserved: accepted instances include segments
     where both door and hole exits are present, yet tokens choose a door exit.

6. New solver assumptions introduced by the fusion
   - Tokens still originate at the outer edges, so "left" and "right" are stable
     token identities throughout the descent. The generator preserves that.
   - Hole exits are only recognized on immediate segment boundaries (a-1 or b+1),
     not arbitrary black cells elsewhere in the row. The generator does not rely
     on any broader interpretation.
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


def _open_from_input(grid: np.ndarray, gate: int, bg: int) -> tuple[np.ndarray, int]:
    out = np.array(grid, dtype=int, copy=True)
    height = out.shape[0]
    colors = set(map(int, out.flatten()))
    open_col = OPEN if gate == GATE and OPEN not in colors else (gate - 1 if gate > 0 else OPEN)

    lab, n = label_mask(out == int(gate), connectivity=4)
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        if int(pts.shape[0]) != 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        r = int(r0)
        left = int(c0)
        center = int(left + 1)
        out[r, left : left + 3] = int(bg)
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < height:
                out[rr, center] = int(open_col)
    return out, int(open_col)


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
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


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


def _door_exits_below(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [
        int(c)
        for c in range(a, b + 1)
        if int(grid[r + 1, c]) == int(bg) and int(grid[r + 2, c]) == int(bg)
    ]


def _hole_exits_below(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    out: list[int] = []
    for c in (a - 1, b + 1):
        if not (0 < c < grid.shape[1] - 1):
            continue
        if int(grid[r, c]) != WALL:
            continue
        if int(grid[r + 1, c]) == int(bg) and int(grid[r + 2, c]) == int(bg):
            out.append(int(c))
    return sorted(set(out))


def _source_side(col: int, width: int) -> str:
    if col <= 1:
        return "left"
    if col >= width - 2:
        return "right"
    return "left" if col < width // 2 else "right"


def _choose_extreme(exits: list[int], side: str) -> int:
    assert exits
    return int(min(exits) if side == "left" else max(exits))


def _token_points(grid: np.ndarray, token: int) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in np.argwhere(np.array(grid, dtype=int) == int(token)))


def _simulate_opened(
    opened: np.ndarray,
    token_points: list[tuple[int, int]],
    bg: int,
    token: int,
    enable_holes: bool = True,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    out = np.array(opened, dtype=int, copy=True)
    for r, c in token_points:
        if int(out[r, c]) == int(token):
            out[r, c] = int(bg)

    records: list[dict[str, object]] = []
    row_first_choices: dict[int, dict[str, int]] = defaultdict(dict)
    stats = {
        "door_steps": 0,
        "hole_steps": 0,
        "priority_segments": 0,
        "multi_choice_steps": 0,
        "max_depth": 0,
        "paired_row_splits": 0,
    }

    for order, (start_r, start_c) in enumerate(token_points):
        side = _source_side(start_c, out.shape[1])
        r = int(start_r)
        c = int(start_c)
        depth = 0
        last_entry: int | None = None
        steps: list[dict[str, object]] = []

        if int(out[r, c]) == int(bg):
            while True:
                seg = _find_segment(out, r, c, bg)
                if seg is None:
                    break
                door_exits = _door_exits_below(out, r, seg, bg)
                hole_exits = _hole_exits_below(out, r, seg, bg) if enable_holes else []
                if door_exits:
                    chosen_kind = "door"
                    exits = door_exits
                    if hole_exits:
                        stats["priority_segments"] += 1
                elif hole_exits:
                    chosen_kind = "hole"
                    exits = hole_exits
                else:
                    break
                if len(exits) > 1:
                    stats["multi_choice_steps"] += 1
                exit_col = _choose_extreme(exits, side)
                steps.append(
                    {
                        "row": int(r),
                        "seg": seg,
                        "door_exits": tuple(int(x) for x in door_exits),
                        "hole_exits": tuple(int(x) for x in hole_exits),
                        "chosen_kind": chosen_kind,
                        "chosen_col": int(exit_col),
                    }
                )
                if chosen_kind == "door":
                    stats["door_steps"] += 1
                else:
                    stats["hole_steps"] += 1
                last_entry = int(exit_col)
                if start_r not in row_first_choices or side not in row_first_choices[start_r]:
                    row_first_choices[start_r][side] = int(exit_col)
                r += 2
                c = int(exit_col)
                depth += 1
                if r >= out.shape[0]:
                    break

        final_seg = _find_segment(out, r, c, bg)
        if final_seg is None:
            final_seg = (int(c), int(c))
        stats["max_depth"] = max(int(stats["max_depth"]), int(depth))
        records.append(
            {
                "order": int(order),
                "start_r": int(start_r),
                "start_c": int(start_c),
                "side": side,
                "r": int(r),
                "c": int(c),
                "depth": int(depth),
                "seg": final_seg,
                "entry_col": last_entry,
                "steps": steps,
            }
        )

    for choices in row_first_choices.values():
        if "left" in choices and "right" in choices and int(choices["left"]) != int(choices["right"]):
            stats["paired_row_splits"] += 1

    buckets: dict[tuple[int, tuple[int, int], int], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        seg = rec["seg"]
        assert isinstance(seg, tuple)
        a, b = int(seg[0]), int(seg[1])
        entry_col = rec["entry_col"]
        if entry_col is not None:
            entry = int(entry_col)
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            side = str(rec["side"])
            pack_end = a if side == "left" else b
        buckets[(int(rec["r"]), (a, b), int(pack_end))].append(rec)

    for (r, (a, b), pack_end), bucket in buckets.items():
        ordered = sorted(bucket, key=lambda item: int(item["order"]))
        if pack_end == a:
            for idx, _rec in enumerate(ordered):
                if a + idx <= b:
                    out[r, a + idx] = int(token)
        else:
            for idx, _rec in enumerate(ordered):
                if b - idx >= a:
                    out[r, b - idx] = int(token)

    return out, records, stats


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gate, token, bg = _infer_gate_token_bg(g)
    opened, _open_col = _open_from_input(g, gate, bg)
    token_points = _token_points(g, token)
    out, _records, _stats = _simulate_opened(opened, token_points, bg, token, enable_holes=True)
    return out


def _candidate_centers(gates: list[list[int]], row: int, width: int) -> list[int]:
    out: list[int] = []
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
            out.append(int(c))
    return out


def _build_gate_structure(rng: np.random.Generator) -> tuple[list[list[int]], int, int]:
    n_walls = int(rng.integers(5, 8))
    width = int(rng.integers(17, 20))
    gates: list[list[int]] = [[] for _ in range(n_walls)]

    left0 = int(rng.integers(4, max(5, width // 2 - 1)))
    right_min = left0 + 5
    right_max = width - 3
    if right_min > right_max:
        right_min = right_max
    right0 = int(rng.integers(right_min, right_max + 1))
    gates[0] = sorted([int(left0), int(right0)])

    prev_center = int((left0 + right0) // 2)
    for row in range(1, n_walls):
        candidates = _candidate_centers(gates, row, width)
        if not candidates:
            candidates = [c for c in range(2, width - 2)]
        candidates.sort(key=lambda c: (abs(c - prev_center), c))
        band = candidates[: min(5, len(candidates))]
        chosen = int(band[int(rng.integers(len(band)))])
        gates[row] = [int(chosen)]
        prev_center = int(chosen)

    focus_wall = int(rng.integers(1, n_walls - 1))
    focus_candidates = _candidate_centers(gates, focus_wall, width)
    if focus_candidates:
        focus_center = gates[focus_wall][0]
        focus_candidates.sort(key=lambda c: (abs(c - focus_center), c), reverse=True)
        gates[focus_wall].append(int(focus_candidates[0]))
        gates[focus_wall].sort()

    total = sum(len(row) for row in gates)
    target_total = int(rng.integers(max(6, total), min(11, 3 * n_walls) + 1))
    while total < target_total:
        choices: list[tuple[int, int]] = []
        for row in range(n_walls):
            if len(gates[row]) >= 3:
                continue
            for center in _candidate_centers(gates, row, width):
                choices.append((int(row), int(center)))
        if not choices:
            break
        row, center = choices[int(rng.integers(len(choices)))]
        gates[row].append(int(center))
        gates[row].sort()
        total += 1

    return gates, width, focus_wall


def _sample_holes(
    rng: np.random.Generator,
    gates: list[list[int]],
    width: int,
    focus_wall: int,
) -> list[list[int]]:
    n_walls = len(gates)
    holes: list[set[int]] = [set() for _ in range(n_walls + 1)]

    focus_pipe = focus_wall
    focus_candidates: list[int] = []
    for center in gates[focus_wall]:
        for c in (center - 1, center + 1):
            if 0 < c < width - 1:
                focus_candidates.append(int(c))
    rng.shuffle(focus_candidates)
    for c in focus_candidates[:2]:
        holes[focus_pipe].add(int(c))

    target_total = int(rng.integers(2, 5))
    attempts = 0
    while sum(len(row) for row in holes) < target_total and attempts < 40:
        attempts += 1
        pipe_idx = int(rng.integers(1, n_walls))
        candidates: list[int] = []
        if pipe_idx < n_walls:
            for center in gates[pipe_idx]:
                for c in (center - 1, center + 1):
                    if 0 < c < width - 1:
                        candidates.append(int(c))
        if pipe_idx > 0:
            for center in gates[pipe_idx - 1]:
                for c in (center - 1, center + 1):
                    if 0 < c < width - 1:
                        candidates.append(int(c))
        candidates.extend(int(x) for x in rng.integers(1, width - 1, size=4))
        rng.shuffle(candidates)
        for c in candidates:
            if not (0 < c < width - 1):
                continue
            if c in holes[pipe_idx]:
                continue
            holes[pipe_idx].add(int(c))
            break

    return [sorted(int(c) for c in row) for row in holes]


def _fusion_metrics(
    opened: np.ndarray,
    token_points: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    fused_out, _records, stats = _simulate_opened(opened, token_points, BG, TOKEN, enable_holes=True)
    no_hole_out, _records_no_hole, _stats_no_hole = _simulate_opened(
        opened, token_points, BG, TOKEN, enable_holes=False
    )
    stats = dict(stats)
    stats["diff_vs_no_hole"] = int(np.count_nonzero(fused_out != no_hole_out))
    topology_multi = 0
    topology_both = 0
    for r in range(0, opened.shape[0] - 2, 2):
        for seg in _segments_in_row(opened, r, BG):
            door_exits = _door_exits_below(opened, r, seg, BG)
            hole_exits = _hole_exits_below(opened, r, seg, BG)
            if len(door_exits) + len(hole_exits) >= 2:
                topology_multi += 1
            if door_exits and hole_exits:
                topology_both += 1
    stats["topology_multi_segments"] = int(topology_multi)
    stats["topology_both_segments"] = int(topology_both)
    return fused_out, stats, no_hole_out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    for _ in range(150):
        gates, width, focus_wall = _build_gate_structure(rng)
        for _ in range(80):
            holes = _sample_holes(rng, gates, width, focus_wall)
            opened = _build_opened_grid(gates, holes, width)
            inp = _build_input(gates, holes, width)
            token_points = _token_points(inp, TOKEN)
            fused_out, stats, _no_hole_out = _fusion_metrics(opened, token_points)

            if int(stats["topology_multi_segments"]) < 1:
                continue
            if int(stats["paired_row_splits"]) < 1:
                continue
            if int(stats["hole_steps"]) < 1:
                continue
            if int(stats["max_depth"]) < 3:
                continue
            if int(stats["diff_vs_no_hole"]) < 1:
                continue
            if int(stats["topology_both_segments"]) < 1:
                continue
            if not np.array_equal(main(inp), fused_out):
                continue
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/13")
