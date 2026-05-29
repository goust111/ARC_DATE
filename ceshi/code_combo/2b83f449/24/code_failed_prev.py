# Combo variant of 2b83f449 (index 24)
# combo_spec: D3/2+D7/0+D9/1
# Name: 惯性+多出口+逐个占位
#
# Design analysis
# ---------------
# a) Concrete fused behavior on the ARC grid:
#    The base task has alternating pipe rows (even rows, mostly color 8) and wall rows (odd rows,
#    mostly color 0). Each 1x3 gate bar opens into a 3-tall vertical post, so each gate creates
#    two doorway columns at center-1 and center+1. In the seed task, every pipe segment had at most
#    one exit below, so a token only needed to fall straight down through unique doors and final
#    packing could be done in bulk.
#
#    In this combo, three changes are coupled:
#    - D7/0: a segment may have two exits below.
#    - D3/2: a token keeps its current directional bias. It starts with downward intent; if the
#      current column is not an exit, it slides horizontally inside the segment to the chosen exit.
#      Which exit is chosen depends on the side from which the token effectively entered the segment:
#      left-entry uses the leftmost exit, right-entry uses the rightmost exit.
#    - D9/1: tokens are processed one by one in row-major source order. As soon as one token settles,
#      its final pixel becomes a new blocker for later tokens, so later segments can split or lose
#      the doorway directly under the next token.
#
#    Small trace:
#    - Suppose a token arrives at row r, column 8, and the segment has exits [5, 11].
#    - If it entered from the left, it slides right to 11 only if 8 is blocked and the left-entry
#      rule says "take the leftmost"? No: left-entry takes 5, right-entry takes 11. Therefore the
#      current approach side matters, not just the current column.
#    - If the token slides from 8 to 5, then the next segment sees it as having approached from the
#      right side of that next chamber (the token moved left before dropping). That updated side can
#      flip the next two-exit decision.
#    - After the token stops, it immediately occupies the opposite end of its current free segment.
#      That occupied pixel is no longer passable, so a later token may no longer share the same free
#      corridor or may lose access to the straight-down doorway and be forced into a different
#      inertial side choice.
#
# b) Complexity calibration:
#    The generator does not accept samples that merely contain two exits. It requires visible
#    horizontal detours, at least one approach-side update, and at least one later token whose route
#    differs from the no-occupancy ablation. That prevents the combo from collapsing back to either
#    the seed task or the simpler D3/2+D7/0 combo.
#
# c) Broken implicit assumptions from the original main():
#    - "Each segment has <=1 exit below" is false, so the old assert-based `seg_exit` logic must be
#      removed.
#    - "Final side can be inferred from a unique entry above" is false, because the same segment may
#      be entered from different sides after inertial detours.
#    - "All tokens can be routed before any token is placed" is false, because later routing depends
#      on already occupied cells.
#    - "Segments are determined only by walls/holes/posts" is false after D9/1; settled tokens also
#      split segments.
#    The solver below replaces those assumptions with explicit state: current approach side, ordered
#    processing, and passability computed on the evolving occupied grid.
#
# d) Placement legality:
#    The generator keeps all gates strictly inside the grid, preserves the original spacing rules
#    (same-row centers differ by >=5, adjacent-wall centers by >=2), and only places holes on pipe
#    rows away from the border. Candidate layouts are rejected unless every pipe segment in the
#    opened network has at most two exits below, so the D7/0 rule is well-defined.
#
# e) Visual observability:
#    A human can see the rule from geometry alone:
#    - some chambers have two visible doorways below;
#    - left and right edge tokens encounter the same chamber from opposite sides;
#    - output tokens from earlier sources visibly occupy corridor ends, changing later routing.
#    No hidden ordering by color value is used.
#
# f) Visual cleanliness:
#    The combo reuses the seed palette and avoids new annotations. Tokens, gates, posts, holes, and
#    walls remain visually distinct. The generator rejects samples where sequential occupation blocks
#    later source cells at the border, because those cases are needlessly confusing.
#
# g) Generator-side anti-degeneration audit:
#    `_behavior_metrics()` reports:
#    - `potential_multi_exit_segments`
#    - `horizontal_detours`
#    - `approach_updates`
#    - `sequential_vs_parallel_pixels`
#    - `occupancy_route_changes`
#    - `combo_vs_static_pixels`
#    Samples are accepted only when all three dimensions are active on the same grid.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask, paste


WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


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
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    exits: list[int] = []
    for c in range(a, b + 1):
        if int(grid[r + 1, c]) == int(bg) and int(grid[r + 2, c]) == int(bg):
            exits.append(c)
    return exits


def _count_tokens(grid: np.ndarray) -> int:
    return int(np.count_nonzero(np.array(grid, dtype=int) == TOKEN))


def _diff_pixels(a: np.ndarray, b: np.ndarray) -> int:
    aa = np.array(a, dtype=int)
    bb = np.array(b, dtype=int)
    return int(np.count_nonzero(aa != bb))


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))

    gate = None
    for col in sorted(colors):
        if int(col) == WALL:
            continue
        lab, n = label_mask(g == int(col), connectivity=4)
        found = False
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(k))
            if int(pts.shape[0]) != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                found = True
                break
        if found:
            gate = int(col)
            break
    assert gate is not None

    counts = Counter(map(int, g.flatten()))
    token_candidates = [c for c in colors if int(c) not in (WALL, gate)]
    assert token_candidates
    token = int(min(token_candidates, key=lambda c: (int(counts[int(c)]), int(c))))
    bg_candidates = [c for c in colors if int(c) not in (WALL, gate, token)]
    assert bg_candidates
    bg = int(max(bg_candidates, key=lambda c: (int(counts[int(c)]), -int(c))))
    open_col = 6 if gate == 7 and 6 not in colors else (gate - 1 if gate > 0 else 6)
    return gate, token, bg, open_col


def _open_from_input(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]], dict[str, int]]:
    gate, token, bg, open_col = _infer_colors(grid)
    g = np.array(grid, dtype=int)
    opened = g.copy()
    height, width = g.shape

    lab, n = label_mask(opened == int(gate), connectivity=4)
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
        center = left + 1
        opened[r, left : left + 3] = int(bg)
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < height:
                opened[rr, center] = int(open_col)

    sources: list[dict[str, int]] = []
    order = 0
    for r, c in np.argwhere(g == int(token)):
        rr = int(r)
        cc = int(c)
        side_code = 0 if cc <= 1 else 1
        sources.append({"r": rr, "c": cc, "pipe_idx": rr // 2, "order": order, "side_code": side_code})
        opened[rr, cc] = int(bg)
        order += 1

    meta = {"gate": int(gate), "token": int(token), "bg": int(bg), "open": int(open_col)}
    return opened, sources, meta


def _initial_side(source: dict[str, int], side_mode: str) -> str:
    if side_mode == "all_left":
        return "left"
    if side_mode == "all_right":
        return "right"
    return "left" if int(source["side_code"]) == 0 else "right"


def _updated_approach(approach: str, current_c: int, chosen_c: int, side_mode: str) -> tuple[str, int]:
    if side_mode != "combo":
        return approach, 0
    new_approach = approach
    if chosen_c > current_c:
        new_approach = "left"
    elif chosen_c < current_c:
        new_approach = "right"
    return new_approach, int(new_approach != approach)


def _entry_column(grid: np.ndarray, r: int, source_c: int, source_side: str, bg: int) -> tuple[int | None, int]:
    if int(grid[r, source_c]) == int(bg):
        return int(source_c), 0
    step = 1 if source_side == "left" else -1
    c = int(source_c)
    while 0 <= c < grid.shape[1] and int(grid[r, c]) != int(bg):
        c += step
    if 0 <= c < grid.shape[1] and int(grid[r, c]) == int(bg):
        return int(c), abs(int(c) - int(source_c))
    return None, 0


def _trace_token(
    grid: np.ndarray,
    source: dict[str, int],
    bg: int,
    side_mode: str,
) -> tuple[dict[str, object], dict[str, int]]:
    stats = defaultdict(int)
    source_side = _initial_side(source, side_mode)
    start_c, shift = _entry_column(grid, int(source["r"]), int(source["c"]), source_side, bg)
    if start_c is None:
        stats["blocked_starts"] += 1
        return {
            "r": int(source["r"]),
            "seg": None,
            "final_c": int(source["c"]),
            "pack_end": int(source["c"]),
            "depth": 0,
            "approach": source_side,
            "source_side": source_side,
            "source": source,
            "route": tuple(),
            "start_shift": 0,
        }, {k: int(v) for k, v in stats.items()}

    if shift > 0:
        stats["start_shifts"] += 1

    r = int(source["r"])
    c = int(start_c)
    approach = source_side
    depth = 0
    route: list[tuple[int, int, int, str]] = []

    while True:
        seg = _find_segment(grid, r, c, bg)
        if seg is None:
            break
        exits = _exits_below(grid, r, seg, bg)
        if len(exits) >= 2:
            stats["multi_exit_steps"] += 1
        if not exits:
            route.append((int(r), int(c), -1, approach))
            break
        if c in exits:
            chosen = int(c)
            new_approach = approach
        else:
            chosen = int(min(exits) if approach == "left" else max(exits))
            stats["horizontal_detours"] += 1
            stats["horizontal_distance"] += abs(int(chosen) - int(c))
            new_approach, changed = _updated_approach(approach, c, chosen, side_mode)
            stats["approach_updates"] += int(changed)
        route.append((int(r), int(c), int(chosen), approach))
        r += 2
        c = int(chosen)
        depth += 1
        if side_mode == "combo":
            approach = new_approach
        if r >= grid.shape[0]:
            break

    final_seg = _find_segment(grid, r, c, bg)
    if final_seg is None and 0 <= c < grid.shape[1] and int(grid[r, c]) == int(bg):
        final_seg = (int(c), int(c))

    if depth == 0:
        pack_end = int(final_seg[0]) if final_seg is not None and source_side == "left" else int(final_seg[1]) if final_seg is not None else int(c)
    else:
        pack_end = int(final_seg[1]) if final_seg is not None and approach == "left" else int(final_seg[0]) if final_seg is not None else int(c)

    stats["max_depth"] = max(int(stats["max_depth"]), int(depth))
    rec = {
        "r": int(r),
        "seg": final_seg,
        "final_c": int(c),
        "pack_end": int(pack_end),
        "depth": int(depth),
        "approach": approach,
        "source_side": source_side,
        "source": source,
        "route": tuple(route),
        "start_shift": int(shift),
    }
    return rec, {k: int(v) for k, v in stats.items()}


def _merge_stats(dst: defaultdict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] += int(value)


def _place_record_sequential(
    occupied: np.ndarray,
    rec: dict[str, object],
    token: int,
    bg: int,
) -> int | None:
    seg = rec["seg"]
    if seg is None:
        return None
    a, b = seg
    pack_end = int(rec["pack_end"])
    if pack_end == int(a):
        rng = range(int(a), int(b) + 1)
    else:
        rng = range(int(b), int(a) - 1, -1)
    for c in rng:
        if int(occupied[int(rec["r"]), c]) == int(bg):
            occupied[int(rec["r"]), c] = int(token)
            rec["pack_pos"] = int(c)
            return int(c)
    return None


def _render_collective(
    opened: np.ndarray,
    records: list[dict[str, object]],
    token: int,
) -> tuple[np.ndarray, int]:
    out = np.array(opened, copy=True)
    per_segment: dict[tuple[int, tuple[int, int]], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {"left": [], "right": []}
    )
    for rec in records:
        seg = rec["seg"]
        if seg is None:
            continue
        r = int(rec["r"])
        if int(rec["pack_end"]) == int(seg[0]):
            per_segment[(r, seg)]["left"].append(rec)
        else:
            per_segment[(r, seg)]["right"].append(rec)

    placed = 0
    for (r, seg), buckets in per_segment.items():
        a, b = seg
        used: set[int] = set()
        left_recs = sorted(
            buckets["left"],
            key=lambda rec: (int(rec["source"]["pipe_idx"]), int(rec["source"]["order"])),
        )
        right_recs = sorted(
            buckets["right"],
            key=lambda rec: (int(rec["source"]["pipe_idx"]), int(rec["source"]["order"])),
        )
        for idx, rec in enumerate(left_recs):
            pos = int(a + idx)
            if pos > int(b) or pos in used:
                continue
            out[r, pos] = int(token)
            rec["pack_pos"] = pos
            used.add(pos)
            placed += 1
        for idx, rec in enumerate(right_recs):
            pos = int(b - idx)
            if pos < int(a) or pos in used:
                continue
            out[r, pos] = int(token)
            rec["pack_pos"] = pos
            used.add(pos)
            placed += 1
    return out, int(placed)


def _solve_collective(
    opened: np.ndarray,
    sources: list[dict[str, int]],
    token: int,
    bg: int,
    side_mode: str,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    stats = defaultdict(int)
    records: list[dict[str, object]] = []
    for source in sorted(sources, key=lambda item: (int(item["r"]), int(item["c"]), int(item["order"]))):
        rec, local = _trace_token(opened, source, bg, side_mode)
        _merge_stats(stats, local)
        records.append(rec)
    out, placed = _render_collective(opened, records, token)
    stats["placed_tokens"] = int(placed)
    stats["source_tokens"] = int(len(sources))
    return out, records, {k: int(v) for k, v in stats.items()}


def _solve_sequential(
    opened: np.ndarray,
    sources: list[dict[str, int]],
    token: int,
    bg: int,
    side_mode: str,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    occupied = np.array(opened, copy=True)
    stats = defaultdict(int)
    records: list[dict[str, object]] = []

    for source in sorted(sources, key=lambda item: (int(item["r"]), int(item["c"]), int(item["order"]))):
        rec, local = _trace_token(occupied, source, bg, side_mode)
        _merge_stats(stats, local)
        pos = _place_record_sequential(occupied, rec, token, bg)
        if pos is None:
            stats["lost_tokens"] += 1
        else:
            stats["placed_tokens"] += 1
        records.append(rec)

    stats["source_tokens"] = int(len(sources))
    return occupied, records, {k: int(v) for k, v in stats.items()}


def _solve_core(
    grid: np.ndarray,
    side_mode: str,
    sequential: bool,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    opened, sources, meta = _open_from_input(np.array(grid, dtype=int))
    if sequential:
        return _solve_sequential(opened, sources, meta["token"], meta["bg"], side_mode)
    return _solve_collective(opened, sources, meta["token"], meta["bg"], side_mode)


def main(grid: np.ndarray) -> np.ndarray:
    out, _records, _stats = _solve_core(np.array(grid, dtype=int), "combo", True)
    return out


def _solve_with_mode(grid: np.ndarray, side_mode: str, sequential: bool) -> np.ndarray:
    out, _records, _stats = _solve_core(np.array(grid, dtype=int), side_mode, sequential)
    return out


def _record_signature(rec: dict[str, object]) -> tuple[object, ...]:
    seg = rec["seg"]
    seg_sig = None if seg is None else (int(seg[0]), int(seg[1]))
    return (
        int(rec["r"]),
        int(rec["final_c"]),
        seg_sig,
        int(rec["depth"]),
        str(rec["approach"]),
        tuple(rec["route"]),
    )


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    core = np.array(grid, dtype=int)
    opened, sources, meta = _open_from_input(core)

    combo, combo_records, combo_stats = _solve_core(core, "combo", True)
    parallel, parallel_records, _parallel_stats = _solve_core(core, "combo", False)
    static_seq, _static_records, _static_stats = _solve_core(core, "static", True)
    all_left, _left_records, _left_stats = _solve_core(core, "all_left", True)
    all_right, _right_records, _right_stats = _solve_core(core, "all_right", True)

    metrics = dict(combo_stats)
    metrics["potential_multi_exit_segments"] = 0
    for r in range(0, opened.shape[0], 2):
        for seg in _segments_in_row(opened, r, meta["bg"]):
            if len(_exits_below(opened, r, seg, meta["bg"])) >= 2:
                metrics["potential_multi_exit_segments"] += 1

    metrics["sequential_vs_parallel_pixels"] = _diff_pixels(combo, parallel)
    metrics["combo_vs_static_pixels"] = _diff_pixels(combo, static_seq)
    metrics["combo_vs_all_left_pixels"] = _diff_pixels(combo, all_left)
    metrics["combo_vs_all_right_pixels"] = _diff_pixels(combo, all_right)

    route_changes = 0
    pack_changes = 0
    for seq_rec, par_rec in zip(combo_records, parallel_records):
        if _record_signature(seq_rec) != _record_signature(par_rec):
            route_changes += 1
        if int(seq_rec.get("pack_pos", -1)) != int(par_rec.get("pack_pos", -1)):
            pack_changes += 1
    metrics["occupancy_route_changes"] = int(route_changes)
    metrics["occupancy_pack_changes"] = int(pack_changes)
    metrics["source_tokens"] = int(len(sources))
    return {k: int(v) for k, v in metrics.items()}


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
        for center in centers:
            grid[r, center - 1] = BG
            grid[r, center] = OPEN
            grid[r, center + 1] = BG
            grid[r - 1, center] = OPEN
            grid[r + 1, center] = OPEN

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
        for center in centers:
            paste(grid, gate_patch, r, center - 1)

    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _candidate_centers(existing: list[int], adj_rows: list[list[int]], width: int) -> list[int]:
    out: list[int] = []
    for c in range(2, width - 2):
        if any(abs(int(c) - int(other)) < 5 for other in existing):
            continue
        if any(abs(int(c) - int(other)) < 2 for row in adj_rows for other in row):
            continue
        out.append(int(c))
    return out


def _segment_exit_limit_ok(opened: np.ndarray) -> bool:
    for r in range(0, opened.shape[0], 2):
        for seg in _segments_in_row(opened, r, BG):
            if len(_exits_below(opened, r, seg, BG)) > 2:
                return False
    return True


def _build_structure(rng: np.random.Generator) -> tuple[list[list[int]], list[list[int]], int] | None:
    n_walls = int(rng.integers(5, 9))
    width = int(rng.integers(16, 20))
    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    center = int(rng.integers(4, width - 4))
    gates[0] = [center]
    for row in range(1, n_walls):
        options = [center + shift for shift in (-4, -3, -2, 2, 3, 4) if 2 <= center + shift <= width - 3]
        options = [c for c in options if all(abs(int(c) - int(other)) >= 2 for other in gates[row - 1])]
        if row + 1 < n_walls:
            options = [c for c in options if all(abs(int(c) - int(other)) >= 2 for other in gates[row + 1])]
        if not options:
            return None
        center = int(rng.choice(options))
        gates[row] = [center]

    branch_count = int(rng.integers(1, min(3, max(2, n_walls - 1))))
    branch_rows = list(rng.choice(np.arange(1, n_walls - 1), size=branch_count, replace=False))
    for row in branch_rows:
        adj_rows = []
        if row - 1 >= 0:
            adj_rows.append(gates[row - 1])
        if row + 1 < n_walls:
            adj_rows.append(gates[row + 1])
        candidates = _candidate_centers(gates[row], adj_rows, width)
        if candidates:
            gates[row].append(int(rng.choice(candidates)))
            gates[row].sort()

    total_gates = sum(len(row) for row in gates)
    extra_target = int(rng.integers(0, 3))
    while total_gates < min(12, n_walls + 2 + extra_target):
        row = int(rng.integers(0, n_walls))
        if len(gates[row]) >= 3:
            break
        adj_rows = []
        if row - 1 >= 0:
            adj_rows.append(gates[row - 1])
        if row + 1 < n_walls:
            adj_rows.append(gates[row + 1])
        candidates = _candidate_centers(gates[row], adj_rows, width)
        if not candidates:
            break
        gates[row].append(int(rng.choice(candidates)))
        gates[row].sort()
        total_gates += 1

    for pipe_idx in range(1, n_walls + 1):
        if rng.random() >= 0.35:
            continue
        blocked = set()
        if pipe_idx - 1 >= 0:
            blocked.update(gates[pipe_idx - 1])
        if pipe_idx < n_walls:
            blocked.update(gates[pipe_idx])
        candidates = [c for c in range(2, width - 2) if c not in blocked]
        if not candidates:
            continue
        holes[pipe_idx] = [int(rng.choice(candidates))]

    if not (6 <= sum(len(row) for row in gates) <= 12):
        return None

    opened = _build_opened_grid(gates, holes, width)
    if not _segment_exit_limit_ok(opened):
        return None
    return gates, holes, width


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    structure = _build_structure(rng)
    if structure is None:
        return None
    gates, holes, width = structure
    inp = _build_input(gates, holes, width)
    expected, _records, stats = _solve_core(inp, "combo", True)

    if int(stats.get("lost_tokens", 0)) != 0:
        return None
    if int(stats.get("blocked_starts", 0)) != 0:
        return None
    if int(stats.get("start_shifts", 0)) != 0:
        return None
    if int(stats.get("max_depth", 0)) < 3:
        return None
    if int(stats.get("placed_tokens", 0)) != int(stats.get("source_tokens", 0)):
        return None
    if _count_tokens(expected) != int(stats.get("source_tokens", 0)):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    metrics = _behavior_metrics(inp)
    if int(metrics.get("potential_multi_exit_segments", 0)) < 1:
        return None
    if int(metrics.get("multi_exit_steps", 0)) < 2:
        return None
    if int(metrics.get("horizontal_detours", 0)) < 2:
        return None
    if int(metrics.get("approach_updates", 0)) < 1:
        return None
    if int(metrics.get("sequential_vs_parallel_pixels", 0)) <= 0:
        return None
    if int(metrics.get("occupancy_route_changes", 0)) <= 0:
        return None
    if int(metrics.get("combo_vs_static_pixels", 0)) <= 0:
        return None
    if max(int(metrics.get("combo_vs_all_left_pixels", 0)), int(metrics.get("combo_vs_all_right_pixels", 0))) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(1200):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/24")
