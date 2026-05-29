# Combo variant of 2b83f449 (index 8)
# combo_spec: D3/2+D7/0
# Name: 惯性移动+多出口选择
#
# Design analysis
# ---------------
# a) Concrete fused behavior:
#    The base task has alternating pipe rows and wall rows. A 1x3 horizontal gate opens into a
#    vertical 3-cell post, and tokens originally only needed to follow a unique downward exit.
#    This fusion breaks that hidden uniqueness. A pipe segment may now have two exits below.
#    Tokens still try to continue straight downward first. If the column directly below is blocked,
#    the token must slide horizontally inside the current segment to find a usable exit. Which exit
#    it chooses depends on the side it entered from: left-entry uses the leftmost exit, right-entry
#    uses the rightmost exit. The key coupling is that a horizontal detour updates the entry side
#    seen by the next segment.
#    Example sketch with exits [4, 9]:
#      - token arrives at c=6 from the left side
#      - base solver would reject because the segment has multiple exits
#      - fused solver picks exit 4, drops there, and a later segment now sees the token as coming
#        from the right if that detour moved left.
#
# b) Complexity calibration:
#    The generator requires both horizontal detours and multi-exit decisions, so the task does not
#    collapse back to the original single-exit behavior. The change is visible from geometry alone.
#
# c) Broken implicit assumptions from the original solver:
#    - "Each segment has <=1 exit below" is invalid.
#    - "Final stacking can be inferred from a unique entry above" is invalid.
#    - The solver must therefore track per-token approach side through the whole simulation.
#
# d) Placement legality:
#    Gates remain interior, holes remain interior, and output tokens are placed only after routing on
#    the opened grid. The generator rejects layouts where final packing would lose tokens.
#
# e) Visual observability:
#    Opposite-side tokens can diverge inside the same chamber, and lower branching can change after
#    earlier horizontal detours. Those repeated geometric cues are visible across examples.
#
# f) Visual cleanliness:
#    The fusion adds no extra marker colors; it reuses the original palette and only changes motion.
#
# g) Anti-degeneration metric:
#    `_behavior_metrics()` counts horizontal detours, multi-exit steps, approach-side updates, and
#    pixel differences against ablations that freeze the side rule. Generated samples are accepted
#    only when those metrics prove the fused control logic is active.

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


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))

    gate = None
    for col in sorted(colors):
        if col == WALL:
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
    out = g.copy()
    height, width = g.shape

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
        center = left + 1
        out[r, left : left + 3] = int(bg)
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < height:
                out[rr, center] = int(open_col)

    sources: list[dict[str, int]] = []
    order = 0
    for r, c in np.argwhere(g == int(token)):
        rr = int(r)
        cc = int(c)
        if cc <= 1:
            side = "left"
        elif cc >= width - 2:
            side = "right"
        else:
            side = "left" if cc < width // 2 else "right"
        sources.append({"r": rr, "c": cc, "pipe_idx": rr // 2, "order": order, "side_code": 0 if side == "left" else 1})
        out[rr, cc] = int(bg)
        order += 1

    meta = {"gate": int(gate), "token": int(token), "bg": int(bg), "open": int(open_col)}
    return out, sources, meta


def _initial_side(source: dict[str, int], mode: str) -> str:
    if mode == "all_left":
        return "left"
    if mode == "all_right":
        return "right"
    return "left" if int(source["side_code"]) == 0 else "right"


def _simulate_records(
    opened: np.ndarray,
    sources: list[dict[str, int]],
    bg: int,
    mode: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    records: list[dict[str, object]] = []
    stats = defaultdict(int)

    for source in sorted(sources, key=lambda item: (int(item["r"]), int(item["c"]), int(item["order"]))):
        r = int(source["r"])
        c = int(source["c"])
        source_side = _initial_side(source, mode)
        approach = source_side
        depth = 0

        while True:
            seg = _find_segment(opened, r, c, bg)
            if seg is None:
                break
            exits = _exits_below(opened, r, seg, bg)
            if not exits:
                break
            if len(exits) >= 2:
                stats["multi_exit_steps"] += 1
            new_approach = approach
            if c in exits:
                chosen = c
            else:
                chosen = min(exits) if approach == "left" else max(exits)
                stats["horizontal_detours"] += 1
                stats["horizontal_distance"] += abs(chosen - c)
                if mode == "combo":
                    if chosen > c:
                        new_approach = "left"
                    elif chosen < c:
                        new_approach = "right"
                    if new_approach != approach:
                        stats["approach_updates"] += 1
            r += 2
            c = chosen
            depth += 1
            if mode == "combo":
                approach = new_approach
            if r >= opened.shape[0]:
                break

        final_seg = _find_segment(opened, r, c, bg)
        if final_seg is None:
            final_seg = (c, c)
        a, b = final_seg
        if depth == 0:
            pack_end = a if source_side == "left" else b
        else:
            pack_end = b if approach == "left" else a
        records.append(
            {
                "r": r,
                "seg": final_seg,
                "final_c": c,
                "pack_end": pack_end,
                "depth": depth,
                "approach": approach,
                "source_side": source_side,
                "source": source,
            }
        )
        stats["max_depth"] = max(int(stats["max_depth"]), depth)

    return records, {k: int(v) for k, v in stats.items()}


def _render_output(
    opened: np.ndarray,
    sources: list[dict[str, int]],
    token: int,
    bg: int,
    mode: str,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.array(opened, copy=True)
    records, stats = _simulate_records(opened, sources, bg, mode)

    per_segment: dict[tuple[int, tuple[int, int]], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {"left": [], "right": []}
    )
    for rec in records:
        seg = rec["seg"]
        r = int(rec["r"])
        pack_end = int(rec["pack_end"])
        if pack_end == int(seg[0]):
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

        for idx, _rec in enumerate(left_recs):
            pos = a + idx
            if pos > b or pos in used:
                continue
            out[r, pos] = int(token)
            used.add(pos)
            placed += 1
        for idx, _rec in enumerate(right_recs):
            pos = b - idx
            if pos < a or pos in used:
                continue
            out[r, pos] = int(token)
            used.add(pos)
            placed += 1

    stats["placed_tokens"] = int(placed)
    stats["source_tokens"] = int(len(sources))
    return out, stats


def _solve_core(grid: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, int]]:
    opened, sources, meta = _open_from_input(grid)
    out, stats = _render_output(opened, sources, meta["token"], meta["bg"], mode)
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    out, _stats = _solve_core(np.array(grid, dtype=int), "combo")
    return out


def _solve_with_mode(grid: np.ndarray, mode: str) -> np.ndarray:
    out, _stats = _solve_core(np.array(grid, dtype=int), mode)
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


def _count_tokens(grid: np.ndarray) -> int:
    return int(np.count_nonzero(np.array(grid, dtype=int) == TOKEN))


def _diff_pixels(a: np.ndarray, b: np.ndarray) -> int:
    aa = np.array(a, dtype=int)
    bb = np.array(b, dtype=int)
    return int(np.count_nonzero(aa != bb))


def _potential_multi_exit_segments(opened: np.ndarray, bg: int) -> int:
    total = 0
    for r in range(0, opened.shape[0], 2):
        for seg in _segments_in_row(opened, r, bg):
            if len(_exits_below(opened, r, seg, bg)) >= 2:
                total += 1
    return int(total)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    core = np.array(grid, dtype=int)
    opened, _sources, meta = _open_from_input(core)
    combo, combo_stats = _solve_core(core, "combo")
    static, _ = _solve_core(np.array(grid, dtype=int), "static")
    all_left, _ = _solve_core(np.array(grid, dtype=int), "all_left")
    all_right, _ = _solve_core(np.array(grid, dtype=int), "all_right")
    metrics = dict(combo_stats)
    metrics["potential_multi_exit_segments"] = _potential_multi_exit_segments(opened, meta["bg"])
    metrics["combo_vs_static_pixels"] = _diff_pixels(combo, static)
    metrics["combo_vs_all_left_pixels"] = _diff_pixels(combo, all_left)
    metrics["combo_vs_all_right_pixels"] = _diff_pixels(combo, all_right)
    return metrics


def _candidate_centers(existing: list[int], adj_rows: list[list[int]], width: int) -> list[int]:
    out: list[int] = []
    for c in range(2, width - 2):
        if any(abs(c - other) < 5 for other in existing):
            continue
        if any(abs(c - other) < 2 for row in adj_rows for other in row):
            continue
        out.append(c)
    return out


def _build_structure(rng: np.random.Generator) -> tuple[list[list[int]], list[list[int]], int] | None:
    n_walls = int(rng.integers(5, 9))
    width = int(rng.integers(16, 20))
    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    center = int(rng.integers(4, width - 4))
    gates[0] = [center]
    for row in range(1, n_walls):
        options = [center + shift for shift in (-4, -3, -2, 2, 3, 4) if 2 <= center + shift <= width - 3]
        options = [c for c in options if all(abs(c - other) >= 2 for other in gates[row - 1])]
        if row + 1 < n_walls:
            options = [c for c in options if all(abs(c - other) >= 2 for other in gates[row + 1])]
        if not options:
            return None
        center = int(rng.choice(options))
        gates[row] = [center]

    branch_rows = list(rng.choice(np.arange(1, n_walls - 1), size=int(rng.integers(1, 3)), replace=False))
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
        if rng.random() >= 0.3:
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

    if not (6 <= total_gates <= 12):
        return None
    return gates, holes, width


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    structure = _build_structure(rng)
    if structure is None:
        return None
    gates, holes, width = structure

    opened = _build_opened_grid(gates, holes, width)
    dummy_sources = []
    order = 0
    for pipe_idx in range((opened.shape[0] - 1) // 2):
        r = 2 * pipe_idx
        dummy_sources.append({"r": r, "c": 0, "pipe_idx": pipe_idx, "order": order, "side_code": 0})
        order += 1
        dummy_sources.append({"r": r, "c": width - 1, "pipe_idx": pipe_idx, "order": order, "side_code": 1})
        order += 1

    expected, stats = _render_output(opened, dummy_sources, TOKEN, BG, "combo")
    if int(stats["placed_tokens"]) != int(stats["source_tokens"]):
        return None
    if int(stats.get("max_depth", 0)) < 3:
        return None

    inp = _build_input(gates, holes, width)
    if not np.array_equal(main(inp), expected):
        return None

    metrics = _behavior_metrics(inp)
    if int(metrics.get("horizontal_detours", 0)) < 3:
        return None
    if int(metrics.get("potential_multi_exit_segments", 0)) < 1:
        return None
    if int(metrics.get("approach_updates", 0)) < 1:
        return None
    if int(metrics.get("combo_vs_static_pixels", 0)) <= 0:
        return None
    if max(int(metrics.get("combo_vs_all_left_pixels", 0)), int(metrics.get("combo_vs_all_right_pixels", 0))) <= 0:
        return None
    if _count_tokens(expected) != int(stats["source_tokens"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/8")
