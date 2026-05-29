# Combo variant of 2b83f449 (index 15)
# combo_spec: D7/0+D9/0
# Name: 多出口+逐门处理

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask


WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


# Design analysis for the combo:
# - Concrete behavior:
#   Start with the original alternating pipe/wall lattice. A gate is still a 1x3
#   horizontal bar on a wall row. When a gate opens, it becomes a 3-tall post and
#   exposes its two flanking cells as downward exits. Unlike the base task, a pipe
#   segment may now have two exits. The choice rule is local and visible: a token
#   sitting on the left side of a segment takes the leftmost exit; one sitting on
#   the right side takes the rightmost exit.
# - Row-batched interpretation of D9:
#   The reference D9 variant already uses top-to-bottom staged opening by wall
#   level. For the combo, that calibration is important: if same-row gates were
#   opened in an arbitrary left/right order, the first one often drains the token
#   before the second can create a visible two-exit choice. So the solver opens all
#   gates on one wall row together, then re-simulates, then proceeds downward.
# - Why the D9 change cannot be bolted on mechanically:
#   If we simply "open one gate at a time" but keep the old all-at-once settling
#   semantics, many instances collapse to the same result as opening everything
#   first. To make the order effect real, each newly opened gate triggers a full
#   re-simulation on the currently open network, and the tokens are re-packed on
#   the segment ends before the next gate opens. That packed position is visible in
#   the grid and becomes the side cue for the next multi-exit choice.
# - Small example:
#   Suppose the top token drops through a newly opened upper gate and stops on the
#   next pipe row because lower gates are still closed. It then packs to one end of
#   that segment. Later, a lower gate opens and creates two exits under that same
#   segment. The token now chooses the exit on the side where it is already sitting.
#   If all gates had opened at once, that same token would have chosen based on its
#   pre-packed transit column instead, so the final distribution can differ.
# - Complexity / observability calibration:
#   The generator keeps the original visual language: only four colors in input,
#   sparse horizontal gate bars, and a few long pipe rows. It requires at least one
#   late-stage multi-exit traversal and rejects samples whose combo output matches
#   the "all gates open immediately" baseline, so the fused behavior is not hidden.
# - Implicit assumptions changed from the base solver:
#   1. A segment may have more than one exit below, so old len(exits)<=1 asserts are
#      invalid.
#   2. Token routing can no longer depend on a hidden original side tag; otherwise
#      progressive opening barely matters. The solver therefore uses the token's
#      current segment-side, which is visible after every intermediate packing step.
#   3. Packing is no longer a single final pass. It happens after each gate-opening
#      stage, because those packed locations affect later exit choices.
# - Generator legality:
#   Gates stay interior, same-row centers are spaced apart, adjacent wall rows avoid
#   near-overlaps, and holes never overwrite gate posts. Inputs are accepted only if
#   `main(input)` reproduces the staged output and token count is preserved.


def _infer_gate_color(grid: np.ndarray) -> int:
    colors = sorted(set(map(int, grid.flatten())))
    for col in colors:
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
                return int(col)
    raise AssertionError("gate color not found")


def _infer_token_bg(grid: np.ndarray, gate: int) -> tuple[int, int]:
    colors = set(map(int, grid.flatten()))
    counts = Counter(map(int, grid.flatten()))
    token_cands = [c for c in colors if c not in (WALL, gate)]
    assert token_cands
    token = int(min(token_cands, key=lambda c: (int(counts[c]), int(c))))
    bg_cands = [c for c in colors if c not in (WALL, gate, token)]
    assert bg_cands
    bg = int(max(bg_cands, key=lambda c: (int(counts[c]), -int(c))))
    return token, bg


def _open_color(gate: int, colors: set[int]) -> int:
    if gate == 7 and 6 not in colors:
        return 6
    return int(gate - 1) if gate > 0 else 6


def _parse_gate_centers(grid: np.ndarray, gate: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    height, width = grid.shape
    for r in range(1, height, 2):
        for c in range(1, width - 1):
            if int(grid[r, c - 1]) == gate and int(grid[r, c]) == gate and int(grid[r, c + 1]) == gate:
                out.append((r, c))
    out.sort(key=lambda rc: (rc[0], rc[1]))
    return out


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    out: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != bg:
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == bg:
            c += 1
        out.append((start, c))
        c += 1
    return out


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int] | None, bg: int) -> list[int]:
    if seg is None or r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == bg and int(grid[r + 2, c]) == bg]


def _choose_exit(cur_c: int, seg: tuple[int, int] | None, exits: list[int]) -> int | None:
    if seg is None or not exits:
        return None
    if len(exits) == 1:
        return int(exits[0])
    a, b = seg
    return int(min(exits) if cur_c <= (a + b) // 2 else max(exits))


def _pack_tokens(
    states: list[dict[str, int | None]],
    opened_grid: np.ndarray,
    bg: int,
) -> list[dict[str, int | None]]:
    buckets: dict[tuple[int, tuple[int, int]], list[dict[str, int | None]]] = defaultdict(list)
    loose: list[dict[str, int | None]] = []
    for st in states:
        seg = _find_segment(opened_grid, int(st["r"]), int(st["c"]), bg)
        if seg is None:
            loose.append({"id": int(st["id"]), "r": int(st["r"]), "c": int(st["c"]), "entry": None})
            continue
        buckets[(int(st["r"]), seg)].append(st)

    packed: list[dict[str, int | None]] = []
    for (r, (a, b)), group in buckets.items():
        left_pack: list[dict[str, int | None]] = []
        right_pack: list[dict[str, int | None]] = []
        for st in group:
            anchor = int(st["entry"]) if st["entry"] is not None else int(st["c"])
            if anchor <= (a + b) // 2:
                right_pack.append(st)
            else:
                left_pack.append(st)
        left_pack.sort(key=lambda st: (int(st["c"]), int(st["id"])))
        right_pack.sort(key=lambda st: (-int(st["c"]), int(st["id"])))
        for i, st in enumerate(left_pack):
            if a + i <= b:
                packed.append({"id": int(st["id"]), "r": r, "c": a + i, "entry": None})
        for i, st in enumerate(right_pack):
            if b - i >= a:
                packed.append({"id": int(st["id"]), "r": r, "c": b - i, "entry": None})

    packed.extend(loose)
    packed.sort(key=lambda st: int(st["id"]))
    return packed


def _simulate_stage(
    opened_grid: np.ndarray,
    states: list[dict[str, int | None]],
    bg: int,
    stage_idx: int,
    stats: dict[str, int] | None = None,
) -> list[dict[str, int | None]]:
    moved: list[dict[str, int | None]] = []
    for st in states:
        r = int(st["r"])
        c = int(st["c"])
        entry: int | None = None
        while True:
            seg = _find_segment(opened_grid, r, c, bg)
            exits = _exits_below(opened_grid, r, seg, bg)
            if stats is not None and len(exits) >= 2:
                stats["multi_exit_seen"] += 1
            ex = _choose_exit(c, seg, exits)
            if ex is None:
                break
            if stats is not None:
                stats["total_drops"] += 1
                if len(exits) >= 2 and stage_idx > 0:
                    stats["late_multi_exit_uses"] += 1
            r += 2
            c = ex
            entry = ex
        moved.append({"id": int(st["id"]), "r": r, "c": c, "entry": entry})
    return _pack_tokens(moved, opened_grid, bg)


def _open_gate_inplace(grid: np.ndarray, wall_r: int, center_c: int, bg: int, open_col: int) -> None:
    grid[wall_r, center_c - 1 : center_c + 2] = bg
    for rr in (wall_r - 1, wall_r, wall_r + 1):
        if 0 <= rr < grid.shape[0]:
            grid[rr, center_c] = open_col


def _render_tokens(base: np.ndarray, states: list[dict[str, int | None]], token: int) -> np.ndarray:
    out = base.copy()
    for st in states:
        out[int(st["r"]), int(st["c"])] = token
    return out


def _parse_states_from_input(grid: np.ndarray, token: int) -> list[dict[str, int | None]]:
    states: list[dict[str, int | None]] = []
    idx = 0
    for r, c in np.argwhere(grid == token):
        states.append({"id": idx, "r": int(r), "c": int(c), "entry": None})
        idx += 1
    states.sort(key=lambda st: (int(st["r"]), int(st["c"]), int(st["id"])))
    for idx, st in enumerate(states):
        st["id"] = idx
    return states


def _row_batches(gate_centers: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    grouped: list[list[tuple[int, int]]] = []
    current_row: int | None = None
    for wall_r, center_c in sorted(gate_centers, key=lambda rc: (rc[0], rc[1])):
        if current_row != wall_r:
            grouped.append([])
            current_row = wall_r
        grouped[-1].append((wall_r, center_c))
    return grouped


def _simulate_combo_from_input(grid: np.ndarray, *, collect_stats: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))
    gate = _infer_gate_color(g)
    token, bg = _infer_token_bg(g, gate)
    open_col = _open_color(gate, colors)
    gate_centers = _parse_gate_centers(g, gate)

    base = g.copy()
    base[base == token] = bg
    states = _parse_states_from_input(g, token)
    stats = {"total_drops": 0, "multi_exit_seen": 0, "late_multi_exit_uses": 0}

    for stage_idx, batch in enumerate(_row_batches(gate_centers)):
        for wall_r, center_c in batch:
            _open_gate_inplace(base, wall_r, center_c, bg, open_col)
        states = _simulate_stage(base, states, bg, stage_idx, stats if collect_stats else None)

    return _render_tokens(base, states, token), stats


def main(grid: np.ndarray) -> np.ndarray:
    out, _stats = _simulate_combo_from_input(np.array(grid, dtype=int), collect_stats=False)
    return out


def _build_input_grid(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL
    for wall_idx, row in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center in row:
            grid[wr, center - 1 : center + 2] = GATE
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN
    return grid


def _build_opened_grid(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL
    for wall_idx, row in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center in row:
            grid[wr, center - 1] = BG
            grid[wr, center] = OPEN
            grid[wr, center + 1] = BG
            grid[wr - 1, center] = OPEN
            grid[wr + 1, center] = OPEN
    return grid


def _initial_states(n_walls: int, width: int) -> list[dict[str, int | None]]:
    states: list[dict[str, int | None]] = []
    idx = 0
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        states.append({"id": idx, "r": r, "c": 0, "entry": None})
        idx += 1
        states.append({"id": idx, "r": r, "c": width - 1, "entry": None})
        idx += 1
    return states


def _simulate_progressive_network(
    gates: list[list[int]],
    holes: list[list[int]],
    width: int,
    *,
    collect_stats: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    n_walls = len(gates)
    base = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        base[r, :] = BG
        for c in holes[pipe_idx]:
            base[r, c] = WALL
    states = _initial_states(n_walls, width)
    stats = {"total_drops": 0, "multi_exit_seen": 0, "late_multi_exit_uses": 0}
    order = [(2 * wall_idx + 1, center) for wall_idx, row in enumerate(gates) for center in sorted(row)]
    for stage_idx, batch in enumerate(_row_batches(order)):
        for wall_r, center_c in batch:
            _open_gate_inplace(base, wall_r, center_c, BG, OPEN)
        states = _simulate_stage(base, states, BG, stage_idx, stats if collect_stats else None)
    return _render_tokens(base, states, TOKEN), stats


def _simulate_all_open_network(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    opened = _build_opened_grid(gates, holes, width)
    states = _simulate_stage(opened, _initial_states(len(gates), width), BG, 0, None)
    return _render_tokens(opened, states, TOKEN)


def _build_candidate_structure(rng: np.random.Generator) -> tuple[list[list[int]], list[list[int]], int] | None:
    n_walls = int(rng.integers(5, 8))
    width = int(rng.integers(16, 20))
    pivot = int(rng.integers(2, n_walls - 1))
    mid_low = max(5, n_walls)
    mid_high = width - 8
    if mid_low > mid_high:
        return None
    mid = int(rng.integers(mid_low, mid_high + 1))
    gates = [[mid] for _ in range(n_walls)]
    gates[pivot] = [mid, mid + 5]
    if bool(rng.integers(2)):
        mirrored = [sorted([width - 1 - c for c in row]) for row in gates]
        if all(all(2 <= c <= width - 3 for c in row) for row in mirrored):
            gates = mirrored
    holes = [[] for _ in range(n_walls + 1)]
    return gates, holes, width


def _quality_metrics(gates: list[list[int]], holes: list[list[int]], width: int) -> dict[str, int]:
    combo_out, combo_stats = _simulate_progressive_network(gates, holes, width, collect_stats=True)
    base_out = _simulate_all_open_network(gates, holes, width)
    return {
        "diff_cells": int(np.sum(combo_out != base_out)),
        "late_multi_exit_uses": int(combo_stats["late_multi_exit_uses"]),
        "multi_exit_seen": int(combo_stats["multi_exit_seen"]),
        "total_drops": int(combo_stats["total_drops"]),
        "token_count": int(np.sum(combo_out == TOKEN)),
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        built = _build_candidate_structure(rng)
        if built is None:
            continue
        gates, holes, width = built
        inp = _build_input_grid(gates, holes, width)
        out_expected, stats = _simulate_progressive_network(gates, holes, width, collect_stats=True)
        if int(np.sum(out_expected == TOKEN)) != 2 * len(gates):
            continue
        baseline = _simulate_all_open_network(gates, holes, width)
        if int(np.sum(out_expected != baseline)) < 2:
            continue
        if int(stats["late_multi_exit_uses"]) < 1:
            continue
        if int(stats["multi_exit_seen"]) < 1:
            continue
        if int(stats["total_drops"]) < 6:
            continue
        if not np.array_equal(main(inp), out_expected):
            continue
        return inp
    raise RuntimeError("failed to generate input for combo 2b83f449/15")
