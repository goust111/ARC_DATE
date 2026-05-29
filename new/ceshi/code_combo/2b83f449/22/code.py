# Combo variant of 2b83f449 (index 22)
# combo_spec: D9/0+D11/1
# Name: 门对齐约束+逐门处理
#
# Design analysis
# a) Concrete behavior and trace:
#    Original 2b83f449 opens every gate first, then removes all tokens, lets each token follow the
#    unique downward route in the fully opened network, and only packs once at the end. In this
#    combo, adjacent wall-row gates are forced into one near-vertical chain: one gate per wall row
#    and each consecutive center differs by at most 1 column. Example centers:
#      [8, 9, 8, 9, 8, 9]
#    The opened state is a staggered vertical pillar. The crucial D9 change is not "open all,
#    simulate once" but:
#      step 1: open the lowest gate -> settle all tokens in the partially opened network,
#      step 2: open the next higher aligned gate -> settle again from those new packed positions,
#      ...
#    This is the one place where the combo deliberately resolves a tension between the standalone
#    D9 wording and the fused behavior description: to make "newly aligned gates immediately create
#    a long tunnel" physically visible, a newly opened higher gate must plug into an already opened
#    lower shaft. Small center shifts matter because each newly opened post can also split the
#    segment where earlier tokens were packed. In the fully opened solve, an upper token may
#    traverse a lower side-pass before that split ever exists; in the progressive solve, the token
#    is first repacked, then a later aligned gate reshapes its current row. This produces outputs
#    that differ from the base task even though both variants use the same final gate geometry.
# b) Complexity calibration:
#    The generator keeps exactly one gate in every wall row. That makes the alignment constraint
#    obvious to a human: the input shows a single almost-straight stack of gates rather than a
#    cluttered multi-gate lattice. At the same time, the centers are not all identical. At least
#    two row-to-row shifts of +/-1 are required, because a perfectly straight chain often collapses
#    back to the original behavior. This keeps the combo visible without introducing extra object
#    types or hidden metadata.
# c) Hidden-assumption audit of the original main():
#    - Original assumption: opening order does not matter because all tokens are simulated only
#      after the final opened grid is known. Broken here; the solver must store intermediate token
#      positions after each gate.
#    - Original assumption: tokens only need to be packed once. Broken here; packing after each
#      gate is what gives later gates something history-dependent to act on.
#    - Original assumption: adjacent wall rows can be treated independently except for the generic
#      single-exit property. Broken here; the combo depends on a whole top-to-bottom aligned chain.
#    - Original assumption: a gate row may have 0-3 gates. The combo narrows this to exactly one
#      gate per wall row so that the aligned-chain opening order is unambiguous.
# d) Placement legality:
#    The generator uses width 16-19 and 6-9 wall rows, matching the original scale. Every gate
#    center stays in [3, W-4], so the 1x3 bar fits and one-column shifts remain in bounds. Tokens
#    stay at both pipe-row edges exactly as in the original task. No extra objects are added, so
#    there are no overlap risks beyond gate bounds, which are explicitly checked by construction.
# e) Visual observability:
#    Across examples, the human-visible pattern is "one nearly vertical gate spine". The output
#    still shows the same colors and opened posts as the original task, but token piles differ in a
#    way that tracks the progressive opening order. Because the inputs always contain repeated
#    near-aligned gates, the solver can plausibly infer that the vertical chain is the structural
#    cue that interacts with the step-by-step opening.
# f) Visual cleanliness:
#    The combo avoids extra markers, legends, or recolors. The input remains the same family of
#    pipe rows, wall rows, gates, and edge tokens; the only change is a stricter gate layout. That
#    keeps the grid readable and makes the gate chain the dominant visual structure.
# g) Change inferability:
#    There is no new color map. Only spatial alignment and processing order change. Because the
#    gate chain repeats across many rows, a human gets repeated evidence that adjacent gates belong
#    to one vertical mechanism rather than unrelated local doors.
# 9) New assumptions introduced by this fused main():
#    - Every wall row contains exactly one 1x3 gate bar.
#    - Consecutive gate centers differ by at most 1 column.
#    - After any prefix of the gate chain is opened, each passable segment still has at most one
#      downward exit and at most one upward entry.
#    - Tokens are repacked after every gate-opening step, not only at the end.
#    - The progressive order follows the aligned shaft from bottom to top so that a newly opened
#      higher gate can connect into an already opened lower tunnel and enable multi-level falls.
#    generate_input() enforces all of these assumptions and rejects any sample whose combo output
#    matches the original simultaneous-opening output.

from __future__ import annotations

from collections import defaultdict

import numpy as np


WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _get_gate_centers(grid: np.ndarray) -> list[int]:
    g = np.array(grid, dtype=int)
    height, width = g.shape
    assert height % 2 == 1
    n_walls = (height - 1) // 2
    centers: list[int] = []
    for wall_idx in range(n_walls):
        wr = 2 * wall_idx + 1
        row = g[wr]
        found: list[int] = []
        c = 1
        while c < width - 1:
            if int(row[c - 1]) == GATE and int(row[c]) == GATE and int(row[c + 1]) == GATE:
                found.append(c)
                c += 3
            else:
                c += 1
        assert len(found) == 1, f"expected exactly one gate on wall row {wr}, got {found}"
        centers.append(int(found[0]))
    for a, b in zip(centers, centers[1:]):
        assert abs(int(a) - int(b)) <= 1, f"adjacent gate centers must align: {centers}"
    return centers


def _open_gate_in_place(state: np.ndarray, wall_idx: int, center: int) -> None:
    wr = 2 * int(wall_idx) + 1
    state[wr, center - 1 : center + 2] = BG
    state[wr - 1, center] = OPEN
    state[wr, center] = OPEN
    state[wr + 1, center] = OPEN


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
        segs.append((int(start), int(c)))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, int(r)):
        if int(seg[0]) <= int(c) <= int(seg[1]):
            return seg
    return None


def _single_exit_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> int | None:
    if int(r) + 2 >= grid.shape[0]:
        return None
    a, b = seg
    exits = [c for c in range(int(a), int(b) + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]
    assert len(exits) <= 1, f"segment {seg} at row {r} has multiple exits {exits}"
    return int(exits[0]) if exits else None


def _single_entry_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> int | None:
    if int(r) < 2:
        return None
    a, b = seg
    entries = [c for c in range(int(a), int(b) + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]
    assert len(entries) <= 1, f"segment {seg} at row {r} has multiple entries {entries}"
    return int(entries[0]) if entries else None


def _pack_positions(
    opened: np.ndarray,
    finals: list[tuple[int, int, int]],
) -> tuple[np.ndarray, dict[str, int]]:
    out = opened.copy()
    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)
    moved_tokens = 0
    max_depth = 0
    for r, c, depth in finals:
        seg = _find_segment(opened, int(r), int(c))
        if seg is None:
            continue
        entry = _single_entry_above(opened, int(r), seg)
        a, b = seg
        if entry is not None:
            pack_end = int(b) if (int(entry) - int(a)) <= (int(b) - int(entry)) else int(a)
        else:
            pack_end = int(a) if abs(int(c) - int(a)) <= abs(int(c) - int(b)) else int(b)
        buckets[(int(r), seg, pack_end)] += 1
        if int(depth) > 0:
            moved_tokens += 1
        max_depth = max(int(max_depth), int(depth))

    for (r, (a, b), pack_end), count in buckets.items():
        if int(pack_end) == int(a):
            for i in range(int(count)):
                if int(a) + i <= int(b):
                    out[int(r), int(a) + i] = TOKEN
        else:
            for i in range(int(count)):
                if int(b) - i >= int(a):
                    out[int(r), int(b) - i] = TOKEN

    return out, {"moved_tokens": int(moved_tokens), "max_depth": int(max_depth)}


def _settle_tokens(state: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    opened = np.array(state, dtype=int)
    token_pts = [(int(r), int(c)) for r, c in np.argwhere(opened == TOKEN)]
    opened[opened == TOKEN] = BG

    finals: list[tuple[int, int, int]] = []
    for start_r, start_c in token_pts:
        r, c = int(start_r), int(start_c)
        depth = 0
        while True:
            seg = _find_segment(opened, int(r), int(c))
            if seg is None:
                break
            exit_col = _single_exit_below(opened, int(r), seg)
            if exit_col is None:
                break
            r += 2
            c = int(exit_col)
            depth += 1
        finals.append((int(r), int(c), int(depth)))

    return _pack_positions(opened, finals)


def _solve_combo_core(grid: np.ndarray, *, with_trace: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    state = np.array(grid, dtype=int)
    centers = _get_gate_centers(state)
    moved_per_step: list[int] = []
    depth_per_step: list[int] = []

    for wall_idx in range(len(centers) - 1, -1, -1):
        center = centers[wall_idx]
        _open_gate_in_place(state, int(wall_idx), int(center))
        state, step_stats = _settle_tokens(state)
        moved_per_step.append(int(step_stats["moved_tokens"]))
        depth_per_step.append(int(step_stats["max_depth"]))

    if with_trace:
        trace = {
            "centers": [int(c) for c in centers],
            "moved_per_step": moved_per_step,
            "max_depth_per_step": depth_per_step,
        }
        return state, trace
    return state


def _solve_base(grid: np.ndarray) -> np.ndarray:
    state = np.array(grid, dtype=int)
    for wall_idx, center in enumerate(_get_gate_centers(state)):
        _open_gate_in_place(state, int(wall_idx), int(center))
    state, _ = _settle_tokens(state)
    return state


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Detect the single near-aligned gate chain.
    2. Open gates one by one along the aligned chain from bottom to top.
    3. After each single gate opens, settle all current tokens in the partially opened network.
    4. Return the fully opened grid after the last progressive settling step.
    """
    return _solve_combo_core(np.array(grid, dtype=int), with_trace=False)


def _build_input(centers: list[int], width: int) -> np.ndarray:
    n_walls = len(centers)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        grid[2 * pipe_idx, :] = BG
    for wall_idx, center in enumerate(centers):
        wr = 2 * wall_idx + 1
        grid[wr, center - 1 : center + 2] = GATE
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN
    return grid


def _sample_centers(rng: np.random.Generator, n_walls: int, width: int) -> list[int] | None:
    start = int(rng.integers(4, width - 4))
    centers = [start]
    nonzero_shifts = 0
    for _ in range(1, n_walls):
        step = int(rng.choice(np.array([-1, 0, 1], dtype=int), p=np.array([0.35, 0.30, 0.35])))
        nxt = int(centers[-1] + step)
        if not (3 <= nxt <= width - 4):
            return None
        centers.append(nxt)
        if step != 0:
            nonzero_shifts += 1
    if nonzero_shifts < 2:
        return None
    if len(set(centers)) < 2:
        return None
    return centers


def combo_behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    combo_out, trace = _solve_combo_core(np.array(grid, dtype=int), with_trace=True)
    base_out = _solve_base(np.array(grid, dtype=int))
    centers = [int(c) for c in trace["centers"]]
    deltas = [abs(int(b) - int(a)) for a, b in zip(centers, centers[1:])]
    return {
        "gate_count": len(centers),
        "nonzero_shifts": sum(1 for d in deltas if d > 0),
        "output_diff_cells": int(np.sum(combo_out != base_out)),
        "active_steps": sum(1 for x in trace["moved_per_step"] if int(x) > 0),
        "max_step_drop": max([0] + [int(x) for x in trace["max_depth_per_step"]]),
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(600):
        n_walls = int(rng.integers(6, 10))
        width = int(rng.integers(16, 20))
        centers = _sample_centers(rng, int(n_walls), int(width))
        if centers is None:
            continue

        inp = _build_input(centers, int(width))
        combo_out, trace = _solve_combo_core(inp, with_trace=True)
        base_out = _solve_base(inp)
        if np.array_equal(combo_out, base_out):
            continue

        metrics = combo_behavior_metrics(inp)
        if int(metrics["output_diff_cells"]) < 4:
            continue
        if int(metrics["active_steps"]) < max(3, int(n_walls) - 2):
            continue
        if int(metrics["nonzero_shifts"]) < 2:
            continue
        if int(metrics["max_step_drop"]) < 2:
            continue

        # Sanity-check that the public solver reproduces the internally accepted sample.
        if not np.array_equal(main(inp), combo_out):
            continue

        # Reject visually weak traces where only the first step moves anything.
        moved_tail = [int(x) for x in trace["moved_per_step"][1:]]
        if sum(1 for x in moved_tail if x > 0) < 2:
            continue

        return inp

    raise RuntimeError("failed to generate input for combo variant 2b83f449/22")
