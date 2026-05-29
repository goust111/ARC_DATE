# Variant of 45a5af55
# Dimension: D13 — 输入色条间的分隔与干扰 (src)
# Change: 色条之间直接相邻，无分隔行，无干扰元素... → 输入中混入不参与构建的干扰色条

from __future__ import annotations

import numpy as np

# Design analysis for this D13 variant
# ===================================
# a) Concrete behavior and trace
# - The input is still a stack of full-width horizontal stripes, each stripe being
#   a solid color with height 1-4.
# - The new visible change is that some colors appear twice in the input stack.
#   Only the first occurrence seen when scanning bottom -> top is real; later
#   occurrences of the same color are distractor stripes and must be ignored.
# - Example, bottom -> top runs:
#   original logic: [(2,2), (5,1), (8,3)] -> build 2-center, then 5 frame, then 8 frame.
#   variant input: [(2,2), (5,1), (8,3), (5,2)].
#   old solver would wrongly add another outer 5-frame.
#   new logic filters to [(2,2), (5,1), (8,3)] because the second 5 is noise.
# - The change only affects the "which stripes participate" stage; the square
#   construction rule itself stays unchanged.
#
# b) Complexity calibration
# - Repeated colors are not enough by themselves because the original task can
#   already contain repeated colors as meaningful layers. The variant therefore
#   tightens the rule to a fixed, consistent policy:
#   "for any repeated color, keep only its first bottom->top occurrence".
# - Generator enforces 1-3 distractor stripes per sample so the difference is
#   usually visible and not a rare corner case.
#
# c) Audit of original implicit assumptions
# - `col = g[r, 0]`: assumes each row belongs to a full-width solid stripe.
#   Still valid; generator preserves full-width stripes.
# - `while ... (g[rr] == col).all()`: assumes equal-colored adjacent rows belong
#   to the same stripe. Still valid; generator never places two same-colored
#   stripes adjacently, so real duplicates do not merge accidentally.
# - `runs.append((col, h))` for every parsed stripe: this is the key old
#   assumption that every stripe participates. No longer valid; we now filter
#   repeated colors after parsing.
# - The nested-square construction loop itself only depends on an ordered list
#   of participating runs and remains correct once distractors are removed.
#
# d) Placement legality
# - A distractor stripe with color C is inserted strictly above the real C stripe
#   and at least one different active stripe away from it, so it remains a
#   separate run instead of merging with the real stripe.
# - Input height and output size are both capped at 30.
# - Every generated input is validated by comparing `main(input)` against the
#   output built directly from the active stripes.
#
# e) Visual observability
# - Each sample has at least one repeated color in the input but only one layer
#   of that color in the output. Across samples, duplicate stripes appear in
#   different positions and with different heights, making the "ignore later
#   repeats" rule observable from few examples.
#
# f) Visual cleanliness
# - No extra symbols or partial-width objects are introduced; the grid remains a
#   clean stack of solid stripes, so the only new signal is repeated-color noise.
#
# g) Color inferability
# - Colors are still drawn from a small ARC palette (1-9), with active colors
#   unique before duplication. The repeated-color relation itself is the clue;
#   there is no hidden recoloring rule.
#
# 9) New assumptions introduced by `main()`
# - If a color repeats, later copies are distractors rather than additional real
#   layers. Generator guarantees this by creating each active color once and only
#   adding optional higher duplicate stripes as noise.
# - Solver assumes the first bottom->top occurrence is the one to keep.
#   Generator guarantees that every duplicate is placed above its real stripe.


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width horizontal stripe runs from bottom to top."""
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        col = int(g[r, 0])
        run_h = 1
        rr = r - 1
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr
    return runs


def _build_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Build the nested squares from participating runs listed bottom -> top."""
    assert runs
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, h in runs[1:]:
        new_n = cur.shape[0] + 2 * h
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stripe stack with repeated-color distractors.

    Variant rule:
    - Parse stripes from bottom to top exactly as in the original task.
    - If a color appears multiple times, only its first bottom->top occurrence
      participates in the construction; later same-color stripes are distractors.
    - The remaining participating stripes are converted to nested square frames
      using the original wrap-by-height rule.
    """
    parsed_runs = _parse_runs(np.array(grid, dtype=int))

    active_runs: list[tuple[int, int]] = []
    seen: set[int] = set()
    for col, h in parsed_runs:
        if col in seen:
            continue
        seen.add(col)
        active_runs.append((col, h))

    return _build_output(active_runs)


def _gen_attempt() -> np.ndarray | None:
    n_active = int(np.random.randint(3, 7))
    active_heights = [int(np.random.randint(1, 5)) for _ in range(n_active)]
    active_cols = list(np.random.permutation(np.arange(1, 10))[:n_active].astype(int))
    active_runs = list(zip(active_cols, active_heights))  # bottom -> top

    output = _build_output(active_runs)
    if output.shape[0] > 30:
        return None

    dup_candidates = list(range(n_active - 1))  # topmost stripe cannot be duplicated cleanly
    max_noise = min(3, len(dup_candidates))
    n_noise = int(np.random.randint(1, max_noise + 1))
    dup_indices = list(np.random.choice(dup_candidates, size=n_noise, replace=False))

    scheduled: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_active)}
    for idx in dup_indices:
        insert_after = int(np.random.randint(idx + 1, n_active))
        noise_h = int(np.random.randint(1, 5))
        scheduled[insert_after].append((active_cols[idx], noise_h))

    final_runs: list[tuple[int, int]] = []
    for i, run in enumerate(active_runs):
        final_runs.append(run)
        slot_items = scheduled[i]
        if slot_items:
            order = np.random.permutation(len(slot_items))
            final_runs.extend(slot_items[int(j)] for j in order)

    total_h = sum(h for _col, h in final_runs)
    if total_h > 30:
        return None

    width = int(np.random.randint(6, 16))
    stripes = [
        np.full((run_h, width), col, dtype=int)
        for col, run_h in reversed(final_runs)
    ]
    grid = np.vstack(stripes)

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
