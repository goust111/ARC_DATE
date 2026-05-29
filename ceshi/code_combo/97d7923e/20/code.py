# Combo variant of 97d7923e (index 20)
# combo_spec: D8/1+D12/0
# Name: 图例嵌入条间+顺序处理

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The visible objects are still bottom-aligned vertical cap/fill/cap bars, but the legend is no
#      longer a single top-left cluster. Each legend entry is a one-column cap-colored stack touching
#      row 0 and embedded directly above one bar column of the same cap.
#    - One group is repeated: its first two bar columns both carry legend stacks. Example with cap K
#      and bar lengths [7, 5, 3], with legend heights [1, 2] above the first two columns:
#        source task: one corner legend per color, independent picks from the full pool
#        fused task: read left embedded (K,1), recolor length-7 bar, remove it; read next embedded
#        (K,2), rerank only [5,3], recolor length-3 bar. The same color thus behaves differently
#        because the second legend entry is both local in space and late in time.
#    - Other groups keep one embedded legend in their first column, so the full left-to-right legend
#      order is spread across multiple bar blocks instead of being concentrated in one corner.
# b) Complexity calibration
#    - The construction keeps 2..4 groups and 2..3 bars per group, so same-cap grouping remains easy
#      to see and rank by eye.
#    - Only one group repeats, and it repeats exactly twice. That is enough to make the sequential
#      dependency visible without overloading the top row with too many same-color markers.
#    - The repeated group always has one extra non-legend bar after its legend-bearing columns. This
#      preserves a clear local group block and stops all legends from collapsing back into one prefix.
# c) Hidden-assumption audit of the source solver
#    - The source `_legend_entries(...)` assumes all legend pixels live in one short top-left prefix.
#      That fails here because later groups place legend stacks farther right among the bars.
#    - The source loop also assumes each legend entry can rank against the immutable input grid. That
#      fails when the same cap appears twice, because the first recolored bar must disappear from the
#      candidate pool before the second request.
#    - `_vertical_sandwiches(...)` itself is still valid: embedded legend stacks do not look like bars
#      because the gap row below the stack is background, so no false sandwich survives the fill check.
# d) Placement legality
#    - Grid height stays 10, width stays 10..20, background stays 0.
#    - Each group occupies a contiguous left-to-right block of unique bar columns spaced by 2. The
#      first `m` columns of a group are its `m` legend columns; every legend column leaves at least one
#      background row between the top stack and the actual bar.
#    - Distinct fill lengths within a group avoid rank ties. The repeated group has 3 bars so its 2
#      legend columns still leave one plain bar column after them, which keeps inter-group legend gaps
#      larger than the old top-left-cluster tolerance.
# e) Visual observability
#    - A solver can see two simultaneous cues in a few samples: legends are attached to local bar
#      blocks, and one color appears twice with two outputs that match sequential consumption rather
#      than independent re-use of the same ranking pool.
#    - The generator rejects degenerate samples where the sequential output would match the old
#      independent interpretation, because such samples would hide the D12 change.
# f) Visual cleanliness
#    - Legend stacks are always one column wide and only occupy the very top of their own bar columns.
#      No object overlaps another, and the bar field remains readable because groups stay contiguous.
# g) Change inferability
#    - No rule depends on numeric color ids. The only color logic is structural: a legend stack uses
#      the same cap color as its group, and the chosen bar is recolored from its fill color to that cap.
# h) New solver assumptions introduced by the combo
#    - Every top-row non-background stack that counts as a legend must sit above a lower same-cap bar
#      in the same column with a background gap in between.
#    - If a cap repeats in the embedded legend, there must be enough bars of that cap for all requests,
#      and the requested ranks must remain valid after earlier removals. `generate_input()` enforces
#      these conditions constructively and rechecks them against the solver.


def _vertical_sandwiches(g: np.ndarray, cap: int) -> list[tuple[int, int, int, int]]:
    """
    Find vertical bar candidates (top_r, bottom_r, col, fill_color) where:
      - g[top_r, col] == cap and g[bottom_r, col] == cap
      - (top_r, bottom_r) are consecutive occurrences of cap in that column
      - interior is a solid run of ONE color fill_color != cap and fill_color != background
      - length >= 1 (bottom_r >= top_r + 2)
    """
    h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int]] = []
    for c in range(w):
        rows = [int(r) for r in np.flatnonzero(g[:, c] == cap)]
        if len(rows) < 2:
            continue
        for a, b in zip(rows, rows[1:]):
            if b <= a + 1:
                continue
            fill = int(g[a + 1, c])
            if fill == cap or fill == bg:
                continue
            if np.all(g[a + 1 : b, c] == fill):
                segs.append((a, b, c, fill))
    return segs


def _embedded_legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legends are distributed across the top row: each one is a same-color vertical stack
    touching row 0, with a background gap below it and a lower bar of the same cap color
    in that same column.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    """
    h, w = g.shape
    entries: list[tuple[int, int, int]] = []

    for c in range(w):
        cap = int(g[0, c])
        if cap == bg:
            continue

        k = 1
        while k < h and int(g[k, c]) == cap:
            k += 1
        if k >= h or int(g[k, c]) != bg:
            continue

        if any(seg[2] == c and seg[0] > k for seg in _vertical_sandwiches(g, cap)):
            entries.append((c, cap, k))

    assert entries, "embedded legend must be non-empty"
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by left-to-right embedded legend requests with removal.

    Input structure:
    - The background is uniform. Along the top row, legend entries appear as one-column
      cap-colored stacks embedded above specific bar columns rather than packed into one
      corner; a repeated group can show two such stacks in its first two columns.
    - The main objects are bottom-aligned vertical bars of the form cap / solid fill / cap.
      Bars sharing a cap color form one group, and their fill lengths are distinct.

    Rule:
    - Scan the embedded legend stacks from left to right.
    - For each entry (K, k), look at the remaining bars capped by K, rank them by interior
      length from longest to shortest, and recolor the current rank-k bar's interior to K.
    - After a bar is recolored, it is removed from the ranking pool for any later legend
      entries of the same cap color.

    Key insight:
    - The top stacks are local markers attached to the bar field, not a separate global
      legend, and repeated same-color markers must be interpreted sequentially because each
      earlier recolor changes which later bar counts as rank k.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _embedded_legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(out, cap)
        assert segs, f"legend cap {cap} must have matching remaining vertical bars"
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} remaining bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MIN_GROUPS = 2
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 400

_BOTTOM_ROW = GRID_H - 1


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _bar_top_row(fill_len: int) -> int:
    return _BOTTOM_ROW - fill_len - 1


def _ranked_bars(cols: list[int], lengths: list[int]) -> list[tuple[int, int]]:
    """Return (col, fill_len) sorted by the solver's ranking rule."""
    return sorted(zip(cols, lengths), key=lambda item: (-item[1], _bar_top_row(item[1]), item[0]))


def _chosen_original_ranks(count: int, ranks: list[int]) -> list[int]:
    """Map sequential remaining-pool ranks back to original 0-based rank indices."""
    active = list(range(count))
    chosen: list[int] = []
    for rank in ranks:
        chosen.append(active.pop(rank - 1))
    return chosen


def _choose_rank_sequence(rng: np.random.Generator, count: int, picks: int) -> list[int]:
    """
    Choose a valid sequential rank request sequence whose final recolored set differs
    from the source task's independent interpretation.
    """
    valid: list[tuple[int, ...]] = []

    def rec(step: int, so_far: list[int]) -> None:
        if step == picks:
            seq = tuple(so_far)
            seq_set = set(_chosen_original_ranks(count, list(seq)))
            independent_set = {rank - 1 for rank in seq}
            if seq_set != independent_set:
                valid.append(seq)
            return
        remaining = count - step
        for rank in range(1, remaining + 1):
            so_far.append(rank)
            rec(step + 1, so_far)
            so_far.pop()

    rec(0, [])
    assert valid, f"no non-degenerate rank sequence for count={count}, picks={picks}"
    return list(valid[int(rng.integers(len(valid)))])


def _choose_group_specs(
    rng: np.random.Generator,
) -> tuple[list[int], list[list[int]], list[int]]:
    """
    Choose per-group bar counts, embedded legend requests, and left-to-right group order.

    Exactly one group repeats, and that group has 3 bars with 2 legend-bearing columns.
    This guarantees both sequential dependence and a trailing plain bar that separates the
    group's legend region from the next group's first legend.
    """
    while True:
        n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
        repeated_group = int(rng.integers(n_groups))

        counts = [int(rng.integers(BARS_PER_GROUP[0], BARS_PER_GROUP[1] + 1)) for _ in range(n_groups)]
        counts[repeated_group] = 3
        if sum(counts) > MAX_TOTAL_BARS:
            continue

        rank_seqs: list[list[int]] = []
        for group_idx, count in enumerate(counts):
            if group_idx == repeated_group:
                rank_seqs.append(_choose_rank_sequence(rng, count, 2))
            else:
                rank_seqs.append([int(rng.integers(1, count + 1))])

        group_order = list(range(n_groups))
        rng.shuffle(group_order)
        return counts, rank_seqs, group_order


def _assign_group_columns(
    rng: np.random.Generator, counts: list[int], group_order: list[int], width: int
) -> list[list[int]]:
    """
    Lay groups out left-to-right as local bar blocks.

    The first columns of each block are the legend-bearing ones, so legend entries are
    embedded in the bar field and remain locally associated with their groups.
    """
    total_bars = sum(counts)
    base_span = COL_SPACING * total_bars - 1
    extra_width = width - base_span
    gap_parts = rng.multinomial(extra_width, np.full(len(group_order) + 1, 1.0 / (len(group_order) + 1)))

    group_columns: list[list[int]] = [[] for _ in counts]
    cur = int(gap_parts[0])
    for idx, group_idx in enumerate(group_order):
        count = counts[group_idx]
        cols = [cur + COL_SPACING * j for j in range(count)]
        group_columns[group_idx] = cols
        if idx + 1 < len(group_order):
            cur = cols[-1] + COL_SPACING + int(gap_parts[idx + 1])
    return group_columns


def _choose_group_lengths(
    rng: np.random.Generator, count: int, legend_ranks: list[int]
) -> list[int]:
    """
    Pick distinct fill lengths for one group, in the same order as that group's columns.

    The first `len(legend_ranks)` columns carry embedded legends, so each of those bars
    must fit below its legend stack with at least one background row of separation.
    """
    max_by_pos = [MAX_FILL_LEN - rank for rank in legend_ranks] + [MAX_FILL_LEN] * (count - len(legend_ranks))
    perms = [
        perm
        for perm in itertools.permutations(range(1, MAX_FILL_LEN + 1), count)
        if all(fill_len <= max_len for fill_len, max_len in zip(perm, max_by_pos))
    ]
    assert perms, f"no valid fill lengths for count={count}, max_by_pos={max_by_pos}"
    return [int(v) for v in perms[int(rng.integers(len(perms)))]]


def _legend_sequence(rank_seqs: list[list[int]], group_order: list[int]) -> list[tuple[int, int]]:
    """Expand per-group embedded legend requests into global left-to-right order."""
    seq: list[tuple[int, int]] = []
    for group_idx in group_order:
        for rank in rank_seqs[group_idx]:
            seq.append((group_idx, rank))
    return seq


def _expected_output_sequential(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    legend_seq: list[tuple[int, int]],
    caps: list[int],
) -> np.ndarray:
    """Apply the fused sequential-removal rule directly from the construction plan."""
    out = inp.copy()
    active = [_ranked_bars(cols, lengths) for cols, lengths in zip(group_columns, group_lengths)]
    for group_idx, rank in legend_seq:
        col, fill_len = active[group_idx].pop(rank - 1)
        top_r = _bar_top_row(fill_len)
        out[top_r + 1 : _BOTTOM_ROW, col] = caps[group_idx]
    return out


def _expected_output_independent(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    legend_seq: list[tuple[int, int]],
    caps: list[int],
) -> np.ndarray:
    """Reference output under the source task's independent per-entry interpretation."""
    out = inp.copy()
    ranked = [_ranked_bars(cols, lengths) for cols, lengths in zip(group_columns, group_lengths)]
    for group_idx, rank in legend_seq:
        col, fill_len = ranked[group_idx][rank - 1]
        top_r = _bar_top_row(fill_len)
        out[top_r + 1 : _BOTTOM_ROW, col] = caps[group_idx]
    return out


def _top_row_prefix_cluster_count(g: np.ndarray, bg: int) -> int:
    """
    Count how many top-row colored columns would be swallowed by the source task's
    old top-left legend-prefix heuristic (gaps <= 2).
    """
    cols = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    if not cols:
        return 0
    count = 1
    for prev, cur in zip(cols, cols[1:]):
        if cur - prev <= 2:
            count += 1
        else:
            break
    return count


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on mismatch or degeneration."""
    rng = np.random.default_rng()

    counts, rank_seqs, group_order = _choose_group_specs(rng)
    n_groups = len(counts)
    legend_seq = _legend_sequence(rank_seqs, group_order)

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_group_columns(rng, counts, group_order, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for group_idx, (cap, fill, count, cols) in enumerate(zip(caps, fills, counts, group_columns)):
        legend_ranks = rank_seqs[group_idx]
        lengths = _choose_group_lengths(rng, count, legend_ranks)
        group_lengths.append(lengths)

        for col, rank in zip(cols[: len(legend_ranks)], legend_ranks):
            paste(grid, np.full((rank, 1), cap, dtype=int), 0, col)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _bar_top_row(fill_len), col)

    expected = _expected_output_sequential(grid, group_columns, group_lengths, legend_seq, caps)
    independent = _expected_output_independent(grid, group_columns, group_lengths, legend_seq, caps)
    legend_total = sum(len(seq) for seq in rank_seqs)

    if np.array_equal(expected, independent):
        return None
    if _top_row_prefix_cluster_count(grid, get_bg(grid)) >= legend_total:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
