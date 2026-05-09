# Variant of 97d7923e
# Dimension: D12 — 多步处理逻辑 (multi_step)
# Change: 所有图例条目独立处理，组间无依赖顺序：对每个图例(K,k)，找到帽色为K的组，按长度排名选第k条，重着色... → 图例条目按序处理，前一步结果影响后续

# Design analysis for the D12 variant
# ==================================
# a) Concrete behavior and trace
#    - Visual objects remain the same as the source task: top-left legend columns and bottom-aligned
#      cap/fill/cap vertical bars.
#    - The visible change is that the legend may now contain the same cap color more than once.
#      Each repeated legend component is still a separate 1-column block because same-colored legend
#      entries are separated by a background gap.
#    - Example for one cap color K with three bars of interior lengths [6, 4, 2] and two legend
#      entries (K,1), (K,2) from left to right:
#        original logic: choose rank-1 from the full list, then rank-2 from the full list -> bars 6 and 4
#        new logic: choose rank-1 from [6,4,2] -> remove 6 -> choose rank-2 from [4,2] -> bars 6 and 2
#      The output differs, so the order dependency is pixel-visible.
#
# b) Complexity calibration
#    - The task stays close to the source task: same bar geometry, same local ranking cue, same legend
#      encoding by height.
#    - Only one extra degree of freedom is added: at least one cap color repeats in the legend.
#      Counts stay small (2-3 bars per color; 2-3 legend requests for the repeated color), which keeps
#      the search space human-readable and avoids ambiguous bookkeeping.
#
# c) Implicit assumption review of the source solver
#    - `_legend_entries(...)` only assumes the legend is a cluster on the top row. It does NOT require
#      colors to be unique, so it can be reused as-is.
#    - `for _x, cap, k in legend:` is still valid, but the old body recomputed candidates from the
#      immutable input `g`, which silently relies on "one legend entry per cap color".
#    - `segs = _vertical_sandwiches(g, cap)` is the key line that breaks under D12. It must read from a
#      mutable state where already-selected bars are no longer candidates.
#    - `out[top_r + 1 : bot_r, col] = cap` is still safe: recoloring a chosen bar to the cap color makes
#      that column disappear from `_vertical_sandwiches(out, cap)`, so the output grid itself can serve
#      as the updated ranking pool.
#
# d) Placement legality
#    - Legend columns are always a subset of bar columns, exactly as in the source generator.
#    - Each legend block sits above its bar with at least one background row between them.
#    - Bars use unique columns, stay in-bounds, and keep distinct interior lengths within each cap group.
#    - Repeated legend entries for one cap are separate connected components because they occupy different
#      columns with a background gap.
#
# e) Visual observability
#    - The change is visible without hidden metadata: the same color appears multiple times in the legend,
#      and those repeated requests recolor multiple bars of that same color group.
#    - The generator rejects degenerate samples where sequential processing would accidentally produce the
#      same output as the original independent processing.
#
# f) Visual cleanliness
#    - Only one repeated-color legend block is introduced per sample. Other groups keep one legend entry.
#    - Column spacing and shared bottom alignment preserve the source task's tidy appearance.
#
# g) Color-change inferability
#    - Colors still have a structural role only: each legend color matches one cap-color group.
#    - No ordering depends on color ids. Ranking depends only on visible bar lengths.
#
# h) New solver assumptions introduced by D12
#    - If a cap color appears `m` times in the legend, the grid must contain enough bars of that cap so
#      each step's rank is valid in the remaining pool.
#    - `generate_input()` guarantees this by constructing a valid per-group request sequence and verifying
#      both the sequential oracle and the non-degeneracy condition.

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path('/home/lixinyu/snap/new/solvers')
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, neighbors4, paste


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


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend is in the top-left: a cluster of non-bg pixels on the top row near the left edge,
    allowing small gaps (typically 1 bg column) between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    the top row within the legend cluster.
    """
    h, w = g.shape
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, 'legend row must be non-empty'
    top_pos.sort()

    cols = [top_pos[0]]
    for c in top_pos[1:]:
        if c - cols[-1] <= 2:
            cols.append(c)
        else:
            break
    legend_max_c = max(cols)

    entries: list[tuple[int, int, int]] = []
    seen = np.zeros((h, w), dtype=bool)
    for c in cols:
        col = int(g[0, c])
        if col == bg or seen[0, c]:
            continue
        st = [(0, c)]
        seen[0, c] = True
        pix = []
        while st:
            r, cc = st.pop()
            pix.append((r, cc))
            for rr, ccc in neighbors4(r, cc):
                if 0 <= rr < h and 0 <= ccc <= legend_max_c and (not seen[rr, ccc]) and int(g[rr, ccc]) == col:
                    seen[rr, ccc] = True
                    st.append((rr, ccc))
        x_pos = min(cc for _, cc in pix)
        rs = [r for r, _ in pix]
        block_count = max(rs) - min(rs) + 1
        entries.append((x_pos, col, int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by sequential legend-driven rank selection.

    Input structure:
    - bg is the background. The top-left legend is a left-to-right row of 1-column blocks; a block's
      color K names a cap-color group, and its height k stores a rank request.
    - The main field contains bottom-aligned vertical bars with form cap / solid fill / cap. Several
      bars may share the same cap color, and that same cap color may appear multiple times in the legend.

    Rule:
    - Read legend entries from left to right.
    - For each entry (K, k), find the remaining bars capped by K and rank them by fill length
      (longest first, with the original row/column tie-break).
    - Recolor the current rank-k bar's interior to K; that bar is then removed from the ranking pool for
      later legend entries of the same color.

    Key insight: A repeated legend color is not redundant. Each occurrence consumes one bar from that
    color group, so later occurrences must re-rank only the bars that still have their original fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, 'legend must be non-empty'

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(out, cap)
        assert segs, f'legend cap {cap} must have matching remaining vertical bars'
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f'legend rank {k} out of range for {len(segs)} remaining bars'
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 400

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _ranked_bars(
    cols: list[int], lengths: list[int]
) -> list[tuple[int, int]]:
    """Return (col, fill_len) sorted by the solver's ranking rule."""
    return sorted(zip(cols, lengths), key=lambda item: (-item[1], _MAX_CAP_ROW - item[1], item[0]))


def _chosen_original_ranks(count: int, ranks: list[int]) -> list[int]:
    """Map sequential remaining-pool ranks back to original 0-based rank indices."""
    active = list(range(count))
    chosen: list[int] = []
    for rank in ranks:
        idx = active.pop(rank - 1)
        chosen.append(idx)
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
    assert valid, f'no non-degenerate rank sequence for count={count}, picks={picks}'
    return list(valid[int(rng.integers(len(valid)))])


def _choose_group_specs(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[list[int]], list[int]]:
    """
    Choose per-group bar counts and legend requests.

    Returns:
      counts: number of bars in each unique cap-color group
      entry_counts: number of legend entries for each group
      rank_seqs: rank requests per group, in that group's internal left-to-right legend order
      legend_groups: global left-to-right legend order as group indices
    """
    while True:
        n_groups = int(rng.integers(1, MAX_GROUPS + 1))
        repeated_group = int(rng.integers(n_groups))

        entry_counts = [1] * n_groups
        entry_counts[repeated_group] = int(rng.integers(2, BARS_PER_GROUP[1] + 1))

        counts: list[int] = []
        for group_idx in range(n_groups):
            lo = max(BARS_PER_GROUP[0], entry_counts[group_idx])
            hi = BARS_PER_GROUP[1]
            counts.append(int(rng.integers(lo, hi + 1)))

        if sum(counts) > MAX_TOTAL_BARS:
            continue

        rank_seqs: list[list[int]] = []
        for count, entry_count in zip(counts, entry_counts):
            if entry_count == 1:
                rank_seqs.append([int(rng.integers(1, count + 1))])
            else:
                rank_seqs.append(_choose_rank_sequence(rng, count, entry_count))

        blocks = [[repeated_group] * entry_counts[repeated_group]]
        for group_idx in range(n_groups):
            if group_idx != repeated_group:
                blocks.append([group_idx])
        rng.shuffle(blocks)
        legend_groups = [group_idx for block in blocks for group_idx in block]
        return counts, entry_counts, rank_seqs, legend_groups


def _assign_columns(
    rng: np.random.Generator, legend_groups: list[int], counts: list[int], width: int
) -> list[list[int]]:
    """
    Assign one unique bar column per bar.

    The first `len(legend_groups)` occupied columns belong to the legend cluster and therefore
    double as legend columns; repeated legend entries simply allocate multiple early columns to
    the same group.
    """
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    extra_width = width - min_width
    gap_bonus = np.zeros(total_bars - 1, dtype=int)
    if total_bars > 1 and extra_width > 0:
        gap_bonus = rng.multinomial(extra_width, np.full(total_bars - 1, 1.0 / (total_bars - 1)))

    occupied = [0]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + COL_SPACING + int(gap_bonus[idx]))

    legend_cols = occupied[: len(legend_groups)]
    extra_cols = occupied[len(legend_groups) :]

    group_columns = [[] for _ in counts]
    for col, group_idx in zip(legend_cols, legend_groups):
        group_columns[group_idx].append(col)

    owners = [group_idx for group_idx, count in enumerate(counts) for _ in range(count - len(group_columns[group_idx]))]
    rng.shuffle(owners)
    for col, owner in zip(extra_cols, owners):
        group_columns[owner].append(col)

    return group_columns


def _choose_group_lengths(
    rng: np.random.Generator, count: int, legend_ranks: list[int]
) -> list[int]:
    """
    Pick distinct fill lengths for one group, in the same order as that group's columns.

    The group's first `len(legend_ranks)` columns are legend columns; each such bar must fit
    below its own legend block with at least one background row of separation.
    """
    max_by_pos = [MAX_FILL_LEN - rank for rank in legend_ranks] + [MAX_FILL_LEN] * (count - len(legend_ranks))
    perms = [
        perm
        for perm in itertools.permutations(range(1, MAX_FILL_LEN + 1), count)
        if all(fill_len <= max_len for fill_len, max_len in zip(perm, max_by_pos))
    ]
    assert perms, f'no valid fill lengths for count={count}, max_by_pos={max_by_pos}'
    return [int(v) for v in perms[int(rng.integers(len(perms)))]]


def _legend_sequence(rank_seqs: list[list[int]], legend_groups: list[int]) -> list[tuple[int, int]]:
    """Expand per-group rank sequences into global left-to-right legend entries."""
    used = [0] * len(rank_seqs)
    seq: list[tuple[int, int]] = []
    for group_idx in legend_groups:
        pos = used[group_idx]
        seq.append((group_idx, rank_seqs[group_idx][pos]))
        used[group_idx] += 1
    return seq


def _expected_output_sequential(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    legend_seq: list[tuple[int, int]],
    caps: list[int],
) -> np.ndarray:
    """Apply the D12 sequential-removal rule directly from the construction plan."""
    out = inp.copy()
    active = [_ranked_bars(cols, lengths) for cols, lengths in zip(group_columns, group_lengths)]
    for group_idx, rank in legend_seq:
        col, fill_len = active[group_idx].pop(rank - 1)
        top_r = _MAX_CAP_ROW - fill_len
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
        top_r = _MAX_CAP_ROW - fill_len
        out[top_r + 1 : _BOTTOM_ROW, col] = caps[group_idx]
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch or degeneration."""
    rng = np.random.default_rng()

    counts, entry_counts, rank_seqs, legend_groups = _choose_group_specs(rng)
    n_groups = len(counts)
    legend_seq = _legend_sequence(rank_seqs, legend_groups)

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, legend_groups, counts, width)

    used = [0] * n_groups
    legend_ranks_by_group = [[] for _ in range(n_groups)]
    for group_idx in legend_groups:
        pos = used[group_idx]
        legend_ranks_by_group[group_idx].append(rank_seqs[group_idx][pos])
        used[group_idx] += 1

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for group_idx, (cap, fill, count, cols) in enumerate(zip(caps, fills, counts, group_columns)):
        legend_ranks = legend_ranks_by_group[group_idx]
        lengths = _choose_group_lengths(rng, count, legend_ranks)
        group_lengths.append(lengths)

        for col, rank in zip(cols[: len(legend_ranks)], legend_ranks):
            paste(grid, np.full((rank, 1), cap, dtype=int), 0, col)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    expected = _expected_output_sequential(grid, group_columns, group_lengths, legend_seq, caps)
    independent = _expected_output_independent(grid, group_columns, group_lengths, legend_seq, caps)

    if np.array_equal(expected, independent):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
