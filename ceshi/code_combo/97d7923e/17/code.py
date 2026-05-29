# Combo variant of 97d7923e (index 17)
# combo_spec: D11/0+D12/0
# Name: 未选中条擦除+顺序处理

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Source task: each legend column (K, k) independently recolors the rank-k bar
#      among all bars capped by K, and every non-selected sibling bar stays visible.
#    - D11 alone changes the final canvas: only the selected bar(s) should survive;
#      all never-selected sibling bars disappear to bg.
#    - D12 alone changes the choice process: if the same cap color appears multiple
#      times in the legend, later requests re-rank only the bars not consumed by
#      earlier requests of that color.
#    - Directly combining "erase every non-selected bar immediately" with repeated
#      same-color legend entries would empty that color group after the first step.
#      Pixel-wise ARC I/O only exposes the final state, not intermediate frames, so
#      the realizable fusion is: process legend entries left-to-right on a remaining
#      pool, record every selected bar, then keep exactly those selected bars in the
#      final output and erase all other bars. Example with one color group of lengths
#      [6, 4, 2] and legend requests (K,1), (K,2):
#      independent source behavior -> recolor 6 and 4, keep 2 visible
#      fused behavior -> choose 6, remove it from the pool, then rank [4,2] and
#      choose 2; final output keeps only 6 and 2, while 4 is erased.
# b) Complexity calibration:
#    - The generator always includes one repeated legend color with exactly two
#      requests and exactly three bars. This is the smallest setting where left-to-
#      right re-ranking can change which bars survive while still leaving at least
#      one erased bar in that repeated group.
#    - Other groups still use 2-3 bars and one legend request each, so the "erase
#      non-selected bars" effect remains obvious across the sample set.
# c) Implicit assumption audit of the source solver:
#    - `_legend_entries(...)` still works because it only assumes the legend is the
#      leftmost top-row cluster; repeated colors are allowed as separate components.
#    - `segs = _vertical_sandwiches(g, cap)` inside the loop is no longer valid for
#      the combo because it silently treats every legend request as independent.
#      The solver now needs one persistent remaining-pool per cap color.
#    - Writing directly into `out` as the pool is also insufficient here: after the
#      D11-style final erasure, bars that should remain eligible for later repeated
#      requests would vanish too early. The solver therefore tracks selection in
#      explicit lists and renders the final keep/erase result afterward.
# d) Placement legality:
#    - Grid size, bottom alignment, bar geometry, and legend-above-bar separation all
#      stay within the source spec.
#    - Each repeated-color legend entry gets its own legend column, so repeated
#      requests are visible as separate top-left stacks without overlap.
#    - Distinct bar lengths within a group preserve unambiguous ranking, and the
#      generator ensures every sequential rank request is valid in the remaining pool.
# e) Visual observability:
#    - The input always shows one color repeated in the legend, making the sequential
#      bookkeeping cue visible to a human.
#    - The output always removes at least one whole bar, so the D11 change is also
#      directly visible. The repeated group is additionally forced to differ from the
#      source task's independent interpretation.
# f) Visual cleanliness:
#    - Only one group repeats, and it repeats exactly twice. This keeps the legend
#      readable while still making order matter.
#    - Column spacing and shared bottom alignment preserve the tidy look of the seed
#      task; the final output contains only solid cap-colored survivor columns.
# g) Change inferability:
#    - No hidden property of the numeric color ids is used. Colors only mark cap
#      groups, legend height only marks rank, and order is conveyed purely by the
#      left-to-right arrangement of repeated legend stacks.
# Post-change solver assumption audit:
#    - `main()` assumes every repeated-color request count is no larger than that
#      group's bar count. `generate_input()` enforces this by construction.
#    - `main()` assumes all bar lengths within one cap group are distinct so rank
#      order is stable. The generator picks distinct lengths per group.
#    - `main()` assumes every cap color named in the legend has at least one bar
#      group and that all bars stay valid cap/fill/cap sandwiches in the input. The
#      generator uses direct constructive placement and validates with an oracle.


def _rank_key(seg: tuple[int, int, int, int]) -> tuple[int, int, int]:
    top_r, bot_r, col, _fill = seg
    return (-(bot_r - top_r - 1), top_r, col)


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
    assert top_pos, "legend row must be non-empty"
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
    Keep only the bars selected by left-to-right legend consumption.

    Input structure:
    - bg is the background. The top-left legend is a left-to-right row of 1-column
      stacks; each stack has color K and height k.
    - The main field contains bottom-aligned vertical cap/fill/cap bars. Bars that
      share cap color K form one rankable group, and one color may appear more than
      once in the legend as separate stacks.

    Rule:
    - Read legend entries from left to right.
    - For each entry (K, k), rank the remaining bars capped by K by interior length
      (longest first, then the original row/column tie-break) and consume the rank-k
      bar from that remaining pool.
    - After all legend entries have been processed, repaint every consumed bar to a
      solid K column and erase every never-consumed sibling bar to bg.

    Key insight:
    - Repeated legend colors are sequential requests, not duplicates. The first such
      request changes which bars are still eligible for the next one, and the final
      picture keeps only the bars that were consumed in that left-to-right sequence.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    all_by_cap: dict[int, list[tuple[int, int, int, int]]] = {}
    active_by_cap: dict[int, list[tuple[int, int, int, int]]] = {}
    selected_by_cap: dict[int, set[tuple[int, int, int, int]]] = {}

    for _x, cap, _k in legend:
        if cap in all_by_cap:
            continue
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        ranked = sorted(segs, key=_rank_key)
        all_by_cap[cap] = ranked
        active_by_cap[cap] = ranked.copy()
        selected_by_cap[cap] = set()

    for _x, cap, k in legend:
        active = active_by_cap[cap]
        assert 1 <= k <= len(active), f"legend rank {k} out of range for {len(active)} remaining bars"
        chosen = active.pop(k - 1)
        selected_by_cap[cap].add(chosen)

    for cap, segs in all_by_cap.items():
        kept = selected_by_cap[cap]
        for top_r, bot_r, col, _fill in segs:
            if (top_r, bot_r, col, _fill) in kept:
                out[top_r : bot_r + 1, col] = cap
            else:
                out[top_r : bot_r + 1, col] = bg

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MIN_GROUPS = 2
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
OTHER_GROUP_BARS = (2, 3)
REPEATED_GROUP_BARS = 3
REPEATED_GROUP_ENTRIES = 2
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 500

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _ranked_bars(cols: list[int], lengths: list[int]) -> list[tuple[int, int]]:
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
    Choose a valid sequential rank request sequence whose survivor set differs from
    the independent-per-entry interpretation.
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
        n_groups = int(rng.integers(MIN_GROUPS, MAX_GROUPS + 1))
        repeated_group = int(rng.integers(n_groups))

        counts = [int(rng.integers(OTHER_GROUP_BARS[0], OTHER_GROUP_BARS[1] + 1)) for _ in range(n_groups)]
        counts[repeated_group] = REPEATED_GROUP_BARS
        if sum(counts) > MAX_TOTAL_BARS:
            continue

        entry_counts = [1] * n_groups
        entry_counts[repeated_group] = REPEATED_GROUP_ENTRIES

        rank_seqs: list[list[int]] = []
        for group_idx, count in enumerate(counts):
            if group_idx == repeated_group:
                rank_seqs.append(_choose_rank_sequence(rng, count, entry_counts[group_idx]))
            else:
                rank_seqs.append([int(rng.integers(1, count + 1))])

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
    double as legend columns; repeated legend entries allocate multiple early columns to one group.
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
    assert perms, f"no valid fill lengths for count={count}, max_by_pos={max_by_pos}"
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


def _expected_output_combo(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    legend_seq: list[tuple[int, int]],
    caps: list[int],
) -> np.ndarray:
    """Apply the combo rule: sequentially consume bars, then erase non-consumed bars."""
    out = inp.copy()
    bg = get_bg(inp)
    active = [_ranked_bars(cols, lengths) for cols, lengths in zip(group_columns, group_lengths)]
    kept: list[set[tuple[int, int]]] = [set() for _ in group_columns]

    for group_idx, rank in legend_seq:
        col, fill_len = active[group_idx].pop(rank - 1)
        kept[group_idx].add((col, fill_len))

    for group_idx, (cols, lengths) in enumerate(zip(group_columns, group_lengths)):
        for col, fill_len in zip(cols, lengths):
            top_r = _MAX_CAP_ROW - fill_len
            if (col, fill_len) in kept[group_idx]:
                out[top_r : _BOTTOM_ROW + 1, col] = caps[group_idx]
            else:
                out[top_r : _BOTTOM_ROW + 1, col] = bg
    return out


def _expected_output_independent_erase(
    inp: np.ndarray,
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    legend_seq: list[tuple[int, int]],
    caps: list[int],
) -> np.ndarray:
    """Reference output if repeated legend entries incorrectly re-used the full pool."""
    out = inp.copy()
    bg = get_bg(inp)
    ranked = [_ranked_bars(cols, lengths) for cols, lengths in zip(group_columns, group_lengths)]
    kept: list[set[tuple[int, int]]] = [set() for _ in group_columns]

    for group_idx, rank in legend_seq:
        col, fill_len = ranked[group_idx][rank - 1]
        kept[group_idx].add((col, fill_len))

    for group_idx, (cols, lengths) in enumerate(zip(group_columns, group_lengths)):
        for col, fill_len in zip(cols, lengths):
            top_r = _MAX_CAP_ROW - fill_len
            if (col, fill_len) in kept[group_idx]:
                out[top_r : _BOTTOM_ROW + 1, col] = caps[group_idx]
            else:
                out[top_r : _BOTTOM_ROW + 1, col] = bg
    return out


def _combo_metric(
    group_lengths: list[list[int]], legend_seq: list[tuple[int, int]]
) -> int:
    """
    Count how many original-rank positions differ between combo and independent survivor sets.
    A positive value means the sequential dependency is actually visible in the final survivors.
    """
    used: dict[int, list[int]] = {}
    for group_idx, rank in legend_seq:
        used.setdefault(group_idx, []).append(rank)

    changed = 0
    for group_idx, ranks in used.items():
        if len(ranks) < 2:
            continue
        count = len(group_lengths[group_idx])
        seq_set = set(_chosen_original_ranks(count, ranks))
        indep_set = {rank - 1 for rank in ranks}
        changed += len(seq_set.symmetric_difference(indep_set))
    return changed


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

    expected = _expected_output_combo(grid, group_columns, group_lengths, legend_seq, caps)
    independent = _expected_output_independent_erase(grid, group_columns, group_lengths, legend_seq, caps)

    if _combo_metric(group_lengths, legend_seq) <= 0:
        return None
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
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
