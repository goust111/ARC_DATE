# Variant of 97d7923e
# Dimension: D12 — 多步处理逻辑 (multi_step)
# Change: 所有图例条目独立处理，组间无依赖顺序：对每个图例(K,k)，找到帽色为K的组，按长度排名选第k条，重着色... → 逆向处理：从输出推断图例

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D12 inversion variant
# a) Concrete behavior and trace:
#    - Original task: the top-left legend is visible in the input, and it tells us which bar
#      to recolor inside each cap-color group.
#    - Variant task: the bars already show the finished recoloring, while the legend area is
#      blank. The solver must read the finished state and reconstruct the missing cause.
#    - Concretely, each group still consists of bottom-aligned one-cell-wide vertical bars that
#      share cap color K and unrecolored fill color F. Exactly one bar per group has already had
#      its fill recolored to K, so that whole bar body becomes monochrome K between the two caps.
#      The legend remains in the original top-left columns, but those cells are zeroed in the
#      input and restored in the output.
#    - Small trace: suppose a cap-2 group has bar fill lengths [6, 4, 2], ordered by length.
#      In the original task, a legend block of height 2 would recolor the length-4 bar. In this
#      variant, the length-4 bar is already monochrome 2 in the input, so the solver infers rank
#      2 and writes a 2-cell legend block above the group's leftmost column.
# b) Complexity calibration:
#    - The inversion is visible because input row 0 becomes empty, while one bar in each group is
#      already highlighted. Human solvers can compare the highlighted bar with the other lengths
#      in the same color group and recover the missing legend height.
#    - Group count stays 1-4 and bars per group stay 2-3, matching the original difficulty and
#      avoiding excessive ambiguity. Distinct lengths inside a group are preserved so ranking is
#      always visible without hidden tie rules.
# c) Implicit-assumption audit against the original solver:
#    - Original `_legend_entries(...)` assumed non-empty top-row legend pixels; that fails here
#      because the top-left region is blank. The new solver must derive legend entries from bars.
#    - Original `_vertical_sandwiches(...)` rejected bars whose fill already equals the cap. That
#      was harmless before, but it would skip the crucial highlighted bars in this variant.
#    - Original main loop iterated legend entries and applied recoloring. That direction is gone;
#      the solver now groups bars by cap color, detects the already highlighted one, computes its
#      rank, and only then writes the legend.
# d) Placement legality:
#    - We keep the original shared-column layout: each group's leftmost bar column doubles as the
#      legend column. The generator constrains that anchor bar's top cap to sit at least one blank
#      row below the future legend block, so adding the legend in the output never overlaps bars.
#    - All bars remain bottom-aligned, one column wide, and spaced by at least one empty column.
# e) Visual observability:
#    - No hidden numeric ordering is used. Ranking depends only on visible bar lengths, and the
#      selected bar is indicated by a visible recoloring to the cap color.
#    - Across examples, solvers can see the consistent pattern: "filled bar encodes rank; output
#      redraws the missing legend block of matching color and height."
# f) Visual cleanliness:
#    - The top-left legend zone is empty in the input, so nothing competes visually with the bars.
#      The output adds only simple one-column legend blocks in reserved columns above the bars.
# g) Color-change inferability:
#    - Each group still has one cap color and one fill color. The only color change in the input
#      is the already-highlighted bar turning from fill color F to cap color K, which is anchored
#      by the surrounding same-cap bars. This keeps the mapping easy to infer from few examples.


def _bottom_bars(g: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """
    Parse bottom-aligned one-column bars as tuples:
    (top_r, bottom_r, col, cap_color, interior_color, fill_length).

    The interior color may equal the cap color for the already-highlighted bar.
    """
    h, w = g.shape
    bg = get_bg(g)
    bottom = h - 1
    bars: list[tuple[int, int, int, int, int, int]] = []

    for c in range(w):
        if int(g[bottom, c]) == bg:
            continue

        top = bottom
        while top > 0 and int(g[top - 1, c]) != bg:
            top -= 1

        run = g[top : bottom + 1, c]
        assert len(run) >= 3, 'bar height must be at least 3'
        cap = int(run[0])
        assert int(run[-1]) == cap, 'bar must have matching top and bottom caps'

        interior = run[1:-1]
        fill = int(interior[0])
        assert np.all(interior == fill), 'bar interior must be a solid run'
        bars.append((top, bottom, c, cap, fill, int(len(interior))))

    return bars


def _legend_from_bars(g: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Infer legend entries as (legend_col, cap_color, rank).

    Groups are ordered left-to-right by their leftmost bar column, which is also the reserved
    legend column used by the generator.
    """
    bars = _bottom_bars(g)
    assert bars, 'there must be at least one bottom-aligned bar'

    grouped: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for bar in bars:
        grouped.setdefault(bar[3], []).append(bar)

    entries: list[tuple[int, int, int]] = []
    for cap, segs in grouped.items():
        segs.sort(key=lambda s: (-s[5], s[0], s[2]))
        selected = [idx for idx, seg in enumerate(segs) if seg[4] == cap]
        assert len(selected) == 1, f'cap {cap} must have exactly one pre-highlighted bar'
        rank = selected[0] + 1
        legend_col = min(seg[2] for seg in segs)
        entries.append((legend_col, cap, rank))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a blank legend from already-highlighted vertical bars.

    Input structure:
    - bg = background. The top-left legend zone is empty, but each group's leftmost bar column is
      still reserved to host the missing legend block in the output.
    - Main grid: bottom-aligned vertical bars with matching top/bottom caps. Within each cap-color
      group, exactly one bar already has its interior recolored to the cap color.

    Rule:
    - Group bars by cap color K and rank each group by interior length (longest = rank 1).
    - Find the already-highlighted bar in each group; its rank is the missing legend height k.
    - Draw a vertical legend block of color K and height k above that group's leftmost column.

    Key insight: The recolored bar is now evidence rather than the target transformation; it
    encodes which legend entry must be reconstructed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for legend_col, cap, rank in _legend_from_bars(g):
        out[:rank, legend_col] = cap

    return out


# --------------- generator (output-first inversion) ---------------

GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """
    Distribute bar columns with the same layout discipline as the original task.

    The first column of each group is the reserved legend column and therefore the group's
    leftmost bar column.
    """
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    gap_slots = total_bars - n_groups + 1
    extra_width = width - min_width
    extra_parts = rng.multinomial(extra_width, np.full(gap_slots, 1.0 / gap_slots))

    inter_gap_bonus = np.zeros(total_bars - 1, dtype=int)
    if total_bars > n_groups:
        inter_gap_bonus[n_groups - 1 :] = extra_parts[:-1]

    occupied = [0]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + COL_SPACING + int(inter_gap_bonus[idx]))

    legend_cols = occupied[:n_groups]
    extra_cols = occupied[n_groups:]

    group_columns = [[legend_cols[i]] for i in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(owners)
    for col, owner in zip(extra_cols, owners):
        group_columns[owner].append(col)

    return group_columns


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """
    Pick distinct fill lengths for one group.

    The anchor bar (the leftmost / legend-sharing column) must leave room for a future legend of
    height `rank` plus one blank row.
    """
    max_anchor_len = MAX_FILL_LEN - rank
    anchor_len = int(rng.integers(1, max_anchor_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != anchor_len]
    rng.shuffle(pool)
    return [anchor_len] + pool[: count - 1]


def _render_output(
    caps: list[int],
    fills: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
    width: int,
) -> np.ndarray:
    """Construct the full target output: bars already highlighted, plus the reconstructed legend."""
    out = np.zeros((GRID_H, width), dtype=int)

    for cap, fill, cols, lengths, rank in zip(caps, fills, group_columns, group_lengths, ranks):
        anchor_col = cols[0]
        paste(out, np.full((rank, 1), cap, dtype=int), 0, anchor_col)

        for col, fill_len in zip(cols, lengths):
            paste(out, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap

    return out


def _erase_legend(full_out: np.ndarray, group_columns: list[list[int]], ranks: list[int]) -> np.ndarray:
    """Remove the legend blocks to produce the variant input."""
    inp = full_out.copy()
    for cols, rank in zip(group_columns, ranks):
        inp[:rank, cols[0]] = 0
    return inp


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    """Try once to construct a valid (input, output) pair. Return None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, count + 1)) for count in counts]

    palette = rng.choice(np.arange(*COLOR_RANGE), size=2 * n_groups, replace=False)
    caps = [int(v) for v in palette[:n_groups]]
    fills = [int(v) for v in palette[n_groups:]]

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    group_lengths: list[list[int]] = []
    for count, rank in zip(counts, ranks):
        group_lengths.append(_choose_fill_lengths(rng, count, rank))

    expected = _render_output(caps, fills, group_columns, group_lengths, ranks, width)
    inp = _erase_legend(expected, group_columns, ranks)

    if not np.all(inp[0, :] == 0):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            inp, _out = sample
            return inp
    raise RuntimeError(f'failed to generate a valid input after {MAX_RETRIES} attempts')
