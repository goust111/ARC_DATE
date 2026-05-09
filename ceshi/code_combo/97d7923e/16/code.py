# Combo variant of 97d7923e (index 16)
# combo_spec: D10/1+D13/1
# Name: 组内填充色各异+填充色消歧
#
# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior trace
#    Start from the original 97d7923e mechanic: each legend column says "within the
#    cap-color group K, recolor the rank-k bar after sorting by fill length
#    descending". The combo changes two linked parts of that story:
#    - D10/1 removes the old "one shared fill color per group" assumption. Bars with
#      the same cap still belong to the same group even if their interiors are all
#      different colors.
#    - D13/1 allows equal fill lengths, so length alone is no longer enough. Equal
#      lengths are resolved by a visible fill-color ladder rather than by hidden
#      generation order.
#    Small trace: suppose one cap group has three bars
#        col 0: cap 5, fill teal,   length 4
#        col 8: cap 5, fill yellow, length 4
#        col 12: cap 5, fill red,   length 2
#      and the legend height for cap 5 is 1.
#      Original solver assumptions would be wrong twice: it would expect one shared
#      fill color per group, and its fallback tie-break would pick the left bar at
#      col 0 among the two length-4 bars. In the combo, the group is still the set of
#      cap-5 bars, but the equal-length pair is ordered by fill color
#      yellow > orange > red > pink > teal, so the yellow bar at col 8 becomes rank 1
#      and is the one recolored.
# b) Complexity calibration
#    The new color channel is intentionally small. Fills are drawn only from the five
#    repeated hues [yellow, orange, red, pink, teal], and at least one group per grid
#    contains an equal-length pair whose order changes under that hue ladder. That
#    keeps the visual novelty obvious without turning the puzzle into a large arbitrary
#    color-mapping exercise.
# c) Hidden assumptions audit against the original solver
#    - `_vertical_sandwiches(...)` still works because each individual bar interior is
#      still a solid monochrome run. What disappears is the assumption that all bars in
#      one cap group share that same fill color.
#    - `_legend_entries(...)` is unchanged: legend height still stores rank, not size.
#    - The original `segs.sort(key=(-(length), top_row, col))` is no longer semantically
#      correct, because equal lengths must be ordered by fill hue before any spatial
#      fallback. The new solver inserts explicit fill-color priority there.
#    - The original generator's `_choose_fill_lengths` required all lengths to differ.
#      That must be replaced by a constructor that sometimes creates equal-length pairs
#      on purpose, while keeping their fill colors distinct.
# d) Placement legality
#    Bars remain one column wide and bottom-aligned, with at least one blank column
#    between occupied columns. The legend still sits in the top-left using the first
#    column of each group. The legend-column bar length is capped by `MAX_FILL_LEN-rank`
#    so there is still a background gap row below the legend block before the bar
#    begins. No objects overlap and all pastes stay within bounds.
# e) Visual observability
#    A human can see that same-cap bars no longer share one body color, because every
#    group contains multiple fill hues. The generator also forces at least one example
#    feature where two equal-length bars differ only by fill hue and the warmer-looking
#    one wins, so the new tie-break is not merely theoretical.
# f) Visual cleanliness
#    The geometry is unchanged from the source task: thin separated columns plus a
#    compact legend. Extra color variation is confined inside bar interiors, so the
#    scene stays readable instead of adding more shapes or overlays in the same area.
# g) Inferability of the change
#    The solver never relies on numeric color ids as a task rule. The tie-break order is
#    documented and generated as a visible warm-to-cool ladder
#    yellow > orange > red > pink > teal. The generator reuses these few hues across
#    samples and rejects cases where the combo output would accidentally match the
#    original length-only ordering, preventing silent regression to the base task.
#
# Post-change assumption audit
#    - `main()` assumes every equal-length decision can be resolved by the visible fill
#      order before spatial fallbacks matter. The generator enforces distinct fill
#      colors within each cap group, so no generated tie needs a hidden fallback.
#    - `main()` still assumes each legend color appears on at least one valid vertical
#      cap-fill-cap bar. The generator constructs bars directly from that schema.
#    - The generator now assumes at least one group's ranked target differs from the
#      original spatial tie-break. A rejection check compares combo output against a
#      legacy baseline and resamples if they match.

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import get_bg, neighbors4, paste


# Equal-length bars are ordered by a small visible hue ladder rather than by color id.
# Earlier entries win the tie-break.
FILL_TIE_ORDER = (4, 7, 2, 6, 8)  # yellow > orange > red > pink > teal
_FILL_PRIORITY = {color: idx for idx, color in enumerate(FILL_TIE_ORDER)}


def _fill_priority(color: int) -> int:
    """Visible warm-to-cool ordering used only for equal-length ties."""
    return _FILL_PRIORITY.get(int(color), len(_FILL_PRIORITY) + int(color))


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
    the top row within the legend cluster (i.e., how many stacked blocks).
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


def _combo_sort_key(seg: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Primary rank is fill length; equal lengths use visible fill-hue priority."""
    top_r, bot_r, col, fill = seg
    fill_len = int(bot_r - top_r - 1)
    return (-fill_len, _fill_priority(fill), top_r, col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor one bar per cap-color group using legend ranks plus fill-color tie-breaks.

    Input structure:
    - Background is 0. In the top-left, each legend entry is a vertical stack of one cap color K.
      The height k of that stack is a rank index.
    - Elsewhere are bottom-aligned vertical bars of the form cap / solid fill / cap.
      Bars in the same group share the same cap color K, but their fill colors may all differ.
    - Bars of the same cap color may also share the same fill length.

    Rule:
    - For each legend entry (K, k), collect all bars whose two caps are color K.
    - Sort those bars by fill length descending.
    - If multiple bars have the same fill length, break that tie by the visible fill-color
      ladder yellow > orange > red > pink > teal, then keep the old spatial fallback only for safety.
    - Recolor the interior of the rank-k bar to cap color K.

    Key insight:
    The legend still encodes a rank, not a size. The new difficulty is that equal-length bars are
    disambiguated by their interior hue, while fill-color differences do not split the cap-color group.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=_combo_sort_key)
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COL_SPACING = 2
MAX_RETRIES = 500

CAP_POOL = (1, 3, 5, 9)  # blue, green, grey, maroon
FILL_POOL = FILL_TIE_ORDER

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [counts for counts in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(counts) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng: np.random.Generator, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns across groups; each group's first column also hosts its legend."""
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


def _legacy_sort_key(seg: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Original task ordering: length first, then top row, then column."""
    top_r, bot_r, col, _fill = seg
    return (-(bot_r - top_r - 1), top_r, col)


def _spec_to_seg(spec: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Convert (col, fill_len, fill) into the solver's segment tuple."""
    col, fill_len, fill = spec
    top_r = _MAX_CAP_ROW - fill_len
    bot_r = _BOTTOM_ROW
    return (top_r, bot_r, col, fill)


def _sample_non_tie_group(
    rng: np.random.Generator, cols: list[int], rank: int
) -> list[tuple[int, int, int]]:
    """Construct one group with distinct fills and distinct lengths."""
    count = len(cols)
    legend_max_len = MAX_FILL_LEN - rank
    legend_len = int(rng.integers(1, legend_max_len + 1))
    other_lengths = [length for length in range(1, MAX_FILL_LEN + 1) if length != legend_len]
    rng.shuffle(other_lengths)
    lengths = [legend_len] + other_lengths[: count - 1]
    fills = [int(v) for v in rng.choice(np.array(FILL_POOL), size=count, replace=False)]
    return [(int(col), int(fill_len), int(fill)) for col, fill_len, fill in zip(cols, lengths, fills)]


def _pick_tie_pair_colors(rng: np.random.Generator) -> tuple[int, int]:
    """Pick (cooler_left, warmer_right) so color order reverses the old left-to-right tie-break."""
    pairs = [
        (left, right)
        for left in FILL_POOL
        for right in FILL_POOL
        if left != right and _fill_priority(left) >= _fill_priority(right) + 2
    ]
    left, right = pairs[int(rng.integers(len(pairs)))]
    return int(left), int(right)


def _sample_tie_group(
    rng: np.random.Generator, cols: list[int]
) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Construct one group whose target rank genuinely depends on fill-color tie-breaking.

    The legend-column bar is always one member of the equal-length pair, while a later
    column gets the warmer fill. That guarantees the combo ordering disagrees with the
    original left-to-right fallback inside the tied bucket.
    """
    count = len(cols)
    assert count in BARS_PER_GROUP

    if count == 2:
        rank = int(rng.integers(1, 3))
        pair_len = int(rng.integers(1, MAX_FILL_LEN - rank + 1))
        left_fill, right_fill = _pick_tie_pair_colors(rng)
        return rank, [(cols[0], pair_len, left_fill), (cols[1], pair_len, right_fill)]

    rank = int(rng.integers(1, 4))
    legend_max_len = MAX_FILL_LEN - rank

    if rank == 1:
        pair_len = int(rng.integers(2, legend_max_len + 1))
        other_len = int(rng.integers(1, pair_len))
    elif rank == 3:
        pair_len = int(rng.integers(1, legend_max_len + 1))
        other_len = int(rng.integers(pair_len + 1, MAX_FILL_LEN + 1))
    else:
        make_pair_longest = bool(rng.integers(0, 2))
        if make_pair_longest and legend_max_len >= 2:
            pair_len = int(rng.integers(2, legend_max_len + 1))
            other_len = int(rng.integers(1, pair_len))
        else:
            pair_len = int(rng.integers(1, legend_max_len + 1))
            other_len = int(rng.integers(pair_len + 1, MAX_FILL_LEN + 1))

    left_fill, right_fill = _pick_tie_pair_colors(rng)
    extra_fill_choices = [color for color in FILL_POOL if color not in {left_fill, right_fill}]
    extra_fill = int(extra_fill_choices[int(rng.integers(len(extra_fill_choices)))])

    return rank, [
        (cols[0], pair_len, left_fill),
        (cols[1], pair_len, right_fill),
        (cols[2], other_len, extra_fill),
    ]


def _render_input(
    caps: list[int], group_specs: list[list[tuple[int, int, int]]], ranks: list[int], width: int
) -> np.ndarray:
    """Render the legend columns plus all bars."""
    grid = np.zeros((GRID_H, width), dtype=int)
    for cap, specs, rank in zip(caps, group_specs, ranks):
        legend_col = specs[0][0]
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, legend_col)
        for col, fill_len, fill in specs:
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)
    return grid


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_specs: list[list[tuple[int, int, int]]],
    ranks: list[int],
    *,
    legacy: bool = False,
) -> np.ndarray:
    """Compute either the combo output or the old length-only baseline output."""
    out = inp.copy()
    sorter = _legacy_sort_key if legacy else _combo_sort_key
    for cap, specs, rank in zip(caps, group_specs, ranks):
        ranked = sorted((_spec_to_seg(spec) for spec in specs), key=sorter)
        top_r, bot_r, col, _fill = ranked[rank - 1]
        out[top_r + 1 : bot_r, col] = cap
    return out


def _tie_groups_metric(group_specs: list[list[tuple[int, int, int]]], ranks: list[int]) -> int:
    """How many groups have duplicate lengths and whose selected rank lies inside a tied bucket."""
    active = 0
    for specs, rank in zip(group_specs, ranks):
        ranked = sorted((_spec_to_seg(spec) for spec in specs), key=_combo_sort_key)
        lengths = [bot_r - top_r - 1 for top_r, bot_r, _col, _fill in ranked]
        target_len = lengths[rank - 1]
        if lengths.count(target_len) >= 2:
            active += 1
    return active


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid non-degenerate combo grid."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    caps = [int(v) for v in rng.choice(np.array(CAP_POOL), size=n_groups, replace=False)]

    tie_group = int(rng.integers(n_groups))
    group_specs: list[list[tuple[int, int, int]]] = []
    ranks: list[int] = []
    for idx, cols in enumerate(group_columns):
        if idx == tie_group:
            rank, specs = _sample_tie_group(rng, cols)
        else:
            rank = int(rng.integers(1, len(cols) + 1))
            specs = _sample_non_tie_group(rng, cols, rank)
        ranks.append(rank)
        group_specs.append(specs)

    grid = _render_input(caps, group_specs, ranks, width)
    expected = _expected_output(grid, caps, group_specs, ranks)
    legacy = _expected_output(grid, caps, group_specs, ranks, legacy=True)

    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(expected, legacy):
        return None
    if _tie_groups_metric(group_specs, ranks) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
