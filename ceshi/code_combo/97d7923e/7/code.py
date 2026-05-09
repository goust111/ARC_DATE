# Combo variant of 97d7923e (index 7)
# combo_spec: D3/0+D11/2
# Name: 部分重着色+帽色也变

from pathlib import Path
import sys
import itertools

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
for _helper_dir in (_ROOT, _ROOT / "solvers"):
    _helper_dir_str = str(_helper_dir)
    if _helper_dir.is_dir() and _helper_dir_str not in sys.path:
        sys.path.append(_helper_dir_str)

from arc_utils import get_bg, neighbors4, paste


# Design analysis for combo D3/0 + D11/2
# ---------------------------------------
# a) concrete_behavior_and_trace:
#    The input scene stays close to the source task: the top-left legend still contains
#    one-column stacks of color K, and each stack height k selects the rank-k bar among
#    all cap/fill/cap bars whose endpoints are K. A literal "caps also recolor to legend
#    color K" reading would be invisible here because the caps already start as K, so the
#    D11 dimension would collapse away. To keep the fusion genuinely multi-dimensional and
#    visually learnable, the selected bar uses two coordinated writes:
#    - D3/0 part: recolor only the first k fill cells to the legend color K;
#    - D11/2 part: recolor both endpoint caps to the bar's own fill color F.
#    If a chosen bar is K / F F F F / K and k=2, then:
#    - original task -> K / K K K K / K;
#    - fused task ->  F / K K F F / F.
#    The output bar therefore has changed caps plus a partially recolored interior.
# b) complexity_calibration:
#    The task still uses only 1-4 groups and 2-3 bars per group, so rank values remain
#    in the small human-readable range 1..3. To keep the D3 part observable, the
#    generator forces the selected bar to satisfy fill_len > k. That guarantees every
#    chosen bar shows both recolored and unrecolored fill pixels instead of collapsing
#    back to a full-fill repaint.
# c) hidden_assumptions_audit:
#    _vertical_sandwiches relies on the input bars having solid interiors and consecutive
#    cap-color endpoints; the generator preserves that. Ranking must be computed from the
#    untouched input grid g, not from out, because earlier output writes would otherwise
#    create mixed columns and could corrupt later detection. The original single slice
#    write `out[top+1:bot] = cap` no longer matches the fused rule; it must become
#    "caps -> fill color, fill prefix -> cap color". A new solver assumption is k <=
#    fill_len for each selected target, and the generator strengthens it to fill_len > k
#    for visibility.
# d) placement_legality:
#    Bars remain bottom-aligned single-column sprites. The first bar of each group may
#    share the legend column, so the legend-column bar length is still capped by
#    MAX_FILL_LEN - rank to leave at least one background cell between the legend block
#    and the bar top cap. Distinct lengths inside each group keep ranking unambiguous.
# e) observability_and_cleanliness:
#    The fixed paint direction "from the top" avoids adding another hidden degree of
#    freedom. Because only the upper prefix of the chosen bar changes, the output stays
#    sparse and legible: each target looks like the top cap has grown downward by k cells
#    while the bottom of the bar keeps its original fill color. This is visually distinct
#    from both the source task and any whole-bar repaint.
# f) nondegeneracy_check_plan:
#    Post-edit validation will measure the selected bars' margins fill_len - k across
#    generated samples. Requiring the margin to be positive proves the generated dataset
#    does not degenerate to the original full-fill behavior on those samples. We also
#    compare the combo outputs against both the source solver and the D3-only solver; the
#    combo must differ from both, otherwise the supposed D11 contribution would still be
#    visually inert.


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


def _ranked_target(g: np.ndarray, cap: int, k: int) -> tuple[int, int, int, int]:
    segs = _vertical_sandwiches(g, cap)
    assert segs, f"legend cap {cap} must have matching vertical bars"
    segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
    assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
    top_r, bot_r, col, fill = segs[k - 1]
    fill_len = bot_r - top_r - 1
    assert k <= fill_len, f"legend height {k} exceeds target fill length {fill_len}"
    return top_r, bot_r, col, fill


def _apply_fused_recolor(out: np.ndarray, top_r: int, bot_r: int, col: int, cap: int, fill: int, k: int) -> None:
    out[top_r, col] = fill
    out[bot_r, col] = fill
    out[top_r + 1 : top_r + 1 + k, col] = cap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partially recolor legend-selected bars and also change both endpoint caps.

    Input structure:
    - The top-left legend is a short leftmost cluster of one-column monochrome stacks.
      Each stack has color K and height k, so the stack height is the visible selector.
    - The main field contains one-column vertical sandwiches: top cap K, solid fill F,
      bottom cap K. Bars sharing cap color K form one group with distinct fill lengths.

    Rule:
    - For each legend entry (K, k), collect all bars whose two endpoint pixels are K and
      sort them by fill length from longest to shortest.
    - Select the rank-k bar in that group.
    - Repaint only the first k fill cells of that chosen bar to K, counted downward from
      the top, while also recoloring the two endpoint cap cells to that bar's original
      fill color F.

    Key insight:
    - The same visible quantity k controls two linked choices at once: it picks which bar
      wins inside the K-group and it fixes how deep the K-colored prefix descends. The
      surviving lower fill tail plus the flipped endpoint caps reveal that two different
      recolor roles are being fused on the same selected bar.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        top_r, bot_r, col, fill = _ranked_target(g, cap, k)
        _apply_fused_recolor(out, top_r, bot_r, col, cap, fill, k)

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
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len + 2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns across groups with random inter-group spacing."""
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
    """Pick distinct lengths with a visible partial-recolor target."""
    max_legend_len = MAX_FILL_LEN - rank
    attempts = 0
    while attempts < 500:
        attempts += 1
        legend_len = int(rng.integers(1, max_legend_len + 1))
        pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != legend_len]
        rng.shuffle(pool)
        lengths = [legend_len] + pool[: count - 1]
        ranked = sorted(lengths, reverse=True)
        if ranked[rank - 1] > rank:
            return lengths
    raise RuntimeError(f"failed to choose visible partial lengths for count={count}, rank={rank}")


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        bot_r = _BOTTOM_ROW
        fill = int(inp[top_r + 1, target_col])
        _apply_fused_recolor(out, top_r, bot_r, target_col, cap, fill, rank)
    return out


def _sample_signature(
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> list[tuple[int, int, int]]:
    signature: list[tuple[int, int, int]] = []
    for cols, lengths, rank in zip(group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        _target_col, target_len = ranked[rank - 1]
        signature.append((rank, target_len, target_len - rank))
    return signature


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid. Returns None on oracle mismatch."""
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

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, fill, count, rank, cols in zip(caps, fills, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

    if any(margin <= 0 for _rank, _target_len, margin in _sample_signature(group_columns, group_lengths, ranks)):
        return None

    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
