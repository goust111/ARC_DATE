# Variant of 97d7923e
# Dimension: D5 — 目标结构：图例的编码方式 (symbol_instructions)
# Change: 图例在左上角：每个条目为一列，从第0行开始向下堆叠k个同色像素，k编码该帽色组内的目标排名；条目间间隔1-2列背景色... → 图例用不同形状编码排名

"""
Design analysis for the D5 variant.

a) Concrete behavior and trace
   - Original legend: one colored vertical stack per group; height k directly meant rank k.
   - Variant legend: one colored glyph per group in the upper-left, still touching row 0 and
     ordered left-to-right, but rank is encoded by glyph class:
       rank 1 -> dot
       rank 2 -> small L
       rank 3 -> small T
   - On an orthogonal pixel grid, literal "2-pixel L" and "3-pixel T" are ambiguous /
     impossible as clear connected symbols, so this variant uses the smallest unambiguous
     orthogonal glyphs instead:
       dot = [[1]]
       L   = [[1,0],
              [1,1]]
       T   = [[1,1,1],
              [0,1,0]]
     The semantic change is still D5: rank is read from symbol shape rather than a stack height.
   - Example trace:
       old: cap color 4 with legend height 2 -> choose 2nd longest bar capped by 4.
       new: cap color 4 with an L-shaped legend glyph -> choose 2nd longest bar capped by 4.
       difference: the solver must classify the upper-left glyph before ranking bars; counting
       vertical extent alone no longer works because L and T both have height 2.

b) Complexity calibration
   - Rank remains a small 3-way category, which keeps the rule inferable from 2-4 examples.
   - The shapes are visually distinct and repeated once per color group, so the clue is present
     several times in a sample set without adding too many free parameters.
   - To keep the wider glyphs physically placeable in a 10x20 board without legend/bar overlap,
     the generator reduces density slightly: at most 3 groups and at most 8 total bars.

c) Implicit assumptions in the original main()
   - Original `_legend_entries()` assumed every legend entry is a 4-connected monochrome column
     whose vertical height equals the target rank. That fails here because:
       * rank 2 and rank 3 are no longer one-column stacks;
       * two different ranks can share the same height;
       * glyph width now matters.
   - `_vertical_sandwiches()` still works unchanged because bar geometry is unchanged.
   - The recoloring loop still works once legend parsing returns `(x_pos, cap_color, rank)`.

d) Placement legality
   - Legend glyphs start at row 0 and are placed left-to-right with 1-2 background columns
     between glyph bounding boxes.
   - Only the leftmost column of each legend glyph is also reused as that group's first bar
     column; the rest of the glyph footprint is reserved, so the legend stays readable.
   - Bars remain bottom-aligned cap/fill/cap columns. The reused legend column gets a stricter
     max fill length so the bar's top cap starts at least one background row below the glyph.

e) Visual observability
   - Human solvers can now see that the changed object is the legend itself: some entries are
     single dots, some are Ls, some are Ts.
   - The old "count the stack depth" heuristic breaks because L and T both span 2 rows.

f) Visual cleanliness
   - The legend stays confined to the upper-left corner.
   - Bars are pushed to the right of the legend region except for the one shared anchor column
     per group, so glyphs and bars do not interleave chaotically.

g) Color-change inferability
   - Color behavior is unchanged: each legend color points to the bar group with the same cap
     color, and the chosen bar is recolored to that cap color.
   - The only new inference burden is symbol classification, which is structurally signaled by
     the legend area itself rather than hidden in numeric color ids.

Post-change assumption review for main()
   - Parsing assumes every legend entry touches row 0, starts at its own leftmost top-row pixel,
     and matches exactly one of the predefined glyph templates. The generator enforces this.
   - Parsing assumes there are only background cells between adjacent glyph bounding boxes. The
     generator enforces 1-2 blank columns between entries.
   - Recoloring assumes every legend color has at least as many bars as its decoded rank. The
     generator chooses rank in `1..bars_in_group`.
"""

import itertools

import numpy as np

from arc_utils import get_bg, paste


_LEGEND_TEMPLATES = {
    1: np.array([[1]], dtype=int),
    2: np.array([[1, 0], [1, 1]], dtype=int),
    3: np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
}


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


def _symbol_sprite(color: int, rank: int) -> np.ndarray:
    templ = _LEGEND_TEMPLATES[rank]
    sprite = np.zeros_like(templ)
    sprite[templ == 1] = int(color)
    return sprite


def _matches_symbol(g: np.ndarray, bg: int, start_c: int, color: int, rank: int) -> bool:
    templ = _LEGEND_TEMPLATES[rank]
    th, tw = templ.shape
    if th > g.shape[0] or start_c + tw > g.shape[1]:
        return False
    sub = g[:th, start_c : start_c + tw]
    return bool(np.all(sub[templ == 1] == color) and np.all(sub[templ == 0] == bg))


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend is in the top-left, touching row 0, with 1-2 bg columns between glyphs.

    Returns entries as (x_pos, legend_color, rank), sorted left-to-right.
    Rank is decoded from the glyph template rather than from component height.
    """
    h, w = g.shape
    assert h >= 2, "grid must be tall enough for legend glyphs"
    assert int(g[0, 0]) != bg, "legend must start at the left edge"

    entries: list[tuple[int, int, int]] = []
    c = 0
    while c < w:
        if int(g[0, c]) == bg:
            if entries and not np.any(g[0, c:] != bg):
                break
            c += 1
            continue

        color = int(g[0, c])
        match = None
        for rank in (3, 2, 1):
            if _matches_symbol(g, bg, c, color, rank):
                tw = _LEGEND_TEMPLATES[rank].shape[1]
                match = (rank, tw)
                break
        assert match is not None, f"unrecognized legend symbol at col {c}"

        rank, tw = match
        entries.append((c, color, rank))
        c += tw
        while c < w and int(g[0, c]) == bg:
            c += 1

    assert entries, "legend must be non-empty"
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor one vertical bar per cap-color group using a shape-coded legend.

    Input structure:
    - Background fills most of the board. In the upper-left, each legend entry is a monochrome
      glyph touching row 0; its color matches one cap-color group, and its glyph class encodes
      the target rank.
    - Elsewhere, the objects are still bottom-aligned vertical cap/fill/cap bars with solid
      interiors.

    Rule:
    - Decode each legend glyph as rank 1 (dot), rank 2 (L), or rank 3 (T).
    - For each legend color K, find all bars capped by K and sort them by interior length
      descending, using row and column only as tie-breakers.
    - Recolor the interior of the rank-selected bar to K.

    Key insight: The upper-left marks no longer tell you the rank by stack height; they are
    symbolic templates whose shapes must be recognized before selecting among length-ranked bars.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    for _x, cap, rank in legend:
        segs = _vertical_sandwiches(g, cap)
        assert segs, f"legend cap {cap} must have matching vertical bars"
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= rank <= len(segs), f"legend rank {rank} out of range for {len(segs)} bars"
        top_r, bot_r, col, _fill = segs[rank - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 3
MAX_TOTAL_BARS = 8
BARS_PER_GROUP = (2, 3)
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / fill*N / cap."""
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _legend_symbol_height(rank: int) -> int:
    return int(_LEGEND_TEMPLATES[rank].shape[0])


def _legend_symbol_width(rank: int) -> int:
    return int(_LEGEND_TEMPLATES[rank].shape[1])


def _layout_columns(
    rng: np.random.Generator, counts: list[int], ranks: list[int]
) -> tuple[int, list[int], list[list[int]]]:
    """
    Place one legend anchor column per group in the top-left, then distribute the
    remaining bar columns to the right of the legend region with one blank column
    between bars.
    """
    n_groups = len(counts)
    widths = [_legend_symbol_width(rank) for rank in ranks]

    gaps = [1] * max(0, n_groups - 1)
    extra_bars = sum(counts) - n_groups
    tail_gap = 1

    while True:
        starts = [0]
        for width, gap in zip(widths, gaps):
            starts.append(starts[-1] + width + gap)
        starts = starts[:n_groups]
        legend_end = starts[-1] + widths[-1]

        first_extra = legend_end + tail_gap
        width_needed = legend_end if extra_bars == 0 else first_extra + 2 * (extra_bars - 1) + 1
        if width_needed > WIDTH_MAX:
            raise ValueError("legend layout does not fit width cap")

        room = WIDTH_MAX - width_needed
        adjustable = [i for i in range(len(gaps)) if gaps[i] == 1]
        if room > 0 and adjustable:
            idx = int(rng.choice(adjustable))
            gaps[idx] = 2
            continue
        break

    width = int(rng.integers(max(WIDTH_MIN, width_needed), WIDTH_MAX + 1))
    group_columns = [[start] for start in starts]

    if extra_bars > 0:
        extra_cols = [first_extra + 2 * i for i in range(extra_bars)]
        owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
        rng.shuffle(owners)
        for col, owner in zip(extra_cols, owners):
            group_columns[owner].append(int(col))

    return width, starts, group_columns


def _choose_fill_lengths(rng: np.random.Generator, count: int, rank: int) -> list[int]:
    """
    Pick distinct fill lengths for one group.

    The first length belongs to the legend anchor column, so its bar must start
    at least one bg row below the legend glyph.
    """
    symbol_h = _legend_symbol_height(rank)
    max_anchor_len = _MAX_CAP_ROW - (symbol_h + 1)
    shared_len = int(rng.integers(1, max_anchor_len + 1))
    pool = [length for length in range(1, MAX_FILL_LEN + 1) if length != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-selected bar to its cap color."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: (-item[1], item[0]))
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


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

    try:
        width, legend_starts, group_columns = _layout_columns(rng, counts, ranks)
    except ValueError:
        return None

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []

    for cap, fill, count, rank, start_c, cols in zip(caps, fills, counts, ranks, legend_starts, group_columns):
        paste(grid, _symbol_sprite(cap, rank), 0, start_c)

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            paste(grid, _build_bar(cap, fill, fill_len), _MAX_CAP_ROW - fill_len, col)

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
