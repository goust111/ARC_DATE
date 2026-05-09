# Variant of 97d7923e
# Dimension: D1 — 匹配条件：端点颜色分组 (match_cond)
# Change: s.endpoint_colors == d.color_group — 垂直条的顶部和底部帽色(cap color)与... → 条的长度区间分组

"""
Design analysis for the D1 variant
==================================

a) Concrete behavior and trace
--------------------------------
- Legend entries are still anchored in the top-left and still use height as a rank index.
- The new visible cue is width: each legend entry is a solid rectangle whose width is the
  size of one length bucket. Reading left-to-right, cumulative widths define consecutive
  length intervals. Example: widths [2, 3, 2] mean buckets [1..2], [3..5], [6..7].
- Bars remain vertical cap/fill/cap columns aligned to the bottom row, but cap colors no
  longer define groups. A single bucket intentionally contains mixed cap colors, and the
  same cap color appears in multiple buckets.
- Small trace:
  original logic: legend (color=red, height=2) -> collect red-capped bars -> sort by
  length -> recolor the 2nd longest red group bar to red.
  variant logic: legend (color=red, height=2, width=3) after a first width-2 block ->
  this block represents lengths [3..5] -> collect all bars with lengths 3, 4, or 5,
  regardless of cap colors -> sort by length -> recolor the 2nd longest one to red.
  The changed step is the match phase: color matching disappears and numeric interval
  membership replaces it.

b) Complexity calibration
-------------------------
- Use 2-3 buckets so the interval idea is visible but still learnable from 2-4 examples.
- Each bucket has width >= 2 and contains exactly one bar for each integer length in that
  interval. That guarantees repeated support for every bucket and removes ambiguity about
  "which lengths belong together".
- Cap colors are drawn from a tiny 2-color palette reused across buckets; fill colors are
  drawn from a tiny 2-color palette reused independently. This keeps the image tidy while
  making it obvious that cap color cannot be the grouping key.

c) Implicit-assumption audit against the original main()
--------------------------------------------------------
- Original `_vertical_sandwiches(g, cap)` assumes a logical group is recoverable from one
  cap color. That is false here, because a bucket mixes cap colors.
- Original loop `for ..., cap, k in legend:` assumes the legend color is also the bar cap
  color. That is false here; legend colors are output labels only.
- Original recolor `out[...] = cap` was harmless because legend color and cap color were
  identical. In the variant that would be wrong; the recolor target must use the legend
  color, not the bar's endpoint color.
- Original legend parser only needed `(x, color, height)`. The variant needs width too,
  because width carries the interval partition.

d) Placement legality
---------------------
- Legend and bars no longer share columns. The legend occupies a dedicated top-left region,
  and bars start after a fixed blank gap, so wide legend rectangles cannot collide with bar
  columns.
- Maximum total covered length is 7, preserving the original vertical fit: a length-7 bar
  has top cap on row 1 and bottom cap on row 9.
- Bars use unique columns chosen from the right-side bar region, so there is no overlap.

e/f/g) Observability, cleanliness, and color inference
------------------------------------------------------
- The width cue is directly visible: short/medium/long-style buckets are encoded by wider
  legend rectangles, and the bars show matching runs of consecutive lengths.
- Legend colors are distinct from all bar cap/fill colors, so the recoloring signal is easy
  to spot and cannot be confused with an existing endpoint color family.
- Because the same cap colors repeat across several buckets, any attempt to solve by color
  grouping fails on the examples themselves rather than only in hidden tests.

9) New solver assumptions guaranteed by the generator
-----------------------------------------------------
- Every legend entry is a solid top-touching rectangle with unique color, height = rank,
  width = bucket span, and the entries appear left-to-right in bucket order.
- The bucket spans partition lengths 1..T exactly, where T is the global maximum bar
  length. Every integer length in that range appears exactly once as a bar.
- Each bucket contains at least as many bars as its legend height, so rank lookup is valid.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break

from arc_utils import get_bg, neighbors4, paste


def _vertical_sandwiches(g: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """
    Find all vertical bar candidates as
    (top_r, bottom_r, col, cap_color, fill_color, fill_length).
    """
    h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int, int, int, int]] = []
    for c in range(w):
        colors = [int(v) for v in np.unique(g[:, c]) if int(v) != bg]
        for cap in colors:
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
                    segs.append((a, b, c, cap, fill, b - a - 1))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """
    Read top-left legend components as (x_pos, legend_color, rank_height, span_width).
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

    entries: list[tuple[int, int, int, int]] = []
    seen = np.zeros((h, w), dtype=bool)
    for c in cols:
        color = int(g[0, c])
        if color == bg or seen[0, c]:
            continue
        st = [(0, c)]
        seen[0, c] = True
        pix: list[tuple[int, int]] = []
        while st:
            r, cc = st.pop()
            pix.append((r, cc))
            for rr, ccc in neighbors4(r, cc):
                if (
                    0 <= rr < h
                    and 0 <= ccc <= legend_max_c
                    and (not seen[rr, ccc])
                    and int(g[rr, ccc]) == color
                ):
                    seen[rr, ccc] = True
                    st.append((rr, ccc))
        rs = [r for r, _ in pix]
        cs = [cc for _, cc in pix]
        x_pos = min(cs)
        rank_height = max(rs) - min(rs) + 1
        span_width = max(cs) - min(cs) + 1
        entries.append((x_pos, color, int(rank_height), int(span_width)))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor vertical bars by legend-defined length-bucket ranking.

    Input structure:
    - bg is the background. The top-left legend consists of solid rectangles touching row 0.
      Each rectangle has a unique color, a height that encodes a rank, and a width that
      encodes how many consecutive bar lengths belong to that bucket.
    - The main field contains bottom-aligned vertical cap/fill/cap bars with solid interiors.
      Cap colors may repeat across different buckets and are not the grouping key.

    Rule:
    - Read legend entries from left to right. Their cumulative widths partition bar lengths
      into consecutive intervals such as [1..2], [3..5], [6..7].
    - For each legend entry, collect every bar whose interior length falls in that interval,
      sort those bars by length descending, and select the rank given by the legend height.
    - Recolor the selected bar's interior to the legend entry's color.

    Key insight: the legend no longer names endpoint colors. Its widths define numeric length
    buckets, and its heights choose a rank within each bucket.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    bars = _vertical_sandwiches(g)
    assert bars, "grid must contain vertical bars"

    low = 1
    for _x, legend_color, rank, span_width in legend:
        high = low + span_width - 1
        group = [s for s in bars if low <= s[5] <= high]
        assert group, f"no bars found in length bucket [{low}, {high}]"
        group.sort(key=lambda s: (-s[5], s[0], s[2]))
        assert 1 <= rank <= len(group), f"legend rank {rank} out of range for {len(group)} bars"
        top_r, bot_r, col, _cap, _fill, _length = group[rank - 1]
        out[top_r + 1 : bot_r, col] = legend_color
        low = high + 1

    return out


# --------------- generator (O2) ---------------

GRID_H = 10
WIDTH_MIN = 12
WIDTH_MAX = 20
MIN_BUCKETS = 2
MAX_BUCKETS = 3
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
LEGEND_GAP = 1
BAR_REGION_GAP = 2
MAX_RETRIES = 400

_BOTTOM_ROW = GRID_H - 1


def _build_bar(cap: int, fill: int, fill_len: int) -> np.ndarray:
    bar = np.full((fill_len + 2, 1), fill, dtype=int)
    bar[0, 0] = cap
    bar[-1, 0] = cap
    return bar


def _compositions(total: int, parts: int, min_part: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)] if total >= min_part else []
    out: list[tuple[int, ...]] = []
    max_first = total - min_part * (parts - 1)
    for first in range(min_part, max_first + 1):
        for rest in _compositions(total - first, parts - 1, min_part):
            out.append((first, *rest))
    return out


_WIDTH_PATTERNS = {
    n: [
        widths
        for total in range(2 * n, MAX_FILL_LEN + 1)
        for widths in _compositions(total, n, 2)
    ]
    for n in range(MIN_BUCKETS, MAX_BUCKETS + 1)
}


def _expected_output(
    inp: np.ndarray,
    legend_colors: list[int],
    ranks: list[int],
    bucket_widths: list[int],
    bars: list[tuple[int, int, int]],
) -> np.ndarray:
    """
    `bars` stores (fill_length, top_r, col).
    """
    out = inp.copy()
    low = 1
    for legend_color, rank, span_width in zip(legend_colors, ranks, bucket_widths):
        high = low + span_width - 1
        group = [b for b in bars if low <= b[0] <= high]
        group.sort(key=lambda item: (-item[0], item[1], item[2]))
        fill_len, top_r, col = group[rank - 1]
        out[top_r + 1 : top_r + 1 + fill_len, col] = legend_color
        low = high + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_buckets = int(rng.integers(MIN_BUCKETS, MAX_BUCKETS + 1))
    bucket_widths = list(_WIDTH_PATTERNS[n_buckets][int(rng.integers(len(_WIDTH_PATTERNS[n_buckets])))])
    ranks = [int(rng.integers(1, span + 1)) for span in bucket_widths]
    total_bars = sum(bucket_widths)

    palette = rng.choice(np.arange(*COLOR_RANGE), size=n_buckets + 4, replace=False)
    legend_colors = [int(v) for v in palette[:n_buckets]]
    cap_palette = [int(v) for v in palette[n_buckets : n_buckets + 2]]
    fill_palette = [int(v) for v in palette[n_buckets + 2 : n_buckets + 4]]

    legend_width = sum(bucket_widths) + LEGEND_GAP * (n_buckets - 1)
    bar_start = legend_width + BAR_REGION_GAP
    min_width = max(WIDTH_MIN, bar_start + total_bars)
    if min_width > WIDTH_MAX:
        return None
    width = int(rng.integers(min_width, WIDTH_MAX + 1))

    grid = np.zeros((GRID_H, width), dtype=int)

    c = 0
    for legend_color, rank, span_width in zip(legend_colors, ranks, bucket_widths):
        paste(grid, np.full((rank, span_width), legend_color, dtype=int), 0, c)
        c += span_width + LEGEND_GAP

    bar_cols = sorted(int(v) for v in rng.choice(np.arange(bar_start, width), size=total_bars, replace=False))

    bars_to_place: list[tuple[int, int, int, int]] = []
    low = 1
    for bucket_idx, span_width in enumerate(bucket_widths):
        for offset in range(span_width):
            fill_len = low + offset
            cap = cap_palette[(bucket_idx + offset) % len(cap_palette)]
            fill = fill_palette[(bucket_idx + offset + 1) % len(fill_palette)]
            bars_to_place.append((fill_len, cap, fill, bucket_idx))
        low += span_width

    rng.shuffle(bars_to_place)

    bars_meta: list[tuple[int, int, int]] = []
    for col, (fill_len, cap, fill, _bucket_idx) in zip(bar_cols, bars_to_place):
        top_r = _BOTTOM_ROW - fill_len - 1
        paste(grid, _build_bar(cap, fill, fill_len), top_r, col)
        bars_meta.append((fill_len, top_r, col))

    expected = _expected_output(grid, legend_colors, ranks, bucket_widths, bars_meta)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
