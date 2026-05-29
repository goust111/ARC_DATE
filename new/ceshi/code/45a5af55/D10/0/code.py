# Variant of 45a5af55
# Dimension: D10 — 符号编码方式 (symbol_instructions)
# Change: 色条高度直接编码框架厚度，色条颜色直接编码框架颜色（隐式编码，无显式图例）... → 添加显式颜色图例区域

from __future__ import annotations

import numpy as np

# Design analysis for the D10 variant
# a) Concrete behavior:
#    The right side still contains the original task's full-width stripe stack.
#    The left side now contains a 2-column palette legend, separated from the
#    stripes by a full background column. Each legend row is a pair
#    [stripe_color, frame_color]. The solver must read the legend first, then
#    read the stripe heights bottom->top, and finally build the nested square
#    output using mapped frame colors instead of the stripe colors themselves.
#    Example:
#      legend rows: [2,7], [4,1], [8,6]
#      stripe stack bottom->top: (8,h=2), (2,h=1), (4,h=3)
#      original logic would build colors [8,2,4]; new logic builds [6,7,1].
#    The only changed stage is color decoding. Thickness decoding by stripe
#    height is intentionally preserved.
# b) Complexity calibration:
#    The legend contains 3-6 mapping pairs, enough for repeated support without
#    overwhelming the grid. Stripe colors remain distinct to avoid ambiguous
#    run parsing. At least one mapping is non-identity so the variant is
#    visually different from the base task.
# c) Hidden assumptions in the original main():
#    - `col = g[r, 0]` assumed the leftmost column belonged to the stripe stack.
#      This fails once a side legend is introduced.
#    - `(g[rr] == col).all()` assumed each row was a uniform stripe across the
#      entire input width. That no longer holds because legend cells occupy the
#      left side. The variant crops to the stripe region before run parsing.
#    - `col` was used directly as the output color. This must be replaced by a
#      legend lookup.
# d) Placement legality:
#    The generator reserves columns 0-1 for legend pairs, column 2 as an all-0
#    separator, and columns 3: for the stripe rectangle. Legend and stripe
#    regions never overlap, stay in-bounds, and the stripe block occupies the
#    full grid height, which keeps run parsing simple and deterministic.
# e) Visual observability:
#    The left legend is compact and explicit. Because legend order is shuffled,
#    it cannot be mistaken for another sequential stripe stack; the right block
#    still visibly drives thicknesses, while the left pairs visibly explain the
#    color substitution.
# f) Visual cleanliness:
#    A full blank separator column cleanly partitions symbol instructions from
#    data. No object overlaps or mixed encodings occur in the same region.
# g) Color-change inferability:
#    Colors are limited to ARC colors 1..9, mappings are shown explicitly, and
#    each stripe color appearing on the right is guaranteed to appear exactly
#    once in the legend on the left.
# h) Generator validation:
#    The generator constructs the expected output from the sampled runs and
#    legend mapping, then checks `main(inp)` against that expected output.
# i) New solver assumptions checked against generator:
#    - exactly one full-background separator column exists: guaranteed.
#    - every used stripe color has one legend entry: guaranteed.
#    - the stripe region is a non-empty full-height rectangle of monochrome rows:
#      guaranteed by construction.


def _build_nested_square(runs: list[tuple[int, int]]) -> np.ndarray:
    """Build the nested-square output from bottom->top (color, height) runs."""
    assert runs
    c0, h0 = runs[0]
    cur = np.full((int(h0), int(h0)), int(c0), dtype=int)
    for col, h in runs[1:]:
        h = int(h)
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt
    return cur


def _parse_regions(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (bg, legend_region, stripe_region)."""
    g = np.array(grid, dtype=int)
    bg = 0 if bool((g == 0).any()) else int(np.unique(g)[0])
    sep_cols = [c for c in range(g.shape[1]) if bool((g[:, c] == bg).all())]
    assert sep_cols
    sep = int(sep_cols[0])
    assert 1 <= sep <= g.shape[1] - 2

    legend = g[:, :sep]
    stripes = g[:, sep + 1 :]
    assert legend.shape[1] >= 2
    assert stripes.shape[1] >= 1
    assert bool((stripes != bg).any())
    return bg, legend, stripes


def _parse_legend(legend: np.ndarray, bg: int) -> dict[int, int]:
    """Read unordered [source_color, target_color] pairs from the legend block."""
    mapping: dict[int, int] = {}
    for row in legend:
        vals = [int(v) for v in row if int(v) != bg]
        if not vals:
            continue
        assert len(vals) == 2
        src, dst = vals
        assert src not in mapping
        mapping[src] = dst
    assert mapping
    return mapping


def _parse_stripe_runs(stripes: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Parse bottom->top monochrome stripe runs from the stripe block."""
    active_rows = [r for r in range(stripes.shape[0]) if bool((stripes[r] != bg).any())]
    assert active_rows
    band = stripes[min(active_rows) : max(active_rows) + 1]

    runs: list[tuple[int, int]] = []
    r = int(band.shape[0] - 1)
    while r >= 0:
        row = band[r]
        col = int(row[0])
        assert col != bg
        assert bool((row == col).all())
        h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((band[rr] == col).all()):
            h += 1
            rr -= 1
        runs.append((col, h))
        r = rr
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from the right-hand stripe stack, but decode each
    stripe color through the explicit legend on the left.
    """
    bg, legend_region, stripe_region = _parse_regions(np.array(grid, dtype=int))
    mapping = _parse_legend(legend_region, bg)
    runs = _parse_stripe_runs(stripe_region, bg)
    mapped_runs = [(int(mapping[col]), int(h)) for col, h in runs]
    return _build_nested_square(mapped_runs)


def _sample_distinct_colors(n: int) -> list[int]:
    colors = np.random.permutation(np.arange(1, 10, dtype=int))[:n]
    return [int(c) for c in colors]


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(3, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    out_size = int(heights[0] + 2 * sum(heights[1:]))
    if out_size > 30:
        return None

    stripe_w = int(np.random.randint(6, 13))
    H = int(sum(heights))
    W = int(stripe_w + 3)

    src_colors = _sample_distinct_colors(n_runs)  # bottom -> top
    dst_colors = _sample_distinct_colors(n_runs)
    if all(int(a) == int(b) for a, b in zip(src_colors, dst_colors)):
        dst_colors[0], dst_colors[1] = dst_colors[1], dst_colors[0]

    mapping = {int(src): int(dst) for src, dst in zip(src_colors, dst_colors)}
    inp = np.zeros((H, W), dtype=int)

    legend_rows = np.random.permutation(np.arange(H))[:n_runs]
    legend_rows.sort()
    order = np.random.permutation(n_runs)
    for rr, idx in zip(legend_rows, order):
        inp[int(rr), 0] = int(src_colors[int(idx)])
        inp[int(rr), 1] = int(dst_colors[int(idx)])

    top = 0
    for src, h in reversed(list(zip(src_colors, heights))):
        inp[top : top + h, 3:] = int(src)
        top += h

    expected = _build_nested_square([(int(mapping[src]), int(h)) for src, h in zip(src_colors, heights)])
    if not np.array_equal(main(inp), expected):
        raise AssertionError("generated sample does not satisfy variant rule")
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
