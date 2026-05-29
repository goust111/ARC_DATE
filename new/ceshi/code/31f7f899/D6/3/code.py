# Variant of 31f7f899
# Dimension: D6 — 基线结构 (src)
# Change: 单条水平基线跨越网格全宽，位于网格中间行（H//2），颜色固定为6... → 基线颜色随机化

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, find_components, get_bg, paste


# Design analysis for the D6 variant
# ==================================
# a) Concrete behavior and trace
# Variant inputs now show a visibly monochrome baseline: the middle row H//2 is
# entirely filled with one non-background color chosen per sample, not fixed to 6.
# Bars stand immediately above that row, so the baseline itself stays intact and can
# be recognized structurally as "the only full-width monochrome non-bg row".
#
# To make the new baseline-identification step operational rather than cosmetic,
# generator always includes one slot whose bar color equals the baseline color.
# Example:
# - base row = 4, baseline color = 3
# - slot colors left->right = [5, 3, 2]
# - input heights by slot = [3, 1, 2]
# - output heights by slot = [1, 2, 3]
# Old logic:
# - infer base color 3 from dominant row
# - skip every component with color 3
# - the middle slot bar disappears because its color matches the baseline
# Variant logic:
# - find row 4 because the whole row is uniform non-bg
# - temporarily blank that row before component extraction
# - recover all three bars, including the bar colored 3
# - sort shapes by height and repaint them into the original left-to-right slots
#
# b) Complexity calibration
# The change is kept narrowly on D6:
# - same global rule: sort bar heights and redistribute shapes into fixed slots
# - same basic objects: monochrome vertical rectangles of width 1 or 3
# - added only the baseline-structure cue and the possibility that a bar shares the
#   baseline color
# This is enough to make the variant visibly different across 2-4 examples without
# adding extra matching dimensions or ambiguous object classes.
#
# c) Implicit-assumption audit against the original main()
# - Original "best row/color by maximal count" was only a proxy. In this variant we
#   require a direct structural test: full-width, monochrome, non-bg row.
# - Original "if col in {bg, base_col}: continue" assumed no bar could legitimately
#   use the baseline color. That is false here, so we first remove the baseline row
#   from a working copy and only then run connected-components.
# - Original anchor extraction preferred cells lying on the base row. Bars no longer
#   occupy that row, so anchor must come from the component's own columns.
# - Original output rewrite of the base row stays valid because the baseline remains
#   an unchanged full-width row in both input and output.
#
# d) Placement legality
# - n is odd so base_r = H//2 is well-defined.
# - Every bar is pasted in rows [base_r-height, base_r), so it never overwrites the
#   baseline and never leaves the grid.
# - Slots use width 1 or 3 with at least one empty column between neighboring bars.
# - Heights are unique per sample, so ordering by height is unambiguous.
# - Removing the baseline row from a working copy disconnects same-color bars from the
#   baseline before component extraction.
#
# e) Visual observability
# Humans can directly see:
# - the middle row is a solid colored stripe, but the stripe color varies by example
# - one bar often uses exactly that same color, so "skip all color-X objects" is not
#   a valid rule
# - outputs preserve slot colors and baseline color while only bar heights move
# This makes the D6 change visible and logically necessary rather than a hidden
# implementation detail.


def _find_baseline(g: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    H, W = g.shape
    for r in range(H):
        row = g[r, :]
        first = int(row[0])
        if first == bg:
            continue
        if np.all(row == first):
            candidates.append((int(r), int(first)))

    if candidates:
        # Variant generator guarantees exactly one such row. If malformed inputs create
        # multiple candidates, prefer the one closest to the prescribed middle row.
        return min(candidates, key=lambda rc: (abs(int(rc[0]) - H // 2), int(rc[0])))

    # Safety fallback for non-variant inputs: keep the old proxy behavior.
    best = None
    for r in range(H):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _extract_objects(g: np.ndarray, base_r: int, bg: int) -> list[dict[str, object]]:
    work = np.array(g, copy=True)
    work[base_r, :] = int(bg)

    objects: list[dict[str, object]] = []
    for comp in find_components(work, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        pts = comp["coords"]
        cols_sorted = sorted(int(c) for _r, c in pts)
        anchor = int(cols_sorted[len(cols_sorted) // 2])
        b = comp["bbox"]
        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(comp["color"]),
                "anchor": int(anchor),
                "height": int(bbox_h(b)),
                "size": int(len(rel)),
                "rel": rel,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Identify the baseline by structure, not by a fixed color value.
    - The baseline is the full-width monochrome non-bg row at H//2.
    - Bars above the baseline are sorted by height and reassigned to left-to-right
      slots, while each slot keeps its original color identity.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_baseline(g, bg)

    objects = _extract_objects(g, base_r, bg)
    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height), int(left))
    grid[base_r, :] = int(base_color)
    return grid


def _pick_colors(k: int, bg: int) -> tuple[int, list[int]]:
    palette = [c for c in range(10) if c != bg]
    base_color = int(np.random.choice(palette))
    other_colors = [c for c in palette if c != base_color]
    chosen = [int(c) for c in np.random.choice(other_colors, size=k - 1, replace=False)]
    colors = [int(base_color), *chosen]
    np.random.shuffle(colors)
    return int(base_color), colors


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_r = n // 2
    bg = 8

    height_choices = list(range(1, base_r + 1))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    base_color, colors = _pick_colors(k, bg)

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, sorted_heights)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_heights = [int(sorted_heights[i]) for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, in_heights)

    if np.array_equal(inp, out):
        return None
    if not any(color == base_color for color in colors):
        return None
    if not np.all(inp[base_r, :] == base_color):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")


# New implicit assumptions introduced by the variant solver
# =========================================================
# - There is exactly one non-background row that is uniform across the full width.
# - All sortable objects become separate monochrome components after removing that row.
# - Bar centers (anchors) are stable under redistribution because all bars use the same
#   width within a sample and have at least one empty column between neighboring slots.
# - At least one bar may share the baseline color, so generator must preserve the
#   structural cue (full-width baseline row) that lets solver separate row vs. bar.
