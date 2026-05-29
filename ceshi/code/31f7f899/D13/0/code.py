# Variant of 31f7f899
# Dimension: D13 — 条形对称性约束 (src)
# Change: 条形关于基线行上下对称（上方延伸 = 下方延伸）... → 条形非对称（上下延伸不等）

from __future__ import annotations

import numpy as np
from arc_utils import find_components, get_bg, paste


# Design analysis for the D13 variant
# a) Concrete behavior:
#    The scene still contains one full-width base line and several monochrome vertical
#    bars sitting on that line. The original generator made every bar vertically
#    symmetric around the base row by pasting an odd-height rectangle centered on
#    `base_r`; for example, height 7 meant 3 cells above the base, the base cell
#    itself, and 3 cells below. In this variant each bar is still a solid rectangle
#    with fixed width, but it is described by `(top_extent, bottom_extent)` and we
#    require `top_extent != bottom_extent`. Example on a 9x9 grid with base row 4:
#    a bar may occupy rows 1..6 (top=3, bottom=2) while another occupies rows 3..8
#    (top=1, bottom=4). The output still keeps slot positions/colors fixed, but the
#    redistributed shapes are ordered by `top_extent` only, not by total height.
#    So if the input has bars `(top,bottom) = (1,4), (2,1), (3,2)`, then the output
#    places them left-to-right as `(1,4), (2,1), (3,2)` even though totals are
#    `6,4,6` and the total-height order would disagree.
# b) Complexity calibration:
#    Keeping rectangles and fixed widths preserves the original task skeleton, so the
#    only new freedom is vertical asymmetry. To make the new rule observable, every
#    generated example enforces three conditions: all bars are asymmetric, the order
#    induced by `top_extent` differs from the order induced by `bottom_extent`, and
#    it also differs from the order induced by total height. This prevents the
#    variant from collapsing back to the old task while keeping the hypothesis space
#    small enough for ARC-style induction.
# c) Hidden-assumption audit of the original `main()`:
#    - Base-row detection via the most frequent non-bg color on a row is still valid,
#      because the base line remains the only long horizontal run of one color.
#    - Component extraction and slot anchoring from the base row still work, because
#      each bar still crosses the base row and remains a single monochrome component.
#    - `bbox_h(...)` as the ordering key is no longer valid: with asymmetric bars,
#      total height mixes the upper and lower extents and would sort some examples
#      incorrectly. The solver must instead measure `base_r - min_row` for each bar.
#    - Repainting via relative coordinates remains valid; asymmetry is carried by the
#      stored coordinate set, so no special placement logic is needed after ranking.
# d) Placement legality:
#    The generator uses odd-sized grids with `base_r = n//2`, so rows above and below
#    the base have equal capacity. Each bar is placed as a rectangle of height
#    `top+bottom+1` with top row `base_r-top`, which guarantees inclusion of the base
#    row, no out-of-bounds writes, and no overlap because slots are horizontally
#    separated by at least one background column.
# e) Visual observability:
#    A human can compare how far each bar rises above the base line in the input and
#    output. The lower tails vary independently and visibly remain attached to the
#    moved shapes, so across 2-4 examples the consistent pattern is that the top
#    reaches become left-to-right sorted while colors stay bound to slots.


def _extract_objects(grid: np.ndarray, base_r: int, base_col: int, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        rows = [int(r) for r, _c in pts]
        cols = [int(c) for _r, c in pts]
        top_extent = int(base_r - min(rows))
        bottom_extent = int(max(rows) - base_r)
        total_height = int(top_extent + bottom_extent + 1)

        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            anchor = int(sorted(cols)[len(cols) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "top": int(top_extent),
                "bottom": int(bottom_extent),
                "height": int(total_height),
                "size": int(len(rel)),
                "rel": rel,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Detect the shared base line.
    - Treat each non-base component as one vertical bar that may extend by different
      amounts above and below the base row.
    - Sort slots by x-position.
    - Sort bar shapes by their upward extent above the base row (shortest rise to
      tallest rise).
    - Move the i-th smallest upward-reaching shape into the i-th leftmost slot and
      recolor it to that slot's original color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    best = None  # (count, row, color)
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

    objects = _extract_objects(g, int(base_r), int(base_col), int(bg))
    assert objects

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (
            int(o["top"]),
            int(o["height"]),
            int(o["bottom"]),
            int(o["size"]),
            int(o["anchor"]),
            int(o["color"]),
        ),
    )

    out = np.full_like(g, bg)
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
    bars: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, color, (top, bottom) in zip(lefts, colors, bars):
        patch = np.full((int(top + bottom + 1), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - top), int(left))
    return grid


def _rank_order(values: list[int]) -> tuple[int, ...]:
    return tuple(i for i, _v in sorted(enumerate(values), key=lambda iv: (int(iv[1]), int(iv[0]))))


def _sample_asymmetric_bars(k: int, max_extent: int) -> list[tuple[int, int]] | None:
    top_choices = sorted(int(x) for x in np.random.choice(np.arange(1, max_extent + 1), size=k, replace=False))
    tops = [int(x) for x in top_choices]

    for _ in range(80):
        perm = np.random.permutation(k)
        if np.array_equal(perm, np.arange(k)):
            continue
        bottoms = [int(tops[i]) for i in perm]
        if any(t == b for t, b in zip(tops, bottoms)):
            continue
        totals = [int(t + b + 1) for t, b in zip(tops, bottoms)]
        if _rank_order(bottoms) == tuple(range(k)):
            continue
        if _rank_order(totals) == tuple(range(k)):
            continue
        return [(int(t), int(b)) for t, b in zip(tops, bottoms)]
    return None


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    max_count = min(5, base_r)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    bars_sorted = _sample_asymmetric_bars(k, base_r)
    if bars_sorted is None:
        return None

    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, bars_sorted)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    bars_input = [bars_sorted[int(i)] for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, bars_input)

    if np.array_equal(inp, out):
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
