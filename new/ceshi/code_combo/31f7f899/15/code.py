# Combo variant of 31f7f899 (index 15)
# combo_spec: D3/4+D6/0
# Name: 对角线条形+垂直基线

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    Original task: a full-width horizontal baseline sits on row H//2. Each bar is an axis-aligned
#    vertical rectangle centered on that row. The solver finds the baseline row, extracts each bar
#    relative to that row, sorts bars by height, then pastes the i-th shortest shape into the i-th
#    leftmost color slot.
#    Combo task: the baseline rotates to a full-height vertical line on column W//2, and the bars
#    stop being rectangles. Each object is now a one-pixel-thick 45-degree diagonal segment of odd
#    length, centered on exactly one pixel of the vertical baseline. Slot identity therefore comes
#    from the segment's intersection row with the baseline, not from x-position. The solver must:
#    1) detect the dominant column as the baseline, 2) use 8-connectivity so each diagonal segment
#    stays whole, 3) use the baseline-touching row as the slot anchor, 4) rank shapes by diagonal
#    length (pixel count), and 5) repaint the sorted shapes into the top-to-bottom row slots while
#    preserving slot colors.
#    Small trace: suppose base column is 6 and there are three slots anchored at rows 3, 7, 11 with
#    colors red/green/yellow. Their input segment lengths are 7/3/5. Original logic would scan rows,
#    miss the true baseline orientation, and treat diagonal pieces as malformed vertical bars.
#    Combo logic scans columns, gets slot rows [3,7,11], sorts lengths [3,5,7], and outputs a red
#    length-3 slash at row 3, a green length-5 slash at row 7, and a yellow length-7 slash at row 11.
#    The key difference is that both the structural axis and the object geometry changed together.
# b) Complexity calibration
#    We keep the core task family small and legible: 3-5 objects, odd lengths from 3 upward, one-
#    pixel-thick diagonals, distinct colors, and a single shared baseline color. Odd lengths guarantee
#    exactly one baseline intersection per object, which avoids ambiguous slot anchors, and excluding
#    length-1 prevents visually degenerate "diagonal" objects that would look like isolated pixels.
#    Distinct lengths prevent ties in the sorting key. Restricting objects to straight diagonal
#    segments rather than arbitrary diagonal sprites keeps the relation human-readable in 2-4 examples
#    while still making the scene visibly unlike the original task.
# c) Hidden assumptions in the original main()
#    - Baseline detection over rows is invalid after D6/0; we must search columns instead.
#    - 4-connectivity is invalid after D3/4 because pure diagonal lines disconnect under 4-neighbors.
#    - bbox_h() as the ranking feature is no longer meaningful for a diagonal segment; the relevant
#      quantity is its diagonal pixel count. For these generated segments, area == length.
#    - The original anchor was a column touching the base row. After fusion, the anchor is a row
#      touching the base column.
#    - Relative coordinates were stored around (base_row, anchor_col); now they must be stored around
#      (anchor_row, base_col).
#    - Output reconstruction can no longer draw a horizontal line or sort by x-position; it must draw
#      a vertical line and sort slots top-to-bottom.
#    All of these assumptions were harmless in the seed task because rectangles were axis-aligned and
#    centered on a horizontal baseline. Under the combo they would either split objects, detect the
#    wrong baseline, or paste shapes into the wrong slots.
# d) Placement legality
#    The generator is output-first. It chooses a single maximum half-length margin and places all slot
#    anchors inside the row interval [margin, n-1-margin], with at least one untouched row between
#    neighboring anchors. That margin guarantees every sampled odd length fits vertically, while the
#    centered baseline column guarantees every sampled odd length fits horizontally. Parallel diagonal
#    segments with different anchors never overlap, and because each segment uses a distinct color and
#    overwrites exactly one baseline cell, baseline detection remains stable.
# e) Visual observability
#    A human looking at the examples sees a vertical pink spine and several colored slashes crossing
#    it at different rows. The colors stay attached to those rows from input to output, while the slash
#    lengths move into sorted order from top to bottom. This is visually distinct from both the seed
#    task and the pure D6 rotation: there are no axis-aligned bars anywhere, and the sortable feature
#    is the diagonal extent itself.
#
# Post-change assumption audit for main()
# - The baseline column must contain the dominant count of one non-bg color. The generator guarantees
#   this because the baseline color fills the whole column except for one overwritten pixel per object.
# - Each non-baseline object must touch the baseline column exactly once. The generator uses odd-length
#   centered diagonal segments, so every object has one unique intersection cell on the baseline.
# - Each object must be an 8-connected monochrome component. The generator uses straight 45-degree
#   pixel chains, which are connected under 8-neighbors.
# - Length ties would make the sort underdetermined for humans. The generator samples distinct odd
#   lengths, so the ranking is unique.
# - Repainting assumes every slot can host every shape. The shared margin construction guarantees that
#   even the longest shape fits around every anchor row selected for the scene.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort diagonal segments by length and redistribute them across fixed row slots.

    Input structure:
    - bg background.
    - A vertical base line of one color spans the full grid height.
    - Several distinct-color diagonal segments cross that base line at different rows.

    Rule:
    - Rank diagonal segments by pixel length (shortest to longest).
    - Rank the original baseline-touch rows top to bottom.
    - Place the i-th shortest segment into the i-th topmost slot, recoloring it to that slot's
      original color.
    - Preserve the vertical baseline, with the segment colors overwriting it at the intersection rows.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    best = None  # (count, col, color)
    for c in range(W):
        vals, counts = np.unique(g[:, c], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(c), int(v))
    assert best is not None
    _cnt, base_c, base_col = best

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue

        pts = comp["coords"]
        base_rows = sorted(int(r) for r, c in pts if int(c) == int(base_c))
        if base_rows:
            anchor = int(base_rows[len(base_rows) // 2])
        else:
            rows_sorted = sorted(int(r) for r, _c in pts)
            anchor = int(rows_sorted[len(rows_sorted) // 2])

        rel = [(int(r) - int(anchor), int(c) - int(base_c)) for r, c in pts]
        area = int(len(rel))
        height = int(bbox_h(comp["bbox"]))
        width = int(bbox_w(comp["bbox"]))
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "length": int(area),
                "size": int(area),
                "height": int(height),
                "width": int(width),
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (int(o["length"]), int(o["height"]), int(o["width"]), int(o["anchor"]), int(o["color"])),
    )

    out = np.full_like(g, bg)
    out[:, int(base_c)] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(slot_anchor) + int(dr)
            cc = int(base_c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_rows(n: int, k: int, margin: int) -> list[int]:
    span = int(n - 2 * margin)
    used = int(k + (k - 1))
    extra = int(span - used)
    if extra < 0:
        raise ValueError("diagonal slots do not fit")
    gaps = _rand_composition(extra, k + 1)
    rows: list[int] = []
    r = int(margin + gaps[0])
    for i in range(k):
        rows.append(int(r))
        r += 1
        if i < k - 1:
            r += 1 + int(gaps[i + 1])
    return rows


def _diag_coords(anchor_r: int, base_c: int, length: int) -> list[tuple[int, int]]:
    half = int(length // 2)
    return [(int(anchor_r + d), int(base_c + d)) for d in range(-half, half + 1)]


def _draw_scene(
    n: int,
    base_c: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    lengths: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[:, int(base_c)] = int(base_color)
    for anchor, color, length in zip(anchors, colors, lengths):
        for r, c in _diag_coords(int(anchor), int(base_c), int(length)):
            grid[int(r), int(c)] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = 8
    base_color = 6

    k = int(np.random.randint(3, 6))
    feasible_ns = [int(n) for n in range(11, 20, 2) if int(n) >= 4 * k - 1]
    if not feasible_ns:
        return None
    n = int(np.random.choice(feasible_ns))
    base_c = n // 2

    max_half_lo = int(k)
    max_half_hi = int(min(base_c, (n - (2 * k - 1)) // 2))
    if max_half_hi < max_half_lo:
        return None
    max_half = int(np.random.randint(max_half_lo, max_half_hi + 1))

    anchors = _slot_rows(n, k, max_half)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    length_choices = list(range(3, 2 * max_half + 2, 2))
    sorted_lengths = sorted(int(x) for x in np.random.choice(length_choices, size=k, replace=False))
    out = _draw_scene(n, base_c, base_color, bg, anchors, colors, sorted_lengths)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    input_lengths = [int(sorted_lengths[i]) for i in perm]
    inp = _draw_scene(n, base_c, base_color, bg, anchors, colors, input_lengths)

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
