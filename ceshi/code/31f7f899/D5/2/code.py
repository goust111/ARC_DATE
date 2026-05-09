# Variant of 31f7f899
# Dimension: D5 — 变换：重新着色 (transform)
# Change: recolor — 条形形状放入新槽位后，颜色变为该槽位的原始颜色... → 条形形状旋转后放入

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, find_components, get_bg, paste


# Design analysis for this D5 variant
# a) Concrete behavior:
#    Input stays close to the seed task: a full-width base row plus several monochrome
#    vertical bars centered on that row. Output still sorts bar shapes by height and keeps
#    slot order/color fixed, but each selected bar shape is first rotated 90 degrees before
#    being pasted into its destination slot. A 5x1 vertical bar therefore becomes a 1x5
#    horizontal bar centered on the slot anchor; a 7x3 vertical rectangle becomes a 3x7
#    horizontal rectangle.
#    Trace example: anchors [3, 9, 15], slot colors [red, green, yellow], input heights
#    [7, 3, 5]. Seed solver would sort to [3, 5, 7] and paste vertical bars at anchors
#    [3, 9, 15]. Variant solver sorts to the same order, then rotates each bar so lengths
#    along the base row become [3, 5, 7], producing horizontal bars in those three slots.
#    The affected stage is exactly the final placement loop: relative coordinates are no
#    longer reused as-is, they are transformed by (dr, dc) -> (dc, -dr) before placement.
# b) Complexity calibration:
#    To make the rotation visually obvious, generator bars use odd heights strictly larger
#    than their thickness. This avoids square cases such as 3x3 where rotation would be
#    invisible. Thickness is still low-cardinality (1 or 3), so examples remain readable.
#    Output anchors are spaced according to rotated lengths, preventing overlap between the
#    new horizontal bars while keeping the original left-to-right slot identity intact.
# c) Hidden-assumption audit of the seed solver:
#    - Base-row detection assumes one non-bg color dominates a row. Generator keeps the
#      full-width baseline, so that remains true.
#    - Component extraction assumes every non-baseline object is a single monochrome bar.
#      Generator still produces isolated monochrome rectangles, so this remains true.
#    - Anchor detection assumes a bar intersects the base row and its middle base-row cell
#      is the slot center. Generator preserves centered odd-height bars so this stays valid.
#    - The seed placement loop assumes reusing raw rel coords is harmless. That is the one
#      invalidated assumption here: in the variant the stored coords must be rotated before
#      placement or the output collapses back to the original task.
# d) Placement legality:
#    Generator is output-first. It first chooses sorted heights and computes non-overlapping
#    horizontal output intervals, then reuses the resulting anchors for the input. Because
#    vertical input bars are thinner than their rotated output spans, the same anchors also
#    guarantee input bars stay separated and in bounds. Each candidate is oracle-checked via
#    main(input) == output.
# e) Visual observability:
#    Across 2-4 examples a human should see two linked effects: bars are reordered by height
#    exactly as before, but the destination objects are horizontal rather than vertical and
#    inherit the slot color. Generator additionally rejects cases where the old no-rotation
#    solver would match the same target, so the new D5 behavior is genuinely exercised.


def _solve(grid: np.ndarray, *, rotate: bool) -> np.ndarray:
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

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue

        pts = comp["coords"]
        b = comp["bbox"]
        height = int(bbox_h(b))

        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": col,
                "anchor": anchor,
                "height": height,
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        rel_coords = shape["rel"]
        for dr, dc in rel_coords:
            rr_off, cc_off = (int(dc), int(-dr)) if rotate else (int(dr), int(dc))
            rr = int(base_r) + int(rr_off)
            cc = int(slot_anchor) + int(cc_off)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height, then place the sorted shapes into the fixed slots
    after rotating each bar 90 degrees and recoloring it to the destination slot color.
    """
    return _solve(grid, rotate=True)


def _solve_without_rotation(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, rotate=False)


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts_for_lengths(n: int, lengths: list[int]) -> list[int]:
    used = int(sum(int(x) for x in lengths) + max(0, len(lengths) - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("rotated bars do not fit")
    gaps = _rand_composition(extra, len(lengths) + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i, length in enumerate(lengths):
        lefts.append(int(c))
        c += int(length)
        if i < len(lengths) - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_input_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for anchor, color, height in zip(anchors, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(anchor - width // 2))
    return grid


def _draw_output_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    width: int,
    colors: list[int],
    lengths: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for anchor, color, length in zip(anchors, colors, lengths):
        patch = np.full((int(width), int(length)), int(color), dtype=int)
        paste(grid, patch, int(base_r - width // 2), int(anchor - length // 2))
    return grid


def _non_identity_perm(k: int) -> np.ndarray:
    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    return perm


def _gen_attempt() -> np.ndarray | None:
    bg = 8
    base_color = 6

    width = int(np.random.choice([1, 3]))
    height_choices = list(range(width + 2, 12, 2))
    if len(height_choices) < 3:
        return None

    k = int(np.random.randint(3, min(4, len(height_choices)) + 1))
    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))

    min_n = max(int(sorted_heights[-1]), int(sum(sorted_heights) + (k - 1) + 2))
    candidate_ns = [n for n in range(max(9, min_n), 30) if n % 2 == 1]
    if not candidate_ns:
        return None
    n = int(np.random.choice(candidate_ns))
    base_r = n // 2

    lefts = _slot_lefts_for_lengths(n, sorted_heights)
    anchors = [int(left + length // 2) for left, length in zip(lefts, sorted_heights)]
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    out = _draw_output_scene(n, base_r, base_color, bg, anchors, width, colors, sorted_heights)

    perm = _non_identity_perm(k)
    in_heights = [int(sorted_heights[i]) for i in perm]
    inp = _draw_input_scene(n, base_r, base_color, bg, anchors, width, colors, in_heights)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_solve_without_rotation(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
