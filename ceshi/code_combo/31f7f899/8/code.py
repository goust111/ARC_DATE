# Combo variant of 31f7f899 (index 8)
# combo_spec: D4/2+D7/1
# Name: 紧凑裁剪输出+输出裁剪为条形区域

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, bbox_h, find_components, get_bg, paste

"""
Design analysis for the combo variant D4/2 + D7/1

a) Concrete behavior and trace
   - Original task:
       input is a square canvas with a full-width baseline and several colored vertical
       bars crossing that baseline. Solver sorts the bar shapes by height and redraws
       them into the original left-to-right slots on the same full canvas.
   - Combo task:
       input stays a square canvas with outer background margins, but the baseline is
       restricted to the slot span instead of the whole row. After sorting the bars into
       the original slots, the solver returns only the minimal bbox of the resulting
       non-background pixels.
   - Small trace:
       suppose the input has slot colors at x=[3,7,11] and unsorted heights [7,3,5].
       Original output: draw sorted heights [3,5,7] back at x=[3,7,11] on the original
       15x15 canvas; blank margins remain.
       Combo output: draw that same sorted arrangement first, but only over the original
       slot-span baseline, then crop away every blank outer margin. The visible result is
       a much smaller rectangle whose height equals the tallest sorted bar and whose width
       equals the slot span.
   - Difference:
       D4 changes the destination semantics from “keep the original canvas” to “return a
       compact rendering”, while D7 makes that compact rendering a different grid shape by
       forcing the output to the tight bar region. Together they require both sorting and
       output-shape inference.

b) Complexity calibration
   - Bars use distinct odd heights so shortest-to-tallest order is unambiguous.
   - Each sample uses 3..5 bars and a uniform width of 1 or 3, which keeps the sorting
     task visible without adding unrelated shape-matching ambiguity.
   - The input canvas stays square and includes positive margins on all four sides; the
     output is strictly smaller in both dimensions and is forced to be non-square in the
     generator, making the crop visually obvious.
   - Baseline color appears in multiple separated runs on the baseline row because bars
     overwrite it and there is at least one gap between neighboring bars. This gives the
     solver a stable cue for detecting the baseline after cropping-related changes.

c) Hidden assumptions in the original main()
   - `out = np.full_like(g, bg); out[base_r, :] = base_col` assumed the baseline spans
     the whole row. That breaks the combo because it would erase the crop signal. Fix:
     preserve only the visible baseline cells from the input row.
   - “Most frequent non-bg color on a row” was enough when the baseline covered the whole
     width. After narrowing the baseline, a tall bar can rival that count. Fix: detect
     the row/color that forms multiple separated runs, which matches the overwritten
     baseline row but not a bar.
   - Anchor recovery assumes every bar intersects the baseline row. The generator keeps
     odd heights and centers every rectangle on the baseline, so that still holds.
   - Reusing the original slot anchors is still valid because sorting changes only which
     shape goes to which slot, not the slot x-positions.
   - Tight cropping assumes at least one non-bg pixel exists. The generator always places
     bars and a baseline, so the output bbox is non-empty.

d) Placement legality
   - The generator enforces at least one column gap between neighboring bars, so distinct
     colors never merge into one component.
   - Heights are chosen so the tallest bar leaves at least a two-cell top and bottom
     margin inside the square input; widths and slot spans are chosen so there are also
     left and right outer margins. No paste call goes out of bounds.
   - The baseline spans exactly the slot region, ensuring all bars intersect the same row
     and the cropped output bbox is exactly the intended bar region.

e) Visual observability
   - From a few examples, a human can see that bar heights are being sorted left to right
     while slot colors stay tied to their original columns.
   - The second visible cue is equally strong: the output is no longer the same square
     canvas. Every outer blank border disappears, leaving only the tight rectangle around
     the baseline-and-bars structure.

Post-change solver assumptions audited against the generator
   - Base color appears only on the baseline row: guaranteed by construction.
   - Baseline row has at least two visible runs of the base color: guaranteed by k>=3 and
     at least one column gap between adjacent bars.
   - Each non-base component is one monochrome rectangle intersecting the baseline row:
     guaranteed by odd heights and centered placement.
   - Sorting by `(height, size, anchor, color)` is deterministic. Heights are distinct in
     generated samples, and the secondary keys keep the solver stable anyway.
   - The sorted full-canvas rendering always has positive margins on all four sides, so
     cropping changes the output shape on every generated example and the combo does not
     degenerate back to the original full-canvas task.
"""


def _count_runs(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(starts.sum())


def _find_base_row_color(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, _W = g.shape
    best_multi = None  # (runs, count, row, color)
    best_any = None  # (count, row, color)
    for r in range(H):
        row = g[r, :]
        for v in sorted(set(map(int, row)) - {int(bg)}):
            mask = row == int(v)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            cand_any = (cnt, int(r), int(v))
            if best_any is None or cand_any > best_any:
                best_any = cand_any
            runs = _count_runs(mask)
            if runs >= 2:
                cand_multi = (runs, cnt, int(r), int(v))
                if best_multi is None or cand_multi > best_multi:
                    best_multi = cand_multi
    if best_multi is not None:
        _runs, _cnt, base_r, base_col = best_multi
        return int(base_r), int(base_col)
    assert best_any is not None
    _cnt, base_r, base_col = best_any
    return int(base_r), int(base_col)


def _tight_crop(g: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the bars by height, redraw them in the original slots, then tight-crop.

    Input:
    - square canvas with background margins
    - one segmented baseline row of a single color spanning only the slot region
    - several monochrome vertical bars of distinct colors crossing that baseline

    Output:
    - the sorted arrangement, cropped to the minimal rectangle containing the bars and
      baseline; all outer background margins are removed
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    base_r, base_col = _find_base_row_color(g, bg)
    base_mask = np.zeros((H, W), dtype=bool)
    base_mask[int(base_r), :] = g[int(base_r), :] == int(base_col)

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        height = int(bbox_h(comp["bbox"]))
        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "height": int(height),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full((H, W), int(bg), dtype=int)
    out[base_mask] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return _tight_crop(out, bg)


def _rand_composition(total: int, parts: int) -> list[int]:
    probs = np.full(int(parts), 1.0 / int(parts))
    return [int(x) for x in np.random.multinomial(int(total), probs)]


def _slot_lefts(span_w: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(span_w - used)
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
    base_left: int,
    base_right: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((int(n), int(n)), int(bg), dtype=int)
    grid[int(base_r), int(base_left) : int(base_right) + 1] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _sorted_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _gen_attempt() -> np.ndarray | None:
    bg = 8
    base_color = 6
    palette = [1, 2, 3, 4, 5, 7, 9]

    n = int(np.random.choice(np.arange(13, 22, 2)))
    base_r = n // 2

    height_choices = list(range(1, min(11, n - 4) + 1, 2))
    max_count = min(5, len(height_choices))
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if int(k * 3 + (k - 1)) <= n - 4:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    used_w = int(k * width + (k - 1))
    max_internal_extra = int(n - used_w - 4)
    if max_internal_extra < 0:
        return None
    span_w = int(used_w + np.random.randint(0, max_internal_extra + 1))
    if span_w < used_w or span_w > n - 4:
        return None

    outer_total = int(n - span_w)
    if outer_total < 4:
        return None
    outer_left = int(np.random.randint(2, outer_total - 1))
    outer_right = int(outer_total - outer_left)
    if outer_left < 2 or outer_right < 2:
        return None

    local_lefts = _slot_lefts(span_w, k, width)
    lefts = [int(outer_left + x) for x in local_lefts]
    base_left = int(outer_left)
    base_right = int(outer_left + span_w - 1)

    colors = [int(c) for c in np.random.choice(palette, size=k, replace=False)]
    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    if span_w == sorted_heights[-1]:
        return None

    out_full = _draw_scene(
        n=n,
        base_r=base_r,
        base_left=base_left,
        base_right=base_right,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        colors=colors,
        heights=sorted_heights,
    )
    out = _tight_crop(out_full, bg)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_heights = [int(sorted_heights[i]) for i in perm]
    inp = _draw_scene(
        n=n,
        base_r=base_r,
        base_left=base_left,
        base_right=base_right,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        colors=colors,
        heights=in_heights,
    )

    nz = np.argwhere(out_full != int(bg))
    r0, c0, r1, c1 = bbox_coords(nz)
    margins = (int(r0), int(c0), int(n - 1 - r1), int(n - 1 - c1))

    if np.array_equal(inp, out_full):
        return None
    if out.shape[0] >= inp.shape[0] or out.shape[1] >= inp.shape[1]:
        return None
    if out.shape[0] == out.shape[1]:
        return None
    if min(margins) <= 0:
        return None
    if not np.array_equal(_sorted_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
