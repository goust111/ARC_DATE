# Combo variant of 31f7f899 (index 1)
# combo_spec: D1/2+D4/3
# Name: 降序排列+右起槽位

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_h, find_components, get_bg, paste


"""
Design analysis for combo D1/2 + D4/3
=====================================

a) Concrete trace on a small example
------------------------------------
Original task:
- Slots are ordered by x from left to right.
- Shapes are ordered by height from short to tall.
- So with heights [3, 5, 7, 9], output left->right becomes [3, 5, 7, 9].

Literal combo rule:
- Slots are ordered from right to left.
- Shapes are ordered by height from tall to short.
- With strictly distinct heights [3, 5, 7, 9], the pairing is:
  9->rightmost, 7->next, 5->next, 3->leftmost.
- Read back left->right, that again becomes [3, 5, 7, 9].

Key consequence:
- A naive generator with all heights distinct would degenerate: combo output would be
  identical to the original task even though the code changed.
- To make the fused behavior actually observable, the generator injects one repeated
  height with different widths, e.g. bars [(3,w=1), (5,w=1), (5,w=3), (7,w=1)].
- Original assignment yields left->right [(3,1), (5,1), (5,3), (7,1)].
- Combo assignment yields left->right [(3,1), (5,3), (5,1), (7,1)] because shapes are
  sorted by (-height, size, anchor, color) while slots are consumed right->left.
- The tallest bar is still at the far right, but the equal-height block flips internally,
  so the combo no longer collapses to the original mapping.

b) Complexity calibration
-------------------------
- Bars remain simple vertical rectangles, so the task still reads as a bar-sorting task.
- Each sample uses 3 or 4 bars. This is enough to show the ordering pattern clearly.
- Exactly one duplicated height is injected, and only for one pair. That is enough to
  break degeneration without turning the task into a complicated multi-attribute puzzle.
- The duplicate pair uses widths 1 and 3, making the swapped order visible by eye.

c) Implicit-assumption audit of the original `main()`
-----------------------------------------------------
- Base-row detection by maximal per-row non-bg count still works only if the uncovered
  baseline cells outnumber any bar-width run on a single row. The generator therefore
  enforces `grid_width - sum(widths) >= 4`, while widths are only 1 or 3.
- Component extraction assumes each bar is a separate monochrome component touching the
  base row. The generator keeps at least one blank column between 3-column slot footprints,
  so moved 3-wide bars still stay disjoint.
- Anchor recovery assumes each bar intersects the base row. The generator keeps odd
  heights and centers each rectangle on the base row so that assumption remains valid.
- Reassignment used to sort `height` ascending and slots ascending by x. The combo changes
  both: shapes use descending height, but slot order is descending x as well.
- To keep the combo distinct from the original task, ties on height must matter. The new
  solver therefore preserves the old secondary keys (`size`, `anchor`, `color`) instead of
  fully reversing the whole tuple. The generator guarantees one equal-height pair with
  different sizes, so this secondary ordering becomes active.

d) Placement legality
---------------------
- Every slot reserves a 3-column footprint with at least one blank column between slots.
- Actual bar widths are 1 or 3 and are centered inside that footprint, so any moved shape
  fits in any slot footprint without overlap.
- Heights are odd and chosen so `base_r - h//2 >= 1` and `base_r + h//2 <= n-2`, which
  keeps all bars in bounds with visible margins above and below.

e) Visual observability
-----------------------
- Across examples, a solver can still see the coarse fused rule: the tallest bar ends on
  the right and bars generally get shorter toward the left.
- The non-degenerate clue is that an equal-height pair swaps internal order compared with
  the original task; the wider twin ends up to the left of the thinner twin in output.
- Because only one pair ties, the main visual story remains simple enough for 2-4 examples.

Audit of new solver assumptions against the generator
-----------------------------------------------------
- There is exactly one full-width baseline color and it appears on the chosen base row.
- Every non-baseline component is a centered odd-height rectangle.
- At least one equal-height pair has different widths, so combo output differs from the
  original ascending+left-to-right mapping.
- All slots are wide enough for the widest moved bar because slot footprints are fixed to
  width 3 even when an input bar is width 1.
"""


def _find_base_row_color(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    H, _W = g.shape
    for r in range(H):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == int(bg):
                continue
            cand = (cnt, int(r), int(v))
            if best is None or cand > best:
                best = cand
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _extract_objects(g: np.ndarray, bg: int, base_r: int, base_col: int) -> list[dict[str, object]]:
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
    return objects


def _render(objects: list[dict[str, object]], H: int, W: int, bg: int, base_r: int, base_col: int, *, combo: bool) -> np.ndarray:
    if combo:
        slots = sorted(objects, key=lambda o: (-int(o["anchor"]), int(o["color"])))
        shapes = sorted(
            objects,
            key=lambda o: (-int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])),
        )
    else:
        slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
        shapes = sorted(
            objects,
            key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])),
        )

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Rank bar shapes by height in descending order.
    - Rank slots by x-position from right to left.
    - Place the i-th tallest shape into the i-th rightmost slot, recoloring it to that
      slot's original color.
    - Preserve the baseline row.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base_row_color(g, bg)
    objects = _extract_objects(g, bg, base_r, base_col)
    return _render(objects, H, W, bg, base_r, base_col, combo=True)


def _rand_composition(total: int, parts: int) -> list[int]:
    probs = np.full(int(parts), 1.0 / int(parts))
    return [int(x) for x in np.random.multinomial(int(total), probs)]


def _copy_bar(bar: dict[str, int]) -> dict[str, int]:
    return {"height": int(bar["height"]), "width": int(bar["width"])}


def _slot_lefts(n: int, k: int, slot_w: int) -> list[int]:
    used = int(k * slot_w + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("slots do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(slot_w)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _make_bar_pool(k: int) -> list[dict[str, int]]:
    # One unique tallest bar plus one equal-height pair with widths 1 and 3.
    height_choices = [1, 3, 5, 7, 9]
    tallest = int(np.random.choice([5, 7, 9]))
    lower = [h for h in height_choices if h < tallest]
    dup_h = int(np.random.choice(lower))
    remaining_needed = int(k - 3)
    others = []
    if remaining_needed > 0:
        pool = [h for h in lower if h != dup_h]
        if len(pool) < remaining_needed:
            raise ValueError("not enough distinct lower heights")
        others = [int(h) for h in np.random.choice(pool, size=remaining_needed, replace=False)]

    bars = [{"height": int(dup_h), "width": 1}, {"height": int(dup_h), "width": 3}]
    for h in others:
        bars.append({"height": int(h), "width": int(np.random.choice([1, 3]))})
    bars.append({"height": int(tallest), "width": int(np.random.choice([1, 3]))})

    perm = np.random.permutation(len(bars))
    return [_copy_bar(bars[int(i)]) for i in perm]


def _bar_shape_key(bar: dict[str, int], *, descending: bool) -> tuple[int, int]:
    height = int(bar["height"])
    size = int(bar["height"]) * int(bar["width"])
    if descending:
        return (-height, size)
    return (height, size)


def _assign_bars_to_slots(bars: list[dict[str, int]], *, combo: bool) -> list[dict[str, int]]:
    k = len(bars)
    slot_indices = list(range(k - 1, -1, -1)) if combo else list(range(k))
    shape_order = sorted(range(k), key=lambda i: _bar_shape_key(bars[int(i)], descending=combo))
    assigned: list[dict[str, int] | None] = [None] * k
    for slot_idx, bar_idx in zip(slot_indices, shape_order):
        assigned[int(slot_idx)] = _copy_bar(bars[int(bar_idx)])
    assert all(bar is not None for bar in assigned)
    return [bar for bar in assigned if bar is not None]


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    colors: list[int],
    bars_by_slot: list[dict[str, int]],
) -> np.ndarray:
    grid = np.full((int(n), int(n)), int(bg), dtype=int)
    grid[int(base_r), :] = int(base_color)
    for left, color, bar in zip(lefts, colors, bars_by_slot):
        height = int(bar["height"])
        width = int(bar["width"])
        anchor = int(left) + 1
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(anchor - width // 2))
    return grid


def _original_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base_row_color(g, bg)
    objects = _extract_objects(g, bg, base_r, base_col)
    return _render(objects, H, W, bg, base_r, base_col, combo=False)


def _gen_attempt() -> np.ndarray | None:
    bg = 8
    base_color = 6
    slot_w = 3
    palette = [1, 2, 3, 4, 5, 7, 9]

    k = int(np.random.choice([3, 4]))
    bars = _make_bar_pool(k)
    max_height = max(int(bar["height"]) for bar in bars)
    width_sum = sum(int(bar["width"]) for bar in bars)

    feasible_ns = [
        n
        for n in [11, 13, 15, 17]
        if n >= int(max_height + 2) and n >= int(k * slot_w + (k - 1)) and n - int(width_sum) >= 4
    ]
    if not feasible_ns:
        return None
    n = int(np.random.choice(feasible_ns))
    base_r = int(n // 2)

    lefts = _slot_lefts(n, k, slot_w)
    colors = [int(c) for c in np.random.choice(palette, size=k, replace=False)]

    out_bars = _assign_bars_to_slots(bars, combo=True)
    orig_bars = _assign_bars_to_slots(bars, combo=False)

    perm = np.random.permutation(k)
    inp_bars = [_copy_bar(bars[int(i)]) for i in perm]
    if inp_bars == out_bars:
        a, b = np.random.choice(k, size=2, replace=False)
        inp_bars[int(a)], inp_bars[int(b)] = inp_bars[int(b)], inp_bars[int(a)]

    inp = _draw_scene(n, base_r, base_color, bg, lefts, colors, inp_bars)
    out = _draw_scene(n, base_r, base_color, bg, lefts, colors, out_bars)
    orig_out = _draw_scene(n, base_r, base_color, bg, lefts, colors, orig_bars)

    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, orig_out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_original_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
