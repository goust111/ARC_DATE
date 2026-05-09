# Combo variant of 31f7f899 (index 12)
# combo_spec: D8/1+D10/0
# Name: 冒泡单步+轨迹标记

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    Original task:
#    - Parse the baseline row and the set of colored vertical bars.
#    - Globally sort all bar shapes by height.
#    - Reassign those sorted shapes to the left-to-right slots, while each slot
#      keeps its original color identity.
#    Combo behavior:
#    - Keep the same visual scaffold: background, one full-width baseline, and
#      several monochrome bars centered on that baseline.
#    - Do not globally sort. Instead read the current left-to-right height
#      sequence, find the first adjacent inversion h[i] > h[i+1], and swap only
#      that neighboring pair, i.e. one bubble-sort step.
#    - After the swap, mark the two source slots with a special trail color. The
#      marker is not drawn inside the bar body because the arriving bar would
#      overwrite it; instead each slot has a reserved adjacent empty column where
#      the output shows a vertical trail stripe matching the moved-away bar's
#      original height.
#    Example: with slot colors [blue, red, yellow, maroon] and input heights
#    [3, 9, 5, 7], the original task would output heights [3, 5, 7, 9]. The combo
#    outputs [3, 5, 9, 7] and additionally places trail stripes beside slots 2 and
#    3, because those are the two bars that swapped in this single bubble step.
#    The visible differences are therefore both algorithmic (partial swap instead
#    of full sort) and representational (source-slot trails appear only in output).
#
# b) Complexity calibration
#    - Heights are distinct odd numbers >= 3, so the first inversion is unique and
#      every bar has a visible nontrivial trail stripe.
#    - The generator samples genuine intermediate bubble states and rejects cases
#      where the next step would already fully sort the sequence; this keeps the
#      D10 behavior visibly different from the original one-shot sort.
#    - Trail color is excluded from slot colors, so the extra markers cannot be
#      mistaken for another bar or another slot identity.
#    - Every slot gets a reserved adjacent gap column; without that reservation the
#      trail would either be clipped at the border or collide with a neighbor.
#
# c) Audit of old hidden assumptions
#    - Old assumption: every non-background/non-baseline component is a bar and can
#      be globally sorted by height. The first part still holds for inputs because
#      trails exist only in outputs, but the global-sort assumption is now false.
#    - Old assumption: the desired output is determined only by the multiset of bar
#      heights. The combo breaks this: the current left-to-right order matters
#      because only the first inversion swaps.
#    - Old assumption: output contains only baseline + recolored bars. The combo
#      adds a new rendering layer for source-slot trails, so output synthesis must
#      preserve both the moved bars and the extra marker channel.
#    - Potential conflict from combining D8 and D10: if the trail were drawn on the
#      vacated bar cells themselves, the incoming swapped bar would erase it. The
#      reserved side column resolves this interaction cleanly.
#
# d) Placement legality
#    - Bars use one global width per sample and remain separated by at least one
#      empty column.
#    - Slot generation reserves one dedicated empty column immediately to the left
#      of every slot, used only for possible output trails.
#    - Heights are odd and bounded by the grid size, so both bars and trail stripes
#      stay in bounds when centered on the baseline row.
#
# e) Human observability
#    - Across a few training pairs, a human can see that outputs are not globally
#      sorted: only one neighboring pair changes.
#    - The two trail stripes isolate exactly which source slots participated in the
#      swap, so the extra output density is not decorative; it corroborates the
#      single-step bubble interpretation.
#
# f) New assumptions introduced by this solver, and generator guarantees
#    - There is exactly one full-width baseline row with one non-background color.
#    - Input bars are monochrome 4-connected components distinct from the baseline.
#    - Slot order is the left-to-right order of bar components.
#    - Each slot has at least one adjacent free column available for the trail; the
#      generator guarantees a reserved left-side gap for every slot.
#    - Output trail color does not appear as an input bar color; generator enforces
#      colors from {1,2,3,4,7,9} while bg/base/trail use 8/6/5.


def _find_base(grid: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    H, _W = grid.shape
    for r in range(H):
        vals, counts = np.unique(grid[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == int(bg):
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = [(int(r), int(c)) for r, c in comp["coords"]]
        _r0, c0, _r1, c1 = comp["bbox"]
        rel = [(int(r) - int(base_r), int(c) - int(c0)) for r, c in pts]
        row_offsets = sorted({int(r) - int(base_r) for r, _c in pts})
        objects.append(
            {
                "color": int(col),
                "left": int(c0),
                "right": int(c1),
                "height": int(bbox_h(comp["bbox"])),
                "size": int(len(pts)),
                "rel": rel,
                "row_offsets": row_offsets,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["left"]), int(o["color"])))
    return int(bg), int(base_r), int(base_col), slots


def _first_inversion(heights: list[int]) -> int | None:
    for i in range(len(heights) - 1):
        if int(heights[i]) > int(heights[i + 1]):
            return int(i)
    return None


def _trail_column(slots: list[dict[str, object]], idx: int, width: int) -> int | None:
    slot = slots[int(idx)]
    left = int(slot["left"])
    right = int(slot["right"])
    prev_right = int(slots[idx - 1]["right"]) if idx > 0 else -10
    next_left = int(slots[idx + 1]["left"]) if idx + 1 < len(slots) else int(width) + 10

    for cand in (left - 1, right + 1):
        if 0 <= int(cand) < int(width) and int(prev_right) < int(cand) < int(next_left):
            return int(cand)
    if 0 <= int(left - 1) < int(width):
        return int(left - 1)
    if 0 <= int(right + 1) < int(width):
        return int(right + 1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Perform one bubble-sort step on the slot-height sequence and mark sources.

    The slot x-positions and slot colors remain fixed. We inspect the current
    left-to-right bar heights, swap only the first adjacent inversion, and draw a
    trail stripe next to each of the two source slots involved in that swap.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, base_r, base_col, slots = _extract_scene(g)

    heights = [int(obj["height"]) for obj in slots]
    swap_i = _first_inversion(heights)

    source_idx = list(range(len(slots)))
    trail_sources: list[int] = []
    if swap_i is not None:
        source_idx[swap_i], source_idx[swap_i + 1] = source_idx[swap_i + 1], source_idx[swap_i]
        trail_sources = [int(swap_i), int(swap_i + 1)]

    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    for slot_i, slot in enumerate(slots):
        shape = slots[source_idx[slot_i]]
        slot_left = int(slot["left"])
        slot_col = int(slot["color"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_left) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    trail_color = 5
    for src_idx in trail_sources:
        trail_c = _trail_column(slots, int(src_idx), W)
        if trail_c is None:
            continue
        for dr in slots[int(src_idx)]["row_offsets"]:
            rr = int(base_r) + int(dr)
            if 0 <= rr < H and 0 <= int(trail_c) < W:
                out[rr, int(trail_c)] = int(trail_color)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * (width + 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars plus reserved trail gaps do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        c += 1  # reserved empty column immediately left of the slot
        lefts.append(int(c))
        c += int(width)
        c += int(gaps[i + 1])
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
    grid[base_r, :] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _draw_expected_output(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    trail_color: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    in_heights: list[int],
) -> np.ndarray | None:
    swap_i = _first_inversion(in_heights)
    if swap_i is None:
        return None

    out_heights = [int(h) for h in in_heights]
    out_heights[swap_i], out_heights[swap_i + 1] = out_heights[swap_i + 1], out_heights[swap_i]

    grid = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, out_heights)
    for idx in (int(swap_i), int(swap_i + 1)):
        trail_c = int(lefts[idx]) - 1
        height = int(in_heights[idx])
        for rr in range(int(base_r - height // 2), int(base_r + height // 2) + 1):
            if 0 <= rr < n and 0 <= trail_c < n:
                grid[int(rr), int(trail_c)] = int(trail_color)
    return grid


def _bubble_states(heights: list[int]) -> tuple[list[list[int]], list[int]]:
    states = [[int(h) for h in heights]]
    swap_positions: list[int] = []
    current = [int(h) for h in heights]
    while True:
        idx = _first_inversion(current)
        if idx is None:
            break
        nxt = current.copy()
        nxt[idx], nxt[idx + 1] = nxt[idx + 1], nxt[idx]
        swap_positions.append(int(idx))
        states.append(nxt)
        current = nxt
    return states, swap_positions


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6
    trail_color = 5

    height_choices = list(range(3, min(n, 11) + 1, 2))
    configs: list[tuple[int, int]] = []
    max_k = min(5, len(height_choices))
    for k in range(3, max_k + 1):
        for width in (1, 3):
            if int(k * (width + 1)) <= n:
                configs.append((int(k), int(width)))
    if not configs:
        return None

    k, width = configs[int(np.random.randint(len(configs)))]
    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 7, 9], size=k, replace=False)]
    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))

    for _ in range(80):
        perm = np.random.permutation(k)
        in0 = [int(sorted_heights[i]) for i in perm]
        states, _ = _bubble_states(in0)
        valid_steps = [int(i) for i in range(len(states) - 1) if _first_inversion(states[i]) is not None and _first_inversion(states[i + 1]) is not None]
        if not valid_steps:
            continue

        step = int(valid_steps[np.random.randint(len(valid_steps))])
        mid_heights = [int(h) for h in states[step]]
        inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, mid_heights)
        out = _draw_expected_output(n, base_r, base_color, bg, trail_color, lefts, width, colors, mid_heights)
        if out is None:
            continue

        # Reject samples where the D10 behavior collapses to the original global sort.
        if [int(h) for h in sorted(mid_heights)] == [int(h) for h in states[step + 1]]:
            continue
        if np.array_equal(inp, out):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
