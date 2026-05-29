# Variant of 31f7f899
# Dimension: D10 — 多步骤序列化处理 (meta)
# Change: 单步全局排序——一次性将所有条形按高度排序分配到所有槽位... → 逐步冒泡排序

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, find_components, get_bg, paste

# Design analysis for the D10 variant
# a) Concrete behavior:
#    The scene stays visually identical to the original task: a full-width base row,
#    and several vertical monochrome bars centered on that row. What changes is the
#    redistribution rule. The original solver extracted every bar shape, globally
#    sorted those shapes by height, then placed them into left-to-right slots while
#    preserving each slot's color. In this variant, the input is a genuine
#    intermediate state of a bubble-sort-like process over the left-to-right height
#    sequence. We scan slots from left to right, locate the first adjacent inversion
#    h[i] > h[i+1], and swap only that neighboring pair of shapes. Example:
#    heights [1, 7, 3, 5] used to become [1, 3, 5, 7] in one shot; now it becomes
#    [1, 3, 7, 5]. The visible difference is that only two adjacent bars exchange
#    silhouettes, while all other bars stay in place for this sample.
# b) Complexity calibration:
#    Heights remain distinct odd numbers, so the swapped pair is visually
#    distinguishable and the left-to-right sequence has a unique "first inversion".
#    Generator samples only states that still contain another inversion after the
#    swap, so the task does not collapse back into the old one-step global sort.
# c) Hidden-assumption audit of the original main():
#    - Base-row detection by most frequent non-background color on one row still
#      works because the baseline structure is unchanged.
#    - Component extraction excluding {bg, base_col} still works because each bar is
#      still a monochrome component disconnected by horizontal gaps.
#    - Sorting all shapes globally by height is the key old assumption that no longer
#      holds; we must preserve the current left-to-right order except for one
#      adjacent swap.
#    - Recoloring by slot color remains valid because slots still keep their color
#      identities; only silhouettes move between neighboring slots.
# d) Placement legality:
#    Bars are placed in non-overlapping slots with at least one empty column between
#    them. Heights are odd and chosen to fit around the fixed middle base row, so no
#    pasted patch leaves the grid. Swapping adjacent shapes is safe because both
#    shapes already fit in every slot of the same fixed width.
# e) Human observability:
#    From 2-4 examples, a solver can see that outputs are not fully sorted: only the
#    leftmost out-of-order neighboring pair changes. Generator states are authentic
#    intermediate states of repeated "swap first inversion" dynamics, which makes the
#    sequential nature visible without introducing a hidden pointer marker.
# f) New assumptions introduced by this main():
#    - There is at most one bar per slot anchor; generator enforces distinct slots.
#    - Bar heights are comparable by bbox height; generator keeps simple rectangles.
#    - If no inversion exists, output is unchanged; generator avoids this degenerate
#      case for training/test production by requiring at least one pending swap.


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]]]:
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
                "color": int(col),
                "anchor": int(anchor),
                "height": int(height),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    return bg, int(base_r), int(base_col), slots


def _first_inversion(heights: list[int]) -> int | None:
    for i in range(len(heights) - 1):
        if int(heights[i]) > int(heights[i + 1]):
            return int(i)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Perform one bubble-sort-style adjacent swap on the bar-height sequence.

    The left-to-right slot anchors and slot colors stay fixed. We inspect the
    current bar heights in slot order, find the first adjacent inversion, and swap
    only that pair of bar silhouettes. Everything else is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, base_r, base_col, slots = _extract_scene(g)

    heights = [int(obj["height"]) for obj in slots]
    swap_i = _first_inversion(heights)

    source_idx = list(range(len(slots)))
    if swap_i is not None:
        source_idx[swap_i], source_idx[swap_i + 1] = source_idx[swap_i + 1], source_idx[swap_i]

    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    for slot_i, slot in enumerate(slots):
        shape = slots[source_idx[slot_i]]
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
    grid[base_r, :] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
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

    height_choices = list(range(1, min(n, 11) + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]
    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))

    for _ in range(50):
        perm = np.random.permutation(k)
        in_heights = [int(sorted_heights[i]) for i in perm]
        states, _swap_positions = _bubble_states(in_heights)
        if len(states) < 3:
            continue

        step = int(np.random.randint(0, len(states) - 2))
        mid_heights = [int(h) for h in states[step]]
        out_heights = [int(h) for h in states[step + 1]]

        if _first_inversion(out_heights) is None:
            continue

        inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, mid_heights)
        out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, out_heights)

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
