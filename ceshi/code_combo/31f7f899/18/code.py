# Combo variant of 31f7f899 (index 18)
# combo_spec: D7/2+D12/2
# Name: 输入输出分离区域+N:M不等基数

from __future__ import annotations

"""
Design analysis for the combo variant "输入输出分离区域+N:M不等基数".

a) Concrete behavior and trace
   - Pixel layout keeps the original "bars crossing a shared baseline" motif, but the board is
     split by one full-height separator column.
   - Left half = source region. It contains four unsorted monochrome vertical bars touching the
     baseline row.
   - Right half = destination region. Input contains only top-row slot markers there; output
     keeps the markers and paints bars below them.
   - The D12 fusion twist is that there are fewer slot markers than source bars.
   - Example trace:
       left-source bar heights by x-order = [7, 1, 5, 3]
       right-half slot markers            = [red, yellow]
       original task would use all four bars and four slots in one region.
       D7-only split variant would still require 4 source bars and 4 right-half markers.
       combo behavior sorts heights to [1, 3, 5, 7], keeps only the first M=2 shapes, discards
       the taller two, and paints heights [1, 3] into the two right-half marker positions with
       colors [red, yellow].
     The solver must jointly infer spatial separation and selective retention of only the
     shortest M bars.

b) Complexity calibration
   - The separator is always present and centered, so the two-region structure is obvious.
   - Source count is fixed to 4 while slot count is only 2 or 3. This keeps the N>M signal
     consistent across samples without exploding layout freedom.
   - Bar width is fixed to 1. The original task only cares about height silhouettes, and the
     split layout already adds one extra reasoning step; keeping width constant avoids accidental
     overlap and keeps the visual cue focused on height + selection.
   - Heights are distinct odd values, and source order is shuffled away from ascending order.
     This prevents the samples from visually collapsing into a trivial already-sorted layout.

c) Audit of implicit assumptions changed by the combo
   - Original `main()` assumed one undivided region where the bars themselves define the slots.
     That breaks under D7/2. The new solver must separate source bars (left half) from slot
     markers (top row of the right half).
   - Original `main()` also assumed 1:1 cardinality. That breaks under D12/2. The new solver
     must tolerate `len(source_bars) > len(slots)` and define which shapes survive.
   - The fused rule chooses the `M` shortest source bars, where `M` is the number of slots. This
     reuses the source task's "sort by height ascending" principle and extends it with a stable
     truncation rule.
   - Copying the whole input before painting still matters: rebuilding from scratch would erase
     the left source region, the separator, and the right-half markers that remain visible in the
     output.
   - Component extraction now has extra assumptions: true source bars are monochrome components
     fully left of the separator and touching the baseline row, while true slots are one-cell
     top-row markers strictly right of the separator. The generator enforces both.

d) Placement legality
   - Grid sizes are restricted to 15 or 17 so the left half can safely hold four source bars
     with one background column between occupied columns.
   - Right-half markers also keep at least one background gap between occupied columns.
   - Bar heights are odd and capped so every bar stays in bounds and never reaches the marker row.
   - Output is built by clearing only the drawable body of the right half, then painting the
     retained shortest bars at the marker columns. Every accepted sample is verified with
     `main(input) == expected_output`.

e) Visual observability
   - Across examples, a human sees that the output changes only in the right half while the left
     half keeps all four source bars. That exposes the D7 split-region behavior.
   - The right half contains fewer bars than the left half in every sample, making the D12
     cardinality mismatch visible without hidden bookkeeping.
   - Because the kept bars are always the shortest ones, the dropped bars are the visibly taller
     leftovers that never appear on the right. This makes the selection rule legible from a few
     demonstrations rather than seeming arbitrary.

New implicit assumptions after the change
   - `main()` assumes exactly one full-height non-bg, non-baseline separator column exists.
   - `main()` assumes there are at least as many source bars as destination slots.
   - `main()` assumes destination slots are encoded only by top-row markers in the right half.
   - `generate_input()` guarantees those assumptions by construction and rejects samples where
     the source heights would already be left-to-right ascending.
"""

import numpy as np

from arc_utils import bbox_h, find_components, get_bg


def _detect_base(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    for r in range(g.shape[0]):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == int(bg):
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    if best is None:
        raise ValueError("failed to detect baseline")
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _find_separator(g: np.ndarray, bg: int, base_col: int) -> tuple[int, int]:
    H, W = g.shape
    candidates: list[tuple[int, int, int]] = []
    for c in range(W):
        vals = np.unique(g[:, c])
        if len(vals) != 1:
            continue
        col = int(vals[0])
        if col in {int(bg), int(base_col)}:
            continue
        candidates.append((abs(int(c) - W // 2), int(c), int(col)))
    if not candidates:
        raise ValueError("failed to detect separator column")
    _dist, sep_c, sep_col = min(candidates)
    return int(sep_c), int(sep_col)


def _extract_source_bars(
    g: np.ndarray,
    *,
    bg: int,
    base_r: int,
    base_col: int,
    sep_c: int,
    sep_col: int,
) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col), int(sep_col)}:
            continue

        b = comp["bbox"]
        if int(b[3]) >= int(sep_c):
            continue

        pts = comp["coords"]
        if not any(int(r) == int(base_r) for r, _c in pts):
            continue

        height = int(bbox_h(b))
        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        anchor = int(base_cols[len(base_cols) // 2])
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

    if not objects:
        raise ValueError("no source bars found")
    return objects


def _extract_slots(g: np.ndarray, *, bg: int, sep_c: int, sep_col: int) -> list[dict[str, int]]:
    slots: list[dict[str, int]] = []
    for c in range(int(sep_c) + 1, g.shape[1]):
        col = int(g[0, c])
        if col in {int(bg), int(sep_col)}:
            continue
        slots.append({"anchor": int(c), "color": int(col)})
    if not slots:
        raise ValueError("no destination slots found")
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - detect the centered separator and shared baseline,
    - read all source bars from the left half,
    - read right-half slot anchors/colors from the top row,
    - sort source bars by height ascending,
    - keep only the first M shapes where M is the number of slots,
    - paint those retained shortest shapes into the right half.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    base_r, base_col = _detect_base(g, bg)
    sep_c, sep_col = _find_separator(g, bg, base_col)

    objects = _extract_source_bars(g, bg=bg, base_r=base_r, base_col=base_col, sep_c=sep_c, sep_col=sep_col)
    slots = _extract_slots(g, bg=bg, sep_c=sep_c, sep_col=sep_col)
    if len(objects) < len(slots):
        raise ValueError("not enough source bars for slots")

    slots = sorted(slots, key=lambda o: int(o["anchor"]))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))
    chosen_shapes = shapes[: len(slots)]

    out = g.copy()
    for r in range(H):
        if r in {0, int(base_r)}:
            continue
        out[r, int(sep_c) + 1 :] = int(bg)

    for slot, shape in zip(slots, chosen_shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and int(sep_c) < cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _choose_local_cols(span: int, k: int) -> list[int]:
    used = int(2 * k - 1)  # width-1 objects with at least one bg gap
    extra = int(span - used)
    if extra < 0:
        raise ValueError("objects do not fit in half-width")
    gaps = _rand_composition(extra, k + 1)
    cols: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        cols.append(int(c))
        c += 2
        if i < k - 1:
            c += int(gaps[i + 1])
    return cols


def _draw_vertical_bar(grid: np.ndarray, *, base_r: int, c: int, color: int, height: int) -> None:
    half = int(height // 2)
    grid[int(base_r) - half : int(base_r) + half + 1, int(c)] = int(color)


def _build_input(
    n: int,
    *,
    bg: int,
    base_color: int,
    sep_color: int,
    base_r: int,
    src_cols: list[int],
    src_colors: list[int],
    src_heights: list[int],
    slot_cols: list[int],
    slot_colors: list[int],
) -> np.ndarray:
    g = np.full((int(n), int(n)), int(bg), dtype=int)
    sep_c = int(n // 2)
    g[:, sep_c] = int(sep_color)
    g[int(base_r), :sep_c] = int(base_color)
    g[int(base_r), sep_c + 1 :] = int(base_color)

    for c, color in zip(slot_cols, slot_colors):
        g[0, int(c)] = int(color)

    for c, color, height in zip(src_cols, src_colors, src_heights):
        _draw_vertical_bar(g, base_r=base_r, c=int(c), color=int(color), height=int(height))

    return g


def _build_output(
    inp: np.ndarray,
    *,
    bg: int,
    base_r: int,
    sep_c: int,
    slot_cols: list[int],
    slot_colors: list[int],
    kept_heights: list[int],
) -> np.ndarray:
    out = np.array(inp, copy=True)
    for r in range(out.shape[0]):
        if r in {0, int(base_r)}:
            continue
        out[r, int(sep_c) + 1 :] = int(bg)

    for c, color, height in zip(slot_cols, slot_colors, kept_heights):
        _draw_vertical_bar(out, base_r=base_r, c=int(c), color=int(color), height=int(height))
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([15, 17]))
    half_w = int(n // 2)
    base_r = int(n // 2)
    sep_c = int(n // 2)

    bg = 8
    base_color = 6
    sep_color = 5
    palette = [1, 2, 3, 4, 7, 9]

    src_count = 4
    slot_count = int(np.random.choice([2, 3]))

    src_local = _choose_local_cols(half_w, src_count)
    slot_local = _choose_local_cols(half_w, slot_count)
    src_cols = [int(c) for c in src_local]
    slot_cols = [int(sep_c + 1 + c) for c in slot_local]

    src_colors = [int(c) for c in np.random.choice(palette, size=src_count, replace=False)]
    slot_colors = [int(c) for c in np.random.choice(palette, size=slot_count, replace=False)]

    height_choices = list(range(1, min(9, n - 4) + 1, 2))
    sorted_pool = sorted(int(h) for h in np.random.choice(height_choices, size=src_count, replace=False))
    perm = np.random.permutation(src_count)
    if np.array_equal(perm, np.arange(src_count)):
        a, b = np.random.choice(src_count, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    src_heights = [int(sorted_pool[i]) for i in perm]
    if src_heights == sorted(src_heights):
        return None

    kept_heights = sorted(sorted_pool[:slot_count])
    discarded_heights = sorted(sorted_pool[slot_count:])
    if not discarded_heights:
        return None
    if int(discarded_heights[0]) <= int(kept_heights[-1]):
        return None

    inp = _build_input(
        n,
        bg=bg,
        base_color=base_color,
        sep_color=sep_color,
        base_r=base_r,
        src_cols=src_cols,
        src_colors=src_colors,
        src_heights=src_heights,
        slot_cols=slot_cols,
        slot_colors=slot_colors,
    )
    out = _build_output(
        inp,
        bg=bg,
        base_r=base_r,
        sep_c=sep_c,
        slot_cols=slot_cols,
        slot_colors=slot_colors,
        kept_heights=kept_heights,
    )

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
