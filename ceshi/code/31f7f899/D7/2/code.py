# Variant of 31f7f899
# Dimension: D7 — 网格形状与尺寸 (grid_shape)
# Change: 正方形网格，7-17行列，奇数边长... → 输入输出分离为两个区域
#
# Design analysis for this D7 variant
# -----------------------------------
# a) 具体化与行为追踪
# - The grid is still an odd-sized square, but the center column is now a solid separator frame.
#   This divides the canvas into a left source half and a right destination half.
# - Left half: unsorted monochrome vertical bars cross the shared baseline row, exactly as in the
#   original task's "objects to be ranked".
# - Right half: no bars are present initially; only one-pixel colored slot markers appear on the
#   top row. Their left-to-right order defines the destination slots and their colors.
# - Output: preserve the whole left source half and fill the right half with the same bar shapes,
#   sorted by height from shortest to tallest and recolored by the right-half slot markers.
# - Small trace:
#   source heights on the left by x-order = [5, 1, 3]
#   slot markers on the right by x-order = [red, yellow, blue]
#   original task on one undivided baseline would sort those three source bars and put them back
#   onto the same x-slots. In this variant the sorted shapes [1, 3, 5] are drawn only in the
#   right half, at the marker x-positions, with colors [red, yellow, blue]. The key difference is
#   that source extraction and destination placement happen in different spatial regions.
#
# b) 复杂度校准
# - The separator is mandatory and spans the full height, so the left/right partition is visually
#   obvious in every sample.
# - There are always 3-4 bars/slots, enough to show an ordering pattern but few enough that the
#   half-width remains readable on 11-17 sized boards.
# - Bar width is fixed to 1. This is an intentional adaptation to the split layout: it keeps the
#   original "sort by height" rule intact while fitting multiple objects into each half without
#   overlap or ambiguity.
# - Heights are distinct odd values and are always shuffled away from sorted order on the source
#   side, so the transfer is visually nontrivial.
#
# c) 原始 main() 的隐式假设审查
# - Original base-row detection is still valid because the baseline row remains the row with the
#   largest same-color run. The separator only contributes one cell per row, so it does not win.
# - Original component scan assumed every non-baseline component was a candidate bar. That breaks
#   here because right-half slot markers are also components. New solver therefore filters source
#   bars by region (strictly left of the separator) and by touching the baseline row.
# - Original slot extraction used the bars' own x-positions/colors as slots. That assumption no
#   longer holds because destination slots are decoupled from source bars. New solver reads slots
#   from top-row markers in the right half.
# - Original output started from a blank canvas and rebuilt the whole scene. That would erase the
#   separator and the left source half. New solver starts from a copy of the input, clears only the
#   drawable body of the right half, and paints the sorted result there.
#
# d) 放置合法性
# - Generator uses the center column as the only separator, leaving equal half-widths.
# - In each half, occupied columns are chosen with at least one background gap between them, so
#   source bars and slot markers stay visually separated.
# - Heights are capped at n-2, reserving the top row for markers and the bottom row as background.
# - Source bars all cross the shared baseline row; right-half output bars use the same vertical
#   footprint, so main() can reuse relative coordinates safely.
#
# e) 视觉可观察性
# - A human sees a consistent template: left half contains unsorted bars, center has a divider,
#   right half contains only colored markers, and the output populates the right half.
# - Because the source bars remain visible in the output, the "copy sorted silhouettes into the
#   other region" behavior is easier to infer from 2-3 examples than if the source half vanished.
# - The separator is the key D7 signal. Without understanding the spatial partition, a solver that
#   only sorts bars in-place cannot explain why the output changes only on the right side.
#
# Reference-task note
# - The prompt cites 5545f144 as the inspiration for frame_separation. In the provided workspace
#   its official JSON is available, but a solver file for 5545f144 is not present after repo-wide
#   search. The implementation below therefore uses only the observable "horizontal concatenation
#   of equal-purpose regions separated by solid columns" pattern from that task's I/O examples.
#
# New main() assumptions introduced by the variant
# - Exactly one full-height solid separator column exists; generator fixes it to the center column.
# - Right-half slot markers live on the top row and use colors distinct from bg/base/separator.
# - Every true source bar lies entirely in the left half and touches the baseline row.
# - The number of source bars equals the number of right-half slot markers.
# The generator explicitly enforces all four conditions.

from __future__ import annotations

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
    Variant rule:
    - detect a full-height separator column that splits the grid into left/right halves,
    - read unsorted vertical bars from the left half,
    - read destination slot colors/anchors from top-row markers in the right half,
    - sort source bar shapes by height and draw them into the right half only.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    base_r, base_col = _detect_base(g, bg)
    sep_c, sep_col = _find_separator(g, bg, base_col)

    objects = _extract_source_bars(g, bg=bg, base_r=base_r, base_col=base_col, sep_c=sep_c, sep_col=sep_col)
    slots = _extract_slots(g, bg=bg, sep_c=sep_c, sep_col=sep_col)
    if len(objects) != len(slots):
        raise ValueError("bar/slot count mismatch")

    slots = sorted(slots, key=lambda o: int(o["anchor"]))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = g.copy()
    for r in range(H):
        if r in {0, int(base_r)}:
            continue
        out[r, int(sep_c) + 1 :] = int(bg)

    for slot, shape in zip(slots, shapes):
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
    used = int(2 * k - 1)  # width 1 bars/markers with at least one bg gap between occupied columns
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
    sorted_heights: list[int],
) -> np.ndarray:
    out = np.array(inp, copy=True)
    for r in range(out.shape[0]):
        if r in {0, int(base_r)}:
            continue
        out[r, int(sep_c) + 1 :] = int(bg)

    for c, color, height in zip(slot_cols, slot_colors, sorted_heights):
        half = int(height // 2)
        out[int(base_r) - half : int(base_r) + half + 1, int(c)] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(11, 18, 2)))
    half_w = int(n // 2)
    base_r = int(n // 2)
    sep_c = int(n // 2)

    bg = 8
    base_color = 6
    sep_color = 5
    palette = [1, 2, 3, 4, 7, 9]

    height_choices = list(range(1, min(9, n - 2) + 1, 2))
    max_count = min(4, len(height_choices), (half_w + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    src_local = _choose_local_cols(half_w, k)
    slot_local = _choose_local_cols(half_w, k)
    src_cols = [int(c) for c in src_local]
    slot_cols = [int(sep_c + 1 + c) for c in slot_local]

    slot_colors = [int(c) for c in np.random.choice(palette, size=k, replace=False)]
    src_colors = [int(c) for c in np.random.permutation(slot_colors)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    src_heights = [int(sorted_heights[i]) for i in perm]

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
        sorted_heights=sorted_heights,
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
