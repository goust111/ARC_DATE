# Variant of 31f7f899
# Dimension: D4 — 目标槽位：原地替换 (dst)
# Change: 条形的原始列位置即为槽位，保持位置不变，只替换条形形状... → 填入框架洞中

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


# Design analysis for this D4 variant
# -----------------------------------
# a) Concrete behavior and trace
# - Input keeps the original "source bars on a baseline" idea: several colored vertical bars
#   touch a long horizontal baseline row. Their heights are all distinct.
# - The destination is no longer the bars' original x-columns. Instead, the grid also contains
#   one large solid frame block with several empty vertical holes carved inside it. Each hole
#   is a destination slot. A same-width colored marker strip sits directly below each hole and
#   gives that hole its slot color.
# - Output erases the source bars, preserves the baseline, preserves the frame + markers, and
#   fills the left-to-right holes with the source bar heights sorted shortest->tallest. The
#   i-th hole receives the i-th shortest height and is colored with that hole's own marker color.
# - Small trace: suppose the source bars (left->right colors) are blue/red/green with heights
#   5/2/4. The frame has three holes, markers blue/red/green left->right. Original logic would
#   redraw heights 2/4/5 back at the three source columns. Variant logic instead clears the
#   source bars, then fills the three frame holes with heights 2/4/5, colored blue/red/green.
#   The sorting rule is unchanged; only the destination representation changes from raw columns
#   to nested holes inside a container.
#
# b) Complexity calibration
# - The frame is a large solid monochrome rectangle, so the "container with internal holes"
#   is visually obvious from 2-3 examples.
# - Each sample uses 3-5 holes, not just one or two, so left-to-right ordering is clearly
#   active rather than incidental.
# - Hole markers repeat the slot colors already seen on the source side. This keeps the original
#   "slot keeps its color identity" idea visible even though the slots moved into a frame.
# - Hole heights are all the same (a capacity / socket), while fill heights vary. This avoids
#   collapsing back to direct shape-match-by-hole-size; the solver still needs the source bars'
#   global height ordering.
#
# c) Original-code assumptions that no longer hold
# - Original base-row detection used the row/color with maximal count. That breaks once a large
#   frame is present, because a frame row can have more pixels than the baseline. New solver
#   instead finds the unique non-bg row color that spans both left and right edges, which the
#   generator guarantees only for the baseline.
# - Original destination slots were inferred from each source bar's anchor column. That no longer
#   works because destination holes are separate objects. New solver must explicitly locate the
#   largest frame, then enumerate bg-hole components inside its bbox.
# - Original rendering started from a blank canvas with only the baseline restored. That would
#   destroy the frame/markers. New solver starts from the input, erases only the source bars,
#   and then fills the frame holes.
# - Original relative-shape transfer around a shared baseline is unnecessary here. To make hole
#   filling legible and robust, this variant uses bottom-aligned solid rectangles both for source
#   bars and for fills inside the holes.
#
# d) Placement legality guaranteed by the generator
# - Source bars are spaced on the baseline, never touch the grid edges, and fit above the base.
# - The frame is a solid rectangle that does not touch grid edges and does not intersect the
#   baseline. Holes are fully enclosed bg components inside the frame bbox, separated by at least
#   one frame-colored column, so component extraction is unambiguous.
# - Marker strips lie one row below the frame and stay outside the frame bbox, so they do not
#   contaminate hole extraction.
# - Source bar width always equals hole width, and every bar height is <= hole capacity.
#
# e) Visual observability
# - The rule difference is visible: input shows bars outside the frame and empty holes inside;
#   output shows the bars gone and the holes filled. This cannot be mistaken for the original
#   "same x-position slots" behavior.
# - Because source colors and hole-marker colors share the same left-to-right palette order,
#   humans can see that slot identity is preserved while the destination geometry changed.
#
# New implicit assumptions added by main(), and how generate_input() guarantees them
# - There is exactly one edge-spanning baseline row/color: generator draws a full-width baseline
#   and keeps bars away from the extreme columns.
# - The largest non-baseline monochrome component is the frame: generator makes the frame a large
#   solid block, much larger than any bar or marker.
# - Source bars are exactly the monochrome components touching the baseline and not using the
#   frame/base colors: generator draws bars to include the baseline row, while markers never
#   touch it.
# - Each hole has exactly one non-bg marker color directly below its column span: generator draws
#   one same-color strip per hole and leaves the rest of that row as bg.


def _find_base_row_and_color(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, W = g.shape
    best = None  # (count, row, color)
    for r in range(H):
        edge_col = int(g[r, 0])
        if edge_col == bg or edge_col != int(g[r, W - 1]):
            continue
        cnt = int(np.count_nonzero(g[r] == edge_col))
        if best is None or cnt > best[0]:
            best = (cnt, int(r), int(edge_col))
    assert best is not None
    return int(best[1]), int(best[2])


def _find_frame_component(g: np.ndarray, bg: int, base_col: int) -> dict[str, object]:
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    frame = max((comp for comp in comps if int(comp["color"]) != int(base_col)), key=lambda d: int(d["area"]))
    return frame


def _extract_source_bars(
    g: np.ndarray,
    bg: int,
    base_r: int,
    base_col: int,
    frame_col: int,
) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col), int(frame_col)}:
            continue
        pts = comp["coords"]
        if not np.any(pts[:, 0] == int(base_r)):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        objects.append(
            {
                "color": int(col),
                "coords": pts,
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "anchor": int(c0),
            }
        )
    return objects


def _extract_holes(g: np.ndarray, bg: int, frame: dict[str, object]) -> list[dict[str, int]]:
    H, _W = g.shape
    frame_col = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])
    sub = crop_bbox(g, frame["bbox"])
    hole_mask = (sub == int(bg))

    holes: list[dict[str, int]] = []
    marker_r = int(r1 + 1)
    for coords in components_mask(hole_mask, connectivity=4):
        if coords.size == 0:
            continue
        hr0, hc0, hr1, hc1 = map(int, bbox_coords(coords))
        if hr0 == 0 or hc0 == 0 or hr1 == sub.shape[0] - 1 or hc1 == sub.shape[1] - 1:
            continue
        gc0 = int(c0 + hc0)
        gc1 = int(c0 + hc1)
        vals: list[int] = []
        if marker_r < H:
            vals = [int(v) for v in g[marker_r, gc0 : gc1 + 1] if int(v) not in {int(bg), int(frame_col)}]
        assert vals
        slot_col = int(vals[0])
        holes.append(
            {
                "color": int(slot_col),
                "r0": int(r0 + hr0),
                "c0": int(gc0),
                "r1": int(r0 + hr1),
                "c1": int(gc1),
                "width": int(gc1 - gc0 + 1),
            }
        )
    return sorted(holes, key=lambda h: (int(h["c0"]), int(h["color"])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Read source bars from the shared baseline area.
    - Read destination slots as holes inside the large frame.
    - Sort source bars by height ascending.
    - Sort holes left to right.
    - Erase the source bars and fill the i-th hole with the i-th shortest height, recolored to
      that hole's marker color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_r, base_col = _find_base_row_and_color(g, bg)
    frame = _find_frame_component(g, bg, base_col)
    frame_col = int(frame["color"])

    objects = _extract_source_bars(g, bg, base_r, base_col, frame_col)
    holes = _extract_holes(g, bg, frame)
    assert objects and holes and len(objects) == len(holes)

    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["width"]), int(o["anchor"]), int(o["color"])))

    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            rr = int(r)
            cc = int(c)
            out[rr, cc] = int(base_col) if rr == int(base_r) else int(bg)

    for hole, shape in zip(holes, shapes):
        height = int(shape["height"])
        width = int(shape["width"])
        assert width == int(hole["width"])
        top = int(hole["r1"]) - int(height) + 1
        patch = np.full((height, width), int(hole["color"]), dtype=int)
        paste(out, patch, top, int(hole["c0"]))

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(span: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(span - used)
    if extra < 0:
        raise ValueError("slots do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    cur = int(gaps[0])
    for i in range(k):
        lefts.append(int(cur))
        cur += int(width)
        if i < k - 1:
            cur += 1 + int(gaps[i + 1])
    return lefts


def _draw_base_frame_scene(
    n: int,
    bg: int,
    base_r: int,
    base_color: int,
    frame_color: int,
    frame_top: int,
    frame_left: int,
    hole_height: int,
    hole_lefts: list[int],
    width: int,
    colors: list[int],
    inner_w: int,
) -> tuple[np.ndarray, int]:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)

    frame_height = int(hole_height + 2)
    frame_bottom = int(frame_top + frame_height - 1)
    frame_right = int(frame_left + inner_w + 1)
    grid[frame_top : frame_bottom + 1, frame_left : frame_right + 1] = int(frame_color)

    hole_top = int(frame_top + 1)
    hole_bottom = int(frame_bottom - 1)
    for local_left in hole_lefts:
        left = int(frame_left + 1 + local_left)
        grid[hole_top : hole_bottom + 1, left : left + width] = int(bg)

    marker_r = int(frame_bottom + 1)
    for local_left, color in zip(hole_lefts, colors):
        left = int(frame_left + 1 + local_left)
        grid[marker_r, left : left + width] = int(color)

    return grid, int(hole_bottom)


def _add_source_bars(
    grid: np.ndarray,
    base_r: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    out = grid.copy()
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(out, patch, int(base_r - height + 1), int(left))
    return out


def _fill_holes(
    grid: np.ndarray,
    hole_bottom: int,
    hole_lefts: list[int],
    frame_left: int,
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    out = grid.copy()
    for local_left, color, height in zip(hole_lefts, colors, heights):
        left = int(frame_left + 1 + local_left)
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(out, patch, int(hole_bottom - height + 1), int(left))
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(23, 28))
    bg = 8
    base_color = 6
    frame_color = 4
    base_r = int(n - 2)

    width = 1
    possible_k = [k for k in range(3, 6) if int(4 * k + 3) <= int(n)]
    if not possible_k:
        return None
    k = int(np.random.choice(possible_k))

    max_hole_height = int(min(7, base_r - 4))
    possible_hole_heights = [h for h in range(max(4, k + 1), max_hole_height + 1)]
    if not possible_hole_heights:
        return None
    hole_height = int(np.random.choice(possible_hole_heights))

    frame_height = int(hole_height + 2)
    max_frame_top = int(base_r - frame_height - 2)
    if max_frame_top < 1:
        return None
    frame_top = int(np.random.randint(1, max_frame_top + 1))

    min_inner_w = int(k * width + (k - 1))
    min_source_span = int(k * width + (k - 1))
    max_inner_w = int(n - min_source_span - 5)
    if min_inner_w > max_inner_w:
        return None
    inner_w = int(np.random.randint(min_inner_w, max_inner_w + 1))
    hole_lefts = _slot_lefts(inner_w, k, width)
    frame_width = int(inner_w + 2)
    frame_left = int(n - frame_width - 1)
    source_span = int(frame_left - 2)
    if source_span < min_source_span:
        return None
    source_lefts = [int(x + 1) for x in _slot_lefts(source_span, k, width)]

    palette = [1, 2, 3, 5, 7, 9]
    colors = [int(c) for c in np.random.choice(palette, size=k, replace=False)]
    sorted_heights = sorted(int(h) for h in np.random.choice(np.arange(2, hole_height + 1), size=k, replace=False))

    base_scene, hole_bottom = _draw_base_frame_scene(
        n=n,
        bg=bg,
        base_r=base_r,
        base_color=base_color,
        frame_color=frame_color,
        frame_top=frame_top,
        frame_left=frame_left,
        hole_height=hole_height,
        hole_lefts=hole_lefts,
        width=width,
        colors=colors,
        inner_w=inner_w,
    )

    out = _fill_holes(
        grid=base_scene,
        hole_bottom=hole_bottom,
        hole_lefts=hole_lefts,
        frame_left=frame_left,
        width=width,
        colors=colors,
        heights=sorted_heights,
    )

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_heights = [int(sorted_heights[i]) for i in perm]
    inp = _add_source_bars(
        grid=base_scene,
        base_r=base_r,
        lefts=source_lefts,
        width=width,
        colors=colors,
        heights=in_heights,
    )

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
