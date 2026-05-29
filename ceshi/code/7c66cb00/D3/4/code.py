# Variant of 7c66cb00
# Dimension: D3 — 源对象：精灵结构与颜色组成 (src)
# Change: 双色精灵：外层色（多数像素）包裹内层色（少数像素，可选），2x2到5x9尺寸，左右对称，在顶部区域互不重叠... → 分离式双组件精灵
#
# Design analysis
# ===============
# variant_spec:
#   preserved:
#     grid: 30x24
#     bars:
#       count: 2-3
#       geometry: full-width horizontal bars with edge border colors
#       heights: 4-8
#       vertical_layout: stacked bottom-up with exactly 1 bg row between bars
#     transform_core:
#       - clear the top area
#       - match each sprite to the bar whose fill color equals the sprite outer color
#       - bottom-align the sprite at its original x-position
#       - recolor outer -> target border, inner -> target fill
#     implementation_quirk_preserved:
#       - the inner-color sub-object also docks independently into its own bar
#   changed:
#     top_sprite_definition:
#       old: one connected two-tone sprite
#       new: exactly two disconnected monochrome blobs that are spatially near
#     grouping_rule:
#       - the two nearby blobs form one logical sprite
#       - outer blob = larger-area blob
#       - inner blob = smaller-area blob
#       - the pair is recovered by proximity, not by connectivity
#   concrete_geometry:
#     sprite_group:
#       count: 2-3
#       colors: exactly 2 bar-fill colors per group, colors differ
#       outer_blob:
#         shape: solid rectangle
#         area: larger than inner area
#       inner_blob:
#         shape: solid rectangle
#         area: smaller than outer area
#       union_bbox:
#         height: 2-5
#         width: 3-9
#       within_group_spacing:
#         - blobs are separated by exactly one blank row or one blank column
#         - small offset is allowed on the orthogonal axis
#       between_group_spacing:
#         - union bboxes are separated by at least 3 cells
#         - this guarantees proximity-grouping does not merge distinct sprites
#
# behavior_trace:
#   original_solver_failure_point:
#     - it scans top-area monochrome connected components
#     - each component bbox is treated as a full sprite
#     - with separated blobs, the outer blob and inner blob would be misread as two independent sprites
#   new_solver_behavior:
#     - detect all monochrome components in the top area
#     - dilate the entire non-bg mask by one cell and label those proximity regions
#     - components falling in the same dilated region are grouped as one sprite
#     - choose the larger component as outer, smaller as inner
#     - move the grouped bbox into the outer bar with recoloring
#     - also dock the inner blob by itself into its own bar, preserving the original implementation quirk
#   small_example:
#     - input top: a 3x2 red block, one blank column, then a 1x2 blue block
#     - old logic: red block docks alone into red bar, blue block docks alone into blue bar
#     - new logic: the pair docks together into the red bar as one opaque local bbox; the blue block inside the
#       pair turns red-fill, the blank gap around it is preserved as bg, and the blue block also appears once in
#       the blue bar as a border-colored sub-object
#
# complexity_calibration:
#   - every generated sprite has two blobs, so the D3 change is always visible
#   - only 2-3 bars and 2-3 sprite groups are used, keeping the hypothesis space small
#   - outer-vs-inner is structurally recoverable from area, not from hidden color ordering
#   - proximity grouping is supported by repeated examples because multiple top objects use the same rule
#
# implicit_assumption_audit:
#   original_main_dependencies_that_break:
#     - iterating per-color connected components assumes each logical sprite is connected enough that the
#       outer component bbox already contains the inner part; this is false now
#     - picking the most common color in a bbox as outer still works only after correct grouping
#     - in-place sequential recoloring can accidentally cascade when a border color equals another bar fill;
#       this was mostly harmless before but is unsafe in the variant too, so recoloring now uses masks
#   new_main_assumptions:
#     - each proximity group contains exactly one larger blob and one smaller blob of different colors
#     - distinct sprite groups are far enough apart that one-cell dilation does not merge them
#     - the larger blob color matches exactly one bar fill
#   generator_guarantees:
#     - exactly two blobs per group
#     - outer area > inner area
#     - inter-group spacing >= 3 cells
#     - target-bar occupancy is checked for both grouped docking and inner-only docking
#
# visual_observability:
#   - the top area visibly changes from connected two-tone patches to pairs of nearby detached blobs
#   - transparent pasting would make the inner blob disappear after recoloring, so grouped sprites are pasted
#     opaquely into the outer bar; this is the minimal supporting change needed to make D3 observable
#   - the output preserves each pair's relative layout and the bg gap inside the target bar, so the grouping is legible
#   - no sorting depends on color ids; only visible size and visible proximity matter
#
# cleanliness:
#   - blobs are solid rectangles with blank space around/between them
#   - groups are kept well separated to avoid clutter
#   - all extra output placements occur inside the bottom bars, as in the original implementation family

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (3, 9)
GEN_RETRIES = 400
TOP_GROUP_MARGIN = 3


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    bars: list[dict] = []
    current_bar = None
    for r in range(H):
        row = g[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(W * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar:
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                    }
            elif current_bar:
                bars.append(current_bar)
                current_bar = None
        elif current_bar:
            bars.append(current_bar)
            current_bar = None
    if current_bar:
        bars.append(current_bar)
    return bars


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - 1)
        r1 = min(H, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(W, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _collect_top_components(top: np.ndarray, bg: int) -> list[dict]:
    components: list[dict] = []
    for color in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(color)
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                    "patch": crop_bbox(top, (int(r0), int(c0), int(r1), int(c1))),
                }
            )
    return components


def _group_components(top: np.ndarray, bg: int) -> list[dict]:
    components = _collect_top_components(top, bg)
    if not components:
        return []

    expanded = _dilate_once(top != int(bg))
    group_labels, _ = label_mask(expanded, connectivity=8)
    grouped: dict[int, list[dict]] = {}
    for comp in components:
        first_r, first_c = map(int, comp["coords"][0])
        group_id = int(group_labels[first_r, first_c])
        grouped.setdefault(group_id, []).append(comp)

    patterns: list[dict] = []
    for comps in grouped.values():
        all_coords = np.vstack([comp["coords"] for comp in comps])
        r0, c0, r1, c1 = bbox_coords(all_coords)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        content = crop_bbox(top, bbox)

        ordered = sorted(comps, key=lambda comp: (-int(comp["area"]), int(comp["bbox"][1]), int(comp["bbox"][0])))
        outer = ordered[0]
        inner = None
        for comp in ordered[1:]:
            if int(comp["color"]) != int(outer["color"]):
                inner = comp
                break

        inner_color = None
        inner_patch = None
        inner_abs_c = None
        inner_h = None
        if inner is not None:
            ir0, ic0, ir1, ic1 = map(int, inner["bbox"])
            inner_color = int(inner["color"])
            inner_patch = crop_bbox(top, inner["bbox"])
            inner_abs_c = int(ic0)
            inner_h = int(ir1 - ir0 + 1)

        patterns.append(
            {
                "outer_color": int(outer["color"]),
                "inner_color": inner_color,
                "r0": int(r0),
                "c0": int(c0),
                "h": int(content.shape[0]),
                "w": int(content.shape[1]),
                "content": content,
                "inner_patch": inner_patch,
                "inner_abs_c": inner_abs_c,
                "inner_h": inner_h,
            }
        )

    return patterns


def _recolor_group_patch(
    patch: np.ndarray,
    outer_color: int,
    inner_color: int | None,
    new_outer: int,
    new_inner: int,
) -> np.ndarray:
    transformed = np.array(patch, dtype=int).copy()
    outer_mask = patch == int(outer_color)
    transformed[outer_mask] = int(new_outer)
    if inner_color is not None:
        inner_mask = patch == int(inner_color)
        transformed[inner_mask] = int(new_inner)
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(Counter(g.flatten()).most_common(1)[0][0])

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    pattern_area_end = min(int(bar["r_min"]) for bar in bars)
    top = g[:pattern_area_end, :]
    patterns = _group_components(top, bg)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    fill_to_bar = {int(bar["fill"]): bar for bar in bars}
    for pat in patterns:
        outer_bar = fill_to_bar.get(int(pat["outer_color"]))
        if outer_bar is None:
            continue

        transformed = _recolor_group_patch(
            pat["content"],
            int(pat["outer_color"]),
            pat["inner_color"],
            int(outer_bar["border"]),
            int(outer_bar["fill"]),
        )
        place_r = int(outer_bar["r_max"] - pat["h"] + 1)
        paste(out, transformed, place_r, int(pat["c0"]))

        if pat["inner_color"] is not None and pat["inner_patch"] is not None and pat["inner_abs_c"] is not None:
            inner_bar = fill_to_bar.get(int(pat["inner_color"]))
            if inner_bar is not None:
                inner_border_patch = np.full_like(np.array(pat["inner_patch"], dtype=int), int(inner_bar["border"]))
                inner_place_r = int(inner_bar["r_max"] - inner_border_patch.shape[0] + 1)
                paste(out, inner_border_patch, inner_place_r, int(pat["inner_abs_c"]))

    return out


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = heights[i]
        r0, r1 = r - h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = fills[i]
        grid[r0 : r1 + 1, 0] = borders[i]
        grid[r0 : r1 + 1, GRID_W - 1] = borders[i]
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = r0 - 1
    return grid, bars


def _make_group_patch(bg: int, outer_color: int, inner_color: int, max_outer_h: int) -> dict | None:
    for _ in range(200):
        outer_h = random.randint(1, min(3, max_outer_h, SPRITE_H_RANGE[1]))
        outer_w = random.randint(2, 5)
        inner_h = random.randint(1, min(2, max_outer_h, SPRITE_H_RANGE[1]))
        inner_w = random.randint(1, 3)
        if outer_h * outer_w <= inner_h * inner_w:
            continue

        if random.random() < 0.5:
            row_delta = random.randint(-1, 1)
            outer_top = max(0, -row_delta)
            inner_top = max(0, row_delta)
            h = max(outer_top + outer_h, inner_top + inner_h)
            w = outer_w + 1 + inner_w
            if not (SPRITE_H_RANGE[0] <= h <= SPRITE_H_RANGE[1] and SPRITE_W_RANGE[0] <= w <= SPRITE_W_RANGE[1]):
                continue
            outer_left = 0
            inner_left = outer_w + 1
            if random.random() < 0.5:
                outer_left, inner_left = inner_w + 1, 0
        else:
            col_delta = random.randint(-1, 1)
            outer_left = max(0, -col_delta)
            inner_left = max(0, col_delta)
            h = outer_h + 1 + inner_h
            w = max(outer_left + outer_w, inner_left + inner_w)
            if not (SPRITE_H_RANGE[0] <= h <= SPRITE_H_RANGE[1] and SPRITE_W_RANGE[0] <= w <= SPRITE_W_RANGE[1]):
                continue
            outer_top = 0
            inner_top = outer_h + 1
            if random.random() < 0.5:
                outer_top, inner_top = inner_h + 1, 0

        patch = np.full((h, w), bg, dtype=int)
        patch[outer_top : outer_top + outer_h, outer_left : outer_left + outer_w] = outer_color
        patch[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = inner_color
        if np.any(patch == outer_color) and np.any(patch == inner_color):
            return {
                "patch": patch,
                "h": int(h),
                "w": int(w),
                "inner_bbox": (int(inner_top), int(inner_left), int(inner_h), int(inner_w)),
            }
    return None


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[start : start + length].any()


def _rect_separated(
    rects: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    margin: int = TOP_GROUP_MARGIN,
) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - margin or rr1 < r0 - margin or c1 < cc0 - margin or cc1 < c0 - margin):
            return False
    return True


def _make_sprite(outer_idx: int, bars: list[dict], bg: int) -> dict | None:
    outer_color = int(bars[outer_idx]["fill"])
    inner_choices = [i for i in range(len(bars)) if i != outer_idx]
    inner_idx = random.choice(inner_choices)
    inner_color = int(bars[inner_idx]["fill"])
    shape = _make_group_patch(bg, outer_color, inner_color, int(bars[outer_idx]["h"]))
    if shape is None:
        return None
    return {
        "outer_idx": int(outer_idx),
        "inner_idx": int(inner_idx),
        "patch": shape["patch"],
        "h": int(shape["h"]),
        "w": int(shape["w"]),
        "inner_bbox": shape["inner_bbox"],
    }


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx, inner_idx = int(sp["outer_idx"]), int(sp["inner_idx"])
        ir0, ic0, ih, iw = map(int, sp["inner_bbox"])

        candidates = []
        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_idx], c0, w):
                continue
            if not _interval_free(bar_occ[inner_idx], c0 + ic0, iw):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][c0 : c0 + w] = True
        bar_occ[inner_idx][c0 + ic0 : c0 + ic0 + iw] = True
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()
    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[int(sp["outer_idx"])]
        inner_bar = bars[int(sp["inner_idx"])]
        h, c0 = int(sp["h"]), int(sp["c0"])
        ir0, ic0, ih, iw = map(int, sp["inner_bbox"])

        paste(inp, patch, int(sp["r0"]), c0, transparent=int(bg))

        transformed = _recolor_group_patch(
            patch,
            int(outer_bar["fill"]),
            int(inner_bar["fill"]),
            int(outer_bar["border"]),
            int(outer_bar["fill"]),
        )
        paste(out, transformed, int(outer_bar["r_max"]) - h + 1, c0)

        inner_patch = patch[ir0 : ir0 + ih, ic0 : ic0 + iw]
        inner_only = np.full_like(inner_patch, int(inner_bar["border"]))
        paste(out, inner_only, int(inner_bar["r_max"]) - ih + 1, c0 + ic0)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)

    sprites: list[dict] = []
    for i in range(n_sprites):
        sp = _make_sprite(int(outer_slots[i]), bars, bg)
        if sp is None:
            return None
        sprites.append(sp)
    sprites.sort(key=lambda sp: (-int(sp["w"]), -int(sp["h"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

    if get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
