# Combo variant of 2ba387bc (index 24)
# combo_spec: D8/0+D15/0
# Name: 多级分类+颜色图例

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the combo variant
=====================================

This variant fuses:
- D8/0: first group by color, then classify frame vs solid inside each color group
- D15/0: the color-group order is not implicit; it is explicitly given by a corner legend

1. Concrete behavior and small trace
------------------------------------
- Original task:
  find every scattered 4x4 object -> classify globally by shape -> stack all frames
  into one left column and all solids into one right column.
- Combo input:
  the scattered 4x4 objects remain, but the top-left corner now contains a small
  vertical legend, one colored cell per row. The legend order defines the output
  order of color groups from left to right.
- Combo output:
  each legend color gets its own 8-column block:
  - cols 8k..8k+3: objects of that color that are visual frames
  - cols 8k+4..8k+7: objects of that color that are visual solids
  All objects are copied unchanged, and each subgroup is stacked top-to-bottom by
  original `(row, col)` order.
- Example:
  legend = `[8, 3]` from top to bottom.
  scattered objects include:
  - color 3 frame at row 6
  - color 8 solid at row 10
  - color 8 frame at row 15
  - color 3 solid at row 18
  Original solver:
  - both frames go to the single global left column
  - both solids go to the single global right column
  Combo solver:
  - first create color-group 8, because legend says 8 comes first
    - frame(8) goes to output cols 0..3
    - solid(8) goes to output cols 4..7
  - then create color-group 3
    - frame(3) goes to output cols 8..11
    - solid(3) goes to output cols 12..15
  The key difference is the extra reasoning step:
  read legend-defined color groups first, then do shape classification inside each group.

2. Complexity calibration
-------------------------
- The generator uses 2 or 3 legend colors. One color would collapse the task back
  toward the original two-column layout.
- Every legend color appears in the scattered field, and every color group contains
  both at least one frame and at least one solid. This makes the within-group left/
  right split visible instead of accidental.
- The legend order is randomized independently from object placement. The generator
  rejects cases where the legend order equals the row-first appearance order of
  colors in the field, so the corner legend is genuinely needed to recover the
  left-to-right order of color groups.

3. Original-solver assumption audit
-----------------------------------
- `bg = get_bg(g)` remains valid because the background still dominates the grid.
- `find_components(... monochromatic=False)` remains valid if objects are separated.
  The legend cells form only a thin corner strip and never a 4x4 component.
- `if bbox_h(b) != 4 or bbox_w(b) != 4: continue` remains valid. Only sortable
  objects are 4x4; legend cells are 1x1.
- Original global branch:
    frame if interior is bg else solid
  is no longer sufficient by itself. Shape still matters, but only after grouping
  the object by legend color.
- Original fixed `outW = 8` is invalid. The combo output needs `8 * n_groups`
  because each color group gets its own frame/solid column pair.
- Sorting by `(row, col)` still works, but now it is applied inside each subgroup
  `(legend_color, shape_type)` rather than only inside the two global groups.

4. Placement legality
---------------------
- The legend is a contiguous top-left vertical strip in column 0.
- The main object field begins at row `legend_h + 2` and column `4`, leaving a
  blank gap so legend and objects cannot touch.
- Each 4x4 object is placed with `random_free_location(... padding=1,
  padding_connectivity=4)`, so different objects never become 4-connected.
- All patches are pasted fully inside the canvas, so there is no overlap and no
  clipping.

5. Visual observability
-----------------------
- A human can see that the output width changes from one pair of columns to
  several pairs of columns.
- The order of those column-pairs matches the legend, not the object positions and
  not the numeric color ids.
- Inside each pair, hollow 4x4 objects consistently occupy the left half and solid
  4x4 objects the right half.
- The legend disappears in the output, which makes it clearly an instruction area.

6. Visual cleanliness
---------------------
- The legend is confined to one corner and spatially separated from the sortable
  objects.
- Objects stay as plain 4x4 monochrome frame/solid blocks, so the field remains
  sparse and easy to parse even with repeated colors.

7. Change inferability
----------------------
- No recoloring occurs. Colors act only as group labels.
- The number of legend colors is small and every legend color is reused in the
  field, so the color-group rule is supported by repeated evidence.
- Because the generator rejects samples whose legend order equals the field's
  first-appearance color order, the legend is not just decorative.

8. Generator non-degeneration metrics
-------------------------------------
- Metric 1: `group_count = len(legend_colors)`. Reject `group_count < 2`.
- Metric 2: `mixed_group_count = #legend colors having both a frame and a solid`.
  Reject `mixed_group_count < group_count`.
- Metric 3: `legend_order_differs = legend_colors != row_first_color_order`.
  Reject if false; otherwise the legend affects output column-pair order.
- Metric 4: `combo_differs_from_original = main(grid) != original_shape_only_output`.
  This is checked in post-test statistics to verify the fused behavior is active.

9. New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: the legend is the contiguous run of non-background cells in column 0
  starting from row 0.
  Generator guarantee: legend is placed exactly that way and row `legend_h` is bg.
- Assumption: each sortable object is a monochrome 4x4 frame or a monochrome 4x4
  solid containing exactly one non-background color.
  Generator guarantee: only `_make_frame` and `_make_solid` are used.
- Assumption: every sortable object color appears exactly once in the legend.
  Generator guarantee: legend colors are distinct and objects only use those colors.
- Assumption: there is at least one object for every legend color.
  Generator guarantee: each color gets at least one frame and one solid object.
"""


ObjectRec = tuple[int, str, int, int, np.ndarray]


def _extract_legend(g: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    H, W = g.shape
    if W < 1:
        return colors
    r = 0
    while r < H and int(g[r, 0]) != bg:
        colors.append(int(g[r, 0]))
        r += 1
    return colors


def _object_color(sub: np.ndarray, bg: int) -> int | None:
    vals = sorted(set(map(int, sub[sub != bg].flatten())))
    if len(vals) != 1:
        return None
    return int(vals[0])


def _is_visual_frame(sub: np.ndarray, bg: int) -> bool:
    return bool((sub[1:3, 1:3] == int(bg)).all())


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    out = np.full((4, 4), int(col), dtype=int)
    out[1:3, 1:3] = int(bg)
    return out


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _stack_grouped(
    grouped: dict[int, dict[str, list[tuple[int, int, np.ndarray]]]],
    legend_colors: list[int],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(max(len(grouped[col]["frame"]), len(grouped[col]["solid"])) for col in legend_colors)
    out_w = 8 * len(legend_colors)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for gi, col in enumerate(legend_colors):
        base_c = 8 * gi
        for i, (_r0, _c0, sub) in enumerate(grouped[col]["frame"]):
            paste(out, sub, 4 * i, base_c)
        for i, (_r0, _c0, sub) in enumerate(grouped[col]["solid"]):
            paste(out, sub, 4 * i, base_c + 4)
    return out


def _group_objects(
    objects: list[ObjectRec],
    legend_colors: list[int],
    bg: int,
) -> dict[int, dict[str, list[tuple[int, int, np.ndarray]]]]:
    grouped: dict[int, dict[str, list[tuple[int, int, np.ndarray]]]] = {
        int(col): {"frame": [], "solid": []} for col in legend_colors
    }
    allowed = set(int(col) for col in legend_colors)
    for color, _kind, r0, c0, sub in objects:
        if int(color) not in allowed:
            continue
        bucket = "frame" if _is_visual_frame(sub, bg) else "solid"
        grouped[int(color)][bucket].append((int(r0), int(c0), sub))
    for col in legend_colors:
        grouped[int(col)]["frame"].sort(key=lambda t: (int(t[0]), int(t[1])))
        grouped[int(col)]["solid"].sort(key=lambda t: (int(t[0]), int(t[1])))
    return grouped


def _scan_objects(g: np.ndarray, bg: int) -> list[ObjectRec]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    objects: list[ObjectRec] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        color = _object_color(sub, bg)
        if color is None:
            continue
        kind = "frame" if _is_visual_frame(sub, bg) else "solid"
        objects.append((int(color), kind, int(r0), int(c0), sub))
    return objects


def _row_first_color_order(objects: list[ObjectRec]) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for color, _kind, r0, c0, _sub in sorted(objects, key=lambda t: (int(t[2]), int(t[3]))):
        if int(color) not in seen:
            seen.add(int(color))
            order.append(int(color))
    return order


def _original_shape_only_output(g: np.ndarray) -> np.ndarray:
    bg = int(get_bg(g))
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []
    for _color, kind, r0, c0, sub in _scan_objects(g, bg):
        if kind == "frame":
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))
    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, 4 * i, 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, 4 * i, 4)
    return out


def analyze_sample(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend_colors = _extract_legend(g, bg)
    objects = _scan_objects(g, bg)
    grouped = _group_objects(objects, legend_colors, bg)
    mixed_group_count = sum(
        int(len(grouped[col]["frame"]) > 0 and len(grouped[col]["solid"]) > 0) for col in legend_colors
    )
    row_first = _row_first_color_order(objects)
    combo_out = main(g)
    orig_out = _original_shape_only_output(g)
    same_as_original = int(combo_out.shape == orig_out.shape and np.array_equal(combo_out, orig_out))
    return {
        "group_count": int(len(legend_colors)),
        "mixed_group_count": int(mixed_group_count),
        "legend_order_matches_row_first": int(list(map(int, legend_colors)) == list(map(int, row_first))),
        "same_as_original": int(same_as_original),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Read the top-left legend column to get the ordered list of color groups.
    2. Collect all 4x4 monochrome objects on the background.
    3. For each legend color, split that color's objects into frames and solids.
    4. Output one 8-column block per legend color:
       left 4 columns = frames of that color, right 4 columns = solids of that color.
    5. Inside each subgroup, preserve the original top-to-bottom `(row, col)` order.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend_colors = _extract_legend(g, bg)
    objects = _scan_objects(g, bg)
    grouped = _group_objects(objects, legend_colors, bg)
    return _stack_grouped(grouped, legend_colors, bg)


def _place_legend(g: np.ndarray, legend_colors: list[int]) -> None:
    for r, col in enumerate(legend_colors):
        g[int(r), 0] = int(col)


def _counts_for_groups(group_count: int) -> tuple[list[int], list[int]]:
    frame_counts = [int(np.random.randint(1, 3)) for _ in range(group_count)]
    solid_counts = [int(np.random.randint(1, 3)) for _ in range(group_count)]
    if max(frame_counts + solid_counts) <= 1:
        if bool(np.random.randint(0, 2)):
            frame_counts[int(np.random.randint(0, group_count))] = 2
        else:
            solid_counts[int(np.random.randint(0, group_count))] = 2
    return frame_counts, solid_counts


def _place_objects(
    g: np.ndarray,
    legend_colors: list[int],
    frame_counts: list[int],
    solid_counts: list[int],
    bg: int,
) -> list[ObjectRec] | None:
    legend_h = len(legend_colors)
    field_r0 = legend_h + 2
    field_c0 = 4
    if g.shape[0] - field_r0 < 10 or g.shape[1] - field_c0 < 10:
        return None

    _place_legend(g, legend_colors)
    field = g[field_r0:, field_c0:]
    objects: list[ObjectRec] = []
    try:
        for color, n_frames, n_solids in zip(legend_colors, frame_counts, solid_counts):
            for _ in range(int(n_frames)):
                patch = _make_frame(int(color), bg)
                r0, c0 = random_free_location(
                    field,
                    patch,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=4,
                )
                paste(g, patch, field_r0 + r0, field_c0 + c0, transparent=bg)
                objects.append((int(color), "frame", field_r0 + int(r0), field_c0 + int(c0), patch))
            for _ in range(int(n_solids)):
                patch = _make_solid(int(color))
                r0, c0 = random_free_location(
                    field,
                    patch,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=4,
                )
                paste(g, patch, field_r0 + r0, field_c0 + c0, transparent=bg)
                objects.append((int(color), "solid", field_r0 + int(r0), field_c0 + int(c0), patch))
    except Exception:
        return None
    return objects


def _build_expected(objects: list[ObjectRec], legend_colors: list[int], bg: int) -> np.ndarray:
    grouped = _group_objects(objects, legend_colors, bg)
    return _stack_grouped(grouped, legend_colors, bg)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    palette = [int(x) for x in np.random.permutation(np.arange(1, 10))]
    group_count = int(np.random.randint(2, 4))
    legend_colors = palette[:group_count]
    frame_counts, solid_counts = _counts_for_groups(group_count)

    H = int(np.random.randint(28, 35))
    W = int(np.random.randint(28, 35))
    g = np.full((H, W), bg, dtype=int)

    objects = _place_objects(g, legend_colors, frame_counts, solid_counts, bg)
    if objects is None:
        return None

    row_first = _row_first_color_order(objects)
    if list(map(int, row_first)) == list(map(int, legend_colors)):
        return None

    expected = _build_expected(objects, legend_colors, bg)
    actual = main(g)
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        return None

    orig = _original_shape_only_output(g)
    if expected.shape == orig.shape and np.array_equal(expected, orig):
        return None

    metrics = analyze_sample(g)
    if metrics["group_count"] < 2:
        return None
    if metrics["mixed_group_count"] < metrics["group_count"]:
        return None
    if metrics["legend_order_matches_row_first"] != 0:
        return None
    if metrics["same_as_original"] != 0:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 800 attempts")
