# Combo variant of 2ba387bc (index 29)
# combo_spec: D10/1+D15/0
# Name: 三列输出+颜色图例

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo variant D10/1 + D15/0
================================================

Reference-task note
-------------------
- The prompt points to `9385bd28` as the source for the legend-style instruction.
- In this workspace there is an evaluation JSON and prompt notes for `9385bd28`,
  but no standalone solver file to inspect directly.
- The usable baseline for this combo is therefore:
  1. the original `2ba387bc` solver/generator,
  2. the existing D15 single-variant implementation for `2ba387bc`,
  3. the existing 3-type single-variant implementation for `2ba387bc`,
  4. the prompt note that `9385bd28` communicates rules through a corner legend.

a) Concrete behavior and small trace
------------------------------------
- Original task:
  - scattered 4x4 objects on a background;
  - geometry alone decides the bucket:
    hollow frame -> left output column,
    solid block -> right output column.
- D10/1 adds a third visually distinct 4x4 template and a third output column.
  In this combo the three visible templates are:
  - frame: border color + 2x2 background center,
  - solid: all 16 cells same color,
  - diagonal-center: border color + center `[[color, bg], [bg, color]]`.
- D15/0 changes the instruction source:
  - the top-left corner contains a 3-column color legend;
  - legend column 0 lists colors for output cols 0..3,
  - legend column 1 lists colors for output cols 4..7,
  - legend column 2 lists colors for output cols 8..11.
- Critical fusion point:
  - visible template no longer determines the destination column;
  - object color membership in the legend determines the destination column.
- Small trace:
  - suppose legend rows are `[8, 3, 2]` then `[6, 0, 7]`, where `0` means blank
    background in that legend cell, so effective column sets are:
    col0={8,6}, col1={3}, col2={2,7}.
  - scattered objects include a solid `8`, a frame `3`, and a frame `7`.
  - Old 3-type geometry solver would send `8` to the solid column and `7` to the
    frame column.
  - Combo solver must send color `8` to output column 0 and color `7` to output
    column 2, because the legend overrides visible template.
  - The output still copies each 4x4 object exactly as seen; only the stacking
    column changes.

b) Complexity calibration
-------------------------
- The legend has exactly 3 explicit columns because the output also has exactly
  3 object columns. This makes the new control structure visually obvious.
- Each semantic column uses only 1-3 colors, and each listed color is guaranteed
  to appear in at least one scattered object. This keeps the rule compact.
- The generator reuses colors within a semantic column across multiple objects,
  so the legend is not a one-off token with no repeated evidence.
- The scene still uses only three fixed 4x4 templates. That is enough visual
  variety to make the third output column noticeable without introducing a large
  pattern family or rotational ambiguity.
- To avoid simplification back to a geometry-only task, every sample contains at
  least two objects whose visual template index disagrees with their legend column.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- `bg = get_bg(g)`: still valid. The background remains the dominant color.
- `find_components(... monochromatic=False)`: still valid if objects stay
  separated. The legend may form a small multicolor component, but it is never
  4x4, so it will be ignored by the object pass.
- `if bbox_h(b) != 4 or bbox_w(b) != 4: continue`: still valid. Sortable objects
  remain exactly 4x4; legend entries are 1x1 cells in a corner block.
- Original classification branch:
    `if frame-like then frames else solids`
  is invalid twice in the combo:
  1. there are now three visible templates rather than two;
  2. destination is legend-controlled anyway, so even correct visual recognition
     is not enough to decide output placement.
- Original output width `8` is invalid. The combo needs width `12`.
- Original output height `4 * max(#frames, #solids)` is invalid. The combo needs
  `4 * max(counts across all three legend buckets)`.
- Original interpretation of object color as incidental is invalid. Color is now
  the instruction key because it connects scattered objects to the corner legend.

d) Placement legality
---------------------
- The legend occupies only the top-left 3 columns for a small number of rows.
- The scattered-object field starts at least two rows below the legend and at
  least two blank columns to its right, so legend and objects cannot touch.
- Objects are still placed with `random_free_location(... padding=1,
  padding_connectivity=4)`, which prevents overlap and 4-neighbor contact.
- Every placed patch is exactly 4x4 and is pasted wholly inside a reserved field,
  so nothing crosses grid boundaries.

e) Visual observability
-----------------------
- A human can see that:
  - the corner legend disappears in the output;
  - output width is 12 with three independent stacks;
  - colors listed in legend column k consistently populate output column k;
  - visible template is not a reliable cue because the same template can land in
    different output columns across examples.
- The rule depends only on visible structure: corner legend position and object
  colors. It does not rely on hidden numeric properties of color ids.

f) Visual cleanliness
---------------------
- The legend is isolated in one corner with blank margin around it.
- Objects remain sparse, separated 4x4 squares on a plain background.
- No overlays or intersecting markers are used, so the scene remains readable.

g) Change inferability
----------------------
- There is no recoloring transformation. Colors are preserved exactly.
- The only color-based inference is membership in one of three legend columns.
- The palette size per sample is intentionally small and every legend color is
  used in the main field, so the mapping can be inferred from 2-4 examples.

8) Generator non-degeneration metric
------------------------------------
- Metric 1: `legend_override_count`
  = number of objects whose visual-template column (`frame=0`, `solid=1`,
    `diag=2`) differs from their legend-assigned output column.
- If `legend_override_count == 0`, the sample degenerates into the simple 3-type
  geometry task because grouping by shape would match grouping by legend.
- Metric 2: `distinct_visuals_per_semantic`
  = for each legend bucket, how many different visual templates appear there.
- Samples are rejected unless:
  - `legend_override_count >= 2`, and
  - at least one semantic bucket contains at least 2 visual templates.
- These conditions ensure the fusion really combines the two dimensions instead
  of leaving one dimension inactive.

9) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: the legend is the contiguous run of top rows where at least one of
  columns 0..2 is non-background. Generator guarantee: legend is placed exactly
  there and is followed by a blank separator row.
- Assumption: every sortable 4x4 object uses exactly one non-background color.
  Generator guarantee: all three templates are monochrome with optional bg cells.
- Assumption: each object color appears in exactly one legend column.
  Generator guarantee: the three legend color sets are pairwise disjoint.
- Assumption: only objects with legend-listed colors are task objects.
  Generator guarantee: every scattered object color is drawn from the legend.
- Assumption: objects are spatially separated enough to remain distinct 4-neighbor
  components. Generator guarantee: padded placement enforces that separation.
"""


FRAME_KIND = 0
SOLID_KIND = 1
DIAG_KIND = 2
KIND_NAMES = ("frame", "solid", "diag")


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _make_diag(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = np.array([[int(col), int(bg)], [int(bg), int(col)]], dtype=int)
    return patch


def _make_patch(kind: int, col: int, bg: int) -> np.ndarray:
    if int(kind) == FRAME_KIND:
        return _make_frame(col, bg)
    if int(kind) == SOLID_KIND:
        return _make_solid(col)
    return _make_diag(col, bg)


def _extract_legend(g: np.ndarray, bg: int) -> list[list[int]]:
    legend_cols: list[list[int]] = [[], [], []]
    if g.shape[1] < 3:
        return legend_cols

    r = 0
    while r < g.shape[0] and any(int(g[r, c]) != int(bg) for c in range(3)):
        for c in range(3):
            val = int(g[r, c])
            if val != int(bg):
                legend_cols[c].append(val)
        r += 1
    return legend_cols


def _object_color(sub: np.ndarray, bg: int) -> int | None:
    cols = sorted(set(map(int, sub[sub != int(bg)].flatten())))
    if len(cols) != 1:
        return None
    return int(cols[0])


def _visual_kind(sub: np.ndarray, bg: int) -> int | None:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_cols = set(map(int, border.flatten()))
    center = sub[1:3, 1:3]

    if len(set(map(int, sub.flatten()))) == 1 and int(sub[0, 0]) != int(bg):
        return SOLID_KIND

    if len(border_cols) != 1:
        return None
    col = int(next(iter(border_cols)))
    if col == int(bg):
        return None
    if bool((center == int(bg)).all()):
        return FRAME_KIND

    diag_center = np.array([[col, int(bg)], [int(bg), int(col)]], dtype=int)
    if bool((center == diag_center).all()):
        return DIAG_KIND
    return None


def _stack_groups(groups: list[list[tuple[int, int, np.ndarray]]], bg: int) -> np.ndarray:
    out_h = 4 * max((len(group) for group in groups), default=0)
    out = np.full((out_h, 12), int(bg), dtype=int)
    for group_idx, group in enumerate(groups):
        for i, (_r0, _c0, sub) in enumerate(group):
            paste(out, sub, int(i * 4), int(group_idx * 4))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top-left 3-column color legend and stack all scattered 4x4 objects
    into 3 output columns by legend membership, preserving top-to-bottom order
    within each legend-defined group.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend_cols = _extract_legend(g, bg)
    legend_sets = [set(col_list) for col_list in legend_cols]

    groups: list[list[tuple[int, int, np.ndarray]]] = [[], [], []]
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        col = _object_color(sub, bg)
        if col is None:
            continue
        for idx, legend_set in enumerate(legend_sets):
            if col in legend_set:
                groups[idx].append((int(r0), int(c0), sub))
                break

    for group in groups:
        group.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _stack_groups(groups, bg)


def _build_expected(
    objects: list[tuple[int, int, int, np.ndarray, int]],
    bg: int,
) -> np.ndarray:
    groups: list[list[tuple[int, int, np.ndarray]]] = [[], [], []]
    for semantic_idx, r0, c0, sub, _kind in objects:
        groups[int(semantic_idx)].append((int(r0), int(c0), sub))
    for group in groups:
        group.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _stack_groups(groups, bg)


def _legend_rows(legend_cols: list[list[int]]) -> int:
    return max((len(col_list) for col_list in legend_cols), default=0)


def _visual_schedule(counts: list[int]) -> list[list[int]]:
    total = sum(int(x) for x in counts)
    visuals = [int(np.random.randint(0, 3)) for _ in range(total)]

    if total >= 3:
        visuals[:3] = [FRAME_KIND, SOLID_KIND, DIAG_KIND]
        np.random.shuffle(visuals)

    offsets = np.cumsum([0] + counts)
    mismatch_count = sum(
        visuals[i] != semantic_idx
        for semantic_idx in range(3)
        for i in range(int(offsets[semantic_idx]), int(offsets[semantic_idx + 1]))
    )
    if mismatch_count < 2:
        changed = 0
        for semantic_idx in range(3):
            lo = int(offsets[semantic_idx])
            hi = int(offsets[semantic_idx + 1])
            if hi <= lo:
                continue
            pos = lo
            visuals[pos] = int((semantic_idx + 1 + changed) % 3)
            changed += 1
            if changed >= 2:
                break

    per_group: list[list[int]] = []
    for semantic_idx in range(3):
        lo = int(offsets[semantic_idx])
        hi = int(offsets[semantic_idx + 1])
        per_group.append([int(v) for v in visuals[lo:hi]])

    bucket_visual_variety = [len(set(group_visuals)) for group_visuals in per_group]
    if max(bucket_visual_variety, default=0) < 2:
        target = int(np.argmax(counts))
        if counts[target] >= 2:
            per_group[target][0] = FRAME_KIND
            per_group[target][1] = SOLID_KIND if target != SOLID_KIND else DIAG_KIND
        elif counts[target] >= 1:
            per_group[target][0] = int((target + 1) % 3)
            other = int((target + 2) % 3)
            for idx in range(3):
                if idx != target and counts[idx] >= 1:
                    per_group[idx][0] = other
                    break
    return per_group


def _metric_legend_override(objects: list[tuple[int, int, int, np.ndarray, int]]) -> int:
    return sum(int(semantic_idx) != int(kind) for semantic_idx, _r0, _c0, _sub, kind in objects)


def _metric_bucket_visual_variety(objects: list[tuple[int, int, int, np.ndarray, int]]) -> list[int]:
    per_bucket: list[set[int]] = [set(), set(), set()]
    for semantic_idx, _r0, _c0, _sub, kind in objects:
        per_bucket[int(semantic_idx)].add(int(kind))
    return [len(s) for s in per_bucket]


def _place_objects(
    g: np.ndarray,
    legend_cols: list[list[int]],
    counts: list[int],
    bg: int,
) -> list[tuple[int, int, int, np.ndarray, int]] | None:
    legend_h = _legend_rows(legend_cols)
    field_r0 = legend_h + 2
    field_c0 = 5
    if g.shape[0] - field_r0 < 8 or g.shape[1] - field_c0 < 8:
        return None

    for c, col_list in enumerate(legend_cols):
        for r, col in enumerate(col_list):
            g[int(r), int(c)] = int(col)

    visual_groups = _visual_schedule(counts)
    field = g[field_r0:, field_c0:]
    objects: list[tuple[int, int, int, np.ndarray, int]] = []

    try:
        for semantic_idx in range(3):
            colors = list(legend_cols[semantic_idx])
            assigned_colors = list(colors)
            while len(assigned_colors) < counts[semantic_idx]:
                assigned_colors.append(int(np.random.choice(colors)))
            np.random.shuffle(assigned_colors)

            for col, kind in zip(assigned_colors, visual_groups[semantic_idx]):
                patch = _make_patch(kind, int(col), bg)
                r0, c0 = random_free_location(
                    field,
                    patch,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=4,
                )
                paste(g, patch, field_r0 + int(r0), field_c0 + int(c0), transparent=bg)
                objects.append((int(semantic_idx), field_r0 + int(r0), field_c0 + int(c0), patch, int(kind)))
    except Exception:
        return None

    override_count = _metric_legend_override(objects)
    bucket_visual_variety = _metric_bucket_visual_variety(objects)
    if override_count < 2:
        return None
    if max(bucket_visual_variety, default=0) < 2:
        return None
    return objects


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    palette = [int(x) for x in np.random.permutation(np.arange(1, 10))]

    color_counts = [int(np.random.randint(1, 4)) for _ in range(3)]
    if sum(color_counts) > len(palette):
        return None

    legend_cols: list[list[int]] = []
    start = 0
    for count in color_counts:
        legend_cols.append(palette[start : start + count])
        start += count

    counts = [int(np.random.randint(max(2, len(legend_cols[i])), 6)) for i in range(3)]
    legend_h = _legend_rows(legend_cols)
    min_h = legend_h + 2 + 10
    min_w = 5 + 10
    h = int(np.random.randint(max(24, min_h), 34))
    w = int(np.random.randint(max(24, min_w), 34))
    g = np.full((h, w), bg, dtype=int)

    objects = _place_objects(g, legend_cols, counts, bg)
    if objects is None:
        return None

    expected = _build_expected(objects, bg)
    actual = main(g)
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")
