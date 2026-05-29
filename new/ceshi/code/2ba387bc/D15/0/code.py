# Variant of 2ba387bc
# Dimension: D15 — 符号指令——分类规则的编码方式 (symbol_instructions)
# Change: 无显式符号指令，分类规则完全通过观察方块形状特征隐式推断
#   → 增加颜色图例——指示哪些颜色为框架、哪些为实心

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the D15 variant
===================================

Reference-task note
-------------------
- The prompt asks to read the source-task solver for `9385bd28` first.
- In this workspace there is an evaluation JSON for `9385bd28`, plus prompt notes
  describing it as a legend-driven task, but there is no readable standalone
  `9385bd28.py` solver file to inspect.
- The concrete baseline used here is therefore:
  1. the original `2ba387bc` solver/generator code,
  2. the visible I/O behavior of `9385bd28`,
  3. the prompt note that `9385bd28` uses a corner legend whose local color pairs
     instruct the interpretation of the larger scene.

1. Concrete behavior and small trace
------------------------------------
- Original input: several scattered 4x4 monochrome objects on background.
  Visual frames have a hollow 2x2 center; visual solids are fully filled.
- Original output: same 4x4 objects are copied unchanged, but re-stacked into two
  columns: left for visually hollow objects, right for visually solid objects.
- Variant input: the scattered 4x4 objects remain, but the top-left corner now
  contains a tiny two-column legend.
  - legend col 0 = colors that should go to the left output column
  - legend col 1 = colors that should go to the right output column
- Critical change: object geometry no longer determines the semantic bucket.
  Any listed color may appear as either a hollow-looking 4x4 frame or a solid
  4x4 block in the scattered field. The output still copies each object exactly
  as seen; only its destination column is legend-controlled.
- Example trace:
  - suppose legend rows are `[8|3]` then `[6|2]`.
  - scattered objects include a solid `8` block, a hollow `3` frame, and a hollow
    `2` frame.
  - Original solver would place hollow `2` on the left and solid `8` on the right.
  - Variant solver must place color `8` on the left and color `2` on the right,
    because category comes from the legend rather than from the visible interior.
  - That is the exact point where the old logic and new logic diverge.

2. Complexity calibration
-------------------------
- The legend uses only two explicit columns and a small number of colors per side
  (1-3 frame-colors, 1-3 solid-colors). This keeps the instruction compact and
  human-readable.
- Colors are reused across multiple objects when possible so the legend is not a
  one-off decoration. A color name in the legend usually governs more than one
  object across the training distribution.
- To avoid a visually invisible variant, the generator enforces at least one
  semantic/geometry mismatch in every sample:
  - either a color listed on the left appears as a solid-looking 4x4 block,
  - or a color listed on the right appears as a hollow-looking 4x4 frame,
  - often both.
- This is enough to break the original shape-based heuristic while preserving the
  rest of the task skeleton.

3. Original-solver assumption audit
-----------------------------------
- `bg = get_bg(g)`: still valid. Background remains the most common color.
- `find_components(... monochromatic=False)`: still valid if objects stay spatially
  separated. The legend is small and never 4x4, so it will not be mistaken for a
  sortable object.
- `if bbox_h(b) != 4 or bbox_w(b) != 4: continue`: still valid. Only scattered
  target objects are 4x4; legend entries are 1x1 and never expand to 4x4.
- Old branch:
    `if len(bcols) == 1 and (sub[1:3,1:3] == bg).all(): frames else solids`
  This is invalid in the variant because geometry is no longer the instruction.
  A hollow-looking object can be semantically "solid" and vice versa.
- Sorting by `(row, col)`: still valid and kept unchanged.
- Output width 8 and 4-row stacking cadence: still valid and kept unchanged.

4. Placement legality
---------------------
- The legend is fixed to the top-left corner using only cols 0 and 1.
- The scattered-object field starts at least two rows below the legend and four
  columns to the right of it, so legend and objects cannot touch.
- Objects are placed with `random_free_location(... padding=1, connectivity=4)`,
  which prevents 4-neighbor contact between 4x4 objects. Since the solver uses
  `connectivity=4`, distinct objects remain distinct components.
- Because object patches are 4x4 and pasted wholly within a reserved sub-canvas,
  they never cross the legend boundary or the outer grid boundary.

5. Visual observability
-----------------------
- A human seeing train pairs can notice:
  - the tiny corner legend disappears in the output,
  - colors listed in legend col 0 consistently populate output col 0,
  - colors listed in legend col 1 consistently populate output col 1,
  - visible hollowness is not a reliable cue anymore.
- The rule does not rely on numeric color ids; it relies on visible placement of
  colors inside the legend block.

6. Visual cleanliness
---------------------
- The legend is confined to a corner and separated from the main field by blank
  margin, avoiding overlap or ambiguous layering.
- Objects stay as simple 4x4 monochrome shapes, so even with repeated colors the
  grid remains easy to parse.

7. Color-change inference
-------------------------
- There is no recoloring transformation. Colors act only as category membership
  tokens via the legend.
- The number of legend colors is small and each legend color is guaranteed to
  appear in the scattered field, so every listed token has visible evidence.

8. Generator non-degeneration metric
------------------------------------
- Metric: `mismatch_count = #objects whose visible geometry disagrees with the
  legend-assigned semantic bucket`.
- If `mismatch_count == 0`, the sample degenerates back toward the original task,
  because shape-only reasoning would still work.
- The generator rejects such samples and only returns grids with
  `mismatch_count >= 1`.

9. New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: the legend is the contiguous top-left run of non-background rows in
  columns 0 and 1. Generator guarantee: objects start at row `legend_h + 2` and
  col `4`, leaving a blank separator row/column region.
- Assumption: every sortable 4x4 object contains exactly one non-background color.
  Generator guarantee: every object is either a monochrome hollow frame or a
  monochrome solid block.
- Assumption: every sortable object color appears in exactly one legend column.
  Generator guarantee: frame-color and solid-color palettes are disjoint.
"""


def _extract_legend(g: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    frame_cols: list[int] = []
    solid_cols: list[int] = []
    H, W = g.shape
    if W < 2:
        return frame_cols, solid_cols

    r = 0
    while r < H and (int(g[r, 0]) != bg or int(g[r, 1]) != bg):
        if int(g[r, 0]) != bg:
            frame_cols.append(int(g[r, 0]))
        if int(g[r, 1]) != bg:
            solid_cols.append(int(g[r, 1]))
        r += 1
    return frame_cols, solid_cols


def _object_color(sub: np.ndarray, bg: int) -> int | None:
    cols = sorted(set(map(int, sub[sub != bg].flatten())))
    if len(cols) != 1:
        return None
    return int(cols[0])


def _is_visual_frame(sub: np.ndarray, bg: int) -> bool:
    return bool((sub[1:3, 1:3] == int(bg)).all())


def _stack_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, i * 4, 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, i * 4, 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Read the top-left 2-column legend.
      - left legend column => colors that belong in output cols 0..3
      - right legend column => colors that belong in output cols 4..7
    - Collect all scattered 4x4 monochrome objects on the background.
    - Ignore each object's visible geometry when deciding the destination column;
      use only the legend-defined color membership.
    - Within each semantic group, preserve the original top-to-bottom order.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_colors, solid_colors = _extract_legend(g, bg)
    frame_set = set(frame_colors)
    solid_set = set(solid_colors)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        col = _object_color(sub, bg)
        if col is None:
            continue
        if col in frame_set:
            frames.append((int(r0), int(c0), sub))
        elif col in solid_set:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _stack_output(frames, solids, bg)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _build_expected(
    objects: list[tuple[str, int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    frames = [(r0, c0, sub) for semantic, r0, c0, sub in objects if semantic == "frame"]
    solids = [(r0, c0, sub) for semantic, r0, c0, sub in objects if semantic == "solid"]
    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _stack_output(frames, solids, bg)


def _legend_rows(frame_colors: list[int], solid_colors: list[int]) -> int:
    return max(len(frame_colors), len(solid_colors))


def _place_objects(
    g: np.ndarray,
    frame_colors: list[int],
    solid_colors: list[int],
    n_frame_objs: int,
    n_solid_objs: int,
    bg: int,
) -> list[tuple[str, int, int, np.ndarray]] | None:
    legend_h = _legend_rows(frame_colors, solid_colors)
    field_r0 = legend_h + 2
    field_c0 = 4
    if g.shape[0] - field_r0 < 6 or g.shape[1] - field_c0 < 6:
        return None

    for i, col in enumerate(frame_colors):
        g[i, 0] = int(col)
    for i, col in enumerate(solid_colors):
        g[i, 1] = int(col)

    frame_visuals = [bool(np.random.randint(0, 2)) for _ in range(n_frame_objs)]
    solid_visuals = [bool(np.random.randint(0, 2)) for _ in range(n_solid_objs)]

    mismatch_count = frame_visuals.count(False) + solid_visuals.count(True)
    if mismatch_count == 0:
        if n_frame_objs > 0 and bool(np.random.randint(0, 2)):
            frame_visuals[int(np.random.randint(0, n_frame_objs))] = False
        else:
            solid_visuals[int(np.random.randint(0, n_solid_objs))] = True

    frame_assign = list(frame_colors)
    while len(frame_assign) < n_frame_objs:
        frame_assign.append(int(np.random.choice(frame_colors)))
    solid_assign = list(solid_colors)
    while len(solid_assign) < n_solid_objs:
        solid_assign.append(int(np.random.choice(solid_colors)))
    np.random.shuffle(frame_assign)
    np.random.shuffle(solid_assign)

    field = g[field_r0:, field_c0:]
    objects: list[tuple[str, int, int, np.ndarray]] = []
    try:
        for col, visual_frame in zip(frame_assign, frame_visuals):
            patch = _make_frame(col, bg) if visual_frame else _make_solid(col)
            r0, c0 = random_free_location(field, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, field_r0 + r0, field_c0 + c0, transparent=bg)
            objects.append(("frame", field_r0 + int(r0), field_c0 + int(c0), patch))
        for col, visual_frame in zip(solid_assign, solid_visuals):
            patch = _make_frame(col, bg) if visual_frame else _make_solid(col)
            r0, c0 = random_free_location(field, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, field_r0 + r0, field_c0 + c0, transparent=bg)
            objects.append(("solid", field_r0 + int(r0), field_c0 + int(c0), patch))
    except Exception:
        return None

    observed_mismatch_count = sum(
        (_is_visual_frame(sub, bg) and semantic == "solid")
        or ((not _is_visual_frame(sub, bg)) and semantic == "frame")
        for semantic, _r0, _c0, sub in objects
    )
    if observed_mismatch_count <= 0:
        return None
    return objects


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    palette = list(np.random.permutation(np.arange(1, 10)))

    n_frame_cols = int(np.random.randint(1, 4))
    n_solid_cols = int(np.random.randint(1, 4))
    frame_colors = [int(x) for x in palette[:n_frame_cols]]
    solid_colors = [int(x) for x in palette[n_frame_cols : n_frame_cols + n_solid_cols]]

    n_frame_objs = int(np.random.randint(max(2, n_frame_cols), 6))
    n_solid_objs = int(np.random.randint(max(2, n_solid_cols), 6))

    legend_h = _legend_rows(frame_colors, solid_colors)
    min_h = legend_h + 2 + 8
    min_w = 4 + 8
    H = int(np.random.randint(max(22, min_h), 31))
    W = int(np.random.randint(max(22, min_w), 31))
    g = np.full((H, W), bg, dtype=int)

    objects = _place_objects(g, frame_colors, solid_colors, n_frame_objs, n_solid_objs, bg)
    if objects is None:
        return None

    expected = _build_expected(objects, bg)
    actual = main(g)
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
