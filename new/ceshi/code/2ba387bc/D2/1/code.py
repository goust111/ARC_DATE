# Variant of 2ba387bc
# Dimension: D2 — 排序规则——组内排列顺序 (match_cond)
# Change: 每组（框架组和实心组）内部按输入中的行位置(row)从上到下排序，行相同时按列位置(col)排序... → 按行位置降序排列（从下到上）

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for this D2 variant
# a) Concrete behavior and trace:
#    - Input still contains scattered 4x4 objects on a uniform background.
#    - Two object types remain unchanged:
#      1. Frame: one color on the 1-cell border, 2x2 center equals background.
#      2. Solid: the full 4x4 block is filled by one non-background color.
#    - Output still has width 8, with frames stacked in cols 0..3 and solids in cols 4..7.
#    - The only rule change is the order *within* each type-group:
#      old rule: sort by (row ascending, col ascending),
#      new rule: sort by (row descending, col ascending).
#    - Small trace:
#      Suppose frames are at top-left rows [2, 10, 17]. Old output order is rows 2 -> 10 -> 17.
#      New output order is rows 17 -> 10 -> 2. The object colors/shapes are unchanged; only the
#      vertical stacking order in the left output column is reversed. Solids follow the same change.
# b) Complexity calibration:
#    - The task is still visually simple: only two shape categories and one ordering cue (input row).
#    - To make the reversal observable, the generator enforces at least two frames and two solids,
#      unique colors across objects, and distinct top rows within each group.
# c) Implicit assumptions in the original main():
#    - `find_components(..., monochromatic=False)` assumes each object is isolated by background.
#      This still holds, so the generator preserves at least 1-cell padding.
#    - `bbox_h == bbox_w == 4` assumes every relevant object occupies a 4x4 bbox. Unchanged.
#    - The frame test assumes a monochrome border plus bg-filled 2x2 interior. Unchanged.
#    - The old `sort(key=(row, col))` encoded the original ascending order; this is the only solver
#      line that depends on the old D2 property and must be changed.
# d) Placement legality:
#    - Objects are placed with `random_free_location(..., border=1, padding=1, padding_connectivity=4)`,
#      so they do not overlap, do not touch, and stay in bounds.
#    - Post-generation validation re-runs the object collector to ensure the intended counts survive.
# e) Visual observability:
#    - Color ids are never used as a hidden ordering key.
#    - Ordering depends only on visible vertical position in the input.
#    - The generator rejects any sample whose descending output matches the old ascending output.
# f) Visual cleanliness:
#    - Objects remain sparse, non-overlapping, and easy to segment by eye.
#    - Each object uses a unique color to make reordered stacks easy to track.
# g) Color inference:
#    - Colors carry identity only; there is no color mapping rule to infer.
# 9) New assumptions introduced by the modified main():
#    - If multiple objects of the same type share the same top row, ordering falls back to col ascending.
#      The generator avoids same-row ties so the intended row-direction reversal is always visible.
#    - The solver still assumes that any non-frame 4x4 component should be treated as a solid. The
#      generator only emits true frames and true solids, so this assumption remains safe.


def _collect_objects(grid: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)

        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        border_colors = set(map(int, border.flatten()))
        is_frame = len(border_colors) == 1 and bool((sub[1:3, 1:3] == bg).all())

        if is_frame:
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    return bg, frames, solids


def _sort_group(group: list[tuple[int, int, np.ndarray]], *, descending: bool) -> list[tuple[int, int, np.ndarray]]:
    if descending:
        return sorted(group, key=lambda t: (-int(t[0]), int(t[1])))
    return sorted(group, key=lambda t: (int(t[0]), int(t[1])))


def _render_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    *,
    descending: bool,
) -> np.ndarray:
    ordered_frames = _sort_group(frames, descending=descending)
    ordered_solids = _sort_group(solids, descending=descending)

    out_h = 4 * max(int(len(ordered_frames)), int(len(ordered_solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(ordered_frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(ordered_solids):
        paste(out, sub, int(i * 4), 4)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type: frames left, solids right.

    Variant rule:
    - Classify each 4x4 object as frame or solid by its interior.
    - Sort each group bottom-to-top by input row (row descending).
    - If rows tie, keep column ascending as the tiebreak.
    - Stack frames in left column (cols 0-3), solids in right (cols 4-7).
    """
    bg, frames, solids = _collect_objects(np.array(grid, dtype=int))
    return _render_output(bg, frames, solids, descending=True)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _pick_unique_color(used_cols: set[int]) -> int:
    col = int(np.random.randint(1, 10))
    while col in used_cols:
        col = int(np.random.randint(1, 10))
    used_cols.add(col)
    return col


def _place_with_new_row(
    grid: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    used_rows: set[int],
) -> tuple[int, int]:
    for _ in range(80):
        r0, c0 = random_free_location(
            grid,
            patch,
            transparent=bg,
            border=1,
            padding=1,
            padding_connectivity=4,
        )
        if int(r0) not in used_rows:
            return int(r0), int(c0)
    raise ValueError("could not find a placement with a fresh row")


def _variant_is_observable(grid: np.ndarray) -> bool:
    bg, frames, solids = _collect_objects(grid)
    if len(frames) < 2 or len(solids) < 2:
        return False

    frame_rows = [int(r0) for r0, _c0, _sub in frames]
    solid_rows = [int(r0) for r0, _c0, _sub in solids]
    if len(set(frame_rows)) != len(frame_rows):
        return False
    if len(set(solid_rows)) != len(solid_rows):
        return False

    desc = _render_output(bg, frames, solids, descending=True)
    asc = _render_output(bg, frames, solids, descending=False)
    return not np.array_equal(desc, asc)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    grid = np.full((h, w), bg, dtype=int)

    # Both groups need multiple members so the row-order reversal is learnable from examples.
    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(2, 5))

    try:
        used_cols: set[int] = set()
        frame_rows: set[int] = set()
        solid_rows: set[int] = set()

        for _ in range(n_frames):
            col = _pick_unique_color(used_cols)
            patch = _make_frame(col, bg)
            r0, c0 = _place_with_new_row(grid, patch, bg=bg, used_rows=frame_rows)
            frame_rows.add(int(r0))
            paste(grid, patch, r0, c0, transparent=bg)

        for _ in range(n_solids):
            col = _pick_unique_color(used_cols)
            patch = _make_solid(col)
            r0, c0 = _place_with_new_row(grid, patch, bg=bg, used_rows=solid_rows)
            solid_rows.add(int(r0))
            paste(grid, patch, r0, c0, transparent=bg)

        _bg2, frames, solids = _collect_objects(grid)
        if len(frames) != n_frames or len(solids) != n_solids:
            return None
        if not _variant_is_observable(grid):
            return None

        # Final solver pass confirms this input is valid for the variant solver.
        main(grid)
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
