# Variant of 2ba387bc
# Dimension: D5 — 目标布局——两列排列方式 (dst)
# Change: 框架放入左列（cols 0-3），实心放入右列（cols 4-7），两列紧邻无间隙... → 上下排列替代左右排列

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the D5 variant
# a) Concrete behavior:
#    Input stays the same as the original task: several separated 4x4 objects lie on a
#    background. A frame is a 4x4 border with a bg-colored 2x2 hole; a solid is a full
#    4x4 block. The original output placed the i-th frame at rows 4*i, cols 0:4 and the
#    i-th solid at rows 4*i, cols 4:8, so both categories occupied parallel vertical
#    columns. In this variant, frames still appear first and solids still appear second,
#    but the layout becomes one single 4-cell-wide column: all sorted frames are stacked
#    from the top, then all sorted solids continue immediately underneath. Example with
#    two frames F1/F2 and one solid S1 sorted by input row:
#    - Original: rows 0:4 -> F1 at left, S1 at right; rows 4:8 -> F2 at left, bg at right.
#    - Variant: rows 0:4 -> F1; rows 4:8 -> F2; rows 8:12 -> S1.
#    The change affects only output assembly; detection/classification and sorting remain.
# b) Complexity calibration:
#    The objects keep two visually distinct shape classes and distinct colors, so the new
#    rule is still inferable from 2-4 examples. The output difference is strong because
#    width shrinks from 8 to 4 and the category split changes from horizontal to vertical.
#    Generator keeps 1-4 frames and 1-4 solids so both segments are normally present.
# c) Implicit assumption audit of main():
#    - get_bg / find_components assume objects are disconnected foreground regions. The
#      generator preserves 1-cell padding so this still holds.
#    - bbox_h==bbox_w==4 assumes every relevant object is exactly 4x4. Generator still
#      only places 4x4 sprites.
#    - Frame test "single border color + bg 2x2 interior" still defines the class.
#    - Sorting by (row, col) remains valid because the variant changes destination only.
#    - The old outH=max(len(frames), len(solids))*4 and outW=8 no longer hold; they were
#      tied to side-by-side columns. They must become outH=(len(frames)+len(solids))*4 and
#      outW=4, with solids pasted after the frame segment.
# d) Placement legality:
#    Generator uses random_free_location with border=1 and padding=1, so 4x4 objects do
#    not overlap, do not touch, and stay in bounds. That guarantees component extraction
#    matches the intended object list.
# e) Visual observability:
#    The rule is visible without hidden metadata: the same scattered input objects now
#    produce a tall single column where the category boundary is horizontal. Because both
#    categories are always present in generated samples, the D5 change is observable.
# f) Visual cleanliness:
#    Input objects are sparse and non-overlapping; output is a clean 4-wide stack with no
#    overdraw or ambiguous layering.
# g) Color inference:
#    No color remapping is introduced. Colors only help distinguish separate objects; the
#    operative cue is shape type (frame vs solid), which is directly visible.
# h) Generator correctness strategy:
#    Build the input by scattering known objects, compute the expected variant output from
#    the recorded placements, and assert np.array_equal(main(input), expected). Also assert
#    that expected differs from the old two-column assembly on the same sorted objects, so
#    samples do not silently collapse back to the original layout.
# i) New main() assumptions introduced by the variant:
#    main() now assumes the output is allowed to be taller and only 4 cells wide. The
#    generator guarantees at least one frame and one solid, so both top and bottom segments
#    exist and the vertical category split is visible in produced examples.


def _classify_objects(grid: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
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

        if len(border_colors) == 1 and bool((sub[1:3, 1:3] == bg).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, frames, solids


def _assemble_variant_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    out_h = 4 * (len(frames) + len(solids))
    out = np.full((out_h, 4), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    solid_base = 4 * len(frames)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(solid_base + i * 4), 0)
    return out


def _assemble_original_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares by type, then stack them vertically in one column.

    Frames are sorted top-to-bottom and placed first.
    Solids are sorted top-to-bottom and placed immediately below the frame segment.
    """
    bg, frames, solids = _classify_objects(np.array(grid, dtype=int))
    return _assemble_variant_output(bg, frames, solids)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), col, dtype=int)
    a[1:3, 1:3] = bg
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    try:
        used_cols: set[int] = set()
        for _ in range(n_frames):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            sub = _make_frame(col, bg)
            r0, c0 = random_free_location(g, sub, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, sub, r0, c0, transparent=bg)
            frames.append((int(r0), int(c0), sub))

        for _ in range(n_solids):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            sub = _make_solid(col)
            r0, c0 = random_free_location(g, sub, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, sub, r0, c0, transparent=bg)
            solids.append((int(r0), int(c0), sub))

        frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _assemble_variant_output(bg, frames, solids)
        original = _assemble_original_output(bg, frames, solids)
        if expected.shape == original.shape and np.array_equal(expected, original):
            return None
        if not np.array_equal(main(g), expected):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
