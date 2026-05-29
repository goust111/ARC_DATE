# Variant of 2ba387bc
# Dimension: D14 — 堆叠方向与对齐方式 (dst)
# Change: 两列均从顶部向下堆叠，每个方块占4行，左列从col 0开始，右列从col 4开始
#   → 水平方向堆叠——框架排成上行，实心排成下行

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for the D14 variant
# a) Concrete behavior and trace
#    - Input stays the same as the source task: several disconnected 4x4 objects on a plain background.
#      A frame has one border color and a bg-colored 2x2 interior; a solid is filled 4x4.
#    - Original output placed frames in a left vertical column and solids in a right vertical column.
#      Example with frames F1,F2 and solids S1,S2, sorted by input row:
#        original: rows 0-3 = F1 | S1, rows 4-7 = F2 | S2
#        variant : top row  = F1 F2, bottom row = S1 S2
#      The classification and ordering stage is unchanged; only the final placement coordinates and
#      canvas aspect ratio change.
# b) Complexity calibration
#    - The task still has exactly two visually distinct classes, so the rule stays legible.
#    - Generator rejects the smallest 1-frame + 1-solid case to avoid under-supported samples.
#      This keeps the horizontal ordering visible in most generated examples without adding new rules.
# c) Implicit-assumption review of the original main()
#    - `bbox_h(b) != 4 or bbox_w(b) != 4`: assumes every relevant object is exactly 4x4. Still valid.
#    - `len(bcols) == 1 and interior == bg`: assumes frames are recognizable by empty interior only.
#      Still valid because D14 changes layout, not object appearance.
#    - sorting by `(row, col)`: assumes output order follows input top-to-bottom. Still valid.
#    - `outH = 4 * max(...)`, `outW = 8`, and vertical `paste(..., i * 4, fixed_col)` encode the old
#      destination axis. These are the lines that must change for D14.
# d) Placement legality
#    - Input generator uses `random_free_location(..., padding=1)` so objects do not overlap or touch.
#    - Unique non-bg colors prevent any accidental merging across objects of different classes.
#    - After generation, `main(input)` is executed as a self-check before the sample is accepted.
# e) Visual observability
#    - The new rule is visible from shape alone: outputs become two 4-pixel-high bands instead of two
#      4-pixel-wide columns. No hidden dependence on numeric color ids is introduced.
#    - Having at least one group with size >= 2 makes left-to-right ordering readable across examples.
# f) Visual cleanliness
#    - Objects remain isolated in input, and output has clean 4x4 tiling with no overlap.
#    - Empty cells only appear as right-side padding when one class has fewer objects than the other.
# g) Color reasoning
#    - Colors are preserved exactly; they are identifiers for individual objects, not rule controls.
# 9) New assumptions introduced by the variant solver
#    - The solver now assumes the output always has height 8 and width `4 * max(n_frames, n_solids)`.
#      The generator guarantees at least one object of each class, so width is always >= 8 in samples.
#    - The paste loops assume each extracted object remains 4x4; generator construction enforces this.


def _extract_objects(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    """Return frames and solids sorted by input position."""
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
        is_frame = len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())

        if is_frame:
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects by type, then stack them horizontally by row.

    Variant rule:
    - Frames go in the top 4 rows from left to right.
    - Solids go in the bottom 4 rows from left to right.
    - Width is 4 * max(#frames, #solids); shorter row is bg-padded on the right.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _extract_objects(g, bg)

    out_h = 8
    out_w = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, 0, int(i * 4))
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, 4, int(i * 4))

    return out


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), col, dtype=int)
    a[1:3, 1:3] = bg
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


def _draw_random_object(g: np.ndarray, patch: np.ndarray, bg: int) -> None:
    r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
    paste(g, patch, r0, c0, transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    if n_frames == 1 and n_solids == 1:
        return None

    try:
        used_cols: set[int] = set()

        for _ in range(n_frames):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            _draw_random_object(g, _make_frame(col, bg), bg)

        for _ in range(n_solids):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            _draw_random_object(g, _make_solid(col), bg)

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        candidate = _gen_attempt()
        if candidate is None:
            continue

        bg = int(get_bg(candidate))
        frames, solids = _extract_objects(candidate, bg)
        if not frames or not solids:
            continue

        out = main(candidate)
        expected_shape = (8, 4 * max(len(frames), len(solids)))
        if out.shape != expected_shape:
            continue

        return candidate

    raise RuntimeError("generate_input failed after 300 attempts")
