# Combo variant of 2ba387bc (index 51)
# combo_spec: D5/1+D7/0+D14/0+D16/0
# Name: 框架重着色+旋转+反向堆叠+分隔列

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = PROJECT_ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


FRAME_OUT_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


"""
Design analysis for combo D5/1 + D7/0 + D14/0 + D16/0
======================================================

1. Concrete behavior and traced example
   - Input stays in the original 2ba387bc regime: several disconnected 4x4 objects on a
     plain background. A frame is a 4x4 one-pixel ring with a bg-colored 2x2 hole; a
     solid is a fully filled 4x4 monochrome block.
   - Classification is unchanged: detect all 4x4 components, split them into frames vs
     solids, and sort each type by input `(top_row, left_col)`.
   - The combo changes only the output assembly stage, but it changes it in four coupled ways:
       * D5/1: output width becomes 9 instead of 8, because col 4 is a pure background
         separator between the two type columns.
       * D7/0: every frame is recolored to a single canonical output color
         `FRAME_OUT_COLOR`, while solids keep their original colors.
       * D14/0: left column still stacks top-down, but the right column stacks bottom-up.
       * D16/0: each frame is rotated 90 degrees before recoloring and placing.
   - Example with frames F1,F2 and solids S1,S2,S3 after sorting by input row:
       original output:
         rows  0..3  -> F1 at cols 0..3, S1 at cols 4..7
         rows  4..7  -> F2 at cols 0..3, S2 at cols 4..7
         rows  8..11 -> bg left,          S3 at cols 4..7
       combo output:
         rows  0..3  -> rot(recolor(F1)) at cols 0..3, separator at col 4, bg on right
         rows  4..7  -> rot(recolor(F2)) at cols 0..3, separator at col 4, S3 on right
         rows  8..11 -> bg left,                    separator at col 4, S2 on right
         rows 12..15 -> bg left,                    separator at col 4, S1 on right
     So the right column visibly reverses vertical order and gravitates toward the bottom.

2. Complexity calibration
   - The frame recolor is made observable by fixing one canonical frame output color and
     banning that color from all input objects.
   - The reverse right-stack is made observable by requiring at least two solids and using
     unique solid colors; otherwise bottom-up placement could be mistaken for ordinary
     top-down stacking.
   - The separator is kept to exactly one column, which is visually clear but not noisy.
   - D16/0 needs special scrutiny here: a standard 4x4 hollow square is invariant under
     90-degree rotation, so the rotation is a real solver-side operation but it does not
     create a new pixel pattern on this source task. We preserve the operation faithfully
     instead of adding artificial asymmetry that would break the original object definition.

3. Hidden-assumption audit of the original `main()`
   - `bg = get_bg(g)`: still valid. Background inference is unchanged.
   - `find_components(... monochromatic=False)`: still valid if every object stays an
     isolated connected component. Generator keeps 1-cell padding to enforce this.
   - `bbox_h(b) != 4 or bbox_w(b) != 4`: still valid because the input object geometry
     remains the original 4x4 shape family.
   - Frame test = "uniform border + bg-filled 2x2 center": still valid. The combo does not
     alter how frames are recognized, only how they are rendered in the output.
   - Original `outW = 8` is invalidated by D5/1 and must become 9.
   - Original frame paste `paste(out, sub, i * 4, 0)` is invalidated by D7/0 and D16/0:
     frames must be rotated and recolored before pasting.
   - Original solid paste `paste(out, sub, i * 4, 4)` is invalidated by both D5/1 and
     D14/0: solids move to cols 5..8 and their destination rows must be counted from the
     bottom of the canvas instead of the top.

4. Placement legality
   - Every input object is still exactly 4x4.
   - `random_free_location(... border=1, padding=1, padding_connectivity=4)` guarantees
     that objects stay in bounds, do not overlap, and do not merge under 4-connectivity.
   - The generator re-parses the finished grid with the same extraction logic and rejects
     any sample where the recovered counts disagree with the intended counts.

5. Visual observability and cleanliness
   - A human can infer the rule from a few examples because three visible signatures repeat:
       * all left-column frames become the same color;
       * there is always a blank separator column between groups;
       * the right column's order is reversed vertically and anchored to the bottom.
   - The scene remains clean because output contains only two 4-column bands plus one blank
     separator. No extra markers or overlays are introduced.

6. Color-change inferability
   - The recolor is structural and low-entropy: "if an object is a frame, its output color
     becomes 5". Solids preserve their own colors, so the mapping is easy to isolate.
   - Keeping input frame colors distinct from 5 avoids ambiguous cases where recoloring
     appears to do nothing.

7. Generator anti-degeneration strategy
   - Reject any sample where the combo output equals the original 8-wide top-down output.
   - Reject any sample where the combo output equals a weaker variant that keeps the new
     separator and frame recolor but stacks solids top-down; this ensures the reverse stack
     is actually doing visible work.
   - Reject any sample where the combo output equals a weaker variant that keeps the new
     separator and reverse stack but skips frame recoloring; this ensures recolor is visible.
   - D16/0 cannot be made separately visible without violating the source task's symmetric
     frame definition, so the solver executes the rotation explicitly and the generator
     validates the other three observable controls.

8. New assumptions introduced by this combo solver
   - The solver assumes the canonical frame output color differs from the background.
     Generator fixes `bg = 0`, so `FRAME_OUT_COLOR = 5` is safe.
   - The solver assumes both object classes still arise from the original 4x4 geometry
     rule. Generator only emits those shapes.
   - The solver assumes there is at least one frame and one solid so both columns exist.
     Generator guarantees at least two of each to make the asymmetric stacking legible.
"""


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    return len(set(map(int, border.flatten()))) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _collect_objects(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        if _is_frame(sub, bg):
            frames.append((int(r0), int(c0), np.array(sub, dtype=int, copy=True)))
        else:
            solids.append((int(r0), int(c0), np.array(sub, dtype=int, copy=True)))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _transform_frame(sub: np.ndarray, bg: int) -> np.ndarray:
    """
    Apply the combo frame transform: rotate 90 degrees, then recolor every non-bg pixel.

    For this task's canonical 4x4 hollow square, the rotation is pixel-invariant because the
    shape has 90-degree symmetry, but the operation is still executed explicitly.
    """
    patch = np.rot90(np.array(sub, dtype=int, copy=True), k=1)
    patch[patch != int(bg)] = int(FRAME_OUT_COLOR)
    return patch


def _build_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
    *,
    recolor_frames: bool,
    reverse_solids: bool,
    separator: bool,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out_w = 9 if separator else 8
    solid_col = 5 if separator else 4
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        patch = _transform_frame(sub, bg) if recolor_frames else np.rot90(np.array(sub, dtype=int, copy=True), k=1)
        paste(out, patch, int(i * 4), 0, transparent=bg)

    for i, (_r0, _c0, sub) in enumerate(solids):
        dest_r = int(out_h - 4 * (i + 1)) if reverse_solids else int(i * 4)
        paste(out, sub, dest_r, solid_col)

    return out


def _build_original_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects into two separated columns by type.

    Combo rule:
    - Frames go in the left 4 columns, after a 90-degree rotation and recolor to one
      canonical output color.
    - Solids keep their original colors and go in the right 4 columns.
    - Col 4 is a blank separator column.
    - Frames stack from top to bottom; solids stack from bottom to top.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    return _build_output(frames, solids, bg, recolor_frames=True, reverse_solids=True, separator=True)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _pick_counts() -> tuple[int, int]:
    options = [
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 4),
        (4, 2),
        (4, 3),
        (2, 2),
        (3, 3),
        (4, 4),
    ]
    idx = int(np.random.randint(0, len(options)))
    return options[idx]


def _place_with_distinct_row(
    grid: np.ndarray,
    patch: np.ndarray,
    bg: int,
    used_rows: set[int],
) -> tuple[int, int]:
    for _ in range(100):
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
    raise ValueError("failed to find a distinct row for object placement")


def _frame_pixels_all_canonical(out: np.ndarray, n_frames: int, bg: int) -> bool:
    if n_frames <= 0:
        return False
    left = out[: 4 * int(n_frames), :4]
    fg = left[left != int(bg)]
    return fg.size > 0 and set(map(int, fg.tolist())) == {int(FRAME_OUT_COLOR)}


def _right_column_top_color(out: np.ndarray, bg: int) -> int | None:
    right = out[:, 5:9]
    nz = np.argwhere(right != int(bg))
    if nz.size == 0:
        return None
    rr, cc = map(int, nz[0])
    return int(right[rr, cc])


def _right_column_bottom_color(out: np.ndarray, bg: int) -> int | None:
    right = out[:, 5:9]
    nz = np.argwhere(right != int(bg))
    if nz.size == 0:
        return None
    rr, cc = map(int, nz[-1])
    return int(right[rr, cc])


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 33))
    w = int(np.random.randint(22, 33))
    g = np.full((h, w), int(bg), dtype=int)

    n_frames, n_solids = _pick_counts()
    total = int(n_frames + n_solids)
    if total > len(OBJECT_COLORS):
        return None

    colors = list(np.random.permutation(np.array(OBJECT_COLORS, dtype=int))[:total])

    frame_rows: set[int] = set()
    solid_rows: set[int] = set()

    try:
        for i in range(n_frames):
            patch = _make_frame(int(colors[i]), bg)
            r0, c0 = _place_with_distinct_row(g, patch, bg, frame_rows)
            paste(g, patch, r0, c0, transparent=bg)
            frame_rows.add(int(r0))

        for i in range(n_solids):
            patch = _make_solid(int(colors[n_frames + i]))
            r0, c0 = _place_with_distinct_row(g, patch, bg, solid_rows)
            paste(g, patch, r0, c0, transparent=bg)
            solid_rows.add(int(r0))

        frames, solids = _collect_objects(g, bg)
        if len(frames) != n_frames or len(solids) != n_solids:
            return None

        combo = _build_output(frames, solids, bg, recolor_frames=True, reverse_solids=True, separator=True)
        legacy = _build_original_output(frames, solids, bg)
        weak_no_reverse = _build_output(frames, solids, bg, recolor_frames=True, reverse_solids=False, separator=True)
        weak_no_recolor = _build_output(frames, solids, bg, recolor_frames=False, reverse_solids=True, separator=True)
        actual = main(g)

        if not np.array_equal(actual, combo):
            return None
        if np.array_equal(combo, legacy):
            return None
        if np.array_equal(combo, weak_no_reverse):
            return None
        if np.array_equal(combo, weak_no_recolor):
            return None
        if combo.shape != (4 * max(n_frames, n_solids), 9):
            return None
        if not bool((combo[:, 4] == int(bg)).all()):
            return None
        if not _frame_pixels_all_canonical(combo, n_frames, bg):
            return None
        if _right_column_top_color(combo, bg) != int(solids[-1][2][0, 0]):
            return None
        if _right_column_bottom_color(combo, bg) != int(solids[0][2][0, 0]):
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
