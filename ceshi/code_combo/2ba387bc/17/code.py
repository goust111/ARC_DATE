# Combo variant of 2ba387bc (index 17)
# combo_spec: D5/3+D7/2
# Name: 重力堆叠+跨列颜色互换

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "solvers" / "arc_utils.py").exists()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate project root containing solvers/arc_utils.py")
SOLVER_DIR = PROJECT_ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D5/3 + D7/2
=====================================

a) Concrete behavior and trace
   - Input remains the seed task's scene: several disjoint 4x4 monochrome objects on a
     uniform background. Frames are hollow 4x4 rings; solids are filled 4x4 blocks.
   - The seed task did three things: classify by shape, sort each type by input row,
     then stack frames top-down on the left and solids top-down on the right, keeping
     each object's original color.
   - This combo changes two stages at once:
       1. Gravity layout: frames still go to the left column but stack from bottom to
          top; solids still go to the right column and stack from top to bottom.
       2. Cross-column color swap: after per-type sorting, the i-th frame and i-th solid
          exchange colors. Their shapes do not change; only their colors do.
   - Example with row-sorted objects:
       frames = [F1(red), F2(green), F3(yellow)]
       solids = [S1(blue), S2(maroon), S3(teal)]
     Seed output:
       left top->bottom  = [F1(red), F2(green), F3(yellow)]
       right top->bottom = [S1(blue), S2(maroon), S3(teal)]
     Combo output:
       paired color swap -> frames use [blue, maroon, teal], solids use [red, green, yellow]
       left bottom-up placement means top->bottom left becomes [F3(teal), F2(maroon), F1(blue)]
       right top-down placement stays               [S1(red), S2(green), S3(yellow)]
     The solver must therefore track both a position reversal on the left and a color
     transfer across columns by sorted index.
   - If the counts were unequal, only the first min(#frames, #solids) pairs could swap
     and leftovers would keep their original colors. The solver supports that general
     rule, but the generator uses equal counts so every object participates visibly.

b) Complexity calibration
   - Cross-column color swapping is hard to infer if only one pair exists or if some
     objects keep the same color. The generator therefore creates 2-4 frames and the
     same number of solids, with unique colors across all objects.
   - Bottom gravity is still visible with equal counts because the left-column order is
     reversed relative to the frame order in the input. Using at least two frames avoids
     degenerate cases where top-down and bottom-up placement look identical.
   - Equal counts also keep the color-swap relation low-entropy: every frame has exactly
     one partner solid, so a human can infer "same ordinal within each type swaps color"
     rather than having to guess what happens to leftovers.

c) Hidden-assumption audit for main()
   - `bg = get_bg(g)`: still valid; the combo does not change background detection.
   - `find_components(... monochromatic=False)`: still valid because each 4x4 object is
     a single 4-connected component. The generator enforces gaps so components do not
     merge.
   - `bbox_h/bbox_w == 4`: still valid because object geometry is unchanged.
   - Frame test = "uniform border + bg 2x2 interior": still valid because the combo
     changes only output layout/color, not input recognition.
   - Original code assumed `paste(out, sub, i*4, left_or_right)` was enough. That old
     line relied on two now-false assumptions:
       * frame output position equals top-down order;
       * output color equals the object's own input color.
     The combo requires reconstructing each output patch from two independent decisions:
       * destination row from gravity;
       * paint color after the paired swap.
   - New pairing logic assumes the per-type sorted lists are the canonical order for
     both stacking and swapping. The generator guarantees that within each type the input
     top rows are distinct, so the ordinal pairing is unambiguous.

d) Placement legality
   - Generator placement still uses `random_free_location(... border=1, padding=1,
     padding_connectivity=4)`, so every 4x4 object stays in bounds, does not overlap,
     and does not touch another object orthogonally. This preserves clean component
     extraction under 4-connectivity.
   - Additional retries enforce distinct top rows within frames and within solids, which
     preserves a visible ordering signal for the per-type sort.

e) Visual observability
   - A human can see two independent regularities in 2-4 examples:
       * left-column frames appear in reverse vertical order compared with their input
         frame order;
       * colors jump across columns between same-ordinal frame/solid pairs.
   - Colors are never used as hidden keys for sorting; they only make the swap legible.
     Unique colors across all objects ensure every transferred color is visible.

f) Visual cleanliness
   - The input remains sparse and uncluttered: one object per 4x4 footprint, background
     gaps everywhere else.
   - The output remains the seed task's tidy 8-column format, with only one frame or one
     solid occupying each 4x4 slot. No overlays, markers, or extra decoration are added.

g) Color-change inferability
   - The color rule has a small and structured hypothesis space: "same-ranked frame and
     solid exchange colors". Because counts are equal and all colors are unique, every
     object provides evidence for one swap relation and there are no ambiguous untouched
     leftovers in training-style examples.
   - Equal counts make the effect especially readable: the left column shows the swapped
     solid colors in reverse order, while the right column shows the swapped frame colors
     in forward order.

h) Generator-side anti-degeneration checks
   - We reject any sample where the fused output equals:
       * the seed task output;
       * gravity-only output (D5/3 without color swap);
       * color-swap-only output (D7/2 without gravity).
   - We also compute two quantitative fusion metrics:
       * `swapped_pairs`: number of ordinal frame/solid pairs whose colors actually
         changed on both sides;
       * `moved_frames`: number of frames whose destination row differs from legacy
         top-down stacking.
     The generator requires `swapped_pairs >= 2` and `moved_frames >= 2`.

i) New assumptions introduced by the combo solver
   - `_object_color()` assumes each valid 4x4 object is monochrome on all non-bg cells.
     That was already true in the seed task and is guaranteed by the generator.
   - `_swap_colors_by_index()` assumes the meaningful correspondence is the sorted ordinal
     within each type, not physical row alignment after gravity. This is the intended new
     combo rule and is exactly what the generator evidences.
"""


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    return len(set(map(int, border.flatten()))) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _object_color(sub: np.ndarray, bg: int) -> int:
    vals = sub[sub != int(bg)]
    if vals.size == 0:
        raise ValueError("4x4 object has no non-background color")
    return int(vals.flat[0])


def _extract_objects(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, int]] = []
    solids: list[tuple[int, int, int]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        color = _object_color(sub, bg)
        if _is_frame(sub, bg):
            frames.append((r0, c0, color))
        else:
            solids.append((r0, c0, color))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, frames, solids


def _swap_colors_by_index(
    frames: list[tuple[int, int, int]],
    solids: list[tuple[int, int, int]],
) -> tuple[list[int], list[int]]:
    frame_colors = [int(col) for _r0, _c0, col in frames]
    solid_colors = [int(col) for _r0, _c0, col in solids]
    pair_count = min(len(frame_colors), len(solid_colors))

    for i in range(pair_count):
        frame_colors[i], solid_colors[i] = solid_colors[i], frame_colors[i]

    return frame_colors, solid_colors


def _build_output(
    bg: int,
    frames: list[tuple[int, int, int]],
    solids: list[tuple[int, int, int]],
    *,
    frame_from_bottom: bool,
    swap_colors: bool,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    if swap_colors:
        frame_colors, solid_colors = _swap_colors_by_index(frames, solids)
    else:
        frame_colors = [int(col) for _r0, _c0, col in frames]
        solid_colors = [int(col) for _r0, _c0, col in solids]

    for i, color in enumerate(frame_colors):
        dest_r = int(out_h - 4 * (i + 1)) if frame_from_bottom else int(i * 4)
        paste(out, _make_frame(int(color), bg), dest_r, 0)

    for i, color in enumerate(solid_colors):
        paste(out, _make_solid(int(color)), int(i * 4), 4)

    return out


def _fusion_metrics(
    frames: list[tuple[int, int, int]],
    solids: list[tuple[int, int, int]],
) -> tuple[int, int]:
    frame_colors, solid_colors = _swap_colors_by_index(frames, solids)
    pair_count = min(len(frames), len(solids))
    swapped_pairs = 0
    for i in range(pair_count):
        if frame_colors[i] != int(frames[i][2]) and solid_colors[i] != int(solids[i][2]):
            swapped_pairs += 1

    out_h = 4 * max(int(len(frames)), int(len(solids)))
    moved_frames = 0
    for i in range(len(frames)):
        legacy_r = int(i * 4)
        gravity_r = int(out_h - 4 * (i + 1))
        if legacy_r != gravity_r:
            moved_frames += 1

    return swapped_pairs, moved_frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects by type, then apply fused gravity + color-swap logic.

    Combo rule:
    - Frames go to cols 0..3 and stack from bottom upward.
    - Solids go to cols 4..7 and stack from top downward.
    - After sorting each type by input row, the i-th frame and i-th solid swap colors.
    """
    bg, frames, solids = _extract_objects(grid)
    return _build_output(bg, frames, solids, frame_from_bottom=True, swap_colors=True)


def _pick_count() -> int:
    return int(np.random.choice(np.array([2, 3, 4], dtype=int)))


def _place_with_distinct_row(
    grid: np.ndarray,
    patch: np.ndarray,
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
    raise ValueError("failed to place object with a distinct row")


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    g = np.full((h, w), bg, dtype=int)
    n = _pick_count()

    try:
        colors = list(map(int, np.random.permutation(np.arange(1, 10))[: 2 * n]))
        if len(colors) != 2 * n:
            return None

        frame_rows: set[int] = set()
        solid_rows: set[int] = set()

        for i in range(n):
            patch = _make_frame(colors[i], bg)
            r0, c0 = _place_with_distinct_row(g, patch, bg, frame_rows)
            paste(g, patch, r0, c0, transparent=bg)
            frame_rows.add(int(r0))

        for i in range(n):
            patch = _make_solid(colors[n + i])
            r0, c0 = _place_with_distinct_row(g, patch, bg, solid_rows)
            paste(g, patch, r0, c0, transparent=bg)
            solid_rows.add(int(r0))

        parsed_bg, frames, solids = _extract_objects(g)
        if len(frames) != n or len(solids) != n:
            raise ValueError("component extraction mismatch")

        fused = _build_output(parsed_bg, frames, solids, frame_from_bottom=True, swap_colors=True)
        seed = _build_output(parsed_bg, frames, solids, frame_from_bottom=False, swap_colors=False)
        gravity_only = _build_output(parsed_bg, frames, solids, frame_from_bottom=True, swap_colors=False)
        swap_only = _build_output(parsed_bg, frames, solids, frame_from_bottom=False, swap_colors=True)

        swapped_pairs, moved_frames = _fusion_metrics(frames, solids)
        if swapped_pairs < 2 or moved_frames < 2:
            raise ValueError("fusion metrics too weak")
        if np.array_equal(fused, seed):
            raise ValueError("combo degenerated to seed task")
        if np.array_equal(fused, gravity_only):
            raise ValueError("color-swap effect missing")
        if np.array_equal(fused, swap_only):
            raise ValueError("gravity effect missing")
        if not np.array_equal(main(g), fused):
            raise ValueError("round-trip check failed")
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
