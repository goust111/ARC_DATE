# Combo variant of 2ba387bc (index 45)
# combo_spec: D4/0+D7/0+D12/2
# Name: 彩色内部框架+框架重着色+不同输出背景

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/lixinyu/snap/new")
SOLVER_DIR = PROJECT_ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


FRAME_OUT_COLOR = 5
OUTPUT_BG = 0
INPUT_BGS = (2, 4, 8)


"""
Design analysis for combo D4/0 + D7/0 + D12/2
==============================================

1. Concrete behavior and trace
   - Original task: each 4x4 object is either
       * frame = monochrome border + background-filled 2x2 center
       * solid = full 4x4 monochrome block
     The solver detects frames by `center == bg`, sorts both groups by `(row, col)`,
     then stacks frames in the left 4 columns and solids in the right 4 columns, while
     preserving all colors and reusing the input background in the output.
   - Fused variant input:
       * frame = border color `b`, center color `i`, both non-background, and `b != i`
       * solid = full 4x4 monochrome color `s`
     So a frame is no longer a "hole on background"; it is a fully filled two-color tile.
   - Fused variant output:
       * frames are still stacked on the left, solids on the right
       * every frame is rebuilt as a canonical hollow frame whose border is fixed
         `FRAME_OUT_COLOR`
       * the output background is fixed `OUTPUT_BG`, different from the input background
       * therefore each frame center also becomes `OUTPUT_BG`
   - Small trace:
       input bg = 8
       input frame A =
         3 3 3 3
         3 7 7 3
         3 7 7 3
         3 3 3 3
       input solid B = all 7
       Original solver would misread frame A as a solid because its center is not bg.
       D4-only logic would classify A as a frame but keep its colors.
       Fused logic classifies A from the border/center relation, then outputs
         5 5 5 5
         5 0 0 5
         5 0 0 5
         5 5 5 5
       while solid B keeps color 7. This makes all three changes visible at once:
       colored-core input frames, frame recolor, and new output background.

2. Complexity calibration
   - The generator always creates at least 2 frames so the "many different frame colors
     all collapse to one canonical output color" pattern is visible.
   - Input object colors exclude both `OUTPUT_BG` and `FRAME_OUT_COLOR`, so the D7 and D12
     changes cannot visually degenerate into "nothing happened".
   - At least one solid reuses a color that also appears in a frame border or frame center.
     This prevents a trivial color-only shortcut such as "some colors always mean solid".
   - Input backgrounds are drawn from a small repeated set `{2,4,8}` while output
     background is always `0`. This keeps the background change easy to infer from 2-4
     examples without introducing a per-sample arbitrary mapping.

3. Hidden-assumption audit of the original `main()`
   - `bg = get_bg(g)`: still valid, but now only for interpreting the input scene.
   - `find_components(... monochromatic=False)`: still valid. A colored-core frame is a
     single 4-connected 4x4 component because its center is filled, not empty.
   - `bbox_h(b) == bbox_w(b) == 4`: preserved; generator still places only 4x4 objects.
   - Original frame test:
       `len(border_colors) == 1 and sub[1:3,1:3] == bg`
     is invalid after D4/0 because frame centers are no longer background.
   - Original frame pasting:
       `paste(out, sub, ...)`
     is invalid after D7/0 and D12/2 because output frames must no longer preserve input
     colors, and their centers must no longer preserve the input center color.
   - Original output initialization:
       `out = np.full(..., bg)`
     is invalid after D12/2 because output background is fixed to `OUTPUT_BG`, not input bg.
   - Combined effect: the solver must classify frames using border/center color relation,
     then reconstruct canonical output frames instead of reusing the raw input crops.

4. Placement legality
   - Generator places every object as one explicit 4x4 patch.
   - `random_free_location(... border=1, padding=1, padding_connectivity=4)` guarantees
     no overlap, no edge-touching, and no out-of-bounds placement.
   - Because objects stay disconnected under 4-connectivity, `find_components` returns the
     intended per-object components reliably.

5. Visual observability
   - A human sees scattered two-color frame tiles and single-color solids in the input.
   - In the output, all left-column frames become the same hollow grey frame on black,
     while right-column solids keep their own colors.
   - The rule is inferable from visible structure only: border-vs-center relation,
     left/right grouping, and consistent recolor/background change. No step depends on
     numeric color ordering or any hidden metadata.

6. Visual cleanliness
   - Only one object layer exists per location; there are no overlays or markers.
   - The scene remains sparse and separated by at least one cell, so the 4x4 objects are
     easy to parse as individual units.
   - Output layout remains the original clean 2-column stack, which keeps the fusion easy
     to compare against the base task.

7. Color-change inferability
   - The color logic is low-entropy and structural:
       * input frame recognition depends on "center color differs from border color"
       * output frame border is always `FRAME_OUT_COLOR`
       * output background is always `OUTPUT_BG`
       * solids preserve their color
   - Since multiple different input frame colors map to the same output frame color, the
     solver must infer a type-conditioned normalization, not a global palette swap.

8. Non-degeneracy metric used for validation
   - Metric A: `frame_delta_pixels` = Hamming distance between
       (a) stacking the detected frames in their original colored-core appearance and
       (b) stacking them in the fused canonical appearance.
     Every valid frame contributes 16 changed pixels, so this must be `16 * n_frames > 0`.
   - Metric B: fused output must differ from the original task's interpretation on the
     same input. Because colored-core frames are never background-centered, the original
     task would route them incorrectly. The generator rejects any accidental equality.

9. New assumptions introduced by the fused solver and generator guarantees
   - Assumption: a frame has a monochrome border and a monochrome 2x2 center with
     different colors. Generator constructs frames exactly that way.
   - Assumption: solids are full monochrome 4x4 blocks. Generator constructs solids that way.
   - Assumption: `FRAME_OUT_COLOR` and `OUTPUT_BG` remain visually distinct from any input
     object color. Generator excludes them from the input palette.
   - Assumption: at least one frame exists, otherwise recolor/background-on-frame behavior
     would be invisible. Generator uses 2-4 frames.
   - Assumption: output background differs from input background. Generator samples input bg
     only from `INPUT_BGS`, which excludes `OUTPUT_BG`.
"""


def _frame_colors(sub: np.ndarray) -> tuple[int, int] | None:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    inner = sub[1:3, 1:3]
    border_colors = set(map(int, border.flatten()))
    inner_colors = set(map(int, inner.flatten()))
    if len(border_colors) != 1 or len(inner_colors) != 1:
        return None
    border_col = int(next(iter(border_colors)))
    inner_col = int(next(iter(inner_colors)))
    if border_col == inner_col:
        return None
    return border_col, inner_col


def _make_input_frame(border_col: int, inner_col: int) -> np.ndarray:
    tile = np.full((4, 4), int(border_col), dtype=int)
    tile[1:3, 1:3] = int(inner_col)
    return tile


def _make_output_frame() -> np.ndarray:
    tile = np.full((4, 4), int(FRAME_OUT_COLOR), dtype=int)
    tile[1:3, 1:3] = int(OUTPUT_BG)
    return tile


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _collect_objects(
    g: np.ndarray,
    input_bg: int,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    comps = find_components(g, bg=input_bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, int, int]] = []
    solids: list[tuple[int, int, int]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        colors = _frame_colors(sub)
        if colors is not None:
            border_col, inner_col = colors
            frames.append((int(r0), int(c0), int(border_col), int(inner_col)))
        else:
            solids.append((int(r0), int(c0), int(sub[0, 0])))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _assemble_output(
    frames: list[tuple[int, int, int, int]],
    solids: list[tuple[int, int, int]],
    *,
    output_bg: int,
    canonicalize_frames: bool,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(output_bg), dtype=int)

    for i, (_r0, _c0, border_col, inner_col) in enumerate(frames):
        patch = _make_output_frame() if canonicalize_frames else _make_input_frame(border_col, inner_col)
        paste(out, patch, int(i * 4), 0, transparent=int(output_bg))

    for i, (_r0, _c0, col) in enumerate(solids):
        paste(out, _make_solid(col), int(i * 4), 4)

    return out


def _original_solver_output_like(
    g: np.ndarray,
    input_bg: int,
) -> np.ndarray:
    comps = find_components(g, bg=input_bg, connectivity=4, monochromatic=False, return_format="dict")
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
        if len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(input_bg)).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(input_bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def _frame_delta_pixels(frames: list[tuple[int, int, int, int]]) -> int:
    if not frames:
        return 0
    baseline = np.full((4 * len(frames), 4), -1, dtype=int)
    fused = np.full((4 * len(frames), 4), -1, dtype=int)
    for i, (_r0, _c0, border_col, inner_col) in enumerate(frames):
        paste(baseline, _make_input_frame(border_col, inner_col), int(i * 4), 0)
        paste(fused, _make_output_frame(), int(i * 4), 0)
    return int(np.count_nonzero(baseline != fused))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Detect 4x4 two-color colored-core frames and 4x4 monochrome solids.
    - Sort each class by input position.
    - Stack frames in the left column and solids in the right column.
    - Rebuild every frame as the same canonical hollow frame color.
    - Use a fixed output background different from the input background.
    """
    g = np.array(grid, dtype=int)
    input_bg = int(get_bg(g))
    frames, solids = _collect_objects(g, input_bg)
    return _assemble_output(frames, solids, output_bg=OUTPUT_BG, canonicalize_frames=True)


def _rand_color(exclude: set[int]) -> int:
    choices = [c for c in range(1, 10) if c not in set(map(int, exclude))]
    if not choices:
        raise ValueError("no colors available")
    return int(np.random.choice(np.array(choices, dtype=int)))


def _gen_attempt() -> np.ndarray | None:
    input_bg = int(np.random.choice(np.array(INPUT_BGS, dtype=int)))
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    g = np.full((h, w), int(input_bg), dtype=int)

    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(1, 5))

    usable = [c for c in range(1, 10) if c not in {int(input_bg), int(OUTPUT_BG), int(FRAME_OUT_COLOR)}]
    if len(usable) < 3:
        return None

    try:
        shared_color = int(np.random.choice(np.array(usable, dtype=int)))
        border_seed = shared_color
        inner_seed = _rand_color({input_bg, OUTPUT_BG, FRAME_OUT_COLOR, border_seed})

        frame_specs: list[tuple[int, int]] = []
        frame_specs.append((int(border_seed), int(inner_seed)))

        second_border = _rand_color({input_bg, OUTPUT_BG, FRAME_OUT_COLOR, shared_color})
        frame_specs.append((int(second_border), int(shared_color)))

        while len(frame_specs) < n_frames:
            border_col = _rand_color({input_bg, OUTPUT_BG, FRAME_OUT_COLOR})
            inner_col = _rand_color({input_bg, OUTPUT_BG, FRAME_OUT_COLOR, border_col})
            frame_specs.append((int(border_col), int(inner_col)))

        solid_colors = [int(shared_color)]
        while len(solid_colors) < n_solids:
            solid_colors.append(_rand_color({input_bg, OUTPUT_BG, FRAME_OUT_COLOR}))

        placed_frames: list[tuple[int, int, int, int]] = []
        placed_solids: list[tuple[int, int, int]] = []

        for border_col, inner_col in frame_specs:
            patch = _make_input_frame(border_col, inner_col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=int(input_bg),
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0)
            placed_frames.append((int(r0), int(c0), int(border_col), int(inner_col)))

        for col in solid_colors:
            patch = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=int(input_bg),
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0)
            placed_solids.append((int(r0), int(c0), int(col)))

        expected = _assemble_output(
            sorted(placed_frames, key=lambda t: (int(t[0]), int(t[1]))),
            sorted(placed_solids, key=lambda t: (int(t[0]), int(t[1]))),
            output_bg=OUTPUT_BG,
            canonicalize_frames=True,
        )
        actual = main(g)
        if not np.array_equal(actual, expected):
            return None

        detected_frames, detected_solids = _collect_objects(g, input_bg)
        if len(detected_frames) != n_frames or len(detected_solids) != n_solids:
            return None

        if _frame_delta_pixels(detected_frames) <= 0:
            return None

        d4_only = _assemble_output(
            detected_frames,
            detected_solids,
            output_bg=input_bg,
            canonicalize_frames=False,
        )
        if np.array_equal(actual, d4_only):
            return None

        old_like = _original_solver_output_like(g, input_bg)
        if actual.shape == old_like.shape and np.array_equal(actual, old_like):
            return None

        if input_bg == OUTPUT_BG:
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
