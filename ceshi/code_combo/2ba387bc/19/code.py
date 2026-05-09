# Combo variant of 2ba387bc (index 19)
# combo_spec: D5/1+D16/1
# Name: 分隔列+实心缩小

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

"""
Design analysis

a) Concrete behavior trace
   - Input stays the same as the original task: several disconnected 4x4 objects on a dominant
     background, where each object is either a frame (1-pixel border, 2x2 bg interior) or a
     solid 4x4 block.
   - Original output: frames are sorted by input row and stacked in cols 0..3; solids are sorted
     by input row and stacked in cols 4..7. Both groups use a 4-row stride, so output height is
     `4 * max(n_frames, n_solids)`.
   - Combo output: frames still go to cols 0..3 with a 4-row stride, but a full background
     separator is inserted at col 4 and the solid group moves to the right side after that gap.
     Each solid shrinks from 4x4 to 2x2, so solids now use a 2-row stride and are pasted at
     cols 5..6, while the overall width remains 9 because D5 reserves a 4-column right lane
     (cols 5..8) separated from the left lane by the explicit gap.
   - Example: if the input contains frames F1,F2 and solids S1,S2,S3 in top-to-bottom order, the
     original task would produce an 8x8 or 12x8 canvas with F1/F2 at rows 0 and 4, and S1/S2/S3
     at rows 0,4,8. The combo instead outputs width 9, puts frames at rows 0 and 4 in cols 0..3,
     keeps col 4 empty, and pastes 2x2 solids at rows 0,2,4 in cols 5..6. The left and right
     stacks therefore no longer share the same vertical rhythm.

b) Complexity calibration
   - The combo is visually stronger than either change alone: the separator makes the two output
     regions explicit, and the shrunk solids make the right stack denser vertically.
   - I keep the input object vocabulary unchanged so the task still has one clean inference chain:
     detect 4x4 objects, classify them, sort them, then apply the new output layout.
   - Both classes are forced to appear in generated samples so the separator and the 2-row solid
     stride are both visible instead of degenerating into a single-column case.

c) Original implicit assumptions that break
   - `find_components(... monochromatic=False)` assumes objects remain disconnected by 4-neighbor
     connectivity. The generator still guarantees that with placement padding.
   - `bbox_h(b) == bbox_w(b) == 4` assumes every relevant input object is still 4x4. The combo
     does not change the input object size, so that stays valid.
   - Frame detection via `sub[1:3, 1:3] == bg` assumes the frame interior remains bg. That stays
     valid because only the output transform changes.
   - `outH = 4 * max(...)` is no longer valid because solids now consume 2 rows each, not 4.
   - `outW = 8` and pasting solids at col 4 are no longer valid because D5 inserts a separator;
     the right side starts at col 5 and total width becomes 9.
   - Treating the right group as 4x4 patches is no longer valid because D16 shrinks solids to 2x2.

d) Placement legality
   - The generator still scatters 4x4 objects into a larger background grid using
     `random_free_location(... padding=1, padding_connectivity=4)` so objects cannot overlap,
     cannot leave the canvas, and cannot merge into one component through 4-neighbor contact.
   - After placement, the generator re-runs the same collector used by the solver and rejects any
     accidental sample where the intended counts are not recovered exactly.

e) Visual observability
   - A human solver can see the rule from a few examples because two independent cues co-occur:
     a permanent blank separator column and a right stack that advances every 2 rows instead of 4.
   - No invisible feature such as numeric color id is used for sorting; ordering is still by input
     row position, which is directly visible.

f) Visual cleanliness
   - The output has two non-overlapping regions: a 4-column frame lane on the left and a separated
     right lane after a blank column. Nothing overlaps, and the blank separator prevents the denser
     2x2 right stack from being visually confused with the 4x4 frame stack.
   - The unused part of the reserved right lane (cols 7..8) stays background everywhere, so the
     canvas remains structured rather than cluttered.

g) Color-rule inferability
   - There is no color remapping in this combo. Colors are preserved from the input objects, which
     keeps the task about shape/layout transformation rather than arbitrary recoloring.

h) Anti-degeneration checks in the generator
   - Each generated sample contains at least one frame and at least one solid.
   - The generator validates that the combo solver output has width 9, a fully background
     separator column at col 4, and exactly `4 * n_solids` non-background pixels in cols 5..6,
     which proves every solid became exactly one 2x2 patch.
   - The generator also compares against the original baseline transformation and rejects a sample
     if the combo output would accidentally match it. With both classes present, the width/layout
     change already makes equality impossible, but the explicit check documents the intended metric.

i) New assumptions introduced by the combo solver
   - The solver assumes the right lane begins immediately after the separator at col 5, and that
     only cols 5..6 are used by the shrunken solids. The generator is aligned with that exact
     convention because it validates the produced output layout.
   - The solver assumes every solid can be shrunk by taking its center 2x2 (`sub[1:3, 1:3]`).
     Since generated solids are uniform 4x4 color blocks, any 2x2 crop would be equivalent, and
     the generator preserves that invariant.
   - The solver assumes sorting ties can be broken by column after row. The generator places each
     object as a separate component and does not rely on any hidden attribute beyond visible
     position, so that tie-break remains safe.
"""


def _collect_objects(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
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
    return frames, solids


def _shrink_solid(sub: np.ndarray) -> np.ndarray:
    return np.array(sub[1:3, 1:3], dtype=int, copy=True)


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)

    out = np.full((4 * max(len(frames), len(solids)), 8), bg, dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, 4 * i, 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, 4 * i, 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo D5/1 + D16/1 for task 2ba387bc.

    - Frames stay 4x4 in the left lane at cols 0..3 with 4-row spacing.
    - Col 4 is an explicit background separator.
    - Solids shrink to 2x2 and are stacked at cols 5..6 with 2-row spacing.
    - Total width is fixed at 9, so the right lane remains visually separated even though the
      shrunken solids only occupy part of it.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)

    out_h = max(4 * len(frames), 2 * len(solids))
    out = np.full((out_h, 9), bg, dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, 4 * i, 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, _shrink_solid(sub), 2 * i, 5)

    return out


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), col, dtype=int)
    patch[1:3, 1:3] = bg
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


def _validate_generated_sample(g: np.ndarray, n_frames: int, n_solids: int, bg: int) -> bool:
    frames, solids = _collect_objects(g, bg)
    if len(frames) != n_frames or len(solids) != n_solids:
        return False

    out = main(g)
    if out.shape != (max(4 * n_frames, 2 * n_solids), 9):
        return False
    if not bool((out[:, 4] == bg).all()):
        return False
    if int((out[:, 5:7] != bg).sum()) != 4 * n_solids:
        return False
    if not bool((out[:, 7:9] == bg).all()):
        return False
    if np.array_equal(out, _baseline_main(g)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    try:
        used_colors: set[int] = set()

        for _ in range(n_frames):
            col = int(np.random.randint(1, 10))
            while col in used_colors:
                col = int(np.random.randint(1, 10))
            used_colors.add(col)

            patch = _make_frame(col, bg)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        for _ in range(n_solids):
            col = int(np.random.randint(1, 10))
            while col in used_colors:
                col = int(np.random.randint(1, 10))
            used_colors.add(col)

            patch = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        if not _validate_generated_sample(g, n_frames, n_solids, bg):
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
