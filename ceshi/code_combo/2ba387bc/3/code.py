# Combo variant of 2ba387bc (index 3)
# combo_spec: D1/1+D7/0
# Name: 颜色计数分类+框架统一着色

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "solvers" / "arc_utils.py").exists())
SOLVER_DIR = PROJECT_ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


FRAME_OUT_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


"""
Design analysis for combo D1/1 + D7/0
=====================================

1. Concrete behavior and trace
   - The scene is still a plain background with isolated 4x4 objects.
   - Input object types remain the original two:
       * solid = one non-background color everywhere in the 4x4 patch;
       * frame = border in one color, 2x2 center in background.
   - Original task:
       * detect frame vs solid from geometry (`sub[1:3, 1:3] == bg`);
       * keep input colors unchanged;
       * stack frames on the left and solids on the right.
   - Combo task:
       * detect type by color count instead of by interior geometry:
           len(colors in patch) == 1 -> solid
           len(colors in patch) == 2 -> frame
         In the valid task distribution, the two colors for a frame are
         {border_color, bg}.
       * keep the same row-order sorting and left/right stacking;
       * recolor every output frame to a single canonical color.
   - Example trace:
       input objects = teal frame, red solid, yellow frame
       original output = [teal frame; yellow frame] | [red solid]
       combo output    = [grey frame; grey frame]   | [red solid]
     The visible difference is the canonical frame color; the reasoning-path
     difference is that type is inferred from "1 color vs 2 colors".

2. Complexity calibration
   - If a sample had only one frame, the recolor rule would be weaker. The
     generator therefore always creates at least 2 frames.
   - If an input object already used the canonical output color, the recolor
     could become invisible. The generator excludes color 5 from all inputs.
   - Object colors are drawn from a small palette and are unique per object, so
     multiple differently colored frames provide repeated evidence that all
     2-color objects collapse to the same output frame color.

3. Hidden-assumption audit for `main()`
   - `bg = get_bg(g)`: still valid because the generator keeps a dominant plain
     background.
   - `find_components(... monochromatic=False)`: still valid because each 4x4
     object is isolated by at least one background cell under 4-connectivity.
   - `bbox_h(b) != 4 or bbox_w(b) != 4`: still valid because the generator only
     emits 4x4 objects.
   - Original frame test relied on geometry:
       `len(border_colors) == 1 and sub[1:3, 1:3] == bg`
     That assumption is intentionally replaced. The combo solver must not depend
     on reading the 2x2 center to decide the class; it now depends only on the
     number of distinct colors in the patch.
   - Original output pasting relied on "frame output keeps input color". D7/0
     breaks that assumption, so frames must be rebuilt as canonical patches
     before they are pasted into the output.
   - New solver assumption:
       every frame has exactly two colors and every solid exactly one color.
     The generator guarantees this by constructing only standard solid squares
     and standard frame rings.

4. Placement legality
   - Every patch is 4x4 and is placed with
     `random_free_location(... border=1, padding=1, padding_connectivity=4)`.
   - This prevents overlap, out-of-bounds placement, and accidental component
     merging through edge contact.

5. Visual observability
   - A human can see that all 2-color ring-like objects become the same grey in
     the output, while 1-color filled blocks keep their own colors.
   - The left/right grouping remains simple and unchanged, so the new burden is
     specifically "classify by color-count, then apply a type-conditioned recolor".
   - Requiring at least two differently colored frames makes the many-to-one
     recolor visible within a single example.

6. Visual cleanliness
   - No overlays, markers, or mixed-size sprites are introduced.
   - The output layout stays the original tidy 2-column stack.
   - Unique input colors keep each source object easy to track.

7. Change inferability
   - The color mapping is low-entropy: every frame, regardless of input border
     color, becomes `FRAME_OUT_COLOR`.
   - Because solids do not change color, the recolor is clearly conditioned on
     object type rather than on a global palette permutation.

8. Generator-side anti-degeneracy checks
   - Reject samples if combo output equals original output.
   - Reject samples if any input frame already uses the canonical frame color.
   - Re-run both the color-count classifier and the original geometry classifier
     and require them to agree on the generated scene, so the task stays within
     the intended canonical distribution.

9. Post-edit assumption audit
   - `_is_color_count_frame(...)` assumes the background color is one of the two
     colors inside a frame patch. The generator guarantees that by using the
     original 4x4 ring geometry for all frames.
   - `_build_output(...)` assumes recolored frames should keep frame geometry.
     The generator guarantees every frame is the standard 4x4 ring, so the
     canonical rewritten patch is always valid.
"""


def _color_set(sub: np.ndarray) -> set[int]:
    return set(map(int, sub.flatten()))


def _is_color_count_frame(sub: np.ndarray, bg: int) -> bool:
    cols = _color_set(sub)
    return len(cols) == 2 and int(bg) in cols


def _is_original_frame_shape(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    return len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _canonical_frame(bg: int) -> np.ndarray:
    out = np.full((4, 4), int(FRAME_OUT_COLOR), dtype=int)
    out[1:3, 1:3] = int(bg)
    return out


def _collect_objects(
    g: np.ndarray,
    bg: int,
    *,
    frame_classifier: Callable[[np.ndarray, int], bool],
) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        target = frames if frame_classifier(sub, bg) else solids
        target.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _build_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
    *,
    recolor_frames: bool,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        patch = _canonical_frame(bg) if recolor_frames else sub
        paste(out, patch, int(i * 4), 0, transparent=bg)

    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type: frames left, solids right.

    Combo rule:
    - classify by the number of colors in each 4x4 object;
    - objects with 2 colors are frames and are recolored to one fixed color;
    - objects with 1 color are solids and keep their original colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg, frame_classifier=_is_color_count_frame)
    return _build_output(frames, solids, bg, recolor_frames=True)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _combo_metrics(g: np.ndarray, bg: int) -> dict[str, int]:
    frames, solids = _collect_objects(g, bg, frame_classifier=_is_color_count_frame)
    old_frames, old_solids = _collect_objects(g, bg, frame_classifier=_is_original_frame_shape)
    combo_out = _build_output(frames, solids, bg, recolor_frames=True)
    old_out = _build_output(old_frames, old_solids, bg, recolor_frames=False)
    diff_pixels = int(np.count_nonzero(combo_out != old_out)) if combo_out.shape == old_out.shape else -1
    recolored_frames = sum(int(sub[0, 0]) != int(FRAME_OUT_COLOR) for _r0, _c0, sub in frames)
    return {
        "frames": int(len(frames)),
        "solids": int(len(solids)),
        "old_frames": int(len(old_frames)),
        "old_solids": int(len(old_solids)),
        "diff_pixels": diff_pixels,
        "recolored_frames": int(recolored_frames),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    g = np.full((h, w), int(bg), dtype=int)

    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(1, 5))
    total = n_frames + n_solids
    if total > len(OBJECT_COLORS):
        return None

    colors = list(np.random.permutation(np.array(OBJECT_COLORS, dtype=int))[:total])
    objects: list[tuple[str, int]] = [("frame", int(c)) for c in colors[:n_frames]]
    objects += [("solid", int(c)) for c in colors[n_frames:]]
    np.random.shuffle(objects)

    try:
        for kind, col in objects:
            patch = _make_frame(col, bg) if kind == "frame" else _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        metrics = _combo_metrics(g, bg)
        if metrics["frames"] != n_frames or metrics["solids"] != n_solids:
            return None
        if metrics["frames"] != metrics["old_frames"] or metrics["solids"] != metrics["old_solids"]:
            return None
        if metrics["diff_pixels"] <= 0:
            return None
        if metrics["recolored_frames"] != n_frames:
            return None

        frames, solids = _collect_objects(g, bg, frame_classifier=_is_color_count_frame)
        if any(len(_color_set(sub)) != 2 for _r0, _c0, sub in frames):
            return None
        if any(len(_color_set(sub)) != 1 for _r0, _c0, sub in solids):
            return None

        _ = main(g)
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
