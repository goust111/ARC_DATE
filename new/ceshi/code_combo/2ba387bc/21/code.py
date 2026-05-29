# Combo variant of 2ba387bc (index 21)
# combo_spec: D7/0+D16/0
# Name: 框架重着色+框架旋转

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "solvers" / "arc_utils.py").exists():
            return candidate
    raise RuntimeError("could not locate project root containing solvers/arc_utils.py")


PROJECT_ROOT = _find_project_root()
SOLVER_DIR = PROJECT_ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


FRAME_OUT_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_MARKERS = ((1, 1), (1, 2), (2, 2), (2, 1))


"""
Design analysis for the combo variant
=====================================

1. Concrete behavior and trace
   - The original task uses two 4x4 object classes on a plain background:
       * frame: a hollow border block;
       * solid: a filled 4x4 square.
     It sorts each class by input row and stacks frames on the left, solids on the right.
   - This combo asks for two frame-only transforms in the output:
       * D7/0: every frame is recolored to one fixed canonical color;
       * D16/0: every frame is rotated 90 degrees before stacking.
     Solids keep both their color and their orientation.
   - Hidden problem: the original hollow 4x4 ring is perfectly 90-degree symmetric, so a
     literal rotation would be invisible and the D16 half would degenerate.
   - To make rotation observable while preserving the "frame vs solid" structure, this
     generator uses a frame subtype with one extra same-color interior pixel attached to
     one corner of the 2x2 hole. Visually it is still a frame, but it now has orientation.
   - Example trace:
       input frame A = red border with an inner tab at the top-left of the hole
       input solid B = blue 4x4 square
       original-style output would place [red tab-frame | blue solid]
       combo output places [grey tab-frame rotated 90 deg clockwise | blue solid]
     The tab moves to the top-right corner after rotation, so the direction change is
     directly visible, while the solid remains identical to its input form.

2. Complexity calibration
   - Recolor alone would already differ from the original task, but the user explicitly
     requires a multi-dimensional fusion where rotation also matters.
   - The frame tab is therefore the minimum extra asymmetry needed to expose rotation
     without adding a third object class, labels, separators, or color-coded metadata.
   - The generator always creates at least 2 frames and forces at least 2 distinct frame
     orientations in each sample, so the rotation rule is supported by repeated evidence.
   - Input object colors exclude the canonical output frame color 5, making recoloring
     visually unambiguous.

3. Hidden-assumption audit for `main()`
   - `bg = get_bg(g)`: still valid. The background remains the globally most common color.
   - `find_components(... monochromatic=False)`: still valid if each object is isolated.
     The generator keeps 1-cell padding to prevent accidental component merges.
   - `bbox_h(b) != 4 or bbox_w(b) != 4`: still valid. All input objects remain 4x4.
   - Original frame test: "uniform border + 2x2 interior all bg".
     This assumption is no longer sufficient, because our observable frame subtype has
     one same-color interior tab. The solver must instead accept:
       * border is one non-bg color;
       * each interior cell is either bg or that border color;
       * the number of colored interior cells is 0 or 1.
   - Original frame output line:
       `paste(out, sub, i * 4, 0)`
     This relied on two old invariants:
       * the frame output color equals the frame input color;
       * the frame orientation is unchanged.
     Both become false here, so the frame patch must be rebuilt by rotating the input
     object and then recoloring every non-background pixel to `FRAME_OUT_COLOR`.
   - Solid pasting still relies on the old invariants and remains correct, because solids
     intentionally do not change in this combo.

4. Placement legality
   - Every object is still a 4x4 patch pasted inside a larger background grid.
   - Placement uses `random_free_location(... padding=1, padding_connectivity=4)`, so
     objects do not overlap, do not leave the canvas, and do not touch in a way that would
     merge them under 4-connectivity.
   - After construction, the generator re-runs the same collection logic as the solver and
     rejects any sample where counts or frame orientations are not recovered as expected.

5. Visual observability
   - Frames and solids are still separated structurally by shape: tab-frames are outline
     objects with a hole; solids are filled 4x4 squares.
   - In the output, every frame becomes grey and its tab turns 90 degrees clockwise.
     Solids keep their original colors and geometry.
   - Because each sample contains multiple frame orientations and at least one solid, a
     human can compare input/output pairs and infer the type-conditioned rule:
     "frame-like objects rotate and normalize to grey; solid blocks do not."

6. Visual cleanliness
   - No object overlaps, no guide lines, and no additional markers are added outside the
     objects themselves.
   - The asymmetry is only a single interior tab, which keeps the scene clean and close to
     the original task while still making rotation legible.
   - Output layout remains the original 8-column two-stack arrangement, so the only visual
     novelty is inside the frame objects, not in the overall canvas organization.

7. Color-change inferability
   - The recolor rule is low-entropy and type-conditioned: every frame becomes color 5,
     every solid keeps its own color.
   - Since frames appear in several different input colors, the grey output cannot be
     mistaken for a global palette swap.
   - Since solids preserve color, the left/right type split provides a strong structural
     cue for which objects participate in the recolor.

8. Generator anti-degeneracy checks
   - The generator compares the combo output against two weaker baselines on the same input:
       * original behavior: no recolor, no rotation;
       * recolor-only behavior: recolor frames but do not rotate them.
     Samples are rejected if the combo output matches either baseline.
   - This specifically guards against the failure mode where D7 happens but D16 is
     invisible, which would violate the requested fusion semantics.

9. New solver assumptions introduced here
   - Frame objects are assumed to be 4x4, single-color outline-like components whose
     interior contains at most one same-color tab pixel.
   - Rotation direction is fixed to 90 degrees clockwise for all frames.
   - The canonical frame output color is assumed different from background; the generator
     fixes `bg = 0` and excludes color 5 from inputs to guarantee that.
"""


def _is_frame_like(sub: np.ndarray, bg: int) -> tuple[bool, int | None]:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    if len(border_colors) != 1:
        return False, None

    border_color = int(next(iter(border_colors)))
    if border_color == int(bg):
        return False, None

    interior = np.array(sub[1:3, 1:3], dtype=int, copy=False)
    allowed = (interior == int(bg)) | (interior == int(border_color))
    if not bool(allowed.all()):
        return False, None

    colored_coords = np.argwhere(interior == int(border_color))
    if int(colored_coords.shape[0]) > 1:
        return False, None
    if int(colored_coords.shape[0]) == 0:
        return True, None

    rr, cc = map(int, colored_coords[0])
    for orient, (mr, mc) in enumerate(FRAME_MARKERS):
        if rr + 1 == int(mr) and cc + 1 == int(mc):
            return True, int(orient)
    return False, None


def _collect_objects(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, int, np.ndarray, int | None]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray, int | None]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        is_frame, orientation = _is_frame_like(sub, bg)
        if is_frame:
            frames.append((int(r0), int(c0), sub, orientation))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _recolor_non_bg(sub: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = np.full(sub.shape, int(bg), dtype=int)
    out[sub != int(bg)] = int(color)
    return out


def _transform_frame(sub: np.ndarray, bg: int, *, rotate: bool, recolor: bool) -> np.ndarray:
    out = np.array(sub, dtype=int, copy=True)
    if rotate:
        out = np.rot90(out, k=3)
    if recolor:
        out = _recolor_non_bg(out, bg, FRAME_OUT_COLOR)
    return out


def _build_output(
    frames: list[tuple[int, int, np.ndarray, int | None]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
    *,
    rotate_frames: bool,
    recolor_frames: bool,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub, _orientation) in enumerate(frames):
        patch = _transform_frame(sub, bg, rotate=rotate_frames, recolor=recolor_frames)
        paste(out, patch, int(i * 4), 0, transparent=bg)

    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects into two columns by type.

    Combo rule:
    - Frame-like objects stack on the left after a 90-degree clockwise rotation and a
      recolor to one fixed output color.
    - Solid 4x4 blocks stack on the right unchanged.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    return _build_output(frames, solids, bg, rotate_frames=True, recolor_frames=True)


def _make_frame(col: int, bg: int, orientation: int) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    rr, cc = FRAME_MARKERS[int(orientation) % 4]
    a[int(rr), int(cc)] = int(col)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _sample_frame_orientations(count: int) -> list[int]:
    while True:
        orientations = list(map(int, np.random.randint(0, 4, size=int(count))))
        if len(set(orientations)) >= 2:
            return orientations


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 33))
    w = int(np.random.randint(22, 33))
    g = np.full((h, w), int(bg), dtype=int)

    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(1, 5))
    total = n_frames + n_solids
    if total > len(OBJECT_COLORS):
        return None

    colors = list(np.random.permutation(np.array(OBJECT_COLORS, dtype=int))[:total])
    frame_colors = [int(c) for c in colors[:n_frames]]
    solid_colors = [int(c) for c in colors[n_frames:]]
    frame_orientations = _sample_frame_orientations(n_frames)

    objects: list[tuple[str, int, int | None]] = []
    for col, orientation in zip(frame_colors, frame_orientations):
        objects.append(("frame", int(col), int(orientation)))
    for col in solid_colors:
        objects.append(("solid", int(col), None))
    np.random.shuffle(objects)

    try:
        for kind, col, orientation in objects:
            patch = _make_frame(col, bg, int(orientation)) if kind == "frame" else _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        frames, solids = _collect_objects(g, bg)
        if len(frames) != n_frames or len(solids) != n_solids:
            return None

        recovered_orientations = [orientation for _r0, _c0, _sub, orientation in frames if orientation is not None]
        if len(recovered_orientations) != n_frames:
            return None
        if len(set(map(int, recovered_orientations))) < 2:
            return None

        combo = _build_output(frames, solids, bg, rotate_frames=True, recolor_frames=True)
        original_like = _build_output(frames, solids, bg, rotate_frames=False, recolor_frames=False)
        recolor_only = _build_output(frames, solids, bg, rotate_frames=False, recolor_frames=True)
        if np.array_equal(combo, original_like):
            return None
        if np.array_equal(combo, recolor_only):
            return None

        for _r0, _c0, sub, orientation in frames:
            if orientation is None:
                return None
            if int(sub[0, 0]) == int(FRAME_OUT_COLOR):
                return None
            rotated = _transform_frame(sub, bg, rotate=True, recolor=True)
            unrotated = _transform_frame(sub, bg, rotate=False, recolor=True)
            if np.array_equal(rotated, unrotated):
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
