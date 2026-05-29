# Variant of 2ba387bc
# Dimension: D7 — 变换——堆叠时的颜色处理 (transform)
# Change: 方块在分类和堆叠过程中保持原始颜色不变... → 框架重着色为统一颜色，实心保持原色

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = PROJECT_ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


FRAME_OUT_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


"""
Design analysis for the D7 variant
==================================

1. Concrete behavior and trace
   - The input still contains scattered 4x4 objects on a plain background.
   - A frame is still a 4x4 ring whose 2x2 center is background; a solid is still a
     fully filled 4x4 monochrome square.
   - The original task stacked both kinds without changing their colors. This variant
     changes only the frame rendering step in the output: every detected frame is
     rewritten to the same fixed color `FRAME_OUT_COLOR`, while solids keep their own
     input colors.
   - Example trace:
       input frame color = 8, input solid color = 3
       original output block pair = [8-frame | 3-solid]
       variant output block pair  = [5-frame | 3-solid]
     The classification, sorting by row, and two-column stacking all stay identical;
     only the pixels of frame borders are normalized.

2. Complexity calibration
   - Recoloring would be visually weak if a sample had only one frame or if a frame
     already used the canonical output color. The generator therefore:
       * always creates at least 2 frames;
       * never uses color 5 for any input object.
   - This keeps the rule observable across a few examples without introducing extra
     object types, extra markers, or hidden color-dependent sorting.

3. Hidden-assumption audit for `main()`
   - `bg = get_bg(g)`: still valid. The variant does not change background detection.
   - `find_components(... monochromatic=False)`: still valid as long as each 4x4 object
     is isolated as one connected component. The generator enforces 1-cell padding.
   - `bbox_h(b) != 4 or bbox_w(b) != 4`: unchanged; the object size rule is preserved.
   - Frame test = "uniform border + 2x2 interior == bg": unchanged. The variant only
     changes output color, not how a frame is recognized.
   - Original bug-risk line:
       `paste(out, sub, i * 4, 0)`
     This relied on the old invariant "frame output color equals frame input color".
     That assumption is false in D7, so the frame patch must be reconstructed in the
     canonical color before pasting.
   - Solid pasting keeps the old assumption and remains correct because solids preserve
     their color.

4. Placement legality
   - All objects are still 4x4.
   - `random_free_location(... padding=1, connectivity=4)` guarantees no overlap, no
     edge-touching, and no accidental merged component under 4-connectivity.
   - The generator validates the produced grid by re-running the solver logic and
     rejecting any sample where the D7 recolor would not visibly differ from the
     original non-recolor behavior.

5. Visual observability and cleanliness
   - The only new visual signal is that every frame becomes grey in the output.
   - Keeping solids in their original colors makes the left/right type split easy to
     compare against the input.
   - Excluding grey from input objects prevents ambiguous cases where a frame appears
     unchanged after recoloring.
   - No extra overlays or internal decorations are added, so the scene stays as clean
     as the original task.

6. Color-change inferability
   - The color mapping is low-entropy and structural: "if the 4x4 object is a frame,
     then its output color is always grey".
   - Because multiple differently colored frames appear in the same input distribution,
     the solver can infer a type-conditioned recolor rather than a global palette swap.

7. New assumptions introduced by the variant solver
   - The solver now assumes that the canonical frame color is visible against the
     background. The generator guarantees `bg = 0`, so `FRAME_OUT_COLOR = 5` is safe.
   - The solver assumes the frame shape remains a standard 4x4 ring after recoloring.
     The generator only emits that exact frame geometry.
"""


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.flatten()))
    return len(bcols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _canonical_frame(bg: int) -> np.ndarray:
    a = np.full((4, 4), int(FRAME_OUT_COLOR), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


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
        if _is_frame(sub, bg):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

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

    D7 variant rule:
    - Classification and stacking are unchanged.
    - Every frame is recolored to one fixed output color before stacking.
    - Solids keep their original colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    return _build_output(frames, solids, bg, recolor_frames=True)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    g = np.full((h, w), bg, dtype=int)

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

        frames, solids = _collect_objects(g, bg)
        if len(frames) != n_frames or len(solids) != n_solids:
            return None

        out_new = _build_output(frames, solids, bg, recolor_frames=True)
        out_old = _build_output(frames, solids, bg, recolor_frames=False)
        if np.array_equal(out_new, out_old):
            return None

        for _r0, _c0, sub in frames:
            if int(sub[0, 0]) == int(FRAME_OUT_COLOR):
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
