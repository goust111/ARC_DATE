# Variant of 2ba387bc
# Dimension: D3 — 源对象的形状与尺寸 (src)
# Change: 所有对象均为固定的4x4正方形，框架为1像素宽边框+2x2背景内部，实心为全色填充... → 矩形替代正方形（如4x6或3x4）

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the D3 variant
==================================

No source_task is provided, so the design baseline is the original 2ba387bc solver/generator only.

a) Concrete behavior and trace
------------------------------
- Original objects are always 4x4. A frame is a one-color 4x4 border with a 2x2 background hole.
- In this variant, every object in one sample shares one rectangular size (h, w) with h != w and h, w >= 3.
  Examples used by the generator are limited to a small visible set such as 3x5, 4x6, 5x3, 6x4.
- Frames still have a 1-pixel border and a background-filled interior of shape (h-2, w-2).
- Solids are fully filled h x w rectangles.
- Input still scatters multiple objects on a background canvas.
- Output still has two columns: frames on the left, solids on the right, both sorted by input row then input col.
- Because the shared object width is now w instead of 4, output width becomes 2*w.
- Because the shared object height is now h instead of 4, output height becomes h * max(num_frames, num_solids).

Trace on a small example:
- Suppose all objects are 3x5.
- Original logic would reject them because `bbox_h != 4 or bbox_w != 4`.
- New logic accepts a component if it is either:
  1. a solid monochrome rectangle, or
  2. a one-color rectangular border whose full interior is background.
- If the top-most frame is color 8 and the second frame is color 3, while solids are 1 then 6,
  the output becomes:
  left column width 5: frame(8) stacked above frame(3)
  right column width 5: solid(1) stacked above solid(6)
  total output shape: (3 * 2, 10).

b) Complexity calibration
-------------------------
- The change must be visually obvious, so every generated sample uses non-square objects only.
- The size choices are restricted to a small set of four rectangle sizes. This avoids overfitting to one exact size
  while keeping the inference space small enough for humans.
- All objects inside one sample share the same rectangle size. That preserves the original task's "uniform source object"
  structure and avoids introducing a second major rule about mixed slot sizes.
- Colors remain unique per object, matching the original generator's disambiguation strategy.

c) Implicit assumption audit of the original main()
---------------------------------------------------
- `if bbox_h(b) != 4 or bbox_w(b) != 4: continue`
  Assumption: every relevant object is exactly 4x4.
  Variant impact: invalid. Must be replaced by generic rectangular detection.
- `sub[1:3, 1:3] == bg`
  Assumption: frame interiors are always exactly 2x2.
  Variant impact: invalid. Interior must become `sub[1:-1, 1:-1]`.
- `outH = 4 * max(...)`, `outW = 8`, and paste row steps of `i * 4`
  Assumption: every object has height 4 and each column has width 4.
  Variant impact: invalid. Must use the detected shared rectangle size `(h, w)`.
- Border extraction itself is still valid for rectangles:
  `sub[0, :]`, `sub[-1, :]`, `sub[1:-1, 0]`, `sub[1:-1, -1]` generalizes unchanged.

d) Placement legality
---------------------
- Generator chooses `h, w >= 3`, so a 1-pixel frame border always leaves a non-empty interior.
- Each object is placed with `random_free_location(..., border=1, padding=1, padding_connectivity=4)`.
  This prevents overlap, keeps objects in bounds, and avoids accidental touching that would merge components.
- Canvas size is scaled from the chosen rectangle size and object counts, so placement remains feasible.
- Generator verifies `main(input) == expected_output` before accepting a sample.

e) Visual observability
-----------------------
- The key D3 change is directly visible because no object is square in generated data.
- The output width and height now visibly track the rectangle dimensions.
- Within one sample, repeated same-size rectangles give the solver enough evidence that size matters structurally,
  not just incidentally for one object.

f) Visual cleanliness
---------------------
- Objects are isolated by at least one cell of padding, so connected-component boundaries remain clean.
- Only two object types exist, each with a single simple silhouette: hollow rectangle or solid rectangle.
- Output keeps the original tidy two-column arrangement, now with rectangle-aware width/height.

g) Color-rule audit
-------------------
- No color remapping is introduced.
- Colors only identify distinct objects; sorting depends only on visible positions, not color ids.

h) Variant-specific quantitative check
--------------------------------------
- Metric 1: proportion of generated objects with `height != width`. Expected: 100%.
- Metric 2: fraction of samples where the new solver output differs from the original 4x4 solver output.
  Expected: near 100%, because the original solver ignores non-4x4 objects.

i) New main() assumptions and generator guarantees
--------------------------------------------------
- Assumption: all relevant objects are axis-aligned rectangles with height/width >= 3.
  Guarantee: generator only creates such objects.
- Assumption: all relevant objects in one sample share one size `(h, w)`.
  Guarantee: generator samples one size per grid and uses it for every frame/solid.
- Assumption: every accepted object is either a monochrome solid rectangle or a one-color frame with bg interior.
  Guarantee: generator constructs exactly those two patterns.
- Assumption: different objects do not touch orthogonally.
  Guarantee: placement uses `padding=1` with 4-connectivity.
"""


RECT_SIZES: tuple[tuple[int, int], ...] = (
    (3, 5),
    (4, 6),
    (5, 3),
    (6, 4),
)


def _is_frame_rect(sub: np.ndarray, bg: int) -> bool:
    h, w = map(int, sub.shape)
    if h < 3 or w < 3:
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.flatten()))
    if len(bcols) != 1 or int(next(iter(bcols))) == int(bg):
        return False
    return bool((sub[1:-1, 1:-1] == int(bg)).all())


def _is_solid_rect(sub: np.ndarray, bg: int) -> bool:
    return int(sub[0, 0]) != int(bg) and bool((sub == int(sub[0, 0])).all())


def _collect_objects(g: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if bbox_h(b) < 3 or bbox_w(b) < 3:
            continue
        if _is_frame_rect(sub, bg):
            frames.append((int(r0), int(c0), sub))
        elif _is_solid_rect(sub, bg):
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, frames, solids


def _build_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    objs = [sub for _r0, _c0, sub in frames + solids]
    if not objs:
        raise ValueError("no valid rectangular objects found")

    obj_h = int(objs[0].shape[0])
    obj_w = int(objs[0].shape[1])
    for sub in objs[1:]:
        if sub.shape != (obj_h, obj_w):
            raise ValueError("mixed object sizes are not supported in this D3 variant")

    out_h = obj_h * max(int(len(frames)), int(len(solids)))
    out_w = obj_w * 2
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * obj_h), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * obj_h), int(obj_w))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered same-size rectangles into two columns by type.

    - Frames: one-color rectangular border with background interior.
    - Solids: one-color filled rectangles.
    - Both groups are sorted by input row, then input col.
    - Output uses the shared rectangle size: width = 2*w, height = h * max(group sizes).
    """
    g = np.array(grid, dtype=int)
    bg, frames, solids = _collect_objects(g)
    return _build_output(bg, frames, solids)


def _make_frame(col: int, h: int, w: int, bg: int = 0) -> np.ndarray:
    a = np.full((h, w), int(col), dtype=int)
    a[1:-1, 1:-1] = int(bg)
    return a


def _make_solid(col: int, h: int, w: int) -> np.ndarray:
    return np.full((h, w), int(col), dtype=int)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h, w = RECT_SIZES[int(np.random.randint(0, len(RECT_SIZES)))]
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    min_h = max(22, h * (n_frames + n_solids) + 10)
    min_w = max(22, w * 3 + 10)
    canvas_h = int(np.random.randint(min_h, min_h + 8))
    canvas_w = int(np.random.randint(min_w, min_w + 8))
    g = np.full((canvas_h, canvas_w), bg, dtype=int)

    try:
        used_cols: set[int] = set()
        placed_frames: list[tuple[int, int, np.ndarray]] = []
        placed_solids: list[tuple[int, int, np.ndarray]] = []

        for _ in range(n_frames):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            patch = _make_frame(col, h, w, bg)
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            placed_frames.append((int(r0), int(c0), patch))

        for _ in range(n_solids):
            col = int(np.random.randint(1, 10))
            while col in used_cols:
                col = int(np.random.randint(1, 10))
            used_cols.add(col)
            patch = _make_solid(col, h, w)
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            placed_solids.append((int(r0), int(c0), patch))

        placed_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _build_output(bg, placed_frames, placed_solids)
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
