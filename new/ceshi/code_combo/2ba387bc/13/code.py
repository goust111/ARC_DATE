# Combo variant of 2ba387bc (index 13)
# combo_spec: D3/1+D16/0
# Name: 矩形方块+框架旋转

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

THIS_FILE = Path(__file__).resolve()
for _parent in THIS_FILE.parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        if str(_parent / "solvers") not in sys.path:
            sys.path.append(str(_parent / "solvers"))
        break

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D3/1 + D16/0
======================================

This combo starts from the original 2ba387bc rule "classify scattered 4x4 squares into
frame-vs-solid columns" and fuses two changes:

- D3/1: every object becomes the same non-square rectangle (for example 4x6 or 3x5)
- D16/0: frame objects rotate 90 degrees before being placed into the left output column

a) Concrete behavior and trace
------------------------------
- Original behavior:
  every relevant component is 4x4; frames and solids keep the same 4x4 shape in output;
  output width is always 8 because both columns are 4 cells wide.
- Combo behavior:
  suppose one sample uses 4x6 rectangles.
  A frame is a 4x6 one-color border with a 2x4 background hole.
  A solid is a 4x6 filled rectangle.
  After classification, each frame is rotated to 6x4 before going into the left column,
  while each solid stays 4x6 in the right column.
- Small trace:
  if two top-most frames are colors 8 then 3, and two top-most solids are 2 then 7,
  original logic would either ignore the objects (`!= 4x4`) or, in a D3-only rectangle variant,
  stack 4x6 frames on the left and 4x6 solids on the right.
  In this combo, the left column instead contains 6x4 rotated frames stacked with step 6,
  the right column contains 4x6 solids stacked with step 4, total width becomes 4 + 6 = 10,
  and total height becomes `max(2*6, 2*4) = 12`.
  The visible difference is not only that objects are rectangular, but also that the frame
  column has swapped width/height relative to the solid column.

b) Complexity calibration
-------------------------
- The generator only uses a small visible set of non-square sizes: (3,5), (4,6), (5,3), (6,4).
- All objects inside one sample share the same source size. This keeps the task readable and
  avoids introducing a second rule about mixed source sizes.
- Both frames and solids appear in every sample, so the asymmetric rotation rule is visible.
- Colors stay unique per object to avoid ambiguous grouping by color.

c) Implicit assumption audit of the original main()
---------------------------------------------------
- `bbox_h(b) != 4 or bbox_w(b) != 4`
  Broken by D3/1. The combo must detect general rectangles instead of fixed 4x4 blocks.
- `sub[1:3, 1:3] == bg`
  Broken by D3/1. Frame interior must be checked with `sub[1:-1, 1:-1]`.
- `outH = 4 * max(...)`, `outW = 8`, paste steps `i * 4`
  Broken by both changes. Output geometry now depends on:
  frame size after rotation `(w, h)` and solid size `(h, w)`.
- "Left and right columns have equal width"
  Broken specifically by D16/0 on non-square rectangles. The combo must allocate
  `frame_col_width = h` and `solid_col_width = w`.

d) Placement legality
---------------------
- The generator uses only `h, w >= 3`, so each frame has a non-empty interior.
- Objects are placed with `random_free_location(..., border=1, padding=1, padding_connectivity=4)`,
  so they neither overlap nor touch orthogonally.
- Canvas size is scaled from the chosen rectangle size and object counts, keeping placement feasible.
- Each candidate sample is validated by recomputing the exact expected output and checking
  `np.array_equal(main(input), expected)`.

e) Visual observability
-----------------------
- Non-square source objects make the D3 change obvious in the input.
- In the output, the left column has the rotated frame aspect ratio while the right column keeps
  the original solid aspect ratio, so the D16 change is also visible.
- Using both frame and solid objects in each sample ensures a solver can compare the two columns
  and infer that only frames change orientation.

f) Visual cleanliness
---------------------
- Only two object morphologies exist: hollow rectangles and filled rectangles.
- Objects are isolated with padding, so connected-component boundaries stay clean.
- The output is still a simple two-column arrangement, just with unequal column widths.

g) Change inferability / anti-degeneracy
----------------------------------------
- No color remapping is introduced, so the only new inference burden is geometric.
- A degenerate generator could accidentally collapse back to a D3-only rectangle task if it
  allowed square objects or if the output builder silently treated both columns the same.
  This generator prevents that by forcing `h != w` and by checking that combo output differs
  from the "no frame rotation" rectangle output on accepted samples.

h) Quantitative fusion indicator
--------------------------------
- Indicator 1: source rectangles are non-square, i.e. `h != w`. Expected: 100% of objects.
- Indicator 2: combo output differs from a D3-only rectangle sorter that does not rotate frames.
  Expected: 100% of accepted samples, because non-square frame shapes swap width/height.

i) New main() assumptions and generator guarantees
--------------------------------------------------
- Assumption: all relevant objects are axis-aligned rectangles with height/width >= 3.
  Guarantee: generator creates only such rectangles.
- Assumption: all relevant objects in one sample share one source size `(h, w)`.
  Guarantee: generator samples one size per grid and uses it for all objects.
- Assumption: each relevant object is either a monochrome solid rectangle or a one-color
  frame with full background interior.
  Guarantee: generator constructs exactly those two patterns.
- Assumption: `np.rot90(frame)` is the left-column transform for every frame.
  Guarantee: generator computes expected outputs with the same transform.
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
    if len(bcols) != 1:
        return False
    only_border_col = int(next(iter(bcols)))
    if only_border_col == int(bg):
        return False
    return bool((sub[1:-1, 1:-1] == int(bg)).all())


def _is_solid_rect(sub: np.ndarray, bg: int) -> bool:
    return int(sub[0, 0]) != int(bg) and bool((sub == int(sub[0, 0])).all())


def _collect_objects(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) < 3 or bbox_w(b) < 3:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _is_frame_rect(sub, bg):
            frames.append((int(r0), int(c0), sub))
        elif _is_solid_rect(sub, bg):
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, frames, solids


def _validate_shared_size(frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]]) -> tuple[int, int]:
    objs = [sub for _r0, _c0, sub in frames + solids]
    if not objs:
        raise ValueError("no valid rectangular objects found")

    src_h = int(objs[0].shape[0])
    src_w = int(objs[0].shape[1])
    if src_h < 3 or src_w < 3:
        raise ValueError("objects must be at least 3x3")
    if src_h == src_w:
        raise ValueError("combo expects non-square source rectangles")

    for sub in objs[1:]:
        if sub.shape != (src_h, src_w):
            raise ValueError("mixed source rectangle sizes are not supported in this combo")

    return src_h, src_w


def _build_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    src_h, src_w = _validate_shared_size(frames, solids)
    frame_h, frame_w = int(src_w), int(src_h)
    solid_h, solid_w = int(src_h), int(src_w)

    out_h = max(int(len(frames)) * frame_h, int(len(solids)) * solid_h)
    out_w = frame_w + solid_w
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        rotated = np.rot90(sub, k=1)
        paste(out, rotated, int(i * frame_h), 0)

    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * solid_h), int(frame_w))

    return out


def _build_output_without_rotation(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    src_h, src_w = _validate_shared_size(frames, solids)
    out_h = src_h * max(int(len(frames)), int(len(solids)))
    out_w = src_w * 2
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * src_h), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * src_h), int(src_w))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered same-size rectangles into two output columns by type.

    - Frames: one-color rectangular border with background interior.
    - Solids: one-color filled rectangles.
    - Groups are sorted by input row, then input col.
    - Frames rotate 90 degrees before entering the left column.
    - Solids keep their original orientation in the right column.
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


def _sample_distinct_color(used_cols: set[int]) -> int:
    col = int(np.random.randint(1, 10))
    while col in used_cols:
        col = int(np.random.randint(1, 10))
    used_cols.add(col)
    return col


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h, w = RECT_SIZES[int(np.random.randint(0, len(RECT_SIZES)))]
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    min_h = max(22, h * n_solids + w * n_frames + 10)
    min_w = max(22, w * 3 + h + 10)
    canvas_h = int(np.random.randint(min_h, min_h + 8))
    canvas_w = int(np.random.randint(min_w, min_w + 8))
    g = np.full((canvas_h, canvas_w), bg, dtype=int)

    try:
        used_cols: set[int] = set()
        placed_frames: list[tuple[int, int, np.ndarray]] = []
        placed_solids: list[tuple[int, int, np.ndarray]] = []

        for _ in range(n_frames):
            col = _sample_distinct_color(used_cols)
            patch = _make_frame(col, h, w, bg)
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            placed_frames.append((int(r0), int(c0), patch))

        for _ in range(n_solids):
            col = _sample_distinct_color(used_cols)
            patch = _make_solid(col, h, w)
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            placed_solids.append((int(r0), int(c0), patch))

        placed_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_solids.sort(key=lambda t: (int(t[0]), int(t[1])))

        expected = _build_output(bg, placed_frames, placed_solids)
        d3_only = _build_output_without_rotation(bg, placed_frames, placed_solids)
        if np.array_equal(expected, d3_only):
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
