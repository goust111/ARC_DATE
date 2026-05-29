# Combo variant of 2ba387bc (index 47)
# combo_spec: D1/3+D3/1+D16/0
# Name: 对称性分类+旋转+矩形

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo variant 47
====================================

This is a fused variant of 2ba387bc combining:
- D1/3: classification is no longer "empty interior => frame" vs "filled => solid";
  instead it is based on whether the whole colored object has a reflection symmetry axis.
- D3/1: source objects are non-square rectangles rather than fixed 4x4 squares.
- D16/0: the left-group objects are rotated 90 degrees before being stacked.

a) Concrete behavior and trace
------------------------------
- Original task:
  each object is a 4x4 square. Frames are hollow, solids are fully filled.
  The solver checks the 2x2 interior, sorts by input row, then stacks frames on the left
  and solids on the right without changing their shape.
- Fused task here:
  every object is a same-size non-square rectangle, such as 5x7 or 7x5.
  There are two generated object families, and both are frame-like so that "hollow vs filled"
  is no longer enough to solve the task:
  1. Symmetric frame-like rectangles:
     a one-color rectangular frame with a centered vertical spine of another color.
     The full color pattern is mirror-symmetric, and the object remains connected.
  2. Asymmetric frame-like rectangles:
     a one-color rectangular frame with a small corner hook of another color attached to the border.
     The full color pattern has no horizontal or vertical reflection symmetry.
- Classification rule:
  if the whole cropped object equals its vertical flip or horizontal flip, it belongs
  to the left group; otherwise it belongs to the right group.
- Transformation rule:
  left-group objects are rotated 90 degrees CCW before output, so their width/height swap.
  Right-group objects are pasted unchanged.

Small trace:
- Suppose input objects are all 5x7.
- A symmetric object might look like a 5x7 hollow rectangle with a centered colored spine.
  The original 2ba387bc logic would fail because it expects exactly 4x4 with a fully empty 2x2 middle.
  The new solver instead notices mirror symmetry, classifies it into the left group, rotates it
  to 7x5, and stacks it in the left column.
- An asymmetric object might look like a 5x7 hollow rectangle with a corner-attached hook near the top-left.
  The new solver sees no reflection symmetry, leaves it unrotated as 5x7, and stacks it on the right.
- The crucial difference is that classification and transformation now interact:
  the same non-square bounding box can end up with different output dimensions depending on symmetry.

b) Complexity calibration
-------------------------
- Non-square sizes are restricted to a short list of odd-dimension rectangles:
  5x7, 7x5, 5x9, 9x5.
  This makes the D3 change visually obvious while keeping the inference space small.
- Odd dimensions let the centered symmetry spine be unambiguous to humans.
- The asymmetric class uses only four corner-anchored hook positions, so the asymmetry is visible
  but not visually chaotic.
- Each sample has repeated objects from both classes, which gives human solvers repeated evidence
  for "symmetric ones rotate and go left; asymmetric ones stay and go right."

c) Implicit assumption audit of the original main()
---------------------------------------------------
- `bbox_h(b) != 4 or bbox_w(b) != 4`
  Original assumption: all relevant objects are fixed 4x4 squares.
  Status here: invalid. Objects are rectangles with multiple possible non-square sizes.
- `sub[1:3, 1:3] == bg`
  Original assumption: frame detection is equivalent to an empty 2x2 interior.
  Status here: invalid. Both output classes are frame-like and both may contain interior non-background
  structure, so the empty-middle test is neither sufficient nor the intended grouping rule.
- Fixed output geometry `outH = 4 * ...`, `outW = 8`, and row steps of `i * 4`
  Original assumption: both output columns use the same 4x4 footprint.
  Status here: invalid. The left group is rotated, so its pasted footprint is (w,h),
  while the right group remains (h,w).
- Implicit assumption that classification is independent from transformation
  Status here: invalid. The very property used to classify an object determines whether its
  width and height swap in the output.

d) Placement legality
---------------------
- The generator uses `random_free_location(..., border=1, padding=1, padding_connectivity=4)`.
  That prevents overlap, out-of-bounds placement, and accidental touching between components.
- All generated objects are connected under 4-connectivity:
  the symmetric frame-like spine touches the outer frame, and the asymmetric corner hook
  is attached directly to the frame border.
- Canvas dimensions are sized from the rectangle shape and object counts, leaving ample room.
- The generator accepts a sample only if `main(input)` exactly matches the explicitly built output.

e) Visual observability
-----------------------
- The D3 change is obvious because every object is visibly rectangular rather than square.
- The D1 change is human-visible because one class has a centered balanced structure and the other
  has a clear corner-biased hook, even though both classes are otherwise the same rectangular frame motif.
- The D16 change is human-visible because the left-column rectangles change orientation:
  their tall/wide aspect flips, and the centered spine changes from vertical to horizontal.
- No hidden numeric property such as color id or generation order is used for sorting or classification.

f) Visual cleanliness
---------------------
- Objects do not overlap or touch, so each component is easy to isolate by eye.
- The symmetric class always uses one centered spine, and the asymmetric class always uses one small
  corner hook. This keeps the motifs simple.
- Output remains a tidy two-column stack, just with different column widths caused by rotation.

g) Change inferability and anti-degeneracy
------------------------------------------
- If generated frames had an entirely empty interior, a simpler "look for a hollow rectangle" solver
  would be enough. To avoid that degeneration, every symmetric object includes a central spine and
  every asymmetric object includes a corner hook, while both remain frames.
- The key anti-degeneracy metrics used after implementation are:
  1. every accepted sample has non-square objects;
  2. every accepted sample has at least one symmetric and one asymmetric object;
  3. every symmetric object changes shape after rotation (`rotated.shape != original.shape`);
  4. the fused solver output differs from a "group by hollow-frame only" baseline on essentially all samples.

h) New main() assumptions and generator guarantees
--------------------------------------------------
- Assumption: relevant objects are isolated connected components on the background.
  Guarantee: generator places them with 1-cell padding.
- Assumption: all relevant objects in one sample share a single rectangle size.
  Guarantee: generator samples one size per grid.
- Assumption: left-group objects have at least one reflection symmetry axis; right-group objects do not.
  Guarantee: generator constructs only centered-spine symmetric frames and corner-hook asymmetric frames.
- Assumption: there is at least one object in each class.
  Guarantee: generator samples counts from 1..4 for both classes.
- Assumption: the output should stack each column independently from the top.
  Guarantee: expected outputs are built with that exact convention and validated against `main()`.
"""


RECT_SIZES: tuple[tuple[int, int], ...] = (
    (5, 7),
    (7, 5),
    (5, 9),
    (9, 5),
)

ASYM_CORNERS: tuple[str, ...] = ("tl", "tr", "bl", "br")


def _has_reflection_symmetry(sub: np.ndarray) -> bool:
    return bool(np.array_equal(sub, np.flipud(sub)) or np.array_equal(sub, np.fliplr(sub)))


def _collect_objects(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    symmetric: list[tuple[int, int, np.ndarray]] = []
    asymmetric: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        sub = crop_bbox(g, comp["bbox"])
        h, w = map(int, sub.shape)
        if h < 5 or w < 5:
            continue
        if _has_reflection_symmetry(sub):
            symmetric.append((int(r0), int(c0), sub))
        else:
            asymmetric.append((int(r0), int(c0), sub))

    symmetric.sort(key=lambda t: (int(t[0]), int(t[1])))
    asymmetric.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, symmetric, asymmetric


def _build_output(
    bg: int,
    symmetric: list[tuple[int, int, np.ndarray]],
    asymmetric: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    if not symmetric or not asymmetric:
        raise ValueError("expected at least one symmetric and one asymmetric object")

    frame_seq = [np.rot90(sub, k=1) for _r0, _c0, sub in symmetric]
    solid_seq = [sub.copy() for _r0, _c0, sub in asymmetric]

    left_w = max(int(a.shape[1]) for a in frame_seq)
    right_w = max(int(a.shape[1]) for a in solid_seq)
    left_h_total = sum(int(a.shape[0]) for a in frame_seq)
    right_h_total = sum(int(a.shape[0]) for a in solid_seq)

    out_h = max(int(left_h_total), int(right_h_total))
    out_w = int(left_w + right_w)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    rr = 0
    for a in frame_seq:
        paste(out, a, int(rr), 0)
        rr += int(a.shape[0])

    rr = 0
    for a in solid_seq:
        paste(out, a, int(rr), int(left_w))
        rr += int(a.shape[0])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group non-square rectangles by reflection symmetry, then rotate the symmetric group.

    - Symmetric objects go to the left column after a 90-degree CCW rotation.
    - Asymmetric objects go to the right column unchanged.
    - Both groups are sorted by input row, then input col.
    - Because the left group is rotated, output width is `rotated_width + original_width`,
      and total height is the taller of the two stacked columns.
    """
    g = np.array(grid, dtype=int)
    bg, symmetric, asymmetric = _collect_objects(g)
    return _build_output(bg, symmetric, asymmetric)


def _make_symmetric_frame(outer: int, accent: int, h: int, w: int, bg: int) -> np.ndarray:
    a = np.full((h, w), int(outer), dtype=int)
    a[1:-1, 1:-1] = int(bg)
    center_c = int(w // 2)
    a[:, center_c] = int(accent)
    return a


def _make_asymmetric_frame(outer: int, accent: int, h: int, w: int, bg: int, corner: str) -> np.ndarray:
    a = np.full((h, w), int(outer), dtype=int)
    a[1:-1, 1:-1] = int(bg)

    if corner == "tl":
        cells = ((0, 1), (1, 1), (2, 1), (2, 2))
    elif corner == "tr":
        cells = ((0, w - 2), (1, w - 2), (2, w - 2), (2, w - 3))
    elif corner == "bl":
        cells = ((h - 1, 1), (h - 2, 1), (h - 3, 1), (h - 3, 2))
    else:
        cells = ((h - 1, w - 2), (h - 2, w - 2), (h - 3, w - 2), (h - 3, w - 3))

    for r, c in cells:
        a[int(r), int(c)] = int(accent)
    return a


def _pick_accent(avoid: set[int]) -> int:
    choices = [c for c in range(1, 10) if int(c) not in {int(x) for x in avoid}]
    return int(choices[int(np.random.randint(0, len(choices)))])


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h, w = RECT_SIZES[int(np.random.randint(0, len(RECT_SIZES)))]
    n_symmetric = int(np.random.randint(1, 5))
    n_asymmetric = int(np.random.randint(1, 5))

    total = int(n_symmetric + n_asymmetric)
    outer_colors = [int(x) for x in np.random.permutation(np.arange(1, 10))[:total]]

    min_h = max(24, h * total + 12)
    min_w = max(24, w * 3 + 12)
    canvas_h = int(np.random.randint(min_h, min_h + 8))
    canvas_w = int(np.random.randint(min_w, min_w + 8))
    g = np.full((canvas_h, canvas_w), int(bg), dtype=int)

    try:
        placed_symmetric: list[tuple[int, int, np.ndarray]] = []
        placed_asymmetric: list[tuple[int, int, np.ndarray]] = []

        for idx in range(n_symmetric):
            outer = int(outer_colors[idx])
            accent = _pick_accent({bg, outer})
            patch = _make_symmetric_frame(outer, accent, h, w, bg)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)
            placed_symmetric.append((int(r0), int(c0), patch))

        for idx in range(n_asymmetric):
            outer = int(outer_colors[n_symmetric + idx])
            accent = _pick_accent({bg, outer})
            corner = str(ASYM_CORNERS[int(np.random.randint(0, len(ASYM_CORNERS)))])
            patch = _make_asymmetric_frame(outer, accent, h, w, bg, corner)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)
            placed_asymmetric.append((int(r0), int(c0), patch))

        placed_symmetric.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_asymmetric.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _build_output(bg, placed_symmetric, placed_asymmetric)
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
