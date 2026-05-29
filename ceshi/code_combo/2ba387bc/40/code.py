# Combo variant of 2ba387bc (index 40)
# combo_spec: D3/1+D5/1+D16/0
# Name: 矩形方块+框架旋转+分隔列

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D3/1 + D5/1 + D16/0
==============================================

This combo starts from original task 2ba387bc and fuses three changes:
1. D3/1: source objects are same-size rectangles instead of fixed 4x4 squares.
2. D5/1: output has a 1-cell separator column between the two groups.
3. D16/0: frames rotate 90 degrees before being placed in the left column.

a) Concrete behavior and trace
------------------------------
- Original task:
  every object is 4x4; frames and solids keep their shape; output is two adjacent 4-cell columns.
- Combo task:
  every object in one sample shares one non-square size (h, w) with h > w >= 3.
  Frames are hollow h x w rectangles; solids are filled h x w rectangles.
  In the output, each frame is rotated to w x h, while each solid stays h x w.
  Because of that rotation, the left column width becomes h and the right column width stays w.
  A 1-cell background separator sits between them, so total width is h + 1 + w.

Trace on a concrete sample:
- Suppose source objects are size (6, 4). This is the code's row-major form of the prose's "4x6" rectangle.
- Original solver assumptions fail immediately:
  it only accepts 4x4 bboxes, checks a hard-coded 2x2 interior, and always emits width 8.
- Combo solver instead:
  1. detects same-size rectangular components,
  2. classifies a frame by "single-color border + all-bg interior",
  3. sorts frames and solids by (input row, input col),
  4. rotates each frame to shape (4, 6),
  5. pastes frames down the left column using step 4,
  6. pastes solids down the right column using step 6,
  7. leaves column 6 as background separator.
- Result:
  left column width = 6, separator width = 1, right column width = 4, total output width = 11.
  The two columns can also have different total heights because rotated frames are shorter/taller than solids.

b) Complexity calibration
-------------------------
- The rectangle size is chosen from a small visible set and is always non-square.
  This makes the D3 change obvious while keeping inference manageable.
- All objects inside one sample share the same size. This avoids introducing a second hidden rule
  about mixed sizes and keeps the task focused on classification + rotation + layout.
- The generator enforces at least one frame and one solid, so both columns are always present.
- Using only h > w makes the rotation effect visually stable: the left column is always wider than the right.

c) Implicit assumption audit of original main()
-----------------------------------------------
- `if bbox_h(b) != 4 or bbox_w(b) != 4: continue`
  Original assumption: every relevant object is exactly 4x4.
  Combo impact: invalid. Replaced with rectangle-aware checks and a sample-level same-size check.
- `sub[1:3, 1:3] == bg`
  Original assumption: the frame hole is always 2x2.
  Combo impact: invalid. Replaced with `sub[1:-1, 1:-1] == bg`.
- `outH = 4 * max(...)`, `outW = 8`, `paste(..., i * 4, ...)`
  Original assumption: all pasted objects have the same 4x4 shape in output.
  Combo impact: invalid twice:
  solids still use source height h, but rotated frames use output height w.
  The new build step therefore uses independent row strides for the two columns,
  output height `max(len(frames)*w, len(solids)*h)`, and output width `h + 1 + w`.
- `border` uniformity logic remains valid for rectangles and survives the fusion.

d) Placement legality
---------------------
- Generator only uses sizes with h > w >= 3, so a 1-cell border always leaves non-empty interior.
- Objects are scattered with `random_free_location(..., border=1, padding=1, padding_connectivity=4)`,
  which prevents overlap, keeps them in bounds, and avoids accidental component merging.
- Canvas size scales with object count and rectangle size, so placement stays feasible.
- Each accepted sample is checked by comparing `main(input)` against a directly built expected output.

e) Visual observability
-----------------------
- Non-square source objects make the D3 change visible from the input alone.
- In the output, frames clearly change aspect ratio while solids do not, so D16/0 is visible.
- The separator column is a persistent empty stripe, so D5/1 is also visible.
- Because there is at least one object of each type in every sample, a human can compare the two columns
  within a single example instead of needing hidden color knowledge.

f) Visual cleanliness
---------------------
- Objects remain isolated in the input by padding, so the scene is easy to segment.
- Output uses only two aligned stacks plus one separator stripe; no overlays or shared regions occur.
- Restricting to a few rectangle sizes avoids chaotic aspect-ratio variation across objects in one sample.

g) Change inferability and color audit
--------------------------------------
- No color remapping is introduced.
- Colors only identify individual objects; ordering is determined by visible positions only.
- The fusion rule is structural: object type controls rotation, and column identity controls placement.
  There is no hidden dependency on numeric color ids.

h) Quantitative anti-degeneracy checks
--------------------------------------
- Metric 1: `source_h != source_w` for every sample. Expected: 100%.
- Metric 2: `rotated_frame_width - solid_width = source_h - source_w`. Expected: always > 0.
- Metric 3: combo output differs from a "same rectangles + separator but no frame rotation" baseline.
  Expected: 100%, because all samples contain at least one non-square frame.

i) New main() assumptions and generator guarantees
--------------------------------------------------
- Assumption: every relevant object is an axis-aligned rectangle with height/width >= 3.
  Guarantee: generator only creates such objects.
- Assumption: all relevant objects in one sample share one source size (h, w).
  Guarantee: generator samples exactly one rectangle size per grid.
- Assumption: every accepted object is either a monochrome solid rectangle or a one-color frame with bg interior.
  Guarantee: generator constructs exactly those two patterns.
- Assumption: at least one frame and one solid exist, so both columns and the separator are meaningful.
  Guarantee: generator samples counts from 1..4 for each group.
- Assumption: different objects do not touch orthogonally.
  Guarantee: placement uses 4-connectivity padding of 1.
"""


RECT_SIZES: tuple[tuple[int, int], ...] = (
    (4, 3),
    (5, 3),
    (6, 4),
)

SEPARATOR_W = 1


def _is_frame_rect(sub: np.ndarray, bg: int) -> bool:
    h, w = map(int, sub.shape)
    if h < 3 or w < 3:
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.flatten()))
    if len(bcols) != 1:
        return False
    border_col = int(next(iter(bcols)))
    if border_col == int(bg):
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


def _shared_source_size(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> tuple[int, int]:
    objs = [sub for _r0, _c0, sub in frames + solids]
    if not objs:
        raise ValueError("no valid rectangular objects found")

    src_h = int(objs[0].shape[0])
    src_w = int(objs[0].shape[1])
    if src_h < 3 or src_w < 3:
        raise ValueError("objects must be at least 3x3")

    for sub in objs[1:]:
        if sub.shape != (src_h, src_w):
            raise ValueError("mixed object sizes are not supported in this combo variant")
    return src_h, src_w


def _build_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    src_h, src_w = _shared_source_size(frames, solids)
    frame_out_h = src_w
    frame_out_w = src_h
    solid_out_h = src_h
    solid_out_w = src_w

    out_h = max(int(len(frames)) * frame_out_h, int(len(solids)) * solid_out_h)
    out_w = frame_out_w + SEPARATOR_W + solid_out_w
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, np.rot90(sub, k=1), int(i * frame_out_h), 0)
    solid_col = frame_out_w + SEPARATOR_W
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * solid_out_h), int(solid_col))

    return out


def _build_no_rotation_baseline(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    src_h, src_w = _shared_source_size(frames, solids)
    out_h = src_h * max(int(len(frames)), int(len(solids)))
    out_w = src_w * 2 + SEPARATOR_W
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * src_h), 0)
    solid_col = src_w + SEPARATOR_W
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * src_h), int(solid_col))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered same-size rectangles into two separated columns by type.

    - Frames: one-color rectangular border with background interior.
    - Solids: one-color filled rectangles.
    - Both groups are sorted by input row, then input col.
    - Frames rotate 90 degrees before placement in the left column.
    - Solids keep source orientation in the right column.
    - A 1-cell background separator sits between the columns.
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


def _sample_unique_color(used_cols: set[int]) -> int:
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

    min_h = max(24, h * (n_frames + n_solids) + 10)
    min_w = max(24, (h + w) * 2 + 8)
    canvas_h = int(np.random.randint(min_h, min_h + 8))
    canvas_w = int(np.random.randint(min_w, min_w + 8))
    g = np.full((canvas_h, canvas_w), bg, dtype=int)

    try:
        used_cols: set[int] = set()
        placed_frames: list[tuple[int, int, np.ndarray]] = []
        placed_solids: list[tuple[int, int, np.ndarray]] = []

        for _ in range(n_frames):
            patch = _make_frame(_sample_unique_color(used_cols), h, w, bg)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)
            placed_frames.append((int(r0), int(c0), patch))

        for _ in range(n_solids):
            patch = _make_solid(_sample_unique_color(used_cols), h, w)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)
            placed_solids.append((int(r0), int(c0), patch))

        placed_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _build_output(bg, placed_frames, placed_solids)
        if not np.array_equal(main(g), expected):
            return None
        if np.array_equal(expected, _build_no_rotation_baseline(bg, placed_frames, placed_solids)):
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
