# Combo variant of 2ba387bc (index 37)
# combo_spec: D1/2+D3/0+D11/0
# Name: 面积分类+可变尺寸+非标干扰

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVERS_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D1/2 + D3/0 + D11/0
==============================================

a) Concrete behavior and trace
   - Original task:
     * only isolated 4x4 objects exist;
     * frame vs solid is detected from the 2x2 interior;
     * every accepted object is stacked in a fixed 4-row band;
     * no distractors exist.
   - Fused variant:
     * valid objects are monochrome squares with side length n in {3, 4, 5};
     * two semantic families still exist, but they are recognized through pixel area:
       frame-like squares have area 4n-4 and solid squares have area n^2;
     * non-valid distractors are also present and may touch valid squares with a
       different color, so component extraction must be monochromatic;
     * output keeps the original two-column layout idea, but column width is the
       maximum valid square size and row-band heights are paired maxima.
   - Small trace:
     * Suppose the input contains a 5x5 frame at row 3, a 4x4 solid at row 10, and
       an attached 2x8 bar touching the frame. The frame and the solid both have area
       16, but they belong to different columns because their sizes differ:
       (size, area) = (5, 16) means frame-family, while (4, 16) means solid-family.
     * The original solver fails twice: it merges the attached bar with the frame when
       using multicolor components, and it also hard-codes size 4.
     * The fused solver first isolates monochrome components, filters to square bboxes
       in 3..5, then uses size-aware area formulas to classify them.

b) Complexity calibration
   - Sizes are restricted to 3, 4, 5 only. This is enough to visibly differ from the
     original task without exploding the hypothesis space.
   - The generator always includes one 5x5 frame and one 4x4 solid, so same-area=16
     objects appear in opposite groups. This prevents a degenerate dataset where area
     alone without size would still work.
   - Distractors come from a small rectangle vocabulary, including some with areas
     equal to valid object areas such as 8, 9, 12, or 16. This forces "size + area"
     filtering rather than simple area counting.

c) Hidden-assumption audit of the original main()
   - `find_components(... monochromatic=False)` is invalid now because attached
     distractors can merge with a valid square into one multicolor blob.
   - `bbox_h(b) != 4 or bbox_w(b) != 4` is invalid now because valid square sizes are
     3..5, not fixed at 4.
   - `sub[1:3, 1:3] == bg` is invalid now because the interior slice depends on n and,
     more importantly, the combo rule is area-based rather than direct interior testing.
   - `else: solids.append(...)` is unsafe now because distractors can have arbitrary
     shapes; every accepted object must explicitly satisfy one of the valid
     (size, area, local-pattern) combinations.
   - `outH = 4 * max(...)`, `outW = 8`, and `i * 4` are invalid now because object
     sizes vary and output geometry must follow the largest square and paired band
     heights.
   - New combo-specific conflict:
     area 16 is ambiguous across the two semantic families. Therefore the correct
     classifier cannot be "area only"; it must be "area interpreted under the square's
     size". That is the key non-trivial interaction between D1 and D3.

d) Placement legality
   - The generator reserves each valid square's full bbox in a separate occupancy grid.
     This prevents distractors from being placed inside a frame's background interior.
   - Attached distractors are allowed to touch exactly one target valid square and are
     rejected if they would touch any other reserved object.
   - Free distractors use `random_free_location` on the occupancy grid with padding, so
     they do not overlap or ambiguously cling to unrelated objects.

e) Visual observability
   - Human-visible cues support each needed inference:
     * sizes 3/4/5 are directly visible;
     * frames vs solids differ in colored-pixel count relative to size;
     * distractors are visibly not valid 3..5 squares and never appear in output;
     * output band heights clearly adapt to mixed sizes.
   - The forced (5x5 frame, 4x4 solid) pair makes the same-area=16 ambiguity visible
     in the examples, so a human can infer that raw area alone is not the whole rule.

f) Visual cleanliness
   - Colors are unique per object, which keeps monochromatic components distinct.
   - Distractors are simple filled rectangles, not noisy composites.
   - Valid objects are spaced with padding before distractors are attached, so the grid
     remains readable instead of collapsing into dense clutter.

g) New solver assumptions after modification
   - Every meaningful object is a monochromatic component.
   - Valid output candidates are exactly square components with side length in 3..5.
   - For a valid size n, accepted areas are only 4n-4 (frame-family) or n^2
     (solid-family), and the local crop must match that family.
   - The generator guarantees unique colors, valid square construction, at least one
     attached distractor, and a forced ambiguous-area pair, then round-trips through
     `main()` before returning a sample.
"""


_VALID_SIZES = (3, 4, 5)
_AMBIGUOUS_DISTRACTOR_SHAPES: tuple[tuple[int, int], ...] = (
    (2, 4),   # area 8
    (1, 9),   # area 9
    (2, 6),   # area 12
    (2, 8),   # area 16
)
_DISTRACTOR_SHAPES: tuple[tuple[int, int], ...] = (
    (2, 4),
    (1, 9),
    (2, 6),
    (2, 8),
    (1, 7),
    (2, 5),
    (3, 4),
    (4, 6),
    (5, 6),
    (3, 7),
)


def _frame_area(size: int) -> int:
    return int(4 * size - 4)


def _solid_area(size: int) -> int:
    return int(size * size)


def _make_frame(size: int, col: int, bg: int = 0) -> np.ndarray:
    a = np.full((int(size), int(size)), int(col), dtype=int)
    a[1:-1, 1:-1] = int(bg)
    return a


def _make_solid(size: int, col: int) -> np.ndarray:
    return np.full((int(size), int(size)), int(col), dtype=int)


def _classify_valid_square(sub: np.ndarray, color: int, area: int, bg: int) -> str | None:
    h, w = map(int, sub.shape)
    if h != w or h not in _VALID_SIZES:
        return None

    size = int(h)
    color = int(color)
    area = int(area)
    bg = int(bg)
    if color == bg:
        return None

    if area == _solid_area(size) and bool((sub == color).all()):
        return "solid"

    if area == _frame_area(size):
        border_ok = bool((sub[0, :] == color).all()) and bool((sub[-1, :] == color).all())
        border_ok = border_ok and bool((sub[1:-1, 0] == color).all()) and bool((sub[1:-1, -1] == color).all())
        if border_ok and bool((sub[1:-1, 1:-1] == bg).all()):
            return "frame"

    return None


def _assemble_output(objects: list[dict], bg: int) -> np.ndarray:
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    frames.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))
    solids.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))

    if not frames and not solids:
        return np.zeros((0, 0), dtype=int)

    max_size = max(int(obj["size"]) for obj in objects)
    row_count = max(len(frames), len(solids))
    row_heights = [
        max(
            int(frames[i]["size"]) if i < len(frames) else 0,
            int(solids[i]["size"]) if i < len(solids) else 0,
        )
        for i in range(row_count)
    ]

    out = np.full((int(sum(row_heights)), int(2 * max_size)), int(bg), dtype=int)
    r = 0
    for i, band_h in enumerate(row_heights):
        if i < len(frames):
            paste(out, frames[i]["sprite"], int(r), 0)
        if i < len(solids):
            paste(out, solids[i]["sprite"], int(r), int(max_size))
        r += int(band_h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only valid 3x3..5x5 squares, classify them by size-aware colored area,
    then stack frame-family squares left and solid-family squares right.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    objects: list[dict] = []

    for comp in comps:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        if h != w or h not in _VALID_SIZES:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        kind = _classify_valid_square(sub, int(comp["color"]), int(comp["area"]), bg)
        if kind is None:
            continue

        objects.append(
            {
                "kind": kind,
                "size": int(h),
                "r0": int(r0),
                "c0": int(c0),
                "sprite": sub,
            }
        )

    return _assemble_output(objects, bg)


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)}


def _can_place_attached(
    occ: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> bool:
    h, w = map(int, patch.shape)
    H, W = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    if bool((occ[r0 : r0 + h, c0 : c0 + w] != bg).any()):
        return False

    target_cells = _bbox_cells(target_bbox)
    touched_target = False
    touched_other = False
    for rr in range(r0, r0 + h):
        for cc in range(c0, c0 + w):
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(occ[nr, nc]) == bg:
                    continue
                if (nr, nc) in target_cells:
                    touched_target = True
                else:
                    touched_other = True
    return touched_target and not touched_other


def _attached_location(
    occ: np.ndarray,
    patch: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = target_bbox
    h, w = map(int, patch.shape)
    candidates: list[tuple[int, int]] = []

    if c0 - w >= 0:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((rr, int(c0 - w)))
    if c1 + 1 + w <= occ.shape[1]:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((rr, int(c1 + 1)))
    if r0 - h >= 0:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r0 - h), cc))
    if r1 + 1 + h <= occ.shape[0]:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r1 + 1), cc))

    np.random.shuffle(candidates)
    for rr, cc in candidates:
        if _can_place_attached(occ, patch, int(rr), int(cc), target_bbox, bg):
            return int(rr), int(cc)
    return None


def _place_distractor(
    occ: np.ndarray,
    g: np.ndarray,
    patch: np.ndarray,
    bg: int,
    anchors: list[tuple[int, int, int, int]],
    force_attach: bool,
) -> bool:
    occ_patch = np.full(patch.shape, 1, dtype=int)
    anchor_order = list(range(len(anchors)))
    np.random.shuffle(anchor_order)

    if force_attach:
        for idx in anchor_order:
            loc = _attached_location(occ, occ_patch, anchors[idx], bg)
            if loc is not None:
                paste(occ, occ_patch, loc[0], loc[1], transparent=bg)
                paste(g, patch, loc[0], loc[1], transparent=bg)
                return True
        return False

    if bool(np.random.rand() < 0.5):
        for idx in anchor_order:
            loc = _attached_location(occ, occ_patch, anchors[idx], bg)
            if loc is not None:
                paste(occ, occ_patch, loc[0], loc[1], transparent=bg)
                paste(g, patch, loc[0], loc[1], transparent=bg)
                return True

    try:
        r0, c0 = random_free_location(occ, occ_patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
        paste(occ, occ_patch, int(r0), int(c0), transparent=bg)
        paste(g, patch, int(r0), int(c0), transparent=bg)
        return True
    except Exception:
        return False


def _sample_valid_specs() -> list[dict]:
    n_frames = int(np.random.randint(1, 4))
    n_solids = int(np.random.randint(1, 4))

    frame_sizes = [5]
    solid_sizes = [4]
    for _ in range(n_frames - 1):
        frame_sizes.append(int(np.random.randint(3, 6)))
    for _ in range(n_solids - 1):
        solid_sizes.append(int(np.random.randint(3, 6)))

    all_sizes = frame_sizes + solid_sizes
    if len(all_sizes) >= 3 and 3 not in all_sizes:
        if len(frame_sizes) > 1 and bool(np.random.rand() < 0.5):
            frame_sizes[-1] = 3
        elif len(solid_sizes) > 1:
            solid_sizes[-1] = 3

    specs = [{"kind": "frame", "size": int(size)} for size in frame_sizes]
    specs.extend({"kind": "solid", "size": int(size)} for size in solid_sizes)
    np.random.shuffle(specs)
    return specs


def _sample_distractor_shapes(count: int) -> list[tuple[int, int]]:
    shapes = [_AMBIGUOUS_DISTRACTOR_SHAPES[int(np.random.randint(0, len(_AMBIGUOUS_DISTRACTOR_SHAPES)))]]
    while len(shapes) < count:
        shapes.append(_DISTRACTOR_SHAPES[int(np.random.randint(0, len(_DISTRACTOR_SHAPES)))])
    return shapes


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(28, 41))
    W = int(np.random.randint(28, 41))
    g = np.full((H, W), bg, dtype=int)
    occ = np.full((H, W), bg, dtype=int)

    valid_specs = _sample_valid_specs()
    n_valid = len(valid_specs)
    n_distractors = int(np.random.randint(2, min(4, 10 - n_valid)))
    total = n_valid + n_distractors
    colors = list(np.random.permutation(np.arange(1, 10))[:total].astype(int))
    valid_colors = colors[:n_valid]
    distractor_colors = colors[n_valid:]

    objects: list[dict] = []
    anchor_bboxes: list[tuple[int, int, int, int]] = []

    try:
        for spec, color in zip(valid_specs, valid_colors):
            size = int(spec["size"])
            sprite = _make_frame(size, int(color), bg) if spec["kind"] == "frame" else _make_solid(size, int(color))
            reserve = np.full((size, size), 1, dtype=int)
            r0, c0 = random_free_location(occ, reserve, transparent=bg, border=1, padding=2, padding_connectivity=4)
            paste(occ, reserve, int(r0), int(c0), transparent=bg)
            paste(g, sprite, int(r0), int(c0), transparent=bg)
            anchor_bboxes.append((int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)))
            objects.append(
                {
                    "kind": spec["kind"],
                    "size": size,
                    "r0": int(r0),
                    "c0": int(c0),
                    "sprite": sprite,
                }
            )

        attached_count = 0
        distractor_shapes = _sample_distractor_shapes(n_distractors)
        valid_area_like_count = 0
        for i, (shape, color) in enumerate(zip(distractor_shapes, distractor_colors)):
            patch = np.full(shape, int(color), dtype=int)
            if int(np.prod(shape)) in {_frame_area(3), _solid_area(3), _frame_area(4), _solid_area(4), _frame_area(5)}:
                valid_area_like_count += 1
            force_attach = i == 0
            ok = _place_distractor(occ, g, patch, bg, anchor_bboxes, force_attach=force_attach)
            if not ok:
                return None
            if force_attach:
                attached_count += 1

        frame_rows = [int(obj["r0"]) for obj in objects if obj["kind"] == "frame"]
        solid_rows = [int(obj["r0"]) for obj in objects if obj["kind"] == "solid"]
        if len(frame_rows) != len(set(frame_rows)) or len(solid_rows) != len(set(solid_rows)):
            return None

        size_area_pairs = {(int(obj["size"]), int(np.count_nonzero(obj["sprite"] != bg)), obj["kind"]) for obj in objects}
        if (5, 16, "frame") not in size_area_pairs or (4, 16, "solid") not in size_area_pairs:
            return None
        if attached_count == 0 or valid_area_like_count == 0:
            return None

        expected = _assemble_output(objects, bg)
        actual = main(g)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
