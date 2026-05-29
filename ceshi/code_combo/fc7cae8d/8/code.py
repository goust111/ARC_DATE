# Combo variant of fc7cae8d (index 8)
# combo_spec: D1/0+D5/0
# Name: 形状嵌合+保留源
#
# Design analysis for the fused variant
# ------------------------------------
# 1. Concrete behavior and trace:
#    Original fc7cae8d:
#    - detect two almost-solid frame control sides by color,
#    - find the interior object whose two guide edges use those same two colors,
#    - rotate/flip so guide colors face the matching frame sides,
#    - crop away the matched 1px guide strips,
#    - output only the cropped core rectangle.
#
#    Fused D1/0 + D5/0:
#    - the interior source object still has two 1px guide layers, but each guide side is now a jagged
#      silhouette made from one guide color plus bg holes, so colors no longer identify orientation;
#    - each active frame side is still almost solid on the outer border, but the adjacent inner border
#      line carries a same-colored protrusion mask;
#    - solve orientation by shape complement: after one dihedral transform, the object's bg-hole mask
#      on each active side must equal the frame protrusion mask on that side;
#    - crop those matched guide layers away to recover the aligned core;
#    - keep the source object in place and paste one extra copy of the aligned core into the matching
#      control corner while preserving full canvas size.
#
#    Small trace on a top+left example:
#    - old rule: "top guide is ctrl_h color, left guide is ctrl_v color, so crop and return the core".
#    - fused rule: "top border and left border are the active controls, but the source object's two
#      active sides share one guide color; orientation is instead the unique transform whose top-hole
#      mask equals the top protrusion comb and whose left-hole mask equals the left protrusion comb.
#      After cropping, paste that core at the top-left interior control corner."
#    - integration detail: because D1 uses the second row/column for protrusions, the D5 copy target
#      is shifted one extra cell inward to `(2,2)` / `(2,W-w-2)` / `(H-h-2,2)` / `(H-h-2,W-w-2)`.
#      That avoids overwriting the very signatures needed to read the puzzle.
#
# 2. Complexity calibration:
#    - Both guide masks are non-palindromic and contain both holes and solids, so shape matching is
#      visually salient and not reducible to a plain colored strip.
#    - The core is forced asymmetric and the source is generated via a non-identity transform, so the
#      keep-source copy is visibly a transformed duplicate instead of a trivial same-looking stamp.
#    - The generator rejects cases with multiple shape-matching transforms, so the puzzle has a unique
#      intended orientation.
#
# 3. Hidden assumptions audited from the original solver:
#    - Original `guide_edges()` assumed each active side was almost monochrome. That fails here because
#      each guide side mixes guide color and bg holes. The fused solver therefore reads hole masks.
#    - Original output shape was the cropped core only. That fails under D5, so `main()` now preserves
#      the full grid and pastes an extra transformed core copy.
#    - Naively reusing the D5 corner target `(1,1)` would overwrite D1 protrusions on row/col 1; the
#      fused solver and generator both shift the copy target inward by one more cell.
#    - `extract_center_object()` assumes the interior source is the largest non-border component. The
#      generator preserves this by reserving the control corner and keeping a blank moat to the border.
#
# 4. Placement legality:
#    - The source object contains bg holes, so placement treats 0 as transparent when collision-testing.
#    - The control corner copy region and a moat around it are reserved before the source is placed.
#    - The frame protrusion masks are kept away from the literal corner, so the two control colors never
#      overwrite each other and the copied core does not overlap the protrusion lane.
#
# 5. Visual observability and cleanliness:
#    - Humans can see one jagged source object in the middle and one clean unguided core copy in the
#      control corner. The original source stays put, making the "copy + transform" behavior explicit.
#    - The protrusion lane remains visible in the output because the corner copy is offset inward; this
#      avoids stacking the copied core on top of the control signatures.
#
# 6. Color inference:
#    - Colors only identify which two frame sides are active. They do not identify which object side
#      matches which control because both object guide sides share one distinct guide color.
#    - The palette is kept small: two control colors, one guide color, one fill color, and 2-4 pattern
#      colors. This keeps the shape cue readable from a few examples.
#
# 7. New solver assumptions and generator guarantees:
#    - `_frame_signature()` assumes each active frame side has a contiguous signature span whose first
#      and last cells are present; the generator enforces that in `_random_signature()`.
#    - `_object_signatures()` assumes only the two active guide sides contain bg holes; the core is a
#      solid rectangle with colored texture but no bg holes, so inactive sides stay plain.
#    - `_matching_transforms()` assumes the correct shape match is unique; `_gen_attempt()` rejects any
#      sample where the aligned object admits more than one satisfying dihedral transform.
#    - `_corner_core_location()` assumes the control corner region of size `core.shape` is empty; the
#      generator reserves that exact footprint plus a moat before placing the source object.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[4] / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    H, W = arr.shape
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, line in sides.items():
        vals = list(map(int, line))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(color), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, color) for side, color, _cnt, _L in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1):
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy(), (0, 0, H - 1, W - 1)
    return crop_bbox(arr, best[1]), best[1]


def _frame_signature(arr: np.ndarray, side: str, color: int) -> np.ndarray | None:
    H, W = arr.shape
    if side == "top":
        line = arr[1, 1:-1]
    elif side == "bottom":
        line = arr[H - 2, 1:-1]
    elif side == "left":
        line = arr[1:-1, 1]
    else:
        line = arr[1:-1, W - 2]

    mask = np.array(line == int(color), dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return None
    lo, hi = int(idx[0]), int(idx[-1])
    sig = mask[lo : hi + 1]
    if int(sig.sum()) < 3 or int(sig.sum()) == len(sig):
        return None
    return sig


def _object_side_signature(arr: np.ndarray, side: str, bg: int) -> np.ndarray | None:
    if side == "top":
        edge = arr[0, :]
    elif side == "bottom":
        edge = arr[-1, :]
    elif side == "left":
        edge = arr[:, 0]
    else:
        edge = arr[:, -1]

    holes = np.array(edge == int(bg), dtype=bool)
    if int(holes.sum()) < 2 or int(holes.sum()) >= len(holes):
        return None
    if int((~holes).sum()) < 2:
        return None
    return holes


def _object_signatures(arr: np.ndarray, bg: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side in ("top", "bottom", "left", "right"):
        sig = _object_side_signature(arr, side, bg)
        if sig is not None:
            out[side] = sig
    return out


def _matching_transforms(
    obj: np.ndarray, desired_shapes: dict[str, np.ndarray], bg: int
) -> list[tuple[int, bool, np.ndarray]]:
    matches: list[tuple[int, bool, np.ndarray]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            got = _object_signatures(t, bg)
            if len(got) != len(desired_shapes):
                continue
            ok = True
            for side, sig in desired_shapes.items():
                cur = got.get(side)
                if cur is None or len(cur) != len(sig) or not np.array_equal(cur, sig):
                    ok = False
                    break
            if ok:
                matches.append((int(k), bool(flip), t))
    return matches


def _crop_aligned_core(aligned_obj: np.ndarray, desired_shapes: dict[str, np.ndarray]) -> np.ndarray:
    top = 1 if "top" in desired_shapes else 0
    bottom = int(aligned_obj.shape[0]) - (1 if "bottom" in desired_shapes else 0)
    left = 1 if "left" in desired_shapes else 0
    right = int(aligned_obj.shape[1]) - (1 if "right" in desired_shapes else 0)
    return aligned_obj[int(top) : int(bottom), int(left) : int(right)]


def _corner_core_location(
    grid_shape: tuple[int, int], core_shape: tuple[int, int], desired_sides: dict[str, object]
) -> tuple[int, int]:
    H, W = map(int, grid_shape)
    h, w = map(int, core_shape)
    r0 = 2 if "top" in desired_sides else H - h - 2
    c0 = 2 if "left" in desired_sides else W - w - 2
    return int(r0), int(c0)


def _plain_color_guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, line in edges.items():
        vals = list(map(int, line))
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        color, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(color)
    return out


def _color_only_matching_transforms(obj: np.ndarray, desired_ctrls: dict[str, int]) -> list[tuple[int, bool]]:
    matches: list[tuple[int, bool]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            gedges = _plain_color_guide_edges(t)
            if all(side in gedges and int(gedges[side]) == int(color) for side, color in desired_ctrls.items()):
                matches.append((int(k), bool(flip)))
    return matches


def _bbox_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return bool(a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _fusion_diagnostics(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrls = _detect_controls_frame(g, bg)
    desired_ctrls = {side: color for side, color in ctrls}
    desired_shapes: dict[str, np.ndarray] = {}
    for side, color in ctrls:
        sig = _frame_signature(g, side, color)
        if sig is not None:
            desired_shapes[side] = sig

    obj, source_bbox = _extract_center_object(g, bg)
    shape_matches = _matching_transforms(obj, desired_shapes, bg)
    color_matches = _color_only_matching_transforms(obj, desired_ctrls)

    core = None
    target_bbox = None
    non_identity = False
    if shape_matches:
        k, flip, aligned = shape_matches[0]
        non_identity = not (int(k) == 0 and not bool(flip))
        core = _crop_aligned_core(aligned, desired_shapes)
        tr, tc = _corner_core_location(g.shape, core.shape, desired_shapes)
        target_bbox = (int(tr), int(tc), int(tr + core.shape[0] - 1), int(tc + core.shape[1] - 1))

    return {
        "shape_match_count": int(len(shape_matches)),
        "color_match_count": int(len(color_matches)),
        "non_identity_match": bool(non_identity),
        "source_bbox": source_bbox,
        "target_bbox": target_bbox,
        "source_target_disjoint": bool(
            target_bbox is not None and _bbox_disjoint(tuple(map(int, source_bbox)), tuple(map(int, target_bbox)))
        ),
        "desired_sides": tuple(sorted(desired_shapes)),
        "core_shape": None if core is None else tuple(map(int, core.shape)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the fused shape-match + keep-source variant.

    Steps:
    1. Detect the two active frame sides from their almost-solid outer-border colors.
    2. Read the inner protrusion masks on those two sides.
    3. Find the unique dihedral transform whose object-side hole masks equal those protrusion masks.
    4. Crop away the matched 1px guide layers to recover the aligned core.
    5. Keep the original canvas and source object, and paste one extra core copy into the control corner.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired_shapes: dict[str, np.ndarray] = {}
    for side, color in ctrls:
        sig = _frame_signature(g, side, color)
        if sig is not None:
            desired_shapes[side] = sig
    if len(desired_shapes) != 2:
        return g.copy()

    obj, _bbox = _extract_center_object(g, bg)
    matches = _matching_transforms(obj, desired_shapes, bg)
    if not matches:
        return g.copy()

    aligned = matches[0][2]
    core = _crop_aligned_core(aligned, desired_shapes)

    out = g.copy()
    tr, tc = _corner_core_location(out.shape, core.shape, desired_shapes)
    paste(out, core, tr, tc)
    return out


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors) + 5, int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors) + 4, area // 8), max_nonfill)

    seed_cells = [
        (0, 0),
        (h - 1, max(0, w // 2 - 1)),
        (max(0, h // 2 - 1), w - 1),
        (min(h - 1, 1), min(w - 1, 2)),
        (min(h - 1, 2), max(0, w // 3)),
    ]
    for idx, (r, c) in enumerate(seed_cells):
        core[int(r), int(c)] = int(pattern_colors[idx % len(pattern_colors)])

    while int(np.count_nonzero(core != int(fill))) < target_nonfill:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(["dot", "line", "rect", "elbow"])
        if shape_kind == "dot":
            cells = [(random.randrange(h), random.randrange(w))]
        elif shape_kind == "line":
            line_len = random.randint(2, min(5, max(h, w)))
            if random.random() < 0.5:
                rr = random.randrange(h)
                c0 = random.randrange(w - line_len + 1)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                r0 = random.randrange(h - line_len + 1)
                cc = random.randrange(w)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        elif shape_kind == "elbow":
            r0 = random.randrange(max(1, h - 2))
            c0 = random.randrange(max(1, w - 2))
            cells = [(r0, c0), (r0 + 1, c0), (r0 + 1, c0 + 1)]
        else:
            mh = random.randint(1, min(3, h))
            mw = random.randint(1, min(3, w))
            r0 = random.randrange(h - mh + 1)
            c0 = random.randrange(w - mw + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        for r, c in cells:
            if core[r, c] == int(fill):
                core[r, c] = int(color)
            if int(np.count_nonzero(core != int(fill))) >= target_nonfill:
                break

    return core


def _ensure_asymmetry(core: np.ndarray, fill: int, pattern_colors: list[int]) -> np.ndarray:
    out = core.copy()
    non_identity = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
    for _ in range(32):
        symmetric = False
        for k, flip in non_identity:
            t = _transform(out, k, flip)
            if t.shape == out.shape and np.array_equal(t, out):
                symmetric = True
                break
        if not symmetric:
            return out
        fill_cells = np.argwhere(out == int(fill))
        if fill_cells.size == 0:
            break
        r, c = map(int, fill_cells[random.randrange(len(fill_cells))])
        out[r, c] = int(random.choice(pattern_colors))
    return out


def _random_signature(length: int) -> np.ndarray:
    while True:
        sig = np.zeros(int(length), dtype=bool)
        sig[0] = True
        sig[-1] = True
        interior = list(range(1, length - 1))
        if len(interior) < 3:
            continue
        extra_n = random.randint(2, min(len(interior) - 1, max(3, length // 2)))
        for idx in random.sample(interior, extra_n):
            sig[int(idx)] = True
        if int(sig.sum()) < 4:
            continue
        if int((~sig).sum()) < 2:
            continue
        if np.array_equal(sig, sig[::-1]):
            continue
        return sig


def _nondegenerate_pair(sig_a: np.ndarray, sig_b: np.ndarray) -> bool:
    if np.array_equal(sig_a, sig_b):
        return False
    if len(sig_a) == len(sig_b) and np.array_equal(sig_a, sig_b[::-1]):
        return False
    return True


def _attach_shape_guides(
    core: np.ndarray,
    guide_sides: tuple[str, str],
    guide_color: int,
    guide_shapes: dict[str, np.ndarray],
) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[0, :] = int(guide_color)
        out[0, np.array(guide_shapes["top"], dtype=bool)] = 0
    if has_bottom:
        out[-1, :] = int(guide_color)
        out[-1, np.array(guide_shapes["bottom"], dtype=bool)] = 0
    if has_left:
        out[:, 0] = int(guide_color)
        out[np.array(guide_shapes["left"], dtype=bool), 0] = 0
    if has_right:
        out[:, -1] = int(guide_color)
        out[np.array(guide_shapes["right"], dtype=bool), -1] = 0
    return out


def _paint_frame(
    grid: np.ndarray,
    horiz_side: str,
    vert_side: str,
    ctrl_h: int,
    ctrl_v: int,
    horiz_sig: np.ndarray,
    vert_sig: np.ndarray,
) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
        grid[1, 1:-1] = 0
    else:
        grid[-1, :] = int(ctrl_h)
        grid[-2, 1:-1] = 0
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
        grid[1:-1, 1] = 0
    else:
        grid[:, -1] = int(ctrl_v)
        grid[1:-1, -2] = 0

    c_start_min = 2
    c_start_max = n - len(horiz_sig) - 2
    r_start_min = 2
    r_start_max = n - len(vert_sig) - 2
    if c_start_min > c_start_max or r_start_min > r_start_max:
        raise ValueError("frame too small for signatures")

    c0 = random.randint(c_start_min, c_start_max)
    r0 = random.randint(r_start_min, r_start_max)

    if horiz_side == "top":
        grid[1, c0 : c0 + len(horiz_sig)][horiz_sig] = int(ctrl_h)
    else:
        grid[-2, c0 : c0 + len(horiz_sig)][horiz_sig] = int(ctrl_h)

    if vert_side == "left":
        grid[r0 : r0 + len(vert_sig), 1][vert_sig] = int(ctrl_v)
    else:
        grid[r0 : r0 + len(vert_sig), -2][vert_sig] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _reserve_corner(canvas: np.ndarray, core_shape: tuple[int, int], desired_sides: dict[str, object], moat: int = 1) -> None:
    h, w = map(int, core_shape)
    r0, c0 = _corner_core_location(canvas.shape, core_shape, desired_sides)
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    rr0 = max(0, r0 - moat)
    cc0 = max(0, c0 - moat)
    rr1 = min(canvas.shape[0] - 1, r1 + moat)
    cc1 = min(canvas.shape[1] - 1, c1 + moat)
    canvas[rr0 : rr1 + 1, cc0 : cc1 + 1] = 9


def _expected_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    ctrls = _detect_controls_frame(g, bg)
    desired_shapes: dict[str, np.ndarray] = {}
    for side, color in ctrls:
        sig = _frame_signature(g, side, color)
        if sig is not None:
            desired_shapes[side] = sig
    obj, _bbox = _extract_center_object(g, bg)
    matches = _matching_transforms(obj, desired_shapes, bg)
    if len(matches) != 1:
        raise ValueError("expected exactly one shape-matching transform")
    core = _crop_aligned_core(matches[0][2], desired_shapes)
    out = g.copy()
    tr, tc = _corner_core_location(out.shape, core.shape, desired_shapes)
    paste(out, core, tr, tc)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(25, 31)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    guide_color = random.choice(remaining)
    remaining = [c for c in remaining if c != guide_color]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(11, n - 10)
    max_core_w = min(11, n - 10)
    if max_core_h < 5 or max_core_w < 5:
        return None

    core_h = random.randint(5, max_core_h)
    core_w_choices = [w for w in range(5, max_core_w + 1) if w != core_h]
    core_w = random.choice(core_w_choices) if core_w_choices else random.randint(5, max_core_w)
    core = _ensure_asymmetry(_build_core(core_h, core_w, fill, pattern_colors), fill, pattern_colors)

    horiz_len = core_w + 1
    vert_len = core_h + 1
    desired_sides = {horiz_side: True, vert_side: True}
    for _ in range(80):
        horiz_sig = _random_signature(horiz_len)
        vert_sig = _random_signature(vert_len)
        if not _nondegenerate_pair(horiz_sig, vert_sig):
            continue
        guide_shapes = {horiz_side: horiz_sig, vert_side: vert_sig}
        aligned_object = _attach_shape_guides(core, (horiz_side, vert_side), guide_color, guide_shapes)
        matches = _matching_transforms(aligned_object, guide_shapes, 0)
        if len(matches) != 1:
            continue

        transforms = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
        k, flip = random.choice(transforms)
        input_object = _inverse_transform(aligned_object, k, flip)
        if np.array_equal(input_object, aligned_object):
            continue
        break
    else:
        return None

    grid = np.zeros((n, n), dtype=int)
    try:
        _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v, horiz_sig, vert_sig)
    except ValueError:
        return None

    occupancy = grid.copy()
    _reserve_corner(occupancy, core.shape, desired_sides, moat=1)
    try:
        r0, c0 = random_free_location(occupancy, input_object, transparent=0, border=2)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    try:
        expected = _expected_output(grid)
    except ValueError:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    diag = _fusion_diagnostics(grid)
    if diag["shape_match_count"] != 1:
        return None
    if diag["color_match_count"] != 0:
        return None
    if not diag["non_identity_match"]:
        return None
    if not diag["source_target_disjoint"]:
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D1/0+D5/0")
