# Combo variant of fc7cae8d (index 16)
# combo_spec: D2/4+D4/3
# Name: 空心源框+擦除填充
#
# Design analysis for this combo
# a) Concrete behavior trace:
#    Original fc7cae8d uses a filled core. After D4 alignment and guide-crop, it
#    returns that dense rectangle unchanged.
#    This combo changes two linked stages:
#    - D2/4: the source core is now a thick hollow rectangular frame whose inner
#      hole is background 0. The two guide strips are still attached outside two
#      adjacent sides.
#    - D4/3: after the same alignment and crop, erase the dominant support color
#      of the hollow frame to background 0.
#    Example aligned object before crop:
#        [[0,2,2,2,2,2,2],
#         [3,4,4,4,4,4,4],
#         [3,4,6,6,6,4,4],
#         [3,4,6,0,6,4,4],
#         [3,4,6,6,6,4,4],
#         [3,4,4,4,4,4,4]]
#      top-guide=2, left-guide=3, support fill=4, inner wireframe=6.
#      Original logic would align and crop to:
#        [[4,4,4,4,4,4],
#         [4,6,6,6,4,4],
#         [4,6,0,6,4,4],
#         [4,6,6,6,4,4],
#         [4,4,4,4,4,4]]
#      This combo then erases support fill 4 and outputs:
#        [[0,0,0,0,0,0],
#         [0,6,6,6,0,0],
#         [0,6,0,6,0,0],
#         [0,6,6,6,0,0],
#         [0,0,0,0,0,0]]
#      The difference is not a separate independent tweak: the hollow source
#      makes the erase step produce a minimal line drawing rather than a sparse
#      punctured solid block.
# b) Complexity calibration:
#    A 1px hollow frame would let the hole dominate and break fill inference.
#    The generator therefore uses a thick frame (2-3 px) so the support color is
#    still the most common non-background color. The retained pattern is limited
#    to one full inner loop plus a few optional accents, which keeps the visual
#    rule strong without creating extra matching ambiguity.
# c) Implicit assumptions review:
#    - `detect_controls_frame` still assumes exactly one horizontal and one
#      vertical near-solid control edge on the outer frame. The generator keeps
#      that and only adds grey noise to the other two frame sides.
#    - `extract_center_object` assumes one largest interior component. The guided
#      hollow frame remains one connected non-background component because its
#      ring and guide strips are 4-connected.
#    - Original `guide_edges` assumed the overall dominant color was the solid
#      fill. Hollow frames violate that because the inner hole can make 0 common.
#      The solver now uses "most common non-background color" as the support fill.
#    - `crop_guides` assumes guides occur only on the two desired sides. The
#      generator preserves that by placing the decorative line on the inner rim
#      of the frame, never on the outer rim where it could mimic a guide strip.
#    - `erase_fill` now assumes there is a single dominant non-background support
#      color after crop. The generator enforces a thick frame so this remains true.
# d) Placement legality:
#    The transformed object is inserted with `random_free_location(..., border=1)`
#    and `transparent=0`, so the hole does not cause overlap, the object stays
#    fully inside the frame, and it never touches the border controls.
# e) Visual observability:
#    Across examples, input still shows "frame-edge colors tell orientation", but
#    output changes from a dense block to a thin wireframe-like motif. A human can
#    infer: first align/crop the hollow source, then remove its support color.
# f) Visual cleanliness:
#    Only three layers coexist in the object bbox: outer guides, support frame,
#    inner wireframe. No marks are stacked on the same outer edge, so guide sides
#    stay easy to distinguish from non-guide sides.
# g) Color inferability:
#    Colors are few and structurally separated: two frame-control colors, one
#    support fill color, one or two retained motif colors, and grey frame noise.
#    The erase rule is therefore visible as "remove the thick support ring",
#    not as an arbitrary recoloring table.
# 9) New solver assumptions after modification:
#    - `_dominant_nonbg` expects at least one non-background color in the object;
#      generator always creates a non-empty framed object.
#    - `_guide_edges` expects the support color to dominate non-background counts;
#      generator rejects samples where that is false.
#    - `_aligned_cropped_core` expects exactly one D4 pose to satisfy both guide
#      colors. Because the two controls are on adjacent sides and use distinct
#      colors, that pose is identifiable.

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.insert(0, ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def _dominant_nonbg(arr: np.ndarray, bg: int) -> int:
    values = [int(x) for x in arr.flatten() if int(x) != int(bg)]
    if not values:
        return int(bg)
    return int(Counter(values).most_common(1)[0][0])


def _detect_controls_frame(arr: np.ndarray, bg: int) -> dict[str, int]:
    h, w = arr.shape
    sides = {
        "top": arr[0, 1 : w - 1],
        "bottom": arr[h - 1, 1 : w - 1],
        "left": arr[1 : h - 1, 0],
        "right": arr[1 : h - 1, w - 1],
    }
    candidates: list[tuple[str, int, int, int]] = []
    for side, values in sides.items():
        vals = [int(x) for x in values]
        nonbg = [x for x in vals if x != int(bg)]
        if not nonbg:
            continue
        color, count = Counter(nonbg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            candidates.append((side, int(color), int(count), int(len(vals))))
    candidates.sort(key=lambda item: (item[2] / item[3], item[2]), reverse=True)
    return {side: color for side, color, _count, _length in candidates[:2]}


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return np.array(arr, copy=True)
    return crop_bbox(arr, best[1])


def _guide_edges(arr: np.ndarray, bg: int) -> dict[str, int]:
    fill = _dominant_nonbg(arr, bg)
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        vals = [int(x) for x in values]
        nonguide_fill = [x for x in vals if x != int(fill) and x != int(bg)]
        if not nonguide_fill:
            continue
        color, count = Counter(nonguide_fill).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(color)
    return out


def _crop_guides(arr: np.ndarray, sides: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in sides else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in sides else 0)
    left = 1 if "left" in sides else 0
    right = int(arr.shape[1]) - (1 if "right" in sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)]


def _erase_fill(arr: np.ndarray, bg: int) -> np.ndarray:
    fill = _dominant_nonbg(arr, bg)
    out = np.array(arr, copy=True)
    if fill == int(bg):
        return out
    out[out == int(fill)] = int(bg)
    return out


def _aligned_cropped_core(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_controls_frame(g, bg)
    obj = _extract_center_object(g, bg)

    for k in range(4):
        rotated = np.rot90(obj, k)
        for flip in (False, True):
            transformed = np.fliplr(rotated) if flip else rotated
            guides = _guide_edges(transformed, bg)
            if all(side in guides and int(guides[side]) == int(color) for side, color in desired.items()):
                return _crop_guides(transformed, desired)

    fallback_guides = _guide_edges(obj, bg)
    return _crop_guides(obj, fallback_guides)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the guided hollow frame to the border controls, crop the guide strips,
    then erase the dominant support color so only the sparse wireframe motif remains.
    """
    cropped = _aligned_cropped_core(grid)
    return _erase_fill(cropped, bg=0)


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _build_hollow_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> tuple[np.ndarray, int]:
    max_thickness = min(3, (min(h, w) - 3) // 2)
    if max_thickness < 2:
        raise ValueError("dimensions too small for a visible hollow frame")
    thickness = random.randint(2, max_thickness)

    core = np.full((h, w), int(fill), dtype=int)
    core[thickness : h - thickness, thickness : w - thickness] = 0

    top = thickness - 1
    bottom = h - thickness
    left = thickness - 1
    right = w - thickness

    horiz_color = int(pattern_colors[0])
    vert_color = int(pattern_colors[1] if len(pattern_colors) > 1 and random.random() < 0.6 else horiz_color)
    core[top, left : right + 1] = horiz_color
    core[bottom, left : right + 1] = horiz_color
    core[top : bottom + 1, left] = vert_color
    core[top : bottom + 1, right] = vert_color

    # Optional sparse accents stay inside the thick ring so they survive erase
    # without interfering with outer guide detection.
    if thickness >= 3 and len(pattern_colors) > 1 and random.random() < 0.7:
        accent_color = int(random.choice(pattern_colors))
        band = random.choice(["top", "bottom", "left", "right"])
        if band in ("top", "bottom"):
            rr = 1 if band == "top" else h - 2
            c0 = random.randint(1, max(1, w - 4))
            c1 = min(w - 2, c0 + random.randint(1, 3))
            core[rr, c0 : c1 + 1] = accent_color
        else:
            cc = 1 if band == "left" else w - 2
            r0 = random.randint(1, max(1, h - 4))
            r1 = min(h - 2, r0 + random.randint(1, 3))
            core[r0 : r1 + 1, cc] = accent_color

    return core, thickness


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int]) -> np.ndarray:
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
        out[0, c0 : c0 + w] = int(guide_colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_colors["right"])

    if has_top and has_left:
        out[0, 0] = 0
    if has_top and has_right:
        out[0, -1] = 0
    if has_bottom and has_left:
        out[-1, 0] = 0
    if has_bottom and has_right:
        out[-1, -1] = 0
    return out


def _paint_frame(grid: np.ndarray, horiz_side: str, vert_side: str, ctrl_h: int, ctrl_v: int) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        noise_count = random.randint(int(np.ceil(0.35 * len(coords))), int(np.floor(0.60 * len(coords))))
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _core_metrics(core: np.ndarray, fill: int, erased: np.ndarray) -> dict[str, float]:
    return {
        "hole_area": float(np.count_nonzero(core == 0)),
        "fill_cells_removed": float(np.count_nonzero(core == int(fill))),
        "survivor_cells": float(np.count_nonzero(erased != 0)),
        "erase_delta": float(np.count_nonzero(core != erased)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 30)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([color for color in all_colors if color != 5], 2)
    remaining = [color for color in all_colors if color not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [color for color in remaining if color != fill]
    pattern_count = random.randint(1, min(2, len(remaining)))
    pattern_colors = random.sample(remaining, pattern_count)

    max_core_h = min(18, n - 3)
    max_core_w = min(18, n - 3)
    if max_core_h < 8 or max_core_w < 8:
        return None
    core_h = random.randint(8, max_core_h)
    core_w = random.randint(8, max_core_w)
    core, thickness = _build_hollow_core(core_h, core_w, fill, pattern_colors)
    expected = _erase_fill(core, bg=0)
    metrics = _core_metrics(core, fill, expected)

    if metrics["hole_area"] < 9:
        return None
    if metrics["survivor_cells"] <= 0:
        return None
    if metrics["erase_delta"] <= 0:
        return None
    if metrics["survivor_cells"] >= metrics["fill_cells_removed"]:
        return None

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)
    if _dominant_nonbg(aligned_object, 0) != int(fill):
        return None
    if set(_guide_edges(aligned_object, 0)) != {horiz_side, vert_side}:
        return None

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    cropped = _aligned_cropped_core(grid)
    if not np.array_equal(cropped, core):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if thickness == 2 and np.count_nonzero(expected) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D2/4+D4/3")
