# Combo variant of fc7cae8d (index 22)
# combo_spec: D3/3+D4/4
# Name: 十字分割+条件旋转
#
# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - The seed task uses two border control edges and one interior guided rectangle. The solver
#      rotates/reflection-matches the rectangle so its two colored guide edges face the matching
#      border sides, then crops those guide edges away.
#    - This combo moves the controls from the outer border to an internal cross that splits the
#      canvas into four quadrants. The four half-segments of the cross have four distinct colors:
#      horizontal-left, horizontal-right, vertical-top, vertical-bottom.
#    - Each quadrant is identified by one horizontal half-segment plus one vertical half-segment.
#      The guided rectangle must be aligned as if it were placed inside that quadrant and touching
#      the cross, so the retained guide sides are the inward-facing sides:
#        TL -> bottom/right, TR -> bottom/left, BL -> top/right, BR -> top/left.
#    - Example: suppose the cross colors are h_left=7, h_right=4, v_top=2, v_bottom=8, and the
#      rectangle carries guide colors {7, 2}. In the seed task a solver would think "top/left".
#      In this combo the pair (h_left, v_top) means the target quadrant is top-left, so the guides
#      must face inward toward the cross, i.e. bottom=7 and right=2. After the matching D4
#      transform, those two guide strips are cropped. Then the secondary rule fires: since 7 > 2,
#      the cropped core is rotated 90 degrees CCW once more.
#    - The key difference from the seed is that color pair -> side mapping is no longer an outer
#      corner lookup. The solver must first infer the internal cross topology, recover the target
#      quadrant, and only then apply the conditional post-rotation.
# b) Complexity calibration:
#    - The spatial change is visible because the canvas now has a salient full-row/full-column cross
#      with four different segment colors, not two border edges.
#    - The conditional rule is visible because the generator forces non-square asymmetric cores, so
#      the extra 90-degree rotation changes both silhouette and pattern placement whenever it fires.
#    - Control freedom is intentionally limited to four distinct segment colors and four quadrants.
#      That is enough to require spatial reasoning without creating an unstructured color puzzle.
# c) Implicit assumption review against the seed solver:
#    - Border-side detection no longer applies; `main()` must detect a unique internal full row and
#      full column instead.
#    - The largest non-background component is no longer the object, because the cross itself is
#      larger. The solver therefore masks out the cross before component extraction.
#    - The old side mapping ("top/left", "top/right", ...) is wrong here. The solver must use the
#      inward-facing side mapping derived from quadrant geometry.
#    - The old "crop and return" end state is wrong because D4/4 adds a conditional final rotation.
#    - Guide extraction still assumes the core fill stays dominant inside the object; the generator
#      preserves that by keeping guide strips as the only near-solid full-edge colors.
# d) Placement legality:
#    - The generator builds the internal cross first, then places exactly one scrambled guided
#      rectangle strictly inside one quadrant with at least one-cell clearance from the outer border
#      and from the cross. This prevents overlap, out-of-bounds placement, and accidental merges.
# e) Visual observability:
#    - Across samples, a human can see that the object may appear in a quadrant unrelated to the
#      guide-color target quadrant, so current location is not the rule.
#    - The repeated structural cue is the same internal cross with four colored half-segments; the
#      output orientation depends on which quadrant-color pair the object matches, plus a separate
#      color-order trigger for the final rotation.
# f) Visual cleanliness:
#    - Only three visual layers exist: background, one internal cross, and one object. They never
#      overlap. The cross spans a full row/column, the object stays compact, and the center marker is
#      a single fixed color. This keeps the scene readable.
# g) Color-change inferability:
#    - No arbitrary recoloring happens. Colors are only identifiers for cross half-segments and for
#      the object's two guide strips. The conditional rule uses the observable numeric ordering of
#      those two matched control colors; it does not remap colors.
# h) Non-degeneration guard in the generator:
#    - Each sample is rejected if the same scrambled object would yield the same final output under a
#      degenerate "treat the pair like an outer corner" interpretation. This enforces that the D3
#      cross/quadrant change is behaviorally active, not just cosmetic.
#    - Triggered samples are also rejected if the final conditional rotation would be invisible.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes exactly one full non-background row and one full non-background column.
#      `generate_input()` creates exactly one such cross.
#    - `main()` assumes the four half-segments are recoverable by dominant color. The generator makes
#      each half-segment solid and distinct.
#    - `main()` assumes the guided rectangle remains a single component after masking the cross. The
#      generator keeps the object away from the cross and from the border.
#    - `main()` assumes exactly one quadrant pair matches the object's two guide colors. The generator
#      uses four distinct cross colors, so every quadrant pair is unique.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


QUADRANTS = ("tl", "tr", "bl", "br")
INWARD_SIDES = {
    "tl": ("bottom", "right"),
    "tr": ("bottom", "left"),
    "bl": ("top", "right"),
    "br": ("top", "left"),
}
OUTWARD_SIDES = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
SEGMENT_NAMES = ("h_left", "h_right", "v_top", "v_bottom")
MARKER = 5
USABLE_CTRL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _locate_cross(arr: np.ndarray, bg: int) -> tuple[int, int]:
    full_rows = [r for r in range(arr.shape[0]) if bool(np.all(arr[r, :] != int(bg)))]
    full_cols = [c for c in range(arr.shape[1]) if bool(np.all(arr[:, c] != int(bg)))]
    if len(full_rows) != 1 or len(full_cols) != 1:
        raise ValueError("expected exactly one internal cross row and one internal cross column")
    return int(full_rows[0]), int(full_cols[0])


def _dominant_non_bg(values: np.ndarray, bg: int) -> int:
    non_bg = [int(x) for x in values if int(x) != int(bg)]
    if not non_bg:
        raise ValueError("expected a non-background control segment")
    return int(Counter(non_bg).most_common(1)[0][0])


def _quadrant_control_info(arr: np.ndarray, cross_r: int, cross_c: int, bg: int) -> dict[str, dict[str, object]]:
    if cross_r <= 0 or cross_r >= arr.shape[0] - 1 or cross_c <= 0 or cross_c >= arr.shape[1] - 1:
        raise ValueError("cross must be strictly internal")

    seg = {
        "h_left": _dominant_non_bg(arr[cross_r, :cross_c], bg),
        "h_right": _dominant_non_bg(arr[cross_r, cross_c + 1 :], bg),
        "v_top": _dominant_non_bg(arr[:cross_r, cross_c], bg),
        "v_bottom": _dominant_non_bg(arr[cross_r + 1 :, cross_c], bg),
    }
    return _quadrant_control_info_from_segments(seg)


def _quadrant_control_info_from_segments(seg: dict[str, int]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for quadrant in QUADRANTS:
        h_name = "h_left" if quadrant in ("tl", "bl") else "h_right"
        v_name = "v_top" if quadrant in ("tl", "tr") else "v_bottom"
        ctrl_h = int(seg[h_name])
        ctrl_v = int(seg[v_name])
        inward = {
            INWARD_SIDES[quadrant][0]: ctrl_h,
            INWARD_SIDES[quadrant][1]: ctrl_v,
        }
        outward = {
            OUTWARD_SIDES[quadrant][0]: ctrl_h,
            OUTWARD_SIDES[quadrant][1]: ctrl_v,
        }
        out[quadrant] = {
            "ctrl_h": ctrl_h,
            "ctrl_v": ctrl_v,
            "desired": inward,
            "outer": outward,
        }
    return out


def _extract_center_object(arr: np.ndarray, bg: int, cross_r: int, cross_c: int) -> np.ndarray:
    masked = np.array(arr, copy=True)
    masked[cross_r, :] = int(bg)
    masked[:, cross_c] = int(bg)
    comps = find_components(masked, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    if not comps:
        raise ValueError("no interior guided object found after masking cross")
    best = max(comps, key=lambda comp: int(comp["area"]))
    return crop_bbox(masked, tuple(map(int, best["bbox"])))


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        vals = list(map(int, values))
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        col, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(col)
    return out


def _align_and_crop(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray | None:
    target_sides = set(desired)
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            gedges = _guide_edges(t)
            if set(gedges) != target_sides:
                continue
            if all(int(gedges[side]) == int(col) for side, col in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                return t[int(top) : int(bottom), int(left) : int(right)]
    return None


def _conditional_rotate(core: np.ndarray, ctrl_h: int, ctrl_v: int) -> np.ndarray:
    if int(ctrl_h) > int(ctrl_v):
        return np.rot90(core, 1)
    return core


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the internal cross, infer which quadrant's inward-facing edge pair the guided rectangle
    must match, crop away those guides, then conditionally rotate the cropped core.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cross_r, cross_c = _locate_cross(g, bg)
    qinfo = _quadrant_control_info(g, cross_r, cross_c, bg)
    obj = _extract_center_object(g, bg, cross_r, cross_c)

    raw_guides = _guide_edges(obj)
    raw_colors = set(raw_guides.values())
    candidate_quadrants = [
        quadrant
        for quadrant, info in qinfo.items()
        if raw_colors == {int(info["ctrl_h"]), int(info["ctrl_v"])}
    ]
    search_order = candidate_quadrants if candidate_quadrants else list(QUADRANTS)

    for quadrant in search_order:
        info = qinfo[quadrant]
        cropped = _align_and_crop(obj, info["desired"])
        if cropped is not None:
            return _conditional_rotate(cropped, int(info["ctrl_h"]), int(info["ctrl_v"]))

    return obj


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    colors = list(pattern_colors)

    core[0, 1 : min(w, 1 + max(2, w // 2))] = int(colors[0])
    core[min(h - 1, 1) : min(h, 1 + max(2, h // 2)), w - 2] = int(colors[1])

    elbow_r = max(1, h - 3)
    core[elbow_r : elbow_r + 2, 0] = int(colors[2])
    core[h - 1, : min(w, 3)] = int(colors[2])

    if len(colors) > 3:
        core[h // 2, max(1, w // 3)] = int(colors[3])

    area = h * w
    painted = int(np.count_nonzero(core != int(fill)))
    max_nonfill = min(area - 1, max(painted + 4, int(area * 0.28)))
    target_nonfill = random.randint(painted + 3, max_nonfill)

    while painted < target_nonfill:
        color = int(random.choice(colors))
        shape_kind = random.choice(["dot", "line", "rect"])
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
        else:
            mh = random.randint(1, min(3, h))
            mw = random.randint(1, min(3, w))
            r0 = random.randrange(h - mh + 1)
            c0 = random.randrange(w - mw + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        free_cells = [(r, c) for r, c in cells if int(core[r, c]) == int(fill)]
        if len(free_cells) > remaining:
            free_cells = random.sample(free_cells, remaining)
        for r, c in free_cells:
            core[r, c] = int(color)
        painted += len(free_cells)

    return core


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


def _quadrant_capacities(n: int, cross_r: int, cross_c: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (int(cross_r - 2), int(cross_c - 2)),
        "tr": (int(cross_r - 2), int(n - cross_c - 2)),
        "bl": (int(n - cross_r - 2), int(cross_c - 2)),
        "br": (int(n - cross_r - 2), int(n - cross_c - 2)),
    }


def _choose_ordered_pair(trigger: bool) -> tuple[int, int]:
    if trigger:
        hi = random.choice(USABLE_CTRL_COLORS[1:])
        lo = random.choice([c for c in USABLE_CTRL_COLORS if c < hi])
        return int(hi), int(lo)
    lo = random.choice(USABLE_CTRL_COLORS[:-1])
    hi = random.choice([c for c in USABLE_CTRL_COLORS if c > lo])
    return int(lo), int(hi)


def _choose_segments(target_quadrant: str, trigger: bool) -> dict[str, int]:
    ctrl_h, ctrl_v = _choose_ordered_pair(bool(trigger))
    seg: dict[str, int] = {}
    h_name = "h_left" if target_quadrant in ("tl", "bl") else "h_right"
    v_name = "v_top" if target_quadrant in ("tl", "tr") else "v_bottom"
    seg[h_name] = int(ctrl_h)
    seg[v_name] = int(ctrl_v)
    remaining_colors = [c for c in USABLE_CTRL_COLORS if c not in {ctrl_h, ctrl_v}]
    other_colors = random.sample(remaining_colors, 2)
    for name, color in zip([n for n in SEGMENT_NAMES if n not in seg], other_colors):
        seg[name] = int(color)
    return seg


def _paint_cross(grid: np.ndarray, cross_r: int, cross_c: int, seg: dict[str, int]) -> None:
    grid[cross_r, :cross_c] = int(seg["h_left"])
    grid[cross_r, cross_c + 1 :] = int(seg["h_right"])
    grid[:cross_r, cross_c] = int(seg["v_top"])
    grid[cross_r + 1 :, cross_c] = int(seg["v_bottom"])
    grid[cross_r, cross_c] = int(MARKER)


def _place_in_quadrant(grid: np.ndarray, patch: np.ndarray, cross_r: int, cross_c: int, quadrant: str) -> bool:
    n = int(grid.shape[0])
    h, w = map(int, patch.shape)

    if quadrant in ("tl", "tr"):
        r_lo, r_hi = 1, int(cross_r - h - 1)
    else:
        r_lo, r_hi = int(cross_r + 2), int(n - h - 1)

    if quadrant in ("tl", "bl"):
        c_lo, c_hi = 1, int(cross_c - w - 1)
    else:
        c_lo, c_hi = int(cross_c + 2), int(n - w - 1)

    if r_lo > r_hi or c_lo > c_hi:
        return False

    r0 = random.randint(r_lo, r_hi)
    c0 = random.randint(c_lo, c_hi)
    paste(grid, patch, int(r0), int(c0), transparent=0)
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 29)
    cross_r = random.randint(7, n - 8)
    cross_c = random.randint(7, n - 8)

    target_quadrant = random.choice(QUADRANTS)
    trigger = bool(random.randrange(2))
    seg = _choose_segments(target_quadrant, trigger)
    qinfo = _quadrant_control_info_from_segments(seg)
    info = qinfo[target_quadrant]

    capacities = _quadrant_capacities(n, cross_r, cross_c)
    base_fit = [q for q, (ah, aw) in capacities.items() if ah >= 6 and aw >= 7]
    if not base_fit:
        return None
    size_quadrant = random.choice(base_fit)
    max_core_h = min(12, capacities[size_quadrant][0] - 1)
    max_core_w = min(14, capacities[size_quadrant][1] - 1)
    if max_core_h < 5 or max_core_w < 6:
        return None

    core_h = random.randint(5, max_core_h)
    core_w_choices = [w for w in range(6, max_core_w + 1) if w != core_h]
    if not core_w_choices:
        return None
    core_w = random.choice(core_w_choices)

    unavailable = {MARKER, *(int(seg[name]) for name in SEGMENT_NAMES)}
    remaining = [c for c in range(1, 10) if c not in unavailable]
    if len(remaining) < 4:
        return None
    fill = random.choice(remaining)
    pattern_pool = [c for c in remaining if c != fill]
    pattern_colors = random.sample(pattern_pool, random.randint(3, min(4, len(pattern_pool))))

    core = _build_core(core_h, core_w, int(fill), pattern_colors)
    expected = _conditional_rotate(core, int(info["ctrl_h"]), int(info["ctrl_v"]))

    aligned_object = _attach_guides(core, INWARD_SIDES[target_quadrant], info["desired"])
    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    fit_quadrants = [
        q
        for q, (ah, aw) in capacities.items()
        if int(input_object.shape[0]) <= int(ah) and int(input_object.shape[1]) <= int(aw)
    ]
    if not fit_quadrants:
        return None
    place_choices = [q for q in fit_quadrants if q != target_quadrant]
    place_quadrant = random.choice(place_choices if place_choices else fit_quadrants)

    fused_crop = _align_and_crop(input_object, info["desired"])
    outer_crop = _align_and_crop(input_object, info["outer"])
    if fused_crop is None or outer_crop is None:
        return None
    if not np.array_equal(fused_crop, core):
        return None
    fused_final = _conditional_rotate(fused_crop, int(info["ctrl_h"]), int(info["ctrl_v"]))
    outer_final = _conditional_rotate(outer_crop, int(info["ctrl_h"]), int(info["ctrl_v"]))
    if np.array_equal(fused_final, outer_final):
        return None
    if bool(trigger) and np.array_equal(fused_crop, fused_final):
        return None

    grid = np.zeros((n, n), dtype=int)
    _paint_cross(grid, cross_r, cross_c, seg)
    if not _place_in_quadrant(grid, input_object, cross_r, cross_c, place_quadrant):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D3/3+D4/4")
