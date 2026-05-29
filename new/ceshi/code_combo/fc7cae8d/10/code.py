# Combo variant of fc7cae8d (index 10)
# combo_spec: D1/0+D10/0
# Name: 形状嵌合+干扰矩形
#
# Design analysis for this combo:
# a) Concrete behavior trace:
#    Original task: one interior rectangle carried two almost-solid guide strips whose colors matched two frame
#    control edges. The solver could crop the largest interior component, rotate/flip it, read guide colors, and
#    trim away the matched 1px guide strips.
#    Combo behavior: the true source still becomes the output core after a dihedral transform, but its two active
#    guide sides are now jagged 1px borders with background holes. The frame still exposes one horizontal and one
#    vertical control edge, yet orientation is encoded by inward protrusion masks on the inner border line rather
#    than by edge color equality. At the same time, there are 1-2 extra interior rectangles with no guide layer.
#    Example trace: suppose the frame activates `top` and `left`. The true object may appear rotated so its jagged
#    guides sit on `bottom` and `right`, while a larger plain distractor sits elsewhere. The old solver would first
#    pick the largest component and then look for solid control-colored edges, which now fails twice: the distractor
#    is not the target, and the target guides are not monochrome. The combo solver must instead scan every interior
#    component, test every dihedral transform, compare frame protrusion masks to object bg-hole masks on candidate
#    guide sides, and crop the uniquely matching component.
# b) Complexity calibration:
#    The generator forces non-palindromic jagged signatures with both holes and solids, rejects objects admitting
#    multiple matching dihedral transforms, and always places 1-2 distractors. One distractor is biased to be at
#    least as large as the true source bbox so that "pick the largest interior rectangle" fails often. This keeps
#    the visual change obvious without exploding the search space beyond the eight dihedral transforms.
# c) Implicit assumption audit against the original `main()`:
#    - `detect_controls_frame()` assumed the active frame sides are almost solid. That still holds; inactive sides
#      are now left blank instead of noisy because D10 replaces the old border-noise meta variation.
#    - `extract_center_object()` assumed the largest interior component is the source. D10 breaks this, so the combo
#      solver enumerates all interior components.
#    - `guide_edges()` assumed guide sides are almost monochrome. D1 breaks this because each guide side contains bg
#      holes. The combo solver replaces color-count logic with shape-signature extraction.
#    - The crop step assumes guide thickness is exactly 1px on the matched sides. The generator preserves that.
#    - Component separation was implicit before because there was only one object. The combo generator now enforces
#      at least one cell of padding so distractors remain separate 4-connected components.
# d) Placement legality:
#    Frame protrusions live on row/col 1 or n-2, so all interior objects are placed with `border=2`. The true
#    guided object uses `transparent=0` because its guide holes are real background cells; distractors are solid
#    rectangles and are also placed with padding so nothing overlaps or touches.
# e) Visual observability:
#    Across examples, a human sees three stable cues: two frame sides with comb-like inward marks, one interior
#    rectangle with matching jagged cutouts on two sides, and 1-2 plain rectangles that do not have those cutouts.
#    The output is consistently the fill-patterned core of the cutout-bearing rectangle after alignment.
# f) Visual cleanliness:
#    To avoid clutter, inactive frame sides are blank, protrusions stay away from corners, the number of distractors
#    is capped at two, and all objects are separated by at least one background cell. This prevents overlap between
#    frame cues and interior rectangles.
# g) Color-change inferability:
#    Colors are not the main carrier of the new rule. Control colors stay on the frame, a separate guide color marks
#    the jagged source border, and core/distractor interiors avoid the frame colors. That keeps color variety modest
#    and makes the geometric matching cue visually dominant.
#
# New solver assumptions introduced here and guaranteed by the generator:
# - Exactly two frame control sides exist: one horizontal and one vertical.
# - Exactly one interior component admits exactly one dihedral transform whose bg-hole guide signatures match the two
#   frame protrusion signatures.
# - Distractors are plain rectangles with no bg holes on their boundary, so they never satisfy the shape-match test.
# - The matched guide signatures correspond to 1px-thick removable borders.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, values in sides.items():
        vals = list(map(int, values))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(color), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, color) for side, color, _count, _length in cand[:2]]


def _frame_signature(arr: np.ndarray, side: str, color: int) -> np.ndarray | None:
    h, w = arr.shape
    if side == "top":
        line = arr[1, 1:-1]
    elif side == "bottom":
        line = arr[h - 2, 1:-1]
    elif side == "left":
        line = arr[1:-1, 1]
    else:
        line = arr[1:-1, w - 2]

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
        outer = arr[0, :]
    elif side == "bottom":
        outer = arr[-1, :]
    elif side == "left":
        outer = arr[:, 0]
    else:
        outer = arr[:, -1]

    holes = np.array(outer == int(bg), dtype=bool)
    hole_count = int(holes.sum())
    if hole_count < 2 or hole_count >= len(holes):
        return None
    solid_count = int((~holes).sum())
    if solid_count < 2:
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
    desired_sides = set(desired_shapes)
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            got = _object_signatures(transformed, bg)
            if set(got) != desired_sides:
                continue
            ok = True
            for side, sig in desired_shapes.items():
                cur = got.get(side)
                if cur is None or len(cur) != len(sig) or not np.array_equal(cur, sig):
                    ok = False
                    break
            if ok:
                matches.append((int(k), bool(flip), transformed))
    return matches


def _collect_component_matches(
    grid: np.ndarray, bg: int, desired_shapes: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    h, w = grid.shape
    comps = find_components(grid, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    matches: list[dict[str, object]] = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        obj = crop_bbox(grid, (r0, c0, r1, c1))
        for k, flip, transformed in _matching_transforms(obj, desired_shapes, bg):
            matches.append(
                {
                    "component_index": int(idx),
                    "bbox": (r0, c0, r1, c1),
                    "area": int(comp["area"]),
                    "k": int(k),
                    "flip": bool(flip),
                    "transformed": transformed,
                }
            )
    return matches


def _crop_aligned_core(arr: np.ndarray, desired_shapes: dict[str, np.ndarray]) -> np.ndarray:
    top = 1 if "top" in desired_shapes else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in desired_shapes else 0)
    left = 1 if "left" in desired_shapes else 0
    right = int(arr.shape[1]) - (1 if "right" in desired_shapes else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - The frame has one horizontal and one vertical control side, each with an inward protrusion mask.
    - Exactly one interior component has jagged guide edges whose bg-hole masks can align to those protrusions.
    - Other interior rectangles are distractors without removable guide edges.
    - Apply the unique dihedral transform that shape-matches the frame, then trim the matched 1px guide layer.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired_shapes: dict[str, np.ndarray] = {}
    for side, color in ctrls:
        sig = _frame_signature(g, side, color)
        if sig is None:
            raise ValueError(f"missing frame signature on side {side}")
        desired_shapes[side] = sig
    if len(desired_shapes) != 2:
        raise ValueError("expected exactly two frame control signatures")

    matches = _collect_component_matches(g, bg, desired_shapes)
    if not matches:
        raise ValueError("no interior component matches the frame signatures")
    if len(matches) != 1:
        raise ValueError(f"expected a unique matching component/transform, got {len(matches)}")
    matches.sort(key=lambda m: (int(m["area"]), -int(m["component_index"])), reverse=True)
    return _crop_aligned_core(np.array(matches[0]["transformed"], dtype=int), desired_shapes)


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors), int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors), area // 10), max_nonfill)
    painted = 0

    for color in pattern_colors:
        r = random.randrange(h)
        c = random.randrange(w)
        core[r, c] = int(color)
        painted += 1

    while painted < target_nonfill:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [(random.randrange(h), random.randrange(w))]
        elif shape_kind == "line":
            if random.random() < 0.5:
                line_len = random.randint(2, min(5, w))
                rr = random.randrange(h)
                c0 = random.randrange(w - line_len + 1)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                line_len = random.randint(2, min(5, h))
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
        fill_cells = [(r, c) for r, c in cells if core[r, c] == fill]
        if len(fill_cells) > remaining:
            fill_cells = random.sample(fill_cells, remaining)
        for r, c in fill_cells:
            core[r, c] = int(color)
        painted += len(fill_cells)
        for r, c in cells:
            if core[r, c] != fill or painted >= target_nonfill:
                continue
            core[r, c] = int(color)
            painted += 1
    return core


def _random_signature(length: int) -> np.ndarray:
    while True:
        sig = np.zeros(int(length), dtype=bool)
        sig[0] = True
        sig[-1] = True
        interior = list(range(1, length - 1))
        max_extra = min(len(interior) - 1, max(2, length // 2))
        if max_extra < 1:
            continue
        extra_n = random.randint(1, max_extra)
        for idx in random.sample(interior, extra_n):
            sig[int(idx)] = True
        if int(sig.sum()) < 3:
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
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    c_min = 2
    c_max = n - len(horiz_sig) - 2
    r_min = 2
    r_max = n - len(vert_sig) - 2
    if c_min > c_max or r_min > r_max:
        raise ValueError("frame too small for protrusion signatures")

    c0 = random.randint(c_min, c_max)
    r0 = random.randint(r_min, r_max)

    if horiz_side == "top":
        span = grid[1, c0 : c0 + len(horiz_sig)]
    else:
        span = grid[-2, c0 : c0 + len(horiz_sig)]
    span[np.array(horiz_sig, dtype=bool)] = int(ctrl_h)

    if vert_side == "left":
        span = grid[r0 : r0 + len(vert_sig), 1]
    else:
        span = grid[r0 : r0 + len(vert_sig), -2]
    span[np.array(vert_sig, dtype=bool)] = int(ctrl_v)


def _choose_core_dims(n: int) -> tuple[int, int] | None:
    max_core_h = min(11, n - 11)
    max_core_w = min(11, n - 11)
    if max_core_h < 6 or max_core_w < 6:
        return None
    return random.randint(6, max_core_h), random.randint(6, max_core_w)


def _choose_large_distractor_dims(n: int, source_h: int, source_w: int) -> tuple[int, int] | None:
    max_h = min(n - 6, source_h + 4)
    max_w = min(n - 6, source_w + 4)
    min_h = max(5, min(max_h, source_h - 1))
    min_w = max(5, min(max_w, source_w - 1))
    if min_h > max_h or min_w > max_w:
        return None
    source_area = int(source_h * source_w)
    for _ in range(80):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        if h * w >= source_area:
            return h, w
    return None


def _choose_small_distractor_dims(n: int, source_h: int, source_w: int) -> tuple[int, int] | None:
    max_h = min(n - 7, max(5, source_h + 1))
    max_w = min(n - 7, max(5, source_w + 1))
    if max_h < 4 or max_w < 4:
        return None
    return random.randint(4, max_h), random.randint(4, max_w)


def _build_distractor(n: int, source_h: int, source_w: int, colors: list[int], want_large: bool) -> np.ndarray | None:
    dims = _choose_large_distractor_dims(n, source_h, source_w) if want_large else _choose_small_distractor_dims(
        n, source_h, source_w
    )
    if dims is None:
        return None
    h, w = dims
    fill = random.choice(colors)
    pattern_pool = [c for c in colors if c != fill]
    if len(pattern_pool) < 2:
        return None
    pattern_count = random.randint(2, min(4, len(pattern_pool)))
    return _build_core(h, w, fill, random.sample(pattern_pool, pattern_count))


def _largest_interior_bbox_area(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    best = 0
    comps = find_components(grid, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        best = max(best, int((r1 - r0 + 1) * (c1 - c0 + 1)))
    return best


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(26, 32)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample(all_colors, 2)
    remaining = [c for c in all_colors if c not in {ctrl_h, ctrl_v}]
    guide_color = random.choice(remaining)
    remaining = [c for c in remaining if c != guide_color]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_count = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, pattern_count)

    dims = _choose_core_dims(n)
    if dims is None:
        return None
    core_h, core_w = dims
    core = _build_core(core_h, core_w, fill, pattern_colors)

    horiz_len = core_w + 1
    vert_len = core_h + 1
    desired_shapes: dict[str, np.ndarray]
    aligned_object: np.ndarray
    for _ in range(80):
        horiz_sig = _random_signature(horiz_len)
        vert_sig = _random_signature(vert_len)
        if not _nondegenerate_pair(horiz_sig, vert_sig):
            continue
        desired_shapes = {horiz_side: horiz_sig, vert_side: vert_sig}
        aligned_object = _attach_shape_guides(core, (horiz_side, vert_side), guide_color, desired_shapes)
        if len(_matching_transforms(aligned_object, desired_shapes, 0)) == 1:
            break
    else:
        return None

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    try:
        _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v, horiz_sig, vert_sig)
    except ValueError:
        return None

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=2, padding=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    distractor_colors = [c for c in all_colors if c not in {ctrl_h, ctrl_v}]
    n_distractors = random.randint(1, 2)
    source_obj_h, source_obj_w = input_object.shape
    source_bbox_area = int(source_obj_h * source_obj_w)
    placed_large = False

    for idx in range(n_distractors):
        distractor = _build_distractor(
            n,
            source_obj_h,
            source_obj_w,
            distractor_colors,
            want_large=(idx == 0 or (not placed_large and idx == n_distractors - 1)),
        )
        if distractor is None:
            return None
        try:
            dr, dc = random_free_location(grid, distractor, transparent=0, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, distractor, dr, dc, transparent=0)
        placed_large = placed_large or distractor.shape[0] * distractor.shape[1] >= source_bbox_area

    if not placed_large:
        return None

    matches = _collect_component_matches(grid, 0, desired_shapes)
    if len(matches) != 1:
        return None

    target_bbox = matches[0]["bbox"]
    target_bbox_area = int((target_bbox[2] - target_bbox[0] + 1) * (target_bbox[3] - target_bbox[1] + 1))
    if _largest_interior_bbox_area(grid, 0) <= target_bbox_area:
        return None

    solved = main(grid)
    if not np.array_equal(solved, core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant D1/0+D10/0")
