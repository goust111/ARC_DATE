# Combo variant of fc7cae8d (index 6)
# combo_spec: D1/0+D3/0
# Name: 形状嵌合+对边控制
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    Original fc7cae8d used two adjacent frame controls and two adjacent object guide strips.
#    The solver read guide-strip colors, searched a dihedral transform, then cropped the aligned strips.
#    This combo changes both cues at once:
#    - D3/0: the active frame sides are opposite, not adjacent, so the task is axis alignment instead of corner alignment.
#    - D1/0: the object guides are no longer color-identifiable strips. They are a single guide color with bg holes,
#      while the frame exposes matching same-colored protrusion masks on the inner line of each active side.
#    Small trace on a top+bottom sample:
#    - old rule: "find top color X and bottom color Y on the object, rotate until they face the top and bottom frame edges".
#    - fused rule: "find that the frame activates the vertical axis (top/bottom); read the top and bottom protrusion masks;
#      rotate/flip the object until its two opposite guide sides land on top and bottom and each hole mask equals the
#      corresponding frame protrusion mask; then crop those two guide rows".
#    The visible difference is that the object can start with left/right jagged guides, so the solver must reason about
#    axis and shape, not color or corner.
# b) Complexity calibration:
#    To keep the fusion visible and non-trivial, the generator enforces:
#    - the object guide color is different from both frame control colors, so color-only matching cannot solve it;
#    - the active pair is always opposite and the input object is rotated by 90 or 270 degrees relative to the target axis;
#    - the two guide signatures on the active axis are distinct, non-palindromic, and not reverses of each other;
#    - only one dihedral transform may satisfy the pair of signatures.
#    This keeps the search space rich enough for the combo while avoiding ambiguous samples.
# c) Hidden assumptions review against the seed solver:
#    - `guide_edges()` from the seed assumed guides were almost monochrome; this is false because guide sides now mix
#      guide color and bg holes. Replaced with explicit hole-mask extraction.
#    - The seed only needed any two strong frame edges; now the solver must prefer an opposite pair, because adjacency no
#      longer encodes the answer.
#    - The seed's color equality test between frame and object sides no longer holds. Colors now only mark which frame
#      sides are active; the match itself is geometric.
#    - Cropping still assumes a 1px guide thickness. The generator preserves exactly that.
# d) Placement legality:
#    The object contains bg holes, so placement uses `transparent=0` for collision testing and pasting.
#    Frame protrusions sit on the inner line next to the active border and stay away from corners, so they do not collide
#    with the opposite control edge or with the central object. The object is placed with `border=1`, keeping it strictly
#    inside the frame.
# e) Visual observability:
#    A human sees two opposite colored border controls and matching comb-like protrusions facing inward. The object shows
#    two opposite jagged guide sides in a third color. Because the object starts on the orthogonal axis, the "axis switch"
#    is obvious from a few examples, and the matching shapes reveal which side maps to which side.
# f) Visual cleanliness:
#    Only three structural layers occupy the scene: frame borders, inner protrusion masks, and one central object.
#    Protrusions are thin and localized; the object remains a single rectangular body with sparse interior patterning.
#    This avoids overlapping annotations in the same region.
# g) Color inference:
#    No extra color mapping needs to be inferred. Frame control colors identify active frame sides; a separate guide color
#    marks both jagged object guides. Because guide color never equals either control color, the intended "shape not color"
#    reasoning is externally visible.
# h) Fusion-behavior metric used during verification:
#    - `axis_changed`: input object guide axis differs from the frame control axis.
#    - `guide_color_distinct`: the object guide color is absent from frame control colors.
#    - `old_solver_same_output`: the original seed solver returns the same output on this fused input.
#    Healthy fusion should give `axis_changed=True`, `guide_color_distinct=True`, and usually
#    `old_solver_same_output=False`.
# i) New solver assumptions and generator guarantees:
#    - `_frame_signature()` assumes each active frame side has one contiguous signature span whose endpoints are active.
#      The generator paints signatures with first/last cells set.
#    - `_object_side_signature()` assumes only guide sides contain bg holes. The generator keeps the core border solid.
#    - `_matching_transforms()` assumes the signature pair uniquely identifies one dihedral pose. The generator rejects
#      any sample with zero or multiple matches.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.util
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[5] / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

OPPOSITE_PAIRS = (("top", "bottom"), ("left", "right"))


def _edge_slices(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }


def _detect_opposite_controls(arr: np.ndarray, bg: int) -> dict[str, int]:
    candidates: dict[str, tuple[int, int]] = {}
    for side, edge in _edge_slices(arr).items():
        vals = [int(x) for x in edge if int(x) != int(bg)]
        if not vals:
            continue
        color, count = Counter(vals).most_common(1)[0]
        if int(count) >= len(edge) - 1:
            candidates[side] = (int(color), int(count))

    for side_a, side_b in OPPOSITE_PAIRS:
        if side_a in candidates and side_b in candidates:
            return {side_a: candidates[side_a][0], side_b: candidates[side_b][0]}

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][1], reverse=True)[:2]
    return {side: color for side, (color, _count) in ranked}


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
        return arr.copy()
    return crop_bbox(arr, best[1])


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
    if int((~sig).sum()) < 2:
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
    hole_count = int(holes.sum())
    if hole_count < 3 or hole_count >= len(holes):
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


def _crop_guides(arr: np.ndarray, sides: dict[str, np.ndarray] | dict[str, int]) -> np.ndarray:
    r0 = 1 if "top" in sides else 0
    r1 = arr.shape[0] - (1 if "bottom" in sides else 0)
    c0 = 1 if "left" in sides else 0
    c1 = arr.shape[1] - (1 if "right" in sides else 0)
    return arr[r0:r1, c0:c1]


def _matching_transforms(
    obj: np.ndarray,
    desired_shapes: dict[str, np.ndarray],
    bg: int,
) -> list[tuple[int, bool, np.ndarray]]:
    matches: list[tuple[int, bool, np.ndarray]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            cur = np.fliplr(base) if flip else base
            got = _object_signatures(cur, bg)
            if set(got) != set(desired_shapes):
                continue
            ok = True
            for side, sig in desired_shapes.items():
                cur_sig = got.get(side)
                if cur_sig is None or len(cur_sig) != len(sig) or not np.array_equal(cur_sig, sig):
                    ok = False
                    break
            if ok:
                matches.append((int(k), bool(flip), cur))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align an opposite-guide object to opposite frame controls using shape complement.

    The active frame axis is detected from the opposite pair of almost-solid border edges.
    Each active side exposes an inward protrusion mask; the matching object side is the one
    whose guide border has bg holes with the same boolean signature.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    desired_colors = _detect_opposite_controls(g, bg)
    desired_shapes: dict[str, np.ndarray] = {}
    for side, color in desired_colors.items():
        sig = _frame_signature(g, side, color)
        if sig is not None:
            desired_shapes[side] = sig

    obj = _extract_center_object(g, bg)
    matches = _matching_transforms(obj, desired_shapes, bg)
    if matches:
        return _crop_guides(matches[0][2], desired_shapes)
    return obj


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    inner_rows = list(range(1, h - 1))
    inner_cols = list(range(1, w - 1))
    inner_cells = [(r, c) for r in inner_rows for c in inner_cols]
    random.shuffle(inner_cells)

    target_nonfill = random.randint(
        max(len(pattern_colors) + 2, len(inner_cells) // 9),
        max(len(pattern_colors) + 6, int(len(inner_cells) * 0.35)),
    )
    painted = 0

    for color in pattern_colors:
        if not inner_cells:
            break
        r, c = inner_cells.pop()
        core[r, c] = int(color)
        painted += 1

    while painted < target_nonfill and inner_cells:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [random.choice(inner_cells)]
        elif shape_kind == "line":
            if random.random() < 0.5:
                rr = random.choice(inner_rows)
                line_len = random.randint(2, min(4, w - 2))
                c0 = random.randint(1, w - 1 - line_len)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                cc = random.choice(inner_cols)
                line_len = random.randint(2, min(4, h - 2))
                r0 = random.randint(1, h - 1 - line_len)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            mh = random.randint(1, min(3, h - 2))
            mw = random.randint(1, min(3, w - 2))
            r0 = random.randint(1, h - 1 - mh)
            c0 = random.randint(1, w - 1 - mw)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        for r, c in cells:
            if painted >= target_nonfill:
                break
            if core[r, c] == int(fill):
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
    bg: int = 0,
) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.full((h + has_top + has_bottom, w + has_left + has_right), int(bg), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[0, c0 : c0 + w] = int(guide_color)
        out[0, c0 : c0 + w][np.array(guide_shapes["top"], dtype=bool)] = int(bg)
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_color)
        out[-1, c0 : c0 + w][np.array(guide_shapes["bottom"], dtype=bool)] = int(bg)
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_color)
        out[r0 : r0 + h, 0][np.array(guide_shapes["left"], dtype=bool)] = int(bg)
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_color)
        out[r0 : r0 + h, -1][np.array(guide_shapes["right"], dtype=bool)] = int(bg)
    return out


def _paint_frame(
    grid: np.ndarray,
    control_sides: tuple[str, str],
    control_colors: dict[str, int],
    guide_shapes: dict[str, np.ndarray],
) -> None:
    n = grid.shape[0]
    for side in control_sides:
        color = int(control_colors[side])
        if side == "top":
            grid[0, :] = color
        elif side == "bottom":
            grid[-1, :] = color
        elif side == "left":
            grid[:, 0] = color
        else:
            grid[:, -1] = color

    if control_sides == ("top", "bottom"):
        sig_len = len(guide_shapes["top"])
        c_min = 2
        c_max = n - sig_len - 2
        if c_min > c_max:
            raise ValueError("frame too small for horizontal signatures")
        c0 = random.randint(c_min, c_max)
        top_sig = np.array(guide_shapes["top"], dtype=bool)
        bottom_sig = np.array(guide_shapes["bottom"], dtype=bool)
        grid[1, c0 : c0 + sig_len][top_sig] = int(control_colors["top"])
        grid[-2, c0 : c0 + sig_len][bottom_sig] = int(control_colors["bottom"])
    else:
        sig_len = len(guide_shapes["left"])
        r_min = 2
        r_max = n - sig_len - 2
        if r_min > r_max:
            raise ValueError("frame too small for vertical signatures")
        r0 = random.randint(r_min, r_max)
        left_sig = np.array(guide_shapes["left"], dtype=bool)
        right_sig = np.array(guide_shapes["right"], dtype=bool)
        grid[r0 : r0 + sig_len, 1][left_sig] = int(control_colors["left"])
        grid[r0 : r0 + sig_len, -2][right_sig] = int(control_colors["right"])

    for side in ("top", "bottom", "left", "right"):
        if side in control_sides:
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        for r, c in random.sample(coords, random.randint(min_count, max_count)):
            grid[r, c] = 5


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _guide_axis(sides: dict[str, np.ndarray] | tuple[str, str] | dict[str, int]) -> str:
    keys = tuple(sides) if not isinstance(sides, tuple) else sides
    return "vertical" if "top" in keys else "horizontal"


def _load_original_main():
    seed_path = ARC_UTILS_DIR / "fc7cae8d.py"
    spec = importlib.util.spec_from_file_location("fc7cae8d_seed", seed_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "main", None)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 31)
    control_sides = random.choice(OPPOSITE_PAIRS)

    all_colors = list(range(1, 10))
    ctrl_a, ctrl_b = random.sample([c for c in all_colors if c != 5], 2)
    control_colors = {control_sides[0]: ctrl_a, control_sides[1]: ctrl_b}
    remaining = [c for c in all_colors if c not in {5, ctrl_a, ctrl_b}]
    guide_color = random.choice(remaining)
    remaining = [c for c in remaining if c != guide_color]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    max_core_h = min(16, n - 6)
    max_core_w = min(16, n - 6)
    if max_core_h < 6 or max_core_w < 6:
        return None

    for _ in range(30):
        core_h = random.randint(6, max_core_h)
        core_w = random.randint(6, max_core_w)
        if core_h != core_w:
            break
    else:
        return None

    core = _build_core(core_h, core_w, fill, pattern_colors)

    sig_len = core_w if control_sides == ("top", "bottom") else core_h
    desired_shapes: dict[str, np.ndarray] = {}
    for _ in range(80):
        sig_a = _random_signature(sig_len)
        sig_b = _random_signature(sig_len)
        if not _nondegenerate_pair(sig_a, sig_b):
            continue
        desired_shapes = {control_sides[0]: sig_a, control_sides[1]: sig_b}
        aligned_object = _attach_shape_guides(core, control_sides, guide_color, desired_shapes)
        matches = _matching_transforms(aligned_object, desired_shapes, 0)
        if len(matches) == 1:
            break
    else:
        return None

    input_object = _inverse_transform(aligned_object, random.choice([1, 3]), False)

    grid = np.zeros((n, n), dtype=int)
    try:
        _paint_frame(grid, control_sides, control_colors, desired_shapes)
    except ValueError:
        return None

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    solved = main(grid)
    if not np.array_equal(solved, core):
        return None

    obj = _extract_center_object(grid, 0)
    if len(_matching_transforms(obj, desired_shapes, 0)) != 1:
        return None

    if _guide_axis(_object_signatures(obj, 0)) == _guide_axis(control_sides):
        return None
    if guide_color in set(control_colors.values()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant D1/0+D3/0")
