# Combo variant of fc7cae8d (index 33)
# combo_spec: D6/1+D8/1
# Name: 含控制边裁剪+双层迭代裁剪
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace.
#    Original task: one guide layer sits on two sides of the central rectangle.
#    Solver finds the frame's horizontal/vertical control sides, applies one
#    dihedral transform so the guide colors face those same sides, then crops
#    away that single guide layer and returns the bare core.
#
#    Fused task:
#    - The object now contains two nested guide layers that reuse the same two
#      control colors. The outer layer is on the same corner as the frame
#      controls. The inner layer is on the opposite corner.
#    - Step 1: align the outer layer to the frame and crop it away.
#    - Step 2: the newly exposed inner layer is still on the opposite corner,
#      so it must be aligned again to the frame and cropped away.
#    - Output is not the naked core. We re-wrap the final core with a compact
#      1px control-colored border on the frame-controlled sides.
#
#    Small example with desired controls on top+left:
#    - Canonical aligned object before scrambling:
#        outer(top,left) around inner(bottom,right) around core
#    - After step 1 crop:
#        inner(bottom,right) around core
#    - After step 2 alignment+crop:
#        rotated/reflected core only
#    - Final output:
#        core with a compact top+left control border.
#
# b) Complexity calibration.
#    The inner layer is forced to the opposite corner, so the second step is
#    always meaningful and visually obvious, but the number of relation types is
#    still small enough for humans to infer from 2-4 examples.
#
# c) Hidden-assumption audit against the original solver.
#    - Original `guide_edges()` assumed the only meaningful guide layer sat on
#      the bbox edge. This remains true per stage, but now it must be applied
#      twice on two different cropped objects.
#    - Original `main()` assumed one crop finished the task. That is false now;
#      after the first crop the exposed object must still have full-length guide
#      stripes on its bbox.
#    - Original output assumed "crop to core". D6/1 breaks that: the final
#      answer must preserve control-side context as a compact border.
#
# d) Placement legality.
#    The generator builds one connected object, scrambles it with a single
#    dihedral transform, and places it strictly inside the frame with a 1-cell
#    margin, so it never overlaps the frame or leaves the grid.
#
# e) Visual observability.
#    Humans can see two nested control-colored stripes in the input: one on the
#    outside and one one-cell inward on the opposite corner. The output keeps a
#    compact control-colored border, making "what is preserved" visible.
#
# f) Visual cleanliness.
#    Only one object is placed. Guide colors are reused from the frame, while
#    internal pattern colors exclude control and noise colors. This keeps the
#    nested structure readable instead of cluttered.
#
# g) Color inference.
#    No arbitrary recoloring is introduced. The same two control colors appear
#    on the frame, on both guide layers, and on the retained output border, so
#    the color semantics stay local and structural.

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


SIDE_ORDER = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = ("top", "bottom")
VERTICAL_SIDES = ("left", "right")
CORNER_SIDES = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
]


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _opposite_side(side: str) -> str:
    return {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }[side]


def _opposite_corner(sides: tuple[str, str]) -> tuple[str, str]:
    return (_opposite_side(sides[0]), _opposite_side(sides[1]))


def _detect_controls_frame(arr: np.ndarray, bg: int) -> dict[str, int]:
    def detect(side: str) -> tuple[str, int, float] | None:
        if side == "top":
            line = arr[0, 1:-1]
        elif side == "bottom":
            line = arr[-1, 1:-1]
        elif side == "left":
            line = arr[1:-1, 0]
        else:
            line = arr[1:-1, -1]
        vals = [int(x) for x in line]
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            return None
        color, count = Counter(non_bg).most_common(1)[0]
        ratio = float(count) / float(len(vals))
        if ratio < 0.85:
            return None
        return side, int(color), ratio

    horiz = [cand for cand in (detect("top"), detect("bottom")) if cand is not None]
    vert = [cand for cand in (detect("left"), detect("right")) if cand is not None]
    if len(horiz) != 1 or len(vert) != 1:
        raise ValueError("frame controls not uniquely detectable")
    hs, hc, _ = horiz[0]
    vs, vc, _ = vert[0]
    return {hs: int(hc), vs: int(vc)}


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        raise ValueError("no interior object found")
    return crop_bbox(arr, best[1])


def _visible_guides(arr: np.ndarray, allowed_colors: set[int]) -> dict[str, int]:
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, line in edges.items():
        counts = Counter(int(x) for x in line if int(x) in allowed_colors)
        if not counts:
            continue
        color, count = counts.most_common(1)[0]
        if int(count) >= len(line) - 1:
            out[side] = int(color)
    return out


def _align_object_to_controls(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    allowed = set(map(int, desired.values()))
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            cand = np.fliplr(base) if flip else base
            guides = _visible_guides(cand, allowed)
            if all(guides.get(side) == int(color) for side, color in desired.items()):
                return cand
    raise ValueError("no dihedral transform matches the control sides")


def _crop_control_guides(arr: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in desired else 0
    bottom = arr.shape[0] - (1 if "bottom" in desired else 0)
    left = 1 if "left" in desired else 0
    right = arr.shape[1] - (1 if "right" in desired else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)].copy()


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


def _solve_object(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    outer_aligned = _align_object_to_controls(obj, desired)
    after_outer_crop = _crop_control_guides(outer_aligned, desired)
    inner_aligned = _align_object_to_controls(after_outer_crop, desired)
    core = _crop_control_guides(inner_aligned, desired)
    return _attach_guides(core, tuple(desired.keys()), desired)


def _single_step_output(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    outer_aligned = _align_object_to_controls(obj, desired)
    return _crop_control_guides(outer_aligned, desired)


def fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    """
    Quantify whether the generated sample really exhibits the fused behavior.

    Metrics used in manual validation:
    - `inner_sides_after_first` should be the opposite corner of `desired_sides`.
    - `same_as_single_step` should be False, proving the second crop/context kept
      the sample from collapsing back to the original one-step behavior.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_controls_frame(g, bg)
    obj = _extract_center_object(g, bg)
    outer_aligned = _align_object_to_controls(obj, desired)
    after_outer = _crop_control_guides(outer_aligned, desired)
    inner_sides = tuple(side for side in SIDE_ORDER if side in _visible_guides(after_outer, set(desired.values())))
    fused = _solve_object(obj, desired)
    single = _single_step_output(obj, desired)
    return {
        "desired_sides": tuple(side for side in SIDE_ORDER if side in desired),
        "inner_sides_after_first": inner_sides,
        "same_as_single_step": bool(np.array_equal(fused, single)),
        "fused_shape": tuple(map(int, fused.shape)),
        "single_shape": tuple(map(int, single.shape)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_controls_frame(g, bg)
    obj = _extract_center_object(g, bg)
    return _solve_object(obj, desired)


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
            line_len = random.randint(2, min(5, max(h, w)))
            if random.random() < 0.5:
                rr = random.randrange(h)
                c0 = random.randrange(max(1, w - line_len + 1))
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                r0 = random.randrange(max(1, h - line_len + 1))
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
            if core[r, c] != fill:
                continue
            if painted >= target_nonfill:
                break
            core[r, c] = int(color)
            painted += 1

    return core


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

    for side in SIDE_ORDER:
        if side in (horiz_side, vert_side):
            continue
        if side in HORIZONTAL_SIDES:
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


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 30)
    outer_sides = random.choice(CORNER_SIDES)
    inner_sides = _opposite_corner(outer_sides)
    horiz_side, vert_side = outer_sides

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    max_core_h = min(15, n - 6)
    max_core_w = min(15, n - 6)
    if max_core_h < 5 or max_core_w < 5:
        return None
    core_h = random.randint(5, max_core_h)
    core_w = random.randint(5, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v, inner_sides[0]: ctrl_h, inner_sides[1]: ctrl_v}
    inner_aligned = _attach_guides(core, inner_sides, guide_colors)
    outer_aligned = _attach_guides(inner_aligned, outer_sides, guide_colors)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(outer_aligned, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1, padding=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)
    if int(get_bg(grid)) != 0:
        return None

    expected = _solve_object(input_object, {horiz_side: ctrl_h, vert_side: ctrl_v})
    if not np.array_equal(main(grid), expected):
        return None
    metrics = fusion_metrics(grid)
    if metrics["same_as_single_step"]:
        return None
    if tuple(metrics["inner_sides_after_first"]) != tuple(inner_sides):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fused fc7cae8d")
