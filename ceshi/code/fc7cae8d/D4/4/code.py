# Variant of fc7cae8d
# Dimension: D4 — 变换类型 (transform)
# Change: 对齐后裁掉引导边（crop），输出纯核心矩形... → 条件性旋转附加
#
# Design analysis for this variant
# a) Concrete behavior:
#    The input still contains two frame controls (one horizontal edge color, one vertical edge color)
#    plus one interior rectangle carrying matching 1px guide edges. The solver first performs the
#    original task: find the dihedral transform that aligns the guide colors to the matching frame
#    sides, then crop those aligned guide edges away. The new rule is a second-stage transform:
#    after cropping, rotate the core 90 degrees CCW iff ctrl_h > ctrl_v, where ctrl_h is the color
#    on the horizontal frame control and ctrl_v is the color on the vertical frame control.
#    Small trace: suppose the frame uses top=7 and left=3, while the interior object is a rotated
#    version whose guide edges currently sit on bottom/right. Original logic rotates/flips until the
#    7-guide reaches top and the 3-guide reaches left, then crops to the raw core. New logic does
#    exactly that first, but then notices 7 > 3 and rotates the cropped core once more. If the same
#    aligned example instead had top=2 and left=6, the crop result would be returned unchanged.
# b) Complexity calibration:
#    The added rule is intentionally as light as the source task 291dc1e1: it keeps the original
#    primary transform and appends one extra orientation step keyed off a hidden property. To make
#    the extra rotation visually observable, the generator forces the core to be non-square and also
#    paints asymmetric internal motifs. This avoids accidental invisibility from symmetric objects.
# c) Implicit assumptions review:
#    - detect_controls_frame still assumes exactly one dominant horizontal control and one dominant
#      vertical control on the border; the generator preserves that.
#    - extract_center_object still assumes the target object is the largest non-border component; the
#      generator places exactly one interior guided rectangle.
#    - guide_edges still assumes the dominant color inside the object is the core fill; the generator
#      keeps fill as the majority color.
#    - The old unconditional return-after-crop would now be wrong, because it silently assumes crop
#      is the final stage. The variant fixes this by routing the cropped core through
#      _conditional_rotate before returning.
# d) Placement legality:
#    The generator still constructs the aligned object first, applies an inverse D4 transform, then
#    uses random_free_location with border=1 so the object never touches the frame, never overlaps the
#    control edges, and remains a single interior component.
# e) Visual observability:
#    The key observable metric is whether the extra-rotation branch changes the output relative to the
#    original task. The generator guarantees core_h != core_w, so every triggered sample has a swapped
#    output shape after rotation. It also balances ctrl_h > ctrl_v and ctrl_h < ctrl_v cases so a set
#    of examples can expose both branches of the hidden rule.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    H, W = arr.shape
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
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, col) for side, col, _count, _length in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
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
        return arr.copy()
    return crop_bbox(arr, best[1])


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


def _control_axes(desired: dict[str, int]) -> tuple[int, int]:
    ctrl_h = None
    ctrl_v = None
    for side, color in desired.items():
        if side in ("top", "bottom"):
            ctrl_h = int(color)
        else:
            ctrl_v = int(color)
    if ctrl_h is None or ctrl_v is None:
        raise ValueError("expected one horizontal and one vertical control")
    return int(ctrl_h), int(ctrl_v)


def _conditional_rotate(core: np.ndarray, ctrl_h: int, ctrl_v: int) -> np.ndarray:
    if int(ctrl_h) > int(ctrl_v):
        return np.rot90(core, 1)
    return core


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the guided rectangle to the frame controls, crop the matched guide edges,
    then conditionally rotate the cropped core based on the control-color ordering.

    The extra rule mirrors the complexity of reference task 291dc1e1:
    a primary geometric normalization is followed by one additional rotation
    controlled by a secondary latent property.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    ctrl_h, ctrl_v = _control_axes(desired)
    obj = _extract_center_object(g, bg)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            gedges = _guide_edges(t)
            if all(side in gedges and int(gedges[side]) == int(col) for side, col in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                cropped = t[int(top) : int(bottom), int(left) : int(right)]
                return _conditional_rotate(cropped, ctrl_h, ctrl_v)

    return obj


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    colors = list(pattern_colors)

    # Deterministic asymmetric anchors: these survive randomization and make rotation visible.
    core[0, 1 : min(w, 1 + max(2, w // 2))] = int(colors[0])
    core[min(h - 1, 1) : min(h, 1 + max(2, h // 2)), w - 2] = int(colors[1])

    elbow_r = max(1, h - 3)
    elbow_c = 0
    core[elbow_r : elbow_r + 2, elbow_c] = int(colors[2])
    core[h - 1, elbow_c : min(w, elbow_c + 3)] = int(colors[2])

    if len(colors) > 3:
        core[h // 2, max(1, w // 3)] = int(colors[3])

    area = h * w
    max_nonfill = min(area - 1, max(len(colors) + 6, int(area * 0.28)))
    target_nonfill = random.randint(len(colors) + 6, max_nonfill)
    painted = int(np.count_nonzero(core != int(fill)))

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
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _choose_controls() -> tuple[int, int]:
    usable = [1, 2, 3, 4, 6, 7, 8, 9]
    trigger = bool(random.randrange(2))
    if trigger:
        hi = random.choice(usable[1:])
        lo_choices = [c for c in usable if c < hi]
        return int(hi), int(random.choice(lo_choices))
    lo = random.choice(usable[:-1])
    hi_choices = [c for c in usable if c > lo]
    return int(lo), int(random.choice(hi_choices))


def _expected_output(core: np.ndarray, ctrl_h: int, ctrl_v: int) -> np.ndarray:
    return _conditional_rotate(core, ctrl_h, ctrl_v)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    ctrl_h, ctrl_v = _choose_controls()
    remaining = [c for c in range(1, 10) if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(3, min(4, len(remaining))))

    max_core_h = min(13, n - 4)
    max_core_w = min(15, n - 4)
    if max_core_h < 5 or max_core_w < 6:
        return None

    core_h = random.randint(5, max_core_h)
    core_w_choices = [w for w in range(6, max_core_w + 1) if w != core_h]
    if not core_w_choices:
        return None
    core_w = random.choice(core_w_choices)

    core = _build_core(core_h, core_w, fill, pattern_colors)
    expected = _expected_output(core, ctrl_h, ctrl_v)

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)

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

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D4 conditional_rotation")
