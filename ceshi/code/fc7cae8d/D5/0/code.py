# Variant of fc7cae8d
# Dimension: D5 — 轨迹/源对象保留 (trajectory)
# Change: 默认行为——源对象在输入中存在，输出中被裁剪变换后替代，无轨迹
#   → 保留源对象原位；输出保持原尺寸；额外把对齐后裁边得到的核心复制到控制边定义的内角位置。

from __future__ import annotations

import sys
from collections import Counter
import random

import numpy as np

sys.path.append("/home/lixinyu/snap/new/arc2_solver_select")
from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the D5 keep-source variant
# ----------------------------------------------
# 1. Concrete behavior:
#    Original fc7cae8d returns only the cropped core. This variant keeps the full input canvas.
#    The interior source object stays exactly where it started. We still infer the correct dihedral
#    transform from the colored guide edges, crop those guide edges away, and then stamp the core
#    into the control corner's *interior* location:
#      - controls on top+left  -> core top-left at (1, 1)
#      - controls on top+right -> core top-left at (1, W-core_w-1)
#      - controls on bottom+left -> core top-left at (H-core_h-1, 1)
#      - controls on bottom+right -> core top-left at (H-core_h-1, W-core_w-1)
#    Small trace:
#      - old rule: object with guides is aligned, guides are cut, output is only that cropped core.
#      - new rule: the same aligned-and-cropped core is produced, but instead of replacing the whole
#        output, it is copied into the frame corner while the original source object remains visible.
#      - difference: the solver now needs "copy + transform", not "move + crop".
#
# 2. Complexity calibration:
#    If the core were rotationally/reflection symmetric, the new copy could look identical under
#    multiple transforms and the D5 change would be hard to see. The generator therefore forces a
#    visibly asymmetric interior pattern and samples only non-identity input transforms.
#
# 3. Old assumptions that break under D5:
#      - `main()` formerly returned only the cropped core; that assumes the source object disappears.
#      - unconditional `return t[...]` is no longer valid because the original canvas must survive.
#      - pasting into the control corner assumes that corner interior is empty in the input. The
#        generator now reserves that region so the copied core does not overwrite the source object.
#
# 4. Placement legality:
#    The generator paints the border controls first, reserves the target corner interior plus a
#    one-cell moat, and only then places the source object in a remaining free location. This
#    guarantees: no overlap with borders, no overlap with the copied target core, and a source bbox
#    that still satisfies the solver's "largest interior component" assumption.
#
# 5. Visual observability:
#    Across examples, a human sees the original guided rectangle still present and an extra unguided
#    copy appearing in the frame corner. Because the source orientation is always a non-identity
#    transform of the aligned state, the change is visually distinguishable from a trivial duplicate.


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
    for side, v in sides.items():
        vals = list(map(int, v))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(s, c) for s, c, _cnt, _L in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None  # (area, bbox)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1):
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy(), (0, 0, H - 1, W - 1)
    _area, bbox = best
    return crop_bbox(arr, bbox), bbox


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, v in edges.items():
        vals = list(map(int, v))
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        col, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(col)
    return out


def _align_and_crop(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray:
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
                return t[int(top) : int(bottom), int(left) : int(right)]
    raise ValueError("no matching dihedral transform found")


def _corner_core_location(grid_shape: tuple[int, int], core_shape: tuple[int, int], desired: dict[str, int]) -> tuple[int, int]:
    H, W = map(int, grid_shape)
    h, w = map(int, core_shape)
    r0 = 1 if "top" in desired else H - h - 1
    c0 = 1 if "left" in desired else W - w - 1
    return int(r0), int(c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the interior guided rectangle to the frame controls, but keep the source object in place.

    Output rule for this variant:
    - detect the two control sides on the outer frame,
    - rotate/reflect the interior object so its guide colors face those sides,
    - crop the matched guide edges away,
    - copy that cropped core into the control corner interior,
    - leave the original source object untouched and preserve the full grid size.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    obj, _bbox = _extract_center_object(g, bg)
    core = _align_and_crop(obj, desired)

    out = g.copy()
    tr, tc = _corner_core_location(out.shape, core.shape, desired)
    paste(out, core, tr, tc)
    return out


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors) + 4, int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors) + 3, area // 8), max_nonfill)

    # Seed an unmistakably asymmetric signature before adding random texture.
    signature = [
        (0, 0),
        (h - 1, max(0, w // 2 - 1)),
        (max(0, h // 2 - 1), w - 1),
        (min(h - 1, 1), min(w - 1, 2)),
    ]
    for idx, (r, c) in enumerate(signature):
        core[int(r), int(c)] = int(pattern_colors[idx % len(pattern_colors)])

    painted = int(np.count_nonzero(core != int(fill)))
    while painted < target_nonfill:
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

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        fill_cells = [(r, c) for r, c in cells if core[r, c] == int(fill)]
        if len(fill_cells) > remaining:
            fill_cells = random.sample(fill_cells, remaining)
        for r, c in fill_cells:
            core[r, c] = int(color)
        painted = int(np.count_nonzero(core != int(fill)))

    return core


def _ensure_asymmetry(core: np.ndarray, fill: int, pattern_colors: list[int]) -> np.ndarray:
    out = core.copy()
    non_identity = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
    for _ in range(24):
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


def _reserve_corner(canvas: np.ndarray, core_shape: tuple[int, int], desired: dict[str, int], moat: int = 1) -> None:
    h, w = map(int, core_shape)
    r0, c0 = _corner_core_location(canvas.shape, core_shape, desired)
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    rr0 = max(0, r0 - moat)
    cc0 = max(0, c0 - moat)
    rr1 = min(canvas.shape[0] - 1, r1 + moat)
    cc1 = min(canvas.shape[1] - 1, c1 + moat)
    canvas[rr0 : rr1 + 1, cc0 : cc1 + 1] = 9


def _expected_output(inp: np.ndarray) -> np.ndarray:
    bg = int(get_bg(inp))
    desired = {side: col for side, col in _detect_controls_frame(inp, bg)}
    obj, _bbox = _extract_center_object(inp, bg)
    core = _align_and_crop(obj, desired)
    out = inp.copy()
    tr, tc = _corner_core_location(out.shape, core.shape, desired)
    paste(out, core, tr, tc)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 30)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(12, n - 8)
    max_core_w = min(12, n - 8)
    if max_core_h < 5 or max_core_w < 5:
        return None

    core_h = random.randint(5, max_core_h)
    core_w_choices = [w for w in range(5, max_core_w + 1) if w != core_h]
    core_w = random.choice(core_w_choices) if core_w_choices else random.randint(5, max_core_w)
    core = _ensure_asymmetry(_build_core(core_h, core_w, fill, pattern_colors), fill, pattern_colors)

    desired = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), desired)

    transforms = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
    k, flip = random.choice(transforms)
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    occupancy = grid.copy()
    _reserve_corner(occupancy, core.shape, desired, moat=1)
    # `extract_center_object()` searches for the largest non-border connected component with
    # monochromatic=False, so the source object must stay at least one full blank moat away from
    # the non-bg border frame. Using border=2 prevents accidental 4-neighbor attachment.
    try:
        r0, c0 = random_free_location(occupancy, input_object, transparent=0, border=2)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    expected = _expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D5 variant")
