# Combo variant of fc7cae8d (index 25)
# combo_spec: D4/3+D5/0
# Name: 擦除填充+保留源
#
# Design analysis for the fused variant
# ------------------------------------
# 1. Concrete behavior trace:
#    Original fc7cae8d:
#      - detect the two colored control edges on the outer frame,
#      - rotate/reflect the interior guided rectangle until its two guide colors face
#        those same sides,
#      - crop away the matched 1px guide strips,
#      - return only that dense cropped core.
#    This fused D4/3+D5/0 variant keeps the same D4 matching step, but changes both
#    the post-transform representation and the trajectory:
#      - after crop, identify the dominant support/fill color inside the core and erase
#        it to background 0, keeping only the pattern cells,
#      - keep the original source object fully intact at its original location,
#      - preserve the full input canvas size,
#      - stamp the erased sparse core into the frame corner indicated by the two control
#        sides.
#    Small trace:
#      aligned dense core:
#          [[4, 4, 1],
#           [4, 3, 4],
#           [4, 4, 4]]
#      original task output:
#          [[4, 4, 1],
#           [4, 3, 4],
#           [4, 4, 4]]
#      fused variant sparse copy:
#          [[0, 0, 1],
#           [0, 3, 0],
#           [0, 0, 0]]
#      final output is the full original grid plus that sparse copy pasted into the
#      control-corner interior. The dense guided source and the hollow target copy are
#      deliberately contrasted.
#
# 2. Complexity calibration:
#    The generator keeps the core fill dominant so `guide_edges()` can still treat that
#    dominant color as the support color, but also forces a multi-color asymmetric motif.
#    Pattern density is capped below half of the core so the erase step remains visible.
#    The sparse motif is required to touch all four sides of the cropped core, which makes
#    the erased copy span the same rectangle footprint instead of collapsing into a tiny
#    ambiguous cluster.
#
# 3. Implicit assumption review for `main()`:
#    - `_detect_controls_frame()` assumes there are exactly two near-solid control sides.
#      The generator paints one horizontal and one vertical control edge and adds noise
#      only to the other two frame sides.
#    - `_extract_center_object()` assumes the source object is the largest interior
#      non-background component. The generator places exactly one interior guided object
#      and reserves the target corner so no second interior component competes with it.
#    - `_guide_edges()` assumes the aligned object's dominant color is the dense core fill,
#      not a guide or motif color. The generator enforces fill-majority.
#    - `_align_and_crop()` assumes the correct D4 transform is identifiable. The generator
#      avoids identity-only ambiguity by enforcing asymmetry in the erased motif and by
#      sampling only non-identity source orientations.
#    - The new keep-source logic assumes the control-corner interior is empty in the input.
#      `_reserve_corner()` guarantees that the source object cannot be placed there.
#    - The new erase step assumes the cropped dense core still has a unique dominant fill.
#      `_build_core()` and `_ensure_sparse_asymmetry()` preserve that.
#
# 4. Placement legality:
#    The frame is painted first. Then the future sparse-copy corner rectangle is reserved
#    with a one-cell moat on an occupancy canvas. The transformed source object is placed
#    only in remaining free space with `border=2`, so it stays strictly inside the frame,
#    does not overlap the reserved target corner, and remains separated from the border
#    for component extraction.
#
# 5. Visual observability and cleanliness:
#    Across examples, the input always shows a single dense guided rectangle away from the
#    frame corner. The output keeps that dense source untouched and adds a sparse hollow
#    copy in the control corner. The contrast "solid source vs hollow aligned copy" is
#    visible without relying on color ids. Reserving the target corner avoids visual
#    clutter from object overlap.
#
# 6. Color-change inferability:
#    The only color change is structural, not arbitrary: within the copied aligned core,
#    the dominant support/fill color becomes background while minority motif colors stay.
#    Because the source keeps the original dense fill, a human can compare source vs target
#    and infer that the copied version is the same aligned core with support pixels erased.

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.insert(0, ARC_UTILS_DIR)

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
    for side, vec in sides.items():
        vals = list(map(int, vec))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, col) for side, col, _count, _length in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy(), (0, 0, H - 1, W - 1)
    return crop_bbox(arr, best[1]), best[1]


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, vec in edges.items():
        vals = list(map(int, vec))
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        col, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(col)
    return out


def _crop_guides(arr: np.ndarray, sides: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in sides else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in sides else 0)
    left = 1 if "left" in sides else 0
    right = int(arr.shape[1]) - (1 if "right" in sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)]


def _align_and_crop(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            gedges = _guide_edges(transformed)
            if all(side in gedges and int(gedges[side]) == int(col) for side, col in desired.items()):
                return _crop_guides(transformed, desired)
    fallback_edges = _guide_edges(obj)
    return _crop_guides(obj, fallback_edges)


def _erase_fill(core: np.ndarray, fill: int | None = None, bg: int = 0) -> np.ndarray:
    if core.size == 0:
        return core.copy()
    erase_color = int(fill) if fill is not None else int(Counter(map(int, core.flatten())).most_common(1)[0][0])
    out = np.array(core, copy=True)
    out[out == erase_color] = int(bg)
    return out


def _corner_core_location(
    grid_shape: tuple[int, int], core_shape: tuple[int, int], desired: dict[str, int]
) -> tuple[int, int]:
    H, W = map(int, grid_shape)
    h, w = map(int, core_shape)
    r0 = 1 if "top" in desired else H - h - 1
    c0 = 1 if "left" in desired else W - w - 1
    return int(r0), int(c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the guided source object in place, align/crop its core by the frame controls,
    erase the core fill to background, and stamp that sparse copy into the control corner.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    desired = {side: col for side, col in _detect_controls_frame(g, bg)}
    obj, _bbox = _extract_center_object(g, bg)
    dense_core = _align_and_crop(obj, desired)
    sparse_core = _erase_fill(dense_core, bg=bg)

    out = g.copy()
    tr, tc = _corner_core_location(out.shape, sparse_core.shape, desired)
    paste(out, sparse_core, tr, tc)
    return out


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    min_nonfill = max(len(pattern_colors) + 5, int(np.ceil(area * 0.22)))
    max_nonfill = max(min_nonfill, int(np.floor(area * 0.36)))
    target_nonfill = random.randint(min_nonfill, max_nonfill)

    # Seed visible structure on all sides so the erased copy still spans the whole core.
    seeds = [
        (0, random.randrange(w), int(random.choice(pattern_colors))),
        (h - 1, random.randrange(w), int(random.choice(pattern_colors))),
        (random.randrange(h), 0, int(random.choice(pattern_colors))),
        (random.randrange(h), w - 1, int(random.choice(pattern_colors))),
        (0, 0, int(pattern_colors[0])),
        (h - 1, max(0, w // 2 - 1), int(pattern_colors[1 % len(pattern_colors)])),
        (max(0, h // 2), w - 1, int(pattern_colors[2 % len(pattern_colors)])),
    ]
    for r, c, color in seeds:
        core[int(r), int(c)] = int(color)

    painted = int(np.count_nonzero(core != int(fill)))
    attempts = 0
    while painted < target_nonfill and attempts < area * 12:
        attempts += 1
        color = int(random.choice(pattern_colors))
        shape_kind = random.choice(["dot", "line", "rect", "elbow"])
        if shape_kind == "dot":
            cells = [(random.randrange(h), random.randrange(w))]
        elif shape_kind == "line":
            horizontal = random.random() < 0.5
            if horizontal:
                line_len = random.randint(2, min(5, w))
                rr = random.randrange(h)
                c0 = random.randrange(w - line_len + 1)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                line_len = random.randint(2, min(5, h))
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
            if painted >= target_nonfill:
                break
            if core[r, c] != int(fill):
                continue
            core[r, c] = int(color)
            painted += 1

    # Guarantee every edge keeps at least one surviving motif cell after erase.
    edge_sets = {
        "top": [(0, c) for c in range(w)],
        "bottom": [(h - 1, c) for c in range(w)],
        "left": [(r, 0) for r in range(h)],
        "right": [(r, w - 1) for r in range(h)],
    }
    for coords in edge_sets.values():
        if all(core[r, c] == int(fill) for r, c in coords):
            r, c = random.choice(coords)
            core[r, c] = int(random.choice(pattern_colors))

    return core


def _ensure_sparse_asymmetry(core: np.ndarray, fill: int, pattern_colors: list[int]) -> np.ndarray:
    out = np.array(core, copy=True)
    non_identity = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
    for _ in range(32):
        sparse = _erase_fill(out, fill=fill, bg=0)
        symmetric = False
        for k, flip in non_identity:
            transformed = _transform(sparse, k, flip)
            if transformed.shape == sparse.shape and np.array_equal(transformed, sparse):
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
    dense_core = _align_and_crop(obj, desired)
    sparse_core = _erase_fill(dense_core, bg=bg)
    out = inp.copy()
    tr, tc = _corner_core_location(out.shape, sparse_core.shape, desired)
    paste(out, sparse_core, tr, tc)
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
    core = _ensure_sparse_asymmetry(_build_core(core_h, core_w, fill, pattern_colors), fill, pattern_colors)

    desired = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), desired)

    transforms = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
    k, flip = random.choice(transforms)
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    occupancy = grid.copy()
    _reserve_corner(occupancy, core.shape, desired, moat=1)
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
    raise RuntimeError("failed to generate input for fc7cae8d combo variant D4/3+D5/0")
