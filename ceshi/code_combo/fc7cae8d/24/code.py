# Combo variant of fc7cae8d (index 24)
# combo_spec: D4/1+D5/0
# Name: 颜色互换+保留源

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))
from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for combo D4/1 + D5/0
# --------------------------------------
# a) Concrete behavior and trace:
#    The original task detects two frame-edge control colors, aligns the guided interior
#    rectangle by one D4 transform, crops away the matched 1-cell guides, and returns only
#    the cropped core.
#
#    The fused task keeps the same first half:
#      detect controls -> align source object -> crop matched guide edges
#    but then changes the final half in two coupled ways:
#      crop core -> swap the two pattern colors in that core -> paste the swapped core into
#      the interior control corner while preserving the entire original input canvas.
#
#    Small trace with fill=1, pattern colors {2,3}, controls top+left:
#      aligned guided object:
#        02220
#        21131
#        21111
#        23331
#      cropped aligned core:
#        1131
#        1111
#        3331
#      swapped core:
#        1121
#        1111
#        2221
#      original task output:
#        1131 / 1111 / 3331
#      fused output:
#        full input grid, source object unchanged in place, plus an extra corner copy
#        1121 / 1111 / 2221 pasted just inside the control corner.
#
#    The difference is not "do D4 then do D5 separately". The same inferred aligned core is
#    the single intermediate object that both defines the color swap and the kept-source copy.
#
# b) Complexity calibration:
#    The generator uses exactly two pattern colors, repeats both several times, keeps fill
#    dominant, and rejects D4-symmetric cores. That prevents ambiguous guide detection,
#    makes the swap visible, and ensures the copied corner patch is informative rather than
#    degenerate. The source orientation is sampled from a non-identity transform so examples
#    do not collapse into "already aligned, just duplicate it".
#
# c) Hidden assumption audit against the original main():
#    - Control detection still assumes two near-solid frame edges. Kept unchanged.
#    - Center-object extraction still assumes the guided object is the largest interior
#      non-border component. The generator now reserves the output corner region so there is
#      no competing interior object in the input.
#    - Guide-edge detection assumes the core fill is the dominant object color. The generator
#      enforces sparse enough pattern density and exactly two non-fill colors.
#    - The original direct return of the cropped core is no longer valid. In the fused rule,
#      the cropped core is only an intermediate that must be swapped and then pasted onto a
#      copy of the full input grid.
#    - New assumption introduced here: the cropped core contains exactly two pattern colors.
#      The generator guarantees that by construction and rejects failed builds.
#
# d) Placement legality:
#    The corner destination is the interior corner determined by the two controls:
#      top+left -> (1,1), top+right -> (1,W-w-1), bottom+left -> (H-h-1,1),
#      bottom+right -> (H-h-1,W-w-1).
#    The generator reserves that rectangle plus a 1-cell moat before placing the source
#    object, so source and pasted copy never overlap and both stay strictly inside bounds.
#
# e) Visual observability:
#    A solver can compare the unchanged source object with the extra corner copy and see:
#    (1) the corner copy matches the source only after undoing the source D4 transform,
#    and (2) after alignment, the two pattern colors have been exchanged. Because the source
#    stays visible, the swap has an in-grid color key instead of being a hidden remapping.
#
# f) Visual cleanliness:
#    Only three visual layers exist: frame controls on the border, one guided source object
#    in the interior, and one unguided corner copy in an empty reserved corner. The copy is
#    not stacked on top of the source, so the grid stays readable.
#
# g) Color-change inferability:
#    Restricting to exactly two pattern colors makes the color transform a simple swap rather
#    than an arbitrary palette permutation. Both colors appear multiple times, and the source
#    object remains visible in original colors, giving repeated local evidence for the swap.
#
# h) Post-test fusion metric:
#    A generated sample counts as non-degenerate only if:
#      - output shape equals input shape (keep-source is active),
#      - the corner patch equals the swapped aligned core,
#      - the corner patch differs from the unswapped aligned core on at least one cell.
#    This separates the true fusion behavior from samples that would also satisfy plain D5.
#
# i) New solver assumptions and generator guarantees:
#    - `_align_and_crop()` assumes one D4 orientation matches both control colors. The
#      generator builds the object from the aligned state and applies one inverse D4 transform.
#    - `_swap_two_pattern_colors()` assumes exactly two non-fill colors in the cropped core.
#      The generator samples exactly two pattern colors and checks both survive in the core.
#    - Corner pasting assumes the destination is background in the input. The generator
#      reserves that region before placing the source object.


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
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
    cand.sort(key=lambda item: (item[2] / item[3], item[2]), reverse=True)
    return [(side, col) for side, col, _count, _length in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy(), (0, 0, h - 1, w - 1)
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
    for side, values in edges.items():
        vals = list(map(int, values))
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


def _swap_two_pattern_colors(core: np.ndarray) -> np.ndarray:
    counts = Counter(map(int, core.flatten()))
    fill = int(counts.most_common(1)[0][0])
    pattern_colors = sorted(int(col) for col in counts if int(col) != fill)
    if len(pattern_colors) != 2:
        return core.copy()
    a, b = pattern_colors
    out = core.copy()
    mask_a = core == a
    mask_b = core == b
    out[mask_a] = b
    out[mask_b] = a
    return out


def _corner_core_location(
    grid_shape: tuple[int, int], core_shape: tuple[int, int], desired: dict[str, int]
) -> tuple[int, int]:
    h, w = map(int, core_shape)
    grid_h, grid_w = map(int, grid_shape)
    r0 = 1 if "top" in desired else grid_h - h - 1
    c0 = 1 if "left" in desired else grid_w - w - 1
    return int(r0), int(c0)


def _expected_output(inp: np.ndarray) -> np.ndarray:
    bg = int(get_bg(inp))
    desired = {side: col for side, col in _detect_controls_frame(inp, bg)}
    obj, _bbox = _extract_center_object(inp, bg)
    aligned_core = _align_and_crop(obj, desired)
    swapped_core = _swap_two_pattern_colors(aligned_core)
    out = np.array(inp, dtype=int, copy=True)
    tr, tc = _corner_core_location(out.shape, swapped_core.shape, desired)
    paste(out, swapped_core, tr, tc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the full canvas, preserve the original guided source object, and add one extra copy:
    the source aligned to the control corner, cropped to its core, and with its two pattern
    colors swapped.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    desired = {side: col for side, col in _detect_controls_frame(g, bg)}
    obj, _bbox = _extract_center_object(g, bg)
    aligned_core = _align_and_crop(obj, desired)
    swapped_core = _swap_two_pattern_colors(aligned_core)

    out = g.copy()
    tr, tc = _corner_core_location(out.shape, swapped_core.shape, desired)
    paste(out, swapped_core, tr, tc)
    return out


def _counts_by_color(arr: np.ndarray) -> dict[int, int]:
    return {int(col): int(count) for col, count in Counter(map(int, arr.flatten())).items()}


def _d4_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        base = np.rot90(arr, int(k))
        variants.append(base)
        variants.append(np.fliplr(base))
    return variants


def _is_d4_distinct(arr: np.ndarray) -> bool:
    seen: list[np.ndarray] = []
    for variant in _d4_variants(arr):
        if any(np.array_equal(variant, prev) for prev in seen):
            return False
        seen.append(variant)
    return True


def _build_core(h: int, w: int, fill: int, pattern_colors: tuple[int, int]) -> np.ndarray | None:
    a, b = map(int, pattern_colors)
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    min_per_color = max(5, area // 18)
    max_nonfill = min(area - 1, max(2 * min_per_color + 2, int(area * 0.34)))
    if max_nonfill < 2 * min_per_color:
        return None

    total_nonfill = random.randint(2 * min_per_color, max_nonfill)
    targets = [min_per_color, min_per_color]
    extra = total_nonfill - 2 * min_per_color
    if extra > 0:
        extra_a = random.randint(0, extra)
        targets[0] += extra_a
        targets[1] += extra - extra_a

    signature = [
        (0, 0, a),
        (h - 1, max(0, w // 2 - 1), b),
        (max(0, h // 2 - 1), w - 1, a),
        (min(h - 1, 1), min(w - 1, 2), b),
    ]
    for r, c, color in signature:
        core[int(r), int(c)] = int(color)

    seeded = [int(np.count_nonzero(core == a)), int(np.count_nonzero(core == b))]
    for idx, color in enumerate((a, b)):
        painted = seeded[idx]
        attempts = 0
        target = targets[idx]
        while painted < target and attempts < area * 12:
            attempts += 1
            shape_kind = random.choice(["dot", "line", "rect", "elbow"])
            if shape_kind == "dot":
                cells = [(random.randrange(h), random.randrange(w))]
            elif shape_kind == "line":
                line_len = random.randint(2, min(6, max(h, w)))
                if random.random() < 0.5:
                    rr = random.randrange(h)
                    c0 = random.randrange(w - line_len + 1)
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    r0 = random.randrange(h - line_len + 1)
                    cc = random.randrange(w)
                    cells = [(r0 + dr, cc) for dr in range(line_len)]
            elif shape_kind == "rect":
                rh = random.randint(1, min(4, h))
                rw = random.randint(1, min(4, w))
                r0 = random.randrange(h - rh + 1)
                c0 = random.randrange(w - rw + 1)
                cells = [(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw)]
            else:
                rh = random.randint(2, min(4, h))
                rw = random.randint(2, min(4, w))
                r0 = random.randrange(h - rh + 1)
                c0 = random.randrange(w - rw + 1)
                cells = [(r0 + dr, c0) for dr in range(rh)] + [(r0 + rh - 1, c0 + dc) for dc in range(rw)]

            random.shuffle(cells)
            for r, c in cells:
                if core[r, c] != fill:
                    continue
                core[r, c] = int(color)
                painted += 1
                if painted >= target:
                    break
        if painted < target:
            return None

    counts = _counts_by_color(core)
    if counts.get(int(fill), 0) <= max(counts.get(a, 0), counts.get(b, 0)):
        return None
    if counts.get(a, 0) < min_per_color or counts.get(b, 0) < min_per_color:
        return None
    if not _is_d4_distinct(core):
        return None
    if np.array_equal(core, _swap_two_pattern_colors(core)):
        return None
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
    pattern_colors = tuple(random.sample(remaining, 2))

    max_core_h = min(12, n - 8)
    max_core_w = min(12, n - 8)
    if max_core_h < 6 or max_core_w < 6:
        return None

    core_h = random.randint(6, max_core_h)
    core_w_choices = [w for w in range(6, max_core_w + 1) if w != core_h]
    core_w = random.choice(core_w_choices) if core_w_choices else random.randint(6, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)
    if core is None:
        return None

    desired = {horiz_side: ctrl_h, vert_side: ctrl_v}
    swapped_core = _swap_two_pattern_colors(core)
    if np.array_equal(core, swapped_core):
        return None

    aligned_object = _attach_guides(core, (horiz_side, vert_side), desired)
    non_identity = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
    k, flip = random.choice(non_identity)
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

    tr, tc = _corner_core_location(grid.shape, core.shape, desired)
    actual_patch = expected[tr : tr + core.shape[0], tc : tc + core.shape[1]]
    if not np.array_equal(actual_patch, swapped_core):
        return None
    if np.array_equal(actual_patch, core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant")
