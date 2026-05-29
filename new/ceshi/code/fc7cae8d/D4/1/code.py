# Variant of fc7cae8d
# Dimension: D4 — 变换类型 (transform)
# Change: 对齐后裁掉引导边（crop），输出纯核心矩形... → 裁剪后颜色互换

from __future__ import annotations

from collections import Counter
import random
from pathlib import Path
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))
from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for this D4 variant:
# a) Concrete behavior:
#    The input keeps the original frame-edge controls and the internal guided rectangle.
#    The solver still has to infer one D4 transform that aligns the guide colors with the
#    matching frame sides. The changed step happens after alignment: instead of returning
#    the cropped core directly, crop first and then swap the two pattern colors inside
#    the core. Example: if aligned crop is
#      11121
#      13331
#      11131
#    with fill=1 and pattern colors {2,3}, the variant output is
#      11131
#      12221
#      11121
#    So the pipeline changes only at the final output stage:
#    original = align -> crop -> return core
#    variant  = align -> crop -> swap the two non-fill colors -> return swapped core
# b) Complexity calibration:
#    To make the color-swap rule visible and non-degenerate, the generator now uses exactly
#    two pattern colors, forces both to appear multiple times, and rejects D4-symmetric
#    cores. This prevents samples where the transform is visually ambiguous or where the
#    post-crop swap has little evidence.
# c) Hidden assumptions in the original solver:
#    - detect_controls_frame() assumes two strong border controls with near-solid colors.
#      This still holds; generator keeps the same frame construction.
#    - extract_center_object() assumes the guided object is the largest non-border component.
#      Still valid because only the output rule changes.
#    - guide_edges() assumes the core fill is the dominant object color. This becomes a real
#      requirement once the core is restricted to exactly two non-fill colors, so generator
#      now caps pattern density to keep fill dominant.
#    - The original return slice was final output. In the variant that slice is only an
#      intermediate "aligned core"; returning it unchanged would silently collapse back to
#      the old task, so main() must explicitly post-process that cropped array.
# d) Placement legality:
#    The generator still builds the aligned guided object first, applies an inverse D4
#    transform, and then uses random_free_location(..., border=1) before pasting with
#    transparent background. This preserves non-overlap, in-bounds placement, and the
#    solver's expectation that the central object does not touch the frame.
# e) Visual observability:
#    Across a few examples, a human can still see that guide colors determine orientation,
#    but now the output colors are not copied verbatim from the aligned crop. Because the
#    generator enforces two visible pattern colors and D4-asymmetric cores, both the
#    orientation step and the final color-swap step remain observable from examples.
# Reference note:
#    The requested source task a25697e4 is available here only as evaluation data plus
#    the merged mechanism annotation "swap colors of the two components of s if flip is
#    used"; its standalone solver code is not present in the provided files. This variant
#    uses that color-swap mechanism as the baseline transform change.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a central rectangle to two colored frame-edge controls via a dihedral transform.
    After removing the aligned guide edges, swap the two pattern colors in the cropped core.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def detect_controls_frame(arr: np.ndarray) -> list[tuple[str, int]]:
        sides = {
            "top": arr[0, 1:-1],
            "bottom": arr[-1, 1:-1],
            "left": arr[1:-1, 0],
            "right": arr[1:-1, -1],
        }
        cand: list[tuple[str, int, int, int]] = []
        for side, v in sides.items():
            values = list(map(int, v))
            non_bg = [x for x in values if x != bg]
            if not non_bg:
                continue
            col, count = Counter(non_bg).most_common(1)[0]
            if int(count) >= len(values) - 1:
                cand.append((side, int(col), int(count), int(len(values))))
        cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
        return [(side, col) for side, col, _count, _length in cand[:2]]

    def extract_center_object(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        comps = find_components(arr, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
        best = None
        for comp in comps:
            r0, c0, r1, c1 = map(int, comp["bbox"])
            if bool(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
                continue
            area = int(comp["area"])
            if best is None or area > best[0]:
                best = (area, (r0, c0, r1, c1))
        if best is None:
            return arr.copy()
        return crop_bbox(arr, best[1])

    def guide_edges(arr: np.ndarray) -> dict[str, int]:
        fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
        edges = {
            "top": arr[0, :],
            "bottom": arr[-1, :],
            "left": arr[:, 0],
            "right": arr[:, -1],
        }
        out: dict[str, int] = {}
        for side, v in edges.items():
            values = list(map(int, v))
            non_fill = [x for x in values if x != fill]
            if not non_fill:
                continue
            col, count = Counter(non_fill).most_common(1)[0]
            if int(count) >= len(values) - 1:
                out[side] = int(col)
        return out

    def swap_pattern_colors(arr: np.ndarray) -> np.ndarray:
        counts = Counter(map(int, arr.flatten()))
        fill = int(counts.most_common(1)[0][0])
        pattern_colors = sorted(col for col in counts if int(col) != fill)
        if len(pattern_colors) != 2:
            return arr.copy()
        a, b = map(int, pattern_colors)
        out = arr.copy()
        mask_a = arr == a
        mask_b = arr == b
        out[mask_a] = b
        out[mask_b] = a
        return out

    desired = {side: col for side, col in detect_controls_frame(g)}
    obj = extract_center_object(g)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            gedges = guide_edges(t)
            if all(side in gedges and int(gedges[side]) == int(col) for side, col in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                core = t[int(top) : int(bottom), int(left) : int(right)]
                return swap_pattern_colors(core)

    return obj


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _swap_pattern_colors(core: np.ndarray, fill: int, pattern_colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, pattern_colors)
    out = core.copy()
    out[core == a] = b
    out[core == b] = a
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
    seen = []
    for variant in _d4_variants(arr):
        if any(np.array_equal(variant, prev) for prev in seen):
            return False
        seen.append(variant)
    return True


def _build_core(h: int, w: int, fill: int, pattern_colors: tuple[int, int]) -> np.ndarray | None:
    a, b = map(int, pattern_colors)
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    min_per_color = max(4, area // 18)
    max_nonfill = max(2 * min_per_color, int(area * 0.36))
    total_nonfill = random.randint(2 * min_per_color, max_nonfill)
    targets = [min_per_color, min_per_color]
    extra = total_nonfill - 2 * min_per_color
    if extra > 0:
        give_a = random.randint(0, extra)
        targets[0] += give_a
        targets[1] += extra - give_a

    for color, target in zip((a, b), targets):
        painted = 0
        attempts = 0
        while painted < target and attempts < area * 10:
            attempts += 1
            shape_kind = random.choice(["dot", "line", "rect", "el"])
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
    if np.array_equal(core, _swap_pattern_colors(core, fill, (a, b))):
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


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = tuple(random.sample(remaining, 2))

    max_core_h = min(20, n - 3)
    max_core_w = min(20, n - 3)
    if max_core_h < 6 or max_core_w < 6:
        return None
    core_h = random.randint(6, max_core_h)
    core_w = random.randint(6, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)
    if core is None:
        return None
    expected = _swap_pattern_colors(core, fill, pattern_colors)

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
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D4 variant")
