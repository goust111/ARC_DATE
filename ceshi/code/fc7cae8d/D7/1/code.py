# Variant of fc7cae8d
# Dimension: D7 — 源-目标基数关系 (meta)
# Change: 1:1——单个源对象对应单个目标角，一次对齐操作... → 多对象同角 (many:1)
#
# Design analysis for this variant:
# - Reference-task step: the variant spec lists no separate source_task, so the baseline is the
#   original fc7cae8d solver/generator only. The preserved mechanism is still "read two frame
#   controls -> find guide-edge colors on rectangle objects -> apply a dihedral transform so the
#   guides face the same frame sides -> remove the guides".
# - Concrete many:1 behavior: the input now contains 2-3 separate inner rectangles. Every object
#   uses the same two guide colors, so all of them point to the same frame corner. After each
#   object is aligned and cropped, its core is anchored to that shared corner on one output canvas.
#   Example: if the controls are top+left and the cropped cores are 12x10, 8x13, and 6x7, then the
#   output canvas is 12x13. All three cores are pasted with top-left anchoring; conflicts are
#   resolved by priority so smaller-area cores overwrite larger ones.
# - Why this is visibly different from the original: the original pipeline extracts one largest
#   center component and returns one cropped core. That would silently ignore extra rectangles here.
#   In the variant, multiple scattered input objects all survive into the output, and the output is
#   a layered merge rather than a single extracted rectangle.
# - Complexity calibration: the generator gives each core a distinct fill color, unique area, and
#   asymmetric interior motifs. Unique areas eliminate priority ties; asymmetry makes the required
#   rotation/reflection visible and prevents the variant from collapsing into a trivial "same shape"
#   case.
# - Implicit-assumption audit of the original code:
#   1. `extract_center_object()` assumed the largest internal component is the only relevant one.
#      This fails under many:1, so the new solver scans all internal multicolor components.
#   2. The original `main()` returned after the first matching transform. That is only valid for
#      1:1; the new solver aligns every valid object, then merges their cropped cores.
#   3. The original output size equaled one cropped core. That no longer holds; the merged output
#      uses the max height/width needed to place all aligned cores on the same target corner.
# - Placement legality in the generator: objects are still placed with `random_free_location`, one
#   by one, so they do not overlap or touch the border. The largest core defines the final output
#   canvas, all guide-bearing objects fit inside the input grid with a one-cell margin, and every
#   generated sample is accepted only if `main(input)` reproduces the constructed layered output.
# - Visual observability check: a human now sees multiple interior rectangles carrying the same two
#   guide colors toward the same frame corner, while the output shows a single composite rectangle
#   with visible bands/patches contributed by different source objects. This makes the many:1
#   cardinality change and the overlap-resolution rule inspectable from a few ARC examples.
# - New solver assumptions introduced here:
#   1. Each relevant object is an internal 4-connected multicolor component.
#   2. Valid objects have exactly the desired two guide sides after some dihedral transform.
#   3. Priority is ordered by unique core area (smaller area = higher priority / pasted later).
#   The generator enforces all three assumptions.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[5]
    sys.path.append(str(ROOT / "arc2_solver_select"))
    from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    H, W = arr.shape
    sides = {
        "top": arr[0, 1 : W - 1],
        "bottom": arr[H - 1, 1 : W - 1],
        "left": arr[1 : H - 1, 0],
        "right": arr[1 : H - 1, W - 1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, values in sides.items():
        v = list(map(int, values))
        non_bg = [x for x in v if x != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(v) - 1:
            cand.append((side, int(color), int(count), int(len(v))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, color) for side, color, _count, _length in cand[:2]]


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
        v = list(map(int, values))
        non_fill = [x for x in v if x != int(fill)]
        if not non_fill:
            continue
        color, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(v) - 1:
            out[side] = int(color)
    return out


def _candidate_objects(arr: np.ndarray, bg: int) -> list[np.ndarray]:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    objs: list[np.ndarray] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        objs.append(crop_bbox(arr, (r0, c0, r1, c1)))
    objs.sort(key=lambda obj: obj.shape[0] * obj.shape[1], reverse=True)
    return objs


def _crop_aligned_object(arr: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in desired else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in desired else 0)
    left = 1 if "left" in desired else 0
    right = int(arr.shape[1]) - (1 if "right" in desired else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)]


def _align_object(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray | None:
    matches: list[np.ndarray] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed)
            if set(guides) != set(desired):
                continue
            if all(int(guides[side]) == int(color) for side, color in desired.items()):
                matches.append(_crop_aligned_object(transformed, desired))
    if not matches:
        return None
    matches.sort(key=lambda core: (core.shape[0] * core.shape[1], core.shape[0], core.shape[1]), reverse=True)
    return matches[0]


def _extract_aligned_cores(arr: np.ndarray) -> tuple[dict[str, int], list[np.ndarray]]:
    g = np.array(arr, dtype=int)
    bg = int(get_bg(g))
    desired = {side: color for side, color in _detect_controls_frame(g, bg)}
    cores: list[np.ndarray] = []
    for obj in _candidate_objects(g, bg):
        aligned = _align_object(obj, desired)
        if aligned is not None:
            cores.append(aligned)
    return desired, cores


def _priority_key(core: np.ndarray) -> tuple[int, int, int]:
    return (int(core.shape[0] * core.shape[1]), int(core.shape[0]), int(core.shape[1]))


def _anchor_location(desired: dict[str, int], out_h: int, out_w: int, core_h: int, core_w: int) -> tuple[int, int]:
    r0 = 0 if "top" in desired else int(out_h - core_h)
    c0 = 0 if "left" in desired else int(out_w - core_w)
    return int(r0), int(c0)


def _compose_output(cores: list[np.ndarray], desired: dict[str, int]) -> np.ndarray:
    if not cores:
        return np.zeros((0, 0), dtype=int)
    out_h = max(int(core.shape[0]) for core in cores)
    out_w = max(int(core.shape[1]) for core in cores)
    out = np.zeros((out_h, out_w), dtype=int)
    for core in sorted(cores, key=_priority_key, reverse=True):
        r0, c0 = _anchor_location(desired, out_h, out_w, int(core.shape[0]), int(core.shape[1]))
        paste(out, core, r0, c0)
    return out


def _largest_candidate_fallback(arr: np.ndarray) -> np.ndarray:
    g = np.array(arr, dtype=int)
    bg = int(get_bg(g))
    candidates = _candidate_objects(g, bg)
    if candidates:
        return candidates[0]
    return g.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Many:1 variant of fc7cae8d.

    Multiple interior rectangles carry the same two guide colors, so they all target the same
    frame-controlled corner. Each valid object is aligned independently, its guides are cropped,
    and the resulting cores are anchored to that common corner on one shared canvas. Overlaps are
    resolved by priority: larger-area cores are pasted first, smaller-area cores overwrite them.
    """
    desired, cores = _extract_aligned_cores(np.array(grid, dtype=int))
    if len(desired) != 2 or not cores:
        return _largest_candidate_fallback(np.array(grid, dtype=int))
    return _compose_output(cores, desired)


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    cols = [int(pattern_colors[i % len(pattern_colors)]) for i in range(3)]

    core[1 : min(h - 1, 3), 1 : min(w - 1, 4)] = cols[0]
    core[1 : min(h - 1, 4), max(1, w - 2)] = cols[1]
    core[max(1, h - 2), max(1, w - 4) : max(1, w - 1)] = cols[2]

    mid_r = max(1, min(h - 2, h // 2))
    mid_c = max(1, min(w - 2, w // 2))
    core[mid_r : min(h - 1, mid_r + 2), 1 : min(w - 1, 3)] = cols[1]
    core[max(1, h - 4) : max(1, h - 1), max(1, w // 3)] = cols[0]
    core[mid_r, mid_c : min(w - 1, mid_c + 2)] = cols[2]

    n_extra = random.randint(2, 5)
    for _ in range(n_extra):
        rr = random.randrange(1, h - 1)
        cc = random.randrange(1, w - 1)
        core[rr, cc] = random.choice(cols)
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


def _choose_sizes(out_h: int, out_w: int, n_obj: int) -> list[tuple[int, int]] | None:
    sizes = [(int(out_h), int(out_w))]
    seen_areas = {int(out_h * out_w)}
    seen_sizes = {(int(out_h), int(out_w))}

    while len(sizes) < int(n_obj):
        if len(seen_areas) > 80:
            return None
        h = random.randint(5, int(out_h))
        w = random.randint(5, int(out_w))
        area = int(h * w)
        if (h, w) in seen_sizes or area in seen_areas:
            continue
        sizes.append((int(h), int(w)))
        seen_sizes.add((int(h), int(w)))
        seen_areas.add(area)
    return sizes


def _count_variant_objects(grid: np.ndarray) -> int:
    _desired, cores = _extract_aligned_cores(np.array(grid, dtype=int))
    return int(len(cores))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(30, 36)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    usable = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]

    n_obj = random.choice([2, 3, 3])
    out_h = random.randint(8, 13)
    out_w = random.randint(8, 13)
    sizes = _choose_sizes(out_h, out_w, n_obj)
    if sizes is None or len(usable) < n_obj:
        return None

    fill_colors = random.sample(usable, n_obj)
    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}

    cores: list[np.ndarray] = []
    for idx, (h, w) in enumerate(sizes):
        fill = int(fill_colors[idx])
        pool = [c for c in usable if c != fill]
        random.shuffle(pool)
        pattern_colors = pool[: max(2, min(3, len(pool)))]
        if len(pattern_colors) < 2:
            return None
        cores.append(_build_core(int(h), int(w), fill, pattern_colors))

    expected = _compose_output(cores, guide_colors)
    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    objects: list[np.ndarray] = []
    for core in cores:
        aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)
        k = random.randrange(4)
        flip = bool(random.randrange(2))
        objects.append(_inverse_transform(aligned_object, k, flip))

    random.shuffle(objects)
    for obj in objects:
        try:
            r0, c0 = random_free_location(grid, obj, transparent=0, border=1)
        except ValueError:
            return None
        paste(grid, obj, r0, c0, transparent=0)

    got = main(grid)
    if not np.array_equal(got, expected):
        return None
    if _count_variant_objects(grid) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D7 many:1 variant")
