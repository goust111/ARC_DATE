# Combo variant of fc7cae8d (index 18)
# combo_spec: D2/3+D6/2
# Name: 多源+打包网格
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original task: one interior guided rectangle is aligned to the two active
#      frame sides, its 1px guide strips are cropped, and the remaining core is
#      returned directly.
#    - This combo changes two linked stages at once:
#      1. The input now contains 2-3 separate guided rectangles. The outer frame
#         has four solid control sides with distinct colors, so each object's
#         guide-color pair selects one unique corner independently.
#      2. After each object is aligned and cropped, the solver does not build a
#         corner collage. Instead it packs all cropped cores into one compact
#         row or one compact column with 1px background separators.
#    - Small example:
#         input objects A,B,C are placed roughly left-to-right inside the frame;
#         A targets top-left, B targets bottom-right, C targets top-right.
#         Original solver would only keep one largest object and ignore the rest.
#         Fused solver aligns A/B/C separately, removes their guides, then emits
#         core(A) | bg | core(B) | bg | core(C).
#      The corner mapping is still necessary for per-object alignment, but the
#      final output order comes from the input's multi-object spatial layout.
# b) Complexity calibration:
#    - Four frame-side colors are all distinct, so every horizontal+vertical
#      guide-color pair maps to exactly one corner and avoids ambiguous matches.
#    - Objects use asymmetric multicolor cores, so the needed dihedral alignment
#      is visually recoverable and does not collapse under symmetries.
#    - Generator places the objects in a clear horizontal lane or vertical lane.
#      This keeps the row-vs-column packing cue observable instead of noisy.
# c) Implicit-assumption audit against the original main():
#    - `detect_controls_frame(... )[:2]` is no longer valid because all four
#      border sides matter; the fused solver reads every solid border side.
#    - "largest interior component" is invalid because there are 2-3 objects; we
#      must collect every interior component and solve each independently.
#    - Early-return on the first matching transform is invalid globally; each
#      object needs its own transform search, crop, and retention.
#    - The old output assumption "single cropped rectangle" is broken. We now
#      need a packing stage that preserves multi-object order and inserts 1px
#      separators.
# d) Placement legality:
#    - Generator paints the full frame first, then places each transformed object
#      strictly inside rows/cols 2..n-3, leaving a bg moat so no object touches
#      the frame component.
#    - Objects are inserted along a lane with primary-axis gaps of 2-4 cells, so
#      they never overlap and remain separated components.
# e) Visual observability:
#    - Across examples, a human sees several guided rectangles inside one frame,
#      not one central object.
#    - Output examples visibly preserve all objects, but no longer in corner
#      quadrants; instead they become a single row or a single column collage.
# f) Visual cleanliness:
#    - No region mixes frame paint, guides, and other objects on top of each
#      other. Objects are isolated by bg gaps, and output separators are always
#      background-colored 1px bands.
# g) Color-change inferability:
#    - There is no hidden recoloring rule. Colors are stable: frame colors are
#      only used as guide/control keys, and cropped cores retain their own colors.
# h) New implicit assumptions introduced by this combo:
#    - To infer output orientation, the solver assumes object centroids span one
#      dominant axis (horizontal or vertical). The generator guarantees this by
#      placing objects in a row-lane or column-lane with only small perpendicular
#      jitter.
#    - The solver assumes at most one object claims each corner. The generator
#      enforces unique chosen corners.
#    - The solver assumes guide detection can distinguish guide strips from core
#      fill. The generator keeps fill dominant and leaves guide sides almost
#      entirely pure except for the single bg corner.

from __future__ import annotations

from collections import Counter
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
ARC_UTILS_DIR = os.path.join(ROOT, "arc2_solver_select")
if ARC_UTILS_DIR not in sys.path:
    sys.path.insert(0, ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste


CORNER_TO_SIDES: dict[str, tuple[str, str]] = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
SIDES_TO_CORNER = {sides: corner for corner, sides in CORNER_TO_SIDES.items()}


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _frame_controls(arr: np.ndarray, bg: int) -> dict[str, int]:
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    controls: dict[str, int] = {}
    for side, values in sides.items():
        vals = list(map(int, values))
        non_bg = [v for v in vals if v != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if len(non_bg) >= len(vals) - 1 and len(set(non_bg)) == 1:
            controls[side] = int(color)
            continue
        if int(count) >= len(vals) - 1:
            controls[side] = int(color)
    return controls


def _interior_objects(arr: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    objects: list[dict[str, object]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        patch = crop_bbox(arr, (r0, c0, r1, c1))
        objects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "patch": patch,
            }
        )
    objects.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    return objects


def _guide_edges(arr: np.ndarray, bg: int) -> dict[str, int]:
    non_bg_values = [int(v) for v in arr.flatten() if int(v) != int(bg)]
    if not non_bg_values:
        return {}
    fill = int(Counter(non_bg_values).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        vals = list(map(int, values))
        non_bg = [v for v in vals if v != int(bg)]
        if len(non_bg) < len(vals) - 1 or len(set(non_bg)) != 1:
            continue
        color = int(non_bg[0])
        if color != fill:
            out[side] = color
    return out


def _crop_guides(arr: np.ndarray, matched_sides: tuple[str, str]) -> np.ndarray:
    top = 1 if "top" in matched_sides else 0
    bottom = arr.shape[0] - (1 if "bottom" in matched_sides else 0)
    left = 1 if "left" in matched_sides else 0
    right = arr.shape[1] - (1 if "right" in matched_sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)].copy()


def _match_object(obj: np.ndarray, controls: dict[str, int], bg: int) -> tuple[str, np.ndarray] | None:
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed, bg)
            if len(guides) != 2:
                continue
            horiz = [side for side in guides if side in ("top", "bottom")]
            vert = [side for side in guides if side in ("left", "right")]
            if len(horiz) != 1 or len(vert) != 1:
                continue
            sides = (horiz[0], vert[0])
            if any(side not in controls for side in sides):
                continue
            if guides[sides[0]] != controls[sides[0]] or guides[sides[1]] != controls[sides[1]]:
                continue
            return SIDES_TO_CORNER[sides], _crop_guides(transformed, sides)
    return None


def _detect_pack_orientation(objects: list[dict[str, object]]) -> str:
    if len(objects) <= 1:
        return "row"
    centers_r = [0.5 * (int(obj["r0"]) + int(obj["r1"])) for obj in objects]
    centers_c = [0.5 * (int(obj["c0"]) + int(obj["c1"])) for obj in objects]
    row_span = max(centers_r) - min(centers_r)
    col_span = max(centers_c) - min(centers_c)
    return "row" if col_span >= row_span else "col"


def _pack_cores(cores: list[np.ndarray], orientation: str, bg: int) -> np.ndarray:
    if not cores:
        return np.full((1, 1), int(bg), dtype=int)
    if len(cores) == 1:
        return np.array(cores[0], copy=True)

    if orientation == "row":
        out_h = max(int(core.shape[0]) for core in cores)
        out_w = sum(int(core.shape[1]) for core in cores) + (len(cores) - 1)
        out = np.full((out_h, out_w), int(bg), dtype=int)
        c0 = 0
        for core in cores:
            paste(out, core, 0, c0)
            c0 += int(core.shape[1]) + 1
        return out

    out_h = sum(int(core.shape[0]) for core in cores) + (len(cores) - 1)
    out_w = max(int(core.shape[1]) for core in cores)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    r0 = 0
    for core in cores:
        paste(out, core, r0, 0)
        r0 += int(core.shape[0]) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read multiple guided rectangles, align each one to its matching frame corner,
    crop away the matched guides, then pack the recovered cores into a compact
    row or column according to the input objects' dominant layout axis.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    controls = _frame_controls(g, bg)
    objects = _interior_objects(g, bg)

    orientation = _detect_pack_orientation(objects)
    if orientation == "row":
        ordered_objects = sorted(objects, key=lambda item: (int(item["c0"]), int(item["r0"])))
    else:
        ordered_objects = sorted(objects, key=lambda item: (int(item["r0"]), int(item["c0"])))

    cores: list[np.ndarray] = []
    seen_corners: set[str] = set()
    for obj in ordered_objects:
        result = _match_object(np.array(obj["patch"], copy=False), controls, bg)
        if result is None:
            continue
        corner, core = result
        if corner in seen_corners:
            continue
        seen_corners.add(corner)
        cores.append(core)

    if not cores:
        return np.full((1, 1), int(bg), dtype=int)
    return _pack_cores(cores, orientation, bg)


def _dihedral_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _is_dihedrally_asymmetric(arr: np.ndarray) -> bool:
    seen = set()
    for k in range(4):
        base = np.rot90(arr, int(k))
        seen.add(_dihedral_signature(base))
        seen.add(_dihedral_signature(np.fliplr(base)))
    return len(seen) == 8


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    for _ in range(200):
        core = np.full((h, w), int(fill), dtype=int)
        anchors = random.sample([(r, c) for r in range(h) for c in range(w)], k=min(4, h * w))
        for idx, (r, c) in enumerate(anchors):
            core[r, c] = int(pattern_colors[idx % len(pattern_colors)])

        target_nonfill = random.randint(max(4, (h * w) // 6), max(6, int(h * w * 0.45)))
        painted = int(np.count_nonzero(core != int(fill)))

        while painted < target_nonfill:
            color = int(random.choice(pattern_colors))
            shape_kind = random.choice(["dot", "line", "rect", "lshape"])
            if shape_kind == "dot":
                cells = [(random.randrange(h), random.randrange(w))]
            elif shape_kind == "line":
                if random.random() < 0.5:
                    length = random.randint(2, min(5, w))
                    rr = random.randrange(h)
                    c0 = random.randrange(w - length + 1)
                    cells = [(rr, c0 + dc) for dc in range(length)]
                else:
                    length = random.randint(2, min(5, h))
                    r0 = random.randrange(h - length + 1)
                    cc = random.randrange(w)
                    cells = [(r0 + dr, cc) for dr in range(length)]
            elif shape_kind == "rect":
                hh = random.randint(1, min(3, h))
                ww = random.randint(1, min(3, w))
                r0 = random.randrange(h - hh + 1)
                c0 = random.randrange(w - ww + 1)
                cells = [(r0 + dr, c0 + dc) for dr in range(hh) for dc in range(ww)]
            else:
                r0 = random.randrange(max(1, h - 2))
                c0 = random.randrange(max(1, w - 2))
                arm_h = random.randint(2, min(4, h - r0))
                arm_w = random.randint(2, min(4, w - c0))
                cells = [(r0 + dr, c0) for dr in range(arm_h)] + [(r0, c0 + dc) for dc in range(arm_w)]

            for r, c in cells:
                if core[r, c] == int(fill):
                    painted += 1
                core[r, c] = color
                if painted >= target_nonfill:
                    break

        if _is_dihedrally_asymmetric(core):
            return core

    raise RuntimeError("failed to build asymmetric core")


def _attach_guides(core: np.ndarray, corner: str, guide_colors: dict[str, int], bg: int) -> np.ndarray:
    guide_sides = CORNER_TO_SIDES[corner]
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
        out[0, c0 : c0 + w] = int(guide_colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_colors["right"])

    if has_top and has_left:
        out[0, 0] = int(bg)
    if has_top and has_right:
        out[0, -1] = int(bg)
    if has_bottom and has_left:
        out[-1, 0] = int(bg)
    if has_bottom and has_right:
        out[-1, -1] = int(bg)
    return out


def _paint_frame(grid: np.ndarray, controls: dict[str, int], bg: int) -> None:
    grid[:, :] = int(bg)
    grid[0, 1:-1] = int(controls["top"])
    grid[-1, 1:-1] = int(controls["bottom"])
    grid[1:-1, 0] = int(controls["left"])
    grid[1:-1, -1] = int(controls["right"])
    grid[0, 0] = int(bg)
    grid[0, -1] = int(bg)
    grid[-1, 0] = int(bg)
    grid[-1, -1] = int(bg)


def _place_objects(
    grid: np.ndarray,
    objects: list[dict[str, object]],
    orientation: str,
    bg: int,
) -> list[dict[str, object]] | None:
    n = int(grid.shape[0])
    placed: list[dict[str, object]] = []
    gap = random.randint(2, 4)

    if orientation == "row":
        widths = [int(obj["patch"].shape[1]) for obj in objects]
        heights = [int(obj["patch"].shape[0]) for obj in objects]
        total_w = sum(widths) + gap * (len(objects) - 1)
        max_h = max(heights)
        if total_w > n - 4 or max_h + 2 > n - 4:
            return None
        max_start_c = n - 2 - total_w
        max_base_r = n - max_h - 4
        if max_start_c < 2 or max_base_r < 2:
            return None
        c0 = random.randint(2, max_start_c)
        base_r = random.randint(2, max_base_r)
        for obj in objects:
            patch = np.array(obj["patch"], copy=False)
            r0 = base_r + random.randint(0, 2)
            paste(grid, patch, r0, c0, transparent=bg)
            placed.append({**obj, "r0": int(r0), "c0": int(c0), "r1": int(r0 + patch.shape[0] - 1), "c1": int(c0 + patch.shape[1] - 1)})
            c0 += int(patch.shape[1]) + gap
        return placed

    widths = [int(obj["patch"].shape[1]) for obj in objects]
    heights = [int(obj["patch"].shape[0]) for obj in objects]
    total_h = sum(heights) + gap * (len(objects) - 1)
    max_w = max(widths)
    if total_h > n - 4 or max_w + 2 > n - 4:
        return None
    max_start_r = n - 2 - total_h
    max_base_c = n - max_w - 4
    if max_start_r < 2 or max_base_c < 2:
        return None
    r0 = random.randint(2, max_start_r)
    base_c = random.randint(2, max_base_c)
    for obj in objects:
        patch = np.array(obj["patch"], copy=False)
        c0 = base_c + random.randint(0, 2)
        paste(grid, patch, r0, c0, transparent=bg)
        placed.append({**obj, "r0": int(r0), "c0": int(c0), "r1": int(r0 + patch.shape[0] - 1), "c1": int(c0 + patch.shape[1] - 1)})
        r0 += int(patch.shape[0]) + gap
    return placed


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n = random.randint(32, 40)
    grid = np.full((n, n), int(bg), dtype=int)

    all_colors = list(range(1, 10))
    side_colors = random.sample(all_colors, 4)
    controls = {
        "top": side_colors[0],
        "bottom": side_colors[1],
        "left": side_colors[2],
        "right": side_colors[3],
    }
    _paint_frame(grid, controls, bg)

    orientation = random.choice(["row", "col"])
    n_objects = random.randint(2, 3)
    chosen_corners = random.sample(list(CORNER_TO_SIDES), n_objects)
    fills = random.sample([c for c in all_colors if c not in side_colors], n_objects)

    objects: list[dict[str, object]] = []
    for corner, fill in zip(chosen_corners, fills):
        available_patterns = [c for c in all_colors if c not in set(side_colors) | {fill}]
        pattern_count = random.randint(2, min(3, len(available_patterns)))
        pattern_colors = random.sample(available_patterns, pattern_count)
        core_h = random.randint(4, 8)
        core_w = random.randint(4, 8)
        core = _build_core(core_h, core_w, int(fill), pattern_colors)

        guide_colors = {side: controls[side] for side in CORNER_TO_SIDES[corner]}
        aligned = _attach_guides(core, corner, guide_colors, bg)
        k = random.randrange(4)
        flip = bool(random.randrange(2))
        patch = _inverse_transform(aligned, k, flip)
        objects.append({"corner": corner, "core": core, "patch": patch})

    random.shuffle(objects)
    placed = _place_objects(grid, objects, orientation, bg)
    if placed is None:
        return None

    ordered_cores = [np.array(obj["core"], copy=True) for obj in placed]
    expected = _pack_cores(ordered_cores, orientation, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 18")
