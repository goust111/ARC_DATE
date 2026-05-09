# Variant of fc7cae8d
# Dimension: D2 — 源对象结构 (src)
# Change: 单个填充矩形，两相邻边有1px纯色引导条带，引导角（两引导边交汇处）为背景色... → 多个源对象
#
# Design analysis for the D2 variant
# a) Concrete behavior:
#    - Input now has a full four-sided colored frame: top/bottom/left/right are each solid, with corners left as bg.
#    - Inside the frame there are 2-3 separate multicolor rectangles. Each rectangle has exactly two adjacent 1px guide strips,
#      and the two guide-strip colors match one horizontal frame side and one vertical frame side. This uniquely selects one corner.
#    - Solver must no longer "take the largest interior object". It must detect every interior object, try dihedral transforms
#      independently, and map each object to a different target corner. After alignment, the guide strips are removed, leaving a core.
#    - Output is a 2x2 corner layout of the cropped cores: top-left core goes to the top-left quadrant, top-right to top-right, etc.
#      Missing corners remain bg. Example:
#         original: one object -> align -> crop -> return that core
#         variant: three objects (TL, TR, BL) -> align each one separately -> crop each -> place them into TL/TR/BL of one output canvas
#      The key difference is that matching is now per-object and corner-specific, not global and singular.
# b) Complexity calibration:
#    - Four frame-side colors are distinct, so each guide-color pair maps to exactly one corner; this avoids ambiguous multi-match failures.
#    - Objects are different in size/pattern and are separated by bg gaps; otherwise multiple components would be visually unclear.
#    - Core patterns are forced to be asymmetric across dihedral transforms, so each per-object rotation/reflection is observable.
# c) Implicit-assumption audit against the old main():
#    - Old `detect_controls_frame` assumed only two meaningful sides; this breaks once all four sides matter.
#    - Old `extract_center_object` assumed exactly one interior object and chose the largest bbox; that would silently drop the others.
#    - Old `guide_edges` was still mostly reusable, but only if it excludes bg/fill carefully; otherwise a plain fill side could be
#      mistaken for a guide. Here guide detection requires a nearly full side of one non-fill color.
#    - Old early return after the first matching transform is invalid because each component now needs its own transform and crop.
# d) Placement legality:
#    - Generator uses `random_free_location(..., padding=1)` so objects do not overlap, do not touch each other, and stay clear of frame edges.
#    - Each chosen corner is unique; generator never creates two objects claiming the same corner, which the solver also assumes.
# e) Visual observability:
#    - From a few examples, a human can see that (1) there are several guide-strip rectangles, (2) each strip-color pair points to one
#      frame corner, and (3) outputs preserve all of them in a corner-organized collage. This makes the D2 change visible rather than
#      degenerating back to "pick one central rectangle".

from __future__ import annotations

from collections import Counter
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
ARC_UTILS_DIR = os.path.join(ROOT, "arc2_solver_select")
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


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
        if len(set(non_bg)) == 1 and int(count) >= len(vals) - 1:
            controls[side] = int(color)
    return controls


def _interior_objects(arr: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    objects: list[tuple[tuple[int, int], np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        patch = crop_bbox(arr, (r0, c0, r1, c1))
        objects.append(((r0, c0), patch))
    objects.sort(key=lambda item: item[0])
    return [patch for _pos, patch in objects]


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
            corner = SIDES_TO_CORNER[sides]
            return corner, _crop_guides(transformed, sides)
    return None


def _compose_output(corners: dict[str, np.ndarray], bg: int) -> np.ndarray:
    top_h = max((corners[name].shape[0] for name in ("tl", "tr") if name in corners), default=0)
    bottom_h = max((corners[name].shape[0] for name in ("bl", "br") if name in corners), default=0)
    left_w = max((corners[name].shape[1] for name in ("tl", "bl") if name in corners), default=0)
    right_w = max((corners[name].shape[1] for name in ("tr", "br") if name in corners), default=0)

    row_gap = 1 if top_h > 0 and bottom_h > 0 else 0
    col_gap = 1 if left_w > 0 and right_w > 0 else 0
    out_h = top_h + row_gap + bottom_h
    out_w = left_w + col_gap + right_w
    out = np.full((out_h, out_w), int(bg), dtype=int)

    placements: dict[str, tuple[int, int]] = {}
    if "tl" in corners:
        placements["tl"] = (0, 0)
    if "tr" in corners:
        placements["tr"] = (0, left_w + col_gap + (right_w - corners["tr"].shape[1]))
    if "bl" in corners:
        placements["bl"] = (top_h + row_gap + (bottom_h - corners["bl"].shape[0]), 0)
    if "br" in corners:
        placements["br"] = (
            top_h + row_gap + (bottom_h - corners["br"].shape[0]),
            left_w + col_gap + (right_w - corners["br"].shape[1]),
        )

    for corner, (r0, c0) in placements.items():
        paste(out, corners[corner], int(r0), int(c0))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Read all four frame-side colors.
    - For each interior guide-rectangle, find the dihedral transform that makes its two guide strips match one frame corner.
    - Remove the matched guide strips.
    - Assemble the resulting cores into a 2x2 output layout by corner.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    controls = _frame_controls(g, bg)
    objects = _interior_objects(g, bg)

    matched: dict[str, np.ndarray] = {}
    for obj in objects:
        result = _match_object(obj, controls, bg)
        if result is None:
            continue
        corner, core = result
        if corner not in matched:
            matched[corner] = core

    if not matched:
        return np.full((1, 1), int(bg), dtype=int)
    return _compose_output(matched, bg)


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

        # Place a few fixed anchors first so the interior is visibly directional.
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


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n = random.randint(28, 36)
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

    n_objects = random.randint(2, 3)
    chosen_corners = random.sample(list(CORNER_TO_SIDES), n_objects)
    fills = random.sample([c for c in all_colors if c not in side_colors], n_objects)
    cores: dict[str, np.ndarray] = {}

    for corner, fill in zip(chosen_corners, fills):
        available_patterns = [c for c in all_colors if c not in set(side_colors) | {fill}]
        pattern_count = random.randint(2, min(3, len(available_patterns)))
        pattern_colors = random.sample(available_patterns, pattern_count)
        core_h = random.randint(4, 8)
        core_w = random.randint(4, 8)
        core = _build_core(core_h, core_w, int(fill), pattern_colors)
        cores[corner] = core

        guide_colors = {
            side: controls[side]
            for side in CORNER_TO_SIDES[corner]
        }
        aligned = _attach_guides(core, corner, guide_colors, bg)
        k = random.randrange(4)
        flip = bool(random.randrange(2))
        input_object = _inverse_transform(aligned, k, flip)

        try:
            r0, c0 = random_free_location(grid, input_object, transparent=bg, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, input_object, r0, c0, transparent=bg)

    expected = _compose_output(cores, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D2 variant")
