# Combo variant of fc7cae8d (index 17)
# combo_spec: D2/3+D5/0
# Name: 多源+保留源
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace:
#    - Original task: one interior guided rectangle is dihedrally aligned to the two active frame
#      sides, the guide strips are cut away, and only that cropped core is returned.
#    - Fused task here: the frame has four solid colored sides, and there are 2-3 separate guided
#      rectangles inside. Each object uses one horizontal guide color and one vertical guide color,
#      so each object points to one unique frame corner.
#    - For each object independently:
#         input object -> try 8 dihedral transforms -> find transform whose two guide sides match
#         the corresponding frame colors -> crop those two guide strips -> paste the cropped core
#         into that corner's interior location.
#    - Unlike D2 alone, we do not build a new collage canvas; unlike D5 alone, we do not have just
#      one source object. The full input grid is preserved, every source object stays in place, and
#      every matched core is additionally copied into its own target corner.
#    - Small example trace:
#         old: one object tagged by top+left colors -> align -> crop -> return only cropped core.
#         fused: objects tagged by TL, TR, BL -> align each separately -> crop each separately ->
#         output is the original full scene plus three extra cropped copies at the TL/TR/BL interior
#         corners. The source objects do not move or disappear.
#
# b) Complexity calibration:
#    - Four frame-side colors are distinct so each (horizontal, vertical) guide-color pair maps to
#      exactly one corner; this prevents ambiguous object-to-corner assignment.
#    - There are only 2-3 objects, not 4+, so humans can still track them visually.
#    - Each core is forced to be dihedrally asymmetric and each source object is given a non-identity
#      input transform; otherwise the "copy after alignment" behavior would be visually weak.
#    - Objects use different fill colors and stay separated by at least one blank moat, so component
#      extraction and human grouping are both stable.
#
# c) Implicit-assumption audit against the original solver:
#    - Old `detect_controls_frame()` assumed only two meaningful frame sides. That no longer holds:
#      the fused rule needs all four sides so each object can target a different corner.
#    - Old `extract_center_object()` assumed one interior object and picked the largest. That would
#      silently drop the other sources, so we must enumerate all interior components.
#    - Old `return t[...]` on the first matching transform assumed "transform replaces output". The
#      fused rule instead preserves the whole grid and applies multiple copy operations.
#    - Old logic had no need to keep target corners empty. The fused generator now must reserve every
#      corner-copy region so pasted cores remain visible and do not overwrite sources.
#    - New combined assumption: at most one object claims each corner. The generator enforces unique
#      target corners; the solver keeps the first successful match per corner.
#
# d) Placement legality:
#    - The generator paints the full frame first, then reserves each corner interior copy region plus
#      a one-cell moat, then places source objects into remaining free space with `border=2` and
#      `padding=1`. This prevents overlap with the frame, with each other, and with future copies.
#    - Corner copy boxes are checked for overlap before accepting a sample, so copied cores remain
#      visually separate even when two objects target the same row or column band of the border.
#
# e) Visual observability:
#    - Across samples, humans can see several guided rectangles still present in the output and can
#      also see extra unguided cores appear in the corners whose border colors match the guide colors.
#      That exposes both dimensions: multi-source matching and keep-source copying.
#
# f) Visual cleanliness:
#    - The frame is a clean four-sided rectangle with background corners.
#    - Source objects are spaced apart and away from corners.
#    - Copied cores only appear in reserved corner interiors, so there is no confusing overdraw on
#      top of the source objects or border controls.
#
# g) Color inference:
#    - Colors are not remapped. Guide colors are direct references to the frame side colors, which is
#      structurally visible from position and repetition rather than requiring arbitrary color lookup.
#      Using exactly four frame colors keeps the cue system compact and learnable from few examples.
#
# Additional solver-assumption audit after fusion:
#    - `_frame_controls()` assumes each border side is nearly solid in one non-bg color.
#      `_paint_frame()` guarantees exactly that.
#    - `_interior_objects()` assumes sources do not touch the frame or each other in 4-neighborhood.
#      `random_free_location(..., border=2, padding=1)` guarantees that.
#    - `_match_object()` assumes each object has exactly one horizontal and one vertical guide side
#      after the correct transform, and that its guide colors uniquely identify a corner.
#      `_attach_guides()` plus distinct side colors and unique chosen corners guarantee that.
#    - `main()` assumes the pasted core fits in the target corner interior without colliding with an
#      existing source object. `_reserve_corner()` guarantees that during generation.

from __future__ import annotations

from collections import Counter
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
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


def _corner_core_location(grid_shape: tuple[int, int], core_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, core_shape)
    H, W = map(int, grid_shape)
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, W - w - 1
    if corner == "bl":
        return H - h - 1, 1
    return H - h - 1, W - w - 1


def _expected_output_from_spec(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    controls = _frame_controls(g, bg)
    if len(controls) < 4:
        return g.copy()

    out = g.copy()
    used_corners: set[str] = set()
    for obj in _interior_objects(g, bg):
        result = _match_object(obj, controls, bg)
        if result is None:
            continue
        corner, core = result
        if corner in used_corners:
            continue
        used_corners.add(corner)
        r0, c0 = _corner_core_location(out.shape, core.shape, corner)
        paste(out, core, int(r0), int(c0))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Read all four frame-side colors.
    - Detect every interior guided rectangle.
    - For each one, find the dihedral transform whose two guide strips match one frame corner.
    - Remove those guide strips to expose the core.
    - Keep the original grid unchanged in place, and additionally paste each cropped core into the
      corresponding corner interior.
    """
    return _expected_output_from_spec(np.array(grid, dtype=int))


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

        anchors = [
            (0, 0),
            (h - 1, min(w - 1, 1)),
            (min(h - 1, 1), w - 1),
            (max(0, h // 2), max(0, w // 2 - 1)),
        ]
        for idx, (r, c) in enumerate(anchors):
            core[int(r), int(c)] = int(pattern_colors[idx % len(pattern_colors)])

        target_nonfill = random.randint(max(4, (h * w) // 6), max(6, int(h * w * 0.45)))
        painted = int(np.count_nonzero(core != int(fill)))

        while painted < target_nonfill:
            color = int(random.choice(pattern_colors))
            shape_kind = random.choice(["dot", "line", "rect", "elbow"])
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
                cells = [(r0, c0), (r0 + 1, c0), (r0 + 1, c0 + 1), (r0 + 1, min(w - 1, c0 + 2))]

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


def _reserve_corner(canvas: np.ndarray, core_shape: tuple[int, int], corner: str, *, moat: int, marker: int) -> bool:
    h, w = map(int, core_shape)
    r0, c0 = _corner_core_location(canvas.shape, core_shape, corner)
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    rr0 = max(0, r0 - moat)
    cc0 = max(0, c0 - moat)
    rr1 = min(canvas.shape[0] - 1, r1 + moat)
    cc1 = min(canvas.shape[1] - 1, c1 + moat)
    if np.any(canvas[rr0 : rr1 + 1, cc0 : cc1 + 1] != 0):
        return False
    canvas[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(marker)
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n = random.randint(30, 36)
    grid = np.full((n, n), int(bg), dtype=int)

    all_colors = list(range(1, 10))
    side_colors = random.sample(all_colors, 4)
    controls = {
        "top": int(side_colors[0]),
        "bottom": int(side_colors[1]),
        "left": int(side_colors[2]),
        "right": int(side_colors[3]),
    }
    _paint_frame(grid, controls, bg)

    n_objects = random.randint(2, 3)
    chosen_corners = random.sample(list(CORNER_TO_SIDES), n_objects)
    available_fills = [c for c in all_colors if c not in side_colors]
    fills = random.sample(available_fills, n_objects)

    occupancy = grid.copy()
    reserved = np.zeros_like(grid)
    objects: list[np.ndarray] = []

    for corner, fill in zip(chosen_corners, fills):
        available_patterns = [c for c in all_colors if c not in set(side_colors) | {fill}]
        pattern_count = random.randint(2, min(3, len(available_patterns)))
        pattern_colors = random.sample(available_patterns, pattern_count)

        core_h = random.randint(4, 8)
        core_w_choices = [w for w in range(4, 9) if w != core_h]
        core_w = random.choice(core_w_choices) if core_w_choices else random.randint(4, 8)
        core = _build_core(core_h, core_w, int(fill), pattern_colors)

        if not _reserve_corner(reserved, core.shape, corner, moat=1, marker=9):
            return None
        occupancy[reserved != 0] = 9

        aligned = _attach_guides(
            core,
            corner,
            {side: controls[side] for side in CORNER_TO_SIDES[corner]},
            bg,
        )
        transforms = [(k, flip) for k in range(4) for flip in (False, True) if not (k == 0 and not flip)]
        k, flip = random.choice(transforms)
        input_object = _inverse_transform(aligned, k, flip)
        objects.append((corner, core, input_object))

    for _corner, _core, input_object in objects:
        try:
            r0, c0 = random_free_location(occupancy, input_object, transparent=bg, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, input_object, r0, c0, transparent=bg)
        paste(occupancy, input_object, r0, c0, transparent=bg)

    expected = _expected_output_from_spec(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    # Reject degenerate samples where no new corner copy is introduced beyond the source scene.
    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant D2/3+D5/0")
