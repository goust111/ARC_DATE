# Combo variant of fc7cae8d (index 3)
# combo_spec: D1/4+D2/3
# Name: 频率消歧+多源对象
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Original task: one guide-rectangle sits inside the frame. Two frame sides advertise two
#      target colors, and a single rotation/flip aligns the object's guide-strip colors to those
#      sides before the guides are cropped away.
#    - Fused variant: the frame is full 4-sided and the input contains 2-3 separate guide-
#      rectangles. Each object still maps to one output corner as in the multi-source D2 variant.
#      In addition, exactly one chosen corner has the same control color on both of its adjacent
#      frame sides. The object for that corner therefore has two color-valid transforms that land
#      on the same corner. The solver must crop both candidate cores, compute the dominant color
#      of that core, and keep the transform where the dominant color is more frequent on the
#      matched horizontal edge than on the matched vertical edge.
#    - Small example: if `top == left == 2`, `bottom == 6`, `right == 8`, then the TL object
#      carries guide colors `(2, 2)`. Both the identity and a transpose-like transform still
#      satisfy TL by color. The correct one is the candidate whose dominant core color is heavier
#      on the top edge than on the left edge.
# b) Complexity calibration:
#    - Only one corner is ambiguous, so the puzzle clearly differs from the original while not
#      turning every object into a many-way search.
#    - The generator always includes the ambiguous object and adds 1-2 normal objects, so humans
#      can see both the multi-object composition rule and the extra frequency-based disambiguation.
#    - Core palettes stay small: one dominant color, one base color, and 2-3 accent colors. This
#      keeps the dominant-color cue visible instead of visually noisy.
# c) Implicit-assumption audit against the original `main()`:
#    - The original code assumed only one interior object and returned the first matching
#      transform. Both assumptions break here.
#    - Guide detection can no longer rely on the fill being the most common color, because the
#      dominant tiebreak color is intentionally more frequent than the base fill in some objects.
#      The new guide detector instead looks for near-solid outer strips whose color matches a
#      frame-control color.
#    - A color-only "first match wins" strategy is invalid because the repeated-color corner
#      deliberately creates two color-valid transforms for one object.
# d) Placement legality:
#    - Objects are placed with `random_free_location(..., border=2, padding=1)`, so they never
#      overlap each other and never touch the frame.
#    - Output corners are unique; the generator samples distinct target corners and composes the
#      expected output directly from those cores.
# e) Visual observability:
#    - A human sees several interior rectangles that each shrink to a corner-organized collage.
#    - One corner's two frame sides visibly share a color, and the corresponding object has the
#      same-color guides on both adjacent sides. That makes the extra orientation ambiguity
#      visually evident.
#    - The chosen output for that object consistently prefers a dominant-color band along the
#      horizontal side rather than the vertical side, so the frequency tiebreak is learnable.
# f) Visual cleanliness:
#    - Frame signals are clean full border strips with bg corners.
#    - Objects are rectangular, padded apart, and contain no bg holes, so components remain easy
#      to segment.
# g) Color-change inferability:
#    - There is no hidden numeric color remapping. Colors only do two visible jobs: frame/object
#      guide matching and dominant-color comparison inside the ambiguous object's core.
#
# New solver assumptions introduced here, and generator guarantees:
# - Exactly one corner has equal horizontal/vertical control colors; generator enforces that.
# - Every object has exactly one horizontal guide and one vertical guide; generator constructs
#   every object in that format.
# - The ambiguous object has at least two color-valid transforms, and the dominant color count on
#   its matched horizontal edge is strictly larger than on the matched vertical edge; generator
#   rejects samples that violate either condition.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


CORNER_TO_SIDES: dict[str, tuple[str, str]] = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
SIDES_TO_CORNER = {sides: corner for corner, sides in CORNER_TO_SIDES.items()}
HORIZONTAL_SIDES = {"top", "bottom"}
VERTICAL_SIDES = {"left", "right"}


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
    patches: list[tuple[tuple[int, int], np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        patches.append(((r0, c0), crop_bbox(arr, (r0, c0, r1, c1))))
    patches.sort(key=lambda item: item[0])
    return [patch for _pos, patch in patches]


def _guide_edges(arr: np.ndarray, control_colors: set[int], bg: int) -> dict[str, int]:
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
        if len(non_bg) < len(vals) - 1:
            continue
        if len(set(non_bg)) != 1:
            continue
        color = int(non_bg[0])
        if color not in control_colors:
            continue
        out[side] = color
    return out


def _crop_guides(arr: np.ndarray, matched_sides: tuple[str, str]) -> np.ndarray:
    top = 1 if "top" in matched_sides else 0
    bottom = arr.shape[0] - (1 if "bottom" in matched_sides else 0)
    left = 1 if "left" in matched_sides else 0
    right = arr.shape[1] - (1 if "right" in matched_sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)].copy()


def _dominant_color(core: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in core.flatten() if int(v) != int(bg)]
    if not vals:
        return int(bg)
    return int(Counter(vals).most_common(1)[0][0])


def _edge_count(core: np.ndarray, color: int, side: str) -> int:
    if side == "top":
        return int(np.count_nonzero(core[0, :] == int(color)))
    if side == "bottom":
        return int(np.count_nonzero(core[-1, :] == int(color)))
    if side == "left":
        return int(np.count_nonzero(core[:, 0] == int(color)))
    return int(np.count_nonzero(core[:, -1] == int(color)))


def _tiebreak_score(core: np.ndarray, sides: tuple[str, str], bg: int) -> tuple[int, ...]:
    dominant = _dominant_color(core, bg)
    horiz_side, vert_side = sides
    horiz_count = _edge_count(core, dominant, horiz_side)
    vert_count = _edge_count(core, dominant, vert_side)
    return (
        int(horiz_count - vert_count),
        int(horiz_count),
        int(-vert_count),
        int(np.count_nonzero(core == int(dominant))),
        int(-core.shape[0] * core.shape[1]),
    )


def _match_object(obj: np.ndarray, controls: dict[str, int], bg: int) -> tuple[str, np.ndarray] | None:
    control_colors = set(map(int, controls.values()))
    candidates: list[tuple[tuple[int, ...], str, np.ndarray]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed, control_colors, bg)
            if len(guides) != 2:
                continue
            horiz = [side for side in guides if side in HORIZONTAL_SIDES]
            vert = [side for side in guides if side in VERTICAL_SIDES]
            if len(horiz) != 1 or len(vert) != 1:
                continue
            sides = (horiz[0], vert[0])
            if guides[sides[0]] != controls.get(sides[0]) or guides[sides[1]] != controls.get(sides[1]):
                continue
            core = _crop_guides(transformed, sides)
            candidates.append((_tiebreak_score(core, sides, bg), SIDES_TO_CORNER[sides], core))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, corner, core = candidates[0]
    return corner, core


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
    Read all frame-side colors, solve each interior object independently, crop its guides, and
    compose the resulting cores into a 2x2 corner layout. If an object has several color-valid
    transforms, use the dominant-color horizontal-vs-vertical edge count to pick one.
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


def _dominant_margin(core: np.ndarray, corner: str, bg: int) -> int:
    dominant = _dominant_color(core, bg)
    horiz_side, vert_side = CORNER_TO_SIDES[corner]
    return _edge_count(core, dominant, horiz_side) - _edge_count(core, dominant, vert_side)


def _count_color_matches(obj: np.ndarray, controls: dict[str, int], bg: int) -> int:
    control_colors = set(map(int, controls.values()))
    count = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed, control_colors, bg)
            if len(guides) != 2:
                continue
            horiz = [side for side in guides if side in HORIZONTAL_SIDES]
            vert = [side for side in guides if side in VERTICAL_SIDES]
            if len(horiz) != 1 or len(vert) != 1:
                continue
            sides = (horiz[0], vert[0])
            if guides[sides[0]] == controls.get(sides[0]) and guides[sides[1]] == controls.get(sides[1]):
                count += 1
    return int(count)


def _naive_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    controls = _frame_controls(g, bg)
    objects = _interior_objects(g, bg)
    control_colors = set(map(int, controls.values()))

    matched: dict[str, np.ndarray] = {}
    for obj in objects:
        placed = False
        for k in range(4):
            base = np.rot90(obj, int(k))
            for flip in (False, True):
                transformed = np.fliplr(base) if flip else base
                guides = _guide_edges(transformed, control_colors, bg)
                if len(guides) != 2:
                    continue
                horiz = [side for side in guides if side in HORIZONTAL_SIDES]
                vert = [side for side in guides if side in VERTICAL_SIDES]
                if len(horiz) != 1 or len(vert) != 1:
                    continue
                sides = (horiz[0], vert[0])
                if guides[sides[0]] != controls.get(sides[0]) or guides[sides[1]] != controls.get(sides[1]):
                    continue
                corner = SIDES_TO_CORNER[sides]
                if corner not in matched:
                    matched[corner] = _crop_guides(transformed, sides)
                placed = True
                break
            if placed:
                break

    if not matched:
        return np.full((1, 1), int(bg), dtype=int)
    return _compose_output(matched, bg)


def _build_core(h: int, w: int, base: int, dominant: int, accents: list[int], corner: str, bg: int) -> np.ndarray:
    target_top = corner in ("tl", "tr")
    target_left = corner in ("tl", "bl")
    horiz_side, vert_side = CORNER_TO_SIDES[corner]
    band_h = max(2, (h + 1) // 2)
    vert_col = 0 if target_left else w - 1

    for _ in range(200):
        core = np.full((h, w), int(base), dtype=int)

        if target_top:
            core[:band_h, :] = int(dominant)
            ledge_row = min(h - 1, band_h)
        else:
            core[h - band_h :, :] = int(dominant)
            ledge_row = max(0, h - band_h - 1)

        # Thin out the matched vertical edge so the dominant color prefers the horizontal side.
        core[:, int(vert_col)] = int(base)
        ledge_len = max(2, w - 1)
        if target_left:
            core[ledge_row, :ledge_len] = int(dominant)
        else:
            core[ledge_row, w - ledge_len :] = int(dominant)

        gap_cols = random.sample([c for c in range(w) if c != int(vert_col)], k=min(max(1, w // 4), max(1, w - 1)))
        gap_row = 0 if target_top else h - 1
        for c in gap_cols:
            core[gap_row, int(c)] = int(base)

        if len(accents) >= 1:
            accent_a = int(accents[0])
            if target_top:
                core[h - 1, 1 % w] = accent_a
                core[max(band_h, h // 2), w // 2] = accent_a
            else:
                core[0, max(0, w - 2)] = accent_a
                core[max(0, min(h - 1, h // 2)), w // 2] = accent_a

        if len(accents) >= 2:
            accent_b = int(accents[1])
            far_col = w - 1 if target_left else 0
            core[h // 2, int(far_col)] = accent_b
            core[max(0, min(h - 1, h - 2 if target_top else 1)), int(far_col)] = accent_b

        if len(accents) >= 3:
            core[h // 2, w // 2] = int(accents[2])

        counts = Counter(map(int, core.flatten()))
        if counts[int(dominant)] <= max(count for color, count in counts.items() if int(color) != int(dominant)):
            continue
        if _dominant_color(core, bg) != int(dominant):
            continue
        if _edge_count(core, dominant, horiz_side) <= _edge_count(core, dominant, vert_side):
            continue
        if _dominant_margin(core, corner, bg) < 2:
            continue
        return core

    raise RuntimeError("failed to build asymmetric dominant core")


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


def _build_controls(ambiguous_corner: str, colors: tuple[int, int, int]) -> dict[str, int]:
    shared, other_a, other_b = map(int, colors)
    if ambiguous_corner == "tl":
        return {"top": shared, "left": shared, "bottom": other_a, "right": other_b}
    if ambiguous_corner == "tr":
        return {"top": shared, "right": shared, "left": other_a, "bottom": other_b}
    if ambiguous_corner == "bl":
        return {"bottom": shared, "left": shared, "top": other_a, "right": other_b}
    return {"bottom": shared, "right": shared, "top": other_a, "left": other_b}


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n = random.randint(28, 36)
    grid = np.full((n, n), int(bg), dtype=int)

    all_colors = list(range(1, 10))
    ambiguous_corner = random.choice(list(CORNER_TO_SIDES))
    controls = _build_controls(ambiguous_corner, tuple(random.sample(all_colors, 3)))
    _paint_frame(grid, controls, bg)

    n_objects = random.randint(2, 3)
    chosen_corners = [ambiguous_corner] + random.sample(
        [corner for corner in CORNER_TO_SIDES if corner != ambiguous_corner], n_objects - 1
    )

    available_core_colors = [c for c in all_colors if c not in set(map(int, controls.values()))]
    if len(available_core_colors) < 4:
        return None

    cores: dict[str, np.ndarray] = {}
    ambiguous_input_obj: np.ndarray | None = None

    for corner in chosen_corners:
        core_h = random.randint(5, 8)
        core_w = random.randint(5, 8)
        base, dominant = random.sample(available_core_colors, 2)
        accent_pool = [c for c in available_core_colors if c not in {base, dominant}]
        accent_count = random.randint(2, min(3, len(accent_pool)))
        accents = random.sample(accent_pool, accent_count)
        try:
            core = _build_core(core_h, core_w, int(base), int(dominant), list(map(int, accents)), corner, bg)
        except RuntimeError:
            return None
        cores[corner] = core

        guide_colors = {side: controls[side] for side in CORNER_TO_SIDES[corner]}
        aligned = _attach_guides(core, corner, guide_colors, bg)

        transforms = [(k, flip) for k in range(4) for flip in (False, True)]
        random.shuffle(transforms)
        input_object = None
        for k, flip in transforms:
            candidate = _inverse_transform(aligned, k, flip)
            if corner != ambiguous_corner:
                input_object = candidate
                break
            if _count_color_matches(candidate, controls, bg) >= 2:
                input_object = candidate
                break
        if input_object is None:
            return None

        if corner == ambiguous_corner:
            ambiguous_input_obj = input_object

        try:
            r0, c0 = random_free_location(grid, input_object, transparent=bg, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, input_object, r0, c0, transparent=bg)

    if ambiguous_input_obj is None or _count_color_matches(ambiguous_input_obj, controls, bg) < 2:
        return None

    expected = _compose_output(cores, bg)
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(_naive_main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant")
