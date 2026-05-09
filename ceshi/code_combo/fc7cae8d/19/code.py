# Combo variant of fc7cae8d (index 19)
# combo_spec: D2/3+D10/0
# Name: 多源+干扰矩形
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original puzzle: one interior guided rectangle, two frame controls, return one cropped core.
#    - Fused puzzle: the border now uses four solid control colors (top/bottom/left/right, corners left as bg).
#      Inside the frame there are 2-3 true source rectangles and 1-2 distractor rectangles.
#    - Each true source is still "core rectangle + two adjacent 1px guide strips + bg guide corner", but its
#      guide-strip colors match one horizontal frame side and one vertical frame side, so each source maps to
#      exactly one frame corner. The solver must independently test dihedral transforms for every interior object,
#      keep only objects whose two guides match a frame corner, remove those guides, and then compose the cropped
#      cores into a 2x2 corner collage.
#    - Distractors are plain filled rectangles with internal patterns but no guide strips. They stay in the input
#      only; they do not appear in the output. This means the old "pick the largest interior object" shortcut fails
#      twice: there are multiple true sources, and some non-source rectangles can be comparable or larger.
# b) Complexity calibration:
#    - Frame-side colors are all distinct, which keeps the corner mapping unambiguous.
#    - The number of true sources is limited to 2-3 and distractors to 1-2, so the scene is visibly richer than the
#      original without becoming unreadable.
#    - Source cores are forced to be dihedrally asymmetric, making the align-and-crop step visually meaningful.
#    - Distractors never use control colors and are rejected if they accidentally form apparent guide edges, which
#      keeps the "guided vs unguided" distinction stable across examples.
# c) Implicit-assumption audit against the old main():
#    - Old frame detection assumed only two useful sides; that is invalid once every side is a real control.
#    - Old object extraction assumed a single largest interior component; this breaks with multiple true sources and
#      distractors.
#    - Old early-return after the first matching transform is invalid; each interior component must be classified.
#    - Guide detection still works only if guide colors are exclusive to true strips, so the generator forbids control
#      colors inside source cores and distractors.
#    - The new solver additionally assumes that each target corner is claimed by at most one source; the generator
#      enforces unique chosen corners.
# d) Placement legality:
#    - Every interior object is placed with `random_free_location(..., border=2, padding=1)`, so objects do not
#      overlap, do not touch the frame, and remain separate 4-connected components.
# e) Visual observability:
#    - A human sees several interior rectangles, but only some have colored strips hugging two adjacent sides.
#      The outputs keep only those guided objects, arranged by corner, which makes both the multi-source rule and the
#      distractor-filtering rule directly visible from a few examples.
# f) Visual cleanliness:
#    - The frame is clean and fully colored, objects are separated by bg gaps, and distractors are plain rectangles
#      rather than extra noisy overlays. That keeps the scene legible.
# g) Color inferability:
#    - Colors are not remapped arbitrarily. The only meaningful color relation is "guide-strip color equals the
#      corresponding frame-side color", which is structurally anchored by side position and repeated across sources.
#
# New solver assumptions introduced here and guaranteed by the generator:
# - Exactly 2-3 interior components admit a valid guide-to-corner match.
# - Exactly 1-2 interior components do not admit such a match and are distractors.
# - True sources use unique corners, so the corner collage has no overwrite ambiguity.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
    matches: dict[str, np.ndarray] = {}
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
            matches[SIDES_TO_CORNER[sides]] = _crop_guides(transformed, sides)
    if len(matches) != 1:
        return None
    corner = next(iter(matches))
    return corner, matches[corner]


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


def _classify_components(grid: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    controls = _frame_controls(g, bg)
    matched: dict[str, np.ndarray] = {}
    distractors = 0
    for obj in _interior_objects(g, bg):
        result = _match_object(obj, controls, bg)
        if result is None:
            distractors += 1
            continue
        corner, core = result
        if corner in matched:
            return {}, -1
        matched[corner] = core
    return matched, distractors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Read all four frame-side control colors.
    - Scan every interior rectangle, separating guided sources from unguided distractors.
    - For each guided source, apply the dihedral transform that makes its two guide strips match
      one frame corner, then crop away the guide strips.
    - Assemble the recovered cores into a 2x2 corner layout; distractors disappear.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    matched, _distractors = _classify_components(g)
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


def _attach_guides(core: np.ndarray, corner: str, controls: dict[str, int], bg: int) -> np.ndarray:
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
        out[0, c0 : c0 + w] = int(controls["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(controls["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(controls["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(controls["right"])

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


def _build_distractor(
    h: int,
    w: int,
    fill: int,
    pattern_colors: list[int],
    controls: dict[str, int],
    bg: int,
) -> np.ndarray:
    control_colors = set(map(int, controls.values()))
    for _ in range(200):
        arr = _build_core(h, w, fill, pattern_colors)
        if any(int(v) in control_colors for v in arr.flatten()):
            continue
        looks_guided = False
        for k in range(4):
            base = np.rot90(arr, int(k))
            for flip in (False, True):
                transformed = np.fliplr(base) if flip else base
                if _guide_edges(transformed, bg):
                    looks_guided = True
                    break
            if looks_guided:
                break
        if not looks_guided:
            return arr
    raise RuntimeError("failed to build clean distractor")


def _choose_source_dims() -> tuple[int, int]:
    return random.randint(4, 8), random.randint(4, 8)


def _choose_distractor_dims(source_objects: list[np.ndarray], idx: int) -> tuple[int, int]:
    source_heights = [obj.shape[0] for obj in source_objects]
    source_widths = [obj.shape[1] for obj in source_objects]
    if idx == 0 and random.random() < 0.75:
        ref_h = max(source_heights)
        ref_w = max(source_widths)
        h = random.randint(max(4, ref_h - 1), max(5, ref_h + 2))
        w = random.randint(max(4, ref_w - 1), max(5, ref_w + 2))
        return min(h, 10), min(w, 10)
    return random.randint(4, 8), random.randint(4, 8)


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

    n_sources = random.randint(2, 3)
    chosen_corners = random.sample(list(CORNER_TO_SIDES), n_sources)
    source_fill_pool = [c for c in all_colors if c not in side_colors]
    source_fills = random.sample(source_fill_pool, n_sources)

    expected_cores: dict[str, np.ndarray] = {}
    source_objects: list[np.ndarray] = []
    for corner, fill in zip(chosen_corners, source_fills):
        pattern_pool = [c for c in all_colors if c not in set(side_colors) | {fill}]
        pattern_count = random.randint(2, min(3, len(pattern_pool)))
        pattern_colors = random.sample(pattern_pool, pattern_count)
        core_h, core_w = _choose_source_dims()
        core = _build_core(core_h, core_w, int(fill), pattern_colors)
        expected_cores[corner] = core

        aligned = _attach_guides(core, corner, controls, bg)
        k = random.randrange(4)
        flip = bool(random.randrange(2))
        input_object = _inverse_transform(aligned, k, flip)
        source_objects.append(input_object)

        try:
            r0, c0 = random_free_location(grid, input_object, transparent=bg, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, input_object, r0, c0, transparent=bg)

    n_distractors = random.randint(1, 2)
    non_control_colors = [c for c in all_colors if c not in side_colors]
    for idx in range(n_distractors):
        dh, dw = _choose_distractor_dims(source_objects, idx)
        d_fill = random.choice(non_control_colors)
        d_pattern_pool = [c for c in non_control_colors if c != d_fill]
        d_pattern_count = random.randint(2, min(3, len(d_pattern_pool)))
        distractor = _build_distractor(
            dh,
            dw,
            int(d_fill),
            random.sample(d_pattern_pool, d_pattern_count),
            controls,
            bg,
        )
        try:
            dr, dc = random_free_location(grid, distractor, transparent=bg, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, distractor, dr, dc, transparent=bg)

    matched, distractor_count = _classify_components(grid)
    if set(matched) != set(expected_cores):
        return None
    if distractor_count != n_distractors:
        return None

    expected = _compose_output(expected_cores, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 19")
