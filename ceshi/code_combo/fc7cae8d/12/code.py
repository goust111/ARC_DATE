# Combo variant of fc7cae8d (index 12)
# combo_spec: D2/3+D3/2
# Name: 多源+三控制边
#
# Design analysis for the fused D2/3 + D3/2 variant
# a) Concrete behavior and trace:
#    - Original task: one interior rectangle, two guide edges, two frame controls, output is the aligned core.
#    - Fused task: 2-3 interior rectangles, each rectangle has three guide edges, and the frame has three solid control
#      sides plus one sparse grey noise side. Every object is independently dihedrally aligned so that its three guide
#      colors land on the three control sides, then those three guide strips are cropped away.
#    - The D2 single-object assumption no longer works. To keep the multi-object output readable even though all objects
#      share the same three control colors, the output keeps one core per input quadrant: an input object from the
#      top-left quadrant becomes the top-left output tile, etc. Example: if the frame controls are top/left/right and
#      there are objects in input TL, TR, BL, then all three must be rotated/reflected to expose guides on top/left/right,
#      cropped, and placed into TL/TR/BL of the output collage.
# b) Complexity calibration:
#    - All three control colors are distinct, so orientation is constrained by a 3-color side assignment rather than a
#      looser 2-color corner pair.
#    - Objects live in distinct source quadrants and have asymmetric cores, avoiding ambiguous duplicate matches and
#      making the multi-source nature visually obvious.
#    - Only 2-3 objects are used, which is enough to force repeated matching without making the grid unreadable.
# c) Implicit-assumption audit against the original main():
#    - `detect_controls_frame(...)[ :2 ]` assumed exactly two meaningful sides. Here we must detect three solid sides.
#    - `extract_center_object()` assumed exactly one interior object and would silently drop the rest.
#    - `guide_edges()` assumed two guides and could accept a single dominant side accidentally. Here guide detection must
#      tolerate one or two bg corner gaps and require exactly three matched guide sides.
#    - The old early return after the first successful transform is invalid because every interior component needs an
#      independent match, crop, and placement.
# d) Placement legality:
#    - The generator places each object fully inside a distinct quadrant window, with padding from the frame and from
#      other objects, so components never overlap, never touch the border, and never cross the center split.
# e) Visual observability:
#    - Across examples a solver can see three persistent frame controls, every object carrying the same three guide
#      colors in scrambled orientations, and outputs keeping one cropped core per source quadrant. The multi-source and
#      three-control changes are both directly visible.
# f) Visual cleanliness:
#    - Quadrant-restricted placement prevents object tangles. Frame controls stay on the border only, and guide strips
#      stay attached to their own rectangles, so there is no layered clutter.
# g) Color-change inferability:
#    - There is no hidden recoloring rule. Guide colors are copied directly from frame-side colors, and core colors stay
#      unchanged; colors only serve as explicit side-matching cues.
# h) New solver assumptions introduced here:
#    - At most one relevant object occupies each source quadrant. The generator enforces distinct quadrant selection.
#    - Every valid object has exactly three guides matching the three detected controls. The generator builds objects that
#      satisfy this and uses asymmetric cores so the intended transform is observable.

from __future__ import annotations

from collections import Counter
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
ARC_UTILS_DIR = os.path.join(ROOT, "solvers")
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste


QUADRANTS = ("tl", "tr", "bl", "br")
CORNER_TO_SIDES: dict[str, tuple[str, str]] = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
ALL_SIDES = ("top", "bottom", "left", "right")


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
    controls: list[tuple[str, int, int]] = []
    for side, values in sides.items():
        vals = [int(v) for v in values]
        non_bg = [v for v in vals if v != int(bg)]
        if not non_bg or len(set(non_bg)) != 1:
            continue
        color = int(non_bg[0])
        if len(non_bg) >= len(vals) - 1:
            controls.append((side, color, len(non_bg)))
    controls.sort(key=lambda item: item[2], reverse=True)
    return {side: color for side, color, _count in controls[:3]}


def _source_quadrant(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    h, w = map(int, shape)
    mid_r = (h - 1) / 2.0
    mid_c = (w - 1) / 2.0
    center_r = (float(r0) + float(r1)) / 2.0
    center_c = (float(c0) + float(c1)) / 2.0
    vertical = "t" if center_r < mid_r else "b"
    horizontal = "l" if center_c < mid_c else "r"
    return vertical + horizontal


def _interior_objects(arr: np.ndarray, bg: int) -> list[tuple[str, np.ndarray]]:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    objects: list[tuple[tuple[int, int], str, np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        bbox = (r0, c0, r1, c1)
        patch = crop_bbox(arr, bbox)
        quadrant = _source_quadrant(bbox, arr.shape)
        objects.append(((r0, c0), quadrant, patch))
    objects.sort(key=lambda item: item[0])
    return [(quadrant, patch) for _pos, quadrant, patch in objects]


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
    guides: dict[str, int] = {}
    for side, values in edges.items():
        vals = [int(v) for v in values]
        non_bg = [v for v in vals if v != int(bg)]
        if len(non_bg) < len(vals) - 2 or len(set(non_bg)) != 1:
            continue
        color = int(non_bg[0])
        if color != fill:
            guides[side] = color
    return guides


def _crop_guides(arr: np.ndarray, guide_sides: set[str]) -> np.ndarray:
    top = 1 if "top" in guide_sides else 0
    bottom = arr.shape[0] - (1 if "bottom" in guide_sides else 0)
    left = 1 if "left" in guide_sides else 0
    right = arr.shape[1] - (1 if "right" in guide_sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)].copy()


def _match_object(obj: np.ndarray, controls: dict[str, int], bg: int) -> np.ndarray | None:
    expected_sides = set(controls)
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed, bg)
            if set(guides) != expected_sides:
                continue
            if any(int(guides[side]) != int(controls[side]) for side in expected_sides):
                continue
            return _crop_guides(transformed, expected_sides)
    return None


def _compose_output(quadrant_cores: dict[str, np.ndarray], bg: int) -> np.ndarray:
    top_h = max((quadrant_cores[name].shape[0] for name in ("tl", "tr") if name in quadrant_cores), default=0)
    bottom_h = max((quadrant_cores[name].shape[0] for name in ("bl", "br") if name in quadrant_cores), default=0)
    left_w = max((quadrant_cores[name].shape[1] for name in ("tl", "bl") if name in quadrant_cores), default=0)
    right_w = max((quadrant_cores[name].shape[1] for name in ("tr", "br") if name in quadrant_cores), default=0)

    row_gap = 1 if top_h > 0 and bottom_h > 0 else 0
    col_gap = 1 if left_w > 0 and right_w > 0 else 0
    out_h = top_h + row_gap + bottom_h
    out_w = left_w + col_gap + right_w
    out = np.full((out_h, out_w), int(bg), dtype=int)

    placements: dict[str, tuple[int, int]] = {}
    if "tl" in quadrant_cores:
        placements["tl"] = (0, 0)
    if "tr" in quadrant_cores:
        placements["tr"] = (0, left_w + col_gap + (right_w - quadrant_cores["tr"].shape[1]))
    if "bl" in quadrant_cores:
        placements["bl"] = (top_h + row_gap + (bottom_h - quadrant_cores["bl"].shape[0]), 0)
    if "br" in quadrant_cores:
        placements["br"] = (
            top_h + row_gap + (bottom_h - quadrant_cores["br"].shape[0]),
            left_w + col_gap + (right_w - quadrant_cores["br"].shape[1]),
        )

    for corner, (r0, c0) in placements.items():
        paste(out, quadrant_cores[corner], int(r0), int(c0))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Detect the three solid control sides of the frame.
    - Find every interior object and remember which input quadrant it came from.
    - For each object, find the dihedral transform whose three guide colors align to the three control sides.
    - Crop away those three guides and place the remaining core into the matching output quadrant.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    controls = _frame_controls(g, bg)
    objects = _interior_objects(g, bg)

    matched: dict[str, np.ndarray] = {}
    for quadrant, obj in objects:
        core = _match_object(obj, controls, bg)
        if core is None:
            continue
        if quadrant not in matched:
            matched[quadrant] = core

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


def _attach_guides(core: np.ndarray, control_sides: tuple[str, ...], controls: dict[str, int], bg: int) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in control_sides
    has_bottom = "bottom" in control_sides
    has_left = "left" in control_sides
    has_right = "right" in control_sides
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


def _paint_frame(grid: np.ndarray, controls: dict[str, int], noise_side: str, bg: int) -> None:
    n = grid.shape[0]
    grid[:, :] = int(bg)
    if "top" in controls:
        grid[0, 1:-1] = int(controls["top"])
    if "bottom" in controls:
        grid[-1, 1:-1] = int(controls["bottom"])
    if "left" in controls:
        grid[1:-1, 0] = int(controls["left"])
    if "right" in controls:
        grid[1:-1, -1] = int(controls["right"])
    grid[0, 0] = int(bg)
    grid[0, -1] = int(bg)
    grid[-1, 0] = int(bg)
    grid[-1, -1] = int(bg)

    if noise_side in ("top", "bottom"):
        r = 0 if noise_side == "top" else n - 1
        coords = [(r, c) for c in range(1, n - 1)]
    else:
        c = 0 if noise_side == "left" else n - 1
        coords = [(r, c) for r in range(1, n - 1)]
    noise_count = random.randint(int(np.ceil(0.35 * len(coords))), int(np.floor(0.60 * len(coords))))
    for r, c in random.sample(coords, noise_count):
        grid[r, c] = 5


def _quadrant_window(grid_shape: tuple[int, int], patch_shape: tuple[int, int], quadrant: str) -> tuple[int, int, int, int]:
    n, _m = map(int, grid_shape)
    h, w = map(int, patch_shape)
    mid_r = n // 2
    mid_c = n // 2
    if quadrant[0] == "t":
        r_min = 2
        r_max = mid_r - h - 2
    else:
        r_min = mid_r + 1
        r_max = n - h - 2
    if quadrant[1] == "l":
        c_min = 2
        c_max = mid_c - w - 2
    else:
        c_min = mid_c + 1
        c_max = n - w - 2
    return int(r_min), int(c_min), int(r_max), int(c_max)


def _random_free_in_window(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    window: tuple[int, int, int, int],
    *,
    padding: int = 1,
) -> tuple[int, int]:
    h, w = patch.shape
    r_min, c_min, r_max, c_max = window
    if r_max < r_min or c_max < c_min:
        raise ValueError("window too small")
    mask = patch != int(bg)
    candidates: list[tuple[int, int]] = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            target = canvas[r0 : r0 + h, c0 : c0 + w]
            if np.any(target[mask] != int(bg)):
                continue
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(canvas.shape[0], r0 + h + padding)
            cc1 = min(canvas.shape[1], c0 + w + padding)
            if np.any(canvas[rr0:rr1, cc0:cc1] != int(bg)):
                continue
            candidates.append((int(r0), int(c0)))
    if not candidates:
        raise ValueError("no placement in quadrant window")
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n = random.randint(30, 38)
    grid = np.full((n, n), int(bg), dtype=int)

    noise_side = random.choice(ALL_SIDES)
    control_sides = tuple(side for side in ALL_SIDES if side != noise_side)
    control_colors = random.sample([c for c in range(1, 10) if c != 5], 3)
    controls = {side: color for side, color in zip(control_sides, control_colors)}
    _paint_frame(grid, controls, noise_side, bg)

    n_objects = random.randint(2, 3)
    quadrants = random.sample(list(QUADRANTS), n_objects)
    available_fills = [c for c in range(1, 10) if c not in set(control_colors) | {5}]
    if len(available_fills) < n_objects:
        return None

    quadrant_cores: dict[str, np.ndarray] = {}
    transforms: list[tuple[int, bool]] = []
    used_fills: set[int] = set()
    for quadrant in quadrants:
        remaining_fills = [c for c in available_fills if c not in used_fills]
        if not remaining_fills:
            return None
        fill = int(random.choice(remaining_fills))
        used_fills.add(fill)
        pattern_pool = [c for c in range(1, 10) if c not in set(control_colors) | {5, fill}]
        pattern_count = random.randint(2, min(3, len(pattern_pool)))
        pattern_colors = random.sample(pattern_pool, pattern_count)
        core_h = random.randint(4, 8)
        core_w = random.randint(4, 8)
        core = _build_core(core_h, core_w, fill, pattern_colors)
        quadrant_cores[quadrant] = core

        aligned = _attach_guides(core, control_sides, controls, bg)
        k = random.randrange(4)
        flip = bool(random.randrange(2))
        transforms.append((k, flip))
        input_object = _inverse_transform(aligned, k, flip)

        try:
            window = _quadrant_window(grid.shape, input_object.shape, quadrant)
            r0, c0 = _random_free_in_window(grid, input_object, bg, window, padding=1)
        except ValueError:
            return None
        paste(grid, input_object, r0, c0, transparent=bg)

    if not any(k != 0 or flip for k, flip in transforms):
        return None

    expected = _compose_output(quadrant_cores, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D2/3+D3/2")
