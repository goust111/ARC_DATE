# Combo variant of fc7cae8d (index 36)
# combo_spec: D2/3+D3/2+D6/2
# Name: 多源+三控制边+打包网格
#
# Design analysis for the combo variant
# a) Concrete behavior:
#    - Original task: one interior rectangle, two guide sides, two frame control sides, return one cropped core.
#    - Combo task: the frame now has exactly three solid control sides and one noisy side. There are 2-3 interior objects.
#      Each object carries exactly three guide strips whose colors match the three solid frame sides after the correct
#      dihedral transform. The guide-missing side must face the noisy frame side after alignment.
#    - Example trace:
#         old: detect 2 controls -> find 1 object -> try 8 transforms -> crop 2 matched guide borders -> return 1 core
#         new: detect 3 controls -> find every interior object -> for each, try 8 transforms until all 3 guide sides match
#              the 3 control sides -> crop the 3 matched guide borders -> pack all resulting cores into a 2-column grid
#              in source reading order using 1px background separators.
#      The key difference is not only "more objects", but also that orientation is constrained by three side/color matches
#      instead of two, and the output is a packed multi-object summary rather than a single crop.
# b) Complexity calibration:
#    - Control colors are distinct and never reused by fills/patterns, so three-side matching is unambiguous.
#    - Each core is forced to be dihedrally asymmetric and to contain boundary anchors on all four sides. This prevents
#      the guide-missing side from degenerating into a plain monochrome edge that could be mistaken for a guide.
#    - Only 2-3 objects and a fixed 2-column pack layout are used so the task is visibly different without becoming chaotic.
# c) Implicit-assumption audit against the old main():
#    - Old frame detection assumed only two meaningful sides; invalid once three controls exist.
#    - Old object extraction assumed the largest interior component is the unique target; invalid with 2-3 objects.
#    - Old guide detection tolerated any side that was "mostly non-fill"; here we must explicitly require near-full single-color
#      sides and then demand that the full 3-side guide dictionary equals the frame-control dictionary.
#    - Old early return after matching one object is invalid; every component must be transformed and cropped independently.
#    - New packing introduces an ordering assumption: objects are emitted in bbox reading order. The generator guarantees that
#      components remain visually separated so this order is stable and human-visible.
# d) Placement legality:
#    - Generator uses `random_free_location(..., border=2, padding=1)` so objects do not overlap each other or the frame.
#    - All objects are placed fully inside the frame, remain 4-connected internally, and stay separated by background gaps.
# e) Visual observability:
#    - Across a few examples, a solver can see three things directly: there are multiple source rectangles, each has three
#      colored guide sides, and outputs preserve all cropped interiors in a neat packed collage. This is visibly different
#      from the original single-object/two-edge task.
# f) Visual cleanliness:
#    - The noisy frame is restricted to one border only; the other three borders stay clean solid lines.
#    - Objects are spaced apart and the packed output uses only 1px background separators, so no region contains layered clutter.
# g) Color-change inferability:
#    - Colors are not remapped. Guide colors simply point to same-colored frame sides, and those control colors are reused across
#      all objects within one example. This gives repeated structural evidence instead of arbitrary color substitution.

from __future__ import annotations

from collections import Counter
import math
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
ARC_UTILS_DIR = os.path.join(ROOT, "arc2_solver_select")
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


SIDE_ORDER = ("top", "bottom", "left", "right")


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
    for side in SIDE_ORDER:
        values = list(map(int, sides[side]))
        non_bg = [v for v in values if v != int(bg)]
        if not non_bg:
            continue
        if len(non_bg) < len(values) - 1:
            continue
        if len(set(non_bg)) != 1:
            continue
        controls[side] = int(non_bg[0])
    return controls


def _interior_objects(arr: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    objects: list[tuple[tuple[int, int], np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        objects.append(((r0, c0), crop_bbox(arr, (r0, c0, r1, c1))))
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
    guides: dict[str, int] = {}
    for side in SIDE_ORDER:
        values = list(map(int, edges[side]))
        non_bg = [v for v in values if v != int(bg)]
        if len(non_bg) < len(values) - 2:
            continue
        if len(set(non_bg)) != 1:
            continue
        color = int(non_bg[0])
        if color != fill:
            guides[side] = color
    return guides


def _crop_guides(arr: np.ndarray, matched_sides: set[str]) -> np.ndarray:
    top = 1 if "top" in matched_sides else 0
    bottom = arr.shape[0] - (1 if "bottom" in matched_sides else 0)
    left = 1 if "left" in matched_sides else 0
    right = arr.shape[1] - (1 if "right" in matched_sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)].copy()


def _match_object(obj: np.ndarray, controls: dict[str, int], bg: int) -> np.ndarray | None:
    desired_sides = set(controls)
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed, bg)
            if set(guides) != desired_sides:
                continue
            if any(int(guides[side]) != int(controls[side]) for side in desired_sides):
                continue
            return _crop_guides(transformed, desired_sides)
    return None


def _pack_cores(cores: list[np.ndarray], bg: int, cols: int = 2) -> np.ndarray:
    if not cores:
        return np.full((1, 1), int(bg), dtype=int)

    cols = max(1, min(int(cols), len(cores)))
    rows = int(math.ceil(len(cores) / cols))
    row_heights = [0] * rows
    col_widths = [0] * cols

    for idx, core in enumerate(cores):
        row = idx // cols
        col = idx % cols
        row_heights[row] = max(row_heights[row], int(core.shape[0]))
        col_widths[col] = max(col_widths[col], int(core.shape[1]))

    out_h = sum(row_heights) + max(0, rows - 1)
    out_w = sum(col_widths) + max(0, cols - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    y = 0
    for row in range(rows):
        x = 0
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(cores):
                break
            paste(out, cores[idx], int(y), int(x))
            x += col_widths[col] + 1
        y += row_heights[row] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect the 3 solid frame-control sides.
    - For every interior component, find the dihedral transform whose 3 guide sides exactly match those controls.
    - Remove the guide strips.
    - Pack the cropped cores in bbox reading order into a 2-column grid with 1px separators.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    controls = _frame_controls(g, bg)
    objects = _interior_objects(g, bg)

    cores: list[np.ndarray] = []
    for obj in objects:
        core = _match_object(obj, controls, bg)
        if core is not None:
            cores.append(core)

    return _pack_cores(cores, bg, cols=2)


def _dihedral_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _is_dihedrally_asymmetric(arr: np.ndarray) -> bool:
    seen = set()
    for k in range(4):
        base = np.rot90(arr, int(k))
        seen.add(_dihedral_signature(base))
        seen.add(_dihedral_signature(np.fliplr(base)))
    return len(seen) == 8


def _side_anchor(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randrange(0, w)
    if side == "bottom":
        return h - 1, random.randrange(0, w)
    if side == "left":
        return random.randrange(0, h), 0
    return random.randrange(0, h), w - 1


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    for _ in range(250):
        core = np.full((h, w), int(fill), dtype=int)

        for idx, side in enumerate(SIDE_ORDER):
            r, c = _side_anchor(side, h, w)
            core[r, c] = int(pattern_colors[idx % len(pattern_colors)])

        anchors = random.sample([(r, c) for r in range(h) for c in range(w)], k=min(5, h * w))
        for idx, (r, c) in enumerate(anchors):
            core[r, c] = int(pattern_colors[(idx + 1) % len(pattern_colors)])

        target_nonfill = random.randint(max(6, (h * w) // 5), max(8, int(h * w * 0.45)))
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


def _attach_guides(core: np.ndarray, guide_sides: set[str], guide_colors: dict[str, int], bg: int) -> np.ndarray:
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


def _paint_frame(grid: np.ndarray, controls: dict[str, int], noise_side: str, bg: int, noise_color: int) -> None:
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

    if noise_side in ("top", "bottom"):
        r = 0 if noise_side == "top" else n - 1
        coords = [(r, c) for c in range(1, n - 1)]
    else:
        c = 0 if noise_side == "left" else n - 1
        coords = [(r, c) for r in range(1, n - 1)]

    min_count = int(np.ceil(0.35 * len(coords)))
    max_count = int(np.floor(0.60 * len(coords)))
    for r, c in random.sample(coords, random.randint(min_count, max_count)):
        grid[r, c] = int(noise_color)

    grid[0, 0] = int(bg)
    grid[0, -1] = int(bg)
    grid[-1, 0] = int(bg)
    grid[-1, -1] = int(bg)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    noise_color = 5
    n = random.randint(30, 38)
    grid = np.full((n, n), int(bg), dtype=int)

    all_colors = list(range(1, 10))
    noise_side = random.choice(SIDE_ORDER)
    control_sides = [side for side in SIDE_ORDER if side != noise_side]
    control_colors = random.sample([c for c in all_colors if c != noise_color], 3)
    controls = {side: color for side, color in zip(control_sides, control_colors)}
    _paint_frame(grid, controls, noise_side, bg, noise_color)

    n_objects = random.randint(2, 3)
    object_fills = random.sample([c for c in all_colors if c not in set(control_colors) | {noise_color}], n_objects)

    placed: list[tuple[tuple[int, int], np.ndarray]] = []
    for fill in object_fills:
        available_patterns = [c for c in all_colors if c not in set(control_colors) | {noise_color, fill}]
        pattern_count = random.randint(3, min(4, len(available_patterns)))
        pattern_colors = random.sample(available_patterns, pattern_count)

        core_h = random.randint(4, 8)
        core_w = random.randint(4, 8)
        core = _build_core(core_h, core_w, int(fill), pattern_colors)

        aligned = _attach_guides(core, set(control_sides), controls, bg)
        k = random.randrange(4)
        flip = bool(random.randrange(2))
        input_object = _inverse_transform(aligned, k, flip)

        try:
            r0, c0 = random_free_location(grid, input_object, transparent=bg, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, input_object, r0, c0, transparent=bg)
        placed.append(((r0, c0), core))

    placed.sort(key=lambda item: item[0])
    expected = _pack_cores([core for _pos, core in placed], bg, cols=2)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant")
