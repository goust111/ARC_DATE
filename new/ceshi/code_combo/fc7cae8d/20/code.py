# Combo variant of fc7cae8d (index 20)
# combo_spec: D3/0+D4/2
# Name: 对边控制+翻转后裁剪

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

"""
Design analysis for the combo variant D3/0 + D4/2

a) Concrete behavior and trace
   - Original fc7cae8d:
       the frame exposes two adjacent solid control edges; the object carries two adjacent
       guide strips of the same colors. Solver searches a dihedral transform so each guide
       color lands on the matching frame side, then crops those guide strips away.
   - Combo behavior:
       the frame controls are now opposite edges, either top+bottom or left+right. The
       object also carries opposite guide strips in the aligned state. Solver first aligns
       the guide axis to the frame axis, crops those two guide strips, and then performs an
       extra left-right flip on the cropped core.
   - Small trace:
       suppose the frame controls are top=2 and bottom=7. The aligned object is
       [guide 2]
       [ core   ]
       [guide 7]
       but the input object is rotated 90 degrees, so its visible guides are left=7 and
       right=2. The original puzzle would never use this opposite-side axis logic. The
       combo solver rotates the object back so guides become top=2 and bottom=7, crops
       those rows, and finally outputs fliplr(core). Without the D4 step the result would
       be just core; without the D3 step the opposite-side alignment itself would fail.

b) Complexity calibration
   - Opposite-side guides remove the corner cue that made the seed fully dihedral. To keep
     the task solvable, the generator uses only quarter-turn rotations and always chooses a
     90/270-degree mismatch between object-guide axis and frame-control axis.
   - The core is forced to be non-square and horizontally asymmetric. That does two jobs:
     it makes the axis realignment visible, and it prevents the post-crop horizontal flip
     from collapsing back to the plain cropped core.
   - Pattern colors stay at 2..4 extra colors, with a fill-colored outer ring around the
     core. That keeps guide detection stable and the output visually clean.

c) Hidden assumptions audit against the original main()
   - Original `detect_controls_frame()` only needed “best two near-solid sides”, because
     adjacency encoded the destination corner. That assumption breaks here: the solver must
     explicitly prefer an opposite pair.
   - Original dihedral search over rotations + mirrors is unsafe for opposite guides,
     because a mirror parallel to the guide axis preserves the same visible guide sides and
     colors while changing the core orientation. The combo solver therefore searches only
     rotations, and the generator guarantees that restriction is valid.
   - Original crop returned the final answer. That breaks D4/2: after recovering the core,
     the solver must apply one extra horizontal reflection.
   - Guide detection assumes the fill color is the dominant color inside the object crop.
     The generator preserves a fill border and caps non-fill density so this remains true.

d) Placement legality
   - The frame occupies only the outer border; the object is inserted strictly inside it
     with `border=1`, so it never overwrites controls or border noise.
   - The object is one connected component: the core touches both opposite guide strips.
   - Core dimensions are capped so the guided object fits comfortably inside the frame,
     and `random_free_location()` rejects any illegal placement.

e) Visual observability
   - Across a few samples, a human sees the object guides lie on the wrong axis relative to
     the frame controls, so a rotation is necessary before cropping.
   - The output then differs from the plain cropped core by a consistent left-right mirror.
     Because the core is deliberately asymmetric, this extra step is visible rather than a
     silent no-op.

f) Visual cleanliness
   - Border controls and border noise live on disjoint frame sides.
   - Guides are thin full-length strips on only two object sides, and the core keeps a fill
     margin, so the object reads as “framed core with two colored guide bars” rather than a
     cluttered multi-layer overlay.

g) Color-rule inferability
   - No new color remapping is introduced. The two guide colors still directly match the two
     frame-control colors, and the extra D4 change is geometric only (fixed horizontal flip).

Post-change solver assumptions audited against the generator
   - Exactly one opposite control pair is near-solid on the frame: guaranteed by construction.
   - The input object differs from the aligned state by rotation only, never by a reflection
     parallel to the guide axis: guaranteed by choosing only 90/270-degree inverse rotations.
   - The cropped core is horizontally asymmetric, so `fliplr(core) != core`: enforced by a
     measurable impact threshold in the generator.
   - The object is the largest non-border connected component: guaranteed because border noise
     is sparse and disconnected while the object is a dense rectangle-plus-guides component.
"""

OPPOSITE_PAIRS = (("top", "bottom"), ("left", "right"))


def _edge_slices(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }


def _detect_opposite_controls(arr: np.ndarray, bg: int) -> dict[str, int]:
    candidates: dict[str, tuple[int, int]] = {}
    for side, edge in _edge_slices(arr).items():
        vals = [int(x) for x in edge if int(x) != int(bg)]
        if not vals:
            continue
        color, count = Counter(vals).most_common(1)[0]
        if int(count) >= len(edge) - 1:
            candidates[side] = (int(color), int(count))

    for a, b in OPPOSITE_PAIRS:
        if a in candidates and b in candidates:
            return {a: candidates[a][0], b: candidates[b][0]}

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][1], reverse=True)[:2]
    return {side: color for side, (color, _count) in ranked}


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    out: dict[str, int] = {}
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    for side, edge in edges.items():
        vals = [int(x) for x in edge]
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        color, count = Counter(non_fill).most_common(1)[0]
        if int(color) != int(fill) and int(count) >= len(vals) - 1:
            out[side] = int(color)
    return out


def _pick_opposite_guides(edges: dict[str, int], allowed_colors: set[int]) -> dict[str, int]:
    filtered = {side: color for side, color in edges.items() if int(color) in allowed_colors}
    for a, b in OPPOSITE_PAIRS:
        if a in filtered and b in filtered:
            return {a: filtered[a], b: filtered[b]}
    return filtered


def _crop_guides(arr: np.ndarray, sides: dict[str, int]) -> np.ndarray:
    r0 = 1 if "top" in sides else 0
    r1 = arr.shape[0] - (1 if "bottom" in sides else 0)
    c0 = 1 if "left" in sides else 0
    c1 = arr.shape[1] - (1 if "right" in sides else 0)
    return arr[r0:r1, c0:c1]


def _solve_crop_only(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_opposite_controls(g, bg)
    obj = _extract_center_object(g, bg)
    allowed_colors = set(map(int, desired.values()))

    for k in range(4):
        rotated = np.rot90(obj, int(k))
        guides = _pick_opposite_guides(_guide_edges(rotated), allowed_colors)
        if set(guides) != set(desired):
            continue
        if all(int(guides[side]) == int(color) for side, color in desired.items()):
            return _crop_guides(rotated, desired)

    return obj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the opposite-guide alignment, crop off the guides, then horizontally flip.
    """
    core = _solve_crop_only(grid)
    return np.fliplr(core)


def _horizontal_flip_impact(arr: np.ndarray) -> int:
    return int(np.sum(np.asarray(arr) != np.fliplr(np.asarray(arr))))


def _rotate_back(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -int(k))


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    colors = [int(c) for c in pattern_colors]

    # Fixed asymmetric anchors make the post-crop horizontal flip consistently observable.
    core[1 : min(h - 1, 1 + max(2, h // 2)), 1] = colors[0]

    right_r0 = max(1, h // 2 - 1)
    right_r1 = min(h - 1, right_r0 + 2)
    right_c0 = max(1, w - 3)
    core[right_r0:right_r1, right_c0 : w - 1] = colors[min(1, len(colors) - 1)]

    line_len = max(2, min(4, w - 3))
    line_c0 = max(1, w // 2 - line_len // 2)
    core[1, line_c0 : line_c0 + line_len] = colors[min(2, len(colors) - 1)]

    core[h - 2, min(w - 2, 2)] = colors[-1]

    inner_area = max(1, (h - 2) * (w - 2))
    target_nonfill = random.randint(
        max(len(colors) + 5, inner_area // 7),
        max(len(colors) + 8, int(inner_area * 0.35)),
    )

    while int(np.sum(core != int(fill))) < target_nonfill:
        color = int(random.choice(colors))
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [(random.randint(1, h - 2), random.randint(1, w - 2))]
        elif shape_kind == "line":
            if random.random() < 0.5:
                rr = random.randint(1, h - 2)
                line_len = random.randint(2, min(4, w - 2))
                c0 = random.randint(1, w - 1 - line_len)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                cc = random.randint(1, w - 2)
                line_len = random.randint(2, min(4, h - 2))
                r0 = random.randint(1, h - 1 - line_len)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            rh = random.randint(1, min(3, h - 2))
            rw = random.randint(1, min(3, w - 2))
            r0 = random.randint(1, h - 1 - rh)
            c0 = random.randint(1, w - 1 - rw)
            cells = [(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw)]

        for r, c in cells:
            if core[r, c] == int(fill) or random.random() < 0.35:
                core[r, c] = int(color)

    return core


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int], bg: int = 0) -> np.ndarray:
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
    return out


def _paint_frame(grid: np.ndarray, control_sides: tuple[str, str], control_colors: dict[str, int]) -> None:
    n = grid.shape[0]
    for side in control_sides:
        color = int(control_colors[side])
        if side == "top":
            grid[0, :] = color
        elif side == "bottom":
            grid[-1, :] = color
        elif side == "left":
            grid[:, 0] = color
        else:
            grid[:, -1] = color

    for side in ("top", "bottom", "left", "right"):
        if side in control_sides:
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        for r, c in random.sample(coords, random.randint(min_count, max_count)):
            grid[r, c] = 5


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(23, 30)
    control_sides = random.choice(OPPOSITE_PAIRS)

    all_colors = list(range(1, 10))
    ctrl_a, ctrl_b = random.sample([c for c in all_colors if c != 5], 2)
    control_colors = {control_sides[0]: ctrl_a, control_sides[1]: ctrl_b}
    remaining = [c for c in all_colors if c not in {5, ctrl_a, ctrl_b}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    max_core_h = min(15, n - 5)
    max_core_w = min(16, n - 5)
    if max_core_h < 6 or max_core_w < 7:
        return None

    for _ in range(30):
        core_h = random.randint(6, max_core_h)
        core_w = random.randint(7, max_core_w)
        if core_h != core_w:
            break
    else:
        return None

    core = _build_core(core_h, core_w, fill, pattern_colors)
    flip_impact = _horizontal_flip_impact(core)
    min_impact = max(6, (core_h * core_w) // 10)
    if flip_impact < min_impact:
        return None

    aligned_object = _attach_guides(core, control_sides, control_colors)
    input_object = _rotate_back(aligned_object, random.choice([1, 3]))

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, control_sides, control_colors)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    crop_only = _solve_crop_only(grid)
    out = main(grid)
    expected = np.fliplr(core)

    if not np.array_equal(crop_only, core):
        return None
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, crop_only):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D3/0+D4/2")
