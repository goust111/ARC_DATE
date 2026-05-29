# Combo variant of fc7cae8d (index 13)
# combo_spec: D2/4+D3/0
# Name: 空心源框+对边控制
#
# Design analysis for this combo variant
# a) Concrete behavior:
#    - Seed task: a filled rectangle carries two adjacent 1px guide strips; the border frame
#      carries two adjacent control edges. The solver tries dihedral transforms so guide colors
#      land on the matching frame sides, then trims those guide strips and returns the filled
#      core.
#    - Combo task: the source object is no longer a filled block. It is a hollow rectangle
#      frame: a 1px monochrome border enclosing background-colored interior. The two guide
#      strips sit on opposite outer sides, not adjacent ones. The border controls are also on
#      opposite grid sides. Example: a 7x11 hollow frame with top/bottom guide colors 2 and 8
#      may be rotated into an input whose guides are on left/right; the solver must rotate it
#      back so colors 2/8 align to the frame's top/bottom controls, then crop the guide rows
#      and output the 7x11 hollow frame.
#    - Key visual difference from the seed: output is a hollow frame with an empty interior,
#      not a filled rectangle, and control logic is axis-based rather than corner-based.
# b) Complexity calibration:
#    - The core is always a non-square hollow frame with a visible empty interior, so a 90
#      degree rotation changes the output shape and the D3 change is observable by eye.
#    - Control colors stay to two colors only, plus one frame color and optional grey noise on
#      the unused border sides. This keeps the scene readable.
#    - The generator always uses a 90 or 270 degree input rotation relative to the aligned
#      state, so samples cannot collapse into "already aligned, just crop".
# c) Implicit assumptions broken from the seed:
#    - The seed's guide detection assumed a filled rectangle, using its dominant color as the
#      "non-guide" baseline. That fails here because the object contains a large background hole.
#      New solver detects only fully uniform outer edges as guide strips.
#    - The seed relied on adjacent guides to uniquely pin an element of D4. With opposite guides
#      the task only determines an axis, so the generator avoids needing reflections and the
#      solver searches rotations only.
#    - The seed could crop by checking which sides matched desired controls. Here that still
#      works, but only after matching an opposite pair and validating that the cropped result is
#      a hollow frame.
# d) Placement legality:
#    - The object is placed strictly inside the border frame with border=1, so it never overlaps
#      the control edges or grey noise. Because paste/random_free_location treat 0 as
#      transparent, the hollow interior remains empty and non-overlapping.
# e) Visual observability:
#    - A human sees two opposite solid frame controls, a separate internal object whose two
#      colored guide strips are also opposite, and an output that is the same hollow frame
#      without guides. Across examples the axis mismatch is obvious because the source frame is
#      non-square.
# f) Visual cleanliness:
#    - Only one internal object exists. Unused frame sides receive sparse grey noise, but that
#      stays on the outer border and never intersects the object. The hollow frame uses a single
#      color, so guide strips remain easy to distinguish.
# g) Color inferability:
#    - Colors are structural rather than arbitrary: two guide colors correspond directly to two
#      opposite control edges; one separate color is the hollow frame itself. No extra color
#      mapping is required.
# h) Generator anti-degeneration check:
#    - Besides verifying main(input)==output, generation rejects samples where simply trimming the
#      input object's current guide pair already equals the true output. This ensures the axis
#      control genuinely matters.
# i) New solver assumptions audited:
#    - Control detection assumes one opposite pair of near-solid non-background frame edges; the
#      generator enforces exactly that pattern.
#    - Object extraction assumes the largest interior non-background component is the guided
#      frame; the generator creates only one interior object.
#    - Guide detection assumes the active guide sides are fully monochrome outer edges; the
#      generator draws guide strips as solid 1px rows/columns and never reuses the frame color
#      for guides.
#    - Hollow-frame validation assumes the cropped answer is a 1px rectangular ring with
#      background interior; the generator constructs exactly that family.

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

SIDES = ("top", "bottom", "left", "right")
OPPOSITE_PAIRS = (("top", "bottom"), ("left", "right"))


def _frame_edge_slices(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }


def _object_edge_slices(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }


def _detect_opposite_controls(arr: np.ndarray, bg: int) -> dict[str, int]:
    candidates: dict[str, tuple[int, int]] = {}
    for side, edge in _frame_edge_slices(arr).items():
        vals = [int(x) for x in edge if int(x) != int(bg)]
        if not vals:
            continue
        color, count = Counter(vals).most_common(1)[0]
        if int(count) >= len(edge) - 1:
            candidates[side] = (int(color), int(count))

    for side_a, side_b in OPPOSITE_PAIRS:
        if side_a in candidates and side_b in candidates:
            return {side_a: candidates[side_a][0], side_b: candidates[side_b][0]}

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


def _detect_object_guides(arr: np.ndarray, bg: int, allowed_colors: set[int] | None = None) -> dict[str, int]:
    guides: dict[str, int] = {}
    for side, edge in _object_edge_slices(arr).items():
        vals = [int(x) for x in edge]
        if not vals:
            continue
        color = int(vals[0])
        if color == int(bg):
            continue
        if allowed_colors is not None and color not in allowed_colors:
            continue
        if all(int(v) == color for v in vals):
            guides[side] = color
    return guides


def _pick_opposite_pair(side_to_color: dict[str, int]) -> dict[str, int]:
    for side_a, side_b in OPPOSITE_PAIRS:
        if side_a in side_to_color and side_b in side_to_color:
            return {side_a: side_to_color[side_a], side_b: side_to_color[side_b]}
    return {}


def _crop_guides(arr: np.ndarray, sides: dict[str, int]) -> np.ndarray:
    r0 = 1 if "top" in sides else 0
    r1 = arr.shape[0] - (1 if "bottom" in sides else 0)
    c0 = 1 if "left" in sides else 0
    c1 = arr.shape[1] - (1 if "right" in sides else 0)
    return arr[r0:r1, c0:c1]


def _is_hollow_frame(arr: np.ndarray, bg: int) -> bool:
    h, w = arr.shape
    if h < 3 or w < 3:
        return False
    border_vals = []
    for side in _object_edge_slices(arr).values():
        border_vals.extend(int(v) for v in side)
    if not border_vals:
        return False
    frame_color = int(border_vals[0])
    if frame_color == int(bg):
        return False
    if any(int(v) != frame_color for v in border_vals):
        return False
    return bool(np.all(arr[1:-1, 1:-1] == int(bg)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a guided hollow frame to opposite frame controls, then remove the guides.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_opposite_controls(g, bg)
    obj = _extract_center_object(g, bg)
    allowed_colors = set(map(int, desired.values()))

    for k in range(4):
        rotated = np.rot90(obj, int(k))
        guides = _pick_opposite_pair(_detect_object_guides(rotated, bg, allowed_colors))
        if set(guides) != set(desired):
            continue
        if any(int(guides[side]) != int(color) for side, color in desired.items()):
            continue
        candidate = _crop_guides(rotated, guides)
        if _is_hollow_frame(candidate, bg):
            return candidate

    return obj


def _build_hollow_core(h: int, w: int, frame_color: int, bg: int = 0) -> np.ndarray:
    core = np.full((h, w), int(bg), dtype=int)
    core[0, :] = int(frame_color)
    core[-1, :] = int(frame_color)
    core[:, 0] = int(frame_color)
    core[:, -1] = int(frame_color)
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

    for side in SIDES:
        if side in control_sides:
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        noise_count = random.randint(int(np.ceil(0.35 * len(coords))), int(np.floor(0.60 * len(coords))))
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _rotate_back(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -int(k))


def _naive_crop_current_orientation(obj: np.ndarray, bg: int, allowed_colors: set[int]) -> np.ndarray | None:
    guides = _pick_opposite_pair(_detect_object_guides(obj, bg, allowed_colors))
    if not guides:
        return None
    return _crop_guides(obj, guides)


def describe_fusion_behavior(grid: np.ndarray) -> dict[str, bool | tuple[str, str]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_opposite_controls(g, bg)
    obj = _extract_center_object(g, bg)
    input_guides = _pick_opposite_pair(_detect_object_guides(obj, bg))
    solved = main(g)
    naive = _naive_crop_current_orientation(obj, bg, set(desired.values()))
    return {
        "controls_pair": tuple(sorted(desired)),
        "input_guides_pair": tuple(sorted(input_guides)),
        "axis_mismatch": set(desired) != set(input_guides),
        "naive_differs_from_solution": naive is not None and not np.array_equal(naive, solved),
        "output_is_hollow": _is_hollow_frame(solved, bg),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    control_sides = random.choice(OPPOSITE_PAIRS)

    all_colors = list(range(1, 10))
    ctrl_a, ctrl_b = random.sample([c for c in all_colors if c != 5], 2)
    control_colors = {control_sides[0]: ctrl_a, control_sides[1]: ctrl_b}

    remaining = [c for c in all_colors if c not in {5, ctrl_a, ctrl_b}]
    frame_color = random.choice(remaining)

    max_core_h = min(16, n - 4)
    max_core_w = min(16, n - 4)
    if max_core_h < 6 or max_core_w < 6:
        return None

    for _ in range(40):
        core_h = random.randint(6, max_core_h)
        core_w = random.randint(6, max_core_w)
        if abs(core_h - core_w) >= 2:
            break
    else:
        return None

    core = _build_hollow_core(core_h, core_w, frame_color)
    aligned_object = _attach_guides(core, control_sides, control_colors)

    # Force an orthogonal guide axis so the opposite-side control logic is observable.
    input_object = _rotate_back(aligned_object, random.choice([1, 3]))

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, control_sides, control_colors)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    solved = main(grid)
    if not np.array_equal(solved, core):
        return None

    naive = _naive_crop_current_orientation(input_object, 0, set(control_colors.values()))
    if naive is None or np.array_equal(naive, core):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 13")
