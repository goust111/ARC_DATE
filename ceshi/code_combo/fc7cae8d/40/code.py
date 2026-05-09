# Combo variant of fc7cae8d (index 40)
# combo_spec: D1/0+D3/0+D4/2
# Name: 形状嵌合+对边控制+翻转
#
# Design analysis for this combo:
# a) Concrete behavior / small trace:
#    Original fc7cae8d: the frame activates two adjacent border sides, the inner object carries two adjacent
#    monochrome guide strips with matching colors, and the solver aligns by color then crops those strips away.
#    Combo behavior here: the frame activates two opposite sides instead (`top+bottom` or `left+right`), and each
#    active frame side owns a same-colored inward protrusion mask on the adjacent inner line. The object also has
#    two opposite 1px guide layers, but both guides share one unrelated color and encode information only through
#    bg holes. The solver must first find the unique dihedral transform whose two hole masks equal the two frame
#    protrusion masks on the same sides, then crop the matched guide layer, then apply one extra reflection to the
#    cropped core. Example: if the frame controls are `top+bottom`, the input object is generated with `left+right`
#    jagged guides; after rotating/flipping into the `top+bottom` pose, the solver crops those two rows and finally
#    mirrors the core left-right.
# b) Complexity calibration:
#    This intentionally keeps only two control layouts and one deterministic post-crop flip rule:
#    `top+bottom -> fliplr`, `left+right -> flipud`. That is enough to make the combo visible without creating too
#    many latent cases. The generator rejects palindromic / reversible guide masks, forces the two opposite masks to
#    differ, forces the input object axis to be orthogonal to the frame axis, and rejects samples where the final
#    reflection would leave the core unchanged. That prevents collapse back to "just rotate and crop".
# c) Hidden assumptions from the seed solver that break:
#    - `detect_controls_frame()` in the seed assumed any two near-solid sides were enough because adjacency encoded
#      a corner. Here the solver must explicitly prefer an opposite pair.
#    - `guide_edges()` assumed guide strips were near-monochrome and readable by majority color. That is false now:
#      guide edges mix one guide color with bg holes, so only a hole-mask extractor is valid.
#    - The seed searched all dihedral transforms safely because adjacent colored guides pin orientation. With opposite
#      guides, reflections can be ambiguous unless the two masks are asymmetric and distinct; the generator enforces
#      that uniqueness before accepting a sample.
#    - The seed output was the cropped core directly. This combo adds a second stage: crop first, then reflect based
#      on which opposite control pair is active.
# d) Placement legality:
#    The object contains bg holes, so placement treats color 0 as transparent both in `random_free_location` and
#    `paste`. The frame protrusions live on the inner border line of the active sides only; the object is placed with
#    `border=1`, so it cannot overlap the frame controls or their protrusions.
# e) Visual observability:
#    Across examples, a human sees: (1) the frame controls are on opposite sides rather than a corner, (2) the
#    object's opposite guides use the same color so color alone is useless, (3) the comb-like holes fit the comb-like
#    border protrusions after axis alignment, and (4) the output is the cropped rectangle but reflected along the axis
#    implied by the active control pair. Those are all visually available from 2-4 examples.
# f) Visual cleanliness:
#    The two frame protrusion masks on opposite sides share one projected span, which keeps the scene tidy rather than
#    scattering unrelated marks around the border. The object has only one internal rectangle and one guide color, so
#    shape cues are easy to isolate from the frame cues.
# g) Color-change inferability:
#    No hidden color mapping is introduced. Frame colors only identify which border sides are active; object guide
#    color is intentionally unrelated to frame colors, which makes "shape, not color" the only viable alignment cue.
# h) Quantifiable anti-degeneration checks used by the generator:
#    - the matching transform count over all 8 dihedral transforms must be exactly 1;
#    - the input guide axis must be orthogonal to the frame control axis;
#    - the final reflected output must differ from the unreflected cropped core.
# i) Audit of new solver assumptions:
#    - `_frame_signature()` assumes each active side has exactly one contiguous span of protrusion positions with
#      active cells at both ends; the generator constructs signatures that satisfy this.
#    - `_object_side_signature()` assumes non-guide sides contain no bg holes; the generator keeps a solid fill border
#      around the core so only the guide sides expose bg on the object's bbox boundary.
#    - `_matching_transforms()` assumes the object exposes exactly two valid guide sides; the generator always attaches
#      one opposite pair of guides and rejects ambiguous transforms.
#    - `_post_crop_flip()` assumes the active controls are one of the two opposite pairs; `_detect_opposite_controls()`
#      and the frame painter use exactly those two layouts.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[4] / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

SIDES = ("top", "bottom", "left", "right")
OPPOSITE_PAIRS = (("top", "bottom"), ("left", "right"))


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _edge_slices(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }


def _pair_key(sides: tuple[str, str] | list[str] | set[str] | dict[str, int]) -> tuple[str, str]:
    keys = set(sides if not isinstance(sides, dict) else sides.keys())
    if keys == {"top", "bottom"}:
        return ("top", "bottom")
    return ("left", "right")


def _post_crop_flip(arr: np.ndarray, control_sides: tuple[str, str] | list[str] | set[str] | dict[str, int]) -> np.ndarray:
    if set(_pair_key(control_sides)) == {"top", "bottom"}:
        return np.fliplr(arr)
    return np.flipud(arr)


def _detect_opposite_controls(arr: np.ndarray, bg: int) -> dict[str, int]:
    candidates: dict[str, tuple[int, int]] = {}
    for side, edge in _edge_slices(arr).items():
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


def _frame_signature(arr: np.ndarray, side: str, color: int) -> np.ndarray | None:
    h, w = arr.shape
    if side == "top":
        line = arr[1, 1:-1]
    elif side == "bottom":
        line = arr[h - 2, 1:-1]
    elif side == "left":
        line = arr[1:-1, 1]
    else:
        line = arr[1:-1, w - 2]

    mask = np.array(line == int(color), dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return None
    lo, hi = int(idx[0]), int(idx[-1])
    sig = mask[lo : hi + 1]
    if int(sig.sum()) < 3:
        return None
    if int(sig.sum()) == len(sig) or int((~sig).sum()) < 2:
        return None
    return sig


def _object_side_signature(arr: np.ndarray, side: str, bg: int) -> np.ndarray | None:
    if side == "top":
        outer = arr[0, :]
    elif side == "bottom":
        outer = arr[-1, :]
    elif side == "left":
        outer = arr[:, 0]
    else:
        outer = arr[:, -1]

    holes = np.array(outer == int(bg), dtype=bool)
    if int(holes.sum()) < 3:
        return None
    if int(holes.sum()) == len(holes) or int((~holes).sum()) < 2:
        return None
    return holes


def _object_signatures(arr: np.ndarray, bg: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side in SIDES:
        sig = _object_side_signature(arr, side, bg)
        if sig is not None:
            out[side] = sig
    return out


def _object_guide_pair(arr: np.ndarray, bg: int) -> tuple[str, str] | None:
    sigs = _object_signatures(arr, bg)
    if len(sigs) != 2:
        return None
    for pair in OPPOSITE_PAIRS:
        if set(pair) == set(sigs):
            return pair
    return None


def _matching_transforms(
    obj: np.ndarray, desired_shapes: dict[str, np.ndarray], bg: int
) -> list[tuple[int, bool, np.ndarray]]:
    matches: list[tuple[int, bool, np.ndarray]] = []
    desired_sides = set(desired_shapes)
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            got = _object_signatures(t, bg)
            if set(got) != desired_sides:
                continue
            ok = True
            for side, sig in desired_shapes.items():
                cur = got.get(side)
                if cur is None or len(cur) != len(sig) or not np.array_equal(cur, sig):
                    ok = False
                    break
            if ok:
                matches.append((int(k), bool(flip), t))
    return matches


def _crop_guides(arr: np.ndarray, sides: tuple[str, str] | list[str] | set[str] | dict[str, int]) -> np.ndarray:
    keys = set(sides if not isinstance(sides, dict) else sides.keys())
    r0 = 1 if "top" in keys else 0
    r1 = arr.shape[0] - (1 if "bottom" in keys else 0)
    c0 = 1 if "left" in keys else 0
    c1 = arr.shape[1] - (1 if "right" in keys else 0)
    return arr[r0:r1, c0:c1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align opposite jagged guide edges to opposite frame controls, crop them, then reflect the core.

    `top+bottom` controls mean "match those two shape signatures, crop the two rows, then mirror left-right".
    `left+right` controls mean "match those two shape signatures, crop the two columns, then mirror up-down".
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    desired = _detect_opposite_controls(g, bg)
    desired_shapes: dict[str, np.ndarray] = {}
    for side, color in desired.items():
        sig = _frame_signature(g, side, color)
        if sig is not None:
            desired_shapes[side] = sig

    obj = _extract_center_object(g, bg)
    if set(desired_shapes) != set(desired):
        return obj

    matches = _matching_transforms(obj, desired_shapes, bg)
    if not matches:
        return obj

    aligned = matches[0][2]
    core = _crop_guides(aligned, desired_shapes)
    return _post_crop_flip(core, desired_shapes)


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    inner_rows = list(range(1, h - 1))
    inner_cols = list(range(1, w - 1))
    inner_cells = [(r, c) for r in inner_rows for c in inner_cols]
    random.shuffle(inner_cells)

    lo = max(len(pattern_colors) + 3, len(inner_cells) // 8)
    hi = max(len(pattern_colors) + 6, int(len(inner_cells) * 0.35))
    target_nonfill = random.randint(lo, hi)
    painted = 0

    for color in pattern_colors:
        if not inner_cells:
            break
        r, c = inner_cells.pop()
        core[r, c] = int(color)
        painted += 1

    while painted < target_nonfill and inner_cells:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [random.choice(inner_cells)]
        elif shape_kind == "line":
            if random.random() < 0.5:
                rr = random.choice(inner_rows)
                line_len = random.randint(2, min(4, w - 2))
                c0 = random.randint(1, w - 1 - line_len)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                cc = random.choice(inner_cols)
                line_len = random.randint(2, min(4, h - 2))
                r0 = random.randint(1, h - 1 - line_len)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            mh = random.randint(1, min(3, h - 2))
            mw = random.randint(1, min(3, w - 2))
            r0 = random.randint(1, h - 1 - mh)
            c0 = random.randint(1, w - 1 - mw)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        for r, c in cells:
            if painted >= target_nonfill:
                break
            if core[r, c] == int(fill):
                core[r, c] = int(color)
                painted += 1

    return core


def _random_signature(length: int) -> np.ndarray:
    while True:
        sig = np.zeros(int(length), dtype=bool)
        sig[0] = True
        sig[-1] = True
        interior = list(range(1, length - 1))
        max_extra = min(len(interior) - 1, max(2, length // 2))
        if max_extra < 1:
            continue
        extra_n = random.randint(1, max_extra)
        for idx in random.sample(interior, extra_n):
            sig[int(idx)] = True
        if int(sig.sum()) < 3:
            continue
        if int((~sig).sum()) < 2:
            continue
        if np.array_equal(sig, sig[::-1]):
            continue
        return sig


def _nondegenerate_pair(sig_a: np.ndarray, sig_b: np.ndarray) -> bool:
    if np.array_equal(sig_a, sig_b):
        return False
    if len(sig_a) == len(sig_b) and np.array_equal(sig_a, sig_b[::-1]):
        return False
    return True


def _attach_shape_guides(
    core: np.ndarray,
    guide_sides: tuple[str, str],
    guide_color: int,
    guide_shapes: dict[str, np.ndarray],
    bg: int = 0,
) -> np.ndarray:
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
        out[0, c0 : c0 + w] = int(guide_color)
        top_idx = np.flatnonzero(np.array(guide_shapes["top"], dtype=bool))
        out[0, c0 + top_idx] = int(bg)
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_color)
        bottom_idx = np.flatnonzero(np.array(guide_shapes["bottom"], dtype=bool))
        out[-1, c0 + bottom_idx] = int(bg)
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_color)
        left_idx = np.flatnonzero(np.array(guide_shapes["left"], dtype=bool))
        out[r0 + left_idx, 0] = int(bg)
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_color)
        right_idx = np.flatnonzero(np.array(guide_shapes["right"], dtype=bool))
        out[r0 + right_idx, -1] = int(bg)
    return out


def _paint_frame(
    grid: np.ndarray,
    control_sides: tuple[str, str],
    control_colors: dict[str, int],
    guide_shapes: dict[str, np.ndarray],
) -> None:
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

    if set(control_sides) == {"top", "bottom"}:
        span = len(guide_shapes["top"])
        c_min = 2
        c_max = n - span - 2
        if c_min > c_max:
            raise ValueError("frame too small for horizontal signatures")
        c0 = random.randint(c_min, c_max)
        top_idx = np.flatnonzero(np.array(guide_shapes["top"], dtype=bool))
        bottom_idx = np.flatnonzero(np.array(guide_shapes["bottom"], dtype=bool))
        grid[1, c0 + top_idx] = int(control_colors["top"])
        grid[-2, c0 + bottom_idx] = int(control_colors["bottom"])
    else:
        span = len(guide_shapes["left"])
        r_min = 2
        r_max = n - span - 2
        if r_min > r_max:
            raise ValueError("frame too small for vertical signatures")
        r0 = random.randint(r_min, r_max)
        left_idx = np.flatnonzero(np.array(guide_shapes["left"], dtype=bool))
        right_idx = np.flatnonzero(np.array(guide_shapes["right"], dtype=bool))
        grid[r0 + left_idx, 1] = int(control_colors["left"])
        grid[r0 + right_idx, -2] = int(control_colors["right"])

    for side in SIDES:
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
    n = random.randint(24, 31)
    control_sides = random.choice(OPPOSITE_PAIRS)

    all_colors = list(range(1, 10))
    ctrl_a, ctrl_b = random.sample([c for c in all_colors if c != 5], 2)
    control_colors = {control_sides[0]: ctrl_a, control_sides[1]: ctrl_b}
    remaining = [c for c in all_colors if c not in {5, ctrl_a, ctrl_b}]
    guide_color = random.choice(remaining)
    remaining = [c for c in remaining if c != guide_color]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    max_core_h = min(16, n - 6)
    max_core_w = min(16, n - 6)
    if max_core_h < 6 or max_core_w < 6:
        return None

    for _ in range(30):
        core_h = random.randint(6, max_core_h)
        core_w = random.randint(6, max_core_w)
        if core_h != core_w:
            break
    else:
        return None

    core = _build_core(core_h, core_w, fill, pattern_colors)
    expected = _post_crop_flip(core, control_sides)
    if np.array_equal(expected, core):
        return None

    sig_len = core_w if set(control_sides) == {"top", "bottom"} else core_h
    for _ in range(100):
        sig_a = _random_signature(sig_len)
        sig_b = _random_signature(sig_len)
        if not _nondegenerate_pair(sig_a, sig_b):
            continue
        guide_shapes = {control_sides[0]: sig_a, control_sides[1]: sig_b}
        aligned_object = _attach_shape_guides(core, control_sides, guide_color, guide_shapes)
        if len(_matching_transforms(aligned_object, guide_shapes, 0)) != 1:
            continue
        break
    else:
        return None

    k = random.choice([1, 3])
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)
    input_pair = _object_guide_pair(input_object, 0)
    if input_pair is None or set(input_pair) == set(control_sides):
        return None

    grid = np.zeros((n, n), dtype=int)
    try:
        _paint_frame(grid, control_sides, control_colors, guide_shapes)
    except ValueError:
        return None

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), expected):
        return None
    if len(_matching_transforms(_extract_center_object(grid, 0), guide_shapes, 0)) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo 40")
