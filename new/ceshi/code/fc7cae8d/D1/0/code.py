# Variant of fc7cae8d
# Dimension: D1 — 匹配条件：边颜色对齐 (match_cond)
# Change: 引导边颜色 == 框架控制边颜色（edge_color_match），通过旋转/翻转实现对齐... → 形状嵌合匹配
#
# Design analysis for this variant:
# - Concrete behavior:
#   The original puzzle used two pure-color guide strips on the object, so matching was "which side has color X/Y".
#   Here each guide side is a 1px border layer with bg holes, producing a jagged silhouette. The frame keeps two
#   colored control sides, but each active side now also has a same-colored inward protrusion pattern on the inner
#   adjacent line. The solver must align the object so that its guide-side hole pattern equals the frame protrusion
#   pattern on the corresponding side, then crop away the 1px guide layer and output the core rectangle.
# - Small trace:
#   Old logic on a top+left target: "after transform, object top edge must be color ctrl_h and left edge color ctrl_v".
#   New logic on the same top+left target: "after transform, object top row's bg-hole mask must equal the top frame's
#   inner protrusion mask; object left column's bg-hole mask must equal the left frame's protrusion mask". Colors only
#   identify which frame sides are active; they no longer identify which object side matches which target.
# - Complexity calibration:
#   Pure strips would collapse this back to the original task, so the generator forces non-palindromic jagged masks
#   with both holes and solid cells, and rejects cases where two dihedral transforms satisfy the same pair of masks.
#   This keeps the change visually obvious and prevents accidental symmetry from making the puzzle trivial/ambiguous.
# - Hidden assumptions reviewed from the original solver:
#   `guide_edges()` assumed each guide side was almost monochrome and recoverable via majority color. That is false now:
#   a guide side contains both bg and guide color, so color-count logic would miss it.
#   The crop step still assumes guide thickness is exactly 1px; the generator preserves that.
#   `detect_controls_frame()` still works because active frame borders remain almost solid colors, with at most the
#   corner mismatch introduced by the orthogonal control side.
# - Placement legality:
#   The central object contains bg holes, so placement must treat 0 as transparent during collision checks/paste.
#   Frame protrusions are painted on the inner border line and are kept away from the control-side corner, so the two
#   control colors do not overwrite each other in the protrusion area.
# - Visual observability:
#   Across several samples, humans can see that the object carries jagged cutouts and the frame carries matching
#   comb-like protrusions. The output is still the inner rectangle, but the input no longer supports a color-only rule.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[5] / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    H, W = arr.shape
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, v in sides.items():
        vals = list(map(int, v))
        vals_nb = [x for x in vals if x != int(bg)]
        if not vals_nb:
            continue
        col, count = Counter(vals_nb).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, color) for side, color, _cnt, _L in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1):
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _frame_signature(arr: np.ndarray, side: str, color: int) -> np.ndarray | None:
    H, W = arr.shape
    if side == "top":
        line = arr[1, 1:-1]
    elif side == "bottom":
        line = arr[H - 2, 1:-1]
    elif side == "left":
        line = arr[1:-1, 1]
    else:
        line = arr[1:-1, W - 2]

    mask = np.array(line == int(color), dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return None
    lo, hi = int(idx[0]), int(idx[-1])
    sig = mask[lo : hi + 1]
    if int(sig.sum()) < 2 or int(sig.sum()) == len(sig):
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
    hole_count = int(holes.sum())
    if hole_count < 2 or hole_count >= len(holes):
        return None
    if int((~holes).sum()) < 2:
        return None
    return holes


def _object_signatures(arr: np.ndarray, bg: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side in ("top", "bottom", "left", "right"):
        sig = _object_side_signature(arr, side, bg)
        if sig is not None:
            out[side] = sig
    return out


def _matching_transforms(obj: np.ndarray, desired_shapes: dict[str, np.ndarray], bg: int) -> list[tuple[int, bool, np.ndarray]]:
    matches: list[tuple[int, bool, np.ndarray]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            got = _object_signatures(t, bg)
            if len(got) != len(desired_shapes):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the central object so its jagged guide-edge holes match the frame-side protrusion shapes.

    The two active frame sides are still detected from their almost-solid border colors, but orientation is solved
    by geometric complement: a frame protrusion mask must equal the bg-hole mask on the corresponding object side.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired_shapes: dict[str, np.ndarray] = {}
    for side, color in ctrls:
        sig = _frame_signature(g, side, color)
        if sig is not None:
            desired_shapes[side] = sig

    obj = _extract_center_object(g, bg)
    matches = _matching_transforms(obj, desired_shapes, bg)
    if matches:
        t = matches[0][2]
        top = 1 if "top" in desired_shapes else 0
        bottom = int(t.shape[0]) - (1 if "bottom" in desired_shapes else 0)
        left = 1 if "left" in desired_shapes else 0
        right = int(t.shape[1]) - (1 if "right" in desired_shapes else 0)
        return t[int(top) : int(bottom), int(left) : int(right)]
    return obj


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors), int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors), area // 10), max_nonfill)
    painted = 0

    for color in pattern_colors:
        r = random.randrange(h)
        c = random.randrange(w)
        core[r, c] = int(color)
        painted += 1

    while painted < target_nonfill:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [(random.randrange(h), random.randrange(w))]
        elif shape_kind == "line":
            line_len = random.randint(2, min(5, max(h, w)))
            if random.random() < 0.5:
                rr = random.randrange(h)
                c0 = random.randrange(max(1, w - line_len + 1))
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                r0 = random.randrange(max(1, h - line_len + 1))
                cc = random.randrange(w)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            mh = random.randint(1, min(3, h))
            mw = random.randint(1, min(3, w))
            r0 = random.randrange(h - mh + 1)
            c0 = random.randrange(w - mw + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        fill_cells = [(r, c) for r, c in cells if core[r, c] == fill]
        if len(fill_cells) > remaining:
            fill_cells = random.sample(fill_cells, remaining)
        for r, c in fill_cells:
            core[r, c] = int(color)
        painted += len(fill_cells)
        for r, c in cells:
            if core[r, c] != fill or painted >= target_nonfill:
                continue
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


def _attach_shape_guides(
    core: np.ndarray,
    guide_sides: tuple[str, str],
    guide_color: int,
    guide_shapes: dict[str, np.ndarray],
) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[0, :] = int(guide_color)
    if has_bottom:
        out[-1, :] = int(guide_color)
    if has_left:
        out[:, 0] = int(guide_color)
    if has_right:
        out[:, -1] = int(guide_color)

    if has_top:
        out[0, np.array(guide_shapes["top"], dtype=bool)] = 0
    if has_bottom:
        out[-1, np.array(guide_shapes["bottom"], dtype=bool)] = 0
    if has_left:
        out[np.array(guide_shapes["left"], dtype=bool), 0] = 0
    if has_right:
        out[np.array(guide_shapes["right"], dtype=bool), -1] = 0
    return out


def _paint_frame(
    grid: np.ndarray,
    horiz_side: str,
    vert_side: str,
    ctrl_h: int,
    ctrl_v: int,
    horiz_sig: np.ndarray,
    vert_sig: np.ndarray,
) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    c_start_min = 2
    c_start_max = n - len(horiz_sig) - 2
    r_start_min = 2
    r_start_max = n - len(vert_sig) - 2
    if c_start_min > c_start_max or r_start_min > r_start_max:
        raise ValueError("frame too small for signatures")

    c0 = random.randint(c_start_min, c_start_max)
    r0 = random.randint(r_start_min, r_start_max)

    if horiz_side == "top":
        grid[1, c0 : c0 + len(horiz_sig)][horiz_sig] = int(ctrl_h)
    else:
        grid[-2, c0 : c0 + len(horiz_sig)][horiz_sig] = int(ctrl_h)

    if vert_side == "left":
        grid[r0 : r0 + len(vert_sig), 1][vert_sig] = int(ctrl_v)
    else:
        grid[r0 : r0 + len(vert_sig), -2][vert_sig] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _nondegenerate_pair(sig_a: np.ndarray, sig_b: np.ndarray) -> bool:
    if np.array_equal(sig_a, sig_b):
        return False
    if len(sig_a) == len(sig_b) and np.array_equal(sig_a, sig_b[::-1]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 30)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    guide_color = random.choice(remaining)
    remaining = [c for c in remaining if c != guide_color]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(16, n - 6)
    max_core_w = min(16, n - 6)
    if max_core_h < 6 or max_core_w < 6:
        return None
    core_h = random.randint(6, max_core_h)
    core_w = random.randint(6, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    horiz_len = core_w + 1
    vert_len = core_h + 1
    for _ in range(60):
        horiz_sig = _random_signature(horiz_len)
        vert_sig = _random_signature(vert_len)
        if not _nondegenerate_pair(horiz_sig, vert_sig):
            continue
        guide_shapes = {horiz_side: horiz_sig, vert_side: vert_sig}
        aligned_object = _attach_shape_guides(core, (horiz_side, vert_side), guide_color, guide_shapes)
        desired = {horiz_side: horiz_sig, vert_side: vert_sig}
        matches = _matching_transforms(aligned_object, desired, 0)
        if len(matches) == 1:
            break
    else:
        return None

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    try:
        _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v, horiz_sig, vert_sig)
    except ValueError:
        return None

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    solved = main(grid)
    if not np.array_equal(solved, core):
        return None
    if len(_matching_transforms(_extract_center_object(grid, 0), desired, 0)) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D1 shape-match variant")
