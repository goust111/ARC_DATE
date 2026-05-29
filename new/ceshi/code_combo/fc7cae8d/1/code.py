# Combo variant of fc7cae8d (index 1)
# combo_spec: D1/0+D2/2
# Name: 形状嵌合+非矩形源
#
# Design analysis for the fused variant:
# 1. Concrete behavior and trace:
#    The original task had a rectangular object whose two guide edges were full-color strips,
#    so solving was "find the transform whose top/left/etc. strip colors match the frame colors,
#    then slice away one full row/column".
#    In this combo, the output core is a non-rectangular connected shape (mostly L-shapes, plus
#    some T-shapes). The object is surrounded only on two adjacent viewing directions by a 1px
#    guide layer that follows the directional silhouette of the core rather than a full straight
#    row/column. That guide layer has bg holes, so each active side exposes a jagged binary
#    signature. The frame still marks one horizontal and one vertical control side with solid
#    border colors, but the inner line next to each active frame side also carries a protrusion
#    mask. The solver must find the dihedral transform whose top/bottom and left/right frontier
#    guide masks exactly match those frame protrusion masks, then peel off the guide cells seen
#    from the matched directions and crop the remaining non-rectangular core.
#    Small example: for a top+left target on an L-shape, the old logic would ask whether the
#    object's top row is color A and left column is color B. The fused logic instead asks whether
#    the transformed object's top frontier contains guide-color cells at exactly the same on/off
#    positions as the top frame protrusions, and likewise for the left frontier. Near the notch,
#    frontier cells sit at varying depths, so this is a directional silhouette problem rather
#    than a border-color or fixed-row problem.
# 2. Complexity calibration:
#    The generator keeps the shape family small (L or T) so examples remain readable, but it
#    forces both active side signatures to contain holes and solid runs, rejects palindromic or
#    degenerate signatures, and rejects any sample where more than one transform/color hypothesis
#    satisfies the geometric match. This prevents collapse back to color-matching or to a trivial
#    all-on border.
# 3. Hidden assumptions broken from the original main():
#    - `guide_edges()` assumed each relevant border is almost monochrome. That is false now:
#      guides are sparse and live on irregular silhouette frontier cells.
#    - `return t[top:bottom, left:right]` assumed deleting one whole row/column removes the
#      guides. That is false because only a subset of frontier cells are guide cells.
#    - Using the most common internal color to detect the guide layer is unsafe once the cropped
#      output bbox contains bg cavities from the non-rectangular core.
#    - The old color-equality rule no longer applies because the object uses a single guide color
#      unrelated to the two frame-control colors; only the binary shape signatures align.
#    The only major original assumption kept is that the active frame borders are still almost
#    solid, so the control sides can still be detected by border-color majority.
# 4. Placement legality:
#    The object contains bg holes and guide gaps, so placement uses `transparent=0`. The frame
#    protrusions are painted on the inner border line and the object is placed with `border=1`,
#    so nothing overlaps or touches the outer frame.
# 5. Visual observability:
#    Across examples, humans can see three stable cues: the output is visibly non-rectangular,
#    the input object has a distinct guide-colored jagged fringe only on two adjacent viewing
#    directions, and the frame carries matching comb-like protrusions on one horizontal plus one
#    vertical side. The rule is readable from shape alone; color does not tell which object side
#    maps to which frame side.
# 6. Cleanliness:
#    The guide color is unique and used only for the object fringe. Frame control colors stay on
#    the border/protrusions only. Internal pattern colors stay inside the core. This avoids heavy
#    overlap and keeps the object, frame, and guide system visually separable.
# 7. Color inferability:
#    The transformation does not require learning a color mapping. Colors only partition regions:
#    frame controls identify which two frame sides are active, while the object uses one repeated
#    guide color on both active sides. The actual alignment signal is the repeated binary shape
#    pattern, which is supported twice per input (one horizontal, one vertical).
# 8. New solver assumptions introduced here and guaranteed by the generator:
#    - Every row and column of the cropped object bbox contains at least one occupied cell, so a
#      directional frontier exists for each index on every side.
#    - Exactly one repeated guide color can realize both active-side signatures after the correct
#      transform.
#    - The matched frontier guide cells can be removed independently per side without disconnecting
#      the core into multiple separate outputs that would change the intended bbox.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _bbox_nonbg(arr: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(arr != int(bg))
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, vals in sides.items():
        seq = list(map(int, vals))
        seq_nb = [x for x in seq if x != int(bg)]
        if not seq_nb:
            continue
        color, count = Counter(seq_nb).most_common(1)[0]
        if int(count) >= len(seq) - 1:
            cand.append((side, int(color), int(count), int(len(seq))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, color) for side, color, _cnt, _L in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _silhouette_frontier(arr: np.ndarray, side: str, bg: int) -> list[tuple[int, int]]:
    H, W = arr.shape
    out: list[tuple[int, int]] = []
    if side == "top":
        for c in range(W):
            rows = np.flatnonzero(arr[:, c] != int(bg))
            if rows.size:
                out.append((int(rows[0]), int(c)))
    elif side == "bottom":
        for c in range(W):
            rows = np.flatnonzero(arr[:, c] != int(bg))
            if rows.size:
                out.append((int(rows[-1]), int(c)))
    elif side == "left":
        for r in range(H):
            cols = np.flatnonzero(arr[r, :] != int(bg))
            if cols.size:
                out.append((int(r), int(cols[0])))
    elif side == "right":
        for r in range(H):
            cols = np.flatnonzero(arr[r, :] != int(bg))
            if cols.size:
                out.append((int(r), int(cols[-1])))
    else:
        raise ValueError(f"unknown side: {side}")
    return out


def _frame_signature(arr: np.ndarray, side: str, color: int) -> np.ndarray | None:
    H, W = arr.shape
    if side == "top":
        line = arr[1, 1:-1]
    elif side == "bottom":
        line = arr[H - 2, 1:-1]
    elif side == "left":
        line = arr[1:-1, 1]
    elif side == "right":
        line = arr[1:-1, W - 2]
    else:
        raise ValueError(f"unknown side: {side}")

    mask = np.array(line == int(color), dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return None
    lo, hi = int(idx[0]), int(idx[-1])
    sig = mask[lo : hi + 1]
    if int(sig.sum()) < 3 or int((~sig).sum()) < 2:
        return None
    return sig


def _object_side_signature(arr: np.ndarray, side: str, color: int, bg: int) -> np.ndarray | None:
    frontier = _silhouette_frontier(arr, side, bg)
    if not frontier:
        return None
    sig = np.array([int(arr[r, c]) == int(color) for r, c in frontier], dtype=bool)
    return sig


def _peel_guides(arr: np.ndarray, sides: list[str], guide_color: int, bg: int) -> np.ndarray | None:
    out = np.array(arr, copy=True)
    for side in sides:
        for r, c in _silhouette_frontier(out, side, bg):
            if int(out[r, c]) == int(guide_color):
                out[r, c] = int(bg)
    b = _bbox_nonbg(out, bg)
    if b is None:
        return None
    return crop_bbox(out, b)


def _all_matches(
    obj: np.ndarray, desired_shapes: dict[str, np.ndarray], bg: int
) -> list[tuple[int, bool, int, np.ndarray]]:
    matches: list[tuple[int, bool, int, np.ndarray]] = []
    if len(desired_shapes) != 2:
        return matches

    desired_sides = list(desired_shapes.keys())
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            frontier_counts: Counter[int] = Counter()
            for side in desired_sides:
                for r, c in _silhouette_frontier(t, side, bg):
                    frontier_counts[int(t[r, c])] += 1
            colors = [color for color, count in frontier_counts.items() if color != int(bg) and int(count) >= 3]
            for color in colors:
                ok = True
                for side, sig in desired_shapes.items():
                    cur = _object_side_signature(t, side, color, bg)
                    if cur is None or len(cur) != len(sig) or not np.array_equal(cur, sig):
                        ok = False
                        break
                if not ok:
                    continue
                peeled = _peel_guides(t, desired_sides, color, bg)
                if peeled is None:
                    continue
                matches.append((int(k), bool(flip), int(color), peeled))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a non-rectangular object by matching guide-hole signatures to frame protrusions.

    The frame colors still identify which horizontal and vertical frame sides are active,
    but the object itself does not match those colors. Instead, the correct transform is
    the one whose directional frontier guide mask matches the two frame protrusion masks.
    After alignment, only the guide-colored frontier cells on those matched sides are
    peeled away, and the remaining irregular core is cropped.
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
    matches = _all_matches(obj, desired_shapes, bg)
    if matches:
        return matches[0][3]
    return obj


def _random_signature(length: int) -> np.ndarray:
    while True:
        sig = np.zeros(int(length), dtype=bool)
        sig[0] = True
        sig[-1] = True
        interior = list(range(1, length - 1))
        max_extra = min(len(interior), max(2, length // 2))
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


def _make_l_mask(h: int, w: int, active_corner: tuple[str, str]) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    notch_h = random.randint(2, max(2, h // 2))
    notch_w = random.randint(2, max(2, w // 2))
    notch_h = min(notch_h, h - 2)
    notch_w = min(notch_w, w - 2)

    if active_corner == ("top", "left"):
        mask[:notch_h, :notch_w] = False
    elif active_corner == ("top", "right"):
        mask[:notch_h, w - notch_w :] = False
    elif active_corner == ("bottom", "left"):
        mask[h - notch_h :, :notch_w] = False
    elif active_corner == ("bottom", "right"):
        mask[h - notch_h :, w - notch_w :] = False
    else:
        raise ValueError("active_corner must be adjacent frame sides")
    return mask


def _make_t_mask(h: int, w: int, active_corner: tuple[str, str]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    bar_h = random.randint(2, min(4, h - 3))
    stem_w = random.randint(2, min(max(2, w // 3), w - 2))

    if active_corner[1] == "left":
        lo = 1
        hi = min(max(1, w // 3), w - stem_w - 1)
    else:
        lo = max(1, w - stem_w - max(1, w // 3))
        hi = w - stem_w - 1
    if lo > hi:
        return _make_l_mask(h, w, active_corner)
    stem_c0 = random.randint(lo, hi)

    if active_corner[0] == "top":
        mask[:bar_h, :] = True
        mask[bar_h:, stem_c0 : stem_c0 + stem_w] = True
    else:
        mask[h - bar_h :, :] = True
        mask[: h - bar_h, stem_c0 : stem_c0 + stem_w] = True
    return mask


def _pick_core_mask(h: int, w: int, active_corner: tuple[str, str]) -> np.ndarray:
    kind = random.choice(["L", "L", "L", "T"])
    if kind == "L":
        return _make_l_mask(h, w, active_corner)
    return _make_t_mask(h, w, active_corner)


def _choose_anchor(mask: np.ndarray, active_corner: tuple[str, str]) -> tuple[int, int]:
    h, w = mask.shape
    if active_corner == ("top", "left"):
        prefs = [(h - 1, w - 1), (h - 1, w - 2), (h - 2, w - 1)]
    elif active_corner == ("top", "right"):
        prefs = [(h - 1, 0), (h - 1, 1), (h - 2, 0)]
    elif active_corner == ("bottom", "left"):
        prefs = [(0, w - 1), (1, w - 1), (0, w - 2)]
    else:
        prefs = [(0, 0), (1, 0), (0, 1)]
    for r, c in prefs:
        if 0 <= r < h and 0 <= c < w and bool(mask[r, c]):
            return int(r), int(c)
    occ = np.argwhere(mask)
    rr, cc = occ[random.randrange(len(occ))]
    return int(rr), int(cc)


def _build_nonrect_core(
    h: int, w: int, active_corner: tuple[str, str], fill: int, pattern_colors: list[int]
) -> np.ndarray:
    mask = _pick_core_mask(h, w, active_corner)
    core = np.zeros((h, w), dtype=int)
    core[mask] = int(fill)

    occ = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    protected = set(_silhouette_frontier(core, active_corner[0], 0)) | set(_silhouette_frontier(core, active_corner[1], 0))
    interior_occ = [rc for rc in occ if rc not in protected]
    if not interior_occ:
        interior_occ = occ
    for color in pattern_colors:
        r, c = interior_occ[random.randrange(len(interior_occ))]
        core[r, c] = int(color)

    target_extra = random.randint(max(3, len(pattern_colors)), max(6, int(mask.sum() * 0.22)))
    painted = 0
    while painted < target_extra:
        color = random.choice(pattern_colors)
        kind = random.choice(["dot", "line", "rect"])
        if kind == "dot":
            cells = [occ[random.randrange(len(occ))]]
        elif kind == "line":
            if random.random() < 0.5:
                rr = random.randrange(h)
                c0 = random.randrange(w)
                line_len = random.randint(2, min(5, w))
                cells = [(rr, c0 + dc) for dc in range(line_len) if c0 + dc < w]
            else:
                cc = random.randrange(w)
                r0 = random.randrange(h)
                line_len = random.randint(2, min(5, h))
                cells = [(r0 + dr, cc) for dr in range(line_len) if r0 + dr < h]
        else:
            rh = random.randint(1, min(3, h))
            rw = random.randint(1, min(3, w))
            r0 = random.randrange(h - rh + 1)
            c0 = random.randrange(w - rw + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw)]

        for r, c in cells:
            if not bool(mask[int(r), int(c)]):
                continue
            if (int(r), int(c)) in protected and random.random() < 0.75:
                continue
            if core[int(r), int(c)] == int(fill):
                painted += 1
            core[int(r), int(c)] = int(color)
            if painted >= target_extra:
                break

    anchor_r, anchor_c = _choose_anchor(mask, active_corner)
    core[anchor_r, anchor_c] = int(pattern_colors[0])
    return core


def _guide_targets(core_canvas: np.ndarray, side: str, bg: int) -> list[tuple[int, int]]:
    H, W = core_canvas.shape
    out: list[tuple[int, int]] = []
    if side == "top":
        for c in range(W):
            rows = np.flatnonzero(core_canvas[:, c] != int(bg))
            if rows.size and int(rows[0]) > 0:
                out.append((int(rows[0] - 1), int(c)))
    elif side == "bottom":
        for c in range(W):
            rows = np.flatnonzero(core_canvas[:, c] != int(bg))
            if rows.size and int(rows[-1]) < H - 1:
                out.append((int(rows[-1] + 1), int(c)))
    elif side == "left":
        for r in range(H):
            cols = np.flatnonzero(core_canvas[r, :] != int(bg))
            if cols.size and int(cols[0]) > 0:
                out.append((int(r), int(cols[0] - 1)))
    elif side == "right":
        for r in range(H):
            cols = np.flatnonzero(core_canvas[r, :] != int(bg))
            if cols.size and int(cols[-1]) < W - 1:
                out.append((int(r), int(cols[-1] + 1)))
    else:
        raise ValueError(f"unknown side: {side}")
    return out


def _attach_shape_guides(
    core: np.ndarray,
    guide_sides: tuple[str, str],
    guide_color: int,
    guide_shapes: dict[str, np.ndarray],
) -> np.ndarray:
    out = _blank_core_canvas(core)
    target_map = {side: _guide_targets(out, side, 0) for side in guide_sides}

    for side in guide_sides:
        targets = target_map[side]
        sig = np.array(guide_shapes[side], dtype=bool)
        if len(targets) != len(sig):
            raise ValueError("guide signature length mismatch")
        for use, (r, c) in zip(sig, targets):
            if bool(use):
                out[int(r), int(c)] = int(guide_color)

    b = _bbox_nonbg(out, 0)
    if b is None:
        raise ValueError("empty guided object")
    return crop_bbox(out, b)


def _blank_core_canvas(core: np.ndarray) -> np.ndarray:
    out = np.zeros((core.shape[0] + 2, core.shape[1] + 2), dtype=int)
    paste(out, core, 1, 1, transparent=0)
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
        row = 1
    else:
        row = n - 2
    if vert_side == "left":
        col = 1
    else:
        col = n - 2

    seg_h = grid[row, c0 : c0 + len(horiz_sig)]
    seg_h[np.array(horiz_sig, dtype=bool)] = int(ctrl_h)

    seg_v = grid[r0 : r0 + len(vert_sig), col]
    seg_v[np.array(vert_sig, dtype=bool)] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            rr = 0 if side == "top" else n - 1
            coords = [(rr, c) for c in range(1, n - 1)]
        else:
            cc = 0 if side == "left" else n - 1
            coords = [(r, cc) for r in range(1, n - 1)]
        noise_count = random.randint(int(np.ceil(0.35 * len(coords))), int(np.floor(0.60 * len(coords))))
        for r, c in random.sample(coords, noise_count):
            grid[int(r), int(c)] = 5


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(25, 31)
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

    core_h = random.randint(7, min(14, n - 7))
    core_w = random.randint(7, min(14, n - 7))
    if core_h == core_w:
        core_w = min(core_w + 1, min(15, n - 7))
        if core_h == core_w:
            core_h -= 1
    if core_h < 7 or core_w < 7:
        return None

    core = _build_nonrect_core(core_h, core_w, (horiz_side, vert_side), fill, pattern_colors)
    base_canvas = _blank_core_canvas(core)
    horiz_target_len = len(_guide_targets(base_canvas, horiz_side, 0))
    vert_target_len = len(_guide_targets(base_canvas, vert_side, 0))
    if horiz_target_len < 5 or vert_target_len < 5:
        return None

    for _ in range(100):
        guide_shapes = {
            horiz_side: _random_signature(horiz_target_len),
            vert_side: _random_signature(vert_target_len),
        }
        try:
            aligned_object = _attach_shape_guides(core, (horiz_side, vert_side), guide_color, guide_shapes)
        except ValueError:
            continue

        horiz_obs = _object_side_signature(aligned_object, horiz_side, guide_color, 0)
        vert_obs = _object_side_signature(aligned_object, vert_side, guide_color, 0)
        if horiz_obs is None or vert_obs is None:
            continue
        if int(horiz_obs.sum()) < 3 or int((~horiz_obs).sum()) < 2:
            continue
        if int(vert_obs.sum()) < 3 or int((~vert_obs).sum()) < 2:
            continue
        if np.array_equal(horiz_obs, horiz_obs[::-1]) or np.array_equal(vert_obs, vert_obs[::-1]):
            continue
        if not _nondegenerate_pair(horiz_obs, vert_obs):
            continue

        desired = {horiz_side: horiz_obs, vert_side: vert_obs}
        matches = _all_matches(aligned_object, desired, 0)
        if len(matches) != 1:
            continue
        if not np.array_equal(matches[0][3], core):
            continue
        break
    else:
        return None

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    try:
        _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v, horiz_obs, vert_obs)
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
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D1/0+D2/2")
