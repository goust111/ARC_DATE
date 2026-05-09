# Combo variant of fc7cae8d (index 34)
# combo_spec: D1/0+D2/2+D4/3
# Name: 形状嵌合+非矩形+擦除填充
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Original fc7cae8d:
#    - The frame tells us one horizontal side and one vertical side by color.
#    - The source object is a dense rectangle with two solid 1px guide strips.
#    - We rotate/flip until guide-strip colors line up with the frame colors.
#    - Then we slice away one full row/col and output the dense rectangle.
#    Fused variant:
#    - The frame still marks one horizontal side and one vertical side by color, but the inner adjacent frame line
#      carries a comb-like protrusion mask. The mask is the real orientation clue.
#    - The source core is now a non-rectangular L-shape: a filled rectangle with the corner opposite the control
#      corner removed. This keeps the active control sides visually straight while making the cropped core non-convex.
#    - The object's two active guide borders are not color-coded to the frame colors. Instead, a single guide color is
#      placed only on the positions selected by the protrusion masks. From a side view, "frontier touches bbox edge"
#      means guide present; "frontier is one cell inward" means a hole in that guide pattern.
#    - The solver must find the dihedral transform whose two 1D boundary-touch masks match the frame protrusion masks,
#      strip those guides, crop to the L-core, then erase the dominant fill color to 0 so only sparse motif cells stay.
#    Tiny trace with a top+left target:
#    - Old: top strip is all color 2, left strip is all color 3, so the match channel is purely chromatic.
#    - New: top/left frame sides still use colors 2/3, but the object guide may be color 8 on both sides. The match
#      clue is instead something like top mask [1,0,1,1,0,1] and left mask [1,1,0,1,0,1]. After alignment, deleting
#      those guide cells exposes an L-core; erasing the fill turns that core into a sparse non-convex motif.
# b) Complexity calibration:
#    - I use one non-rectangular family (L-shape) rather than mixing many silhouettes. That keeps the puzzle readable
#      from 2-4 examples while still making the output visibly different from the original dense rectangle.
#    - The guide masks are forced to be non-palindromic and the horizontal/vertical masks are forced to be different
#      enough that exactly one dihedral transform matches, preventing accidental ambiguity.
#    - The notch is on the corner opposite the control corner. This makes the non-rectangular output obvious, but keeps
#      the active matching sides straight enough that the shape-match signal stays 1D and human-readable.
# c) Hidden assumptions broken from the original solver:
#    - `guide_edges()` assumed near-monochrome border rows/cols. False now: each guide side contains deliberate holes.
#    - `return t[top:bottom, left:right]` assumed rectangular cropping solves everything. False now: the cropped core is
#      still non-rectangular inside its bbox, and the final transform must erase the dominant fill.
#    - "match by guide color == frame color" is invalid. The guide color is decoupled from the control colors.
#    - "dense rectangle after crop" is invalid. The generator guarantees positive bbox slack after guide stripping.
# d) Placement legality:
#    - The interior object is a single 4-connected component even though it contains bg holes and a removed corner.
#    - Placement uses `transparent=0`, so the object's internal bg cells do not collide with the frame or with itself.
#    - Frame protrusions live on the inner border line and are kept away from corners so the two control channels do not
#      visually overwrite each other.
# e) Visual observability:
#    - Inputs visibly show a frame with one horizontal and one vertical control side, plus inward protrusions.
#    - The center object visibly has an L-shaped dense region, not a plain rectangle.
#    - Outputs are sparse rather than dense, and their occupied colored cells still spread across the L-shape, so the
#      erase step is obvious and the non-convex source family remains inferable.
# f) Visual cleanliness:
#    - There is only one interior object.
#    - The frame and the center object are separated by at least one bg cell.
#    - Only two guide mechanisms are shown at once: frame protrusions and one shared object-guide color. This avoids
#      unreadable multi-layer overlaps on the same cells.
# g) Color-change inferability:
#    - The output color rule is structural rather than arbitrary: the dominant support color of the cropped core is
#      erased to 0, while smaller motif colors remain. The generator keeps fill dominant and motif colors repeated.
# 9) New assumptions introduced by this fused solver and how the generator guarantees them:
#    - `_frame_signature()` assumes the active inner protrusions form one contiguous span with at least two holes.
#      `_paint_frame()` builds exactly that.
#    - `_object_side_signature()` assumes the guide-present positions on the two active sides can be read as one
#      trimmed 1D mask. `_attach_shape_guides()` creates exactly two such sides and leaves the inactive sides either
#      fully flush (all-True, rejected) or non-matching.
#    - `_strip_guides()` assumes every guide cell on an active side is the frontier cell touching the bbox boundary.
#      The L-core notch is deliberately placed on the opposite corner so the active silhouettes remain straight.
#    - `_erase_fill()` assumes the cropped core still has one dominant non-bg support color. `_build_core()` enforces
#      that by keeping motif density below the fill count.

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
        col, count = Counter(seq_nb).most_common(1)[0]
        if int(count) >= len(seq) - 1:
            cand.append((side, int(col), int(count), int(len(seq))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, col) for side, col, _cnt, _L in cand[:2]]


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
    if int(sig.sum()) < 3 or int(sig.sum()) == len(sig) or int((~sig).sum()) < 2:
        return None
    return sig


def _object_side_signature(arr: np.ndarray, side: str, bg: int) -> np.ndarray | None:
    H, W = arr.shape
    mask: list[bool] = []
    if side == "top":
        for c in range(W):
            rows = np.flatnonzero(arr[:, c] != int(bg))
            if rows.size:
                mask.append(bool(int(rows[0]) == 0))
    elif side == "bottom":
        for c in range(W):
            rows = np.flatnonzero(arr[:, c] != int(bg))
            if rows.size:
                mask.append(bool(int(rows[-1]) == H - 1))
    elif side == "left":
        for r in range(H):
            cols = np.flatnonzero(arr[r, :] != int(bg))
            if cols.size:
                mask.append(bool(int(cols[0]) == 0))
    elif side == "right":
        for r in range(H):
            cols = np.flatnonzero(arr[r, :] != int(bg))
            if cols.size:
                mask.append(bool(int(cols[-1]) == W - 1))
    else:
        raise ValueError(f"unknown side: {side}")

    sig_full = np.array(mask, dtype=bool)
    idx = np.flatnonzero(sig_full)
    if len(idx) < 2:
        return None
    lo, hi = int(idx[0]), int(idx[-1])
    sig = sig_full[lo : hi + 1]
    if int(sig.sum()) < 3 or int(sig.sum()) == len(sig) or int((~sig).sum()) < 2:
        return None
    return sig


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


def _boundary_touch_positions(arr: np.ndarray, side: str, bg: int) -> list[tuple[int, int]]:
    H, W = arr.shape
    out: list[tuple[int, int]] = []
    if side == "top":
        for c in range(W):
            rows = np.flatnonzero(arr[:, c] != int(bg))
            if rows.size and int(rows[0]) == 0:
                out.append((0, int(c)))
    elif side == "bottom":
        for c in range(W):
            rows = np.flatnonzero(arr[:, c] != int(bg))
            if rows.size and int(rows[-1]) == H - 1:
                out.append((H - 1, int(c)))
    elif side == "left":
        for r in range(H):
            cols = np.flatnonzero(arr[r, :] != int(bg))
            if cols.size and int(cols[0]) == 0:
                out.append((int(r), 0))
    elif side == "right":
        for r in range(H):
            cols = np.flatnonzero(arr[r, :] != int(bg))
            if cols.size and int(cols[-1]) == W - 1:
                out.append((int(r), W - 1))
    else:
        raise ValueError(f"unknown side: {side}")
    return out


def _strip_guides(arr: np.ndarray, active_sides: list[str] | tuple[str, ...], bg: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    to_remove: set[tuple[int, int]] = set()
    for side in active_sides:
        to_remove.update(_boundary_touch_positions(arr, side, bg))
    for r, c in to_remove:
        out[int(r), int(c)] = int(bg)
    return out


def _erase_fill(arr: np.ndarray, bg: int) -> np.ndarray:
    vals = [int(x) for x in arr.flatten() if int(x) != int(bg)]
    if not vals:
        return np.array(arr, copy=True)
    fill = int(Counter(vals).most_common(1)[0][0])
    out = np.array(arr, copy=True)
    out[out == fill] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the central object by matching guide-hole silhouettes to frame protrusion masks,
    strip the guides, crop the non-rectangular core, then erase the dominant fill color.
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
        aligned = matches[0][2]
        stripped = _strip_guides(aligned, list(desired_shapes.keys()), bg)
        bbox = _bbox_nonbg(stripped, bg)
        if bbox is not None:
            core = crop_bbox(stripped, bbox)
            return _erase_fill(core, bg)

    return _erase_fill(obj, bg)


def _opposite_corner(corner: tuple[str, str]) -> tuple[str, str]:
    horiz_side, vert_side = corner
    return ("bottom" if horiz_side == "top" else "top", "right" if vert_side == "left" else "left")


def _build_l_mask(h: int, w: int, active_corner: tuple[str, str]) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    notch_corner = _opposite_corner(active_corner)
    notch_h = random.randint(2, max(2, h // 2))
    notch_w = random.randint(2, max(2, w // 2))
    notch_h = min(notch_h, h - 1)
    notch_w = min(notch_w, w - 1)

    horiz_side, vert_side = notch_corner
    if (horiz_side, vert_side) == ("top", "left"):
        mask[:notch_h, :notch_w] = False
    elif (horiz_side, vert_side) == ("top", "right"):
        mask[:notch_h, w - notch_w :] = False
    elif (horiz_side, vert_side) == ("bottom", "left"):
        mask[h - notch_h :, :notch_w] = False
    elif (horiz_side, vert_side) == ("bottom", "right"):
        mask[h - notch_h :, w - notch_w :] = False
    else:
        raise ValueError("invalid corner")
    return mask


def _outline_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    H, W = mask.shape
    raw: list[tuple[int, int]] = []
    for c in range(W):
        rows = np.flatnonzero(mask[:, c])
        if rows.size:
            raw.append((int(rows[0]), int(c)))
            raw.append((int(rows[-1]), int(c)))
    for r in range(H):
        cols = np.flatnonzero(mask[r, :])
        if cols.size:
            raw.append((int(r), int(cols[0])))
            raw.append((int(r), int(cols[-1])))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for cell in raw:
        if cell in seen:
            continue
        seen.add(cell)
        out.append(cell)
    return out


def _build_core(h: int, w: int, active_corner: tuple[str, str], fill: int, pattern_colors: list[int]) -> np.ndarray:
    mask = _build_l_mask(h, w, active_corner)
    core = np.zeros((h, w), dtype=int)
    core[mask] = int(fill)

    area = int(mask.sum())
    min_nonfill = max(len(pattern_colors) + 6, int(np.ceil(area * 0.22)))
    max_nonfill = max(min_nonfill, int(np.floor(area * 0.36)))
    target_nonfill = random.randint(min_nonfill, max_nonfill)

    outline = _outline_cells(mask)
    random.shuffle(outline)
    forced_n = min(len(outline), max(len(pattern_colors) + 4, min(target_nonfill - 1, (2 * target_nonfill) // 3)))
    forced = outline[:forced_n]
    for i, (r, c) in enumerate(forced):
        core[int(r), int(c)] = int(pattern_colors[i % len(pattern_colors)])

    occ = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    painted = int(np.count_nonzero((core != int(fill)) & mask))
    attempts = 0
    while painted < target_nonfill and attempts < area * 12:
        attempts += 1
        color = int(random.choice(pattern_colors))
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
            if not mask[int(r), int(c)]:
                continue
            if core[int(r), int(c)] == int(fill):
                painted += 1
            core[int(r), int(c)] = color
            if painted >= target_nonfill:
                break

    if painted < target_nonfill:
        fill_cells = [(r, c) for r, c in occ if core[int(r), int(c)] == int(fill)]
        for r, c in random.sample(fill_cells, min(target_nonfill - painted, len(fill_cells))):
            core[int(r), int(c)] = int(random.choice(pattern_colors))

    return core


def _guide_targets(core_canvas: np.ndarray, side: str, bg: int) -> list[tuple[int, int]]:
    H, W = core_canvas.shape
    out: list[tuple[int, int]] = []
    if side == "top":
        for c in range(W):
            rows = np.flatnonzero(core_canvas[:, c] != int(bg))
            if rows.size and int(rows[0]) > 0:
                out.append((int(rows[0]) - 1, int(c)))
    elif side == "bottom":
        for c in range(W):
            rows = np.flatnonzero(core_canvas[:, c] != int(bg))
            if rows.size and int(rows[-1]) < H - 1:
                out.append((int(rows[-1]) + 1, int(c)))
    elif side == "left":
        for r in range(H):
            cols = np.flatnonzero(core_canvas[r, :] != int(bg))
            if cols.size and int(cols[0]) > 0:
                out.append((int(r), int(cols[0]) - 1))
    elif side == "right":
        for r in range(H):
            cols = np.flatnonzero(core_canvas[r, :] != int(bg))
            if cols.size and int(cols[-1]) < W - 1:
                out.append((int(r), int(cols[-1]) + 1))
    else:
        raise ValueError(f"unknown side: {side}")
    return out


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
    paste(out, core, r0, c0, transparent=0)

    claims: dict[tuple[int, int], int] = {}
    for side in guide_sides:
        targets = _guide_targets(out, side, 0)
        sig = np.array(guide_shapes[side], dtype=bool)
        if len(targets) != len(sig):
            raise ValueError(f"guide signature length mismatch on {side}: {len(targets)} vs {len(sig)}")
        for pos, keep in zip(targets, sig):
            if bool(keep):
                claims[(int(pos[0]), int(pos[1]))] = int(guide_color)

    for (r, c), color in claims.items():
        out[int(r), int(c)] = int(color)
    return out


def _random_signature(length: int) -> np.ndarray:
    while True:
        sig = np.ones(int(length), dtype=bool)
        interior = list(range(1, length - 1))
        max_holes = min(len(interior), max(2, length // 3))
        if max_holes < 2:
            continue
        hole_n = random.randint(2, max_holes)
        for idx in random.sample(interior, hole_n):
            sig[int(idx)] = False
        if int(sig.sum()) < 3 or int((~sig).sum()) < 2:
            continue
        if np.array_equal(sig, sig[::-1]):
            continue
        return sig


def _nondegenerate_pair(sig_a: np.ndarray, sig_b: np.ndarray) -> bool:
    if len(sig_a) == len(sig_b):
        if np.array_equal(sig_a, sig_b):
            return False
        if np.array_equal(sig_a, sig_b[::-1]):
            return False
    return True


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
        raise ValueError("frame too small for protrusion signatures")

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
            grid[int(r), int(c)] = 5


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

    core_h = random.randint(7, min(13, n - 8))
    core_w = random.randint(7, min(13, n - 8))
    if core_h == core_w:
        core_w = min(core_w + 1, min(14, n - 8))
        if core_h == core_w:
            core_h -= 1
    if core_h < 7 or core_w < 7:
        return None

    core = _build_core(core_h, core_w, (horiz_side, vert_side), fill, pattern_colors)
    expected = _erase_fill(core, 0)

    for _ in range(80):
        horiz_sig = _random_signature(core_w)
        vert_sig = _random_signature(core_h)
        if not _nondegenerate_pair(horiz_sig, vert_sig):
            continue
        guide_shapes = {horiz_side: horiz_sig, vert_side: vert_sig}
        try:
            aligned_object = _attach_shape_guides(core, (horiz_side, vert_side), guide_color, guide_shapes)
        except ValueError:
            continue
        desired = {horiz_side: horiz_sig, vert_side: vert_sig}
        matches = _matching_transforms(aligned_object, desired, 0)
        if len(matches) != 1:
            continue
        stripped = _strip_guides(matches[0][2], (horiz_side, vert_side), 0)
        bbox = _bbox_nonbg(stripped, 0)
        if bbox is None:
            continue
        if not np.array_equal(crop_bbox(stripped, bbox), core):
            continue
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

    if not np.array_equal(main(grid), expected):
        return None

    solved_core = _strip_guides(matches[0][2], (horiz_side, vert_side), 0)
    bbox = _bbox_nonbg(solved_core, 0)
    if bbox is None:
        return None
    cropped = crop_bbox(solved_core, bbox)
    if int(cropped.size - np.count_nonzero(cropped)) <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 34")
