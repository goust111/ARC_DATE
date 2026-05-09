# Variant of fc7cae8d
# Dimension: D2 — 源对象结构 (src)
# Change: 单个填充矩形，两相邻边有1px纯色引导条带，引导角（两引导边交汇处）为背景色... → 非矩形源对象

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


# Design analysis for the D2 variant:
# 1. Concrete behavior:
#    The old object was a filled rectangle with two full guide rows/cols, so after alignment
#    the solver could simply slice away one full row/col. Here the output core is an L-shape:
#    a rectangle with one corner notch removed. The two guide colors no longer form complete
#    outer rows/cols; instead they follow the visible silhouette from the chosen frame sides.
#    Example: if the controls are top+left, then for each column the top guide sits one cell
#    above the first occupied core cell seen from the top, and for each row the left guide
#    sits one cell left of the first occupied core cell seen from the left. Around the notch,
#    these guide segments bend inward, so the solver must remove guide cells by directional
#    silhouette scanning rather than by fixed rectangular cropping.
# 2. Complexity calibration:
#    A corner-notch L-shape is visibly non-rectangular in every sample because the output bbox
#    contains a background notch. Internal colors are still added so different rotations/flips
#    do not all look interchangeable to a human. The generator also rejects ambiguous layouts.
# 3. Old implicit assumptions that break:
#    - guide_edges(...) assumed each relevant border of the cropped object was almost entirely
#      one guide color. That fails once the guide follows an irregular boundary instead of a
#      full row/col.
#    - return t[top:bottom, left:right] assumed deleting a whole row/col exactly removes the
#      guides. That is false here because the guide occupies only the directional silhouette.
#    - fill = most common value inside the object used to detect guide rows. With a notched
#      shape, background is present inside the output bbox, so the fill-based edge heuristic
#      is no longer reliable.
# 4. Placement legality:
#    The object is still a single 4-connected component after adding guides. It is placed away
#    from the outer frame with transparent=0, so the notch background remains empty and the
#    interior object never touches the frame.
# 5. Visual observability:
#    The output always has positive bbox slack: bbox_area - occupied_area > 0. This metric is
#    zero for the old solid-rectangle generator but strictly positive here, so the D2 change is
#    externally visible instead of being a hidden implementation detail.


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
    for side, v in sides.items():
        vals = list(map(int, v))
        vals_nb = [x for x in vals if x != int(bg)]
        if not vals_nb:
            continue
        col, count = Counter(vals_nb).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
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


def _silhouette_positions(arr: np.ndarray, side: str, bg: int) -> list[tuple[int, int]]:
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


def _score_side(arr: np.ndarray, side: str, color: int, bg: int) -> tuple[float, list[tuple[int, int]], int]:
    frontier = _silhouette_positions(arr, side, bg)
    matches = [(r, c) for r, c in frontier if int(arr[r, c]) == int(color)]
    score = len(matches) / max(1, len(frontier))
    return float(score), matches, len(frontier)


def _candidate_after_alignment(
    arr: np.ndarray, desired: dict[str, int], bg: int
) -> tuple[np.ndarray, float, int] | None:
    to_remove: set[tuple[int, int]] = set()
    total_score = 0.0
    for side, color in desired.items():
        score, matches, visible = _score_side(arr, side, color, bg)
        if visible < 4 or len(matches) < 3 or score < 0.45:
            return None
        total_score += score
        to_remove.update(matches)

    out = arr.copy()
    for r, c in to_remove:
        out[int(r), int(c)] = int(bg)

    b = _bbox_nonbg(out, bg)
    if b is None:
        return None
    return crop_bbox(out, b), float(total_score), int(len(to_remove))


def _aligned_candidates(
    obj: np.ndarray, desired: dict[str, int], bg: int
) -> list[tuple[float, int, int, bool, np.ndarray]]:
    candidates: list[tuple[float, int, int, bool, np.ndarray]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            cand = _candidate_after_alignment(t, desired, bg)
            if cand is None:
                continue
            out, score, removed = cand
            candidates.append((float(score), int(removed), int(k), bool(flip), out))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a non-rectangular source object to two colored frame controls.

    The object is now an irregular L-shape whose guide colors follow the object's
    directional silhouette instead of occupying full rows/columns. The correct
    transform is the dihedral transform whose top/bottom and left/right silhouettes
    expose the matching guide colors. Those guide cells are then peeled off and the
    remaining non-background core is cropped.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = {side: col for side, col in _detect_controls_frame(g, bg)}
    obj = _extract_center_object(g, bg)

    candidates = _aligned_candidates(obj, desired, bg)
    if candidates:
        return candidates[0][-1]
    return obj


def _build_nonrect_core(
    h: int, w: int, notch_corner: tuple[str, str], fill: int, pattern_colors: list[int]
) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    notch_h = random.randint(2, max(2, h // 2))
    notch_w = random.randint(2, max(2, w // 2))
    if h == w:
        w += 1
        mask = np.ones((h, w), dtype=bool)
    if notch_h == h - 1:
        notch_h -= 1
    if notch_w == w - 1:
        notch_w -= 1

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
        raise ValueError("guide sides must be an adjacent corner")

    core = np.zeros((h, w), dtype=int)
    core[mask] = int(fill)

    occ = np.argwhere(mask)
    random.shuffle(occ.tolist())
    for color in pattern_colors:
        r, c = map(int, occ[random.randrange(len(occ))])
        core[r, c] = int(color)

    target_extra = random.randint(max(3, len(pattern_colors)), max(5, int(mask.sum() * 0.22)))
    painted = 0
    while painted < target_extra:
        color = random.choice(pattern_colors)
        kind = random.choice(["dot", "line", "rect"])
        if kind == "dot":
            cells = [tuple(map(int, occ[random.randrange(len(occ))]))]
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
            core[int(r), int(c)] = int(color)
            if painted >= target_extra:
                break

    # Force one asymmetric mark near the corner opposite the notch so different
    # transforms are not visually interchangeable.
    if (horiz_side, vert_side) == ("top", "left"):
        anchor = (h - 1, w - 1)
    elif (horiz_side, vert_side) == ("top", "right"):
        anchor = (h - 1, 0)
    elif (horiz_side, vert_side) == ("bottom", "left"):
        anchor = (0, w - 1)
    else:
        anchor = (0, 0)
    if mask[anchor]:
        core[anchor] = int(pattern_colors[0])

    return core


def _guide_targets(core_canvas: np.ndarray, side: str, bg: int) -> list[tuple[int, int]]:
    H, W = core_canvas.shape
    out: list[tuple[int, int]] = []
    if side == "top":
        for c in range(W):
            rows = np.flatnonzero(core_canvas[:, c] != int(bg))
            if rows.size and rows[0] > 0:
                out.append((int(rows[0] - 1), int(c)))
    elif side == "bottom":
        for c in range(W):
            rows = np.flatnonzero(core_canvas[:, c] != int(bg))
            if rows.size and rows[-1] < H - 1:
                out.append((int(rows[-1] + 1), int(c)))
    elif side == "left":
        for r in range(H):
            cols = np.flatnonzero(core_canvas[r, :] != int(bg))
            if cols.size and cols[0] > 0:
                out.append((int(r), int(cols[0] - 1)))
    elif side == "right":
        for r in range(H):
            cols = np.flatnonzero(core_canvas[r, :] != int(bg))
            if cols.size and cols[-1] < W - 1:
                out.append((int(r), int(cols[-1] + 1)))
    else:
        raise ValueError(f"unknown side: {side}")
    return out


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int]) -> np.ndarray:
    bg = 0
    h, w = core.shape
    out = np.zeros((h + 2, w + 2), dtype=int)
    paste(out, core, 1, 1, transparent=0)

    claims: dict[tuple[int, int], list[int]] = {}
    for side in guide_sides:
        for pos in _guide_targets(out, side, bg):
            claims.setdefault(pos, []).append(int(guide_colors[side]))

    for (r, c), colors in claims.items():
        if len(colors) == 1:
            out[int(r), int(c)] = int(colors[0])
        else:
            out[int(r), int(c)] = int(bg)
    return out


def _paint_frame(grid: np.ndarray, horiz_side: str, vert_side: str, ctrl_h: int, ctrl_v: int) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        noise_count = random.randint(int(np.ceil(0.35 * len(coords))), int(np.floor(0.60 * len(coords))))
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
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    core_h = random.randint(7, min(13, n - 6))
    core_w = random.randint(7, min(13, n - 6))
    if core_h == core_w:
        core_w = min(core_w + 1, min(14, n - 6))
        if core_h == core_w:
            core_h -= 1
    if core_h < 7 or core_w < 7:
        return None

    core = _build_nonrect_core(core_h, core_w, (horiz_side, vert_side), fill, pattern_colors)
    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)

    desired = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_candidates = _aligned_candidates(aligned_object, desired, 0)
    if len(aligned_candidates) != 1:
        return None
    if not np.array_equal(aligned_candidates[0][-1], core):
        return None

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D2 variant")
