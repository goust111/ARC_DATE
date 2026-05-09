# Combo variant of fc7cae8d (index 15)
# combo_spec: D2/2+D4/3
# Name: 非矩形源+擦除填充

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.insert(0, ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior trace:
#    Original fc7cae8d used a filled rectangle whose two full outer guide bands matched two
#    colored frame controls. After a D4 alignment, the solver removed one whole row and one
#    whole column and returned the dense rectangle. Here the source core is an L-shape: think
#    of a 7x8 rectangle with its top-left 3x2 corner removed. The two guide colors still
#    correspond to one horizontal and one vertical frame side, but they now trace the visible
#    silhouette of that L-shape rather than spanning an entire row/column. Example: if the
#    frame controls are top=2 and left=7, then the top guide only appears above columns that
#    actually contain core cells, and the left guide only appears left of rows that contain
#    core cells. Around the missing corner they bend inward. The fused output pipeline is:
#    align by dihedral transform -> peel those silhouette guide cells -> keep the cropped
#    L-shaped core canvas -> erase the dominant non-background fill color to 0. The result is
#    a sparse pattern on the same notched canvas instead of the old dense rectangle.
# b) Complexity calibration:
#    The generator deliberately stays with one irregular family: a single corner-notch L-shape.
#    That is enough to visibly break the rectangle assumption without introducing too many
#    shape classes. Ambiguity is reduced by forcing asymmetric internal motifs, fill-majority,
#    and unique D4 alignment acceptance. Pattern density stays around 18%-32% of occupied core
#    cells so erasure is dramatic but the output is still legible.
# c) Implicit assumptions that break in the original main():
#    - `guide_edges(...)` assumed a relevant border was almost fully one color. That is false
#      once guides follow a bent silhouette.
#    - Fixed slicing away one row/column assumed rectangular guides. Now guide cells must be
#      detected by directional frontier scanning and removed cell-by-cell.
#    - Using the global mode as fill is unsafe because the cropped core now contains internal
#      background from the missing corner. The combo solver therefore erases the dominant
#      non-background color instead.
#    - A plain rectangle made several transforms equivalent. The generator now rejects cases
#      where more than one transform satisfies the guide match strongly.
# d) Placement legality:
#    Only one guided object is created. It is transformed before placement and inserted with
#    `random_free_location(..., transparent=0, border=1)`, so it stays strictly inside the
#    frame, never overlaps the border controls, and keeps the notch/background cells empty.
# e) Visual observability:
#    Across a few samples, a human sees three stable clues: the source object is visibly not a
#    rectangle, the guide colors still determine orientation via the frame, and the output is
#    the same aligned canvas with the large support color erased away. Those clues repeat with
#    different colors and orientations, which makes the fusion rule inferable from examples.
# f) Visual cleanliness:
#    The object remains a single component, the frame noise stays on the two unused border
#    sides, and the guides occupy a one-cell halo outside the object silhouette. Pattern seeds
#    are concentrated on shape frontiers plus a few interior marks, so the erased output looks
#    sparse but organized rather than like random confetti.
# g) Color inferability:
#    Colors have a small fixed role set: two control colors, one dominant fill, and 2-4 motif
#    colors. The erased color is not arbitrary; it is the dominant support color inside the
#    cropped core. Because that support color is consistently the most common non-background
#    color before erasure, a human can infer which color disappears.
# New solver assumptions introduced by this combo and guaranteed by the generator:
#    - Each desired guide side exposes at least several silhouette cells of the matching color.
#    - The dominant non-background color inside the peeled core is the support fill.
#    - Exactly one transformed object passes the silhouette-guide tests with a strong score.


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


def _dominant_nonbg(arr: np.ndarray, bg: int) -> int | None:
    vals = [int(x) for x in arr.flatten() if int(x) != int(bg)]
    if not vals:
        return None
    return int(Counter(vals).most_common(1)[0][0])


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
    return [(side, col) for side, col, _count, _length in cand[:2]]


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


def _candidate_after_alignment(
    arr: np.ndarray, desired: dict[str, int], bg: int, min_score: float = 0.5
) -> tuple[np.ndarray, float, int] | None:
    to_remove: set[tuple[int, int]] = set()
    total_score = 0.0
    for side, color in desired.items():
        frontier = _silhouette_positions(arr, side, bg)
        matches = [(r, c) for r, c in frontier if int(arr[r, c]) == int(color)]
        visible = len(frontier)
        score = len(matches) / max(1, visible)
        if visible < 5 or len(matches) < 3 or score < float(min_score):
            return None
        total_score += float(score)
        to_remove.update(matches)

    peeled = np.array(arr, copy=True)
    for r, c in to_remove:
        peeled[int(r), int(c)] = int(bg)

    b = _bbox_nonbg(peeled, bg)
    if b is None:
        return None
    return crop_bbox(peeled, b), float(total_score), int(len(to_remove))


def _aligned_candidates(
    obj: np.ndarray, desired: dict[str, int], bg: int
) -> list[tuple[float, int, int, bool, np.ndarray]]:
    candidates: list[tuple[float, int, int, bool, np.ndarray]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            cand = _candidate_after_alignment(transformed, desired, bg)
            if cand is None:
                continue
            core, score, removed = cand
            candidates.append((float(score), int(removed), int(k), bool(flip), core))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates


def _erase_fill(arr: np.ndarray, bg: int = 0) -> np.ndarray:
    fill = _dominant_nonbg(arr, bg)
    if fill is None:
        return np.array(arr, copy=True)
    out = np.array(arr, copy=True)
    out[out == int(fill)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a guided non-rectangular object to the frame controls, peel off the bent guide
    silhouette, then erase the dominant non-background support color from the cropped core.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = {side: col for side, col in _detect_controls_frame(g, bg)}
    obj = _extract_center_object(g, bg)

    candidates = _aligned_candidates(obj, desired, bg)
    if candidates:
        return _erase_fill(candidates[0][-1], bg)

    b = _bbox_nonbg(obj, bg)
    fallback = crop_bbox(obj, b) if b is not None else obj
    return _erase_fill(fallback, bg)


def _make_l_mask(h: int, w: int, notch_corner: tuple[str, str]) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    notch_h = min(h - 2, random.randint(2, max(2, h // 2)))
    notch_w = min(w - 2, random.randint(2, max(2, w // 2)))
    if notch_h == notch_w and notch_w < w - 2:
        notch_w += 1

    if notch_corner == ("top", "left"):
        mask[:notch_h, :notch_w] = False
    elif notch_corner == ("top", "right"):
        mask[:notch_h, w - notch_w :] = False
    elif notch_corner == ("bottom", "left"):
        mask[h - notch_h :, :notch_w] = False
    elif notch_corner == ("bottom", "right"):
        mask[h - notch_h :, w - notch_w :] = False
    else:
        raise ValueError("guide sides must form a corner")
    return mask


def _mask_frontier(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    H, W = mask.shape
    out: list[tuple[int, int]] = []
    if side == "top":
        for c in range(W):
            rows = np.flatnonzero(mask[:, c])
            if rows.size:
                out.append((int(rows[0]), int(c)))
    elif side == "bottom":
        for c in range(W):
            rows = np.flatnonzero(mask[:, c])
            if rows.size:
                out.append((int(rows[-1]), int(c)))
    elif side == "left":
        for r in range(H):
            cols = np.flatnonzero(mask[r, :])
            if cols.size:
                out.append((int(r), int(cols[0])))
    elif side == "right":
        for r in range(H):
            cols = np.flatnonzero(mask[r, :])
            if cols.size:
                out.append((int(r), int(cols[-1])))
    else:
        raise ValueError(f"unknown side: {side}")
    return out


def _choose_spread(coords: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if not coords or count <= 0:
        return []
    if len(coords) <= count:
        return list(coords)
    if count == 1:
        return [coords[len(coords) // 2]]
    idxs = np.linspace(0, len(coords) - 1, num=count)
    return [coords[int(round(i))] for i in idxs]


def _build_nonrect_core(
    h: int, w: int, notch_corner: tuple[str, str], fill: int, pattern_colors: list[int]
) -> np.ndarray:
    mask = _make_l_mask(h, w, notch_corner)
    core = np.zeros((h, w), dtype=int)
    core[mask] = int(fill)
    area = int(mask.sum())

    frontiers = {side: _mask_frontier(mask, side) for side in ("top", "bottom", "left", "right")}
    painted_positions: set[tuple[int, int]] = set()

    color_cycle = list(pattern_colors)
    for idx, side in enumerate(("top", "bottom", "left", "right")):
        coords = _choose_spread(frontiers[side], 2 if len(frontiers[side]) >= 6 else 1)
        for r, c in coords:
            core[int(r), int(c)] = int(color_cycle[idx % len(color_cycle)])
            painted_positions.add((int(r), int(c)))

    if notch_corner == ("top", "left"):
        anchors = [(h - 1, w - 1), (h - 2, w - 1), (h - 1, w - 2)]
    elif notch_corner == ("top", "right"):
        anchors = [(h - 1, 0), (h - 2, 0), (h - 1, 1)]
    elif notch_corner == ("bottom", "left"):
        anchors = [(0, w - 1), (0, w - 2), (1, w - 1)]
    else:
        anchors = [(0, 0), (0, 1), (1, 0)]
    for idx, pos in enumerate(anchors):
        r, c = pos
        if mask[int(r), int(c)]:
            core[int(r), int(c)] = int(pattern_colors[idx % len(pattern_colors)])
            painted_positions.add((int(r), int(c)))

    target_nonfill = random.randint(max(7, int(np.ceil(area * 0.18))), max(9, int(np.floor(area * 0.32))))
    occ = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    boundary = [(r, c) for r, c in occ if any(
        nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    )]

    attempts = 0
    while len(painted_positions) < target_nonfill and attempts < area * 16:
        attempts += 1
        color = int(random.choice(pattern_colors))
        kind = random.choice(["frontier", "dot", "block"])
        cells: list[tuple[int, int]]
        if kind == "frontier":
            side = random.choice(["top", "bottom", "left", "right"])
            seq = frontiers[side]
            if not seq:
                continue
            seg_len = random.randint(2, min(4, len(seq)))
            start = random.randrange(len(seq) - seg_len + 1)
            cells = seq[start : start + seg_len]
        elif kind == "dot":
            pool = boundary if random.random() < 0.65 and boundary else occ
            cells = [random.choice(pool)]
        else:
            rh = random.randint(1, min(3, h))
            rw = random.randint(1, min(3, w))
            r0 = random.randrange(h - rh + 1)
            c0 = random.randrange(w - rw + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw) if mask[r0 + dr, c0 + dc]]

        for r, c in cells:
            if len(painted_positions) >= target_nonfill:
                break
            if not mask[int(r), int(c)]:
                continue
            core[int(r), int(c)] = int(color)
            painted_positions.add((int(r), int(c)))

    for side, coords in frontiers.items():
        if not any(core[int(r), int(c)] != int(fill) for r, c in coords):
            r, c = random.choice(coords)
            core[int(r), int(c)] = int(random.choice(pattern_colors))
            painted_positions.add((int(r), int(c)))

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
    out = np.zeros((core.shape[0] + 2, core.shape[1] + 2), dtype=int)
    paste(out, core, 1, 1, transparent=0)

    claims: dict[tuple[int, int], list[int]] = {}
    for side in guide_sides:
        for pos in _guide_targets(out, side, 0):
            claims.setdefault(pos, []).append(int(guide_colors[side]))

    for (r, c), colors in claims.items():
        out[int(r), int(c)] = int(colors[0]) if len(colors) == 1 else 0
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
    n = random.randint(24, 31)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    motif_palette = [c for c in remaining if c != fill]
    pattern_colors = random.sample(motif_palette, random.randint(2, min(4, len(motif_palette))))

    max_dim = min(14, n - 7)
    if max_dim < 8:
        return None
    core_h = random.randint(8, max_dim)
    core_w = random.randint(8, max_dim)
    if core_h == core_w:
        if core_w < max_dim:
            core_w += 1
        else:
            core_h -= 1
    if core_h < 8 or core_w < 8:
        return None

    core = _build_nonrect_core(core_h, core_w, (horiz_side, vert_side), fill, pattern_colors)
    expected = _erase_fill(core, 0)
    if np.array_equal(core, expected):
        return None

    dense_nonbg = int(np.count_nonzero(core))
    sparse_nonbg = int(np.count_nonzero(expected))
    bbox_slack = int(core.size - dense_nonbg)
    if bbox_slack <= 0 or sparse_nonbg < 6:
        return None
    erase_ratio = (dense_nonbg - sparse_nonbg) / max(1, dense_nonbg)
    if erase_ratio < 0.45:
        return None

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)
    desired = {horiz_side: ctrl_h, vert_side: ctrl_v}
    candidates = _aligned_candidates(aligned_object, desired, 0)
    if len(candidates) != 1:
        return None
    if not np.array_equal(candidates[0][-1], core):
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

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D2/2+D4/3")
