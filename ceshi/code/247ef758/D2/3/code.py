# Variant of 247ef758
# Dimension: D2 — 源对象：精灵类型与属性 (src)
# Change: 左侧窄条中2-3个可移动精灵（8连通单色色块，3-9像素，外接矩形≤5x5）加0-1个干扰精灵，颜色各不相同... → 对角标记像素对作为精灵
#
# Design analysis
# ===============
# Reference-task note:
# - The prompt asks to read the source task `9385bd28` first. In this workspace, a
#   standalone solver file for `9385bd28` is not present, so the usable baseline is
#   its official evaluation I/O at:
#   `/home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/9385bd28.json`
# - Those examples show the relevant mechanism: two same-color pixels placed at
#   opposite corners of a rectangle act as diagonal markers, and the output fills the
#   entire bounded rectangle. This variant ports exactly that source-object idea into
#   247ef758 while leaving the frame-marker targeting mechanism intact.
#
# variant_spec:
#   unchanged_layout:
#     grid: 10-15 rows × 14-20 cols
#     bg: 0
#     structure: left source strip | one full-height divider column | right frame
#     frame:
#       spans full grid height
#       interior: bg rectangle, 8-13 rows × 8-13 cols
#       top==bottom and left==right
#       mover colors still appear as matching row/col edge markers
#     ordering:
#       movers are still applied bottom-to-top by their source vertical position
#       later (visually higher) sources cover earlier (lower) ones
#   changed_src_objects:
#     movers:
#       count: 2-3
#       each mover color appears in the left strip as exactly two pixels
#       the two pixels are opposite corners of a bbox-sized rectangle
#       bbox height/width: odd sizes, 3 or 5 when space allows, clipped by strip width
#       diagonal orientation: either TL↘BR or TR↙BL
#     distractors:
#       count: 0-1
#       same diagonal-pair object type as movers
#       colors absent from frame markers, so they stay unchanged
#   transformation:
#     detect each left-strip color as a diagonal pair instead of an 8-connected blob
#     row_targets[color] still come from matching left/right edge markers
#     col_targets[color] still come from matching top/bottom edge markers
#     for each target in row_targets × col_targets:
#       center the mover's bbox on the target
#       fill the whole translated rectangle solid with that mover color
#     clip filling to frame interior, same as original clipping semantics
#
# behavior_trace:
#   original:
#     left strip contains a blob sprite. `main()` labels 8-connected components by color,
#     records each blob mask, erases it, then translates the mask onto every target.
#   variant:
#     left strip contains only two same-color pixels, often disconnected even under
#     8-connectivity. Treating them as blobs would split them into two one-pixel sprites,
#     which is wrong. Instead, `main()` must gather all left-strip pixels of a color,
#     require exactly two, verify they are opposite bbox corners, erase those two pixels,
#     and stamp the full bbox rectangle at each target.
#   concrete_example:
#     source pair color 4 at (0,0) and (2,2) in the strip defines a 3×3 rectangle.
#     If row mark 4 is at r=5 and col mark 4 is at c=9, the old solver would place one
#     or two single pixels near (5,9); the variant must place a filled 3×3 block centered
#     on (5,9). This is the observable D2 change.
#
# complexity_calibration:
# - Using only two pixels per source keeps the geometry rule legible.
# - Odd bbox sizes avoid half-cell centers, so the “centered on marker intersection”
#   relation remains visually exact.
# - Colors remain distinct and the frame-marker logic is unchanged, so the solver still
#   has repeated evidence for color→target matching while the source-object type changes.
#
# implicit_assumption_audit:
# - Original `sprites = connected_components(...)` assumes each mover is one connected
#   mono-color blob. This is invalid for diagonal pairs because the two pixels are
#   typically disconnected. Replaced with per-color left-strip pixel collection.
# - Original erase loop assumes a stored mask of all source pixels. Replaced with erasing
#   exactly the two detected source pixels.
# - Original stamping loop copies only source-mask pixels. Replaced with filling every
#   cell in the mover bbox after translation.
# - Original target extraction by frame-edge color equality still holds unchanged because
#   D2 only changes the source object representation, not the row/column marker system.
#
# placement_legality:
# - Generator stacks source objects by their bbox bands in the left strip, so each pair
#   has a visually coherent local rectangle and does not overlap another source.
# - Each mover bbox is chosen before marker placement; row/col markers are then sampled
#   only from target centers that keep the translated rectangle inside the frame interior.
# - Distinct colors for divider, frame base, movers, and distractors prevent ambiguity.
#
# observability_and_cleanliness:
# - Inputs show sparse corner pairs in the left strip, outputs show solid rectangles in
#   the frame. This is visible in a single example and reinforced across 2-4 examples.
# - The frame edges still encode which color goes to which target rows/cols, so humans
#   can separate “where to place” from “what shape to place”.
# - Overlap is allowed but limited by unique row/col markers and mostly 3×3 / 5×3 / 5×5
#   rectangles, keeping the frame readable while preserving the original cover-order rule.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _extract_diagonal_pairs(
    g: np.ndarray,
    *,
    bg: int,
    fc0: int,
    divider_mask: np.ndarray,
) -> list[dict[str, int | tuple[tuple[int, int], tuple[int, int]]]]:
    """Parse left-strip same-color diagonal pixel pairs as source objects."""
    H, W = g.shape
    left_cols = np.arange(W)[None, :] < int(fc0)
    pairs: list[dict[str, int | tuple[tuple[int, int], tuple[int, int]]]] = []

    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        pts = np.argwhere((g == int(col)) & left_cols & (~divider_mask))
        if int(len(pts)) != 2:
            continue

        (r_a, c_a), (r_b, c_b) = [tuple(map(int, pt)) for pt in pts]
        r0, r1 = sorted((int(r_a), int(r_b)))
        c0, c1 = sorted((int(c_a), int(c_b)))
        if r0 == r1 or c0 == c1:
            continue

        corner_set = {(int(r_a), int(c_a)), (int(r_b), int(c_b))}
        if corner_set not in ({(r0, c0), (r1, c1)}, {(r0, c1), (r1, c0)}):
            continue

        pairs.append(
            {
                "color": int(col),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "pixels": ((int(r_a), int(c_a)), (int(r_b), int(c_b))),
            }
        )

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move diagonal-pair sources from the left strip into the framed area at marker-defined
    targets, filling each source pair's bounding rectangle instead of copying sparse pixels.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert not (inner_bbox is None)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    pairs = _extract_diagonal_pairs(g, bg=int(bg), fc0=int(fc0), divider_mask=divider_mask)
    move_colors = {int(p["color"]) for p in pairs}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [p for p in pairs if int(p["color"]) in targets]

    for p in movers:
        for rr, cc in p["pixels"]:
            out[int(rr), int(cc)] = int(bg)

    movers.sort(key=lambda p: (-((int(p["r0"]) + int(p["r1"])) / 2.0), int(p["c0"])))

    for p in movers:
        col = int(p["color"])
        r0, c0, r1, c1 = map(int, (p["r0"], p["c0"], p["r1"], p["c1"]))
        cr = (int(r0) + int(r1)) // 2
        cc = (int(c0) + int(c1)) // 2

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for rr in range(int(r0), int(r1) + 1):
                for cc2 in range(int(c0), int(c1) + 1):
                    nr = int(rr + dr)
                    nc = int(cc2 + dc)
                    if int(ir0) <= nr <= int(ir1) and int(ic0) <= nc <= int(ic1):
                        out[nr, nc] = int(col)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_diagonal_pair(color: int, h: int, w: int) -> np.ndarray:
    """Return an h×w source object containing exactly two same-color diagonal corner pixels."""
    pair = np.zeros((int(h), int(w)), dtype=int)
    if random.choice([False, True]):
        pair[0, 0] = int(color)
        pair[int(h) - 1, int(w) - 1] = int(color)
    else:
        pair[0, int(w) - 1] = int(color)
        pair[int(h) - 1, 0] = int(color)
    return pair


def _choose_object_heights(H: int, n_objects: int) -> list[int] | None:
    """Choose odd bbox heights 3/5 whose sum fits within the full grid height."""
    used = 0
    heights: list[int] = []
    for idx in range(int(n_objects)):
        remaining = int(n_objects - idx - 1)
        max_allowed = min(5, int(H - used - 3 * remaining))
        choices = [v for v in (5, 3) if v <= max_allowed]
        if not choices:
            return None
        if 5 in choices and random.random() < 0.35:
            h = 5
        else:
            h = choices[-1]
        heights.append(int(h))
        used += int(h)
    return heights if sum(heights) <= int(H) else None


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    if H <= 11:
        n_movers = 2
    else:
        n_movers = random.randint(2, 3)
    max_total_objects = H // 3
    n_distractors = 0
    if max_total_objects > n_movers and random.choice([0, 0, 1]) == 1:
        n_distractors = 1

    total_objects = n_movers + n_distractors
    heights = _choose_object_heights(H, total_objects)
    if heights is None:
        return None

    width_choices = [v for v in (3, 5) if v <= int(sprite_area_w)]
    if not width_choices:
        return None
    widths = [random.choice(width_choices) for _ in range(int(total_objects))]

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]
    idx += n_distractors

    mover_pairs: list[np.ndarray] = []
    for k, mc in enumerate(mover_colors):
        mover_pairs.append(_make_diagonal_pair(int(mc), int(heights[k]), int(widths[k])))

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc, pair in zip(mover_colors, mover_pairs):
        sh, sw = pair.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - sh // 2
        min_col = ic0 + sw // 2
        max_col = ic1 - sw // 2
        if min_row > max_row or min_col > max_col:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), sh)
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), sw)
        if not row_marks or not col_marks:
            return None
        mover_targets[int(mc)] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = int(frame_base)
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = int(mc)
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    all_items: list[tuple[int, np.ndarray]] = []
    for mc, pair in zip(mover_colors, mover_pairs):
        all_items.append((int(mc), pair))
    for k, dc_color in enumerate(distractor_colors):
        pair = _make_diagonal_pair(int(dc_color), int(heights[n_movers + k]), int(widths[n_movers + k]))
        all_items.append((int(dc_color), pair))

    cur_row = 0
    for _color, pair in all_items:
        sh, sw = pair.shape
        if sh < 2 or sw < 2 or cur_row + sh > H or sw > sprite_area_w:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(pair[dr, dc]) != int(bg):
                    inp[cur_row + dr, c_off + dc] = int(pair[dr, dc])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None

    left_cols = np.arange(W)[None, :] < int(fc0)
    mover_bbox_areas: list[int] = []
    for mc in mover_colors:
        pts = np.argwhere((inp == int(mc)) & left_cols)
        if int(len(pts)) != 2:
            return None
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area <= 2:
            return None
        mover_bbox_areas.append(area)
    if not mover_bbox_areas or max(mover_bbox_areas) <= 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
