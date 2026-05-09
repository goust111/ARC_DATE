# Combo variant of 247ef758 (index 28)
# combo_spec: D1/1+D4/0
# Name: 颜色尺寸匹配+重着色印刻
#
# Design analysis
# ---------------
# a) Concrete behavior trace:
#    - Keep the original layout: left sprite strip | uniform divider column | full-height frame on the right.
#    - D1/1 changes the edge markers from single cells to mirrored same-color segments.
#      A left/right vertical segment marks one target row, and its segment length must equal the
#      sprite bbox height. A top/bottom horizontal segment marks one target column, and its length
#      must equal the sprite bbox width.
#    - D4/0 changes the stamping color: after matching is done with the sprite's original color
#      and bbox size, the copied sprite is stamped in the frame using the frame base color.
#    - Example trace: a red sprite with bbox 3x2 sees a red vertical segment of length 3 on both
#      side edges and a red horizontal segment of length 2 on both top/bottom edges.
#      Original task: any red row cell x any red column cell, stamp red.
#      Fused variant: use the segment centers only, require lengths 3 and 2 respectively, and
#      stamp the shape in frame_base instead of red.
#      If the frame also contains extra red segments of wrong lengths, those are visible clues but
#      must be ignored by the true rule.
# b) Complexity calibration:
#    - Sprite bbox sizes are constrained to small readable values, mainly 2..4.
#    - Each sample forces at least one mover color to have wrong-length mirrored row and column
#      segments, so color-only matching is insufficient.
#    - Movers remain 2-3 and distractors 0-1, which gives repeated support without exploding the
#      state space.
# c) Hidden assumptions audit:
#    - The original solver assumes marker locations are per-cell; this breaks here and is replaced
#      with maximal mirrored segment scanning.
#    - The original solver keys targets only by color; this breaks here and is replaced with
#      `(color, segment_length)` lookup tied to sprite bbox height/width.
#    - Recoloring breaks the assumption that output color equals mover color, so the solver must
#      explicitly recover `frame_base` from the frame border and stamp with it.
#    - Because all movers recolor to one color, generator placement must prevent stamped copies
#      from touching, otherwise different movers would visually merge into one blob.
# d) Placement legality:
#    - Segment intervals on the border are placed with at least one base-color cell between them,
#      so different segments never merge into a longer ambiguous segment.
#    - Target centers are chosen so each sprite fits entirely in the frame interior.
#    - All stamped bbox pairs are kept at least one cell apart, which preserves separability after
#      recolor.
# e) Visual observability:
#    - A human can see three coupled signals: sprite shape size, matching segment lengths on the
#      border, and output recolor to the enclosing frame base.
#    - Wrong-length same-color segments make the size rule observable, while multiple distinct
#      mover colors collapsing to one frame color make the recolor rule observable.
# f) Cleanliness:
#    - The frame interior stays empty in the input.
#    - Only one layer of border encoding is used, and recolored outputs are kept non-touching.
# g) Inference of color change:
#    - The recolor rule is global and simple: every moved sprite becomes the frame base color.
#      The frame base is visually stable because it remains the dominant border color.
# h) New solver assumptions and generator guarantees:
#    - `main()` now assumes mirrored same-color border runs are meaningful maximal segments, not
#      accidental mergers; generator enforces one-cell gaps between segments.
#    - `main()` assumes frame_base is the most frequent non-background color on the border;
#      generator validates this before accepting a sample.
#    - `main()` assumes at least one mover exists whose color has both row and column matches;
#      generator constructs every mover that way and rejects degenerate cases via ablation metrics.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the border."""
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


def _segment_extents(length: int) -> tuple[int, int]:
    upper = int((int(length) - 1) // 2)
    lower = int(length - 1 - upper)
    return upper, lower


def _segment_bounds(center: int, length: int) -> tuple[int, int]:
    upper, lower = _segment_extents(int(length))
    return int(center - upper), int(center + lower)


def _frame_base_color(g: np.ndarray, bg: int, fr0: int, fc0: int, fr1: int, fc1: int) -> int:
    """Most frequent non-bg color on the rectangular frame border."""
    border_vals: list[int] = []
    border_vals.extend(map(int, g[int(fr0), int(fc0) : int(fc1) + 1]))
    border_vals.extend(map(int, g[int(fr1), int(fc0) : int(fc1) + 1]))
    border_vals.extend(int(g[r, int(fc0)]) for r in range(int(fr0) + 1, int(fr1)))
    border_vals.extend(int(g[r, int(fc1)]) for r in range(int(fr0) + 1, int(fr1)))
    counts = Counter(v for v in border_vals if int(v) != int(bg))
    assert counts
    return int(counts.most_common(1)[0][0])


def _scan_mirrored_row_segments(
    g: np.ndarray,
    fc0: int,
    fc1: int,
    ir0: int,
    ir1: int,
    allowed_colors: set[int],
) -> dict[tuple[int, int], list[int]]:
    """
    Scan maximal mirrored vertical segments on left/right frame edges.
    Keys are (color, segment_length); values are segment floor-centers.
    """
    marks: dict[tuple[int, int], list[int]] = defaultdict(list)
    r = int(ir0)
    while r <= int(ir1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl in allowed_colors:
            r0 = int(r)
            while r + 1 <= int(ir1) and int(g[int(r + 1), int(fc0)]) == cl and int(g[int(r + 1), int(fc1)]) == cl:
                r += 1
            r1 = int(r)
            marks[(int(cl), int(r1 - r0 + 1))].append(int((r0 + r1) // 2))
        r += 1
    return marks


def _scan_mirrored_col_segments(
    g: np.ndarray,
    fr0: int,
    fr1: int,
    ic0: int,
    ic1: int,
    allowed_colors: set[int],
) -> dict[tuple[int, int], list[int]]:
    """
    Scan maximal mirrored horizontal segments on top/bottom frame edges.
    Keys are (color, segment_length); values are segment floor-centers.
    """
    marks: dict[tuple[int, int], list[int]] = defaultdict(list)
    c = int(ic0)
    while c <= int(ic1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct in allowed_colors:
            c0 = int(c)
            while c + 1 <= int(ic1) and int(g[int(fr0), int(c + 1)]) == ct and int(g[int(fr1), int(c + 1)]) == ct:
                c += 1
            c1 = int(c)
            marks[(int(ct), int(c1 - c0 + 1))].append(int((c0 + c1) // 2))
        c += 1
    return marks


def _extract_sprites(g: np.ndarray, bg: int, fc0: int) -> list[dict[str, object]]:
    H, W = g.shape
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)
    sprites: list[dict[str, object]] = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                    "height": int(m.shape[0]),
                    "width": int(m.shape[1]),
                }
            )
    return sprites


def _solve_variant(grid: np.ndarray, *, use_size_match: bool, recolor_to_base: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    frame_base = _frame_base_color(g, bg, fr0, fc0, fr1, fc1)

    sprites = _extract_sprites(g, bg, int(fc0))
    move_colors = {int(s["color"]) for s in sprites}

    row_marks = _scan_mirrored_row_segments(g, int(fc0), int(fc1), int(ir0), int(ir1), move_colors)
    col_marks = _scan_mirrored_col_segments(g, int(fr0), int(fr1), int(ic0), int(ic1), move_colors)

    movers: list[dict[str, object]] = []
    for s in sprites:
        color = int(s["color"])
        if use_size_match:
            rows = sorted(set(row_marks.get((color, int(s["height"])), [])))
            cols = sorted(set(col_marks.get((color, int(s["width"])), [])))
        else:
            rows = sorted({int(center) for (col, _length), vals in row_marks.items() if int(col) == color for center in vals})
            cols = sorted({int(center) for (col, _length), vals in col_marks.items() if int(col) == color for center in vals})
        if rows and cols:
            mover = dict(s)
            mover["targets"] = [(int(r), int(c)) for r in rows for c in cols]
            movers.append(mover)

    out = g.copy()
    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(np.asarray(s["mask"], dtype=bool)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        stamp_color = int(frame_base) if recolor_to_base else int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(np.asarray(s["mask"], dtype=bool))]
        for tr, tc in s["targets"]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(stamp_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Match mover color to mirrored border segments.
    - Also require row-segment length == sprite bbox height and col-segment length == sprite bbox width.
    - Use segment centers as target coordinates.
    - Erase moved sprites on the left, then stamp their shapes recolored to the frame base color.
    """
    return _solve_variant(grid, use_size_match=True, recolor_to_base=True)


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """
    Generate a single-color 8-connected sprite cropped to its bounding box.
    BBoxes stay small so segment lengths remain visually readable.
    """
    if max_h < 2 or max_w < 2:
        return None
    sh = min(random.choice([2, 3, 3]), max_h)
    sw = min(random.choice([2, 3, 3, 4]), max_w)
    if sh < 2 or sw < 2:
        return None

    mask = np.zeros((sh, sw), dtype=bool)
    pivot_r = random.randrange(sh)
    pivot_c = random.randrange(sw)
    mask[int(pivot_r), :] = True
    mask[:, int(pivot_c)] = True

    extras: list[tuple[int, int]] = []
    for r in range(sh):
        for c in range(sw):
            if bool(mask[int(r), int(c)]):
                continue
            if any(
                0 <= int(r + dr) < sh and 0 <= int(c + dc) < sw and bool(mask[int(r + dr), int(c + dc)])
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            ):
                extras.append((int(r), int(c)))
    random.shuffle(extras)
    for r, c in extras[: random.randint(0, min(2, len(extras)))]:
        mask[int(r), int(c)] = True

    sprite = np.zeros((sh, sw), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _can_place_interval(start: int, end: int, occupied: np.ndarray, lo: int, hi: int) -> bool:
    if int(start) < int(lo) or int(end) > int(hi):
        return False
    pad_start = max(int(lo), int(start - 1))
    pad_end = min(int(hi), int(end + 1))
    for pos in range(int(pad_start), int(pad_end) + 1):
        if bool(occupied[int(pos - lo)]):
            return False
    return True


def _occupy_interval(start: int, end: int, occupied: np.ndarray, lo: int) -> None:
    occupied[int(start - lo) : int(end - lo) + 1] = True


def _place_interval_center(
    length: int,
    occupied: np.ndarray,
    lo: int,
    hi: int,
    *,
    center_lo: int | None = None,
    center_hi: int | None = None,
) -> int | None:
    upper, lower = _segment_extents(int(length))
    low = int(lo + upper if center_lo is None else max(int(center_lo), int(lo + upper)))
    high = int(hi - lower if center_hi is None else min(int(center_hi), int(hi - lower)))
    if low > high:
        return None
    candidates = list(range(int(low), int(high) + 1))
    random.shuffle(candidates)
    for center in candidates:
        start, end = _segment_bounds(int(center), int(length))
        if _can_place_interval(int(start), int(end), occupied, int(lo), int(hi)):
            _occupy_interval(int(start), int(end), occupied, int(lo))
            return int(center)
    return None


def _stamp_bbox(height: int, width: int, tr: int, tc: int) -> tuple[int, int, int, int]:
    r0 = int(tr - height // 2)
    c0 = int(tc - width // 2)
    return int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _pick_segment_targets(
    height: int,
    width: int,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    row_occupied: np.ndarray,
    col_occupied: np.ndarray,
    existing_bboxes: list[tuple[int, int, int, int]],
    n_rows: int,
    n_cols: int,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray, list[tuple[int, int, int, int]]] | None:
    row_upper, row_lower = _segment_extents(int(height))
    col_upper, col_lower = _segment_extents(int(width))
    min_row = int(ir0 + row_upper)
    max_row = int(ir1 - row_lower)
    min_col = int(ic0 + col_upper)
    max_col = int(ic1 - col_lower)
    if min_row > max_row or min_col > max_col:
        return None

    for _ in range(120):
        row_occ = row_occupied.copy()
        col_occ = col_occupied.copy()
        rows: list[int] = []
        cols: list[int] = []
        ok = True

        for _j in range(int(n_rows)):
            center = _place_interval_center(int(height), row_occ, int(ir0), int(ir1), center_lo=int(min_row), center_hi=int(max_row))
            if center is None:
                ok = False
                break
            rows.append(int(center))
        if not ok:
            continue

        for _j in range(int(n_cols)):
            center = _place_interval_center(int(width), col_occ, int(ic0), int(ic1), center_lo=int(min_col), center_hi=int(max_col))
            if center is None:
                ok = False
                break
            cols.append(int(center))
        if not ok:
            continue

        rows = sorted(set(rows))
        cols = sorted(set(cols))
        if len(rows) != int(n_rows) or len(cols) != int(n_cols):
            continue

        boxes = [_stamp_bbox(int(height), int(width), int(r), int(c)) for r in rows for c in cols]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _bboxes_too_close(boxes[i], boxes[j]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for box in boxes:
            if any(_bboxes_too_close(box, prev) for prev in existing_bboxes):
                ok = False
                break
        if ok:
            return rows, cols, row_occ, col_occ, boxes
    return None


def _border_counts(g: np.ndarray, fr0: int, fc0: int, fr1: int, fc1: int) -> Counter[int]:
    vals: list[int] = []
    vals.extend(map(int, g[int(fr0), int(fc0) : int(fc1) + 1]))
    vals.extend(map(int, g[int(fr1), int(fc0) : int(fc1) + 1]))
    vals.extend(int(g[r, int(fc0)]) for r in range(int(fr0) + 1, int(fr1)))
    vals.extend(int(g[r, int(fc1)]) for r in range(int(fr0) + 1, int(fr1)))
    return Counter(vals)


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    combo_out = main(inp)
    color_only_out = _solve_variant(inp, use_size_match=False, recolor_to_base=True)
    keep_color_out = _solve_variant(inp, use_size_match=True, recolor_to_base=False)

    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    frame_base = _frame_base_color(g, bg, fr0, fc0, fr1, fc1)

    sprites = _extract_sprites(g, bg, int(fc0))
    sprite_dims = {int(s["color"]): (int(s["height"]), int(s["width"])) for s in sprites}
    move_colors = set(sprite_dims.keys())
    row_marks = _scan_mirrored_row_segments(g, int(fc0), int(fc1), int(ir0), int(ir1), move_colors)
    col_marks = _scan_mirrored_col_segments(g, int(fr0), int(fr1), int(ic0), int(ic1), move_colors)

    wrong_row_segments = sum(
        1
        for (col, length), centers in row_marks.items()
        if int(col) in sprite_dims and int(length) != int(sprite_dims[int(col)][0]) and len(centers) > 0
    )
    wrong_col_segments = sum(
        1
        for (col, length), centers in col_marks.items()
        if int(col) in sprite_dims and int(length) != int(sprite_dims[int(col)][1]) and len(centers) > 0
    )

    interior_mask = np.zeros_like(g, dtype=bool)
    interior_mask[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1] = True
    recolored_pixels = int(np.sum(interior_mask & (combo_out == int(frame_base)) & (keep_color_out != int(frame_base))))
    diff_vs_color_only = int(np.sum(combo_out != color_only_out))
    diff_vs_keep_color = int(np.sum(combo_out != keep_color_out))

    return {
        "wrong_row_segments": int(wrong_row_segments),
        "wrong_col_segments": int(wrong_col_segments),
        "recolored_pixels": int(recolored_pixels),
        "diff_vs_color_only": int(diff_vs_color_only),
        "diff_vs_keep_color": int(diff_vs_keep_color),
    }


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

    div_col = int(sprite_area_w)
    fc0, fc1 = int(sprite_area_w + 1), int(W - 1)
    ir0, ic0 = 1, int(fc0 + 1)
    ir1, ic1 = int(H - 2), int(fc1 - 1)

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 0, 0, 1])
    total_items = int(n_movers + n_distractors)
    max_sprite_h = min(3, max(2, H // max(1, total_items)))

    idx = 0
    divider_color = int(all_colors[idx]); idx += 1
    frame_base = int(all_colors[idx]); idx += 1
    mover_colors = [int(v) for v in all_colors[idx : idx + n_movers]]; idx += n_movers
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    sprites: list[np.ndarray] = []
    sprite_dims: dict[int, tuple[int, int]] = {}
    for mc in mover_colors:
        s = _make_sprite(int(mc), int(max_sprite_h), int(sprite_area_w))
        if s is None:
            return None
        sprites.append(s)
        sprite_dims[int(mc)] = (int(s.shape[0]), int(s.shape[1]))

    total_sprite_h = int(sum(int(sp.shape[0]) for sp in sprites))
    if total_sprite_h > int(H):
        return None

    row_counts = {int(mc): 1 for mc in mover_colors}
    col_counts = {int(mc): 1 for mc in mover_colors}
    if random.random() < 0.25:
        row_counts[int(random.choice(mover_colors))] += 1
    if random.random() < 0.25:
        col_counts[int(random.choice(mover_colors))] += 1

    row_centers: dict[int, list[int]] = {}
    col_centers: dict[int, list[int]] = {}
    row_lengths: dict[int, list[int]] = {}
    col_lengths: dict[int, list[int]] = {}
    row_occupied = np.zeros(int(interior_h), dtype=bool)
    col_occupied = np.zeros(int(interior_w), dtype=bool)
    occupied_target_boxes: list[tuple[int, int, int, int]] = []

    mover_order = mover_colors[:]
    random.shuffle(mover_order)
    mover_order.sort(key=lambda col: (-int(sprite_dims[int(col)][0] * sprite_dims[int(col)][1]), random.random()))
    for mc in mover_order:
        sh, sw = sprite_dims[int(mc)]
        picked = _pick_segment_targets(
            int(sh),
            int(sw),
            int(ir0),
            int(ic0),
            int(ir1),
            int(ic1),
            row_occupied,
            col_occupied,
            occupied_target_boxes,
            int(row_counts[int(mc)]),
            int(col_counts[int(mc)]),
        )
        if picked is None:
            return None
        rows, cols, row_occupied, col_occupied, boxes = picked
        row_centers[int(mc)] = rows
        col_centers[int(mc)] = cols
        row_lengths[int(mc)] = [int(sh)] * len(rows)
        col_lengths[int(mc)] = [int(sw)] * len(cols)
        occupied_target_boxes.extend(boxes)

    ambiguous_color = int(random.choice(mover_colors))
    bad_row_lengths = [v for v in [2, 3, 4] if v <= int(interior_h) and v != int(sprite_dims[int(ambiguous_color)][0])]
    bad_col_lengths = [v for v in [2, 3, 4] if v <= int(interior_w) and v != int(sprite_dims[int(ambiguous_color)][1])]
    if not bad_row_lengths or not bad_col_lengths:
        return None

    bad_row_len = int(random.choice(bad_row_lengths))
    bad_col_len = int(random.choice(bad_col_lengths))
    bad_row_center = _place_interval_center(int(bad_row_len), row_occupied, int(ir0), int(ir1))
    bad_col_center = _place_interval_center(int(bad_col_len), col_occupied, int(ic0), int(ic1))
    if bad_row_center is None or bad_col_center is None:
        return None
    row_centers[int(ambiguous_color)].append(int(bad_row_center))
    row_lengths[int(ambiguous_color)].append(int(bad_row_len))
    col_centers[int(ambiguous_color)].append(int(bad_col_center))
    col_lengths[int(ambiguous_color)].append(int(bad_col_len))

    inp = np.full((H, W), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)
    inp[0, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[H - 1, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[1 : H - 1, int(fc0)] = int(frame_base)
    inp[1 : H - 1, int(fc1)] = int(frame_base)

    for mc in mover_colors:
        for center, length in zip(row_centers[int(mc)], row_lengths[int(mc)]):
            r0, r1 = _segment_bounds(int(center), int(length))
            inp[int(r0) : int(r1) + 1, int(fc0)] = int(mc)
            inp[int(r0) : int(r1) + 1, int(fc1)] = int(mc)
        for center, length in zip(col_centers[int(mc)], col_lengths[int(mc)]):
            c0, c1 = _segment_bounds(int(center), int(length))
            inp[0, int(c0) : int(c1) + 1] = int(mc)
            inp[H - 1, int(c0) : int(c1) + 1] = int(mc)

    border_counts = _border_counts(inp, int(0), int(fc0), int(H - 1), int(fc1))
    if border_counts[int(frame_base)] <= max(count for col, count in border_counts.items() if int(col) != int(frame_base)):
        return None
    if _frame_base_color(inp, int(bg), 0, int(fc0), int(H - 1), int(fc1)) != int(frame_base):
        return None

    all_items: list[tuple[int, np.ndarray]] = [(int(mc), sp) for mc, sp in zip(mover_colors, sprites)]
    remaining_h = int(H - sum(int(sp.shape[0]) for _col, sp in all_items))
    for dc_color in distractor_colors:
        max_h_here = min(4, max(2, remaining_h))
        ds = _make_sprite(int(dc_color), int(max_h_here), int(sprite_area_w))
        if ds is None:
            return None
        all_items.append((int(dc_color), ds))
        remaining_h -= int(ds.shape[0])
    if sum(int(sp.shape[0]) for _col, sp in all_items) > int(H):
        return None

    cur_row = 0
    for _col, sp in all_items:
        sh, sw = sp.shape
        if int(cur_row + sh) > int(H) or int(sw) > int(sprite_area_w):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(int(sh)):
            for dc in range(int(sw)):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[int(dr), int(dc)])
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        metrics = _variant_metrics(inp)
    except Exception:
        return None
    if metrics["wrong_row_segments"] < 1 or metrics["wrong_col_segments"] < 1:
        return None
    if metrics["diff_vs_color_only"] <= 0:
        return None
    if metrics["diff_vs_keep_color"] <= 0:
        return None
    if metrics["recolored_pixels"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 800 attempts")
