# Combo variant of 247ef758 (index 26)
# combo_spec: D2/3+D3/3
# Name: 对角像素对+投影交点目标
#
# Design analysis
# ===============
# fused_rule:
# - Keep the original macro layout: left source strip | one full-height divider | right
#   rectangular frame with a bg interior.
# - D2 change: each movable source is no longer a connected blob. It is exactly two
#   same-color pixels placed at opposite corners of an odd-sized rectangle in the left
#   strip. The output shape is the filled rectangle inferred from that diagonal pair.
# - D3 change: frame markers no longer directly encode a row and a column. For each
#   mover color, top+left markers define one or two "/" projection lines, and top+right
#   markers define one or two "\" projection lines. The target centers are the pairwise
#   intersections of those projection lines.
#
# concrete_behavior_trace:
# - Original task:
#   1. Read one left-strip blob per color.
#   2. Read rows from matching left/right edge colors and columns from matching
#      top/bottom edge colors.
#   3. Stamp the blob mask at every row×col center.
# - Fused task:
#   1. Read one diagonal pair per color in the left strip. Example: color 4 at
#      (r,c) corners of a 3x5 bbox implies a filled 3x5 rectangle shape.
#   2. Read "/" lines from a top marker at column sc and a left marker at row
#      ir0 + (sc - ic0). Read "\" lines from a top marker at column bc and a right
#      marker at row ir0 + (ic1 - bc).
#   3. Intersect them at:
#        tr = ir0 + (sc - bc) // 2
#        tc = (sc + bc) // 2
#      so the target is not literally on a marked row/column and must be inferred.
#   4. Center the filled rectangle there and stamp it. Lower sources still stamp first,
#      higher sources stamp later and cover them.
# - Observable difference from the original:
#   the left strip is sparse corner pairs instead of blobs, and the frame shows only
#   top/left/right projection endpoints instead of mirrored row/column markers.
#
# complexity_calibration:
# - Rectangle sizes are restricted to odd 3 or 5 in each dimension. This keeps centers
#   integral and the geometry legible.
# - Each color gets 1-2 "/" lines and 1-2 "\" lines, so one source may replicate to
#   multiple targets, preserving the original "copy to several places" structure.
# - Colors remain distinct across divider, frame base, movers, and distractors to avoid
#   any hidden value-based reasoning.
#
# implicit_assumption_audit:
# - Original connected-component extraction is invalid because diagonal-pair pixels are
#   usually disconnected; the solver must collect all left-strip pixels of a color and
#   verify they form opposite rectangle corners.
# - Original row_marks / col_marks logic assumes mirrored same-color edge markers. That
#   assumption is intentionally broken here: bottom edge is left as base color, and
#   side markers participate in diagonals rather than direct rows/columns.
# - The new solver assumes each valid source bbox has odd height and odd width so its
#   center is a cell, not a half-cell. The generator guarantees this.
# - The new projection decoding assumes top+left and top+right endpoints are paired by
#   matching corner offset, not by arbitrary ordering. The generator guarantees unique,
#   non-reused marker positions so this decoding is unambiguous.
#
# placement_legality:
# - The generator samples source bbox sizes first, then only chooses projection markers
#   whose induced targets keep the translated rectangle fully inside the frame interior.
# - Left-strip objects are stacked in disjoint vertical bands, so diagonal pairs do not
#   overlap and each color still reads as one coherent source object.
# - Top/left/right marker slots are globally non-reused across mover colors, and target
#   centers are non-reused, keeping projection cues visually separable.
#
# visual_observability_and_cleanliness:
# - A human can see the shape rule because each left-strip mover is only two pixels,
#   while the output is a solid rectangle of the implied bbox.
# - A human can see the location rule because the output center is away from any direct
#   mirrored row/column cue; it aligns with the crossing of two 45-degree projections.
# - Bottom edge is deliberately uninformative to suppress accidental reading as the
#   original top/bottom column-marker rule.
#
# fusion_guardrails:
# - Generator self-checks require every mover to have non-empty projection targets and
#   zero direct Cartesian targets under the original row/column interpretation.
# - Therefore the produced inputs do not collapse back to the original task even though
#   the global layout is still recognizably derived from 247ef758.
#
# new_solver_assumptions_checked_against_generator:
# - Every mover color appears exactly twice in the left strip and those two pixels are
#   diagonal opposite corners of an odd-sized bbox.
# - Every mover color has at least one "/" top+left line and at least one "\" top+right
#   line, with same-parity top columns and sc > bc so intersections are integral.
# - Distractor colors have no valid projection targets and therefore remain unchanged.

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
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
        if ((r1 - r0) & 1) or ((c1 - c0) & 1):
            continue

        corners = {(int(r_a), int(c_a)), (int(r_b), int(c_b))}
        if corners not in ({(r0, c0), (r1, c1)}, {(r0, c1), (r1, c0)}):
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


def _projection_targets_from_top_cols(
    slash_cols: list[int],
    back_cols: list[int],
    *,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
) -> list[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for sc in slash_cols:
        for bc in back_cols:
            if sc <= bc or ((sc - bc) & 1):
                continue
            tr = int(ir0 + (sc - bc) // 2)
            tc = int((sc + bc) // 2)
            if int(ir0) <= tr <= int(ir1) and int(ic0) <= tc <= int(ic1):
                pts.add((tr, tc))
    return sorted(pts)


def _projection_targets_for_color(
    g: np.ndarray,
    color: int,
    *,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fc0: int,
    fc1: int,
) -> list[tuple[int, int]]:
    top_cols = [int(c) for c in range(int(ic0), int(ic1) + 1) if int(g[int(fr0), int(c)]) == int(color)]
    left_rows = {int(r) for r in range(int(ir0), int(ir1) + 1) if int(g[int(r), int(fc0)]) == int(color)}
    right_rows = {int(r) for r in range(int(ir0), int(ir1) + 1) if int(g[int(r), int(fc1)]) == int(color)}

    slash_cols: list[int] = []
    for c in top_cols:
        left_r = int(ir0 + (int(c) - int(ic0)))
        if int(ir0) <= left_r <= int(ir1) and left_r in left_rows:
            slash_cols.append(int(c))

    back_cols: list[int] = []
    for c in top_cols:
        right_r = int(ir0 + (int(ic1) - int(c)))
        if int(ir0) <= right_r <= int(ir1) and right_r in right_rows:
            back_cols.append(int(c))

    return _projection_targets_from_top_cols(slash_cols, back_cols, ir0=ir0, ic0=ic0, ir1=ir1, ic1=ic1)


def _direct_cartesian_targets_for_color(
    g: np.ndarray,
    color: int,
    *,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fr1: int,
    fc0: int,
    fc1: int,
) -> list[tuple[int, int]]:
    row_marks = [
        int(r)
        for r in range(int(ir0), int(ir1) + 1)
        if int(g[int(r), int(fc0)]) == int(color) and int(g[int(r), int(fc1)]) == int(color)
    ]
    col_marks = [
        int(c)
        for c in range(int(ic0), int(ic1) + 1)
        if int(g[int(fr0), int(c)]) == int(color) and int(g[int(fr1), int(c)]) == int(color)
    ]
    return sorted({(int(r), int(c)) for r in row_marks for c in col_marks})


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move diagonal-pair sources from the left strip into the framed area.

    Fused rule:
    - Source shape comes from two diagonal corner pixels of the same color.
    - Target centers come from intersections of top+left "/" projections and
      top+right "\" projections of the same color.
    - The stamped output is the filled source bbox centered at each inferred target.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
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

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in sorted(move_colors):
        pts = _projection_targets_for_color(
            g,
            int(col),
            ir0=ir0,
            ic0=ic0,
            ir1=ir1,
            ic1=ic1,
            fr0=fr0,
            fc0=fc0,
            fc1=fc1,
        )
        if pts:
            targets[int(col)] = pts

    out = g.copy()
    movers = [p for p in pairs if int(p["color"]) in targets]

    for p in movers:
        for rr, cc in p["pixels"]:
            out[int(rr), int(cc)] = int(bg)

    movers.sort(key=lambda p: (-((int(p["r0"]) + int(p["r1"])) / 2.0), int(p["c0"])))

    for p in movers:
        color = int(p["color"])
        r0, c0, r1, c1 = map(int, (p["r0"], p["c0"], p["r1"], p["c1"]))
        cr = (int(r0) + int(r1)) // 2
        cc = (int(c0) + int(c1)) // 2

        for tr, tc in targets[int(color)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for rr in range(int(r0), int(r1) + 1):
                for cc2 in range(int(c0), int(c1) + 1):
                    nr = int(rr + dr)
                    nc = int(cc2 + dc)
                    if int(ir0) <= nr <= int(ir1) and int(ic0) <= nc <= int(ic1):
                        out[nr, nc] = int(color)

    return out


def _make_diagonal_pair(color: int, h: int, w: int) -> np.ndarray:
    """Return an h×w object with exactly two same-color diagonal corner pixels."""
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


def _fit_bounds_for_bbox(h: int, w: int, *, ir0: int, ic0: int, ir1: int, ic1: int) -> tuple[int, int, int, int]:
    top_margin = int(h // 2)
    bot_margin = int(h // 2)
    left_margin = int(w // 2)
    right_margin = int(w // 2)
    return (
        int(ir0 + top_margin),
        int(ir1 - bot_margin),
        int(ic0 + left_margin),
        int(ic1 - right_margin),
    )


def _choose_projection_sets(
    h: int,
    w: int,
    *,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    used_top: set[int],
    used_left: set[int],
    used_right: set[int],
    used_targets: set[tuple[int, int]],
) -> tuple[list[int], list[int], list[tuple[int, int]]] | None:
    fit_r0, fit_r1, fit_c0, fit_c1 = _fit_bounds_for_bbox(h, w, ir0=ir0, ic0=ic0, ir1=ir1, ic1=ic1)
    if fit_r0 > fit_r1 or fit_c0 > fit_c1:
        return None

    max_slash = min(int(ic1), int(ic0 + (ir1 - ir0)))
    min_back = max(int(ic0), int(ic1 - (ir1 - ir0)))
    base_slash = [c for c in range(int(ic0), int(max_slash) + 1) if c not in used_top]
    base_back = [c for c in range(int(min_back), int(ic1) + 1) if c not in used_top]
    if not base_slash or not base_back:
        return None

    count_options = [(2, 2), (1, 2), (2, 1), (1, 1)]
    for n_slash, n_back in count_options:
        for _ in range(300):
            parity = random.randint(0, 1)
            slash_pool = [c for c in base_slash if (int(c) & 1) == parity]
            back_pool = [c for c in base_back if (int(c) & 1) == parity]
            if len(slash_pool) < n_slash or len(back_pool) < n_back:
                continue

            chosen_back = sorted(random.sample(back_pool, n_back))
            higher_slash_pool = [c for c in slash_pool if int(c) > max(chosen_back)]
            if len(higher_slash_pool) < n_slash:
                continue
            chosen_slash = sorted(random.sample(higher_slash_pool, n_slash))

            left_rows = [int(ir0 + (int(c) - int(ic0))) for c in chosen_slash]
            right_rows = [int(ir0 + (int(ic1) - int(c))) for c in chosen_back]
            if any(r in used_left for r in left_rows) or any(r in used_right for r in right_rows):
                continue
            if set(left_rows) & set(right_rows):
                continue

            targets = _projection_targets_from_top_cols(
                chosen_slash,
                chosen_back,
                ir0=ir0,
                ic0=ic0,
                ir1=ir1,
                ic1=ic1,
            )
            if len(targets) != int(n_slash * n_back):
                continue
            if any((tr, tc) in used_targets for tr, tc in targets):
                continue
            if any(not (fit_r0 <= tr <= fit_r1 and fit_c0 <= tc <= fit_c1) for tr, tc in targets):
                continue

            return chosen_slash, chosen_back, targets

    return None


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
    fr0, fr1 = 0, H - 1

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

    mover_pairs: list[np.ndarray] = []
    for k, mc in enumerate(mover_colors):
        mover_pairs.append(_make_diagonal_pair(int(mc), int(heights[k]), int(widths[k])))

    used_top: set[int] = set()
    used_left: set[int] = set()
    used_right: set[int] = set()
    used_targets: set[tuple[int, int]] = set()
    mover_markers: dict[int, tuple[list[int], list[int], list[tuple[int, int]]]] = {}
    for mc, pair in zip(mover_colors, mover_pairs):
        ph, pw = map(int, pair.shape)
        picked = _choose_projection_sets(
            ph,
            pw,
            ir0=ir0,
            ic0=ic0,
            ir1=ir1,
            ic1=ic1,
            used_top=used_top,
            used_left=used_left,
            used_right=used_right,
            used_targets=used_targets,
        )
        if picked is None:
            return None
        slash_cols, back_cols, targets = picked
        mover_markers[int(mc)] = (slash_cols, back_cols, targets)
        used_top.update(slash_cols)
        used_top.update(back_cols)
        used_left.update(int(ir0 + (int(c) - int(ic0))) for c in slash_cols)
        used_right.update(int(ir0 + (int(ic1) - int(c))) for c in back_cols)
        used_targets.update(targets)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = int(divider_color)

    inp[fr0, fc0 : fc1 + 1] = int(frame_base)
    inp[fr1, fc0 : fc1 + 1] = int(frame_base)
    inp[1 : H - 1, fc0] = int(frame_base)
    inp[1 : H - 1, fc1] = int(frame_base)

    for mc, (slash_cols, back_cols, _targets) in mover_markers.items():
        for c in slash_cols:
            inp[fr0, int(c)] = int(mc)
            inp[int(ir0 + (int(c) - int(ic0))), fc0] = int(mc)
        for c in back_cols:
            inp[fr0, int(c)] = int(mc)
            inp[int(ir0 + (int(ic1) - int(c))), fc1] = int(mc)

    all_items: list[tuple[int, np.ndarray]] = []
    for mc, pair in zip(mover_colors, mover_pairs):
        all_items.append((int(mc), pair))
    for k, dc_color in enumerate(distractor_colors):
        pair = _make_diagonal_pair(int(dc_color), int(heights[n_movers + k]), int(widths[n_movers + k]))
        all_items.append((int(dc_color), pair))

    cur_row = 0
    for _color, pair in all_items:
        ph, pw = map(int, pair.shape)
        if ph < 2 or pw < 2 or cur_row + ph > H or pw > sprite_area_w:
            return None
        c_off = random.randint(0, sprite_area_w - pw)
        for dr in range(int(ph)):
            for dc in range(int(pw)):
                if int(pair[dr, dc]) != int(bg):
                    inp[int(cur_row) + int(dr), int(c_off) + int(dc)] = int(pair[dr, dc])
        cur_row += int(ph)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    left_cols = np.arange(W)[None, :] < int(fc0)
    any_multi_target = False
    for mc in mover_colors:
        pts = np.argwhere((inp == int(mc)) & left_cols)
        if int(len(pts)) != 2:
            return None
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 == r1 or c0 == c1:
            return None
        if ((int(r1) - int(r0)) & 1) or ((int(c1) - int(c0)) & 1):
            return None
        if int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)) <= 2:
            return None

        proj_targets = _projection_targets_for_color(
            inp,
            int(mc),
            ir0=ir0,
            ic0=ic0,
            ir1=ir1,
            ic1=ic1,
            fr0=fr0,
            fc0=fc0,
            fc1=fc1,
        )
        if not proj_targets:
            return None
        if len(proj_targets) > 1:
            any_multi_target = True

        direct_targets = _direct_cartesian_targets_for_color(
            inp,
            int(mc),
            ir0=ir0,
            ic0=ic0,
            ir1=ir1,
            ic1=ic1,
            fr0=fr0,
            fr1=fr1,
            fc0=fc0,
            fc1=fc1,
        )
        if direct_targets:
            return None

    if not any_multi_target and n_movers >= 3:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
