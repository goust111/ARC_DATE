# Combo variant of 247ef758 (index 20)
# combo_spec: D3/2+D7/0+D14/0
# Name: 条纹非对称框架+逆序覆盖
#
# Design analysis
# ---------------
# a) Concrete realization and behavior trace
#    The global layout stays "left sprite strip | full-height divider | right frame".
#    The fusion keeps D14's two-color alternating frame, so ordinary frame pixels already use
#    mover colors and markers must be read as stripe breaks, not as "special colors".
#    Rows are still mirrored on left/right: if both vertical edges show the same color and that
#    color breaks the recovered row stripe, that row belongs to that mover.
#    Columns use D3/2's asymmetric rule: top and bottom are parsed independently, and any column
#    where either horizontal edge breaks its own stripe contributes a target column for the
#    observed mover color. Example: if top has an A-break at c=7 and bottom has a B-break at
#    c=10, then A may stamp at c=7 and B at c=10 even though top!=bottom at both columns.
#    Finally D7/0 reverses the stamping order: upper left-strip sprites stamp first, lower ones
#    stamp later and therefore cover overlap pixels.
# b) Complexity calibration
#    The stripe code is only reliable if the frame palette is small and repetitive, so the
#    generator uses exactly two movers as the two stripe colors. That keeps the alternation
#    visually recoverable and avoids ambiguous multi-color periodic fitting. The generator also
#    forces one top-only column anomaly and one bottom-only column anomaly so the non-mirrored
#    column rule is visible instead of silently collapsing back to the source task.
# c) Implicit assumption audit against the source solver
#    The original tests `cl == cr and cl in move_colors` / `ct == cb and ct in move_colors`
#    would now massively over-fire because ordinary stripe pixels already satisfy them. The new
#    solver must first fit the best alternating pattern, then accept only positions that disagree
#    with that expectation. The original horizontal condition `ct == cb` is specifically invalid
#    under D3/2 and is replaced by separate top and bottom anomaly collection. The original
#    bottom-to-top sorting is also invalid and is inverted to ascending center row.
# d) Placement legality
#    The generator samples target centers only from rows/cols where the base stripe expects the
#    opposite mover color, guaranteeing a visible break. Target centers are still clipped by
#    sprite bbox dimensions so each stamp fits inside the frame interior. Left-strip sprites are
#    stacked without overlap and the divider remains a full-height uniform column.
# e) Visual observability
#    A human can now see three independent signals across examples: (1) the frame is a regular
#    two-color stripe, (2) some useful columns appear only on the top edge or only on the bottom
#    edge, and (3) when two stamped sprites overlap, the lower input sprite ends up visible on
#    top because stamping proceeds from top to bottom.
# f) Cleanliness
#    Only one overlapping mover pair is forced, so the D7 signal is present without turning the
#    frame interior into a dense collision field. Distractors remain optional and never appear on
#    the frame, preserving the original "some left sprites do not move" readability.
# g) New solver assumptions guaranteed by the generator
#    - The frame perimeter contains exactly the two mover colors.
#    - Left/right edges share one recoverable alternating phase; top and bottom each have a
#      recoverable alternating phase with only sparse anomalies.
#    - Every row mark is mirrored on left/right; every mover has at least one row mark and at
#      least one column mark from top or bottom.
#    - At least one top-only and one bottom-only column anomaly exist.
#    - The two movers have overlapping target placements, and old-vs-new stamping order changes
#      the output, so D7 does not degenerate away.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the grid border."""
    h, w = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _best_alternating_expectation(seqs: list[list[int]], colors: list[int]) -> list[int]:
    """Fit the lower-error two-color alternation shared by one or more equal-length edge sequences."""
    assert len(colors) == 2
    length = len(seqs[0])
    best_phase = 0
    best_err = None
    for phase in (0, 1):
        err = 0
        for seq in seqs:
            assert len(seq) == length
            for i, v in enumerate(seq):
                if int(v) != int(colors[(i + phase) % 2]):
                    err += 1
        if best_err is None or err < best_err:
            best_err = err
            best_phase = phase
    return [int(colors[(i + best_phase) % 2]) for i in range(length)]


def _find_scene_structure(grid: np.ndarray) -> tuple[int, int, int, int, int, int, int, int, int, np.ndarray]:
    """Recover frame bounds and divider mask from the canonical layout."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((h, w), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(h - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    return bg, ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1, divider_mask


def _extract_sprites(grid: np.ndarray, *, bg: int, fc0: int, divider_mask: np.ndarray) -> list[dict[str, object]]:
    """Extract single-color 8-connected sprites from the left strip."""
    g = np.array(grid, dtype=int)
    _h, w = g.shape
    left_cols = np.arange(w)[None, :] < int(fc0)
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
            cropped = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": cropped,
                }
            )

    return sprites


def _collect_marker_data(
    grid: np.ndarray,
    *,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
    move_colors: set[int],
) -> tuple[dict[int, list[int]], dict[int, list[int]], set[int], set[int]]:
    """
    Recover row/column targets from stripe breaks.

    Rows:
      left/right must agree and disagree with the fitted vertical stripe expectation.
    Columns:
      top and bottom are parsed independently; any anomaly on either edge contributes.
    """
    g = np.array(grid, dtype=int)
    frame_colors = sorted(
        set(map(int, g[int(fr0), int(fc0) : int(fc1) + 1]))
        | set(map(int, g[int(fr1), int(fc0) : int(fc1) + 1]))
        | set(map(int, g[int(fr0) : int(fr1) + 1, int(fc0)]))
        | set(map(int, g[int(fr0) : int(fr1) + 1, int(fc1)]))
    )
    assert len(frame_colors) == 2

    left_seq = [int(g[int(r), int(fc0)]) for r in range(int(ir0), int(ir1) + 1)]
    right_seq = [int(g[int(r), int(fc1)]) for r in range(int(ir0), int(ir1) + 1)]
    top_seq = [int(g[int(fr0), int(c)]) for c in range(int(ic0), int(ic1) + 1)]
    bottom_seq = [int(g[int(fr1), int(c)]) for c in range(int(ic0), int(ic1) + 1)]

    expected_rows = _best_alternating_expectation([left_seq, right_seq], frame_colors)
    expected_top = _best_alternating_expectation([top_seq], frame_colors)
    expected_bottom = _best_alternating_expectation([bottom_seq], frame_colors)

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    top_anomaly_cols: set[int] = set()
    bottom_anomaly_cols: set[int] = set()

    for offset, r in enumerate(range(int(ir0), int(ir1) + 1)):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl in move_colors and cl != int(expected_rows[int(offset)]):
            row_marks[int(cl)].append(int(r))

    for offset, c in enumerate(range(int(ic0), int(ic1) + 1)):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct in move_colors and ct != int(expected_top[int(offset)]):
            col_marks[int(ct)].append(int(c))
            top_anomaly_cols.add(int(c))
        if cb in move_colors and cb != int(expected_bottom[int(offset)]):
            col_marks[int(cb)].append(int(c))
            bottom_anomaly_cols.add(int(c))

    return row_marks, col_marks, top_anomaly_cols, bottom_anomaly_cols


def _targets_from_marks(
    row_marks: dict[int, list[int]],
    col_marks: dict[int, list[int]],
) -> dict[int, list[tuple[int, int]]]:
    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        cols = sorted(set(map(int, col_marks[int(col)])))
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in cols}
        if pts:
            targets[int(col)] = sorted(pts)
    return targets


def _solve(grid: np.ndarray, *, top_to_bottom: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1, divider_mask = _find_scene_structure(g)
    sprites = _extract_sprites(g, bg=int(bg), fc0=int(fc0), divider_mask=divider_mask)
    move_colors = {int(s["color"]) for s in sprites}

    row_marks, col_marks, _top_cols, _bottom_cols = _collect_marker_data(
        g,
        ir0=int(ir0),
        ic0=int(ic0),
        ir1=int(ir1),
        ic1=int(ic1),
        fr0=int(fr0),
        fc0=int(fc0),
        fr1=int(fr1),
        fc1=int(fc1),
        move_colors=move_colors,
    )
    targets = _targets_from_marks(row_marks, col_marks)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    if top_to_bottom:
        movers.sort(key=lambda s: (((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))
    else:
        movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the fused variant:
    - detect stripe-break row marks on mirrored left/right edges,
    - detect stripe-break column marks independently on top and bottom edges,
    - stamp movers from top to bottom so lower input sprites overwrite higher ones.
    """
    return _solve(grid, top_to_bottom=True)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Generator-side audit metrics for the three fused mechanisms."""
    g = np.array(grid, dtype=int)
    bg, ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1, divider_mask = _find_scene_structure(g)
    sprites = _extract_sprites(g, bg=int(bg), fc0=int(fc0), divider_mask=divider_mask)
    move_colors = {int(s["color"]) for s in sprites}
    row_marks, col_marks, top_cols, bottom_cols = _collect_marker_data(
        g,
        ir0=int(ir0),
        ic0=int(ic0),
        ir1=int(ir1),
        ic1=int(ic1),
        fr0=int(fr0),
        fc0=int(fc0),
        fr1=int(fr1),
        fc1=int(fc1),
        move_colors=move_colors,
    )
    targets = _targets_from_marks(row_marks, col_marks)
    new_out = _solve(g, top_to_bottom=True)
    old_out = _solve(g, top_to_bottom=False)
    return {
        "movers_with_targets": int(len(targets)),
        "top_only_cols": int(len(top_cols - bottom_cols)),
        "bottom_only_cols": int(len(bottom_cols - top_cols)),
        "shared_cols": int(len(top_cols & bottom_cols)),
        "order_delta": int(np.count_nonzero(new_out != old_out)),
    }


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick exactly `n` spaced values if possible."""
    pool = list(map(int, candidates))
    if not pool or n < 1:
        return None
    if n == 1:
        return [int(random.choice(pool))]
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == int(n):
                return sorted(picked)
    return None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bbox."""
    sh = min(random.choice([3, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
        )
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            pixels.add((nr, nc))
            r, c = nr, nc
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for pr, pc in pixels:
        sprite[int(pr - min_r), int(pc - min_c)] = int(color)
    return sprite


def _base_stripe_color(colors: list[int], pos: int, phase: int) -> int:
    return int(colors[(int(pos) + int(phase)) % 2])


def _sprite_rel_pixels(sprite: np.ndarray) -> set[tuple[int, int]]:
    center_r = (int(sprite.shape[0]) - 1) // 2
    center_c = (int(sprite.shape[1]) - 1) // 2
    return {(int(r) - center_r, int(c) - center_c) for r, c in np.argwhere(sprite != 0)}


def _translated_pixels(rel_pixels: set[tuple[int, int]], center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = map(int, center)
    return {(int(cr + dr), int(cc + dc)) for dr, dc in rel_pixels}


def _center_ranges(sprite: np.ndarray, ir0: int, ir1: int, ic0: int, ic1: int) -> tuple[list[int], list[int]] | None:
    sh, sw = map(int, sprite.shape)
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    if min_row > max_row or min_col > max_col:
        return None
    return list(range(min_row, max_row + 1)), list(range(min_col, max_col + 1))


def _find_overlap_pair_assignment(
    sprite_a: np.ndarray,
    sprite_b: np.ndarray,
    rows_a: list[int],
    cols_a: list[int],
    rows_b: list[int],
    cols_b: list[int],
) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    """
    Pick one target center for each mover such that their stamped pixels overlap.

    Distinct row and column markers are kept to preserve the original "colors do not share the
    same marker slot" cleanliness.
    """
    rel_a = _sprite_rel_pixels(sprite_a)
    rel_b = _sprite_rel_pixels(sprite_b)
    row_gap_limit = max(int(sprite_a.shape[0]), int(sprite_b.shape[0])) - 1
    col_gap_limit = max(int(sprite_a.shape[1]), int(sprite_b.shape[1])) - 1
    candidates: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []

    for ra in rows_a:
        for rb in rows_b:
            if int(ra) == int(rb) or abs(int(ra) - int(rb)) > int(row_gap_limit):
                continue
            for ca in cols_a:
                pix_a = _translated_pixels(rel_a, (int(ra), int(ca)))
                for cb in cols_b:
                    if int(ca) == int(cb) or abs(int(ca) - int(cb)) > int(col_gap_limit):
                        continue
                    pix_b = _translated_pixels(rel_b, (int(rb), int(cb)))
                    overlap = int(len(pix_a & pix_b))
                    if overlap >= 2:
                        gap = abs(int(ra) - int(rb)) + abs(int(ca) - int(cb))
                        candidates.append((int(overlap), int(gap), (int(ra), int(ca)), (int(rb), int(cb))))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_overlap = int(candidates[0][0])
    shortlist = [item for item in candidates if int(item[0]) == best_overlap][:12]
    overlap, _gap, center_a, center_b = random.choice(shortlist)
    return center_a, center_b, int(overlap)


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    h = interior_h + 2
    if not 10 <= h <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    w = sprite_area_w + 1 + frame_w
    if not 14 <= w <= 20:
        return None

    div_col = int(sprite_area_w)
    fc0, fc1 = int(sprite_area_w + 1), int(w - 1)
    ir0, ic0 = 1, int(fc0 + 1)
    ir1, ic1 = int(h - 2), int(fc1 - 1)

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = 2
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    mover_colors = [int(c) for c in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(c) for c in all_colors[idx : idx + n_distractors]]

    sprites: list[np.ndarray] = []
    center_ranges: list[tuple[list[int], list[int]]] = []
    for mc in mover_colors:
        sprite = _make_sprite(mc, 5, sprite_area_w)
        if sprite is None:
            return None
        ranges = _center_ranges(sprite, ir0, ir1, ic0, ic1)
        if ranges is None:
            return None
        sprites.append(sprite)
        center_ranges.append(ranges)

    row_phase = random.randint(0, 1)
    col_phase = random.randint(0, 1)

    top_idx = random.randint(0, 1)
    bottom_idx = 1 - top_idx

    top_color = int(mover_colors[top_idx])
    bottom_color = int(mover_colors[bottom_idx])

    row_candidates: dict[int, list[int]] = {}
    top_col_candidates: dict[int, list[int]] = {}
    bottom_col_candidates: dict[int, list[int]] = {}
    for idx2, (mc, sprite, ranges) in enumerate(zip(mover_colors, sprites, center_ranges)):
        rows, cols = ranges
        row_candidates[int(mc)] = [
            int(r)
            for r in rows
            if _base_stripe_color(mover_colors, int(r), row_phase) != int(mc)
        ]
        col_candidates = [
            int(c)
            for c in cols
            if _base_stripe_color(mover_colors, int(c - fc0), col_phase) != int(mc)
        ]
        if idx2 == top_idx:
            top_col_candidates[int(mc)] = col_candidates
        else:
            bottom_col_candidates[int(mc)] = col_candidates
        if not row_candidates[int(mc)] or not col_candidates:
            return None

    pair = _find_overlap_pair_assignment(
        sprites[top_idx],
        sprites[bottom_idx],
        row_candidates[int(top_color)],
        top_col_candidates[int(top_color)],
        row_candidates[int(bottom_color)],
        bottom_col_candidates[int(bottom_color)],
    )
    if pair is None:
        return None
    top_center, bottom_center, overlap_pixels = pair
    if overlap_pixels < 2:
        return None

    mover_rows: dict[int, list[int]] = {
        int(top_color): [int(top_center[0])],
        int(bottom_color): [int(bottom_center[0])],
    }
    mover_top_cols: dict[int, list[int]] = {
        int(top_color): [int(top_center[1])],
        int(bottom_color): [],
    }
    mover_bottom_cols: dict[int, list[int]] = {
        int(top_color): [],
        int(bottom_color): [int(bottom_center[1])],
    }

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.array([_base_stripe_color(mover_colors, i, col_phase) for i in range(frame_w)], dtype=int)
    bottom_edge = np.array([_base_stripe_color(mover_colors, i, col_phase) for i in range(frame_w)], dtype=int)
    for mc in mover_colors:
        for cc in mover_top_cols[int(mc)]:
            top_edge[int(cc - fc0)] = int(mc)
        for cc in mover_bottom_cols[int(mc)]:
            bottom_edge[int(cc - fc0)] = int(mc)
    inp[0, int(fc0) : int(fc0) + frame_w] = top_edge
    inp[int(h - 1), int(fc0) : int(fc0) + frame_w] = bottom_edge

    for r in range(h):
        val = _base_stripe_color(mover_colors, int(r), row_phase)
        for mc in mover_colors:
            if int(r) in mover_rows[int(mc)]:
                val = int(mc)
                break
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(int(mc), sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(int(dc_color), min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((int(dc_color), ds))

    for _color, sprite in all_items:
        sh, sw = map(int, sprite.shape)
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sprite[int(dr), int(dc)]) != bg:
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sprite[int(dr), int(dc)])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
        metrics = _fusion_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, output):
        return None
    if int(metrics["movers_with_targets"]) != 2:
        return None
    if int(metrics["top_only_cols"]) < 1 or int(metrics["bottom_only_cols"]) < 1:
        return None
    if int(metrics["order_delta"]) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
