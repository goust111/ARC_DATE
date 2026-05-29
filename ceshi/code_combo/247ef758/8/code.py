# Combo variant of 247ef758 (index 8)
# combo_spec: D4/1+D5/0
# Name: 保留源+旋转印刻+轨迹连线

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Backbone stays the same as the original task: left sprite strip | single-color divider
#   column | framed area on the right. A mover color is still identified by matching markers
#   on both left/right frame edges for rows and on both top/bottom frame edges for cols.
# - Original local behavior for one mover:
#   1. find Cartesian-product targets from row_marks x col_marks
#   2. erase the source sprite from the left strip
#   3. translate-stamp the unrotated sprite at every target center
# - Combo local behavior:
#   1. keep the source sprite visible in the left strip
#   2. for every target, draw a same-color discrete straight line from the source center to
#      the target center
#   3. at that target, rotate the sprite according to the nearest frame side:
#      top -> 0, left -> 90 CCW, bottom -> 180, right -> 270
#   4. stamp the rotated sprite centered at the target
# - Small trace example:
#   if a red sprite centered at (2,1) has targets (3,10) and (8,10), the original output
#   removes the left red blob and places two identical copies on the right. The combo output
#   keeps the left red blob, draws two red lines to the two targets, then stamps one copy in
#   top-facing orientation at (3,10) and another in bottom-facing orientation at (8,10).
#   The visible differences are source retention, line trajectories, and target-dependent
#   rotation.
#
# b) Complexity calibration
# - We keep only 2-3 mover colors and at most 1 distractor, so the scene is still readable.
# - Rotation has only 4 discrete outcomes, and all are determined by the same visible spatial
#   rule "nearest frame side", so the freedom stays low enough for ARC inference.
# - Generator forces at least one mover to appear at two targets with different nearest sides,
#   so rotation is not a rare corner case.
#
# c) Implicit assumptions review
# - The original erase loop is invalid because movers must now remain in the source strip.
# - The original pure-translation stamp loop assumes bbox/orientation never changes. Rotation
#   breaks that assumption, so stamping must rotate the boolean mask first and then recenter
#   using the rotated bbox.
# - After adding trails, painter order matters for both line segments and stamped copies. We
#   keep the original bottom-first, top-last order so later/higher movers can overwrite
#   earlier/lower ones consistently.
# - Source preservation adds a new constraint: trail drawing must not damage any existing
#   left-strip sprite pixels, including distractors and other movers. The solver therefore
#   treats all non-bg left-strip cells as protected during line drawing.
#
# d) Placement legality
# - Generator uses asymmetric sprites whose four rotations are distinct, so rotation is
#   visible.
# - Target centers are chosen from positions that can fit the sprite in any rotation using
#   `span = max(h, w)`. This avoids relying on clipping for the main visual signal.
# - Row and column markers remain unique across mover colors, so no two colors share a target
#   center.
#
# e) Visual observability
# - Each mover color now repeats in four places: source sprite, edge markers, trajectory line,
#   and rotated copies. This gives a human multiple aligned cues across 2-4 examples.
# - Rotation depends only on spatial proximity to top/left/bottom/right borders, not on color
#   ids or hidden priorities. The only tie-break is a solver safety fallback; the generator
#   avoids ties altogether.
#
# f) Visual cleanliness
# - We avoid 2x2 target grids for most movers and require spacing between marks, reducing
#   heavy overlap of trails and copies.
# - Most copies fit fully inside the frame interior, so the output stays structured instead of
#   looking like arbitrary clipped blobs.
#
# g) Anti-degeneracy checks
# - Accepted samples must show nontrivial bg->colored gain so trajectories are actually
#   visible.
# - Accepted samples must also differ from a "keep source + trails + unrotated stamps"
#   baseline, ensuring that rotation is not merely latent in the metadata but visible in the
#   final output.
#
# h) New solver assumptions and generator support
# - The solver still assumes a single enclosed frame interior and a single full-height divider
#   column; the generator preserves that original layout.
# - The solver assumes mover sprites are 8-connected monochrome components in the left strip;
#   the generator constructs exactly that.
# - The solver assumes at least one row mark and one col mark for every mover color it
#   processes; the generator builds markers only for actual mover colors and validates
#   `main(generate_input())` on every accepted sample.


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does not touch the grid border.
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
        if bool(
            (rs == 0).any()
            or (rs == int(H - 1)).any()
            or (cs == 0).any()
            or (cs == int(W - 1)).any()
        ):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _discrete_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer grid line including both endpoints."""
    r, c = int(r0), int(c0)
    r1, c1 = int(r1), int(c1)
    pts = [(r, c)]
    dr = abs(int(r1 - r))
    dc = abs(int(c1 - c))
    sr = 0 if r == r1 else (1 if r1 > r else -1)
    sc = 0 if c == c1 else (1 if c1 > c else -1)

    if dc >= dr:
        err = dc // 2
        while c != c1:
            c += sc
            err -= dr
            if err < 0:
                r += sr
                err += dc
            pts.append((int(r), int(c)))
    else:
        err = dr // 2
        while r != r1:
            r += sr
            err -= dc
            if err < 0:
                c += sc
                err += dr
            pts.append((int(r), int(c)))
    return pts


def _nearest_side_code(
    tr: int, tc: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> int | None:
    dists = {
        0: int(tr - fr0),   # top
        1: int(tc - fc0),   # left
        2: int(fr1 - tr),   # bottom
        3: int(fc1 - tc),   # right
    }
    best = min(dists.values())
    best_codes = [int(code) for code, dist in dists.items() if int(dist) == int(best)]
    if len(best_codes) != 1:
        return None
    return int(best_codes[0])


def _rotation_code_with_fallback(tr: int, tc: int, fr0: int, fc0: int, fr1: int, fc1: int) -> int:
    """
    Return the nearest-side rotation code.

    Priority order top/left/bottom/right is only a safety fallback for accidental ties.
    The generator avoids ties, so human-visible samples should never rely on it.
    """
    dists = [
        (int(tr - fr0), 0),
        (int(tc - fc0), 1),
        (int(fr1 - tr), 2),
        (int(fc1 - tc), 3),
    ]
    dists.sort(key=lambda item: (int(item[0]), int(item[1])))
    return int(dists[0][1])


def _extract_scene(grid: np.ndarray) -> dict:
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

    left_cols = np.arange(W)[None, :] < int(fc0)
    protected_left = left_cols & (~divider_mask) & (g != int(bg))

    sprites = []
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
                    "mask": np.array(m, dtype=bool),
                }
            )

    move_colors = {int(s["color"]) for s in sprites}

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

    return {
        "g": g,
        "H": int(H),
        "W": int(W),
        "bg": int(bg),
        "ir0": int(ir0),
        "ic0": int(ic0),
        "ir1": int(ir1),
        "ic1": int(ic1),
        "fr0": int(fr0),
        "fc0": int(fc0),
        "fr1": int(fr1),
        "fc1": int(fc1),
        "protected_left": protected_left,
        "sprites": sprites,
        "targets": targets,
    }


def _render_scene(
    scene: dict,
    *,
    rotate_targets: bool,
    keep_sources: bool,
    draw_trails: bool,
) -> np.ndarray:
    g = np.array(scene["g"], dtype=int)
    H, W = int(scene["H"]), int(scene["W"])
    ir0, ic0, ir1, ic1 = int(scene["ir0"]), int(scene["ic0"]), int(scene["ir1"]), int(scene["ic1"])
    fr0, fc0, fr1, fc1 = int(scene["fr0"]), int(scene["fc0"]), int(scene["fr1"]), int(scene["fc1"])
    protected_left = np.array(scene["protected_left"], dtype=bool)
    targets = dict(scene["targets"])

    out = g.copy()
    movers = [dict(s) for s in scene["sprites"] if int(s["color"]) in targets]

    if not keep_sources:
        for s in movers:
            r0, c0 = int(s["r0"]), int(s["c0"])
            for rr, cc in np.argwhere(np.array(s["mask"], dtype=bool)):
                out[int(r0) + int(rr), int(c0) + int(cc)] = int(scene["bg"])

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        base_mask = np.array(s["mask"], dtype=bool)
        rot_masks = {0: _crop_bool(base_mask)}
        if rotate_targets:
            for code in (1, 2, 3):
                rot_masks[int(code)] = _crop_bool(np.rot90(base_mask, int(code)))

        for tr, tc in targets[int(col)]:
            if draw_trails:
                for rr, cc2 in _discrete_line(int(cr), int(cc), int(tr), int(tc)):
                    if not (0 <= int(rr) < int(H) and 0 <= int(cc2) < int(W)):
                        continue
                    if bool(protected_left[int(rr), int(cc2)]):
                        continue
                    out[int(rr), int(cc2)] = int(col)

            krot = 0
            if rotate_targets:
                krot = _rotation_code_with_fallback(int(tr), int(tc), int(fr0), int(fc0), int(fr1), int(fc1))
            stamp_mask = rot_masks[int(krot)]
            anchor_r = int((stamp_mask.shape[0] - 1) // 2)
            anchor_c = int((stamp_mask.shape[1] - 1) // 2)

            for rr, cc2 in np.argwhere(stamp_mask):
                gr = int(tr) + int(rr) - int(anchor_r)
                gc = int(tc) + int(cc2) - int(anchor_c)
                if int(ir0) <= gr <= int(ir1) and int(ic0) <= gc <= int(ic1):
                    out[int(gr), int(gc)] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve mover sprites on the left, connect each mover center to all marker-derived targets
    with same-color lines, and stamp rotated copies at those targets inside the frame.
    """
    scene = _extract_scene(grid)
    return _render_scene(scene, rotate_targets=True, keep_sources=True, draw_trails=True)


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return picked
    return picked if picked else None


def _rotation_signatures(mask: np.ndarray) -> set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    sigs: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    for krot in range(4):
        rot = _crop_bool(np.rot90(mask.astype(bool), int(krot)))
        pts = tuple(tuple(map(int, p)) for p in np.argwhere(rot))
        sigs.add(((int(rot.shape[0]), int(rot.shape[1])), pts))
    return sigs


def _make_asymmetric_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Build a single-color sprite whose four rotations are all distinguishable."""
    templates = [
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    ]

    candidates: list[np.ndarray] = []
    for tpl in templates:
        if len(_rotation_signatures(tpl.astype(bool))) != 4:
            continue
        for krot in range(4):
            rot = _crop_bool(np.rot90(tpl.astype(bool), int(krot)))
            if int(rot.shape[0]) <= int(max_h) and int(rot.shape[1]) <= int(max_w):
                arr = np.zeros(rot.shape, dtype=int)
                arr[rot] = int(color)
                candidates.append(arr)
    if not candidates:
        return None
    return np.array(random.choice(candidates), dtype=int)


def _valid_centers(lo: int, hi: int, span: int) -> list[int]:
    """Centers that keep a rotated sprite with max extent `span` fully inside [lo, hi]."""
    anchor = int((int(span) - 1) // 2)
    tail = int(span) - 1 - int(anchor)
    start = int(lo) + int(anchor)
    end = int(hi) - int(tail)
    if start > end:
        return []
    return list(range(int(start), int(end) + 1))


def _pick_special_marks(
    mode: str,
    row_candidates: list[int],
    col_candidates: list[int],
    span: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> tuple[list[int], list[int]] | None:
    rows = list(row_candidates)
    cols = list(col_candidates)
    random.shuffle(rows)
    random.shuffle(cols)

    if mode == "vertical":
        for c in cols:
            top_rows = [
                r
                for r in rows
                if _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)) == 0
            ]
            bot_rows = [
                r
                for r in rows
                if _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)) == 2
            ]
            random.shuffle(top_rows)
            random.shuffle(bot_rows)
            for rt in top_rows:
                for rb in bot_rows:
                    if abs(int(rt) - int(rb)) >= int(span):
                        return sorted([int(rt), int(rb)]), [int(c)]
        return None

    if mode == "horizontal":
        for r in rows:
            left_cols = [
                c
                for c in cols
                if _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)) == 1
            ]
            right_cols = [
                c
                for c in cols
                if _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)) == 3
            ]
            random.shuffle(left_cols)
            random.shuffle(right_cols)
            for cl in left_cols:
                for cr in right_cols:
                    if abs(int(cl) - int(cr)) >= int(span):
                        return [int(r)], sorted([int(cl), int(cr)])
        return None

    return None


def _pick_generic_marks(
    row_candidates: list[int],
    col_candidates: list[int],
    span: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
    options: list[tuple[int, int]],
) -> tuple[list[int], list[int]] | None:
    for row_n, col_n in options:
        for _ in range(60):
            rows = _pick_spaced(list(row_candidates), int(row_n), int(span))
            cols = _pick_spaced(list(col_candidates), int(col_n), int(span))
            if not rows or not cols or len(rows) != int(row_n) or len(cols) != int(col_n):
                continue
            sides = [
                _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1))
                for r in rows
                for c in cols
            ]
            if any(side is None for side in sides):
                continue
            return sorted(map(int, rows)), sorted(map(int, cols))
    return None


def _assign_targets(
    sprites: list[np.ndarray],
    mover_colors: list[int],
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> dict[int, tuple[list[int], list[int]]] | None:
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()

    special_mode = random.choice(["vertical", "horizontal"])

    for idx, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        span = int(max(sp.shape))
        fit_rows = [r for r in _valid_centers(int(ir0), int(ir1), int(span)) if r not in occupied_rows]
        fit_cols = [c for c in _valid_centers(int(ic0), int(ic1), int(span)) if c not in occupied_cols]
        if not fit_rows or not fit_cols:
            return None

        if idx == 0:
            picked = _pick_special_marks(
                str(special_mode),
                fit_rows,
                fit_cols,
                int(span),
                int(fr0),
                int(fc0),
                int(fr1),
                int(fc1),
            )
            if picked is None:
                alt_mode = "horizontal" if special_mode == "vertical" else "vertical"
                picked = _pick_special_marks(
                    str(alt_mode),
                    fit_rows,
                    fit_cols,
                    int(span),
                    int(fr0),
                    int(fc0),
                    int(fr1),
                    int(fc1),
                )
            if picked is None:
                return None
        else:
            weighted_options = [(1, 1), (1, 1), (2, 1), (1, 2)]
            random.shuffle(weighted_options)
            picked = _pick_generic_marks(
                fit_rows,
                fit_cols,
                int(span),
                int(fr0),
                int(fc0),
                int(fr1),
                int(fc1),
                weighted_options,
            )
            if picked is None:
                return None

        rows, cols = picked
        mover_targets[int(mc)] = (list(map(int, rows)), list(map(int, cols)))
        occupied_rows.update(map(int, rows))
        occupied_cols.update(map(int, cols))

    special_color = int(mover_colors[0])
    s_rows, s_cols = mover_targets[int(special_color)]
    special_rotations = {
        _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1))
        for r in s_rows
        for c in s_cols
    }
    if None in special_rotations or len(special_rotations) < 2:
        return None

    return mover_targets


def _target_count_ok(targets_by_color: dict[int, tuple[list[int], list[int]]]) -> bool:
    total = sum(len(rows) * len(cols) for rows, cols in targets_by_color.values())
    return total >= 3


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

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_asymmetric_sprite(int(mc), 5, int(sprite_area_w))
        if s is None:
            return None
        sprites.append(s)

    mover_targets = _assign_targets(
        sprites,
        mover_colors,
        int(ir0),
        int(ic0),
        int(ir1),
        int(ic1),
        int(fr0),
        int(fc0),
        int(fr1),
        int(fc1),
    )
    if mover_targets is None:
        return None
    if not _target_count_ok(mover_targets):
        return None

    inp = np.full((H, W), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - int(fc0)] = int(mc)
    inp[0, int(fc0) : int(fc0 + frame_w)] = top_edge
    inp[H - 1, int(fc0) : int(fc0 + frame_w)] = top_edge

    for r in range(1, H - 1):
        val = int(frame_base)
        for mc, (rmarks, _cmarks) in mover_targets.items():
            if int(r) in set(map(int, rmarks)):
                val = int(mc)
                break
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    all_items: list[tuple[int, np.ndarray]] = [(int(mc), sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_asymmetric_sprite(int(dc_color), 4, int(sprite_area_w))
        if ds is None:
            return None
        all_items.append((int(dc_color), ds))

    cur_row = 0
    for _color, sp in all_items:
        sh, sw = map(int, sp.shape)
        if sw > int(sprite_area_w) or cur_row + sh > H:
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
        scene = _extract_scene(inp)
        output = _render_scene(scene, rotate_targets=True, keep_sources=True, draw_trails=True)
        no_rotation_output = _render_scene(scene, rotate_targets=False, keep_sources=True, draw_trails=True)
    except Exception:
        return None

    if np.array_equal(inp, output):
        return None

    trail_like_gain = int(np.count_nonzero((inp == int(bg)) & (output != int(bg))))
    if trail_like_gain < 6:
        return None

    visible_rotation_delta = int(np.count_nonzero(output != no_rotation_output))
    if visible_rotation_delta < 4:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
