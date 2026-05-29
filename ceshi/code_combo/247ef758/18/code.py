# Combo variant of 247ef758 (index 18)
# combo_spec: D4/1+D7/0+D11/0
# Name: 旋转印刻+逆序覆盖+对角目标

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    The scene backbone stays the same: left sprite strip | solid divider column | framed area.
#    Original task:
#      - a mover color collects all matching row markers and all matching column markers;
#      - targets are the full Cartesian product;
#      - each copy keeps its original orientation;
#      - lower sprites stamp first and upper sprites cover later.
#    Combo behavior:
#      - targets are only the sorted row/col diagonal pairs;
#      - each chosen target rotates the sprite by the nearest frame side
#        (top=0, left=1, bottom=2, right=3 in np.rot90 convention);
#      - movers stamp from top to bottom, so lower sprites cover upper ones.
#    Small trace:
#      - suppose color A has rows [2, 8] and cols [6, 12].
#      - original targets: (2,6),(2,12),(8,6),(8,12), all unrotated.
#      - combo targets: only (2,6) and (8,12).
#      - if (2,6) is closest to the top edge and (8,12) to the right edge,
#        the same left sprite appears once at 0 degrees and once at 270 degrees.
#      - if color B sits lower in the left strip and overlaps A at (8,12),
#        combo order makes B win there, while the source task order would make A win.
#
# b) Complexity calibration
#    The generator must not allow the fusion to collapse into any single-dimension variant.
#    It therefore forces:
#      - one special mover with exactly two row marks and two column marks, so D11 is visible;
#      - that mover's two diagonal targets to have different nearest-side rotations, so D4 is visible;
#      - another mover to overlap one of those diagonal targets, so D7 changes the final colors.
#    This keeps the object vocabulary small but gives repeated support for the fused rule.
#
# c) Implicit assumption review
#    Original assumptions broken by the combo:
#      - `row x col` Cartesian expansion is no longer valid; target construction must zip rows/cols.
#      - plain translation of source pixels is no longer valid; each target may need a different
#        rotated bbox and therefore a different anchor.
#      - bottom-to-top overwrite is no longer valid; sort direction must be reversed.
#    New solver assumptions introduced here:
#      - diagonal pairing is interpreted by ascending scan order on rows and cols;
#      - rotation ties are rare and human-hostile, so the generator avoids them, while the solver
#        keeps a deterministic fallback priority top < left < bottom < right;
#      - the overlap-order signal relies on distinct mover vertical positions, so the generator
#        stacks movers without equal center rows.
#
# d) Placement legality
#    Generator-side legality rules:
#      - centers are chosen from ranges that can contain any 0/90/180/270-degree rotation fully
#        inside the frame interior, using `span=max(h,w)`;
#      - row/col markers are never reused across colors, keeping target matching unambiguous;
#      - left-strip sprites are 8-connected single-color components and never touch the divider;
#      - extra movers/distractors are placed without exceeding the grid height.
#
# e) Visual observability
#    Human-solvable evidence per sample:
#      - the special mover uses an asymmetric template whose four rotations are all distinct;
#      - its 2x2 markers visually suggest four Cartesian sites, but only two diagonal sites are used;
#      - a second mover visibly covers part of one rotated copy, making the top-to-bottom order
#        inferable from the output alone.
#
# f) Visual cleanliness
#    Only one deliberate overlap pair is required. Other targets are kept separated by marker reuse
#    bans and by metric-based rejection of degenerate layouts. The generator prefers small blobs and
#    complete in-frame placements rather than relying on heavy clipping.
#
# g) Color inferability
#    Colors still only indicate which left sprite matches which frame markers. Rotation and ordering
#    depend purely on visible geometry, never on color ids or hidden metadata.

from collections import defaultdict
import itertools
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
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
        if bool(
            (rs == 0).any()
            or (rs == int(h - 1)).any()
            or (cs == 0).any()
            or (cs == int(w - 1)).any()
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
    return np.array(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=bool)


def _rotation_signatures(
    mask: np.ndarray,
) -> set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    sigs: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    for krot in range(4):
        rot = _crop_bool(np.rot90(mask.astype(bool), int(krot)))
        pts = tuple(tuple(map(int, p)) for p in np.argwhere(rot))
        sigs.add(((int(rot.shape[0]), int(rot.shape[1])), pts))
    return sigs


def _nearest_side_code(
    tr: int, tc: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> int:
    """
    Rotation code using deterministic priority for ties.

    Priority only matters as a fallback; the generator rejects tied targets.
    """
    dists = [
        (int(tr - fr0), 0),   # top
        (int(tc - fc0), 1),   # left
        (int(fr1 - tr), 2),   # bottom
        (int(fc1 - tc), 3),   # right
    ]
    dists.sort(key=lambda item: (int(item[0]), int(item[1])))
    return int(dists[0][1])


def _nearest_side_code_strict(
    tr: int, tc: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> int | None:
    dists = {
        0: int(tr - fr0),
        1: int(tc - fc0),
        2: int(fr1 - tr),
        3: int(fc1 - tc),
    }
    best = min(dists.values())
    codes = [int(code) for code, dist in dists.items() if int(dist) == int(best)]
    if len(codes) != 1:
        return None
    return int(codes[0])


def _rotated_mask(mask: np.ndarray, krot: int) -> np.ndarray:
    return _crop_bool(np.rot90(mask.astype(bool), int(krot)))


def _placed_pixels(mask: np.ndarray, center: tuple[int, int], krot: int) -> set[tuple[int, int]]:
    rot = _rotated_mask(mask, int(krot))
    anchor_r = int((rot.shape[0] - 1) // 2)
    anchor_c = int((rot.shape[1] - 1) // 2)
    tr, tc = map(int, center)
    return {
        (int(tr) + int(rr) - int(anchor_r), int(tc) + int(cc) - int(anchor_c))
        for rr, cc in np.argwhere(rot)
    }


def _valid_centers(lo: int, hi: int, span: int) -> list[int]:
    """Centers that keep any rotation of maximal extent `span` inside [lo, hi]."""
    anchor = int((int(span) - 1) // 2)
    tail = int(span) - 1 - int(anchor)
    start = int(lo) + int(anchor)
    end = int(hi) - int(tail)
    if start > end:
        return []
    return list(range(int(start), int(end) + 1))


def _make_asymmetric_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """
    Build a single-color sprite whose four rotations are all distinguishable.
    """
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


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Fallback sprite generator, used only for distractors."""
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
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
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


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
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

    return {
        "g": g,
        "bg": int(bg),
        "ir0": int(ir0),
        "ic0": int(ic0),
        "ir1": int(ir1),
        "ic1": int(ic1),
        "fr0": int(fr0),
        "fc0": int(fc0),
        "fr1": int(fr1),
        "fc1": int(fc1),
        "sprites": sprites,
        "row_marks": row_marks,
        "col_marks": col_marks,
    }


def _build_targets(
    row_marks: dict[int, list[int]],
    col_marks: dict[int, list[int]],
    *,
    diagonal_only: bool,
) -> dict[int, list[tuple[int, int]]]:
    targets: dict[int, list[tuple[int, int]]] = {}
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        rows = sorted(map(int, row_marks[int(col)]))
        cols = sorted(map(int, col_marks[int(col)]))
        if diagonal_only:
            pts = [(int(r), int(c)) for r, c in zip(rows, cols)]
        else:
            pts = sorted({(int(r), int(c)) for r in rows for c in cols})
        if pts:
            targets[int(col)] = pts
    return targets


def _render(scene: dict[str, object], *, diagonal_only: bool, rotate: bool, top_to_bottom: bool) -> np.ndarray:
    g = np.array(scene["g"], dtype=int)
    out = g.copy()
    bg = int(scene["bg"])
    ir0 = int(scene["ir0"])
    ic0 = int(scene["ic0"])
    ir1 = int(scene["ir1"])
    ic1 = int(scene["ic1"])
    fr0 = int(scene["fr0"])
    fc0 = int(scene["fc0"])
    fr1 = int(scene["fr1"])
    fc1 = int(scene["fc1"])
    sprites = list(scene["sprites"])

    targets = _build_targets(scene["row_marks"], scene["col_marks"], diagonal_only=bool(diagonal_only))
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(np.array(s["mask"], dtype=bool)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    if top_to_bottom:
        movers.sort(key=lambda s: (((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))
    else:
        movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        base_mask = np.array(s["mask"], dtype=bool)
        for tr, tc in targets[int(col)]:
            krot = _nearest_side_code(int(tr), int(tc), int(fr0), int(fc0), int(fr1), int(fc1)) if rotate else 0
            rot = _rotated_mask(base_mask, int(krot))
            anchor_r = int((rot.shape[0] - 1) // 2)
            anchor_c = int((rot.shape[1] - 1) // 2)
            for rr, cc in np.argwhere(rot):
                gr = int(tr) + int(rr) - int(anchor_r)
                gc = int(tc) + int(cc) - int(anchor_c)
                if int(ir0) <= gr <= int(ir1) and int(ic0) <= gc <= int(ic1):
                    out[int(gr), int(gc)] = int(col)
    return out


def _solve(
    grid: np.ndarray,
    *,
    diagonal_only: bool,
    rotate: bool,
    top_to_bottom: bool,
) -> np.ndarray:
    scene = _extract_scene(grid)
    return _render(scene, diagonal_only=bool(diagonal_only), rotate=bool(rotate), top_to_bottom=bool(top_to_bottom))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo solver for D4/1 + D7/0 + D11/0.

    - Use only diagonal row/column pairings per mover color.
    - Rotate each stamped copy by the nearest frame side.
    - Stamp movers from top to bottom so lower movers overwrite higher ones.
    """
    return _solve(grid, diagonal_only=True, rotate=True, top_to_bottom=True)


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    combo = _solve(grid, diagonal_only=True, rotate=True, top_to_bottom=True)
    no_rotation = _solve(grid, diagonal_only=True, rotate=False, top_to_bottom=True)
    old_order = _solve(grid, diagonal_only=True, rotate=True, top_to_bottom=False)
    cartesian = _solve(grid, diagonal_only=False, rotate=True, top_to_bottom=True)

    scene = _extract_scene(grid)
    row_marks = scene["row_marks"]
    col_marks = scene["col_marks"]
    fr0 = int(scene["fr0"])
    fc0 = int(scene["fc0"])
    fr1 = int(scene["fr1"])
    fc1 = int(scene["fc1"])

    diagonal_pairs = 0
    extra_cartesian_targets = 0
    multi_rotation_colors = 0
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        rows = sorted(map(int, row_marks[int(col)]))
        cols = sorted(map(int, col_marks[int(col)]))
        diagonal_pairs += min(len(rows), len(cols))
        extra_cartesian_targets += max(0, len(rows) * len(cols) - min(len(rows), len(cols)))
        diag_codes = {
            _nearest_side_code_strict(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1))
            for r, c in zip(rows, cols)
        }
        if None not in diag_codes and len(diag_codes) >= 2:
            multi_rotation_colors += 1

    return {
        "rotation_delta": int(np.count_nonzero(combo != no_rotation)),
        "order_delta": int(np.count_nonzero(combo != old_order)),
        "target_delta": int(np.count_nonzero(combo != cartesian)),
        "extra_cartesian_targets": int(extra_cartesian_targets),
        "diagonal_pairs": int(diagonal_pairs),
        "multi_rotation_colors": int(multi_rotation_colors),
    }


def _find_special_mover_targets(
    mask: np.ndarray,
    row_candidates: list[int],
    col_candidates: list[int],
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> tuple[list[int], list[int], list[tuple[int, int]], dict[tuple[int, int], int]] | None:
    span = int(max(mask.shape))
    row_pairs = [
        (int(r0), int(r1))
        for r0, r1 in itertools.combinations(sorted(map(int, row_candidates)), 2)
        if abs(int(r1) - int(r0)) >= int(span)
    ]
    col_pairs = [
        (int(c0), int(c1))
        for c0, c1 in itertools.combinations(sorted(map(int, col_candidates)), 2)
        if abs(int(c1) - int(c0)) >= int(span)
    ]
    random.shuffle(row_pairs)
    random.shuffle(col_pairs)

    candidates = []
    for r0, r1 in row_pairs:
        for c0, c1 in col_pairs:
            diag = [(int(r0), int(c0)), (int(r1), int(c1))]
            full = diag + [(int(r0), int(c1)), (int(r1), int(c0))]
            full_codes = [
                _nearest_side_code_strict(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1))
                for r, c in full
            ]
            if any(code is None for code in full_codes):
                continue
            diag_codes = full_codes[:2]
            if len(set(map(int, diag_codes))) < 2:
                continue
            spread = abs(int(r1) - int(r0)) + abs(int(c1) - int(c0))
            candidates.append((spread, [int(r0), int(r1)], [int(c0), int(c1)], diag, {
                (int(r), int(c)): int(_nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)))
                for r, c in full
            }))

    if not candidates:
        return None

    candidates.sort(key=lambda item: -int(item[0]))
    spread_best = int(candidates[0][0])
    shortlist = [item for item in candidates if int(item[0]) == int(spread_best)][:12]
    _spread, rows, cols, diag, rot_map = random.choice(shortlist)
    return list(rows), list(cols), list(diag), dict(rot_map)


def _find_overlap_target(
    special_mask: np.ndarray,
    special_diag: list[tuple[int, int]],
    special_rot_map: dict[tuple[int, int], int],
    other_mask: np.ndarray,
    row_candidates: list[int],
    col_candidates: list[int],
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> tuple[int, int, int] | None:
    special_sets = {
        (int(r), int(c)): _placed_pixels(
            special_mask,
            (int(r), int(c)),
            int(special_rot_map[(int(r), int(c))]),
        )
        for r, c in special_diag
    }

    candidates = []
    for rb in sorted(map(int, row_candidates)):
        for cb in sorted(map(int, col_candidates)):
            code = _nearest_side_code_strict(int(rb), int(cb), int(fr0), int(fc0), int(fr1), int(fc1))
            if code is None:
                continue
            pix_b = _placed_pixels(other_mask, (int(rb), int(cb)), int(code))
            best_overlap = 0
            for target in special_diag:
                overlap = int(len(pix_b & special_sets[(int(target[0]), int(target[1]))]))
                best_overlap = max(best_overlap, overlap)
            if best_overlap >= 2:
                center_gap = min(
                    abs(int(rb) - int(target[0])) + abs(int(cb) - int(target[1]))
                    for target in special_diag
                )
                candidates.append((best_overlap, -center_gap, int(rb), int(cb), int(code)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-int(item[0]), -int(item[1])))
    best_overlap = int(candidates[0][0])
    shortlist = [item for item in candidates if int(item[0]) == int(best_overlap)][:12]
    _ov, _gap, rb, cb, code = random.choice(shortlist)
    return int(rb), int(cb), int(code)


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
    fr0, fr1 = 0, int(h - 1)

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    divider_color = int(all_colors[0])
    frame_base = int(all_colors[1])
    mover_colors = [int(all_colors[2]), int(all_colors[3])]
    distractor_colors = [int(all_colors[4])] if random.choice([False, False, True]) else []

    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_asymmetric_sprite(int(mc), 5, int(sprite_area_w))
        if sp is None:
            return None
        mover_sprites.append(sp)

    span_a = int(max(mover_sprites[0].shape))
    span_b = int(max(mover_sprites[1].shape))
    row_candidates_a = _valid_centers(int(ir0), int(ir1), int(span_a))
    col_candidates_a = _valid_centers(int(ic0), int(ic1), int(span_a))
    row_candidates_b_full = _valid_centers(int(ir0), int(ir1), int(span_b))
    col_candidates_b_full = _valid_centers(int(ic0), int(ic1), int(span_b))
    if (
        len(row_candidates_a) < 2
        or len(col_candidates_a) < 2
        or not row_candidates_b_full
        or not col_candidates_b_full
    ):
        return None

    special = _find_special_mover_targets(
        mover_sprites[0].astype(bool),
        row_candidates_a,
        col_candidates_a,
        int(fr0),
        int(fc0),
        int(fr1),
        int(fc1),
    )
    if special is None:
        return None
    rows_a, cols_a, diag_a, rot_map_a = special

    used_rows = set(map(int, rows_a))
    used_cols = set(map(int, cols_a))
    row_candidates_b = [int(r) for r in row_candidates_b_full if int(r) not in used_rows]
    col_candidates_b = [int(c) for c in col_candidates_b_full if int(c) not in used_cols]
    if not row_candidates_b or not col_candidates_b:
        return None

    overlap_target = _find_overlap_target(
        mover_sprites[0].astype(bool),
        diag_a,
        rot_map_a,
        mover_sprites[1].astype(bool),
        row_candidates_b,
        col_candidates_b,
        int(fr0),
        int(fc0),
        int(fr1),
        int(fc1),
    )
    if overlap_target is None:
        return None
    row_b, col_b, _rot_b = overlap_target

    mover_targets: dict[int, tuple[list[int], list[int]]] = {
        int(mover_colors[0]): (sorted(map(int, rows_a)), sorted(map(int, cols_a))),
        int(mover_colors[1]): ([int(row_b)], [int(col_b)]),
    }

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - int(fc0)] = int(mc)
    inp[0, int(fc0) : int(fc0 + frame_w)] = top_edge
    inp[int(h - 1), int(fc0) : int(fc0 + frame_w)] = top_edge

    row_owner = {int(r): int(mc) for mc, (rmarks, _cmarks) in mover_targets.items() for r in rmarks}
    for r in range(1, int(h - 1)):
        val = int(row_owner.get(int(r), int(frame_base)))
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [
        (int(mover_colors[0]), np.array(mover_sprites[0], dtype=int)),
        (int(mover_colors[1]), np.array(mover_sprites[1], dtype=int)),
    ]
    for dc in distractor_colors:
        ds = _make_sprite(int(dc), min(5, int(h - cur_row)), int(sprite_area_w))
        if ds is None:
            return None
        all_items.append((int(dc), ds))

    for color, sp in all_items:
        sh, sw = map(int, sp.shape)
        if sw > int(sprite_area_w) or cur_row + sh > int(h):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(int(sh)):
            for dc in range(int(sw)):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(color)
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        output = main(inp)
        metrics = _combo_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, output):
        return None
    if metrics["rotation_delta"] < 2:
        return None
    if metrics["order_delta"] < 1:
        return None
    if metrics["target_delta"] < 2:
        return None
    if metrics["extra_cartesian_targets"] < 2:
        return None
    if metrics["multi_rotation_colors"] < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
