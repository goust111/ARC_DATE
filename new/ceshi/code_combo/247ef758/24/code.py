# Combo variant of 247ef758 (index 24)
# combo_spec: D3/0+D4/1
# Name: 内部标记+旋转印刻
#
# Design analysis
# a) Concrete behavior trace
#    - Base scaffold stays the same as the source task: left sprite strip | full-height divider | right rectangular frame.
#    - D3/0 removes all colored edge markers. The frame border becomes uniformly `frame_base`, and marker evidence moves
#      inside the frame onto one inner guide row and one inner guide column. We use single-pixel dots rather than tiny
#      crosses because the fusion already adds rotation complexity; dots keep the scene legible while still satisfying
#      the "small cross or dot" requirement.
#    - D4/1 then acts on the decoded Cartesian-product targets: after a target center `(tr, tc)` is inferred from the
#      internal row-dot and column-dot of the same color, the sprite is rotated according to which frame edge is nearest
#      to that target: top->0, left->1, bottom->2, right->3 under `np.rot90`.
#    - Mini trace:
#      Suppose guide dots of color 4 sit at `(row=3, guide_col)` and `(guide_row, col=9)`, so the target is `(3, 9)`.
#      Original task would have read color 4 from opposite frame edges and stamped the sprite unrotated at `(3, 9)`.
#      This fusion instead reads the interior dots, keeps the frame border monochrome, and if `(3, 9)` is closest to the
#      top edge, stamps the sprite at rotation 0. Another target of the same color at `(8, 9)` may be closest to the
#      bottom edge and therefore uses rotation 2. The same color can therefore appear multiple times with different
#      orientations.
#
# b) Complexity calibration
#    - Keep 2-3 movers and 0-1 distractors from the source task so the puzzle remains recognizably the same family.
#    - Keep 1-2 row marks and 1-2 column marks per mover, but force at least one mover to realize two distinct nearest
#      edges. That gives repeated evidence for the rotation rule and prevents degeneration to "everything keeps its
#      original orientation".
#    - Sprites are drawn from asymmetric templates whose four rotations are visually different; otherwise the D4 signal
#      would be invisible even if the code rotated correctly.
#
# c) Implicit assumptions reviewed against `main()`
#    - `largest_enclosed_component_bbox(g == bg)` assumes the framed interior is still the largest enclosed background
#      region. Internal single-pixel dots remove only isolated cells, so the enclosing bbox is preserved.
#    - Divider detection still assumes one nearly full-height uniform non-bg column to the left of the frame. Generator
#      preserves exactly one such column.
#    - Sprite extraction still assumes left-strip objects are monochrome 8-connected components. Generator preserves that.
#    - The original row/column parser assumed informative colors live on the border. That is false here, so target
#      decoding must read `g[r, guide_col]` and `g[guide_row, c]` instead of comparing opposite edges.
#    - The original stamping logic assumed pure translation. That is false here because each target may use a different
#      rotation, so the solver must rotate first and then center the rotated bbox on the target.
#    - The rotation rule introduces a potential tie when a target is equally close to two edges. Solver keeps a stable
#      fallback order `top < left < bottom < right`, but the generator actively avoids ties so humans do not need to
#      infer a hidden tie-break.
#
# d) Placement legality
#    - Guide markers occupy only one inner row and one inner column, and target centers are chosen from the sub-rectangle
#      strictly below and to the right of those guide lanes. This guarantees stamped sprites do not overwrite the marker
#      dots and keeps input-output correspondence visible.
#    - Target centers are chosen from positions that fit the maximum rotated span `max(h, w)` entirely inside the frame.
#      The solver still clips as a safety net, but the generator aims for full placement.
#    - Row and column markers are globally unique across mover colors, so different sprites never share the same target.
#      Within one mover, multiple targets are spaced by at least the sprite span to avoid self-overlap.
#
# e) Visual observability
#    - Humans can see two fused differences immediately: colored evidence is no longer on the frame edge, and output
#      copies can rotate differently depending on where they land inside the same frame.
#    - The rule does not rely on color ids, generation order, or hidden metadata. Colors only link a left sprite to the
#      matching interior row/column dots; rotation depends only on visible geometry.
#    - At least one mover in each generated sample has two targets with different nearest edges, so the rotation rule is
#      observable from a single sample instead of needing a lucky collection of examples.
#
# f) Visual cleanliness
#    - The frame border is uniformly one color and the informative dots occupy only a thin L-shaped guide structure near
#      the top-left of the frame, so the input is less cluttered than if arbitrary internal crosses were scattered across
#      the whole box.
#    - The interior stays otherwise empty before stamping, so output sprites remain easy to segment visually.
#
# g) New solver assumptions and how the generator guarantees them
#    - Assumption: guide row is exactly `ir0 + 1` and guide column is exactly `ic0 + 1`.
#      Guarantee: generator always places internal markers on those lanes only.
#    - Assumption: only mover colors appear on the guide lanes.
#      Guarantee: generator never writes distractor colors onto those lanes.
#    - Assumption: at least one mover has two distinct rotation codes and those rotations change the rendered output.
#      Guarantee: generator uses an asymmetric special mover and rejects samples where rotated output equals the
#      no-rotation control output.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the border."""
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
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _nearest_side_code(
    tr: int, tc: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> int:
    """
    Rotation code chosen from visible geometry.

    Codes follow `np.rot90`:
    top=0, left=1, bottom=2, right=3.
    The generator avoids ties; the secondary sort is only a deterministic fallback.
    """
    dists = [
        (int(tr - fr0), 0),
        (int(tc - fc0), 1),
        (int(fr1 - tr), 2),
        (int(fc1 - tc), 3),
    ]
    dists.sort(key=lambda item: (int(item[0]), int(item[1])))
    return int(dists[0][1])


def _extract_layout_and_sprites(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, int, int, int, int, int, int, int, int, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    assert guide_row <= int(ir1) and guide_col <= int(ic1)

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
            comp = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": comp,
                }
            )

    return (
        g,
        int(bg),
        int(ir0),
        int(ic0),
        int(ir1),
        int(ic1),
        int(fr0),
        int(fc0),
        int(fr1),
        int(fc1),
        int(guide_row),
        int(guide_col),
        sprites,
    )


def _decode_internal_targets(
    g: np.ndarray,
    move_colors: set[int],
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    guide_row: int,
    guide_col: int,
    bg: int,
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[int]], dict[int, list[int]]]:
    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for r in range(int(ir0), int(ir1) + 1):
        val = int(g[int(r), int(guide_col)])
        if val != int(bg) and val in move_colors:
            row_marks[int(val)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        val = int(g[int(guide_row), int(c)])
        if val != int(bg) and val in move_colors:
            col_marks[int(val)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {
            (int(r), int(c))
            for r in row_marks[int(col)]
            for c in col_marks[int(col)]
        }
        targets[int(col)] = sorted(pts)

    return targets, row_marks, col_marks


def _solve(grid: np.ndarray, *, rotate: bool) -> np.ndarray:
    (
        g,
        bg,
        ir0,
        ic0,
        ir1,
        ic1,
        fr0,
        fc0,
        fr1,
        fc1,
        guide_row,
        guide_col,
        sprites,
    ) = _extract_layout_and_sprites(grid)

    move_colors = {int(s["color"]) for s in sprites}
    targets, _row_marks, _col_marks = _decode_internal_targets(
        g,
        move_colors,
        int(ir0),
        int(ic0),
        int(ir1),
        int(ic1),
        int(guide_row),
        int(guide_col),
        int(bg),
    )

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0 = int(s["r0"])
        c0 = int(s["c0"])
        for rr, cc in np.argwhere(np.array(s["mask"], dtype=bool)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(
        key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"]))
    )

    for s in movers:
        col = int(s["color"])
        base_mask = np.array(s["mask"], dtype=bool)
        for tr, tc in targets[int(col)]:
            krot = (
                _nearest_side_code(
                    int(tr), int(tc), int(fr0), int(fc0), int(fr1), int(fc1)
                )
                if rotate
                else 0
            )
            rot_mask = _crop_bool(np.rot90(base_mask, int(krot)))
            anchor_r = int((rot_mask.shape[0] - 1) // 2)
            anchor_c = int((rot_mask.shape[1] - 1) // 2)

            for rr, cc in np.argwhere(rot_mask):
                gr = int(tr) + int(rr) - int(anchor_r)
                gc = int(tc) + int(cc) - int(anchor_c)
                if int(ir0) <= gr <= int(ir1) and int(ic0) <= gc <= int(ic1):
                    out[gr, gc] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Row and column markers are read from internal guide lanes instead of frame edges.
    - Target centers are the Cartesian product of same-color row dots and column dots.
    - Each stamped copy rotates according to the nearest frame edge.
    """
    return _solve(grid, rotate=True)


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to `n` values from candidates with pairwise spacing at least `min_spacing`."""
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
                return sorted(picked)
    return sorted(picked) if picked else None


def _rotation_signatures(
    mask: np.ndarray,
) -> set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    sigs: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    for krot in range(4):
        rot = _crop_bool(np.rot90(mask.astype(bool), int(krot)))
        pts = tuple(tuple(map(int, p)) for p in np.argwhere(rot))
        sigs.add(((int(rot.shape[0]), int(rot.shape[1])), pts))
    return sigs


def _make_asymmetric_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """
    Build a single-color sprite whose rotations are visually distinguishable.
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


def _valid_centers(lo: int, hi: int, span: int) -> list[int]:
    """
    Centers that keep any rotation of span `span` fully inside the inclusive interval [lo, hi].
    """
    anchor = int((int(span) - 1) // 2)
    tail = int(span) - 1 - int(anchor)
    start = int(lo) + int(anchor)
    end = int(hi) - int(tail)
    if start > end:
        return []
    return list(range(int(start), int(end) + 1))


def _nearest_side_code_unique(
    tr: int, tc: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> int | None:
    dists = {
        0: int(tr - fr0),
        1: int(tc - fc0),
        2: int(fr1 - tr),
        3: int(fc1 - tc),
    }
    best = min(dists.values())
    best_codes = [int(code) for code, dist in dists.items() if int(dist) == int(best)]
    if len(best_codes) != 1:
        return None
    return int(best_codes[0])


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
                if _nearest_side_code_unique(
                    int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)
                )
                == 0
            ]
            bottom_rows = [
                r
                for r in rows
                if _nearest_side_code_unique(
                    int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)
                )
                == 2
            ]
            random.shuffle(top_rows)
            random.shuffle(bottom_rows)
            for rt in top_rows:
                for rb in bottom_rows:
                    if abs(int(rt) - int(rb)) >= int(span):
                        return sorted([int(rt), int(rb)]), [int(c)]
        return None

    if mode == "horizontal":
        for r in rows:
            left_cols = [
                c
                for c in cols
                if _nearest_side_code_unique(
                    int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)
                )
                == 1
            ]
            right_cols = [
                c
                for c in cols
                if _nearest_side_code_unique(
                    int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)
                )
                == 3
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
            if not rows or not cols:
                continue
            if len(rows) != int(row_n) or len(cols) != int(col_n):
                continue
            sides = [
                _nearest_side_code_unique(
                    int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)
                )
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
    guide_row: int,
    guide_col: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> dict[int, tuple[list[int], list[int]]] | None:
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    special_mode = random.choice(["vertical", "horizontal"])

    row_lo = max(int(ir0), int(guide_row) + 1)
    col_lo = max(int(ic0), int(guide_col) + 1)

    for idx, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        span = int(max(sp.shape))
        fit_rows = [
            r
            for r in _valid_centers(int(row_lo), int(ir1), int(span))
            if r not in occupied_rows
        ]
        fit_cols = [
            c
            for c in _valid_centers(int(col_lo), int(ic1), int(span))
            if c not in occupied_cols
        ]
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

    special_rows, special_cols = mover_targets[int(mover_colors[0])]
    special_rotations = {
        _nearest_side_code_unique(
            int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)
        )
        for r in special_rows
        for c in special_cols
    }
    if None in special_rotations or len(special_rotations) < 2:
        return None

    return mover_targets


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
    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    if guide_row > int(ir1) or guide_col > int(ic1):
        return None

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    mover_colors = list(map(int, all_colors[idx : idx + n_movers]))
    idx += n_movers
    distractor_colors = list(map(int, all_colors[idx : idx + n_distractors]))

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_asymmetric_sprite(int(mc), 5, int(sprite_area_w))
        if sp is None:
            return None
        sprites.append(sp)

    mover_targets = _assign_targets(
        sprites,
        mover_colors,
        int(ir0),
        int(ic0),
        int(ir1),
        int(ic1),
        int(guide_row),
        int(guide_col),
        int(fr0),
        int(fc0),
        int(fr1),
        int(fc1),
    )
    if mover_targets is None:
        return None

    inp = np.full((h, w), int(bg), dtype=int)
    inp[:, int(div_col)] = int(divider_color)
    inp[0, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[h - 1, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[:, int(fc0)] = int(frame_base)
    inp[:, int(fc1)] = int(frame_base)

    for mc, (row_marks, col_marks) in mover_targets.items():
        for r in row_marks:
            inp[int(r), int(guide_col)] = int(mc)
        for c in col_marks:
            inp[int(guide_row), int(c)] = int(mc)

    all_items: list[tuple[int, np.ndarray]] = [
        (int(mc), sp) for mc, sp in zip(mover_colors, sprites)
    ]
    for dc_color in distractor_colors:
        ds = _make_asymmetric_sprite(int(dc_color), 4, int(sprite_area_w))
        if ds is None:
            return None
        all_items.append((int(dc_color), ds))

    cur_row = 0
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
        rotated = _solve(inp, rotate=True)
        plain = _solve(inp, rotate=False)
    except Exception:
        return None

    if np.array_equal(inp, rotated):
        return None
    if np.array_equal(rotated, plain):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
