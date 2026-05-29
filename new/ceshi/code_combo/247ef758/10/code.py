# Combo variant of 247ef758 (index 10)
# combo_spec: D1/2+D11/0
# Name: 单侧匹配+对角目标

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    Base scene stays the same as 247ef758: left sprite strip | one solid divider column |
#    framed rectangle on the right whose interior is bg.
#    The fused rule changes two linked stages:
#    1. Row-marker extraction becomes asymmetric: a color is a valid row marker if it appears
#       on the LEFT frame edge only; the right edge is not a confirmation edge.
#    2. Target selection is not the full row x col Cartesian product; instead, the sorted row
#       list and sorted column list are paired index-wise, giving only diagonal/order-matched
#       targets.
#    Example for sprite color 4:
#    - Input edges show left rows [3, 8] in color 4, right edge stays frame_base at those rows,
#      and top/bottom columns [10, 14] both show color 4.
#    - Original task: rows [3, 8] are rejected because right edge is not 4, so color 4 does not move.
#    - D1-only task: rows [3, 8] are accepted and columns [10, 14] are accepted, producing four
#      Cartesian targets (3,10), (3,14), (8,10), (8,14).
#    - This combo task: the same marks produce only two diagonal targets (3,10) and (8,14).
#    The visible difference is therefore joint: the sprite moves despite lacking right-edge mirrors,
#    and among the 2x2 apparent intersections only the matched-order diagonal is used.
#
# b) Complexity calibration
#    To keep the combo visible, the generator forces at least one mover color to have exactly
#    two row marks and two column marks while all row marks remain left-only. That prevents
#    collapse to either source task or a single-dimension variant. Other movers can still have
#    one or two pairs, giving repeated evidence without too many degrees of freedom.
#
# c) Implicit assumption audit for main()
#    - largest_enclosed_component_bbox(): still assumes the frame interior is the largest bg
#      component enclosed away from the border. Generator preserves the original frame geometry.
#    - divider detection: still assumes a nearly full-height uniform non-bg column left of the
#      frame. Generator keeps exactly one such divider column.
#    - sprite extraction: still assumes left-strip components are 8-connected monochrome blobs
#      and divider cells are excluded. Generator keeps that unchanged.
#    - row mark extraction: the original solver assumed left/right mirrored confirmation. That
#      assumption is intentionally broken; the combo solver must read only the left edge.
#    - diagonal pairing adds a new assumption that row and column marks admit a stable order.
#      The solver uses ascending scan order; the generator guarantees equal row/col counts per
#      mover so zip(rows, cols) is the intended complete pairing.
#    - rendering/erase/order logic does not depend on symmetry or Cartesian expansion and remains valid.
#
# d) Placement legality
#    Marker rows/cols are sampled from sprite-dependent legal center ranges, so every stamped copy
#    fits inside the frame interior. Marker rows and marker cols are not reused across colors.
#    Left-strip sprites are stacked without overlap. Generation rejects any sample where solver
#    execution fails or where the intended combo behavior is not measurable.
#
# e) Visual observability
#    Humans can see both nonstandard rules directly:
#    - row evidence is asymmetric because color appears on the left edge but not the right;
#    - target selection is sparse because a 2-row x 2-col color leaves the off-diagonal crossings empty.
#    Repeating this for at least one forced 2x2 mover makes the combo inferable from 2-4 examples.
#
# f) Visual cleanliness
#    The right edge is kept as frame_base instead of noisy distractor markers, which makes the
#    left-only rule legible. Colors are unique across movers, divider, frame, and distractors.
#    Marker reuse is forbidden, so the edge coding stays readable.
#
# g) Change inferability
#    There is no hidden color remapping. Colors still identify the corresponding sprite and edge
#    marks; only the matching condition and target geometry changed. This keeps the reasoning
#    structural and visible.
#
# Post-change audit of new solver assumptions
# -------------------------------------------
# - _collect_marks(..., row_mode="left_only") assumes every movable color intended by the task
#   has at least one left-edge row mark and at least one mirrored top/bottom column mark.
#   The generator constructs both for every mover.
# - _build_targets(..., mode="diagonal") assumes row and column counts are equal in generated
#   examples when we want all visible marks to participate. The generator uses a shared n_pairs
#   per mover and additionally forces one mover to have n_pairs == 2.
# - The metrics helper assumes combo activity should be measurable against two baselines:
#   source task (symmetric + Cartesian) and D1-only (left-only + Cartesian). The generator rejects
#   degenerate samples where the combo output matches either baseline.

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """Return the bbox of the largest connected component that does not touch the border."""
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


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    """
    Parse the shared 247ef758 scene.

    The combo-specific logic is not baked into parsing; it is isolated in mark collection and
    target construction so we can compare against source-task and partial-variant baselines.
    """
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
                {"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m}
            )

    return {
        "g": g,
        "h": int(h),
        "w": int(w),
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
    }


def _collect_marks(
    scene: dict[str, object],
    *,
    row_mode: str,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Extract row and column marks.

    row_mode:
    - "symmetric": original task rows require left == right.
    - "left_only": combo rows require only the left edge.
    Columns always require top == bottom, which remains unchanged across this combo.
    """
    g = np.array(scene["g"], dtype=int)
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

    move_colors = {int(s["color"]) for s in sprites}
    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)

    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if row_mode == "symmetric":
            ok = cl == cr and cl != int(bg) and cl in move_colors
            color = cl
        elif row_mode == "left_only":
            ok = cl != int(bg) and cl in move_colors
            color = cl
        else:
            raise ValueError(f"unknown row mode: {row_mode}")
        if ok:
            row_marks[int(color)].append(int(r))

    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    return row_marks, col_marks


def _build_targets(
    row_marks: dict[int, list[int]],
    col_marks: dict[int, list[int]],
    *,
    mode: str,
) -> dict[int, list[tuple[int, int]]]:
    targets: dict[int, list[tuple[int, int]]] = {}
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        rows = sorted(map(int, row_marks[int(col)]))
        cols = sorted(map(int, col_marks[int(col)]))
        if mode == "cartesian":
            pts = sorted({(int(r), int(c)) for r in rows for c in cols})
        elif mode == "diagonal":
            pts = [(int(r), int(c)) for r, c in zip(rows, cols)]
        else:
            raise ValueError(f"unknown target mode: {mode}")
        if pts:
            targets[int(col)] = pts
    return targets


def _render_targets(scene: dict[str, object], targets: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    g = np.array(scene["g"], dtype=int)
    out = g.copy()
    bg = int(scene["bg"])
    ir0 = int(scene["ir0"])
    ic0 = int(scene["ic0"])
    ir1 = int(scene["ir1"])
    ic1 = int(scene["ic1"])
    sprites = list(scene["sprites"])

    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0 = int(s["r0"])
        c0 = int(s["c0"])
        for rr, cc in np.argwhere(np.array(s["mask"], dtype=bool)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [
            (int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2))
            for rr, cc2 in np.argwhere(np.array(s["mask"], dtype=bool))
        ]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def _solve_with_modes(grid: np.ndarray, *, row_mode: str, target_mode: str) -> np.ndarray:
    scene = _extract_scene(grid)
    row_marks, col_marks = _collect_marks(scene, row_mode=row_mode)
    targets = _build_targets(row_marks, col_marks, mode=target_mode)
    return _render_targets(scene, targets)


def _original_output(grid: np.ndarray) -> np.ndarray:
    return _solve_with_modes(grid, row_mode="symmetric", target_mode="cartesian")


def _leftonly_cartesian_output(grid: np.ndarray) -> np.ndarray:
    return _solve_with_modes(grid, row_mode="left_only", target_mode="cartesian")


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether both combo controls are active on the same generated sample.

    Metrics:
    - asymmetric_row_marks: count of row marks accepted only by the combo row rule.
    - diagonal_omissions: how many Cartesian targets the combo omits.
    - combo_active_colors: colors with both asymmetry and a 2x2-or-larger diagonal-vs-Cartesian gap.
    """
    scene = _extract_scene(grid)
    row_left, col_marks = _collect_marks(scene, row_mode="left_only")
    row_sym, _ = _collect_marks(scene, row_mode="symmetric")

    asymmetric_row_marks = 0
    diagonal_omissions = 0
    combo_active_colors = 0

    for col in sorted(set(row_left.keys()) & set(col_marks.keys())):
        rows_left = sorted(map(int, row_left[int(col)]))
        rows_sym = set(map(int, row_sym.get(int(col), [])))
        cols = sorted(map(int, col_marks[int(col)]))
        asym_rows = [int(r) for r in rows_left if int(r) not in rows_sym]

        asymmetric_row_marks += len(asym_rows)
        diagonal_omissions += max(0, len(rows_left) * len(cols) - min(len(rows_left), len(cols)))
        if asym_rows and len(rows_left) >= 2 and len(cols) >= 2:
            combo_active_colors += 1

    return {
        "asymmetric_row_marks": int(asymmetric_row_marks),
        "diagonal_omissions": int(diagonal_omissions),
        "combo_active_colors": int(combo_active_colors),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo solver for D1/2 + D11/0.

    - Row marks: only the LEFT frame edge needs the mover color.
    - Column marks: TOP and BOTTOM must still match.
    - Targets: sorted row marks are paired with sorted column marks index-wise, not Cartesian-expanded.
    """
    return _solve_with_modes(grid, row_mode="left_only", target_mode="diagonal")


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick exactly n values with at least min_spacing between picks."""
    if len(candidates) < n:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return sorted(picked)
    return None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
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

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, w - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = h - 2, fc1 - 1

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
        sprite = _make_sprite(mc, 5, sprite_area_w)
        if sprite is None:
            return None
        sprites.append(sprite)

    forced_combo = random.randrange(n_movers)
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()

    for i, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None

        n_pairs = 2 if i == forced_combo else random.choice([1, 2])
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        row_marks = _pick_spaced(avail_rows, n_pairs, sh)
        col_marks = _pick_spaced(avail_cols, n_pairs, sw)
        if row_marks is None or col_marks is None:
            return None

        mover_targets[int(mc)] = (row_marks, col_marks)
        occupied_rows.update(map(int, row_marks))
        occupied_cols.update(map(int, col_marks))

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = top_edge

    left_edge = np.full(h - 2, int(frame_base), dtype=int)
    for mc, (rmarks, _) in mover_targets.items():
        for rr in rmarks:
            left_edge[int(rr - 1)] = int(mc)
    inp[1 : h - 1, fc0] = left_edge

    # Keep the right edge neutral so the left-only rule is both necessary and visible.
    inp[1 : h - 1, fc1] = int(frame_base)

    cur_row = 0
    items: list[tuple[int, np.ndarray]] = [(int(mc), sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        items.append((int(dc_color), ds))

    for _, sp in items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[int(dr), int(dc)])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        combo_output = main(inp)
        original_output = _original_output(inp)
        leftonly_cart_output = _leftonly_cartesian_output(inp)
        metrics = _variant_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, combo_output):
        return None
    if np.array_equal(combo_output, original_output):
        return None
    if np.array_equal(combo_output, leftonly_cart_output):
        return None
    if metrics["combo_active_colors"] < 1:
        return None
    if metrics["asymmetric_row_marks"] < 2:
        return None
    if metrics["diagonal_omissions"] < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
