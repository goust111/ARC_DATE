# Combo variant of 247ef758 (index 33)
# combo_spec: D3/0+D7/1+D11/0
# Name: 内部标记+面积排序+对角目标

from __future__ import annotations

# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace
#    Source task:
#    - Right side is a framed rectangle whose informative colors sit on the border.
#    - Matching left/right edge pixels select rows, matching top/bottom edge pixels
#      select columns, and every selected row combines with every selected column.
#    - Movers are stamped in bottom-to-top order from the left strip.
#
#    Fused task:
#    - The border is now visually uniform. Informative marker dots move inside the
#      frame onto one guide column and one guide row.
#    - For each mover color X, rows are the r values where grid[r, guide_col] == X,
#      and columns are the c values where grid[guide_row, c] == X.
#    - Targets are not the full Cartesian product. After sorting those rows and
#      columns, only the diagonal pairs are used: zip(rows, cols).
#    - When multiple movers overlap, stamping order is by visible sprite area:
#      smaller blobs stamp first, larger blobs stamp last and cover them.
#
#    Small trace:
#    - Suppose color 4 has row dots at rows [5, 9] and column dots at cols [11, 15].
#      Original rule would stamp at (5,11), (5,15), (9,11), (9,15) and would use
#      strip position to decide overwrite priority.
#      Fused rule stamps only at (5,11) and (9,15), and a larger overlapping sprite
#      can cover a smaller one even if the larger sprite sits lower in the strip.
#
# b) Complexity calibration
#    - Keep 2-3 mover colors and 0-1 distractor, matching the source task scale.
#    - Force at least one mover to have two row dots and two column dots so the
#      diagonal-vs-Cartesian distinction is visible.
#    - Force distinct mover areas and reject samples where area-order output equals
#      position-order output, so the D7 change is visible instead of latent.
#
# c) Implicit-assumption audit
#    - `largest_enclosed_component_bbox(g == bg)` still works because a few internal
#      marker dots do not disconnect the frame interior under 4-connectivity.
#    - Divider detection and left-strip sprite extraction remain valid.
#    - The source assumption "markers live on frame edges" is invalid; parsing must
#      read the internal guide row/column instead.
#    - The source assumption "all row marks cross all column marks" is invalid; the
#      target builder must zip sorted rows and cols.
#    - The source assumption "overwrite order comes from vertical strip position" is
#      invalid; the renderer must sort movers by area instead.
#    - New solver assumptions introduced here:
#      1. the frame border is uniform and markers live only on the internal guides;
#      2. row/col mark counts for a color are paired by sorted order;
#      3. guide lanes are kept clear enough that stamped sprites do not erase the
#         evidence needed to infer the rule.
#
# d) Placement legality
#    - Generator chooses sprite-dependent legal target-center ranges so every stamp
#      fits entirely inside the frame interior.
#    - Centers are kept away from the guide row/column.
#    - Row marks and column marks are unique across mover colors, so matching is
#      unambiguous.
#
# e) Visual observability
#    - Internal dots make the D3 change obvious because the border is now plain.
#    - At least one color shows a 2-row/2-col setup, so omitted off-diagonal targets
#      reveal the D11 change.
#    - Movers are stacked small-to-large in the left strip while primary targets are
#      clustered, making area-based covering differ from position-based covering.
#
# f) Visual cleanliness
#    - Only one guide row and one guide column carry marker dots.
#    - Secondary diagonal pairs are spaced away from their primary pair so the frame
#      stays readable.
#    - Distractors stay in the left strip and do not add extra symbols inside the
#      frame.
#
# g) Inferability of changes
#    - No hidden use of color ids: matching is by visible same-color reuse, and
#      ordering is by visible area only.
#    - Each altered rule has repeated structural evidence in a single sample, and the
#      generator rejects samples where the fused output collapses to a weaker rule.

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
for _extra in (_ROOT, _ROOT / "solvers", _ROOT / "arc2_solver_select"):
    _p = str(_extra)
    if _p not in sys.path:
        sys.path.append(_p)

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Bbox of the largest connected component in `mask` that does not touch the border.
    """
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


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
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
            sprite_mask = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": sprite_mask,
                    "area": int(sprite_mask.sum()),
                }
            )

    move_colors = {int(s["color"]) for s in sprites}
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

    return {
        "grid": g,
        "bg": int(bg),
        "inner_bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
        "frame_bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
        "guide_row": int(guide_row),
        "guide_col": int(guide_col),
        "sprites": sprites,
        "row_marks": row_marks,
        "col_marks": col_marks,
    }


def _build_targets(
    row_marks: dict[int, list[int]],
    col_marks: dict[int, list[int]],
    *,
    target_mode: str,
) -> dict[int, list[tuple[int, int]]]:
    targets: dict[int, list[tuple[int, int]]] = {}
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        rows = sorted(map(int, row_marks[int(col)]))
        cols = sorted(map(int, col_marks[int(col)]))
        if target_mode == "diagonal":
            pts = [(int(r), int(c)) for r, c in zip(rows, cols)]
        elif target_mode == "cartesian":
            pts = sorted({(int(r), int(c)) for r in rows for c in cols})
        else:
            raise ValueError(f"unknown target_mode: {target_mode}")
        if pts:
            targets[int(col)] = pts
    return targets


def _render_targets(
    scene: dict[str, object],
    targets: dict[int, list[tuple[int, int]]],
    *,
    order_mode: str,
) -> np.ndarray:
    g = np.array(scene["grid"], dtype=int)
    out = g.copy()
    bg = int(scene["bg"])
    ir0, ic0, ir1, ic1 = map(int, scene["inner_bbox"])
    sprites = list(scene["sprites"])

    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(np.array(s["mask"], dtype=bool)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    if order_mode == "position":
        movers.sort(
            key=lambda s: (
                -((int(s["r0"]) + int(s["r1"])) / 2.0),
                int(s["c0"]),
                int(s["color"]),
            )
        )
    elif order_mode == "area_asc":
        movers.sort(
            key=lambda s: (
                int(s["area"]),
                ((int(s["r0"]) + int(s["r1"])) / 2.0),
                int(s["c0"]),
                int(s["color"]),
            )
        )
    else:
        raise ValueError(f"unknown order_mode: {order_mode}")

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


def _solve_with_modes(
    grid: np.ndarray,
    *,
    target_mode: str,
    order_mode: str,
) -> np.ndarray:
    scene = _extract_scene(grid)
    targets = _build_targets(scene["row_marks"], scene["col_marks"], target_mode=target_mode)
    return _render_targets(scene, targets, order_mode=order_mode)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    scene = _extract_scene(grid)
    row_marks = scene["row_marks"]
    col_marks = scene["col_marks"]
    fr0, fc0, fr1, fc1 = map(int, scene["frame_bbox"])
    guide_row = int(scene["guide_row"])
    guide_col = int(scene["guide_col"])
    g = np.array(scene["grid"], dtype=int)
    sprites = list(scene["sprites"])

    diagonal_out = _solve_with_modes(grid, target_mode="diagonal", order_mode="area_asc")
    position_out = _solve_with_modes(grid, target_mode="diagonal", order_mode="position")
    cartesian_out = _solve_with_modes(grid, target_mode="cartesian", order_mode="area_asc")

    edge_vals = []
    edge_vals.extend(map(int, g[int(fr0), int(fc0) : int(fc1) + 1]))
    edge_vals.extend(map(int, g[int(fr1), int(fc0) : int(fc1) + 1]))
    edge_vals.extend(map(int, g[int(fr0) + 1 : int(fr1), int(fc0)]))
    edge_vals.extend(map(int, g[int(fr0) + 1 : int(fr1), int(fc1)]))

    paired_colors = 0
    extra_cartesian_targets = 0
    internal_marker_pixels = 0
    for col in sorted(set(row_marks.keys()) & set(col_marks.keys())):
        nr = len(row_marks[int(col)])
        nc = len(col_marks[int(col)])
        internal_marker_pixels += nr + nc
        if nr >= 2 and nc >= 2:
            paired_colors += 1
        extra_cartesian_targets += max(0, nr * nc - min(nr, nc))

    areas = sorted(int(s["area"]) for s in sprites)
    return {
        "paired_colors": int(paired_colors),
        "extra_cartesian_targets": int(extra_cartesian_targets),
        "internal_marker_pixels": int(internal_marker_pixels),
        "uniform_edge_border": int(len(set(edge_vals)) == 1),
        "guide_row_non_bg": int(np.count_nonzero(g[int(guide_row), :] != int(scene["bg"]))),
        "guide_col_non_bg": int(np.count_nonzero(g[:, int(guide_col)] != int(scene["bg"]))),
        "area_span": int(areas[-1] - areas[0]) if areas else 0,
        "position_diff_pixels": int(np.count_nonzero(diagonal_out != position_out)),
        "cartesian_diff_pixels": int(np.count_nonzero(diagonal_out != cartesian_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused variant rule:
    - read same-color marker dots from an internal guide row/column,
    - pair sorted row marks and column marks diagonally instead of taking the
      full Cartesian product,
    - stamp movers in ascending area order so larger sprites cover smaller ones.
    """
    return _solve_with_modes(grid, target_mode="diagonal", order_mode="area_asc")


def _make_sprite(color: int, max_h: int, max_w: int, n_pixels: int) -> np.ndarray | None:
    """
    Generate a cropped 8-connected single-color sprite with exactly `n_pixels`.
    """
    if max_h < 1 or max_w < 1 or n_pixels < 1 or n_pixels > max_h * max_w:
        return None

    min_h = 2 if max_h >= 2 else 1
    min_w = 2 if max_w >= 2 else 1
    for _ in range(100):
        sh = random.randint(min_h, max_h)
        sw = random.randint(min_w, max_w)
        if sh * sw < n_pixels:
            continue

        pixels = {(sh // 2, sw // 2)}
        while len(pixels) < n_pixels:
            frontier = set()
            for r, c in pixels:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < sh and 0 <= nc < sw and (nr, nc) not in pixels:
                            frontier.add((nr, nc))
            if not frontier:
                break
            pixels.add(random.choice(tuple(frontier)))

        if len(pixels) != n_pixels:
            continue

        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
        for pr, pc in pixels:
            sprite[pr - min_r, pc - min_c] = int(color)
        return sprite

    return None


def _safe_center_bounds(
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    guide_row: int,
    guide_col: int,
    sprite: np.ndarray,
) -> tuple[int, int, int, int]:
    sh, sw = sprite.shape
    top_extent = int(sh // 2)
    bottom_extent = int(sh - 1 - top_extent)
    left_extent = int(sw // 2)
    right_extent = int(sw - 1 - left_extent)

    min_row = int(max(ir0, guide_row + 1) + top_extent + 1)
    max_row = int(ir1 - bottom_extent)
    min_col = int(max(ic0, guide_col + 1) + left_extent + 1)
    max_col = int(ic1 - right_extent)
    return min_row, max_row, min_col, max_col


def _choose_cluster_centers(
    bounds: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]] | None:
    n = len(bounds)
    row_offsets = list(range(n))
    col_offsets = list(range(n))

    row_anchor_lo = -10**9
    row_anchor_hi = 10**9
    col_anchor_lo = -10**9
    col_anchor_hi = 10**9
    for (min_row, max_row, min_col, max_col), ro, co in zip(bounds, row_offsets, col_offsets):
        row_anchor_lo = max(row_anchor_lo, int(min_row - ro))
        row_anchor_hi = min(row_anchor_hi, int(max_row - ro))
        col_anchor_lo = max(col_anchor_lo, int(min_col - co))
        col_anchor_hi = min(col_anchor_hi, int(max_col - co))
    if row_anchor_lo > row_anchor_hi or col_anchor_lo > col_anchor_hi:
        return None

    anchor_r = random.randint(int(row_anchor_lo), int(row_anchor_hi))
    anchor_c = random.randint(int(col_anchor_lo), int(col_anchor_hi))
    return [(int(anchor_r + ro), int(anchor_c + co)) for ro, co in zip(row_offsets, col_offsets)]


def _pick_secondary_pair(
    primary_r: int,
    primary_c: int,
    *,
    min_row: int,
    max_row: int,
    min_col: int,
    max_col: int,
    row_spacing: int,
    col_spacing: int,
    occupied_rows: set[int],
    occupied_cols: set[int],
) -> tuple[int, int] | None:
    for sign in random.sample([-1, 1], 2):
        row_pool = []
        if sign < 0:
            row_pool = [
                r
                for r in range(int(min_row), int(primary_r - row_spacing) + 1)
                if r not in occupied_rows
            ]
            col_pool = [
                c
                for c in range(int(min_col), int(primary_c - col_spacing) + 1)
                if c not in occupied_cols
            ]
        else:
            row_pool = [
                r
                for r in range(int(primary_r + row_spacing), int(max_row) + 1)
                if r not in occupied_rows
            ]
            col_pool = [
                c
                for c in range(int(primary_c + col_spacing), int(max_col) + 1)
                if c not in occupied_cols
            ]
        random.shuffle(row_pool)
        random.shuffle(col_pool)
        if row_pool and col_pool:
            return int(row_pool[0]), int(col_pool[0])
    return None


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
    guide_row = int(ir0 + 1)
    guide_col = int(ic0 + 1)
    if guide_row > ir1 or guide_col > ic1:
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
    mover_colors = [int(v) for v in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    area_choices = list(range(3, 10))
    random.shuffle(area_choices)
    mover_areas = sorted(area_choices[:n_movers])

    mover_info: list[dict[str, object]] = []
    for mc, area in zip(mover_colors, mover_areas):
        sp = _make_sprite(int(mc), 5, int(sprite_area_w), int(area))
        if sp is None:
            return None
        mover_info.append({"color": int(mc), "sprite": sp, "area": int(area)})

    mover_info.sort(key=lambda item: (int(item["area"]), int(item["color"])))

    bounds = [
        _safe_center_bounds(
            int(ir0), int(ic0), int(ir1), int(ic1), int(guide_row), int(guide_col), np.array(item["sprite"], dtype=int)
        )
        for item in mover_info
    ]
    if any(b[0] > b[1] or b[2] > b[3] for b in bounds):
        return None

    centers = _choose_cluster_centers(bounds)
    if centers is None:
        return None

    forced_double = random.randrange(n_movers)
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for i, (item, center, bnd) in enumerate(zip(mover_info, centers, bounds)):
        mc = int(item["color"])
        sp = np.array(item["sprite"], dtype=int)
        sh, sw = sp.shape
        primary_r, primary_c = map(int, center)
        min_row, max_row, min_col, max_col = map(int, bnd)
        if not (min_row <= primary_r <= max_row and min_col <= primary_c <= max_col):
            return None
        if primary_r in occupied_rows or primary_c in occupied_cols:
            return None

        rows = [int(primary_r)]
        cols = [int(primary_c)]
        occupied_rows.add(int(primary_r))
        occupied_cols.add(int(primary_c))

        pair_count = 2 if i == forced_double else random.choice([1, 2])
        if pair_count == 2:
            second = _pick_secondary_pair(
                int(primary_r),
                int(primary_c),
                min_row=int(min_row),
                max_row=int(max_row),
                min_col=int(min_col),
                max_col=int(max_col),
                row_spacing=max(2, int(sh)),
                col_spacing=max(2, int(sw)),
                occupied_rows=occupied_rows,
                occupied_cols=occupied_cols,
            )
            if second is None:
                return None
            r2, c2 = map(int, second)
            rows.append(int(r2))
            cols.append(int(c2))
            occupied_rows.add(int(r2))
            occupied_cols.add(int(c2))

        mover_targets[int(mc)] = (sorted(rows), sorted(cols))

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    inp[0, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[h - 1, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[:, int(fc0)] = int(frame_base)
    inp[:, int(fc1)] = int(frame_base)

    for mc, (rmarks, cmarks) in mover_targets.items():
        for r in rmarks:
            inp[int(r), int(guide_col)] = int(mc)
        for c in cmarks:
            inp[int(guide_row), int(c)] = int(mc)

    cur_row = 0
    for item in mover_info:
        sp = np.array(item["sprite"], dtype=int)
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[dr, dc])
        cur_row += int(sh)

    for dc_color in distractor_colors:
        remaining_h = int(h - cur_row)
        if remaining_h <= 0:
            return None
        d_area = random.randint(3, min(7, max(3, remaining_h * int(sprite_area_w))))
        ds = _make_sprite(int(dc_color), min(5, remaining_h), int(sprite_area_w), int(d_area))
        if ds is None:
            return None
        sh, sw = ds.shape
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(sh):
            for dc in range(sw):
                if ds[dr, dc] != bg:
                    inp[int(cur_row + dr), int(c_off + dc)] = int(ds[dr, dc])
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        fused_out = main(inp)
        position_out = _solve_with_modes(inp, target_mode="diagonal", order_mode="position")
        cartesian_out = _solve_with_modes(inp, target_mode="cartesian", order_mode="area_asc")
        metrics = _fusion_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, fused_out):
        return None
    if np.array_equal(fused_out, position_out):
        return None
    if np.array_equal(fused_out, cartesian_out):
        return None
    if metrics["paired_colors"] < 1:
        return None
    if metrics["extra_cartesian_targets"] < 2:
        return None
    if metrics["uniform_edge_border"] != 1:
        return None
    if metrics["internal_marker_pixels"] < 6:
        return None
    if metrics["area_span"] <= 0:
        return None
    if metrics["position_diff_pixels"] <= 0 or metrics["cartesian_diff_pixels"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 800 attempts")
