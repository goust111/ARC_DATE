# Combo variant of 247ef758 (index 9)
# combo_spec: D1/1+D7/1
# Name: 尺寸双角色（匹配+排序）
#
# Design analysis
# a) Concrete behavior trace
#    - Layout stays the same: left sprite strip | full-height divider | right frame.
#    - The original task used single edge cells as markers and position-order stamping.
#    - This combo turns each mirrored edge marker into a maximal monochrome segment.
#      Left/right segments encode `(color, bbox_height)` and top/bottom segments encode
#      `(color, bbox_width)`. In the fused variant, segment length decides whether a band
#      is eligible, and every cell on an eligible band is a candidate row/col mark.
#    - Example: a red sprite with bbox 3x2 and pixel-area 5.
#      Original rule: every mirrored red row cell and red col cell is valid; overlap order
#      depends on where the sprite sat in the left strip.
#      Combo rule: only mirrored red row-segments of length 3 and mirrored red col-segments
#      of length 2 are valid. Every cell on those matched segments participates in the
#      row×col product, and if red overlaps another mover then the smaller-area sprite
#      lands first and the larger-area sprite covers it later.
#    - We also insert same-color wrong-length segments, so color alone is insufficient.
#
# b) Complexity calibration
#    - Bbox sizes are kept in the readable 2..4 range.
#    - Movers stay at 2-3 and distractors at 0-1, matching the original task family.
#    - Each sample forces two non-degenerate signals:
#      `size-aware output != color-only output` and
#      `area-order output != original position-order output`.
#
# c) Implicit assumptions reviewed
#    - Original `row_marks/col_marks` code assumed one-cell markers; here we must scan
#      maximal mirrored segments, index them by `(color, length)`, and expand only the
#      qualifying segments into candidate row/col cells.
#    - Original mover ordering depended on strip y-position; here the solver must compute
#      per-sprite pixel area.
#    - The solver now assumes same-color segments do not touch each other; the generator
#      guarantees that, preventing accidental merging while still letting different colors abut.
#
# d) Placement legality
#    - Segment placement uses color-aware interval occupancy: different colors may abut, but
#      same-color segments keep a separating gap so the length code stays readable.
#    - Segment placement itself is always legal and non-overlapping on the frame edges.
#      Stamping may clip near the frame boundary because eligible segments contribute full
#      bands rather than a single center; this is intentional and preserved by the solver.
#    - Left-strip sprites are stacked without overlap and always fit in the strip width.
#
# e) Visual observability
#    - Humans can see both relevant size cues: sprite bbox dimensions for matching and
#      sprite pixel counts for cover order. No hidden attribute such as color id magnitude
#      or generation order controls the final behavior.
#    - Same-color wrong-length segments make the D1 change visible; using whole qualified
#      bands rather than single centers lets different movers overlap, which keeps D7 visible.
#
# f) Visual cleanliness
#    - Only the frame edges carry segment codes; the interior stays empty before stamping.
#    - Overlap is encouraged but limited to a small cluster, so outputs remain legible.
#
# g) New solver assumptions checked against the generator
#    - `largest_enclosed_component_bbox`: generator always creates a single rectangular frame
#      with one enclosed background component.
#    - Divider detection: generator always creates one full-height uniform divider column.
#    - Area-order tie-breaks are deterministic in the solver, but the generator avoids equal
#      mover areas so the learned rule remains "sort by visible area".

from __future__ import annotations

from collections import defaultdict
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


def _scan_mirrored_row_segments(
    g: np.ndarray,
    fc0: int,
    fc1: int,
    ir0: int,
    ir1: int,
    allowed_colors: set[int],
) -> dict[tuple[int, int], list[int]]:
    """Scan maximal mirrored vertical edge segments keyed by `(color, length)`."""
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
            length = int(r1 - r0 + 1)
            marks[(int(cl), int(length))].extend(range(int(r0), int(r1) + 1))
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
    """Scan maximal mirrored horizontal edge segments keyed by `(color, length)`."""
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
            length = int(c1 - c0 + 1)
            marks[(int(ct), int(length))].extend(range(int(c0), int(c1) + 1))
        c += 1
    return marks


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
                    "mask": m,
                    "height": int(m.shape[0]),
                    "width": int(m.shape[1]),
                    "area": int(m.sum()),
                }
            )

    move_colors = {int(s["color"]) for s in sprites}
    row_marks = _scan_mirrored_row_segments(g, int(fc0), int(fc1), int(ir0), int(ir1), move_colors)
    col_marks = _scan_mirrored_col_segments(g, int(fr0), int(fr1), int(ic0), int(ic1), move_colors)

    return {
        "grid": g,
        "bg": bg,
        "inner_bbox": (ir0, ic0, ir1, ic1),
        "frame_bbox": (fr0, fc0, fr1, fc1),
        "sprites": sprites,
        "row_marks": row_marks,
        "col_marks": col_marks,
    }


def _centers_for_color(marks: dict[tuple[int, int], list[int]], color: int) -> list[int]:
    vals: set[int] = set()
    for (mark_color, _length), centers in marks.items():
        if int(mark_color) == int(color):
            vals.update(map(int, centers))
    return sorted(vals)


def _solve_with_modes(grid: np.ndarray, *, use_size_match: bool, order_mode: str) -> np.ndarray:
    """
    Shared solver core for:
    - combo rule: `use_size_match=True, order_mode='area_asc'`
    - D1 ablation: `use_size_match=False`
    - D7 ablation: `order_mode='position'`
    """
    scene = _extract_scene(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    ir0, ic0, ir1, ic1 = map(int, scene["inner_bbox"])
    sprites = scene["sprites"]
    row_marks = scene["row_marks"]
    col_marks = scene["col_marks"]

    movers = []
    for s in sprites:
        if use_size_match:
            rows = sorted(set(row_marks.get((int(s["color"]), int(s["height"])), [])))
            cols = sorted(set(col_marks.get((int(s["color"]), int(s["width"])), [])))
        else:
            rows = _centers_for_color(row_marks, int(s["color"]))
            cols = _centers_for_color(col_marks, int(s["color"]))
        if rows and cols:
            movers.append(
                {
                    **s,
                    "targets": [(int(r), int(c)) for r in rows for c in cols],
                }
            )

    out = g.copy()
    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    if order_mode == "position":
        movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"]), int(s["color"])))
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
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in s["targets"]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Solve the fused variant: size-aware matching plus area-based overlap order."""
    return _solve_with_modes(grid, use_size_match=True, order_mode="area_asc")


def _can_place_interval(
    start: int,
    end: int,
    placements: list[tuple[int, int, int]],
    color: int,
    lo: int,
    hi: int,
) -> bool:
    if int(start) < int(lo) or int(end) > int(hi):
        return False
    for prev_start, prev_end, prev_color in placements:
        if not (int(end) < int(prev_start) or int(start) > int(prev_end)):
            return False
        if int(prev_color) == int(color):
            if not (int(end) < int(prev_start - 1) or int(start) > int(prev_end + 1)):
                return False
    return True


def _occupy_interval(start: int, end: int, placements: list[tuple[int, int, int]], color: int) -> None:
    placements.append((int(start), int(end), int(color)))


def _place_interval_center(
    length: int,
    placements: list[tuple[int, int, int]],
    lo: int,
    hi: int,
    *,
    color: int,
    center_lo: int | None = None,
    center_hi: int | None = None,
) -> int | None:
    upper, lower = _segment_extents(int(length))
    low = int(lo + upper if center_lo is None else max(center_lo, lo + upper))
    high = int(hi - lower if center_hi is None else min(center_hi, hi - lower))
    if low > high:
        return None
    candidates = list(range(int(low), int(high) + 1))
    random.shuffle(candidates)
    for center in candidates:
        start, end = _segment_bounds(int(center), int(length))
        if _can_place_interval(int(start), int(end), placements, int(color), int(lo), int(hi)):
            _occupy_interval(int(start), int(end), placements, int(color))
            return int(center)
    return None


def _make_sprite(color: int, max_h: int, max_w: int, n_pixels: int) -> np.ndarray | None:
    """Generate a cropped 8-connected single-color sprite with exact pixel area."""
    if max_h < 2 or max_w < 2 or n_pixels < 3 or n_pixels > max_h * max_w:
        return None

    for _ in range(120):
        sh = random.randint(2, max_h)
        sw = random.randint(2, max_w)
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
        hh = int(max_r - min_r + 1)
        ww = int(max_c - min_c + 1)
        if hh < 2 or ww < 2 or hh > max_h or ww > max_w:
            continue

        sprite = np.zeros((hh, ww), dtype=int)
        for pr, pc in pixels:
            sprite[int(pr - min_r), int(pc - min_c)] = int(color)
        return sprite

    return None


def _make_solid_sprite(color: int, height: int, width: int) -> np.ndarray:
    return np.full((int(height), int(width)), int(color), dtype=int)


def _center_bounds(ir0: int, ic0: int, ir1: int, ic1: int, sprite: np.ndarray) -> tuple[int, int, int, int]:
    sh, sw = sprite.shape
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    return min_row, max_row, min_col, max_col


def _place_primary_centers(
    specs: list[tuple[int, int, int, int]],
    *,
    lo: int,
    hi: int,
) -> tuple[dict[int, int], list[tuple[int, int, int]]] | None:
    """
    Place one primary segment center per mover, clustered near a shared anchor while
    respecting non-touching segment intervals. Each spec is `(color, length, center_lo, center_hi)`.
    """
    if not specs:
        return {}, []

    anchor_lo = min(int(center_lo) for _color, _length, center_lo, _center_hi in specs)
    anchor_hi = max(int(center_hi) for _color, _length, _center_lo, center_hi in specs)
    anchors = list(range(int(anchor_lo), int(anchor_hi) + 1))
    random.shuffle(anchors)
    specs_ordered = sorted(specs, key=lambda x: (-int(x[1]), random.random()))

    for anchor in anchors[: min(24, len(anchors))]:
        occupied: list[tuple[int, int, int]] = []
        centers: dict[int, int] = {}
        ok = True
        for color, length, center_lo, center_hi in specs_ordered:
            placed = None
            for radius in (0, 1, 2, 3, 4, 6, 99):
                lo_here = int(center_lo) if radius == 99 else int(max(center_lo, anchor - radius))
                hi_here = int(center_hi) if radius == 99 else int(min(center_hi, anchor + radius))
                placed = _place_interval_center(
                    int(length),
                    occupied,
                    int(lo),
                    int(hi),
                    color=int(color),
                    center_lo=int(lo_here),
                    center_hi=int(hi_here),
                )
                if placed is not None:
                    break
            if placed is None:
                ok = False
                break
            centers[int(color)] = int(placed)
        if ok:
            return centers, occupied
    return None


def _add_optional_valid_segment(
    *,
    color: int,
    length: int,
    center_lo: int,
    center_hi: int,
    occupied: list[tuple[int, int, int]],
    lo: int,
    hi: int,
    count_bias: float,
) -> list[int]:
    centers: list[int] = []
    if random.random() < float(count_bias):
        extra = _place_interval_center(
            int(length),
            occupied,
            int(lo),
            int(hi),
            color=int(color),
            center_lo=int(center_lo),
            center_hi=int(center_hi),
        )
        if extra is not None:
            centers.append(int(extra))
    return centers


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    combo = main(grid)
    position = _solve_with_modes(grid, use_size_match=True, order_mode="position")
    color_only = _solve_with_modes(grid, use_size_match=False, order_mode="area_asc")
    scene = _extract_scene(grid)

    sprites_by_color = {int(s["color"]): s for s in scene["sprites"]}
    bad_row = 0
    bad_col = 0
    for (color, length), centers in scene["row_marks"].items():
        if int(length) != int(sprites_by_color[int(color)]["height"]):
            bad_row += int(len(set(map(int, centers))))
    for (color, length), centers in scene["col_marks"].items():
        if int(length) != int(sprites_by_color[int(color)]["width"]):
            bad_col += int(len(set(map(int, centers))))

    return {
        "order_diff": int(not np.array_equal(combo, position)),
        "size_diff": int(not np.array_equal(combo, color_only)),
        "bad_row_centers": int(bad_row),
        "bad_col_centers": int(bad_col),
        "bad_total_centers": int(bad_row + bad_col),
    }


def _bump(stats: dict[str, int] | None, key: str) -> None:
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + 1


def _gen_attempt(stats: dict[str, int] | None = None) -> np.ndarray | None:
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

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = 2
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    mover_colors = [int(v) for v in all_colors[idx : idx + n_movers]]
    idx += n_movers
    distractor_colors = [int(v) for v in all_colors[idx : idx + n_distractors]]

    mover_info: list[dict] = []
    rect_choices = [
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
    ]
    rect_choices = [
        (int(h), int(w))
        for h, w in rect_choices
        if int(h) <= 4 and int(w) <= int(sprite_area_w)
    ]
    random.shuffle(rect_choices)
    chosen_rects: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for h, w in rect_choices:
        area = int(h * w)
        if area in used_areas:
            continue
        chosen_rects.append((int(h), int(w)))
        used_areas.add(area)
        if len(chosen_rects) == int(n_movers):
            break
    if len(chosen_rects) < int(n_movers):
        _bump(stats, "area_choices")
        return None
    chosen_rects.sort(key=lambda hw: (int(hw[0] * hw[1]), int(hw[0]), int(hw[1])))
    for mc, (sh, sw) in zip(mover_colors, chosen_rects):
        sp = _make_solid_sprite(int(mc), int(sh), int(sw))
        mover_info.append({"color": int(mc), "sprite": sp, "area": int(sh * sw)})

    if int(sum(int(item["sprite"].shape[0]) for item in mover_info)) > int(H):
        _bump(stats, "mover_height_sum")
        return None

    mover_info.sort(key=lambda item: (int(item["area"]), int(item["color"])))

    row_specs: list[tuple[int, int, int, int]] = []
    col_specs: list[tuple[int, int, int, int]] = []
    bounds_by_color: dict[int, tuple[int, int, int, int]] = {}
    dims_by_color: dict[int, tuple[int, int]] = {}
    for item in mover_info:
        mc = int(item["color"])
        sp = item["sprite"]
        sh, sw = map(int, sp.shape)
        min_row, max_row, min_col, max_col = _center_bounds(int(ir0), int(ic0), int(ir1), int(ic1), sp)
        if min_row > max_row or min_col > max_col:
            _bump(stats, "center_bounds")
            return None
        bounds_by_color[int(mc)] = (int(min_row), int(max_row), int(min_col), int(max_col))
        dims_by_color[int(mc)] = (int(sh), int(sw))
        row_specs.append((int(mc), int(sh), int(min_row), int(max_row)))
        col_specs.append((int(mc), int(sw), int(min_col), int(max_col)))

    row_placed = _place_primary_centers(row_specs, lo=int(ir0), hi=int(ir1))
    col_placed = _place_primary_centers(col_specs, lo=int(ic0), hi=int(ic1))
    if row_placed is None or col_placed is None:
        _bump(stats, "primary_centers")
        return None
    primary_rows, row_occupied = row_placed
    primary_cols, col_occupied = col_placed

    row_centers: dict[int, list[int]] = {int(mc): [int(primary_rows[int(mc)])] for mc in mover_colors}
    col_centers: dict[int, list[int]] = {int(mc): [int(primary_cols[int(mc)])] for mc in mover_colors}
    row_lengths: dict[int, list[int]] = {int(mc): [int(dims_by_color[int(mc)][0])] for mc in mover_colors}
    col_lengths: dict[int, list[int]] = {int(mc): [int(dims_by_color[int(mc)][1])] for mc in mover_colors}

    row_bad_candidates = []
    col_bad_candidates = []
    for mc in mover_colors:
        bad_row_lengths = [v for v in (2, 3, 4) if v <= int(interior_h) and v != int(dims_by_color[int(mc)][0])]
        bad_col_lengths = [v for v in (2, 3, 4) if v <= int(interior_w) and v != int(dims_by_color[int(mc)][1])]
        min_row, max_row, min_col, max_col = bounds_by_color[int(mc)]
        if bad_row_lengths:
            row_slack = int(max_row - min_row + 1)
            row_bad_candidates.append((row_slack, random.random(), int(mc), bad_row_lengths))
        if bad_col_lengths:
            col_slack = int(max_col - min_col + 1)
            col_bad_candidates.append((col_slack, random.random(), int(mc), bad_col_lengths))
    if not row_bad_candidates and not col_bad_candidates:
        _bump(stats, "bad_lengths")
        return None

    row_bad_candidates.sort(key=lambda x: (-int(x[0]), float(x[1])))
    col_bad_candidates.sort(key=lambda x: (-int(x[0]), float(x[1])))

    placed_bad = 0
    if row_bad_candidates:
        _slack, _rnd, bad_row_color, bad_row_lengths = row_bad_candidates[0]
        min_row, max_row, _min_col, _max_col = bounds_by_color[int(bad_row_color)]
        bad_row_length = int(random.choice(bad_row_lengths))
        bad_row = _place_interval_center(
            int(bad_row_length),
            row_occupied,
            int(ir0),
            int(ir1),
            color=int(bad_row_color),
            center_lo=int(min_row),
            center_hi=int(max_row),
        )
        if bad_row is None:
            bad_row = _place_interval_center(
                int(bad_row_length),
                row_occupied,
                int(ir0),
                int(ir1),
                color=int(bad_row_color),
            )
        if bad_row is not None:
            row_centers[int(bad_row_color)].append(int(bad_row))
            row_lengths[int(bad_row_color)].append(int(bad_row_length))
            placed_bad += 1

    if col_bad_candidates:
        _slack, _rnd, bad_col_color, bad_col_lengths = col_bad_candidates[0]
        _min_row, _max_row, min_col, max_col = bounds_by_color[int(bad_col_color)]
        bad_col_length = int(random.choice(bad_col_lengths))
        bad_col = _place_interval_center(
            int(bad_col_length),
            col_occupied,
            int(ic0),
            int(ic1),
            color=int(bad_col_color),
            center_lo=int(min_col),
            center_hi=int(max_col),
        )
        if bad_col is None:
            bad_col = _place_interval_center(
                int(bad_col_length),
                col_occupied,
                int(ic0),
                int(ic1),
                color=int(bad_col_color),
            )
        if bad_col is not None:
            col_centers[int(bad_col_color)].append(int(bad_col))
            col_lengths[int(bad_col_color)].append(int(bad_col_length))
            placed_bad += 1

    if placed_bad == 0:
        _bump(stats, "bad_segments")
        return None

    for mc in mover_colors:
        min_row, max_row, min_col, max_col = bounds_by_color[int(mc)]
        sh, sw = dims_by_color[int(mc)]
        extra_rows = _add_optional_valid_segment(
            color=int(mc),
            length=int(sh),
            center_lo=int(min_row),
            center_hi=int(max_row),
            occupied=row_occupied,
            lo=int(ir0),
            hi=int(ir1),
            count_bias=0.4,
        )
        extra_cols = _add_optional_valid_segment(
            color=int(mc),
            length=int(sw),
            center_lo=int(min_col),
            center_hi=int(max_col),
            occupied=col_occupied,
            lo=int(ic0),
            hi=int(ic1),
            count_bias=0.4,
        )
        for center in extra_rows:
            row_centers[int(mc)].append(int(center))
            row_lengths[int(mc)].append(int(sh))
        for center in extra_cols:
            col_centers[int(mc)].append(int(center))
            col_lengths[int(mc)].append(int(sw))

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

    cur_row = 0
    for item in mover_info:
        sp = item["sprite"]
        sh, sw = map(int, sp.shape)
        if cur_row + sh > int(H) or sw > int(sprite_area_w):
            _bump(stats, "place_movers")
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(int(sh)):
            for dc in range(int(sw)):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[int(dr), int(dc)])
        cur_row += int(sh)

    for dc_color in distractor_colors:
        remaining_h = int(H - cur_row)
        if remaining_h < 2:
            _bump(stats, "distractor_height")
            return None
        max_h_here = min(4, int(remaining_h))
        max_area_here = min(7, int(max_h_here * sprite_area_w))
        if max_area_here < 3:
            _bump(stats, "distractor_area")
            return None
        d_area = random.randint(3, int(max_area_here))
        ds = _make_sprite(int(dc_color), int(max_h_here), int(sprite_area_w), int(d_area))
        if ds is None:
            _bump(stats, "make_distractor")
            return None
        sh, sw = map(int, ds.shape)
        if cur_row + sh > int(H) or sw > int(sprite_area_w):
            _bump(stats, "place_distractor")
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(int(sh)):
            for dc in range(int(sw)):
                if int(ds[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(ds[int(dr), int(dc)])
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        _bump(stats, "bg_majority")
        return None

    try:
        combo = main(inp)
        position = _solve_with_modes(inp, use_size_match=True, order_mode="position")
        color_only = _solve_with_modes(inp, use_size_match=False, order_mode="area_asc")
        metrics = _fusion_metrics(inp)
    except Exception:
        _bump(stats, "solve_exception")
        return None

    if np.array_equal(inp, combo):
        _bump(stats, "no_change")
        return None
    if np.array_equal(combo, position):
        _bump(stats, "no_order_diff")
        return None
    if np.array_equal(combo, color_only):
        _bump(stats, "no_size_diff")
        return None
    if int(metrics["order_diff"]) != 1 or int(metrics["size_diff"]) != 1:
        _bump(stats, "metrics_diff")
        return None
    if int(metrics["bad_total_centers"]) < 1:
        _bump(stats, "metrics_bad_segments")
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 800 attempts")
