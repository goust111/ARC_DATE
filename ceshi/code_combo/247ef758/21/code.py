# Combo variant of 247ef758 (index 21)
# combo_spec: D4/0+D6/0+D7/1
# Name: 重着色+裁剪输出+面积排序
#
# Design analysis
# ---------------
# a) Concretization and behavior trace
#    - Baseline scene: a left sprite strip, one full-height divider column, and a framed rectangle on the right.
#      Matching colors on left/right frame edges choose target rows; matching colors on top/bottom edges choose
#      target cols. A sprite is centered at every row x col target.
#    - Original task:
#      1. erase mover sprites from the left strip
#      2. stamp each mover in its own source color
#      3. keep the whole input-sized canvas
#      4. overlap order is by vertical position in the left strip: lower first, upper last
#    - Fused combo:
#      1. the same target inference still happens from edge markers
#      2. movers are still erased from the left strip
#      3. stamping order changes to ascending visible sprite area, so larger sprites are written later
#      4. every stamped foreground pixel is recolored to the frame base color
#      5. final output is cropped tightly to the frame interior only
#    - Concrete mini-trace:
#      Suppose a 3-pixel red sprite and a 7-pixel yellow sprite have overlapping targets.
#      In the original task, if yellow is lower in the strip and red is higher, yellow stamps first and red
#      stamps last in red, so the overlap ends red. In this combo, the 3-pixel sprite stamps first and the
#      7-pixel sprite stamps last, but both write frame_base, so the final visible overlap is still frame_base.
#      The order still exists as the write order, but D4 collapses its visible color consequence.
#
# b) Complexity calibration
#    - Recoloring is kept global and simple: every mover writes the same frame_base color.
#    - Cropping is also simple and global: output is always exactly the frame interior.
#    - Area sorting is preserved as the canonical write order, but because D4 makes all stamped pixels the same
#      color, D7 becomes visually silent in the final combo output. To avoid silently pretending otherwise, the
#      generator explicitly measures whether area-order would create a visible difference in the source-colored
#      pre-recolor version, and accepts only such samples.
#    - To keep examples readable despite recolor collapsing colors, the generator uses only 2-3 movers and keeps
#      target counts low, usually one primary row mark and one primary col mark per mover.
#
# c) Implicit assumption audit
#    - `largest_enclosed_component_bbox(g == bg)` assumes the frame interior is the largest background component
#      that does not touch the border. The generator preserves that exact layout.
#    - Divider detection assumes at least one near-full-height uniform non-bg column exists left of the frame.
#      The generator still places exactly one such divider.
#    - Sprite extraction assumes movers live strictly left of the frame and are 8-connected single-color blobs.
#      The generator preserves that.
#    - Marker parsing assumes mirrored colors on opposite frame edges. The generator preserves that.
#    - Recoloring adds a new solver need: infer `frame_base`. We do that as the most frequent non-bg color on
#      the frame border, so the generator must keep frame_base in strict majority on that border.
#    - Area sorting adds a new solver assumption: if two mover areas tie, the intended rule is ambiguous from
#      examples. The generator therefore forces distinct mover areas. The solver still uses deterministic ties.
#    - D4 + D7 interaction creates a genuine conflict: once all movers are recolored to the same frame_base,
#      overlap order no longer changes the visible cropped output. This is not fixable by generator layout
#      alone without violating the combo spec. The code therefore keeps the D7 order as a latent write-order
#      rule and exposes metrics that show the source-colored counterfactual does differ by area order.
#
# d) Placement legality
#    - Each target center is chosen from the range where the full sprite bbox fits inside the interior.
#    - Different movers do not reuse their primary row or col marks.
#    - Sprites in the left strip are stacked without overlap.
#    - Target centers are clustered so source-colored area-order would differ from position-order, but still
#      remain legal after centering and clipping.
#
# e) Visual observability
#    - D4 is directly visible: movers of several input colors become one frame_base color in output.
#    - D6 is directly visible: output shape is smaller and equals the frame interior.
#    - D7 is only indirectly observable in this combo because D4 removes its visible color consequence.
#      The generator therefore guarantees that if recolor were disabled, area-order would visibly matter.
#      This is the strongest compatible realization of D7 under the requested fusion.
#
# f) Visual cleanliness
#    - The cropped output contains only the interior, so border noise and distractors disappear.
#    - Overlap is kept moderate by using few movers and few targets; otherwise recolored unions would become
#      unreadably dense.
#
# g) Change inferability
#    - Color change is structurally inferable because all moved shapes adopt the surrounding frame's base color.
#    - The crop boundary is structurally inferable because it is exactly the empty rectangle inside the frame.
#    - The area-order component is preserved as solver logic and as a generator-side structural metric, but the
#      final visible combo intentionally collapses it. This limitation is documented rather than hidden.
#
# main() post-change assumption review
#    - `_frame_base_color(...)`: assumes frame_base is the dominant non-bg border color.
#    - `_extract_scene(...)`: assumes each mover color appears as a left-strip component and matching row/col
#      marks exist for mover colors only.
#    - `_solve_scene(...)`: assumes targets are legal centering positions. The generator guarantees this by
#      choosing target centers inside each sprite's admissible center bounds.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Bbox of the largest connected component in `mask` that does not touch the grid border.
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
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))
    return best_bbox


def _frame_base_color(g: np.ndarray, bg: int, fr0: int, fc0: int, fr1: int, fc1: int) -> int:
    """
    Most frequent non-bg color on the rectangular frame border.
    """
    border_vals: list[int] = []
    border_vals.extend(map(int, g[int(fr0), int(fc0) : int(fc1) + 1]))
    border_vals.extend(map(int, g[int(fr1), int(fc0) : int(fc1) + 1]))
    border_vals.extend(int(g[r, int(fc0)]) for r in range(int(fr0) + 1, int(fr1)))
    border_vals.extend(int(g[r, int(fc1)]) for r in range(int(fr0) + 1, int(fr1)))
    counts = Counter(v for v in border_vals if int(v) != int(bg))
    assert counts
    return int(counts.most_common(1)[0][0])


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
    frame_base = _frame_base_color(g, bg, fr0, fc0, fr1, fc1)

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
            comp_mask = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": comp_mask,
                    "area": int(comp_mask.sum()),
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
        "grid": g,
        "bg": int(bg),
        "inner_bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
        "frame_bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
        "frame_base": int(frame_base),
        "sprites": sprites,
        "targets": targets,
    }


def _sorted_movers(sprites: list[dict[str, object]], targets: dict[int, list[tuple[int, int]]], *, order_mode: str) -> list[dict[str, object]]:
    movers = [s for s in sprites if int(s["color"]) in targets]

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
    return movers


def _solve_scene(
    scene: dict[str, object],
    *,
    order_mode: str,
    color_mode: str,
    crop: bool,
) -> np.ndarray:
    g = np.array(scene["grid"], dtype=int)
    bg = int(scene["bg"])
    ir0, ic0, ir1, ic1 = map(int, scene["inner_bbox"])
    frame_base = int(scene["frame_base"])
    sprites = list(scene["sprites"])
    targets = dict(scene["targets"])

    out = g.copy()
    movers = _sorted_movers(sprites, targets, order_mode=order_mode)

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        mask = np.array(s["mask"], dtype=bool)
        for rr, cc in np.argwhere(mask):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    for s in movers:
        src_color = int(s["color"])
        stamp_color = int(frame_base) if color_mode == "frame_base" else int(src_color)
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(np.array(s["mask"], dtype=bool))]

        for tr, tc in targets[int(src_color)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(stamp_color)

    if crop:
        return out[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - infer mover targets from matching frame markers
    - stamp movers in ascending sprite-area order
    - recolor every stamped pixel to the frame base color
    - return only the frame interior crop
    """
    scene = _extract_scene(grid)
    return _solve_scene(scene, order_mode="area_asc", color_mode="frame_base", crop=True)


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    shuffled = list(candidates)
    random.shuffle(shuffled)
    picked: list[int] = []
    for v in shuffled:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_sprite_exact(color: int, max_h: int, max_w: int, n_pixels: int) -> np.ndarray | None:
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
            sprite[int(pr - min_r), int(pc - min_c)] = int(color)
        return sprite
    return None


def _center_bounds(ir0: int, ic0: int, ir1: int, ic1: int, sprite: np.ndarray) -> tuple[int, int, int, int]:
    sh, sw = sprite.shape
    min_row = int(ir0 + sh // 2)
    max_row = int(ir1 - (sh - 1 - sh // 2))
    min_col = int(ic0 + sw // 2)
    max_col = int(ic1 - (sw - 1 - sw // 2))
    return min_row, max_row, min_col, max_col


def _choose_cluster_centers(
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    sprites: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    """
    Pick nearby but distinct centers so source-colored overlap-order would matter.
    """
    n = len(sprites)
    row_offsets = list(range(n))
    col_offsets = list(range(n))

    row_anchor_lo = -10**9
    row_anchor_hi = 10**9
    col_anchor_lo = -10**9
    col_anchor_hi = 10**9
    for sp, ro, co in zip(sprites, row_offsets, col_offsets):
        min_row, max_row, min_col, max_col = _center_bounds(ir0, ic0, ir1, ic1, sp)
        row_anchor_lo = max(row_anchor_lo, int(min_row - ro))
        row_anchor_hi = min(row_anchor_hi, int(max_row - ro))
        col_anchor_lo = max(col_anchor_lo, int(min_col - co))
        col_anchor_hi = min(col_anchor_hi, int(max_col - co))
    if row_anchor_lo > row_anchor_hi or col_anchor_lo > col_anchor_hi:
        return None

    anchor_r = random.randint(int(row_anchor_lo), int(row_anchor_hi))
    anchor_c = random.randint(int(col_anchor_lo), int(col_anchor_hi))
    return [(int(anchor_r + ro), int(anchor_c + co)) for ro, co in zip(row_offsets, col_offsets)]


def _maybe_add_extra_mark(primary: int, candidates: list[int], occupied: set[int], min_spacing: int) -> list[int]:
    marks = [int(primary)]
    if random.random() < 0.25:
        extra_candidates = [v for v in candidates if v not in occupied and abs(v - primary) >= min_spacing]
        if extra_candidates:
            extra = int(random.choice(extra_candidates))
            marks.append(int(extra))
            occupied.add(int(extra))
    return sorted(marks)


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    scene = _extract_scene(inp)
    ir0, ic0, ir1, ic1 = map(int, scene["inner_bbox"])
    frame_base = int(scene["frame_base"])
    raw_interior = np.array(scene["grid"], dtype=int)[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1]

    combo_out = _solve_scene(scene, order_mode="area_asc", color_mode="frame_base", crop=True)
    area_source = _solve_scene(scene, order_mode="area_asc", color_mode="source", crop=True)
    position_source = _solve_scene(scene, order_mode="position", color_mode="source", crop=True)
    position_combo = _solve_scene(scene, order_mode="position", color_mode="frame_base", crop=True)

    mover_colors = {int(s["color"]) for s in scene["sprites"] if int(s["color"]) in scene["targets"]}
    source_colors_absent = int(all(not np.any(combo_out == int(col)) for col in mover_colors))
    return {
        "cropped_h": int(combo_out.shape[0]),
        "cropped_w": int(combo_out.shape[1]),
        "recolored_pixels": int(np.sum((combo_out == int(frame_base)) & (raw_interior != int(frame_base)))),
        "changed_from_raw": int(not np.array_equal(combo_out, raw_interior)),
        "source_colors_absent": int(source_colors_absent),
        "pre_recolor_order_delta": int(np.sum(area_source != position_source)),
        "combo_visible_order_delta": int(np.sum(combo_out != position_combo)),
    }


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
    n_movers = random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 0, 1])
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
        sp = _make_sprite_exact(int(mc), min(4, interior_h), sprite_area_w, int(area))
        if sp is None:
            return None
        mover_info.append({"color": int(mc), "sprite": sp, "area": int(area)})

    mover_info.sort(key=lambda item: (int(item["area"]), int(item["color"])))
    mover_sprites = [np.array(item["sprite"], dtype=int) for item in mover_info]
    centers = _choose_cluster_centers(ir0, ic0, ir1, ic1, mover_sprites)
    if centers is None:
        return None

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for item, (primary_r, primary_c) in zip(mover_info, centers):
        mc = int(item["color"])
        sp = np.array(item["sprite"], dtype=int)
        sh, sw = sp.shape
        min_row, max_row, min_col, max_col = _center_bounds(ir0, ic0, ir1, ic1, sp)
        if not (int(min_row) <= int(primary_r) <= int(max_row) and int(min_col) <= int(primary_c) <= int(max_col)):
            return None
        if int(primary_r) in occupied_rows or int(primary_c) in occupied_cols:
            return None
        occupied_rows.add(int(primary_r))
        occupied_cols.add(int(primary_c))

        row_marks = _maybe_add_extra_mark(
            int(primary_r),
            list(range(int(min_row), int(max_row) + 1)),
            occupied_rows,
            max(2, int(sh)),
        )
        col_marks = _maybe_add_extra_mark(
            int(primary_c),
            list(range(int(min_col), int(max_col) + 1)),
            occupied_cols,
            max(2, int(sw)),
        )
        mover_targets[int(mc)] = (row_marks, col_marks)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, int(fc0) : int(fc0 + frame_w)] = top_edge
    inp[int(h - 1), int(fc0) : int(fc0 + frame_w)] = top_edge

    for r in range(1, h - 1):
        val = int(frame_base)
        for item in mover_info:
            mc = int(item["color"])
            rmarks, _ = mover_targets[int(mc)]
            if int(r) in rmarks:
                val = int(mc)
                break
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    cur_row = 0
    for item in mover_info:
        sp = np.array(item["sprite"], dtype=int)
        sh, sw = sp.shape
        if int(sw) > int(sprite_area_w) or int(cur_row + sh) > int(h):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[int(dr), int(dc)])
        cur_row += int(sh)

    for dc_color in distractor_colors:
        remaining_h = int(h - cur_row)
        if remaining_h <= 0:
            return None
        d_area = random.randint(3, min(6, max(3, remaining_h * sprite_area_w)))
        ds = _make_sprite_exact(int(dc_color), min(3, remaining_h), sprite_area_w, int(d_area))
        if ds is None:
            return None
        sh, sw = ds.shape
        if int(sw) > int(sprite_area_w) or int(cur_row + sh) > int(h):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(sh):
            for dc in range(sw):
                if int(ds[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(ds[int(dr), int(dc)])
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        out = main(inp)
        metrics = _combo_metrics(inp)
    except Exception:
        return None

    if out.shape != (interior_h, interior_w):
        return None
    if np.array_equal(out, inp[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1]):
        return None
    if metrics["recolored_pixels"] <= 0:
        return None
    if metrics["changed_from_raw"] <= 0:
        return None
    if metrics["source_colors_absent"] <= 0:
        return None
    if metrics["pre_recolor_order_delta"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 800 attempts")
