# Variant of 247ef758
# Dimension: D13 — 像素裁剪行为 (meta)
# Change: 精灵印刻时超出框架内部边界的像素被裁剪（不绘制）... → 超出部分绘制在框架边缘上

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D13 variant
# -----------------------------------
# a) Concrete behavior and trace
#    Original task: sprites are centered at marker-defined interior cells, and any
#    stamped pixel outside the frame interior [ir0..ir1]x[ic0..ic1] is discarded.
#    Variant: pixels may spill one layer outward onto the frame border
#    [fr0..fr1]x[fc0..fc1]. They still may not go outside the outer rectangle.
#    Example: if a sprite has a top pixel at relative offset (-1, +1) and its
#    target row is ir0, original code drops that pixel because rr=fr0 is above the
#    interior; variant keeps it and paints the top edge at column tc+1.
#    This is the only transformation change. Input parsing, marker matching, and
#    mover ordering stay the same.
#
# b) Complexity calibration
#    We keep 2-3 movers and 0-1 distractors, 1-2 row marks and 1-2 col marks per
#    mover, and no row/col mark reuse across movers. To make D13 visible, the
#    generator guarantees at least one mover has a target causing a border hit.
#    To avoid clutter, other movers prefer non-border-touching targets when
#    available.
#
# c) Hidden assumptions review
#    - largest_enclosed_component_bbox(...) still assumes the frame interior is the
#      largest enclosed background component. Generator preserves that.
#    - divider detection still assumes a near-solid single-color column on the
#      left. Generator preserves that.
#    - sprite extraction still assumes left-strip sprites are separate 8-connected
#      components by color. Generator preserves that.
#    - marker detection still assumes targets are interior row/col coordinates read
#      from mirrored frame edges. Generator preserves that.
#    - The changed assumption is the final bounds check in stamping. The original
#      `if ir0 <= rr <= ir1 and ic0 <= cc <= ic1` relied on clipping-to-interior.
#      In the variant, that would silently erase the new behavior, so it must
#      become a frame-wide check `fr0..fr1` / `fc0..fc1`.
#
# d) Placement legality
#    For each sprite, generator computes exact occupied-pixel offsets relative to
#    the solver's center convention ((bbox start + bbox end)//2). Row/col marks are
#    sampled only from interior coordinates that keep every stamped pixel within
#    the full frame rectangle. Thus stamping can overwrite the border but never run
#    outside the frame.
#
# e) Visual observability
#    A border hit is not enough if it lands only on that mover's own marker cell,
#    because that edge cell already has the mover's color. Therefore the generator
#    forces one "special" mover to have an observable side profile: at least one
#    border-touching side has a nonzero orthogonal offset, so when it reaches the
#    border it paints at least one neighboring frame-edge cell too.
#
# f) Visual cleanliness
#    Only one mover is forced to demonstrate the variant. Other movers avoid
#    touching the border if safe positions exist, so the frame remains legible and
#    the change reads as "sprite can overwrite the frame edge" rather than general
#    chaos.
#
# g) Color inferability
#    No new color mapping is introduced. Sprite color still matches row/col marker
#    color. The only change is geometric: some copied pixels now remain visible on
#    the border instead of being clipped away.


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
        if bool((rs == 0).any() or (rs == int(h - 1)).any() or (cs == 0).any() or (cs == int(w - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Variant rule:
    - If a centered sprite extends beyond the frame interior, the overflow is allowed to
      paint the 1-cell frame border instead of being clipped away.
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
                {"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m}
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

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

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
                if int(fr0) <= rr <= int(fr1) and int(fc0) <= cc3 <= int(fc1):
                    out[rr, cc3] = int(col)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _pick_spaced_with_required(
    candidates: list[int], n: int, min_spacing: int, required: int | None = None
) -> list[int] | None:
    if not candidates:
        return None
    ordered = sorted(set(map(int, candidates)))
    if required is None:
        return _pick_spaced(ordered, n, min_spacing)
    if required not in ordered:
        return None
    if n == 1:
        return [int(required)]
    rest = [v for v in ordered if v != int(required)]
    random.shuffle(rest)
    picked = [int(required)]
    for v in rest:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return sorted(picked)
    return sorted(picked) if picked else None


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
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _sprite_geometry(sprite: np.ndarray) -> dict[str, object]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite != 0)]
    sh, sw = sprite.shape
    cr = (int(sh) - 1) // 2
    cc = (int(sw) - 1) // 2
    rel = [(int(r - cr), int(c - cc)) for r, c in coords]

    min_dr = min(dr for dr, _ in rel)
    max_dr = max(dr for dr, _ in rel)
    min_dc = min(dc for _, dc in rel)
    max_dc = max(dc for _, dc in rel)

    top_dcs = [dc for dr, dc in rel if dr == min_dr]
    bot_dcs = [dc for dr, dc in rel if dr == max_dr]
    left_drs = [dr for dr, dc in rel if dc == min_dc]
    right_drs = [dr for dr, dc in rel if dc == max_dc]

    return {
        "height": int(sh),
        "width": int(sw),
        "min_dr": int(min_dr),
        "max_dr": int(max_dr),
        "min_dc": int(min_dc),
        "max_dc": int(max_dc),
        "top_observable": any(int(dc) != 0 for dc in top_dcs),
        "bottom_observable": any(int(dc) != 0 for dc in bot_dcs),
        "left_observable": any(int(dr) != 0 for dr in left_drs),
        "right_observable": any(int(dr) != 0 for dr in right_drs),
    }


def _candidate_targets(
    geom: dict[str, object], interior0: int, interior1: int, frame0: int, frame1: int, axis: str
) -> tuple[list[int], list[int], list[int]]:
    if axis == "row":
        min_rel = int(geom["min_dr"])
        max_rel = int(geom["max_dr"])
        side1 = int(frame0 - min_rel)
        side2 = int(frame1 - max_rel)
        obs1 = bool(geom["top_observable"])
        obs2 = bool(geom["bottom_observable"])
    else:
        min_rel = int(geom["min_dc"])
        max_rel = int(geom["max_dc"])
        side1 = int(frame0 - min_rel)
        side2 = int(frame1 - max_rel)
        obs1 = bool(geom["left_observable"])
        obs2 = bool(geom["right_observable"])

    lo = max(int(interior0), int(frame0 - min_rel))
    hi = min(int(interior1), int(frame1 - max_rel))
    if lo > hi:
        return [], [], []

    all_vals = list(range(int(lo), int(hi) + 1))
    touch_all: list[int] = []
    touch_observable: list[int] = []
    for side_val, obs in ((side1, obs1), (side2, obs2)):
        if int(lo) <= int(side_val) <= int(hi):
            touch_all.append(int(side_val))
            if obs:
                touch_observable.append(int(side_val))
    touch_all = sorted(set(touch_all))
    touch_observable = sorted(set(touch_observable))
    return all_vals, touch_all, touch_observable


def _select_special_index(geoms: list[dict[str, object]]) -> int | None:
    viable = []
    for idx, geom in enumerate(geoms):
        if bool(geom["top_observable"] or geom["bottom_observable"] or geom["left_observable"] or geom["right_observable"]):
            viable.append(int(idx))
    return random.choice(viable) if viable else None


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
    fr0, fr1 = 0, h - 1
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
    geoms: list[dict[str, object]] = []
    for mc in mover_colors:
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)
        geoms.append(_sprite_geometry(s))

    special_idx = _select_special_index(geoms)
    if special_idx is None:
        return None
    if special_idx != 0:
        mover_colors[0], mover_colors[special_idx] = mover_colors[special_idx], mover_colors[0]
        sprites[0], sprites[special_idx] = sprites[special_idx], sprites[0]
        geoms[0], geoms[special_idx] = geoms[special_idx], geoms[0]

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    forced_variant = False

    for idx_m, (mc, sp, geom) in enumerate(zip(mover_colors, sprites, geoms)):
        sh, sw = sp.shape
        row_all, row_touch, row_touch_obs = _candidate_targets(geom, ir0, ir1, fr0, fr1, "row")
        col_all, col_touch, col_touch_obs = _candidate_targets(geom, ic0, ic1, fc0, fc1, "col")
        if not row_all or not col_all:
            return None

        row_safe = [r for r in row_all if r not in row_touch]
        col_safe = [c for c in col_all if c not in col_touch]

        avail_rows = [r for r in row_all if r not in occupied_rows]
        avail_cols = [c for c in col_all if c not in occupied_cols]
        if not avail_rows or not avail_cols:
            return None

        row_required = None
        col_required = None
        if idx_m == 0:
            row_force = [r for r in row_touch_obs if r not in occupied_rows]
            col_force = [c for c in col_touch_obs if c not in occupied_cols]
            if row_force or col_force:
                if row_force and col_force:
                    if random.random() < 0.5:
                        row_required = random.choice(row_force)
                    else:
                        col_required = random.choice(col_force)
                elif row_force:
                    row_required = random.choice(row_force)
                else:
                    col_required = random.choice(col_force)
            else:
                return None

        row_candidates = [r for r in row_safe if r not in occupied_rows] if idx_m != 0 and any(
            r not in occupied_rows for r in row_safe
        ) else avail_rows
        col_candidates = [c for c in col_safe if c not in occupied_cols] if idx_m != 0 and any(
            c not in occupied_cols for c in col_safe
        ) else avail_cols

        row_n = random.randint(1, 2)
        col_n = random.randint(1, 2)
        row_marks = _pick_spaced_with_required(row_candidates, row_n, max(1, sh), row_required)
        col_marks = _pick_spaced_with_required(col_candidates, col_n, max(1, sw), col_required)
        if row_marks is None:
            row_marks = _pick_spaced_with_required(avail_rows, row_n, max(1, sh), row_required)
        if col_marks is None:
            col_marks = _pick_spaced_with_required(avail_cols, col_n, max(1, sw), col_required)
        if not row_marks or not col_marks:
            return None

        mover_targets[int(mc)] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)
        if row_required is not None or col_required is not None:
            forced_variant = True

    if not forced_variant:
        return None

    inp = np.full((h, w), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[h - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, h - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = int(mc)
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, h - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for _color, sp, _ in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > h:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    if _edge_overwrite_count(inp, output) <= 0:
        return None
    return inp


def _edge_overwrite_count(inp: np.ndarray, out: np.ndarray) -> int:
    bg = int(get_bg(inp))
    inner_bbox = largest_enclosed_component_bbox(inp == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    edge = np.zeros_like(inp, dtype=bool)
    edge[fr0, fc0 : fc1 + 1] = True
    edge[fr1, fc0 : fc1 + 1] = True
    edge[fr0 : fr1 + 1, fc0] = True
    edge[fr0 : fr1 + 1, fc1] = True
    return int(np.count_nonzero(edge & (inp != out)))


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
