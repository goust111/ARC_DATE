# Combo variant of 20270e3b (index 20)
# combo_spec: D2/0+D4/2+D7/0
# Name: 水平重力固定画布

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


"""
Design analysis for the fused combo
==================================

a) Concrete behavior and trace
   - Source task:
     * each sprite has one horizontal ctrl bar;
     * the plug is translated so its ctrl row sits exactly one row above the base ctrl row;
     * ctrl overlaps at the seam, then everything is tight-cropped.
   - Fused task:
     * each sprite instead has a vertical ctrl bar of length 2 or 3, so the direction
       change is visually obvious;
     * the connector only fixes vertical alignment: the two ctrl bars share the same rows;
     * the horizontal position is chosen by sideways "gravity":
       - if the base connector's body is on the right, the plug slides in from the left;
       - if the base connector's body is on the left, the plug slides in from the right;
       - move the plug as far toward the base as possible without overlap;
       - therefore the final contact row can be away from the connector rows.
     * ctrl is recolored to obj and the merged tight sprite is pasted at output (0,0) on a
       full-size canvas matching the input shape.
   - Small trace:
     * suppose the base connector is a vertical bar at column 5 with body on its right,
       so the plug must approach from the left.
     * direct side-docking would place the plug connector at column 4.
     * in this combo, if the base also has a non-connector protrusion at column 4 on some
       aligned row, the plug hits that protrusion first and stops with its nearest column at 3.
     * difference vs source: orientation changes (vertical bars), placement target changes
       (gravity contact rather than connector seam), and the final answer keeps the full input
       size instead of cropping tightly.

b) Complexity calibration
   - Connector length is restricted to 2 or 3 so vertical orientation is visible.
   - The generator intentionally keeps only cases with positive horizontal slide delta:
     gravity landing must differ from the "connector-adjacent" column, otherwise the combo
     would visually collapse toward a plain side-docking task.
   - Shapes remain thin 8-connected orthogonal blobs with one connector each, which preserves
     the original object grammar and avoids introducing too many new freedoms at once.

c) Implicit assumptions affected by the combo
   - Original `main()` assumed horizontal bars and vertical seam formulas; those are invalid.
   - Original overlap-at-connector logic is invalid because gravity landing is disjoint:
     the plug stops before overlap and contact is edge-adjacency.
   - Original final tight crop is no longer the output contract; it is now only an internal
     intermediate sprite that gets pasted into a full-size bg canvas.
   - New solver assumptions:
     * each connector is a single vertical bar;
     * the base connector's body side uniquely determines from which side the plug slides;
     * after connector-row alignment there is at least one shared row between occupied pixels,
       so a finite horizontal landing exists;
     * the merged tight sprite fits within the input canvas. The generator enforces all of these.

d) Placement legality
   - The generator builds valid cropped base/plug sprites first, computes the gravity-merged
     result, then chooses an input canvas large enough for both the merged output and the
     scattered plug.
   - Base is pasted at input origin so role inference remains observable.
   - Plug is scattered elsewhere with padding=1 to keep exactly two 8-connected components.
   - The merged output is pasted at output origin on a same-size canvas.

e) Visual observability
   - Human-solvable cues:
     * ctrl bars are vertical, not horizontal;
     * final connection happens from the side, not from above;
     * the plug often stops earlier than connector-adjacent placement, so contact rows visibly
       differ from the connector rows;
     * output size matches the input size with extra bg on the right and/or bottom.
   - These changes are all visible in 2-4 examples without relying on hidden numeric ordering.

f) Visual cleanliness
   - Final assembly is disjoint before recoloring, so there is no ambiguous multi-layer overlap.
   - Colors stay fixed at bg=1, obj=4, ctrl=7; ctrl disappears uniformly after docking.
   - Only two sprites are present in the input, and the scattered plug is kept away from the
     base so the scene remains easy to parse.

g) Color-change inferability
   - The only color change is still ctrl -> obj after docking.
   - Because ctrl marks the vertical bars that also define alignment, the recoloring rule remains
     strongly tied to structure rather than to arbitrary color remapping.
"""


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _conn_pts(
    grid: np.ndarray, sprite: set[tuple[int, int]], ctrl: int
) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for (r, c) in sprite if int(grid[int(r), int(c)]) == int(ctrl)}


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in conn_pts]
    cs = [int(c) for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _assert_vertical_bar(shape: set[tuple[int, int]]) -> None:
    cols = {int(c) for _r, c in shape}
    assert len(cols) == 1
    rows = sorted(int(r) for r, _c in shape)
    assert len(rows) == len(shape) >= 2
    assert rows == list(range(rows[0], rows[-1] + 1))


def _connector_body_side(
    sprite: set[tuple[int, int]],
    conn_pts: set[tuple[int, int]],
) -> int:
    body = sprite - conn_pts
    valid: list[int] = []
    for side in (-1, 1):
        body_ok = all((int(r), int(c + side)) in body for r, c in conn_pts)
        free_ok = all((int(r), int(c - side)) not in body for r, c in conn_pts)
        if body_ok and free_ok:
            valid.append(int(side))
    assert len(valid) == 1
    return int(valid[0])


def _direct_adjacent_tc(
    conn_base: set[tuple[int, int]],
    conn_plug: set[tuple[int, int]],
    base_side: int,
) -> int:
    _sb, (_dr0, dc0) = _norm_conn(conn_base)
    _sp, (_pr0, pc0) = _norm_conn(conn_plug)
    return int((dc0 - base_side) - pc0)


def _horizontal_gravity_shift(
    base: set[tuple[int, int]],
    conn_base: set[tuple[int, int]],
    plug: set[tuple[int, int]],
    conn_plug: set[tuple[int, int]],
    base_side: int,
) -> tuple[int, int, set[tuple[int, int]]]:
    """
    Align connector rows, then slide the plug horizontally toward the base until
    one more step would overlap.
    """
    _sb, (dr0, _dc0) = _norm_conn(conn_base)
    _sp, (pr0, _pc0) = _norm_conn(conn_plug)
    tr = int(dr0 - pr0)
    move_dc = int(base_side)

    base_cols_by_row: dict[int, list[int]] = {}
    for r, c in base:
        base_cols_by_row.setdefault(int(r), []).append(int(c))

    tc_candidates: list[int] = []
    for rp, cp in plug:
        row = int(rp + tr)
        for cb in base_cols_by_row.get(row, []):
            if move_dc == 1:
                tc_candidates.append(int(cb - cp - 1))
            else:
                tc_candidates.append(int(cb - cp + 1))
    assert tc_candidates, "row alignment must yield a finite horizontal landing"

    tc = int(min(tc_candidates) if move_dc == 1 else max(tc_candidates))
    shifted = {(int(r + tr), int(c + tc)) for r, c in plug}
    assert base.isdisjoint(shifted), "gravity landing must stop before overlap"
    assert any((int(r), int(c + move_dc)) in base for r, c in shifted), "landing needs support"
    return int(tr), int(tc), shifted


def _render_shifted_union(
    base_grid: np.ndarray,
    base: set[tuple[int, int]],
    plug_grid: np.ndarray,
    plug: set[tuple[int, int]],
    tr: int,
    tc: int,
    *,
    bg: int,
    obj: int,
    ctrl: int,
) -> np.ndarray:
    shifted = {(int(r + tr), int(c + tc)) for r, c in plug}
    pts_all = base | shifted
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(base_grid[int(r), int(c)])

    for r, c in shifted:
        assert out[int(r - rr0), int(c - cc0)] == int(bg)
        out[int(r - rr0), int(c - cc0)] = int(plug_grid[int(r - tr), int(c - tc)])

    out[out == int(ctrl)] = int(obj)
    cropped = _crop_non_bg(out, bg)
    assert cropped is not None
    assert is_contiguous(cropped != int(bg), background=0, connectivity=8)
    return cropped


def _extract_roles(
    grid: np.ndarray,
    *,
    bg: int,
    ctrl: int,
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
]:
    lab, n = label_mask(grid != int(bg), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    a, b = comps

    conn_a = _conn_pts(grid, a, ctrl)
    conn_b = _conn_pts(grid, b, ctrl)
    assert conn_a and conn_b

    shp_a, _ = _norm_conn(conn_a)
    shp_b, _ = _norm_conn(conn_b)
    _assert_vertical_bar(shp_a)
    _assert_vertical_bar(shp_b)
    assert shp_a == shp_b

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(a)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(b)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        return a, conn_a, b, conn_b
    return b, conn_b, a, conn_a


def _assemble_tight(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    base, conn_base, plug, conn_plug = _extract_roles(g, bg=bg, ctrl=ctrl)
    base_side = _connector_body_side(base, conn_base)
    plug_side = _connector_body_side(plug, conn_plug)
    assert plug_side == -base_side

    tr, tc, _plug_shift = _horizontal_gravity_shift(base, conn_base, plug, conn_plug, base_side)
    merged = _render_shifted_union(
        g,
        base,
        g,
        plug,
        tr,
        tc,
        bg=bg,
        obj=obj,
        ctrl=ctrl,
    )
    direct_tc = _direct_adjacent_tc(conn_base, conn_plug, base_side)
    meta = {
        "base_side": int(base_side),
        "gravity_tr": int(tr),
        "gravity_tc": int(tc),
        "direct_tc": int(direct_tc),
        "slide_delta": int(abs(tc - direct_tc)),
    }
    return merged, meta


def _merge_cropped_sprites(
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    *,
    bg: int,
    obj: int,
    ctrl: int,
) -> tuple[np.ndarray, dict[str, int]]:
    base = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    plug = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}
    conn_base = _conn_pts(base_sprite, base, ctrl)
    conn_plug = _conn_pts(plug_sprite, plug, ctrl)
    assert conn_base and conn_plug

    shp_a, _ = _norm_conn(conn_base)
    shp_b, _ = _norm_conn(conn_plug)
    _assert_vertical_bar(shp_a)
    _assert_vertical_bar(shp_b)
    assert shp_a == shp_b

    base_side = _connector_body_side(base, conn_base)
    plug_side = _connector_body_side(plug, conn_plug)
    assert plug_side == -base_side

    tr, tc, _plug_shift = _horizontal_gravity_shift(base, conn_base, plug, conn_plug, base_side)
    merged = _render_shifted_union(
        base_sprite,
        base,
        plug_sprite,
        plug,
        tr,
        tc,
        bg=bg,
        obj=obj,
        ctrl=ctrl,
    )
    direct_tc = _direct_adjacent_tc(conn_base, conn_plug, base_side)
    meta = {
        "base_side": int(base_side),
        "gravity_tr": int(tr),
        "gravity_tc": int(tc),
        "direct_tc": int(direct_tc),
        "slide_delta": int(abs(tc - direct_tc)),
    }
    return merged, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fuse two sprites by aligning their vertical connector rows, then slide the plug
    horizontally into the base under sideways gravity. Recolor ctrl -> obj and place
    the merged tight sprite at the top-left of a full-size output canvas.
    """
    g = np.array(grid, dtype=int)
    bg = 1
    h, w = map(int, g.shape)

    merged, _meta = _assemble_tight(g)
    mh, mw = map(int, merged.shape)
    assert mh <= h and mw <= w

    out = np.full((h, w), int(bg), dtype=int)
    paste(out, merged, 0, 0, transparent=int(bg))
    return out


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray | None = None,
) -> None:
    if allowed is None:
        allowed = np.ones_like(mask, dtype=bool)
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (
                0 <= r + dr < h
                and 0 <= c + dc < w
                and allowed[int(r + dr), int(c + dc)]
                and not mask[int(r + dr), int(c + dc)]
            )
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        frontier.append((int(nr), int(nc)))
        count += 1


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    pixels = [(int(r), int(c)) for r, c in zip(*np.where(mask))]
    _rng.shuffle(pixels)
    h, w = mask.shape
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _make_base_sprite(
    *,
    bg: int,
    obj: int,
    ctrl: int,
) -> tuple[np.ndarray, dict[str, int]] | None:
    base_h = _rng.randint(4, 7)
    base_w = _rng.randint(4, 9)
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    base_target = _rng.randint(6, min(24, base_h * base_w))
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    candidates: list[tuple[int, int, int, int, int]] = []
    for side in (-1, 1):
        for conn_len in range(2, min(3, base_h) + 1):
            for conn_col in range(base_w):
                body_col = int(conn_col + side)
                impact_col = int(conn_col - side)
                if not (0 <= body_col < base_w and 0 <= impact_col < base_w):
                    continue
                for conn_r0 in range(base_h - conn_len + 1):
                    rows = list(range(conn_r0, conn_r0 + conn_len))
                    if not all(base[int(r), int(conn_col)] for r in rows):
                        continue
                    if not all(base[int(r), int(body_col)] for r in rows):
                        continue
                    if any(base[int(r), int(impact_col)] for r in rows):
                        continue
                    impact_rows = [
                        int(r)
                        for r in range(base_h)
                        if int(r) not in rows and base[int(r), int(impact_col)]
                    ]
                    if not impact_rows:
                        continue
                    candidates.append(
                        (int(side), int(conn_len), int(conn_r0), int(conn_col), _rng.choice(impact_rows))
                    )
    if not candidates:
        return None

    side, conn_len, conn_r0, conn_col, impact_row = _rng.choice(candidates)
    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base] = int(obj)
    for r in range(conn_r0, conn_r0 + conn_len):
        base_grid[int(r), int(conn_col)] = int(ctrl)

    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None
    if int(np.sum(base_grid != int(bg))) < 5:
        return None

    cropped = _crop_non_bg(base_grid, bg)
    if cropped is None:
        return None
    return cropped, {
        "side": int(side),
        "conn_len": int(conn_len),
        "conn_r0": int(conn_r0),
        "impact_row": int(impact_row),
    }


def _make_plug_sprite(
    base_meta: dict[str, int],
    *,
    bg: int,
    obj: int,
    ctrl: int,
) -> np.ndarray | None:
    side = int(base_meta["side"])
    conn_len = int(base_meta["conn_len"])
    base_conn_r0 = int(base_meta["conn_r0"])
    impact_row = int(base_meta["impact_row"])
    plug_body_side = int(-side)

    for _ in range(80):
        plug_h = _rng.randint(max(3, conn_len + 1), 7)
        plug_w = _rng.randint(3, 9)
        if plug_body_side == 1:
            if plug_w < 2:
                continue
            plug_conn_col = _rng.randint(0, plug_w - 2)
        else:
            if plug_w < 2:
                continue
            plug_conn_col = _rng.randint(1, plug_w - 1)

        valid_conn_r0 = []
        for plug_conn_r0 in range(plug_h - conn_len + 1):
            impact_local = int(impact_row - base_conn_r0 + plug_conn_r0)
            if 0 <= impact_local < plug_h:
                valid_conn_r0.append((int(plug_conn_r0), int(impact_local)))
        if not valid_conn_r0:
            continue
        plug_conn_r0, impact_local = _rng.choice(valid_conn_r0)

        plug = np.zeros((plug_h, plug_w), dtype=bool)
        keep: set[tuple[int, int]] = set()
        body_col = int(plug_conn_col + plug_body_side)

        for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
            plug[int(r), int(plug_conn_col)] = True
            plug[int(r), int(body_col)] = True
            keep.add((int(r), int(plug_conn_col)))
            keep.add((int(r), int(body_col)))

        r0 = min(int(impact_local), int(plug_conn_r0))
        r1 = max(int(impact_local), int(plug_conn_r0 + conn_len - 1))
        for r in range(r0, r1 + 1):
            plug[int(r), int(plug_conn_col)] = True
            keep.add((int(r), int(plug_conn_col)))

        plug[int(impact_local), int(body_col)] = True
        keep.add((int(impact_local), int(body_col)))

        plug_target_min = max(6, int(plug.sum()))
        plug_target_max = min(24, plug_h * plug_w)
        if plug_target_min > plug_target_max:
            continue
        _grow_shape(plug, _rng.randint(plug_target_min, plug_target_max), list(zip(*np.where(plug))))
        if not is_contiguous(plug, background=0, connectivity=8):
            continue
        _thin(plug, keep=keep)
        if not is_contiguous(plug, background=0, connectivity=8):
            continue

        free_col = int(plug_conn_col - plug_body_side)
        ok = True
        for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
            if not plug[int(r), int(body_col)]:
                ok = False
                break
            if 0 <= free_col < plug_w and plug[int(r), int(free_col)]:
                ok = False
                break
        if not ok:
            continue

        plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
        plug_grid[plug] = int(obj)
        for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
            plug_grid[int(r), int(plug_conn_col)] = int(ctrl)

        if not is_contiguous(plug_grid != int(bg), background=0, connectivity=8):
            continue
        if int(np.sum(plug_grid != int(bg))) < 5:
            continue

        cropped = _crop_non_bg(plug_grid, bg)
        if cropped is not None:
            return cropped
    return None


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    made = _make_base_sprite(bg=bg, obj=obj, ctrl=ctrl)
    if made is None:
        return None
    base_sprite, base_meta = made

    plug_sprite = _make_plug_sprite(base_meta, bg=bg, obj=obj, ctrl=ctrl)
    if plug_sprite is None:
        return None

    if _rng.random() < 0.5:
        base_sprite = np.fliplr(base_sprite).copy()
        plug_sprite = np.fliplr(plug_sprite).copy()

    merged, meta = _merge_cropped_sprites(base_sprite, plug_sprite, bg=bg, obj=obj, ctrl=ctrl)
    if int(meta["slide_delta"]) < 1:
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    mh, mw = map(int, merged.shape)
    if mh > 11 or mw > 13:
        return None

    size_choices = [
        (int(h), int(w))
        for h in range(max(7, bh, ph, mh), 12)
        for w in range(max(5, bw, pw, mw), 14)
        if h > mh or w > mw
    ]
    if not size_choices:
        return None
    canvas_h, canvas_w = _rng.choice(size_choices)

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(bg))
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=int(bg), padding=1)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=int(bg))

    if np.sum(canvas == int(bg)) <= canvas.size // 2:
        return None

    expected = np.full_like(canvas, int(bg))
    paste(expected, merged, 0, 0, transparent=int(bg))

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
