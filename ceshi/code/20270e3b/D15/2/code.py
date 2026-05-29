# Variant of 20270e3b
# Dimension: D15 — 连接器自由侧约束 (meta)
# Change: base连接器自由侧朝下（下方无body像素），plug连接器自由侧朝上（上方无body像素）... → 连接器在精灵内部（两侧均有body）

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste

# Design analysis for the D15 variant
# - Concretization:
#   Old task used horizontal connectors as vertical docking tabs: base ctrl had body only
#   above, plug ctrl had body only below, so solver could always place plug one row above.
#   Here both connectors are internal in the vertical sense: every ctrl pixel has obj both
#   above and below inside its sprite. Because that removes the up/down "free side" cue, the
#   connector becomes a lateral alignment landmark instead. We fix one observable side-docking
#   rule: keep the base on the left, move the plug to the right, align the two connector rows,
#   and place the plug so its connector starts immediately after the base connector ends.
# - Small behavior trace:
#   Old: tr = (base_conn_row - 1) - plug_conn_row, tc = aligned connector start columns.
#   New: tr = base_conn_row - plug_conn_row, tc = (base_conn_right + 1) - plug_conn_left.
#   Old overlap was a 2 x k vertical docking zone; new docking has no overlap and produces a
#   horizontally merged shape.
# - Complexity calibration:
#   The bar length still has only 1-3 choices and must match across sprites, which keeps the
#   correspondence readable from a few examples. The new visual effect is obvious because the
#   output grows sideways instead of vertically and the ctrl bars are buried inside both
#   sprites rather than sitting on exposed top/bottom edges.
# - Implicit assumptions changed in main():
#   1. The old assertions "base below empty / plug above empty" are invalid and replaced with
#      "body exists on both sides vertically" for both sprites.
#   2. The old placement formula used a fixed row gap of 1 and identical connector columns.
#      That no longer holds; we now use row equality plus right-adjacency of connector spans.
#   3. Overlap is no longer expected even at the connector. Any overlap after translation is an
#      error. The generator guarantees this by giving the base a right-edge connector span and
#      the plug a left-edge connector span in their cropped coordinates.
#   4. Base must still remain the output top-left object. The generator guarantees this by
#      choosing the plug connector row no higher than the base connector row.
# - Placement legality:
#   Generator is still output-first. It first builds a docked left-right assembly that is known
#   to be valid, then separates the plug and places the two cropped sprites apart on a canvas.
#   Placement uses explicit horizontal/vertical layouts with at least one-cell gap, so there
#   are always exactly two components and no accidental touching.
# - Visual observability / tidiness:
#   The input still has exactly two yellow sprites and two orange connector bars of equal
#   length. What changes is easy to see: both bars lie inside their sprites with yellow above
#   and below, and the output joins the shapes laterally into one wider figure.


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn_pts_from_set(pts: set[tuple[int, int]], grid: np.ndarray, ctrl: int) -> set[tuple[int, int]]:
    return {(r, c) for (r, c) in pts if int(grid[int(r), int(c)]) == int(ctrl)}


def _conn_span(conn_pts: set[tuple[int, int]]) -> tuple[int, int, int]:
    assert conn_pts
    rows = {int(r) for r, _c in conn_pts}
    assert len(rows) == 1
    row = int(next(iter(rows)))
    cols = sorted(int(c) for _r, c in conn_pts)
    assert cols == list(range(cols[0], cols[-1] + 1))
    return row, int(cols[0]), int(cols[-1])


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in conn_pts]
    cs = [int(c) for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (r0, c0)


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _dock_grids(base_grid: np.ndarray, plug_grid: np.ndarray, bg: int, obj: int, ctrl: int) -> np.ndarray:
    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_grid != int(bg))}
    plug_pts = {(int(r), int(c)) for r, c in np.argwhere(plug_grid != int(bg))}
    conn_d = _conn_pts_from_set(base_pts, base_grid, ctrl)
    conn_p = _conn_pts_from_set(plug_pts, plug_grid, ctrl)
    assert conn_d and conn_p
    shp_d, _ = _norm_conn(conn_d)
    shp_p, _ = _norm_conn(conn_p)
    assert shp_d == shp_p

    dr, _dc0, dc1 = _conn_span(conn_d)
    pr, pc0, _pc1 = _conn_span(conn_p)
    base_body = base_pts - conn_d
    plug_body = plug_pts - conn_p
    for r, c in conn_d:
        assert (int(r - 1), int(c)) in base_body
        assert (int(r + 1), int(c)) in base_body
    for r, c in conn_p:
        assert (int(r - 1), int(c)) in plug_body
        assert (int(r + 1), int(c)) in plug_body

    tr = int(dr - pr)
    tc = int((dc1 + 1) - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug_pts}

    pts_all = base_pts | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    base_r0, base_c0, _base_r1, _base_c1 = _bbox_of_pts(base_pts)
    assert rr0 == base_r0 and cc0 == base_c0
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(base_grid[int(r), int(c)])

    for r, c in plug_shift:
        val = int(plug_grid[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        assert cur == int(bg)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    return _tight_crop(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    a, b = comps

    conn_a = _conn_pts_from_set(a, g, ctrl)
    conn_b = _conn_pts_from_set(b, g, ctrl)
    assert conn_a and conn_b
    shp_a, _ = _norm_conn(conn_a)
    shp_b, _ = _norm_conn(conn_b)
    assert shp_a == shp_b

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(a)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(b)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, conn_d = a, conn_a
        plug, conn_p = b, conn_b
    else:
        base, conn_d = b, conn_b
        plug, conn_p = a, conn_a

    base_body = base - conn_d
    plug_body = plug - conn_p
    for r, c in conn_d:
        assert (int(r - 1), int(c)) in base_body
        assert (int(r + 1), int(c)) in base_body
    for r, c in conn_p:
        assert (int(r - 1), int(c)) in plug_body
        assert (int(r + 1), int(c)) in plug_body

    dr, _dc0, dc1 = _conn_span(conn_d)
    pr, pc0, _pc1 = _conn_span(conn_p)
    tr = int(dr - pr)
    tc = int((dc1 + 1) - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    base_r0, base_c0, _base_r1, _base_c1 = _bbox_of_pts(base)
    assert rr0 == base_r0 and cc0 == base_c0
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        assert cur == int(bg)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    return _tight_crop(out, bg)


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[r + dr, c + dc]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _make_sprite(
    h: int,
    w: int,
    conn_row: int,
    conn_cols: list[int],
    *,
    require_anchor: bool,
    require_left_of_conn: bool,
    require_right_of_conn: bool,
) -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7
    mask = np.zeros((h, w), dtype=bool)
    keep = set()

    if require_anchor:
        mask[0, 0] = True
        keep.add((0, 0))

    for c in conn_cols:
        for rr in (conn_row - 1, conn_row, conn_row + 1):
            mask[rr, c] = True
            keep.add((rr, c))

    frontier = list(keep)
    target_min = max(len(keep) + 2, 5)
    target_max = min(18, h * w)
    if target_min > target_max:
        return None
    target = _rng.randint(target_min, target_max)
    _grow_shape(mask, target, frontier)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    _thin(mask, keep=keep)

    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if int(mask.sum()) < 5:
        return None
    if require_right_of_conn and not any(mask[:, c].any() for c in range(conn_cols[-1] + 1, w)):
        return None
    if require_left_of_conn and not any(mask[:, c].any() for c in range(0, conn_cols[0])):
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    grid[mask] = int(obj)
    for c in conn_cols:
        grid[int(conn_row), int(c)] = int(ctrl)

    for c in conn_cols:
        if grid[int(conn_row - 1), int(c)] != int(obj):
            return None
        if grid[int(conn_row + 1), int(c)] != int(obj):
            return None
    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    conn_len = _rng.randint(1, 3)
    base_h = _rng.randint(4, 6)
    base_w = _rng.randint(max(3, conn_len + 1), 6)
    conn_row = _rng.randint(1, base_h - 2)
    base_conn_cols = list(range(base_w - conn_len, base_w))
    base_grid = _make_sprite(
        base_h,
        base_w,
        conn_row,
        base_conn_cols,
        require_anchor=True,
        require_left_of_conn=True,
        require_right_of_conn=False,
    )
    if base_grid is None:
        return None

    plug_h = _rng.randint(3, 6)
    plug_w = _rng.randint(max(3, conn_len + 1), 6)
    plug_conn_row_max = min(conn_row, plug_h - 2)
    if plug_conn_row_max < 1:
        return None
    plug_conn_row = _rng.randint(1, plug_conn_row_max)
    plug_conn_cols = list(range(0, conn_len))
    plug_grid = _make_sprite(
        plug_h,
        plug_w,
        plug_conn_row,
        plug_conn_cols,
        require_anchor=False,
        require_left_of_conn=False,
        require_right_of_conn=True,
    )
    if plug_grid is None:
        return None

    expected = _dock_grids(base_grid, plug_grid, bg, obj, ctrl)
    eh, ew = expected.shape
    if eh == 0 or ew == 0:
        return None
    if eh > 9 or ew > 12:
        return None

    base_sprite = _tight_crop(base_grid, bg)
    plug_sprite = _tight_crop(plug_grid, bg)
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    modes = []
    if max(bh, ph) <= 11 and bw + 1 + pw <= 13:
        modes.append("side")
    if bh + 1 + ph <= 11 and max(bw, pw) <= 13:
        modes.append("stack")
    if not modes:
        return None

    mode = _rng.choice(modes)
    if mode == "side":
        min_w = bw + 1 + pw
        canvas_w = _rng.randint(max(7, min_w), 13)
        canvas_h = _rng.randint(max(7, max(bh, ph)), 11)
        gap = _rng.randint(1, canvas_w - (bw + pw))
        pr = _rng.randint(0, canvas_h - ph)
        pc = int(bw + gap)
    else:
        min_h = bh + 1 + ph
        canvas_h = _rng.randint(max(7, min_h), 11)
        canvas_w = _rng.randint(max(3, max(bw, pw)), 13)
        gap = _rng.randint(1, canvas_h - (bh + ph))
        pr = int(bh + gap)
        pc = _rng.randint(0, canvas_w - pw)

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=bg)

    lab, n = label_mask(canvas != int(bg), connectivity=8)
    if int(n) != 2:
        return None
    if np.sum(canvas == int(bg)) <= canvas.size // 2:
        return None

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
