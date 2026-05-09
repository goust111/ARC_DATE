# Combo variant of 20270e3b (index 10)
# combo_spec: D6/2+D15/2
# Name: 标记判定内部连接器

from __future__ import annotations

import random
import sys

import numpy as np

sys.path.insert(0, "/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, is_contiguous, label_mask, paste


# Design analysis for the fused variant
# - Concrete appearance:
#   The input contains exactly two yellow/orange sprites (obj=4, ctrl=7) plus one
#   red singleton marker (marker=2). The orange connector bar inside each sprite is
#   horizontal and internal in the vertical sense: each ctrl cell has yellow body
#   both above and below it. The marker is not part of either sprite; it sits in
#   the 8-neighborhood halo of the base sprite and at least 3 Chebyshev steps from
#   the plug, so it is a visible symbolic cue.
# - Behavior trace on a small example:
#   Old task: choose the sprite whose bbox touches the origin, then move the other
#   sprite upward so the two connector bars share columns with a 1-row vertical
#   offset. The bars disappear into yellow.
#   New task: first find the red marker and choose the adjacent sprite as base.
#   Then detect the internal orange bar in each sprite. Because both bars have body
#   above and below, there is no exposed up/down docking side anymore; instead the
#   plug is translated to the right so its connector row equals the base connector
#   row and its leftmost connector cell sits immediately after the base connector's
#   rightmost cell. The bars are recolored to yellow and the result is tightly
#   cropped. Compared with the original, both the role-selection rule and the
#   geometry of docking visibly change.
# - Complexity calibration:
#   Only one extra symbolic cue is added (a single red marker), and the connector
#   length still has only 1-3 cells and must match across both sprites. This keeps
#   correspondence easy to infer while making the visual change obvious: the output
#   widens sideways instead of docking vertically.
# - Implicit assumptions from the original solver that break:
#   `label_mask(g != bg)` is no longer valid because the red marker would become a
#   third component. We must segment only obj/ctrl colors.
#   `r0+c0==0` no longer identifies the base and is deliberately falsified in many
#   generated samples.
#   `base below empty / plug above empty` is false because both connectors are now
#   internal, with body on both vertical sides.
#   The old translation `(dr0-1)-pr0, dc0-pc0` is invalid; the new translation uses
#   row equality and right-adjacency of the two connector spans.
#   Overlap at the docking seam is no longer allowed, so stamping asserts empty
#   destination cells for every shifted plug pixel.
# - Placement legality:
#   The generator is output-first. It builds a valid docked assembly first, crops
#   the two sprites apart, then places them on a larger canvas with at least one
#   background cell separating them. The marker is placed after sprite placement so
#   it never overlaps a sprite and never touches the plug ambiguously.
# - Visual observability / cleanliness:
#   Human solvers can see two repeated cues across examples: the red singleton marks
#   the base, and both orange bars are buried inside the yellow bodies yet outputs
#   always join the shapes laterally. There is no hidden dependence on color order,
#   component index, or non-visual tie-breaking.
# - New solver assumptions guaranteed by the generator:
#   exactly one red marker exists; exactly two obj/ctrl components exist; the marker
#   is Chebyshev distance 1 from the base and at least 3 from the plug; both
#   connectors are single-row contiguous bars with identical length; every ctrl cell
#   has obj above and below; the base connector sits on the right boundary of the
#   cropped base sprite and the plug connector sits on the left boundary of the
#   cropped plug sprite; the plug connector row is not lower than the base's, so
#   the shifted plug never extends above the base after docking.


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn_pts(sprite: set[tuple[int, int]], grid: np.ndarray, ctrl: int) -> set[tuple[int, int]]:
    return {(r, c) for r, c in sprite if int(grid[int(r), int(c)]) == int(ctrl)}


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in pts]
    cs = [int(c) for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in pts}
    return shape, (r0, c0)


def _conn_span(conn_pts: set[tuple[int, int]]) -> tuple[int, int, int]:
    assert conn_pts
    rows = {int(r) for r, _c in conn_pts}
    assert len(rows) == 1
    row = int(next(iter(rows)))
    cols = sorted(int(c) for _r, c in conn_pts)
    assert cols == list(range(cols[0], cols[-1] + 1))
    return row, int(cols[0]), int(cols[-1])


def _point_to_sprite_cheb(pt: tuple[int, int], sprite: set[tuple[int, int]]) -> int:
    r0, c0 = pt
    return min(max(abs(int(r - r0)), abs(int(c - c0))) for r, c in sprite)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, int(br0 - ar1 - 1), int(ar0 - br1 - 1))
    dc = max(0, int(bc0 - ac1 - 1), int(ac0 - bc1 - 1))
    return int(dr + dc)


def _blocked8(pts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((int(r + dr), int(c + dc)))
    return blocked


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _marker_candidates(
    base_pts: set[tuple[int, int]],
    base_box: tuple[int, int, int, int],
    plug_pts: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = base_box
    occupied = base_pts | plug_pts
    out: set[tuple[int, int]] = set()
    for r, c in base_pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in occupied:
                    continue
                if int(r0) <= nr <= int(r1) and int(c0) <= nc <= int(c1):
                    continue
                if _point_to_sprite_cheb((nr, nc), plug_pts) < 3:
                    continue
                out.add((nr, nc))
    return list(out)


def _assert_internal_connector(sprite: set[tuple[int, int]], conn: set[tuple[int, int]]) -> None:
    body = sprite - conn
    for r, c in conn:
        assert (int(r - 1), int(c)) in body
        assert (int(r + 1), int(c)) in body


def _dock_grids(base_grid: np.ndarray, plug_grid: np.ndarray, bg: int, obj: int, ctrl: int) -> np.ndarray:
    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_grid != int(bg))}
    plug_pts = {(int(r), int(c)) for r, c in np.argwhere(plug_grid != int(bg))}
    conn_d = _conn_pts(base_pts, base_grid, ctrl)
    conn_p = _conn_pts(plug_pts, plug_grid, ctrl)
    assert conn_d and conn_p

    shp_d, _ = _norm_pts(conn_d)
    shp_p, _ = _norm_pts(conn_p)
    assert shp_d == shp_p
    _assert_internal_connector(base_pts, conn_d)
    _assert_internal_connector(plug_pts, conn_p)

    dr, _dc0, dc1 = _conn_span(conn_d)
    pr, pc0, _pc1 = _conn_span(conn_p)
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
    bg, obj, ctrl, marker = 1, 4, 7, 2

    marker_pts = np.argwhere(g == int(marker))
    assert marker_pts.shape[0] == 1
    mr, mc = map(int, marker_pts[0])

    sprite_mask = (g == int(obj)) | (g == int(ctrl))
    lab, n = label_mask(sprite_mask, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    a, b = comps

    conn_a = _conn_pts(a, g, ctrl)
    conn_b = _conn_pts(b, g, ctrl)
    assert conn_a and conn_b
    shp_a, _ = _norm_pts(conn_a)
    shp_b, _ = _norm_pts(conn_b)
    assert shp_a == shp_b

    dist_a = _point_to_sprite_cheb((mr, mc), a)
    dist_b = _point_to_sprite_cheb((mr, mc), b)
    assert dist_a != dist_b
    if int(dist_a) < int(dist_b):
        base, conn_d = a, conn_a
        plug, conn_p = b, conn_b
    else:
        base, conn_d = b, conn_b
        plug, conn_p = a, conn_a

    _assert_internal_connector(base, conn_d)
    _assert_internal_connector(plug, conn_p)

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
        idx = random.randint(0, len(frontier) - 1)
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
        nr, nc = random.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    random.shuffle(pixels)
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
    keep: set[tuple[int, int]] = set()

    if require_anchor:
        mask[0, 0] = True
        keep.add((0, 0))

    for c in conn_cols:
        for rr in (conn_row - 1, conn_row, conn_row + 1):
            mask[int(rr), int(c)] = True
            keep.add((int(rr), int(c)))

    frontier = list(keep)
    target_min = max(len(keep) + 2, 5)
    target_max = min(18, h * w)
    if target_min > target_max:
        return None
    target = random.randint(target_min, target_max)
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
    bg, obj, ctrl, marker = 1, 4, 7, 2

    conn_len = random.randint(1, 3)

    base_h = random.randint(4, 6)
    base_w = random.randint(max(3, conn_len + 1), 6)
    base_conn_row = random.randint(1, base_h - 2)
    base_conn_cols = list(range(base_w - conn_len, base_w))
    base_grid = _make_sprite(
        base_h,
        base_w,
        base_conn_row,
        base_conn_cols,
        require_anchor=True,
        require_left_of_conn=True,
        require_right_of_conn=False,
    )
    if base_grid is None:
        return None

    plug_h = random.randint(3, 6)
    plug_w = random.randint(max(3, conn_len + 1), 6)
    plug_conn_row_max = min(base_conn_row, plug_h - 2)
    if plug_conn_row_max < 1:
        return None
    plug_conn_row = random.randint(1, plug_conn_row_max)
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
    eh, ew = map(int, expected.shape)
    if eh == 0 or ew == 0:
        return None
    if eh > 9 or ew > 12:
        return None

    base_sprite = _tight_crop(base_grid, bg)
    plug_sprite = _tight_crop(plug_grid, bg)
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)

    min_h = max(7, max(bh, ph) + 3)
    min_w = max(7, max(bw, pw) + 3)
    if min_h > 11 or min_w > 13:
        return None
    canvas_h = random.randint(min_h, 11)
    canvas_w = random.randint(min_w, 13)

    base_local = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    plug_local = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}

    base_positions = [
        (int(br), int(bc))
        for br in range(1, canvas_h - bh)
        for bc in range(1, canvas_w - bw)
    ]
    if not base_positions:
        return None
    random.shuffle(base_positions)

    prefer_plug_closer = bool(random.randint(0, 1))
    for br, bc in base_positions:
        base_pts = {(int(br + r), int(bc + c)) for r, c in base_local}
        base_box = (int(br), int(bc), int(br + bh - 1), int(bc + bw - 1))
        blocked = _blocked8(base_pts)
        plug_options: list[tuple[int, int, list[tuple[int, int]]]] = []

        for pr in range(canvas_h - ph + 1):
            for pc in range(canvas_w - pw + 1):
                if prefer_plug_closer and int(pr + pc) >= int(br + bc):
                    continue
                if (not prefer_plug_closer) and int(pr + pc) <= int(br + bc):
                    continue
                plug_pts = {(int(pr + r), int(pc + c)) for r, c in plug_local}
                if any((r, c) in blocked for r, c in plug_pts):
                    continue
                plug_box = (int(pr), int(pc), int(pr + ph - 1), int(pc + pw - 1))
                if not (1 <= _bbox_gap(base_box, plug_box) <= 5):
                    continue
                marker_options = _marker_candidates(base_pts, base_box, plug_pts, canvas_h, canvas_w)
                if not marker_options:
                    continue
                plug_options.append((int(pr), int(pc), marker_options))

        if not plug_options:
            continue

        pr, pc, marker_options = random.choice(plug_options)
        mr, mc = random.choice(marker_options)

        canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
        paste(canvas, base_sprite, br, bc, transparent=bg)
        paste(canvas, plug_sprite, pr, pc, transparent=bg)
        canvas[int(mr), int(mc)] = int(marker)

        sprite_mask = (canvas == int(obj)) | (canvas == int(ctrl))
        _lab, n = label_mask(sprite_mask, connectivity=8)
        if int(n) != 2:
            continue
        if np.sum(canvas == int(bg)) <= canvas.size // 2:
            continue

        try:
            result = main(canvas)
        except Exception:
            continue
        if not np.array_equal(result, expected):
            continue
        return canvas

    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
