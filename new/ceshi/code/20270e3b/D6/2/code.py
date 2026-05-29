# Variant of 20270e3b
# Dimension: D6 — base角色判定规则 (meta)
# Change: 外接矩形左上角满足 r+c==0（即紧贴网格原点）的精灵为base... → 特殊标记像素指示base

from __future__ import annotations

import random
import sys

import numpy as np

sys.path.insert(0, "/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, is_contiguous, label_mask, paste


# Design analysis for this D6 variant
# - Concrete appearance:
#   input keeps the original two yellow/orange sprites (obj=4, ctrl=7), and adds
#   one red marker pixel (marker=2) outside but immediately next to the base sprite.
#   The marker is ignored in the output. The docking motion and connector recolor
#   are unchanged.
# - Behavior trace:
#   original logic picked base by `bbox top-left has r+c==0`; here that rule is
#   intentionally invalidated because base is placed away from the origin, and the
#   plug is often placed even closer to the origin. The solver must first find the
#   red marker, then choose the sprite at Chebyshev distance 1 from that marker.
# - Complexity calibration:
#   only one new cue is introduced: a single fixed-color marker pixel. Its role is
#   repeated in every example, and the plug/base connector mechanics remain exactly
#   the same, so the task changes in one visible dimension without adding extra
#   matching freedoms.
# - Source-task note:
#   the prompt names 9aaea919 as inspiration, but within the allowed local files
#   only the ARC JSON is available and no solver source is present. The marker
#   mechanism below is therefore designed directly from the stated `marker_pixels`
#   idea, not copied from unavailable code.
# - Old implicit assumptions that break:
#   `label_mask(g != bg)` would count the marker as a third component, so sprite
#   extraction must now use only obj/ctrl colors.
#   `assert r0+c0==0 for one sprite` is no longer valid and is removed.
#   choosing the smaller `r0+c0` sprite would now be wrong on many generated cases.
# - Placement legality:
#   generator keeps the two sprites 8-disconnected in the input, keeps the docking
#   overlap restricted to the connector seam in the assembled state, and places the
#   marker outside the base bbox so it reads as a cue rather than body detail.
# - Visual observability / cleanliness:
#   the red singleton is the only new element and sits on the exterior halo of the
#   base, away from the plug. This makes the meta-rule visible in 2-3 examples.
# - New solver assumptions guaranteed by generator:
#   exactly one red marker exists; exactly two obj/ctrl sprite components exist;
#   marker Chebyshev distance to base is 1 and to plug is at least 3; plug still
#   does not extend above or left of base after docking.


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in pts}
    return shape, (int(r0), int(c0))


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


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)

    bg, obj, ctrl, marker = 1, 4, 7, 2

    marker_pts = np.argwhere(g == int(marker))
    assert marker_pts.shape[0] == 1
    mr, mc = map(int, marker_pts[0])

    # Only obj/ctrl belong to the two sprites; marker is a separate symbolic cue.
    sprite_mask = (g == int(obj)) | (g == int(ctrl))
    lab, n = label_mask(sprite_mask, connectivity=8)
    best_comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        best_comps.append(pts)
    best_comps.sort(key=len, reverse=True)
    assert len(best_comps) == 2
    A = set(best_comps[0])
    B = set(best_comps[1])

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    connA = conn(A)
    connB = conn(B)
    assert connA and connB

    shpA, _ = _norm_pts(connA)
    shpB, _ = _norm_pts(connB)
    assert shpA == shpB

    distA = _point_to_sprite_cheb((mr, mc), A)
    distB = _point_to_sprite_cheb((mr, mc), B)
    assert distA != distB
    if int(distA) < int(distB):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    base_body = base - connD
    plug_body = plug - connP
    for r, c in connD:
        assert (int(r + 1), int(c)) not in base_body
    for r, c in connP:
        assert (int(r - 1), int(c)) not in plug_body

    _sp, (pr0, pc0) = _norm_pts(connP)
    _sd, (dr0, dc0) = _norm_pts(connD)

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))

    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)

    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


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


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl, marker = 1, 4, 7, 2

    base_h = random.randint(2, 7)
    base_w = random.randint(3, 10)
    base_target = random.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = random.randint(1, min(3, base_w))
    candidates: list[tuple[int, list[int]]] = []
    for r in range(base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if r == 0:
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            candidates.append((int(r), cols))
    if not candidates:
        return None
    conn_row, conn_cols = random.choice(candidates)

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_h = random.randint(2, 7)
    plug_w = random.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = random.randint(0, plug_w - conn_len)
    dc_offset = int(conn_cols[0] - plug_conn_c0)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        if plug_h > 1:
            plug[1, c] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = random.randint(5, plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = int(conn_row + r)
            dc = int(dc_offset + c)
            if 0 <= dr < base_h and 0 <= dc < base_w:
                if base_grid[dr, dc] == obj:
                    if r == 0 and c in range(plug_conn_c0, plug_conn_c0 + conn_len):
                        continue
                    return None

    if int(dc_offset) < 0:
        return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if plug_h < 2 or plug_grid[1, c] != obj:
            return None
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprite = _crop_non_bg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)

    min_h = max(7, max(bh, ph) + 3)
    min_w = max(5, max(bw, pw) + 3)
    if min_h > 11 or min_w > 13:
        return None
    canvas_h = random.randint(min_h, 11)
    canvas_w = random.randint(min_w, 13)

    base_local = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != bg)}
    plug_local = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != bg)}

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

        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        paste(canvas, base_sprite, br, bc, transparent=bg)
        paste(canvas, plug_sprite, pr, pc, transparent=bg)
        canvas[int(mr), int(mc)] = int(marker)

        if np.sum(canvas == bg) <= canvas.size // 2:
            continue

        try:
            result = main(canvas)
        except Exception:
            continue
        if result.size == 0 or not np.any(result != bg):
            continue
        return canvas

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
