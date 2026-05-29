# Variant of 20270e3b
# Dimension: D4 — 目标对接方式 (dst)
# Change: plug精灵平移到base精灵上方，连接器行相邻（plug连接器在base连接器正上方1行），直接拼接... → 重力下落对接
#
# Design analysis
# ---------------
# spec:
#   preserved:
#     - fixed colors: bg=1, obj=4, ctrl=7
#     - exactly two 8-connected foreground sprites
#     - one horizontal ctrl connector per sprite; the two connector bars match in shape
#     - base is still the sprite whose bbox top-left is anchored at the input origin
#     - output still recolors ctrl -> obj and tight-crops the merged result
#   changed:
#     - old docking placed plug so connector rows were vertically adjacent
#     - new docking first aligns connector columns, then drops the whole plug straight down
#       from above until the next downward step would overlap the base
#     - the final relation is adjacency without overlap, like a Tetris landing
#
# behavior trace:
#   - old rule: if base connector starts at (row=4,col=2..3) and plug connector at local row 0,
#     then plug is shifted so its connector lands at row 3.
#   - new rule: keep the same horizontal connector alignment, but compute the largest vertical
#     shift with no overlap. If the base already has body pixels higher than row 4 in those
#     columns, the plug lands earlier; its connector can stay multiple rows above the base
#     connector while the plug body rests on the base roof.
#   - visible difference: output connectivity comes from surface contact, not from ctrl rows
#     becoming an immediate seam.
#
# complexity calibration:
#   - only one discrete behavioral change is introduced: the vertical target is now determined
#     by gravity rather than by a fixed connector-row formula.
#   - connectors still have only lengths 1..3, keeping role identification stable.
#   - generator enforces positive "drop delta" versus the original direct-attach row so the
#     variant is visible and does not collapse back to the source task.
#
# implicit assumptions review:
#   - original `tr = (dr0 - 1) - pr0` assumed the connector row itself determined the final
#     vertical docking seam; this no longer holds and must be replaced.
#   - original `assert rr0 == br0 and cc0 == bc0` assumed the plug never extends above/left of
#     the base after docking. Gravity docking can place the plug above and/or left of the base,
#     so that assertion becomes invalid.
#   - original overlap logic allowed ctrl-vs-obj overlap at the seam. In this variant, the final
#     placement must be disjoint and connected by edge contact only; overlap would mean the plug
#     tunneled through the base.
#   - new solver assumption: after connector-column alignment there is at least one shared column
#     between plug and base occupancy, so a finite drop distance exists. The generator guarantees
#     this because matched connector bars lie on occupied cells in both sprites.
#
# placement legality:
#   - generator keeps base at input (0,0) and places the plug elsewhere with padding=1, so the
#     input still has exactly two 8-connected components.
#   - assembled output is built output-first: compute the gravity landing in local coordinates,
#     render the merged result, then scatter the cropped sprites onto the input canvas.
#   - connector free sides are preserved: base has no body directly below ctrl; plug has no body
#     directly above ctrl.
#
# visual observability / cleanliness:
#   - the plug always lands at least one row earlier than the old direct-attach formula, so the
#     output shows a nontrivial air gap between the remembered connector rows.
#   - only three colors are used and ctrl still disappears uniformly into obj, so the visual rule
#     remains easy to infer from a few examples.

from __future__ import annotations

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn_pts(g: np.ndarray, sprite: set[tuple[int, int]], ctrl: int) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in conn_pts]
    cs = [int(c) for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _gravity_shift(
    base: set[tuple[int, int]],
    conn_base: set[tuple[int, int]],
    plug: set[tuple[int, int]],
    conn_plug: set[tuple[int, int]],
) -> tuple[int, int, set[tuple[int, int]]]:
    """Align connector columns, then drop plug from above until the next step would overlap."""
    _sb, (_dr0, dc0) = _norm_conn(conn_base)
    _sp, (_pr0, pc0) = _norm_conn(conn_plug)
    tc = int(dc0 - pc0)

    base_rows_by_col: dict[int, list[int]] = {}
    for r, c in base:
        base_rows_by_col.setdefault(int(c), []).append(int(r))

    tr_candidates: list[int] = []
    for rp, cp in plug:
        target_col = int(cp + tc)
        for rb in base_rows_by_col.get(target_col, []):
            tr_candidates.append(int(rb - rp - 1))
    assert tr_candidates, "aligned connector columns must yield a finite gravity landing"

    tr = int(min(tr_candidates))
    shifted = {(int(r + tr), int(c + tc)) for r, c in plug}
    assert base.isdisjoint(shifted), "gravity landing must stop before overlap"
    assert any((int(r + 1), int(c)) in base for r, c in shifted), "gravity landing needs support"
    return tr, tc, shifted


def _crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _render_gravity_union(
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

    _sb, (dr0, _dc0) = _norm_conn(conn_base)
    _sp, (pr0, _pc0) = _norm_conn(conn_plug)
    tr, tc, plug_shift = _gravity_shift(base, conn_base, plug, conn_plug)
    direct_tr = int((dr0 - 1) - pr0)

    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(base_sprite[int(r), int(c)])
    for r, c in plug_shift:
        assert out[int(r - rr0), int(c - cc0)] == int(bg)
        out[int(r - rr0), int(c - cc0)] = int(plug_sprite[int(r - tr), int(c - tc)])

    out[out == int(ctrl)] = int(obj)
    cropped = _crop_non_bg(out, bg)
    assert cropped is not None
    assert is_contiguous(cropped != int(bg), background=0, connectivity=8)
    return cropped, {
        "gravity_tr": int(tr),
        "gravity_tc": int(tc),
        "direct_tr": int(direct_tr),
        "drop_delta": int(direct_tr - tr),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, but choose the final vertical position
    by gravity: align connector columns, then drop the plug onto the base and tight-crop.
    """
    g = np.array(grid, dtype=int)

    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    connA = _conn_pts(g, A, ctrl)
    connB = _conn_pts(g, B, ctrl)
    assert connA and connB

    shpA, _a0 = _norm_conn(connA)
    shpB, _b0 = _norm_conn(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, conn_base = A, connA
        plug, conn_plug = B, connB
    else:
        base, conn_base = B, connB
        plug, conn_plug = A, connA

    base_body = base - conn_base
    plug_body = plug - conn_plug
    for r, c in conn_base:
        assert (int(r + 1), int(c)) not in base_body
    for r, c in conn_plug:
        assert (int(r - 1), int(c)) not in plug_body

    tr, tc, plug_shift = _gravity_shift(base, conn_base, plug, conn_plug)

    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        assert out[int(r - rr0), int(c - cc0)] == int(bg)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    cropped = _crop_non_bg(out, bg)
    assert cropped is not None
    return cropped


import random as _rng


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    """Grow a connected shape on mask from frontier up to target pixels."""
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[int(r + dr), int(c + dc)]
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
    """Remove interior pixels while preserving 8-connectivity."""
    h, w = mask.shape
    keep = keep or set()
    pixels = [(int(r), int(c)) for r, c in zip(*np.where(mask))]
    _rng.shuffle(pixels)
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


def _make_base_sprite(bg: int, obj: int, ctrl: int) -> np.ndarray | None:
    base_h = _rng.randint(3, 6)
    base_w = _rng.randint(4, 9)
    base_target = _rng.randint(6, min(24, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates: list[tuple[int, list[int]]] = []
    for r in range(1, base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[int(r), int(c)] for c in cols):
                continue
            if not all(base[int(r - 1), int(c)] for c in cols):
                continue
            if not all((r + 1 >= base_h) or (not base[int(r + 1), int(c)]) for c in cols):
                continue
            candidates.append((int(r), cols))
    if not candidates:
        return None
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base] = int(obj)
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = int(ctrl)

    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None
    if int(np.sum(base_grid != int(bg))) < 5:
        return None
    return _crop_non_bg(base_grid, bg)


def _make_plug_sprite(conn_len: int, bg: int, obj: int, ctrl: int) -> np.ndarray | None:
    plug_h = _rng.randint(2, 5)
    plug_w = _rng.randint(max(2, conn_len), min(8, conn_len + 5))
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    if int(plug_h * plug_w) < 5:
        return None

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, int(c)] = True
        if plug_h < 2:
            return None
        plug[1, int(c)] = True

    plug_target = _rng.randint(5, min(20, plug_h * plug_w))
    _grow_shape(plug, plug_target, [(int(r), int(c)) for r, c in zip(*np.where(plug))])
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep = {(0, int(c)) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    keep |= {(1, int(c)) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep)

    plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    plug_grid[plug] = int(obj)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, int(c)] = int(ctrl)
        if plug_grid[1, int(c)] != int(obj):
            return None

    if not is_contiguous(plug_grid != int(bg), background=0, connectivity=8):
        return None
    if int(np.sum(plug_grid != int(bg))) < 5:
        return None
    cropped = _crop_non_bg(plug_grid, bg)
    if cropped is None:
        return None
    conn = {(int(r), int(c)) for r, c in np.argwhere(cropped == int(ctrl))}
    _shape, (pr0, _pc0) = _norm_conn(conn)
    if int(pr0) != 0:
        return None
    return cropped


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    base_sprite = _make_base_sprite(bg, obj, ctrl)
    if base_sprite is None:
        return None
    conn_base = {(int(r), int(c)) for r, c in np.argwhere(base_sprite == int(ctrl))}
    conn_len = len(conn_base)

    plug_sprite = _make_plug_sprite(conn_len, bg, obj, ctrl)
    if plug_sprite is None:
        return None

    expected, meta = _render_gravity_union(base_sprite, plug_sprite, bg=bg, obj=obj, ctrl=ctrl)
    if int(meta["drop_delta"]) < 1:
        return None
    eh, ew = map(int, expected.shape)
    if not (5 <= eh <= 10 and 3 <= ew <= 12):
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 12)
    canvas_w = _rng.randint(max(5, max(bw, pw) + 2), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

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
