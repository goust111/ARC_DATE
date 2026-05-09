# Combo variant of 20270e3b (index 24)
# combo_spec: D5/2+D10/0+D11/1
# Name: 双色翻转颜色交换

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


# Design analysis for combo D5/2 + D10/0 + D11/1
# ------------------------------------------------
# a) Concrete trace
#    - Input still has exactly 2 sprites: base at the grid origin and one plug elsewhere.
#      Both sprites have a horizontal orange connector (ctrl=7), but base/plug body colors
#      are different and stable within each sprite.
#    - Original 20270e3b: translate plug as-is so its connector sits one row above the base
#      connector, then recolor every surviving connector to a single global obj color.
#    - This combo: first horizontally flip the entire plug patch, then dock it in the same
#      vertical relation, then swap body ownership colors:
#        * every base pixel (body + connector) becomes plug's original body color
#        * every plug pixel (body + connector) becomes base's original body color
#      Example: base body = 2, plug body = 3, both connectors = 7. If the docked plug shape
#      is right-anchored, the separated input plug is its horizontal mirror with a left-side
#      connector. During output, base recolors to 3, mirrored plug recolors to 2.
#      So the solver must track mirrored geometry and object identity simultaneously.
# b) Complexity calibration
#    - Keep only 4 visible colors total: bg=1, ctrl=7, base_col, plug_col.
#    - Force base_col != plug_col and avoid yellow(4) entirely so the task cannot visually
#      collapse back to the original single-obj-color behavior.
#    - Force the connector to lie on a bbox side (left or right) for both base and docked
#      plug, and reject horizontally symmetric plugs, so the required flip is visible.
# c) Implicit assumption audit
#    - Original `obj=4` assumption breaks: the solver must infer one body color per sprite.
#    - Original direct translation anchor breaks: the solver must recompute the plug
#      connector after `np.fliplr`.
#    - Original `out[out == ctrl] = obj` breaks: connector recolor now depends on sprite
#      identity after color swap, so recoloring is performed during stamping.
#    - Original overlap check using current colors is too weak after recolor; legality must
#      be checked from connector membership in the pre-recolor base/plug patches.
# d) Placement legality
#    - Generator builds the base patch, then the plug in its docked post-flip orientation,
#      mirrors it to obtain the separated input plug, and places the two cropped sprites on
#      a canvas with bbox gap 1..5 and no 8-neighborhood contact.
#    - In the docked pose, overlap is allowed only where connector rows interact with the
#      matching body/connector seam; all other body-body overlaps are rejected.
# e) Visual observability
#    - Across a few samples, the plug connector appears on the opposite horizontal side from
#      the side where it must dock, so a horizontal flip is directly observable.
#    - Output colors visibly exchange ownership between the two shapes, so the color swap is
#      not an arbitrary remapping but tied to object identity after docking.
# f) Visual cleanliness
#    - No extra markers are introduced; only two differently colored sprites and one shared
#      connector color appear in input. Output contains exactly the two swapped body colors.
# g) Color-change inferability
#    - Because each input sprite is monochrome apart from ctrl and the output preserves the
#      same two color values but swaps which silhouette owns them, humans can infer the rule
#      from object correspondence rather than from hidden numeric color identities.
# h) New solver assumptions guaranteed by the generator
#    - Each sprite has exactly one non-bg, non-ctrl body color.
#    - Both connectors are same-length horizontal bars.
#    - Base bbox touches the origin.
#    - Docked plug does not extend above or left of base after flipping.
#    - Docking with and without the flip does not yield the same merged output.


BG = 1
CTRL = 7
BODY_PALETTE = (2, 3, 5, 6, 8, 9)
_rng = random


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _extract_patch(grid: np.ndarray, pts: set[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_pts(pts)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        patch[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return patch


def _conn_points(patch: np.ndarray, ctrl: int) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(patch == int(ctrl))}


def _body_color(patch: np.ndarray, bg: int, ctrl: int) -> int:
    cols = {
        int(patch[int(r), int(c)])
        for r, c in np.argwhere((patch != int(bg)) & (patch != int(ctrl)))
    }
    assert len(cols) == 1
    return int(next(iter(cols)))


def _norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _connector_side(patch: np.ndarray, ctrl: int) -> str:
    conn = _conn_points(patch, ctrl)
    assert conn
    cols = [c for _r, c in conn]
    if int(min(cols)) == 0 and int(max(cols)) < int(patch.shape[1] - 1):
        return "left"
    if int(max(cols)) == int(patch.shape[1] - 1) and int(min(cols)) > 0:
        return "right"
    return "interior"


def _dock_from_patches(
    base_patch: np.ndarray,
    plug_patch: np.ndarray,
    *,
    bg: int,
    ctrl: int,
    flip_plug: bool,
) -> np.ndarray:
    base = np.array(base_patch, dtype=int)
    plug = np.array(plug_patch, dtype=int)
    assert base.size > 0 and plug.size > 0

    base_color = _body_color(base, bg, ctrl)
    plug_color = _body_color(plug, bg, ctrl)
    assert base_color != plug_color

    conn_base = _conn_points(base, ctrl)
    conn_plug_in = _conn_points(plug, ctrl)
    assert conn_base and conn_plug_in

    shp_base, (dr0, dc0) = _norm(conn_base)
    shp_plug_in, _ = _norm(conn_plug_in)
    assert shp_base == shp_plug_in

    base_body = {(int(r), int(c)) for r, c in np.argwhere((base != int(bg)) & (base != int(ctrl)))}
    for r, c in conn_base:
        assert (int(r + 1), int(c)) not in base_body

    work_plug = np.fliplr(plug) if flip_plug else plug.copy()
    conn_work = _conn_points(work_plug, ctrl)
    shp_plug, (pr0, pc0) = _norm(conn_work)
    assert shp_plug == shp_base

    plug_body = {
        (int(r), int(c))
        for r, c in np.argwhere((work_plug != int(bg)) & (work_plug != int(ctrl)))
    }
    for r, c in conn_work:
        assert (int(r - 1), int(c)) not in plug_body

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)

    plug_pixels: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    for r, c in np.argwhere(work_plug != int(bg)):
        rr, cc = int(r), int(c)
        plug_pixels[(int(rr + tr), int(cc + tc))] = ((rr, cc), int(work_plug[rr, cc]))

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base != int(bg))}
    pts_all = base_pts | set(plug_pixels)
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    assert rr0 == 0 and cc0 == 0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)
    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(plug_color)

    for (r, c), (src, val) in plug_pixels.items():
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert (int(r), int(c)) in conn_base or src in conn_work
        out[int(r - rr0), int(c - cc0)] = int(base_color)

    return _crop_non_bg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, ctrl = BG, CTRL

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    connA = {(r, c) for (r, c) in A if int(g[int(r), int(c)]) == int(ctrl)}
    connB = {(r, c) for (r, c) in B if int(g[int(r), int(c)]) == int(ctrl)}
    assert connA and connB
    shpA, _ = _norm(connA)
    shpB, _ = _norm(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base_pts, plug_pts = A, B
    else:
        base_pts, plug_pts = B, A

    base_patch = _extract_patch(g, base_pts, bg)
    plug_patch = _extract_patch(g, plug_pts, bg)
    return _dock_from_patches(
        base_patch,
        plug_patch,
        bg=bg,
        ctrl=ctrl,
        flip_plug=True,
    )


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= int(r + dr) < h and 0 <= int(c + dc) < w and not mask[int(r + dr), int(c + dc)]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r, c = int(r), int(c)
        if (r, c) in keep:
            continue
        adj_bg = any(
            not (0 <= int(r + dr) < h and 0 <= int(c + dc) < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _build_base(base_color: int) -> tuple[np.ndarray, str] | None:
    bg, ctrl = BG, CTRL
    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, int(base_h * base_w)))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    max_conn = min(3, int(base_w - 1))
    if max_conn < 1:
        return None
    conn_len = _rng.randint(1, max_conn)
    side = _rng.choice(["left", "right"])
    candidates: list[tuple[int, list[int]]] = []
    for r in range(1, base_h):
        c0 = 0 if side == "left" else int(base_w - conn_len)
        cols = list(range(int(c0), int(c0 + conn_len)))
        if not all(base[int(r), int(c)] for c in cols):
            continue
        if not all(base[int(r - 1), int(c)] for c in cols):
            continue
        if int(r + 1) < base_h and not all(not base[int(r + 1), int(c)] for c in cols):
            continue
        candidates.append((int(r), cols))
    if not candidates:
        return None

    conn_row, conn_cols = _rng.choice(candidates)
    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base] = int(base_color)
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = int(ctrl)

    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None
    base_sprite = _crop_non_bg(base_grid, bg)
    if base_sprite.size == 0:
        return None
    if _connector_side(base_sprite, ctrl) != side:
        return None
    if _rng.random() < 0.5:
        base_sprite = np.fliplr(base_sprite)
        side = "right" if side == "left" else "left"
    return base_sprite, side


def _build_docked_plug(base_sprite: np.ndarray, side: str, plug_color: int) -> np.ndarray | None:
    bg, ctrl = BG, CTRL
    base_h, base_w = map(int, base_sprite.shape)
    conn_base = _conn_points(base_sprite, ctrl)
    _shape, (base_conn_r0, base_conn_c0) = _norm(conn_base)
    conn_len = int(len(conn_base))

    min_plug_w = max(2, int(conn_len + 1))
    if side == "right":
        max_plug_w = base_w
        if max_plug_w < min_plug_w:
            return None
        plug_w = _rng.randint(min_plug_w, max_plug_w)
        dock_conn_c0 = int(plug_w - conn_len)
    else:
        max_plug_w = min(10, base_w + 4)
        if max_plug_w < min_plug_w:
            return None
        plug_w = _rng.randint(min_plug_w, max_plug_w)
        dock_conn_c0 = 0
    plug_h = _rng.randint(2, 7)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(dock_conn_c0, dock_conn_c0 + conn_len):
        plug[0, int(c)] = True
        if plug_h > 1:
            plug[1, int(c)] = True

    plug_max = min(29, int(plug_h * plug_w))
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, [(int(r), int(c)) for r, c in zip(*np.where(plug))])
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    keep_plug = {(0, int(c)) for c in range(dock_conn_c0, dock_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    if np.array_equal(plug, np.fliplr(plug)):
        return None

    plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    plug_grid[plug] = int(plug_color)
    for c in range(dock_conn_c0, dock_conn_c0 + conn_len):
        plug_grid[0, int(c)] = int(ctrl)
        if plug_h < 2 or plug_grid[1, int(c)] != int(plug_color):
            return None

    if _connector_side(plug_grid, ctrl) != side:
        return None

    tc = int(base_conn_c0 - dock_conn_c0)
    tr = int(base_conn_r0 - 1)
    if tc < 0 or tr < 0:
        return None

    for r, c in np.argwhere(plug_grid != int(bg)):
        rr, cc = int(r), int(c)
        dr = int(rr + tr)
        dc = int(cc + tc)
        if 0 <= dr < base_h and 0 <= dc < base_w:
            cur = int(base_sprite[dr, dc])
            val = int(plug_grid[rr, cc])
            if cur != int(bg) and val != int(bg) and cur != int(ctrl) and val != int(ctrl):
                return None

    return plug_grid


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return int(_gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1))


def _placement_candidates(
    canvas_h: int,
    canvas_w: int,
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    bg: int,
) -> list[tuple[int, int]]:
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return []

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    base_dilated = set(base_pts)
    for r, c in list(base_pts):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                base_dilated.add((int(r + dr), int(c + dc)))
    plug_local = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}

    out: list[tuple[int, int]] = []
    base_bbox = (0, 0, int(bh - 1), int(bw - 1))
    for pr in range(0, int(canvas_h - ph + 1)):
        for pc in range(0, int(canvas_w - pw + 1)):
            if pr == 0 and pc == 0:
                continue
            shifted = {(int(r + pr), int(c + pc)) for r, c in plug_local}
            if any(pt in base_dilated for pt in shifted):
                continue
            gap = _bbox_gap(base_bbox, (int(pr), int(pc), int(pr + ph - 1), int(pc + pw - 1)))
            if 1 <= int(gap) <= 5:
                out.append((int(pr), int(pc)))
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, ctrl = BG, CTRL
    base_color, plug_color = _rng.sample(BODY_PALETTE, 2)

    built = _build_base(int(base_color))
    if built is None:
        return None
    base_sprite, side = built

    docked_plug = _build_docked_plug(base_sprite, side, int(plug_color))
    if docked_plug is None:
        return None

    plug_input = _crop_non_bg(np.fliplr(docked_plug), bg)
    input_side = _connector_side(plug_input, ctrl)
    expected_input_side = "right" if side == "left" else "left"
    if input_side != expected_input_side:
        return None

    try:
        expected = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            ctrl=ctrl,
            flip_plug=True,
        )
    except AssertionError:
        return None
    try:
        direct = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            ctrl=ctrl,
            flip_plug=False,
        )
    except AssertionError:
        direct = None

    if direct is not None and np.array_equal(expected, direct):
        return None

    out_colors = {int(x) for x in np.unique(expected) if int(x) != int(bg)}
    if out_colors != {int(base_color), int(plug_color)}:
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_input.shape)
    min_h = max(7, bh, ph)
    min_w = max(3, bw, pw)
    if min_h > 11 or min_w > 13:
        return None

    for _ in range(20):
        canvas_h = _rng.randint(min_h, 11)
        canvas_w = _rng.randint(min_w, 13)
        placements = _placement_candidates(canvas_h, canvas_w, base_sprite, plug_input, bg)
        if not placements:
            continue
        pr, pc = _rng.choice(placements)
        canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
        paste(canvas, base_sprite, 0, 0, transparent=bg)
        paste(canvas, plug_input, pr, pc, transparent=bg)
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
