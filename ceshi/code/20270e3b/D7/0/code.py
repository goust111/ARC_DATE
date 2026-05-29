# Variant of 20270e3b
# Dimension: D7 — 输出裁剪方式 (grid_shape)
# Change: 对接合并后紧凑裁剪到非背景像素的最小外接矩形... → 固定尺寸输出（填充背景）

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


# Design analysis for the D7 variant
# ----------------------------------
# source_task: None.
#
# a) Concrete behavior and trace
# - The input stays the same: two 8-connected sprites on a blue background (bg=1),
#   with yellow/orange body pixels (obj=4) and one horizontal orange connector bar
#   (ctrl=7) per sprite.
# - The docking logic stays the same: choose the sprite whose bbox touches the grid
#   origin as the base, translate the other sprite so its connector is one row above
#   the base connector, then recolor all connector pixels to obj.
# - The only changed stage is the final output carrier:
#   old: return the merged sprite tightly cropped.
#   new: compute that same merged tight sprite, paste it at output (0,0), and keep
#   the full input grid size with bg padding on the right and/or bottom.
# - Small trace:
#   old flow: 7x13 input -> merged sprite bbox is 7x8 -> output is 7x8.
#   new flow: 7x13 input -> merged sprite bbox is still 7x8 -> output is 7x13 with
#   the 7x8 merged sprite at top-left and the remaining 5 columns filled with bg.
#
# b) Complexity calibration
# - The connector match remains a 1-3 pixel horizontal bar shared by both sprites.
# - To make D7 visible, generation must avoid degenerate cases where the merged tight
#   bbox already equals the input size; otherwise old/new outputs would look identical.
# - No extra colors, roles, or sub-cases are introduced, so the reasoning load stays
#   close to the original task.
#
# c) Old assumptions affected by D7
# - The original solver allocates a minimal bbox and then crops again. That depends on
#   the old "tight crop is the final answer" contract. In the variant, tight assembly
#   is only an intermediate representation.
# - The original assert that the merged bbox starts at the base's top-left was harmless
#   only because the final crop discarded absolute position. The new output rule only
#   needs the merged shape itself, so the solver should not rely on crop origin beyond
#   being able to paste the merged result at (0,0) on the full-size canvas.
# - New required assumption: the merged tight sprite must fit inside the input canvas.
#   The generator now enforces this explicitly.
#
# d) Placement legality
# - The input still places base at (0,0) and plug at a disjoint random location.
# - Before sampling the final input canvas size, the generator computes the merged tight
#   output size and only chooses canvas sizes that can contain it.
# - The plug is then randomly placed without overlap; the solver re-docks it and pastes
#   the result into the output canvas at (0,0).
#
# e) Visual observability
# - Across examples, the merged shape is the same as the original task, but the output
#   now visibly preserves input dimensions: the right and/or bottom part of the output
#   remains background.
# - This is easy to infer from 2-4 examples because the input sizes vary while the merged
#   sprite sits flush to the output top-left instead of being tightly boxed.
#
# f) Visual cleanliness
# - No extra overlays or markers are added. The output simply has one merged sprite plus
#   unused background area, which is cleaner than introducing auxiliary signals.
#
# g) Color mapping
# - Unchanged from the original: ctrl disappears by recoloring to obj. No new color rule
#   is introduced.


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _connector_points(
    grid: np.ndarray, sprite: set[tuple[int, int]], ctrl: int
) -> set[tuple[int, int]]:
    return {(r, c) for (r, c) in sprite if int(grid[int(r), int(c)]) == int(ctrl)}


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _assemble_tight(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    a, b = comps

    conn_a = _connector_points(g, a, ctrl)
    conn_b = _connector_points(g, b, ctrl)
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
        assert (int(r + 1), int(c)) not in base_body
    for r, c in conn_p:
        assert (int(r - 1), int(c)) not in plug_body

    _sp, (pr0, pc0) = _norm_conn(conn_p)
    _sd, (dr0, dc0) = _norm_conn(conn_d)
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
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
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the plug sprite onto the base sprite, recolor the connectors to body color,
    then place the merged tight sprite at the top-left of a full-size output canvas.
    """
    g = np.array(grid, dtype=int)
    bg = 1
    h, w = map(int, g.shape)

    merged = _assemble_tight(g)
    mh, mw = map(int, merged.shape)
    assert mh <= h and mw <= w

    out = np.full((h, w), int(bg), dtype=int)
    paste(out, merged, 0, 0, transparent=bg)
    return out


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray | None = None,
) -> None:
    """Grow a connected shape on mask from frontier up to target pixels."""
    if allowed is None:
        allowed = np.ones_like(mask, dtype=bool)
    count = int(mask.sum())
    while count < target and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (
                0 <= r + dr < h
                and 0 <= c + dc < w
                and allowed[r + dr, c + dc]
                and not mask[r + dr, c + dc]
            )
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
    """Remove interior pixels while keeping the 8-connected shape intact."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (r, c) in keep:
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


def _crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _merged_from_cropped_sprites(base_sprite: np.ndarray, plug_sprite: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    temp = np.full((bh + ph + 2, bw + pw + 2), int(bg), dtype=int)
    paste(temp, base_sprite, 0, 0, transparent=bg)
    paste(temp, plug_sprite, bh + 1, bw + 1, transparent=bg)
    return _assemble_tight(temp)


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(3, 10)
    conn_len = _rng.randint(1, min(3, base_w))
    conn_row = _rng.randint(1, base_h - 1)
    conn_c0 = _rng.randint(0, base_w - conn_len)
    conn_cols = list(range(conn_c0, conn_c0 + conn_len))

    base_allowed = np.ones((base_h, base_w), dtype=bool)
    for c in conn_cols:
        base_allowed[conn_row, c] = False
        if conn_row + 1 < base_h:
            base_allowed[conn_row + 1 :, c] = False

    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    path_col = conn_cols[0]
    for c in range(min(0, path_col), max(0, path_col) + 1):
        base[0, c] = True
    for r in range(0, conn_row):
        base[r, path_col] = True
    for c in conn_cols:
        base[conn_row - 1, c] = True

    base_target_min = max(5, int(base.sum()))
    base_target_max = min(29, int(base_allowed.sum()))
    if base_target_min > base_target_max:
        return None
    _grow_shape(
        base,
        _rng.randint(base_target_min, base_target_max),
        list(zip(*np.where(base))),
        allowed=base_allowed,
    )
    if not is_contiguous(base, background=0, connectivity=8):
        return None

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[conn_row, c] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = conn_cols[0] - plug_conn_c0

    plug_allowed = np.zeros((plug_h, plug_w), dtype=bool)
    if plug_h > 1:
        plug_allowed[1, plug_conn_c0 : plug_conn_c0 + conn_len] = True
    if plug_h > 2:
        plug_allowed[2:, :] = True

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        plug[1, c] = True
    plug_target_min = max(5, int(plug.sum()))
    plug_target_max = min(29, int(plug_allowed.sum()) + conn_len)
    if plug_target_min > plug_target_max:
        return None
    _grow_shape(
        plug,
        _rng.randint(plug_target_min, plug_target_max),
        list(zip(*np.where(plug))),
        allowed=plug_allowed,
    )
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = conn_row - 1 + r
            dc = dc_offset + c
            if 0 <= dr < base_h and 0 <= dc < base_w:
                if base_grid[dr, dc] == obj:
                    if r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len:
                        continue
                    return None

    if dc_offset < 0:
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

    merged = _merged_from_cropped_sprites(base_sprite, plug_sprite, bg)
    mh, mw = map(int, merged.shape)

    size_choices = [
        (h, w)
        for h in range(max(7, bh, ph, mh), 12)
        for w in range(max(3, bw, pw, mw), 14)
        if h > mh or w > mw
    ]
    if not size_choices:
        return None
    canvas_h, canvas_w = _rng.choice(size_choices)

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=0)
    except ValueError:
        return None
    if pr + pc == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    try:
        merged_check = _assemble_tight(canvas)
        result = main(canvas)
    except Exception:
        return None

    if not np.array_equal(merged_check, merged):
        return None

    expected = np.full_like(canvas, bg)
    paste(expected, merged, 0, 0, transparent=bg)
    if not np.array_equal(result, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
