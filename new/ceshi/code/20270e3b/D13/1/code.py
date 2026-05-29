# Variant of 20270e3b
# Dimension: D13 — 重叠处理规则 (meta)
# Change: 对接时仅允许在连接器区域重叠（ctrl遇到obj），其他位置不允许重叠
#         -> 仍按原规则对齐连接器，但当两个精灵的 body 像素重叠时，该位置改为第三种颜色。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for this D13 variant
#
# a) Concretization / behavior trace
#    - Colors stay mostly fixed as in the source task: bg=1, obj=4, ctrl=7.
#      The variant adds one output-only marker color: overlap=2 (red).
#    - Input geometry is unchanged: two 8-connected sprites, each with one
#      horizontal ctrl bar; the base sprite is the one whose bbox starts at
#      the grid origin; the plug is translated so its connector row lands one
#      row above the base connector.
#    - Original docking trace:
#        base obj row
#        base ctrl row
#        plug ctrl row  -> overlaps base obj, then later recolors to obj
#        plug obj row   -> overlaps base ctrl, then later recolors to obj
#      Any deeper obj-vs-obj overlap used to assert.
#    - Variant docking trace:
#      The same connector alignment is used, but if a shifted plug obj lands on
#      an existing base obj, that cell becomes overlap=2 instead of erroring.
#      Connector-related overlaps keep their old meaning and collapse to obj.
#    - Small example:
#        base rows near seam:    444 / 777 / 141
#        plug rows near seam:    777 / 444 / 414
#      Original: only the 777/444 seam is legal; if the bottom 414 also hits
#      base body, solver fails. Variant: those deeper body-body hits become 2.
#
# b) Complexity calibration
#    - Only one new color and one new overlap class are introduced.
#    - The connector still determines the unique translation, so the puzzle does
#      not become ambiguous.
#    - The generator forces at least one body-body overlap, so the new rule is
#      visible in every valid sample rather than appearing as a rare corner case.
#
# c) Hidden-assumption audit of the original main()
#    - `assert cur == ctrl or val == ctrl` depended on "non-connector overlaps
#      never happen". This is the primary line invalidated by D13.
#    - `out[...] = val` was harmless before because every legal overlap involved
#      ctrl and later recolored to obj. In the variant, that same overwrite would
#      erase body-body intersections, so those cells must be intercepted first.
#    - The connector normalization, base selection, translation, and crop logic
#      do not rely on the old overlap prohibition and can stay unchanged.
#
# d) Placement legality
#    - The generator still places the two input sprites without overlap and fully
#      in bounds.
#    - It constructs the plug in docked coordinates first, guaranteeing that
#      after the solver's translation there will be at least one obj-vs-obj
#      intersection and no extension above/left of the base.
#
# e) Visual observability
#    - Humans can still infer the old "dock by matching horizontal orange bars"
#      rule from the input.
#    - The variant is visible because the output now contains red exactly where
#      the merged bodies cross; every generated example contains at least one red
#      cell.
#
# f) Visual cleanliness
#    - Only one extra layer of meaning is added: red marks crossings.
#    - Connector colors disappear in the output as before, so the final image is
#      not cluttered by both seam markers and overlap markers simultaneously.
#
# g) Color-change inferability
#    - The new color is structurally localized: it appears only where both
#      sprites would occupy the same body cell after docking.
#    - There is a single deterministic mapping: obj+obj overlap -> red.
#
# h) Source task review
#    - The variant description cites no separate `source_task`, so there is no
#      additional reference solver to inspect.
#
# i) New assumptions introduced by this variant
#    - `main()` now assumes body-body overlap, if present, uses the fixed obj
#      color on both sprites (still true because inputs keep obj=4).
#    - `generate_input()` guarantees at least one such overlap so the new branch
#      is exercised; otherwise the variant could degenerate back to the source
#      task behavior.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then tight-crop.

    Variant rule:
    - Docking translation is unchanged from the source task.
    - Remaining ctrl pixels still disappear into obj.
    - New behavior: if base body and shifted plug body overlap, paint that cell
      with a dedicated overlap color instead of rejecting the input.
    """
    g = np.array(grid, dtype=int)

    bg, obj, ctrl, overlap = 1, 4, 7, 2

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A = set(comps[0])
    B = set(comps[1])

    def bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        coords = np.array(list(pts), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    def norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
        rs = [r for r, _c in conn_pts]
        cs = [c for _r, c in conn_pts]
        r0, c0 = int(min(rs)), int(min(cs))
        shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
        return shape, (int(r0), int(c0))

    connA = conn(A)
    connB = conn(B)
    assert connA and connB
    shpA, _anchorA = norm(connA)
    shpB, _anchorB = norm(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
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

    _sp, (pr0, pc0) = norm(connP)
    _sd, (dr0, dc0) = norm(connD)
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
        rr = int(r - rr0)
        cc = int(c - cc0)
        cur = int(out[rr, cc])
        if cur == int(bg):
            out[rr, cc] = int(val)
            continue
        if cur == int(obj) and val == int(obj):
            out[rr, cc] = int(overlap)
            continue
        assert cur == int(ctrl) or val == int(ctrl)
        out[rr, cc] = int(val)

    out[out == int(ctrl)] = int(obj)

    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# ---------------------------------------------------------------------------
# Generator (assembled-state-first, but still validates through `main`)
# ---------------------------------------------------------------------------


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    """Grow a connected shape on `mask` from `frontier` up to `target` pixels."""
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
    """Remove removable interior pixels while keeping the shape 8-connected."""
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
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _overlap_candidates(base_grid: np.ndarray, conn_row: int) -> list[tuple[int, int]]:
    bg, obj = 1, 4
    candidates = []
    h, w = base_grid.shape
    for r in range(int(conn_row), h):
        for c in range(w):
            if int(base_grid[r, c]) == int(obj):
                candidates.append((int(r), int(c)))
    return candidates


def _build_plug_mask(
    base_grid: np.ndarray,
    conn_row: int,
    conn_cols: list[int],
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    """
    Build the plug in docked coordinates.

    Local row 0 is the connector row, which the solver will place at
    `base connector row - 1`. The plug uses the same column frame as the base,
    so it can never extend left of the base after docking.
    """
    obj = 4
    base_h, base_w = base_grid.shape
    candidates = _overlap_candidates(base_grid, conn_row)
    if not candidates:
        return None

    near = [
        (r, c)
        for r, c in candidates
        if r <= int(conn_row) + 3 and int(conn_cols[0]) - 3 <= c <= int(conn_cols[-1]) + 3
    ]
    pool = near if near else candidates
    target1 = _rng.choice(pool)
    targets = [target1]
    others = [pt for pt in pool if pt != target1]
    if others and _rng.random() < 0.4:
        targets.append(_rng.choice(others))

    deepest_local = max(int(r - conn_row + 1) for r, _c in targets)
    plug_h = min(7, max(2, deepest_local + 1 + _rng.randint(0, 1)))
    if plug_h <= deepest_local:
        return None

    plug = np.zeros((plug_h, base_w), dtype=bool)
    keep: set[tuple[int, int]] = set()

    for c in conn_cols:
        plug[0, c] = True
        plug[1, c] = True
        keep.add((0, c))
        keep.add((1, c))

    for base_r, base_c in targets:
        local_r = int(base_r - conn_row + 1)
        if not (1 <= local_r < plug_h):
            return None
        if base_c < int(conn_cols[0]):
            start_c = int(conn_cols[0])
        elif base_c > int(conn_cols[-1]):
            start_c = int(conn_cols[-1])
        else:
            start_c = int(base_c)
        for c in range(min(start_c, int(base_c)), max(start_c, int(base_c)) + 1):
            plug[1, c] = True
        for r in range(1, local_r + 1):
            plug[r, int(base_c)] = True
        keep.add((local_r, int(base_c)))

    cur = int(plug.sum())
    target = _rng.randint(max(5, cur), min(18, plug.size))
    _grow_shape(plug, target, list(zip(*np.where(plug))))
    _thin(plug, keep=keep)

    for c in conn_cols:
        if plug_h < 2 or not plug[1, c]:
            return None
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    overlaps_local: set[tuple[int, int]] = set()
    for r, c in zip(*np.where(plug)):
        dr = int(conn_row - 1 + r)
        dc = int(c)
        if 0 <= dr < base_h and 0 <= dc < base_w and int(base_grid[dr, dc]) == int(obj):
            overlaps_local.add((int(r), int(c)))
    if not overlaps_local:
        return None

    return plug, overlaps_local


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl, overlap = 1, 4, 7, 2

    base_h = _rng.randint(3, 6)
    base_w = _rng.randint(3, 9)
    base_target = _rng.randint(6, min(24, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates = []
    for r in range(1, base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            tmp = np.full((base_h, base_w), bg, dtype=int)
            tmp[base] = obj
            for c in cols:
                tmp[r, c] = ctrl
            if _overlap_candidates(tmp, r):
                candidates.append((r, cols))
    if not candidates:
        return None
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[conn_row, c] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    built = _build_plug_mask(base_grid, conn_row, conn_cols)
    if built is None:
        return None
    plug_mask, overlap_local = built
    plug_h, plug_w = plug_mask.shape

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug_mask] = obj
    for c in conn_cols:
        plug_grid[0, c] = ctrl
    for c in conn_cols:
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
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(4, max(bw, pw) + 1), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=0)
    except ValueError:
        return None
    if pr + pc == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    if int(np.sum(canvas == bg)) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != bg):
        return None
    if int(np.sum(result == int(overlap))) == 0:
        return None

    # Sanity check: the docked plug was constructed to exercise the new branch.
    if not overlap_local:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
