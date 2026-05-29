# Combo variant of 20270e3b (index 21)
# combo_spec: D5/1+D9/1+D13/1
# Name: 矩形连接器扩散变色

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


# -----------------------------------------------------------------------------
# Design analysis for the fused variant
# -----------------------------------------------------------------------------
#
# a) Concretization and behavior trace
#    - Relative to the source task, the connector is no longer a 1xL bar. Each
#      sprite now carries one solid 2x2 or 2x3 ctrl rectangle.
#    - Docking is the rectangle-height generalization of the old rule:
#      the plug rectangle is translated so its bottom row ends one row above the
#      base rectangle's top row, with the same columns.
#    - After stamping, visible ctrl is NOT recolored away. Instead it stays
#      visible and spreads one 4-neighbor layer into adjacent obj pixels.
#    - If shifted plug obj lands on existing base obj away from the seam, that
#      cell becomes a dedicated overlap color.
#    - Small trace for a 2x2 connector, around the seam:
#        base support/body row:      44
#        base connector rows:        77 / 77
#        plug connector rows:    77 / 77
#        plug first body rows:       44 / 44...
#      Original task: connector rows would eventually become 44, and deeper
#      obj-obj collisions were illegal.
#      Fused task: the plug connector remains 77, adjacent 44 cells touching it
#      become extra 77 welding marks, and any deeper 44-over-44 collisions turn
#      into overlap color 2. The final output therefore visibly contains:
#      normal body 4, welding ctrl 7, and overlap 2.
#
# b) Complexity calibration
#    - The connector has only two widths (2 or 3) and fixed height 2. That is
#      visually obvious without creating too many cases.
#    - The overlap color is restricted to true obj-vs-obj intersections only.
#      This keeps the new color semantically clean.
#    - The generator forces all three foreground appearances to coexist:
#      retained/diffused ctrl, at least one overlap cell, and at least one
#      untouched obj cell. This prevents degeneration back toward the source
#      task or toward a purely overlap-marking variant.
#
# c) Hidden-assumption audit of the source main()
#    - Source connector normalization assumed a 1-row bar; fused solver must
#      validate a solid 2x2 or 2x3 rectangle instead.
#    - Source docking shift used `dr0 - 1 - pr0`; the hard-coded `1` was the
#      old connector height. It must become `base_top - conn_h - plug_top`.
#    - Source per-pixel seam checks assumed each ctrl pixel directly faced body
#      on one side and empty on the other. For a 2-row rectangle, only the
#      outer support/free rows matter; internal ctrl rows are not body-facing.
#    - Source overlap assert rejected every non-ctrl overlap. Fused solver must
#      intercept obj-vs-obj and paint overlap color 2 instead.
#    - Source `out[out == ctrl] = obj` would erase the D5 behavior, so it is
#      replaced by one-shot ctrl diffusion computed from the pre-diffusion ctrl
#      mask. Diffusion only targets obj cells, so overlap color is preserved.
#
# d) Placement legality
#    - The generator first constructs a legal assembled state in docked
#      coordinates, then crops the two sprites and places them separately on the
#      input canvas. This guarantees the solver's translation is valid.
#    - Base stays anchored at the canvas origin. The plug is placed strictly to
#      the right or below with a positive gap, so the input always has exactly
#      two 8-connected components.
#    - The assembled-state construction keeps the plug entirely within the base
#      column frame before cropping, which guarantees the later docking shift
#      never makes the plug extend left of the base.
#
# e) Visual observability
#    - A human sees the same docking cue repeated twice: both sprites have the
#      same orange rectangle. The output then shows three visually distinct seam
#      effects: orange welding, red overlap, and remaining yellow body.
#    - Because the generator rejects samples missing any of these effects, the
#      fused rule is inferable from a few examples rather than hidden in a rare
#      corner case.
#
# f) Cleanliness
#    - The new visual semantics are spatially separated: ctrl stays concentrated
#      around the seam, overlap appears only where bodies cross deeper inside,
#      and untouched obj remains elsewhere.
#    - Diffusion is limited to exactly one 4-neighbor layer, avoiding noisy
#      color floods.
#
# g) Color-change inferability
#    - There is one structural rule for each non-background output color:
#      7 = connector or one-step weld diffusion around it,
#      2 = body-body overlap after docking,
#      4 = body pixels affected by neither of the above.
#    - No rule depends on hidden numeric ordering of colors; only visible shape,
#      adjacency, and overlap matter.
#
# h) New solver assumptions and generator guarantees
#    - Assumption: each sprite contains exactly one solid ctrl rectangle of size
#      2x2 or 2x3. Guarantee: generator explicitly constructs exactly one.
#    - Assumption: base has a full obj support row above the rectangle and no
#      obj directly below it in connector columns. Guarantee: generator seeds
#      and preserves that support row and blocks the free-side row.
#    - Assumption: plug has a full obj support row directly below its rectangle.
#      Guarantee: generator seeds and preserves that row.
#    - Assumption: after docking, at least one obj-vs-obj overlap exists and at
#      least one obj cell survives unaffected. Guarantee: generator checks the
#      final output statistics and rejects degenerate samples.
# -----------------------------------------------------------------------------


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _connector_info(ctrl_pts: set[tuple[int, int]]) -> dict[str, int | set[tuple[int, int]]]:
    assert ctrl_pts
    r0, c0, r1, c1 = _bbox_of_pts(ctrl_pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == 2 and w in (2, 3)
    shape = {(int(r - r0), int(c - c0)) for r, c in ctrl_pts}
    rect = {(dr, dc) for dr in range(h) for dc in range(w)}
    assert shape == rect
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(h),
        "w": int(w),
        "shape": shape,
    }


def _render_fused(
    g: np.ndarray,
    base: set[tuple[int, int]],
    plug: set[tuple[int, int]],
    base_info: dict[str, int | set[tuple[int, int]]],
    plug_info: dict[str, int | set[tuple[int, int]]],
    *,
    bg: int,
    obj: int,
    ctrl: int,
    overlap: int,
) -> np.ndarray:
    tr = int(base_info["r0"]) - int(plug_info["h"]) - int(plug_info["r0"])
    tc = int(base_info["c0"]) - int(plug_info["c0"])
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))

    base_coords = np.array(list(base), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
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

    ctrl_mask = out == int(ctrl)
    weld_mask = np.zeros_like(ctrl_mask, dtype=bool)
    for r, c in np.argwhere(ctrl_mask):
        r = int(r)
        c = int(c)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and int(out[nr, nc]) == int(obj):
                weld_mask[nr, nc] = True
    out[weld_mask] = int(ctrl)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two sprites by matching 2D rectangular connectors, then tight-crop.

    Fused rule:
    - ctrl is a solid 2x2 or 2x3 rectangle on each sprite.
    - Base is the component whose bbox starts at the input origin.
    - Translate the plug so its rectangle sits directly above the base rectangle.
    - Preserve visible ctrl, diffuse it one 4-neighbor layer into adjacent obj,
      and recolor any obj-vs-obj overlap to a dedicated third color.
    """
    g = np.array(grid, dtype=int)

    bg, obj, ctrl, overlap = 1, 4, 7, 2

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    connA = conn(A)
    connB = conn(B)
    infoA = _connector_info(connA)
    infoB = _connector_info(connB)
    assert infoA["shape"] == infoB["shape"]

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD, infoD = A, connA, infoA
        plug, connP, infoP = B, connB, infoB
    else:
        base, connD, infoD = B, connB, infoB
        plug, connP, infoP = A, connA, infoA

    base_body = base - connD
    plug_body = plug - connP
    for c in range(int(infoD["c0"]), int(infoD["c1"]) + 1):
        assert (int(infoD["r0"]) - 1, int(c)) in base_body
        assert (int(infoD["r1"]) + 1, int(c)) not in base_body
    for c in range(int(infoP["c0"]), int(infoP["c1"]) + 1):
        assert (int(infoP["r0"]) - 1, int(c)) not in plug_body
        assert (int(infoP["r1"]) + 1, int(c)) in plug_body

    out = _render_fused(
        g,
        base,
        plug,
        infoD,
        infoP,
        bg=bg,
        obj=obj,
        ctrl=ctrl,
        overlap=overlap,
    )

    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray,
) -> None:
    count = int(mask.sum())
    frontier = list(frontier)
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h
            and 0 <= c + dc < w
            and bool(allowed[r + dr, c + dc])
            and not bool(mask[r + dr, c + dc])
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
    keep = keep or set()
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        rc = (int(r), int(c))
        if rc in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not bool(mask[r + dr, c + dc])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _build_base(conn_h: int, conn_w: int) -> tuple[np.ndarray, dict[str, int]] | None:
    bg, obj, ctrl = 1, 4, 7

    base_h = _rng.randint(6, 7)
    base_w = _rng.randint(max(5, conn_w + 2), 9)
    conn_r0 = _rng.randint(2, base_h - 3)
    conn_c0 = _rng.randint(0, base_w - conn_w)
    conn_r1 = int(conn_r0 + conn_h - 1)

    blocked = np.zeros((base_h, base_w), dtype=bool)
    blocked[conn_r1 + 1, conn_c0 : conn_c0 + conn_w] = True
    allowed = ~blocked

    stem_choices = [c for c in range(base_w) if c < conn_c0 or c > conn_c0 + conn_w - 1]
    if not stem_choices:
        return None
    stem_c = _rng.choice(stem_choices)
    lower_r = _rng.randint(conn_r1 + 1, base_h - 1)
    lower_choices = [c for c in range(base_w) if not bool(blocked[lower_r, c])]
    if not lower_choices:
        return None
    lower_c = _rng.choice(lower_choices)

    mask = np.zeros((base_h, base_w), dtype=bool)
    keep: set[tuple[int, int]] = set()

    for c in range(0, stem_c + 1):
        mask[0, c] = True
        keep.add((0, c))
    for r in range(base_h):
        mask[r, stem_c] = True
        keep.add((r, stem_c))
    for c in range(conn_c0, conn_c0 + conn_w):
        mask[conn_r0 - 1, c] = True
        keep.add((conn_r0 - 1, c))
    bridge_c = conn_c0 if stem_c < conn_c0 else conn_c0 + conn_w - 1
    for c in range(min(stem_c, bridge_c), max(stem_c, bridge_c) + 1):
        mask[conn_r0 - 1, c] = True
        keep.add((conn_r0 - 1, c))
    for c in range(conn_c0, conn_c0 + conn_w):
        for r in range(conn_r0, conn_r0 + conn_h):
            mask[r, c] = True
            keep.add((r, c))
    for c in range(min(stem_c, lower_c), max(stem_c, lower_c) + 1):
        mask[lower_r, c] = True
        keep.add((lower_r, c))

    target_min = max(int(mask.sum()) + 1, 10)
    target_max = min(26, int(allowed.sum()))
    if target_max < target_min:
        return None
    _grow_shape(mask, _rng.randint(target_min, target_max), list(zip(*np.where(mask))), allowed)
    _thin(mask, keep=keep)

    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if any(mask[conn_r1 + 1, c] for c in range(conn_c0, conn_c0 + conn_w)):
        return None
    if not all(mask[conn_r0 - 1, c] for c in range(conn_c0, conn_c0 + conn_w)):
        return None
    if not any(mask[r, c] for r in range(conn_r1 + 1, base_h) for c in range(base_w)):
        return None

    grid = np.full((base_h, base_w), int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[conn_r0 : conn_r0 + conn_h, conn_c0 : conn_c0 + conn_w] = int(ctrl)
    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        return None

    return grid, {"r0": conn_r0, "c0": conn_c0, "h": conn_h, "w": conn_w}


def _build_plug(
    base_grid: np.ndarray,
    base_info: dict[str, int],
    conn_h: int,
    conn_w: int,
) -> tuple[np.ndarray, int] | None:
    bg, obj, ctrl = 1, 4, 7

    conn_r0 = int(base_info["r0"])
    conn_c0 = int(base_info["c0"])
    base_h, base_w = base_grid.shape
    conn_cols = list(range(conn_c0, conn_c0 + conn_w))

    candidates = [
        (int(r), int(c))
        for r in range(conn_r0 + conn_h, base_h)
        for c in range(base_w)
        if int(base_grid[r, c]) == int(obj)
    ]
    if not candidates:
        return None
    near = [
        (r, c)
        for r, c in candidates
        if r <= conn_r0 + conn_h + 2 and conn_c0 - 3 <= c <= conn_c0 + conn_w + 2
    ]
    pool = near if near else candidates
    targets = [_rng.choice(pool)]
    others = [pt for pt in pool if pt != targets[0]]
    if others and _rng.random() < 0.45:
        targets.append(_rng.choice(others))

    deepest_local = max(int(base_r - conn_r0 + conn_h) for base_r, _base_c in targets)
    plug_h = min(7, max(conn_h + 3, deepest_local + _rng.randint(1, 2)))
    if plug_h <= deepest_local:
        return None

    mask = np.zeros((plug_h, base_w), dtype=bool)
    keep: set[tuple[int, int]] = set()

    for r in range(conn_h):
        for c in conn_cols:
            mask[r, c] = True
            keep.add((r, c))
    for c in conn_cols:
        mask[conn_h, c] = True
        keep.add((conn_h, c))

    for base_r, base_c in targets:
        local_r = int(base_r - conn_r0 + conn_h)
        if not (conn_h <= local_r < plug_h):
            return None
        anchor_c = int(base_c)
        if anchor_c < conn_cols[0]:
            join_c = conn_cols[0]
        elif anchor_c > conn_cols[-1]:
            join_c = conn_cols[-1]
        else:
            join_c = anchor_c
        for c in range(min(join_c, anchor_c), max(join_c, anchor_c) + 1):
            mask[conn_h, c] = True
            keep.add((conn_h, c))
        for r in range(conn_h, local_r + 1):
            mask[r, anchor_c] = True
            keep.add((r, anchor_c))

    allowed = np.ones_like(mask, dtype=bool)
    target_min = max(int(mask.sum()) + 1, 8)
    target_max = min(22, int(mask.size))
    if target_max < target_min:
        return None
    _grow_shape(mask, _rng.randint(target_min, target_max), list(zip(*np.where(mask))), allowed)
    _thin(mask, keep=keep)

    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if not all(mask[conn_h, c] for c in conn_cols):
        return None

    grid = np.full((plug_h, base_w), int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[:conn_h, conn_c0 : conn_c0 + conn_w] = int(ctrl)
    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        return None

    overlap_count = 0
    for r, c in zip(*np.where(grid == int(obj))):
        dr = int(conn_r0 - conn_h + r)
        dc = int(c)
        if 0 <= dr < base_h and 0 <= dc < base_w and int(base_grid[dr, dc]) == int(obj):
            overlap_count += 1
    if overlap_count == 0:
        return None

    return grid, overlap_count


def _place_input(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray | None:
    bg = 1
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    placements: list[tuple[str, int]] = []
    for gap in range(1, 6):
        if bh + gap + ph <= 11:
            placements.append(("below", gap))
        if bw + gap + pw <= 13:
            placements.append(("right", gap))
    if not placements:
        return None

    orient, gap = _rng.choice(placements)
    if orient == "below":
        canvas_h = _rng.randint(max(7, bh + gap + ph), 11)
        canvas_w = _rng.randint(max(5, bw, pw), 13)
        pr = bh + gap
        pc = _rng.randint(0, canvas_w - pw)
    else:
        canvas_h = _rng.randint(max(7, bh, ph), 11)
        canvas_w = _rng.randint(max(5, bw + gap + pw), 13)
        pr = _rng.randint(0, canvas_h - ph)
        pc = bw + gap

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    _lab, n = label_mask(canvas != int(bg), connectivity=8)
    if int(n) != 2:
        return None
    if int(np.sum(canvas == int(bg))) <= canvas.size // 3:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl, overlap = 1, 4, 7, 2

    conn_h = 2
    conn_w = _rng.choice((2, 3))
    built_base = _build_base(conn_h, conn_w)
    if built_base is None:
        return None
    base_grid, base_info = built_base

    built_plug = _build_plug(base_grid, base_info, conn_h, conn_w)
    if built_plug is None:
        return None
    plug_grid, expected_overlap = built_plug

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprite = _crop_non_bg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None

    canvas = _place_input(base_sprite, plug_sprite)
    if canvas is None:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None

    if result.size == 0 or not np.any(result != int(bg)):
        return None
    if int(np.sum(result == int(overlap))) == 0:
        return None
    ctrl_count = int(np.sum(result == int(ctrl)))
    obj_count = int(np.sum(result == int(obj)))
    if ctrl_count <= conn_h * conn_w:
        return None
    if obj_count == 0:
        return None
    if expected_overlap <= 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
