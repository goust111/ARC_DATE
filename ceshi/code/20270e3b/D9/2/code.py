# Variant of 20270e3b
# Dimension: D9 — 连接器形状参数 (meta)
# Change: 连接器为1-3像素长的水平条，位于单行上... → T形或十字形连接器

from __future__ import annotations

import random as _rng
import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D9 variant
# ---------------------------------------------------------------------------
# a) Concrete behavior and trace
# - The old task used a 1xL horizontal ctrl bar, so "match connectors" reduced to
#   matching a length and aligning one row by one row.
# - This variant keeps the same high-level action (identify base at the origin,
#   translate the plug so its connector docks one row above the base connector,
#   then recolor ctrl -> obj and tight-crop), but the connector itself is now a
#   small 2D pattern. We use three fixed templates:
#     * T4   = {(0,0),(0,1),(0,2),(1,1)}
#     * T5   = {(0,0),(0,1),(0,2),(1,1),(2,1)}
#     * X5   = {(0,1),(1,0),(1,1),(1,2),(2,1)}
# - Small trace:
#   old: bar {(0,0),(0,1),(0,2)} -> top-left alignment is enough.
#   new: cross {(0,1),(1,0),(1,1),(1,2),(2,1)} -> the solver must compare the
#   full normalized 2D ctrl set; comparing only row-span or width would confuse a
#   cross with a T-like shape of the same width.
#
# b) Complexity calibration
# - The connector family is intentionally small: three templates, no rotation.
#   This keeps the rule inferable from 2-4 examples while still making the input
#   visually different from the original single-row bars.
# - All generated examples repeat the same connector shape on both sprites inside
#   one grid, so the matching cue remains strong and unambiguous.
#
# c) Audit of old assumptions that break under D9
# - Old `norm(conn)` is still valid, but the meaning changes: it must preserve the
#   entire 2D ctrl set, not just a 1D length.
# - Old free-side checks iterated "for each ctrl pixel, no body below/base or
#   above/plug". That remains safe, but generator support is needed because a 2D
#   connector can have internal vertical neighbors.
# - Old placement `tr = (dr0 - 1) - pr0`, `tc = dc0 - pc0` is still usable if
#   both connectors are interpreted by their normalized bbox top-left. The new
#   generator therefore fixes identical orientation on both sprites.
# - Old overlap logic was harmless for 1-row bars. With 2D shapes, ctrl-on-ctrl
#   overlap can happen inside the docking zone, so the solver must allow overlap
#   whenever at least one side contributes ctrl at that cell.
#
# d) Placement legality
# - The generator keeps the plug connector at its own top-left corner and grows
#   the plug body strictly below the connector bbox; therefore the translated plug
#   never extends above or left of the base.
# - The base body is grown strictly above the connector bbox, so plug body can
#   only overlap the base on ctrl cells during docking, never on obj cells.
# - Canvas placement uses padding=1 with 8-neighborhood, guaranteeing exactly two
#   disconnected foreground components in the input.
#
# e/f) Visual observability and neatness
# - Every connector has bbox height >= 2, so the change is directly visible.
# - Bodies stay above (base) or below (plug) the ctrl shape, avoiding clutter near
#   the docking area and making the connector morphology easy to read.
#
# g) Color rule
# - Colors stay fixed as in the original task: bg=1, obj=4, ctrl=7. The only
#   color change is still deterministic recoloring ctrl -> obj after docking.


BG, OBJ, CTRL = 1, 4, 7

T4 = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
T5 = frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)})
X5 = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})

CONNECTOR_SPECS = {
    T4: {
        "top_attach": frozenset({(0, 0), (0, 1), (0, 2)}),
        "bottom_attach": frozenset({(1, 1)}),
    },
    T5: {
        "top_attach": frozenset({(0, 0), (0, 1), (0, 2)}),
        "bottom_attach": frozenset({(2, 1)}),
    },
    X5: {
        "top_attach": frozenset({(0, 1)}),
        "bottom_attach": frozenset({(2, 1)}),
    },
}

META_CONNECTORS = tuple(CONNECTOR_SPECS)


def _normalize_points(pts: set[tuple[int, int]]):
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (r0, c0)


def _top_attach(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return CONNECTOR_SPECS[shape]["top_attach"]


def _bottom_attach(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return CONNECTOR_SPECS[shape]["bottom_attach"]


def _shape_hw(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    h = max(r for r, _c in shape) + 1
    w = max(c for _r, c in shape) + 1
    return int(h), int(w)


def _crop_non_bg(grid: np.ndarray) -> np.ndarray | None:
    nz = np.argwhere(grid != BG)
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _connector_points(grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid == CTRL)}


def _assemble_output(base_grid: np.ndarray, plug_grid: np.ndarray) -> np.ndarray:
    base_conn = _connector_points(base_grid)
    plug_conn = _connector_points(plug_grid)
    _shape_b, (dr0, dc0) = _normalize_points(base_conn)
    _shape_p, (pr0, pc0) = _normalize_points(plug_conn)
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_grid != BG)}
    plug_pts = {(int(r + tr), int(c + tc)) for r, c in np.argwhere(plug_grid != BG)}
    coords_all = np.array(list(base_pts | plug_pts), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)

    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(base_grid[r, c])
    for r, c in np.argwhere(plug_grid != BG):
        rr = int(r + tr - rr0)
        cc = int(c + tc - cc0)
        out[rr, cc] = int(plug_grid[int(r), int(c)])

    out[out == CTRL] = OBJ
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != BG, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        if pts:
            comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    def bbox_of(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        coords = np.array(list(pts), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[r, c]) == CTRL}

    connA = conn(A)
    connB = conn(B)
    assert connA and connB

    shpA, (ar0, ac0) = _normalize_points(connA)
    shpB, (br0, bc0) = _normalize_points(connB)
    assert shpA == shpB
    assert shpA in META_CONNECTORS

    r0a, c0a, _r1a, _c1a = bbox_of(A)
    r0b, c0b, _r1b, _c1b = bbox_of(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD, shapeD, originD = A, connA, shpA, (ar0, ac0)
        plug, connP, shapeP, originP = B, connB, shpB, (br0, bc0)
    else:
        base, connD, shapeD, originD = B, connB, shpB, (br0, bc0)
        plug, connP, shapeP, originP = A, connA, shpA, (ar0, ac0)
    assert shapeD == shapeP

    base_body = base - connD
    plug_body = plug - connP

    for r, c in connD:
        assert (int(r + 1), int(c)) not in base_body
    for r, c in connP:
        assert (int(r - 1), int(c)) not in plug_body

    for r, c in _top_attach(shapeD):
        assert (int(originD[0] + r - 1), int(originD[1] + c)) in base_body
    for r, c in _bottom_attach(shapeP):
        assert (int(originP[0] + r + 1), int(originP[1] + c)) in plug_body

    tr = int((originD[0] - 1) - originP[0])
    tc = int(originD[1] - originP[1])
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    coords_all = np.array(list(base | plug_shift), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))

    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[r, c])

    for r, c in plug_shift:
        src_r, src_c = int(r - tr), int(c - tc)
        val = int(g[src_r, src_c])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != BG and val != BG:
            assert cur == CTRL or val == CTRL
        out[int(r - rr0), int(c - cc0)] = val

    out[out == CTRL] = OBJ
    nz = np.argwhere(out != BG)
    if nz.size == 0:
        return np.full((0, 0), BG, dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]], allowed: np.ndarray | None = None):
    count = int(mask.sum())
    h, w = mask.shape
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if mask[nr, nc]:
                continue
            if allowed is not None and not bool(allowed[nr, nc]):
                continue
            nbrs.append((nr, nc))
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None):
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]) or not mask[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _make_base(shape: frozenset[tuple[int, int]]) -> np.ndarray | None:
    shape_h, shape_w = _shape_hw(shape)
    top_front = _top_attach(shape)

    body_rows = _rng.randint(2, max(2, 7 - shape_h))
    h = int(body_rows + shape_h)
    w = _rng.randint(max(3, shape_w), 10)
    conn_r0 = int(body_rows)
    conn_c0 = _rng.randint(0, w - shape_w)

    body = np.zeros((h, w), dtype=bool)
    seeds = {(int(conn_r0 + r - 1), int(conn_c0 + c)) for r, c in top_front}
    allowed = np.zeros((h, w), dtype=bool)
    allowed[:conn_r0, :] = True
    if any(not allowed[r, c] for r, c in seeds):
        return None
    for r, c in seeds:
        body[r, c] = True

    max_target = min(29 - len(shape), int(allowed.sum()))
    min_target = max(len(seeds), 5 - len(shape))
    if max_target < min_target:
        return None
    target = _rng.randint(min_target, max_target)
    _grow_shape(body, target, list(seeds), allowed=allowed)
    _thin(body, keep=seeds)

    grid = np.full((h, w), BG, dtype=int)
    grid[body] = OBJ
    for r, c in shape:
        grid[int(conn_r0 + r), int(conn_c0 + c)] = CTRL

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    return _crop_non_bg(grid)


def _make_plug(shape: frozenset[tuple[int, int]]) -> np.ndarray | None:
    shape_h, shape_w = _shape_hw(shape)
    bottom_front = _bottom_attach(shape)

    body_rows = _rng.randint(2, max(2, 7 - shape_h))
    h = int(shape_h + body_rows)
    w = _rng.randint(shape_w, 10)

    body = np.zeros((h, w), dtype=bool)
    seeds = {(int(r + 1), int(c)) for r, c in bottom_front}
    allowed = np.zeros((h, w), dtype=bool)
    allowed[shape_h:, :] = True
    if any(r >= h or c >= w or not allowed[r, c] for r, c in seeds):
        return None
    for r, c in seeds:
        body[r, c] = True

    max_target = min(29 - len(shape), int(allowed.sum()))
    min_target = max(len(seeds), 5 - len(shape))
    if max_target < min_target:
        return None
    target = _rng.randint(min_target, max_target)
    _grow_shape(body, target, list(seeds), allowed=allowed)
    _thin(body, keep=seeds)

    grid = np.full((h, w), BG, dtype=int)
    grid[body] = OBJ
    for r, c in shape:
        grid[int(r), int(c)] = CTRL

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    return _crop_non_bg(grid)


def _valid_docking(base_grid: np.ndarray, plug_grid: np.ndarray) -> bool:
    base_conn = _connector_points(base_grid)
    plug_conn = _connector_points(plug_grid)
    shp_b, (dr0, dc0) = _normalize_points(base_conn)
    shp_p, (pr0, pc0) = _normalize_points(plug_conn)
    if shp_b != shp_p or shp_b not in META_CONNECTORS:
        return False

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    if tr < 0 or tc < 0:
        return False

    has_base_ctrl_overlap = False
    for r, c in np.argwhere(plug_grid != BG):
        rr = int(r + tr)
        cc = int(c + tc)
        if 0 <= rr < base_grid.shape[0] and 0 <= cc < base_grid.shape[1]:
            if int(base_grid[rr, cc]) == OBJ and int(plug_grid[int(r), int(c)]) == OBJ:
                return False
            if int(base_grid[rr, cc]) == CTRL:
                has_base_ctrl_overlap = True
    return has_base_ctrl_overlap


def _gen_attempt() -> np.ndarray | None:
    shape = _rng.choice(META_CONNECTORS)
    base_sprite = _make_base(shape)
    if base_sprite is None:
        return None
    plug_sprite = _make_plug(shape)
    if plug_sprite is None:
        return None
    if not _valid_docking(base_sprite, plug_sprite):
        return None

    expected = _assemble_output(base_sprite, plug_sprite)

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    min_h = max(7, max(bh, ph) + 1)
    min_w = max(3, max(bw, pw) + 1)
    if min_h > 11 or min_w > 13:
        return None
    canvas_h = _rng.randint(min_h, 11)
    canvas_w = _rng.randint(min_w, 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=BG, padding=0)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=BG)

    lab, n = label_mask(canvas != BG, connectivity=8)
    if int(n) != 2:
        return None
    if np.sum(canvas == BG) <= canvas.size // 2:
        return None

    try:
        got = main(canvas)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return canvas


def generate_input():
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
