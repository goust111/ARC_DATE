# Variant of 20270e3b
# Dimension: D6 — base角色判定规则 (meta)
# Change: 外接矩形左上角满足 r+c==0（即紧贴网格原点）的精灵为base... -> 连接器朝下者为base

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


"""
Design analysis for the D6 variant
==================================

Source-task review:
- No external source task was provided for this variant, so the baseline is the
  original 20270e3b solver/generator only.

a) Concretization and behavior trace
------------------------------------
- The visible structure is still two thin 8-connected sprites in colors:
  background=1, body=4, connector=7.
- Each sprite has exactly one horizontal connector bar of length 1..3.
- The connector orientation is visible from local adjacency:
  a downward-facing connector has body pixels directly above it and empty cells
  directly below it; an upward-facing connector has the opposite pattern.
- The output transformation is unchanged: translate the plug so that its
  connector sits one row above the base connector, stamp, recolor connector
  pixels to body color, and tight-crop.

Concrete trace on the changed step:
- Old logic: if sprite P is at input top-left and sprite Q is elsewhere, P is
  declared base even if P's connector faces up.
- New logic: inspect connector neighborhoods. If P has body below / free above
  and Q has body above / free below, then Q is base and P is plug regardless of
  placement.
- Difference: role selection now depends on a local shape cue that remains
  visible after arbitrary input placement, instead of on the global position.

b) Complexity calibration
-------------------------
- The variant changes exactly one decision axis: role selection.
- To make that change observable, the generator intentionally places the plug at
  the input origin and places the base elsewhere. This creates frequent
  disagreement with the old heuristic while keeping every other rule stable.
- Connector length remains in a small range (1..3), so matching is still easy
  and visually repeated within the same example.

c) Hidden assumption audit for main()
------------------------------------
- Foreground component extraction still assumes exactly two 8-connected
  components. The generator enforces non-overlap and non-diagonal-touching.
- Connector matching still assumes both connectors have identical normalized
  shapes. The generator builds them from the same chosen length.
- The original lines that used bbox top-left / r+c==0 are invalid in this
  variant and are replaced by an explicit connector-orientation classifier.
- The output assertion that the docked plug does not extend above/left of the
  base is still valid. It is a property of the docking geometry, not of the
  input placement, and the generator still enforces it while constructing the
  docked pair.

d) Placement legality
---------------------
- Input placement now pastes the plug at (0,0), then places the base with
  `padding=1`, which prevents overlap and 8-neighbor touching.
- Canvas size is chosen only from sizes where at least one separated placement
  axis is feasible, so `random_free_location` has a real chance to succeed.
- The generator still oracle-checks every sample with `main()`.

e) Visual observability
-----------------------
- Humans can now see that the sprite at the top-left is not necessarily the one
  that stays fixed in the output. The deciding cue is the connector direction.
- Because the plug is often the top-left sprite, the old rule is visibly wrong
  on the generated examples, which makes the D6 change learnable.

f) Visual cleanliness
---------------------
- Only two sprites exist; they are separated by at least one cell under
  8-connectivity, so components remain visually distinct.
- No extra markers, colors, or overlays are introduced.

g) Color inference
------------------
- No color remapping rule changed. Color 7 still marks connectors and is
  recolored to body color 4 in the output, exactly as in the original task.

h) New solver assumptions introduced by the variant
---------------------------------------------------
- Each connector must be classifiable as exactly one of {up, down} using direct
  vertical neighbors in the sprite body.
- There must be exactly one downward-facing connector and one upward-facing
  connector among the two sprites.
- The generator guarantees these conditions by construction.
"""


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _connector_facing(
    sprite: set[tuple[int, int]], conn_pts: set[tuple[int, int]]
) -> str:
    body = sprite - conn_pts
    has_body_above = all((int(r - 1), int(c)) in body for r, c in conn_pts)
    has_body_below = all((int(r + 1), int(c)) in body for r, c in conn_pts)
    free_above = all((int(r - 1), int(c)) not in body for r, c in conn_pts)
    free_below = all((int(r + 1), int(c)) not in body for r, c in conn_pts)
    if has_body_above and free_below:
        return "down"
    if has_body_below and free_above:
        return "up"
    raise AssertionError("connector orientation is ambiguous")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then tight-crop.

    Variant D6:
    - base is the sprite whose connector faces down (body above, free side below)
    - plug is the sprite whose connector faces up (body below, free side above)
    - input position is irrelevant for role selection
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    connA = conn(A)
    connB = conn(B)
    assert connA and connB

    shpA, _ = _norm_conn(connA)
    shpB, _ = _norm_conn(connB)
    assert shpA == shpB

    faceA = _connector_facing(A, connA)
    faceB = _connector_facing(B, connB)
    assert {faceA, faceB} == {"down", "up"}
    if faceA == "down":
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    base_body = base - connD
    plug_body = plug - connP
    for r, c in connD:
        assert (int(r - 1), int(c)) in base_body
        assert (int(r + 1), int(c)) not in base_body
    for r, c in connP:
        assert (int(r + 1), int(c)) in plug_body
        assert (int(r - 1), int(c)) not in plug_body

    _sp, (pr0, pc0) = _norm_conn(connP)
    _sd, (dr0, dc0) = _norm_conn(connD)

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))

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
    """Grow a connected shape on mask from frontier up to target pixels."""
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


def _grow_shape_allowed(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray,
) -> bool:
    """Grow a connected shape using only cells marked True in `allowed`."""
    count = int(mask.sum())
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
            and not mask[r + dr, c + dc]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))
    return count == int(target)


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    """Remove interior pixels while preserving 8-connectivity."""
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


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    base_grid = None
    conn_row = -1
    conn_cols: list[int] = []
    for _ in range(20):
        base_h = _rng.randint(2, 7)
        base_w = _rng.randint(3, 10)
        base_target = _rng.randint(5, min(29, base_h * base_w))
        base = np.zeros((base_h, base_w), dtype=bool)
        base[0, 0] = True
        _grow_shape(base, base_target, [(0, 0)])
        if not is_contiguous(base, background=0, connectivity=8):
            continue

        conn_len = _rng.randint(1, min(3, base_w))
        candidates: list[tuple[int, list[int]]] = []
        for r in range(1, base_h):
            for c0 in range(base_w - conn_len + 1):
                cols = list(range(c0, c0 + conn_len))
                if not all(base[r, c] for c in cols):
                    continue
                if not all(base[r - 1, c] for c in cols):
                    continue
                if any(r + 1 < base_h and base[r + 1, c] for c in cols):
                    continue
                candidates.append((int(r), cols))
        if not candidates:
            continue
        conn_row, conn_cols = _rng.choice(candidates)

        keep_base = {(conn_row, c) for c in conn_cols} | {(conn_row - 1, c) for c in conn_cols}
        _thin(base, keep=keep_base)

        if not all(base[conn_row, c] for c in conn_cols):
            continue
        if not all(base[conn_row - 1, c] for c in conn_cols):
            continue
        if any(conn_row + 1 < base_h and base[conn_row + 1, c] for c in conn_cols):
            continue

        base_grid = np.full((base_h, base_w), bg, dtype=int)
        base_grid[base] = obj
        for c in conn_cols:
            base_grid[conn_row, c] = ctrl
        if is_contiguous(base_grid != bg, background=0, connectivity=8):
            break
        base_grid = None
    if base_grid is None:
        return None

    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, min(conn_cols[0], plug_w - conn_len))
    dc_offset = conn_cols[0] - plug_conn_c0

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        plug[1, c] = True

    allowed = np.ones((plug_h, plug_w), dtype=bool)
    for r in range(plug_h):
        for c in range(plug_w):
            dr = (conn_row - 1) + r
            dc = dc_offset + c
            is_connector_pixel = r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len
            if (
                0 <= dr < base_h
                and 0 <= dc < base_w
                and base_grid[dr, dc] == obj
                and not is_connector_pixel
            ):
                allowed[r, c] = False
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if not bool(allowed[0, c]) or not bool(allowed[1, c]):
            return None

    plug_max = min(29, plug_h * plug_w)
    allowed_count = int(np.sum(allowed))
    plug_cap = min(plug_max, allowed_count)
    if plug_cap < 5:
        return None
    plug_target = _rng.randint(5, plug_cap)
    if not _grow_shape_allowed(plug, plug_target, list(zip(*np.where(plug))), allowed):
        return None
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep_plug = {
        (0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)
    } | {
        (1, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)
    }
    _thin(plug, keep=keep_plug)

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if not bool(plug[1, c]):
            return None

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = (conn_row - 1) + r
            dc = dc_offset + c
            is_connector_pixel = r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len
            if (
                0 <= dr < base_h
                and 0 <= dc < base_w
                and base_grid[dr, dc] == obj
                and not is_connector_pixel
            ):
                return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl
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

    can_stack_vertical = ph + 1 + bh <= 11
    can_stack_horizontal = pw + 1 + bw <= 13
    if not (can_stack_vertical or can_stack_horizontal):
        return None

    if can_stack_vertical and can_stack_horizontal:
        mode = _rng.choice(["vertical", "horizontal"])
    elif can_stack_vertical:
        mode = "vertical"
    else:
        mode = "horizontal"

    if mode == "vertical":
        canvas_h = _rng.randint(max(7, ph + 1 + bh), 11)
        canvas_w = _rng.randint(max(3, max(pw, bw)), 13)
    else:
        canvas_h = _rng.randint(max(7, max(ph, bh)), 11)
        canvas_w = _rng.randint(max(3, pw + 1 + bw), 13)

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, plug_sprite, 0, 0, transparent=bg)
    try:
        br, bc = random_free_location(canvas, base_sprite, transparent=bg, padding=1)
    except ValueError:
        return None
    if br + bc == 0:
        return None
    paste(canvas, base_sprite, br, bc, transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != bg):
        return None
    out_h, out_w = map(int, result.shape)
    if not (5 <= out_h <= 8 and 3 <= out_w <= 12):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
