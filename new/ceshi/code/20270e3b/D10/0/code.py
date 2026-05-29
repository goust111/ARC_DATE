# Variant of 20270e3b
# Dimension: D10 — 颜色体系 (meta)
# Change: 固定三色体系：bg=1（蓝），obj=4（黄），ctrl=7（橙）... → 每个精灵不同body色

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


"""
Design analysis for the D10 variant
==================================

spec:
  unchanged_geometry:
    - bg stays fixed at 1 and the connector color stays fixed at 7.
    - input still contains exactly 2 foreground sprites, 8-connected by foreground pixels.
    - each sprite still has exactly one horizontal connector bar of length 1-3.
    - base is still the sprite whose bbox top-left is at the grid origin.
    - plug is translated so its connector row lands exactly one row above the base connector.
    - tight cropping, overlap legality, and all placement constraints stay unchanged.
  D10_change:
    - each sprite now has its own single body color.
    - the two body colors are distinct.
    - after docking, body pixels keep their original sprite color.
    - any surviving connector pixels are recolored to the base sprite's body color.
  concrete_trace:
    - original:
      input might use base body 4, plug body 4, ctrl 7.
      after stamping, remaining 7 cells are blindly recolored to 4.
    - variant:
      input might use base body 3, plug body 4, ctrl 7.
      after stamping, the plug body still overwrites the base connector row at the seam,
      but the surviving upper connector row is recolored to 3, not to a global fixed 4.
      The output therefore visibly contains both 3 and 4, and the former connector row
      matches the base sprite rather than a universal object color.
    - difference:
      the solver can no longer rely on a single global object color and must infer the
      base/plug body colors per component before recoloring the connector.
  complexity_calibration:
    - Keep only 4 colors per input at most: bg, ctrl, base_body, plug_body.
    - The connector color remains unique and shared, so docking is still visually obvious.
    - The base color is forced away from 4 in generation so the variant does not collapse
      back to the original "recolor connector to 4" behavior.
  implicit_assumption_audit:
    - `label_mask(g != bg)` remains valid because both sprite body colors and ctrl are
      still foreground.
    - `conn(sprite)` may still use the fixed ctrl color 7.
    - The old `out[out == ctrl] = obj` is invalid because `obj` is no longer global.
      It must become `out[out == ctrl] = base_color`.
    - Overlap checks must remain color-agnostic except for the connector exception:
      overlap is only legal when one side is ctrl, regardless of body colors.
    - Each sprite must contain exactly one non-bg/non-ctrl color; generator guarantees this.
  placement_legality:
    - Generator still constructs the docked arrangement first, then separates sprites.
    - It forbids plug/body overlap with base body outside the connector seam.
    - Plug is still cropped and placed away from the origin so the solver sees two components.
  visual_observability:
    - Human-solvable cue: two differently colored sprites dock in the same way as the
      original task, but the final merged object keeps both colors and the connector stripe
      resolves to the base color.
    - Because the base color is never 4 in generated examples, the recoloring rule is
      visibly different from the original fixed-yellow output behavior.
  main_new_assumptions_checked_against_generator:
    - each component has one connector and one body color -> enforced when building sprites.
    - the two body colors differ -> enforced by color sampling.
    - base connector free side is downward and plug connector free side is upward ->
      enforced by connector placement and by growing the plug body below its connector.
"""


BG = 1
CTRL = 7
BASE_BODY_PALETTE = (2, 3, 5, 6, 8, 9)
PLUG_BODY_PALETTE = (2, 3, 4, 5, 6, 8, 9)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then tight-crop.

    Variant rule:
    - Two sprites may have different body colors.
    - The merged output keeps both body colors.
    - Remaining connector pixels are recolored to the base sprite's body color.
    """
    g = np.array(grid, dtype=int)
    bg, ctrl = BG, CTRL

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

    def conn(pts: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for r, c in pts if int(g[int(r), int(c)]) == int(ctrl)}

    def body_color(pts: set[tuple[int, int]]) -> int:
        cols = {
            int(g[int(r), int(c)])
            for r, c in pts
            if int(g[int(r), int(c)]) not in (int(bg), int(ctrl))
        }
        assert len(cols) == 1
        return int(next(iter(cols)))

    def norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
        rs = [r for r, _c in conn_pts]
        cs = [c for _r, c in conn_pts]
        r0, c0 = int(min(rs)), int(min(cs))
        shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
        return shape, (int(r0), int(c0))

    connA = conn(A)
    connB = conn(B)
    assert connA and connB

    shpA, _ancA = norm(connA)
    shpB, _ancB = norm(connB)
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

    base_color = body_color(base)
    plug_color = body_color(plug)
    assert base_color != plug_color

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

    out[out == int(ctrl)] = int(base_color)

    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    """Grow a connected shape on mask from frontier up to target pixels."""
    count = int(mask.sum())
    while count < target and frontier:
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
    """Remove interior pixels to make a thin-walled shape."""
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


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_attempt() -> np.ndarray | None:
    bg, ctrl = BG, CTRL
    base_color = int(_rng.choice(BASE_BODY_PALETTE))
    plug_choices = [c for c in PLUG_BODY_PALETTE if int(c) != int(base_color)]
    plug_color = int(_rng.choice(plug_choices))

    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates = []
    for r in range(base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if r == 0:
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            candidates.append((r, cols))
    if not candidates:
        return None
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = int(base_color)
    for c in conn_cols:
        base_grid[conn_row, c] = int(ctrl)
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = conn_cols[0] - plug_conn_c0

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        if plug_h > 1:
            plug[1, c] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = conn_row + r
            dc = dc_offset + c
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == int(base_color):
                if r == 0 and c in range(plug_conn_c0, plug_conn_c0 + conn_len):
                    continue
                return None

    if dc_offset < 0:
        return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = int(plug_color)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = int(ctrl)
        if plug_h < 2 or plug_grid[1, c] != int(plug_color):
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
    canvas_w = _rng.randint(max(3, max(bw, pw)), 13)
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

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != bg):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
