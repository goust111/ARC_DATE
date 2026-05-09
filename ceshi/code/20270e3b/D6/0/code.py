# Variant of 20270e3b
# Dimension: D6 -- base角色判定规则 (meta)
# Change: 外接矩形左上角满足 r+c==0（即紧贴网格原点）的精灵为base... → 面积较大者为base

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete behavior trace
#    Input still contains exactly two 8-connected sprites made from obj=4 and ctrl=7.
#    Each sprite has one horizontal connector bar (1-3 ctrl cells), and the docking
#    transform is unchanged: translate the plug so its connector sits one row above
#    the base connector, then recolor all ctrl to obj and tight-crop.
#
#    The changed step is only role assignment.
#    Original: whichever sprite's bbox top-left is at the grid origin is base.
#    Variant: whichever sprite has strictly more foreground pixels is base.
#
#    Small trace:
#    - Sprite S0: 8 pixels, placed at input origin.
#    - Sprite S1: 14 pixels, placed elsewhere.
#    - Old rule: S0 becomes base only because it touches (0,0).
#    - New rule: S1 becomes base because 14 > 8.
#    - Resulting output differs because the moved object is now S0 instead of S1.
#
# b) Complexity calibration
#    Counting pixels is still a simple ARC-visible operation, but it is meaningfully
#    different from reading position. To avoid ambiguity, the generator enforces a
#    strict area gap of at least 3 pixels. Connector shape, colors, and docking
#    geometry stay fixed so the only new reasoning burden is "count and compare".
#
# c) Old-code assumptions that must change
#    - The original solver asserted that one bbox top-left satisfies r+c==0.
#      This is no longer a semantic requirement for base selection, so that assert
#      must be removed.
#    - The original solver selected base by bbox proximity to (0,0). That must be
#      replaced with a strict foreground-area comparison.
#    - The original generator always pasted base at (0,0). That would let many
#      examples accidentally still obey the old rule, so the generator must instead
#      control which sprite is at the origin independently from which sprite is larger.
#
# d) Placement legality
#    The generator still builds a valid docked pair first, crops both sprites, then
#    separates them on a larger canvas. Exactly one sprite is pasted at the origin.
#    The other is placed with 8-neighbor padding=1 so the two sprites remain two
#    distinct connected components and never touch diagonally.
#
# e) Visual observability
#    Most generated inputs put the smaller sprite at the origin and the larger sprite
#    elsewhere. A human comparing 2-4 examples can see that "touching the origin"
#    does not determine the stationary piece; the larger sprite does.
#
# f) Visual cleanliness
#    Only two colors are used on foreground, connectors remain short horizontal bars,
#    and sprites are separated before docking. No extra markers or overlapping layers
#    are introduced beyond the original connector seam.
#
# g) Color inference
#    Colors are unchanged from the original task: bg=1, obj=4, ctrl=7. The variant
#    does not add a new color mapping, so there is no hidden color-dependent rule.
#
# Post-change solver assumptions and generator guarantees
#    - There are exactly two foreground components.
#    - Each component has at least one ctrl pixel and both connector bars normalize
#      to the same horizontal shape.
#    - One sprite has strictly larger foreground area than the other.
#    - In the intended docked state, the plug does not extend above or left of base.
#    The generator explicitly enforces all four conditions.


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn_pts(g: np.ndarray, sprite: set[tuple[int, int]], ctrl: int) -> set[tuple[int, int]]:
    return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _ in conn_pts]
    cs = [c for _, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then tight-crop.

    Variant D6:
    - base = sprite with strictly larger foreground area
    - origin contact is no longer used for role assignment
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
    assert len(A) != len(B)

    connA = _conn_pts(g, A, ctrl)
    connB = _conn_pts(g, B, ctrl)
    assert connA and connB

    shpA, _ = _norm_conn(connA)
    shpB, _ = _norm_conn(connB)
    assert shpA == shpB

    if len(A) > len(B):
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

    _sp, (pr0, pc0) = _norm_conn(connP)
    _sd, (dr0, dc0) = _norm_conn(connD)

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
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


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(3, 10)
    base_cap = min(29, base_h * base_w)
    if base_cap < 9:
        return None
    base_target = _rng.randint(9, base_cap)
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
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[conn_row, c] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    base_count = int(np.sum(base_grid != bg))
    if base_count < 9:
        return None

    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = conn_cols[0] - plug_conn_c0
    if dc_offset < 0:
        return None

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        if plug_h > 1:
            plug[1, c] = True

    plug_cap = min(29, plug_h * plug_w, base_count - 3)
    if plug_cap < 5:
        return None
    plug_target = _rng.randint(5, plug_cap)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if plug_h < 2 or plug_grid[1, c] != obj:
            return None
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = conn_row + r
            dc = dc_offset + c
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == obj:
                if r == 0 and c in range(plug_conn_c0, plug_conn_c0 + conn_len):
                    continue
                return None

    plug_count = int(np.sum(plug_grid != bg))
    if plug_count < 5 or base_count - plug_count < 3:
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

    origin_holds_base = _rng.random() < 0.25
    if origin_holds_base:
        paste(canvas, base_sprite, 0, 0, transparent=bg)
        try:
            pr, pc = random_free_location(
                canvas, plug_sprite, transparent=bg, padding=1, padding_connectivity=8
            )
        except ValueError:
            return None
        paste(canvas, plug_sprite, pr, pc, transparent=bg)
    else:
        paste(canvas, plug_sprite, 0, 0, transparent=bg)
        try:
            br, bc = random_free_location(
                canvas, base_sprite, transparent=bg, padding=1, padding_connectivity=8
            )
        except ValueError:
            return None
        paste(canvas, base_sprite, br, bc, transparent=bg)

    lab, n = label_mask(canvas != bg, connectivity=8)
    if int(n) != 2:
        return None

    origin_component_sizes = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        r0, c0, _r1, _c1 = bbox_coords(pts)
        origin_component_sizes.append((int(r0 + c0), int(pts.shape[0])))
    if sum(1 for anchor_sum, _area in origin_component_sizes if anchor_sum == 0) != 1:
        return None

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
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
