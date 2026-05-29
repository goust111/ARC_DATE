# Variant of 20270e3b
# Dimension: D7 — 输出裁剪方式 (grid_shape)
# Change: 对接合并后紧凑裁剪到非背景像素的最小外接矩形... → 仅裁剪到目标区域（base范围）
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: after docking, crop to the tight bbox of the merged base+plug foreground.
#     variant: after docking, keep only the base sprite's original bbox window; anything
#       from the shifted plug that lands outside that rectangle is discarded.
#   source_task:
#     requested: a251c730
#     note: the main repo path did not contain this solver, but an auxiliary local copy was
#       available and was inspected. The relevant takeaway was simple: render in a
#       destination-anchored frame directly, instead of rendering the full merged object and
#       tight-cropping afterward. The implementation below applies exactly that idea here.
#   concrete_behavior:
#     input: unchanged from the original task. There are exactly two 8-connected sprites
#       in colors obj=4 and ctrl=7 on bg=1. The base is the sprite anchored at the input
#       origin; the plug is translated so its horizontal ctrl bar lands one row above the
#       base ctrl bar with aligned columns.
#     output: still recolor all ctrl pixels to obj after docking, but the output grid size
#       is exactly the base sprite bbox size. The visible result is therefore a "window"
#       onto the assembled scene, anchored on the base. Plug pixels that extend below or
#       to the right of the base bbox are clipped away.
#     micro_trace:
#       - Suppose base tight bbox is 4x5 and the docked plug reaches rows 0..5, cols 1..6.
#       - Original logic renders a 6x7 merged canvas and tight-crops to that full extent.
#       - Variant logic renders only the 4x5 base window, stamps the same shifted plug with
#         clipping, and returns the 4x5 result. Rows 4..5 and col 6 from the plug vanish.
#   complexity_calibration:
#     - To make D7 observable, the generator must avoid samples where the docked plug fits
#       entirely inside the base bbox, because those would yield the same output as the
#       original task.
#     - The generator therefore requires at least one docked plug foreground pixel to lie
#       outside the base bbox after connector alignment.
#     - Branches are still simple: clipping can happen on the bottom edge, right edge, or
#       both. No extra colors, objects, or matching rules are introduced.
#   implicit_assumption_review:
#     - Component detection, connector matching, base selection, and docking translation are
#       unchanged and still rely on the original fixed-color / two-sprite setup.
#     - The original solver assumed that rendering on the merged bbox was safe, then
#       tight-cropped at the end. That assumption no longer holds: rendering must be done
#       directly in the base bbox frame or plug spillover would incorrectly enlarge output.
#     - The original overlap assertion remains valid, but now it is checked only for plug
#       cells that land inside the base frame; outside cells are intentionally discarded.
#     - The original "plug does not extend above or left of base" invariant is still kept,
#       because otherwise clipping on those sides could hide the anchoring relation and make
#       examples harder to read. The generator guarantees this.
#   placement_legality:
#     - Input placement remains non-overlapping on the source canvas.
#     - In the docked state used conceptually for output, plug overlap with base is allowed
#       only where ctrl/body meet, exactly as in the original task.
#     - The generator computes docked coordinates after sprite cropping and accepts an
#       example only if some plug pixels lie outside the base bbox while none lie above or
#       left of it.
#   visual_observability:
#     - Across examples, the solver still clearly "moves plug onto base", but the output
#       footprint now matches the base rectangle rather than the merged silhouette.
#     - Because the generator forces clipping in every accepted example, the user can infer
#       that the crop is anchored to base instead of to the full merged object.
#   cleanliness:
#     - No extra overlays or markers are added. The only visual difference is the clipped
#       extent of the docked plug, so scenes stay as clean as the original task.
#   color_reasoning:
#     - Colors are unchanged: bg=1, obj=4, ctrl=7 in input, only bg+obj in output. There is
#       no new color mapping burden beyond the original connector recoloring.
#   new_main_assumptions_checked_against_generator:
#     - main() assumes exactly two foreground components with non-empty, shape-matching
#       horizontal connectors.
#     - main() assumes the base bbox top-left is at the input origin and the shifted plug
#       never extends above or left of that bbox.
#     - main() assumes any in-frame overlap is connector-related.
#     - generate_input() enforces all of the above and additionally enforces visible
#       clipping, so the variant does not collapse back to original behavior.

from __future__ import annotations

import random
import numpy as np
from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _docked_clip_stats(base_sprite: np.ndarray, plug_sprite: np.ndarray, bg: int, ctrl: int) -> dict[str, int]:
    base = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    plug = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}
    connD = {(int(r), int(c)) for r, c in np.argwhere(base_sprite == int(ctrl))}
    connP = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite == int(ctrl))}
    _sd, (dr0, dc0) = _norm_conn(connD)
    _sp, (pr0, pc0) = _norm_conn(connP)
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
    bh, bw = map(int, base_sprite.shape)

    outside = 0
    right = 0
    bottom = 0
    above = 0
    left = 0
    for r, c in plug_shift:
        if int(r) < 0:
            above += 1
        if int(c) < 0:
            left += 1
        if int(r) >= int(bh) or int(c) >= int(bw):
            outside += 1
            if int(r) >= int(bh):
                bottom += 1
            if int(c) >= int(bw):
                right += 1
    return {
        "outside": int(outside),
        "right": int(right),
        "bottom": int(bottom),
        "above": int(above),
        "left": int(left),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then crop to base bbox only.

    Input structure:
    - Background color `bg`, sprite color `obj`, connector color `ctrl`.
    - Exactly two 8-connected sprites; one is anchored at the input origin.

    Rule:
    - Identify the base sprite (closest to top-left, in practice anchored at (0,0)).
    - Translate the other sprite so its connector sits one row above the base connector.
    - Stamp the plug onto the base; recolor connectors from `ctrl` to `obj`.
    - Return only the base sprite's original bbox window, clipping any plug spillover.
    """
    g = np.array(grid, dtype=int)

    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
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

    shpA, _anchorA = _norm_conn(connA)
    shpB, _anchorB = _norm_conn(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
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

    _sp, (pr0, pc0) = _norm_conn(connP)
    _sd, (dr0, dc0) = _norm_conn(connD)
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    br0, bc0, br1, bc1 = _bbox_of_pts(base)
    assert int(br0) == 0 and int(bc0) == 0
    if plug_shift:
        assert min(r for r, _c in plug_shift) >= int(br0)
        assert min(c for _r, c in plug_shift) >= int(bc0)

    out = np.full((int(br1 - br0 + 1), int(bc1 - bc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - br0), int(c - bc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        if not (int(br0) <= int(r) <= int(br1) and int(bc0) <= int(c) <= int(bc1)):
            continue
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - br0), int(c - bc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - br0), int(c - bc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    return out


# ---------------------------------------------------------------------------
# Generator (anchor-first: build base, grow plug on it, then separate)
# ---------------------------------------------------------------------------
import random as _rng


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


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    """Remove interior pixels to make a thin-walled shape."""
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


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

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
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base] = int(obj)
    for c in conn_cols:
        base_grid[conn_row, c] = int(ctrl)
    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None

    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = int(conn_cols[0] - plug_conn_c0)

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
            dr = int(conn_row + r)
            dc = int(dc_offset + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and int(base_grid[dr, dc]) == int(obj):
                if r == 0 and c in range(plug_conn_c0, plug_conn_c0 + conn_len):
                    continue
                return None

    if int(dc_offset) < 0:
        return None

    plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    plug_grid[plug] = int(obj)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = int(ctrl)

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if plug_h < 2 or int(plug_grid[1, c]) != int(obj):
            return None
    if not is_contiguous(plug_grid != int(bg), background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != int(bg))) < 5 or int(np.sum(plug_grid != int(bg))) < 5:
        return None

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprite = _crop_non_bg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None

    stats = _docked_clip_stats(base_sprite, plug_sprite, bg, ctrl)
    if int(stats["above"]) != 0 or int(stats["left"]) != 0:
        return None
    if int(stats["outside"]) == 0:
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(3, max(bw, pw)), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(bg))
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=int(bg), padding=0)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=int(bg))

    if int(np.sum(canvas == int(bg))) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None

    if result.size == 0 or not np.any(result != int(bg)):
        return None
    if result.shape != base_sprite.shape:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
