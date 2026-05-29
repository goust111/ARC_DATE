# Variant of 20270e3b
# Dimension: D8 — 精灵数量与配对基数 (multi_step)
# Change: 恰好2个精灵，1:1配对对接... → 三精灵顺序对接

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

# Design analysis for the D8 variant
# ----------------------------------
# a) Concrete behavior
#    - Input now contains exactly 3 scattered 8-connected sprites, all using fixed colors:
#      bg=1, body=4, connector=7.
#    - Sprite A is still the anchor at the grid origin. It has exactly one horizontal
#      connector whose body side is above and free side is below.
#    - Sprite B is the new middle piece. To make the chain physically possible without
#      changing the docking semantics, B carries two horizontal connectors of the same
#      length: one at its top edge facing up, and one at its bottom edge facing down.
#    - Sprite C is the terminal plug. It has exactly one horizontal connector at its top
#      edge facing up.
#    - Output is formed by two sequential docking steps:
#         1. dock B onto A using B's top connector and A's downward-facing connector
#         2. dock C onto shifted B using C's top connector and B's downward-facing connector
#      All connector pixels are recolored from 7 to 4 in the final tight crop.
#    - Small trace:
#         original task: A + B -> one seam -> crop
#         variant:       A + B -> intermediate AB, then AB + C -> longer chain -> crop
#      The second seam depends on the first shift because C aligns to B's lower connector,
#      not directly to A. That is the D8 multi-step change.
# b) Complexity calibration
#    - The new visual cue is explicit but not noisy: only one sprite has two connector bars,
#      so the middle role is visible from examples.
#    - All connectors still share one length per instance, preserving the original matching
#      mechanism and avoiding an unnecessary extra matching dimension.
#    - B's upper and lower connector columns may differ, so the chain can jog horizontally.
#      This makes the second step visible without exploding the branching factor.
# c) Implicit-assumption audit of the original solver
#    - `assert len(best_comps) == 2`: no longer valid; now exactly 3 components are required.
#    - `connA/connB` as one set each: no longer valid because B has 2 connector components.
#    - `assert shpA == shpB`: still valid after being lifted to "all connector bars share one
#      normalized horizontal shape".
#    - base/plug binary role split: replaced by ordered roles A(start), B(middle), C(end).
#    - Single translation (`tr`,`tc`) is insufficient: the solver must compute B's shift,
#      then compute C's shift relative to shifted B.
#    - Single overlap check while stamping plug is insufficient: stamping must allow two seam
#      overlaps, each obeying the same "one side must be ctrl" rule.
# d) Placement legality requirements for the generator
#    - A is pasted at (0,0).
#    - B and C are scattered elsewhere with 8-neighborhood padding so the 3 sprites remain
#      separate connected components in the input.
#    - In the assembled state, B and C are constrained not to extend above or left of A,
#      preserving the original "anchor stays at output top-left after crop" invariant.
#    - Overlap is allowed only inside the two connector seams, where an obj strip overlaps
#      a ctrl strip.
# e) Visual observability
#    - Humans can infer the chain from 2-3 examples because:
#         * one piece clearly has two connector bars and must be in the middle
#         * the third piece always attaches relative to that middle piece's lower connector
#         * the final output is longer than a single A+B merge
#    - No invisible property such as color id ordering is used for sorting; roles are decided
#      only from visible placement (origin anchor) and connector orientation/count.
# f) Visual cleanliness
#    - Colors remain exactly the same as the source task.
#    - Only 3 sprites and at most 4 connector bars are shown, so the input is still sparse.
#    - The final chain uses only pairwise seam overlaps; no triple-stack regions are allowed.
# g) Color reasoning
#    - There is no new color mapping. The only color change remains the original one:
#      connector color 7 disappears into body color 4 after docking.
# Main/generator contract review
#    - `main()` assumes:
#         * exactly 3 foreground components
#         * exactly one origin-anchored sprite with one down connector
#         * exactly one 2-connector sprite with orientations {up, down}
#         * exactly one terminal sprite with one up connector
#         * all connector bars have the same horizontal shape
#         * chain assembly never extends above/left of A and only overlaps at seams
#    - `generate_input()` explicitly constructs inputs satisfying each assumption.


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int), 0, 0
    r0, c0, r1, c1 = bbox_coords(nz)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), int(r0), int(c0)


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _ in conn_pts]
    cs = [c for _, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in conn_pts)
    return shape, (int(r0), int(c0))


def _connector_components(
    g: np.ndarray, sprite: set[tuple[int, int]], ctrl: int
) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox_of_pts(sprite)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in sprite:
        if int(g[r, c]) == int(ctrl):
            mask[int(r - r0), int(c - c0)] = True
    lab, n = label_mask(mask, connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {
            (int(r0 + rr), int(c0 + cc))
            for rr, cc in np.argwhere(lab == int(k))
        }
        comps.append(pts)
    return comps


def _classify_connector(
    sprite: set[tuple[int, int]], conn_pts: set[tuple[int, int]]
) -> dict[str, object]:
    rows = sorted({int(r) for r, _ in conn_pts})
    cols = sorted(int(c) for _, c in conn_pts)
    assert len(rows) == 1
    assert cols == list(range(cols[0], cols[0] + len(cols)))
    body = sprite - conn_pts
    row = int(rows[0])

    up = all((int(row + 1), int(c)) in body for c in cols) and all(
        (int(row - 1), int(c)) not in body for c in cols
    )
    down = all((int(row - 1), int(c)) in body for c in cols) and all(
        (int(row + 1), int(c)) not in body for c in cols
    )
    assert up ^ down
    shape, (r0, c0) = _norm_conn(conn_pts)
    return {
        "pts": conn_pts,
        "shape": shape,
        "row": int(row),
        "c0": int(c0),
        "len": int(len(cols)),
        "orientation": "up" if up else "down",
    }


def _sprite_desc(g: np.ndarray, sprite: set[tuple[int, int]], ctrl: int) -> dict[str, object]:
    connectors = [_classify_connector(sprite, comp) for comp in _connector_components(g, sprite, ctrl)]
    r0, c0, r1, c1 = _bbox_of_pts(sprite)
    return {
        "pts": sprite,
        "connectors": connectors,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "top_left_sum": int(r0 + c0),
    }


def _stamp_shifted(
    out: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    *,
    bg: int,
    ctrl: int,
) -> None:
    h, w = patch.shape
    for r in range(int(h)):
        for c in range(int(w)):
            val = int(patch[r, c])
            if val == int(bg):
                continue
            rr, cc = int(top + r), int(left + c)
            cur = int(out[rr, cc])
            if cur != int(bg):
                assert cur == int(ctrl) or val == int(ctrl)
            out[rr, cc] = int(val)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 3

    sprites = [_sprite_desc(g, pts, ctrl) for pts in comps]
    all_shapes = {desc["shape"] for spr in sprites for desc in spr["connectors"]}
    assert len(all_shapes) == 1

    origin_sprites = [spr for spr in sprites if int(spr["top_left_sum"]) == 0]
    assert len(origin_sprites) == 1
    base = origin_sprites[0]
    others = [spr for spr in sprites if spr is not base]

    base_down = [c for c in base["connectors"] if c["orientation"] == "down"]
    assert len(base["connectors"]) == 1 and len(base_down) == 1

    middle_candidates = []
    end_candidates = []
    for spr in others:
        orientations = sorted(c["orientation"] for c in spr["connectors"])
        if orientations == ["down", "up"]:
            middle_candidates.append(spr)
        elif orientations == ["up"]:
            end_candidates.append(spr)
    assert len(middle_candidates) == 1 and len(end_candidates) == 1
    middle = middle_candidates[0]
    end = end_candidates[0]

    mid_up = [c for c in middle["connectors"] if c["orientation"] == "up"]
    mid_down = [c for c in middle["connectors"] if c["orientation"] == "down"]
    end_up = [c for c in end["connectors"] if c["orientation"] == "up"]
    assert len(mid_up) == 1 and len(mid_down) == 1 and len(end_up) == 1
    base_down = base_down[0]
    mid_up = mid_up[0]
    mid_down = mid_down[0]
    end_up = end_up[0]

    tr_mid = int((int(base_down["row"]) - 1) - int(mid_up["row"]))
    tc_mid = int(int(base_down["c0"]) - int(mid_up["c0"]))
    tr_end = int((int(mid_down["row"]) + tr_mid - 1) - int(end_up["row"]))
    tc_end = int((int(mid_down["c0"]) + tc_mid) - int(end_up["c0"]))

    shifted_mid = {(int(r + tr_mid), int(c + tc_mid)) for r, c in middle["pts"]}
    shifted_end = {(int(r + tr_end), int(c + tc_end)) for r, c in end["pts"]}

    pts_all = set(base["pts"]) | shifted_mid | shifted_end
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))

    base_coords = np.array(list(base["pts"]), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    base_r0, base_c0, base_r1, base_c1 = map(int, base["bbox"])
    mid_r0, mid_c0, mid_r1, mid_c1 = map(int, middle["bbox"])
    end_r0, end_c0, end_r1, end_c1 = map(int, end["bbox"])

    base_patch = g[base_r0 : base_r1 + 1, base_c0 : base_c1 + 1]
    mid_patch = g[mid_r0 : mid_r1 + 1, mid_c0 : mid_c1 + 1]
    end_patch = g[end_r0 : end_r1 + 1, end_c0 : end_c1 + 1]

    _stamp_shifted(out, base_patch, base_r0 - rr0, base_c0 - cc0, bg=bg, ctrl=ctrl)
    _stamp_shifted(out, mid_patch, mid_r0 + tr_mid - rr0, mid_c0 + tc_mid - cc0, bg=bg, ctrl=ctrl)
    _stamp_shifted(out, end_patch, end_r0 + tr_end - rr0, end_c0 + tc_end - cc0, bg=bg, ctrl=ctrl)

    out[out == int(ctrl)] = int(obj)
    return out


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
        frontier.append((int(nr), int(nc)))
        count += 1


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    h, w = mask.shape
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


def _down_connector_candidates(mask: np.ndarray, conn_len: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for r in range(1, h):
        for c0 in range(0, w - conn_len + 1):
            cols = range(c0, c0 + conn_len)
            if not all(mask[r, c] for c in cols):
                continue
            if not all(mask[r - 1, c] for c in cols):
                continue
            if any(r + 1 < h and mask[r + 1, c] for c in cols):
                continue
            out.append((int(r), int(c0)))
    return out


def _render_chain(
    base_grid: np.ndarray,
    mid_grid: np.ndarray,
    end_grid: np.ndarray,
    *,
    mid_shift: tuple[int, int],
    end_shift: tuple[int, int],
    bg: int,
    ctrl: int,
) -> np.ndarray | None:
    mh, mw = mid_grid.shape
    eh, ew = end_grid.shape
    bh, bw = base_grid.shape
    tr_mid, tc_mid = map(int, mid_shift)
    tr_end, tc_end = map(int, end_shift)
    if tr_mid < 0 or tc_mid < 0 or tr_end < 0 or tc_end < 0:
        return None

    H = max(int(bh), int(tr_mid + mh), int(tr_end + eh))
    W = max(int(bw), int(tc_mid + mw), int(tc_end + ew))
    canvas = np.full((H, W), int(bg), dtype=int)
    try:
        _stamp_shifted(canvas, base_grid, 0, 0, bg=bg, ctrl=ctrl)
        _stamp_shifted(canvas, mid_grid, tr_mid, tc_mid, bg=bg, ctrl=ctrl)
        _stamp_shifted(canvas, end_grid, tr_end, tc_end, bg=bg, ctrl=ctrl)
    except AssertionError:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7
    conn_len = _rng.randint(1, 3)

    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(max(3, conn_len), 8)
    base_low = max(6, conn_len + 3)
    base_high = min(24, base_h * base_w)
    if base_low > base_high:
        return None
    base_target = _rng.randint(base_low, base_high)
    base_mask = np.zeros((base_h, base_w), dtype=bool)
    base_mask[0, 0] = True
    _grow_shape(base_mask, base_target, [(0, 0)])
    if not is_contiguous(base_mask, background=0, connectivity=8):
        return None
    _thin(base_mask)

    base_candidates = _down_connector_candidates(base_mask, conn_len)
    if not base_candidates:
        return None
    base_conn_row, base_conn_c0 = _rng.choice(base_candidates)

    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base_mask] = int(obj)
    base_grid[base_conn_row, base_conn_c0 : base_conn_c0 + conn_len] = int(ctrl)
    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None

    mid_h = _rng.randint(4, 8)
    mid_w = _rng.randint(max(3, conn_len), 8)
    mid_top_c0 = _rng.randint(0, mid_w - conn_len)
    mid_bot_c0 = _rng.randint(0, mid_w - conn_len)

    mid_mask = np.zeros((mid_h, mid_w), dtype=bool)
    mid_mask[0, mid_top_c0 : mid_top_c0 + conn_len] = True
    mid_mask[1, mid_top_c0 : mid_top_c0 + conn_len] = True
    mid_mask[mid_h - 1, mid_bot_c0 : mid_bot_c0 + conn_len] = True
    mid_mask[mid_h - 2, mid_bot_c0 : mid_bot_c0 + conn_len] = True
    top_center = int(mid_top_c0 + conn_len // 2)
    bot_center = int(mid_bot_c0 + conn_len // 2)
    for r in range(1, mid_h - 1):
        mid_mask[r, top_center] = True
    for c in range(min(top_center, bot_center), max(top_center, bot_center) + 1):
        mid_mask[mid_h - 2, c] = True

    mid_seed_count = int(mid_mask.sum())
    mid_low = max(mid_seed_count + 1, 8)
    mid_high = min(24, mid_h * mid_w)
    if mid_low > mid_high:
        return None
    mid_target = _rng.randint(mid_low, mid_high)
    _grow_shape(mid_mask, mid_target, list(zip(*np.where(mid_mask))))
    if not is_contiguous(mid_mask, background=0, connectivity=8):
        return None
    keep_mid = {
        (0, c) for c in range(mid_top_c0, mid_top_c0 + conn_len)
    } | {
        (1, c) for c in range(mid_top_c0, mid_top_c0 + conn_len)
    } | {
        (mid_h - 1, c) for c in range(mid_bot_c0, mid_bot_c0 + conn_len)
    } | {
        (mid_h - 2, c) for c in range(mid_bot_c0, mid_bot_c0 + conn_len)
    }
    _thin(mid_mask, keep=keep_mid)

    mid_grid = np.full((mid_h, mid_w), int(bg), dtype=int)
    mid_grid[mid_mask] = int(obj)
    mid_grid[0, mid_top_c0 : mid_top_c0 + conn_len] = int(ctrl)
    mid_grid[mid_h - 1, mid_bot_c0 : mid_bot_c0 + conn_len] = int(ctrl)
    if not is_contiguous(mid_grid != int(bg), background=0, connectivity=8):
        return None
    mid_grid, _mr0, mc0 = _crop_non_bg(mid_grid, bg)
    mid_top_c0 -= int(mc0)
    mid_bot_c0 -= int(mc0)
    mid_h, mid_w = mid_grid.shape

    end_h = _rng.randint(3, 7)
    end_w = _rng.randint(max(2, conn_len), 8)
    end_conn_c0 = _rng.randint(0, end_w - conn_len)
    end_mask = np.zeros((end_h, end_w), dtype=bool)
    end_mask[0, end_conn_c0 : end_conn_c0 + conn_len] = True
    end_mask[1, end_conn_c0 : end_conn_c0 + conn_len] = True
    end_mask[2, end_conn_c0 : end_conn_c0 + conn_len] = True
    end_seed_count = int(end_mask.sum())
    end_low = max(end_seed_count + 1, 7)
    end_high = min(22, end_h * end_w)
    if end_low > end_high:
        return None
    end_target = _rng.randint(end_low, end_high)
    _grow_shape(end_mask, end_target, list(zip(*np.where(end_mask))))
    if not is_contiguous(end_mask, background=0, connectivity=8):
        return None
    keep_end = {
        (r, c)
        for r in (0, 1, 2)
        for c in range(end_conn_c0, end_conn_c0 + conn_len)
    }
    _thin(end_mask, keep=keep_end)

    end_grid = np.full((end_h, end_w), int(bg), dtype=int)
    end_grid[end_mask] = int(obj)
    end_grid[0, end_conn_c0 : end_conn_c0 + conn_len] = int(ctrl)
    if not is_contiguous(end_grid != int(bg), background=0, connectivity=8):
        return None
    end_grid, _er0, ec0 = _crop_non_bg(end_grid, bg)
    end_conn_c0 -= int(ec0)
    end_h, end_w = end_grid.shape

    mid_shift = (int(base_conn_row - 1), int(base_conn_c0 - mid_top_c0))
    end_shift = (
        int(mid_shift[0] + mid_h - 2),
        int(mid_shift[1] + mid_bot_c0 - end_conn_c0),
    )
    chain = _render_chain(
        base_grid,
        mid_grid,
        end_grid,
        mid_shift=mid_shift,
        end_shift=end_shift,
        bg=bg,
        ctrl=ctrl,
    )
    if chain is None:
        return None

    chain_out = chain.copy()
    chain_out[chain_out == int(ctrl)] = int(obj)
    chain_out, _r0, _c0 = _crop_non_bg(chain_out, bg)
    if chain_out.size == 0:
        return None
    if chain_out.shape[0] > 12 or chain_out.shape[1] > 14:
        return None

    if int(np.sum(base_grid != int(bg))) < 6 or int(np.sum(mid_grid != int(bg))) < 8:
        return None
    if int(np.sum(end_grid != int(bg))) < 6:
        return None

    canvas_h_min = max(10, max(base_grid.shape[0], mid_grid.shape[0], end_grid.shape[0]) + 2)
    canvas_w_min = max(11, base_grid.shape[1] + mid_grid.shape[1] + end_grid.shape[1] + 1)
    canvas_h = _rng.randint(canvas_h_min, max(canvas_h_min, 15))
    canvas_w = _rng.randint(canvas_w_min, max(canvas_w_min, 18))
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_grid, 0, 0, transparent=bg)
    try:
        mr, mc = random_free_location(canvas, mid_grid, transparent=bg, padding=1)
        paste(canvas, mid_grid, mr, mc, transparent=bg)
        er, ec = random_free_location(canvas, end_grid, transparent=bg, padding=1)
        paste(canvas, end_grid, er, ec, transparent=bg)
    except ValueError:
        return None

    if (int(mr) + int(mc)) == 0 or (int(er) + int(ec)) == 0:
        return None
    if np.sum(canvas == int(bg)) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, chain_out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
