# Variant of 20270e3b
# Dimension: D1 — 连接器匹配条件 (match_cond)
# Change: 两个精灵的连接器具有相同的归一化形状（相同长度的ctrl色水平条），按形状/颜色匹配... → 连接器形状嵌合匹配

from __future__ import annotations

from collections import deque
import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   kept_from_original:
#     - fixed colors stay bg=1, obj=4, ctrl=7
#     - exactly 2 foreground sprites
#     - base is still the sprite whose bbox touches the grid origin
#     - output is still "dock plug onto base, recolor ctrl->obj, tight-crop"
#   changed_core:
#     - connector is no longer a 1-row horizontal bar
#     - each connector is now a 2 x w zigzag band (w in 3..5)
#     - in every connector column there is exactly one ctrl pixel, either on
#       the upper row or the lower row of that 2-row band
#     - base and plug must use complementary row choices column-by-column
#       (profile bits sum to 1), so together they tile the full 2 x w band
#   concrete_pixel_behavior:
#     - base body lives above the zigzag seam; plug body lives below it
#     - when docked, the two connector bboxes share the same top-left corner
#       instead of the old "plug row = base row - 1" bar alignment
#     - some seam cells are ctrl-vs-obj overlaps, some are ctrl-vs-bg; after
#       recoloring ctrl->obj, the seam becomes a solid interlocked band
#   behavior_trace:
#     - old logic:
#         1. normalize both ctrl sets
#         2. assert the shapes are equal
#         3. shift plug by (base_conn_row - 1 - plug_conn_row, delta_col)
#     - new logic:
#         1. normalize both ctrl sets into a 2 x w bbox
#         2. decode a per-column row profile, e.g. [0,1,1,0]
#         3. assert plug_profile == [1-b for b in base_profile]
#         4. shift plug by (base_conn_top - plug_conn_top, delta_col) so the
#            entire zigzag bands align and interlock
#     - key difference:
#         matching uses complementarity over a 2-row seam, not equality of a
#         1-row bar; the vertical translation formula therefore changes too
#   complexity_calibration:
#     - widths are limited to 3..5 so humans can see the seam shape quickly
#     - profiles are forced to contain both row levels, so samples never
#       collapse back into the original flat-bar behavior
#     - there are still only two sprites, so D1 stays the main change rather
#       than turning into a multi-object search puzzle
#   implicit_assumption_audit_against_old_main:
#     - old norm(conn_pts) + assert shpA == shpB no longer holds; replaced by
#       bbox-height-2 decoding and complement-profile checking
#     - old tr = (dr0 - 1) - pr0 relied on single-row bars separated by one
#       row; this is wrong for a shared 2-row seam and is replaced by tr=dr0-pr0
#     - old overlap assertion "at least one side is ctrl" remains safe because
#       generator only permits overlaps inside the seam and only as ctrl-vs-obj
#     - old free-side checks per ctrl pixel remain valid: base cannot have obj
#       directly below any ctrl pixel; plug cannot have obj directly above any
#       ctrl pixel
#   placement_legality:
#     - generator builds base first, then builds plug against the docked base
#       and rejects any obj-obj overlap in the assembled state
#     - plug connector is placed with local top row 0; base connector top row
#       is >=1, so docked plug never extends above the base
#     - plug connector left offset is chosen so docking never extends left of
#       the base; after tight crop the base stays at output top-left
#     - scattered input placement uses random_free_location(..., padding=1) so
#       the two sprites remain separate under 8-connectivity
#   visual_observability:
#     - every sample shows a visibly jagged seam because both row levels appear
#       in each connector profile
#     - the output solidifies exactly the same seam region, making the new rule
#       inferable from a few examples without hidden numeric cues


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _connector_info(
    sprite: set[tuple[int, int]], g: np.ndarray, ctrl: int
) -> dict[str, object]:
    conn = {(int(r), int(c)) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}
    assert conn
    r0, c0, r1, c1 = _bbox_of_pts(conn)
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    rel = {(int(r - r0), int(c - c0)) for r, c in conn}
    assert h == 2 and w >= 3
    assert len(rel) == w  # exactly one ctrl pixel per connector column
    profile = []
    for dc in range(w):
        rows = [dr for dr, cc in rel if cc == dc]
        assert len(rows) == 1 and rows[0] in (0, 1)
        profile.append(int(rows[0]))
    assert set(profile) == {0, 1}  # forbid degeneration back to a flat bar
    return {
        "pts": conn,
        "r0": int(r0),
        "c0": int(c0),
        "h": int(h),
        "w": int(w),
        "profile": tuple(profile),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using complementary zigzag connectors, then tight-crop.

    Each sprite has one ctrl-colored connector occupying a 2 x w band. In each
    connector column exactly one of the two rows is ctrl. Base and plug use
    complementary row choices, so their seam interlocks when aligned.
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, plug = A, B
    else:
        base, plug = B, A

    info_base = _connector_info(base, g, ctrl)
    info_plug = _connector_info(plug, g, ctrl)
    conn_base = info_base["pts"]
    conn_plug = info_plug["pts"]
    assert isinstance(conn_base, set) and isinstance(conn_plug, set)

    assert info_base["h"] == info_plug["h"] == 2
    assert info_base["w"] == info_plug["w"]
    prof_base = tuple(int(x) for x in info_base["profile"])
    prof_plug = tuple(int(x) for x in info_plug["profile"])
    assert tuple(1 - x for x in prof_base) == prof_plug

    base_body = base - conn_base
    plug_body = plug - conn_plug
    for r, c in conn_base:
        assert (int(r + 1), int(c)) not in base_body
    for r, c in conn_plug:
        assert (int(r - 1), int(c)) not in plug_body

    tr = int(info_base["r0"]) - int(info_plug["r0"])
    tc = int(info_base["c0"]) - int(info_plug["c0"])
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    base_coords = np.array(list(base), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
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
    cropped = _crop_non_bg(out, bg)
    assert cropped is not None
    return cropped


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _random_profile(width: int) -> tuple[int, ...]:
    while True:
        prof = [random.randint(0, 1) for _ in range(int(width))]
        if set(prof) != {0, 1}:
            continue
        if sum(prof[i] != prof[i - 1] for i in range(1, len(prof))) < 1:
            continue
        return tuple(int(x) for x in prof)


def _paint_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int], allowed: np.ndarray) -> bool:
    H, W = mask.shape
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not bool(allowed[nr, nc]):
                continue
            if (nr, nc) in prev:
                continue
            prev[(nr, nc)] = (r, c)
            q.append((nr, nc))
    if goal not in prev:
        return False
    cur: tuple[int, int] | None = goal
    while cur is not None:
        mask[int(cur[0]), int(cur[1])] = True
        cur = prev[cur]
    return True


def _grow_shape(mask: np.ndarray, target: int, allowed: np.ndarray) -> None:
    H, W = mask.shape
    while int(mask.sum()) < int(target):
        frontier = []
        for r, c in zip(*np.where(mask)):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W and bool(allowed[nr, nc]) and not bool(mask[nr, nc]):
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return
        nr, nc = random.choice(frontier)
        mask[int(nr), int(nc)] = True


def _build_base_sprite(
    base_h: int, base_w: int, band_top: int, conn_c0: int, profile: tuple[int, ...], bg: int, obj: int, ctrl: int
) -> np.ndarray | None:
    body = np.zeros((int(base_h), int(base_w)), dtype=bool)
    allowed = np.zeros_like(body, dtype=bool)
    allowed[: int(band_top) + 1, :] = True

    conn_pts = []
    required = {(0, 0)}
    for i, bit in enumerate(profile):
        c = int(conn_c0 + i)
        cr = int(band_top + bit)
        conn_pts.append((cr, c))
        allowed[cr, c] = False
        if int(bit) == 0:
            required.add((int(band_top - 1), c))
        else:
            required.add((int(band_top), c))
    if any(not bool(allowed[r, c]) for r, c in required):
        return None

    req_list = sorted(required)
    body[req_list[0]] = True
    for pt in req_list[1:]:
        if not _paint_path(body, req_list[0], pt, allowed):
            return None
    max_cells = int(allowed.sum())
    cur_cells = int(body.sum())
    if max_cells < 5 or cur_cells > max_cells:
        return None
    lo = max(5, cur_cells)
    hi = min(max_cells, max(lo, 24))
    target = random.randint(lo, hi)
    _grow_shape(body, target, allowed)

    grid = np.full((int(base_h), int(base_w)), int(bg), dtype=int)
    grid[body] = int(obj)
    for r, c in conn_pts:
        grid[int(r), int(c)] = int(ctrl)
    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        return None
    return _crop_non_bg(grid, bg)


def _build_plug_sprite(
    plug_h: int,
    plug_w: int,
    conn_c0: int,
    profile: tuple[int, ...],
    dock_band_top: int,
    dock_left: int,
    base_grid: np.ndarray,
    bg: int,
    obj: int,
    ctrl: int,
) -> np.ndarray | None:
    body = np.zeros((int(plug_h), int(plug_w)), dtype=bool)
    allowed = np.ones_like(body, dtype=bool)
    allowed[0, :] = False

    conn_pts = []
    required: set[tuple[int, int]] = set()
    plug_profile = tuple(1 - int(x) for x in profile)
    for i, bit in enumerate(plug_profile):
        c = int(conn_c0 + i)
        cr = int(bit)
        conn_pts.append((cr, c))
        allowed[cr, c] = False
        if int(bit) == 0:
            required.add((1, c))
        else:
            required.add((2, c))

    if any(not bool(allowed[r, c]) for r, c in required):
        return None
    bottom_anchor = (int(plug_h - 1), random.randint(0, int(plug_w - 1)))
    if not bool(allowed[bottom_anchor]):
        return None
    required.add(bottom_anchor)

    req_list = sorted(required)
    body[req_list[0]] = True
    for pt in req_list[1:]:
        if not _paint_path(body, req_list[0], pt, allowed):
            return None
    max_cells = int(allowed.sum())
    cur_cells = int(body.sum())
    if max_cells < 5 or cur_cells > max_cells:
        return None
    lo = max(5, cur_cells)
    hi = min(max_cells, max(lo, 24))
    target = random.randint(lo, hi)
    _grow_shape(body, target, allowed)

    grid = np.full((int(plug_h), int(plug_w)), int(bg), dtype=int)
    grid[body] = int(obj)
    for r, c in conn_pts:
        grid[int(r), int(c)] = int(ctrl)

    for r in range(int(plug_h)):
        for c in range(int(plug_w)):
            val = int(grid[r, c])
            if val == int(bg):
                continue
            dr = int(dock_band_top + r)
            dc = int(dock_left + c)
            if 0 <= dr < base_grid.shape[0] and 0 <= dc < base_grid.shape[1]:
                base_val = int(base_grid[dr, dc])
                if val == int(obj) and base_val == int(obj):
                    return None
                if val == int(ctrl) and base_val == int(ctrl):
                    return None

    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        return None
    return _crop_non_bg(grid, bg)


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    conn_w = random.randint(3, 5)
    profile = _random_profile(conn_w)

    base_h = random.randint(4, 7)
    base_w = random.randint(max(3, conn_w), min(10, conn_w + 4))
    band_top = random.randint(1, base_h - 2)
    conn_c0_base = random.randint(0, base_w - conn_w)
    base_sprite = _build_base_sprite(base_h, base_w, band_top, conn_c0_base, profile, bg, obj, ctrl)
    if base_sprite is None:
        return None

    base_conn_cols = [c for r, c in np.argwhere(base_sprite == int(ctrl))]
    base_conn_rows = [r for r, c in np.argwhere(base_sprite == int(ctrl))]
    base_band_top = int(min(base_conn_rows))
    base_conn_c0 = int(min(base_conn_cols))

    plug_h = random.randint(3, 7)
    plug_w = random.randint(max(3, conn_w), min(10, conn_w + 4))
    max_plug_conn_c0 = min(int(plug_w - conn_w), int(base_conn_c0))
    if max_plug_conn_c0 < 0:
        return None
    conn_c0_plug = random.randint(0, max_plug_conn_c0)
    dock_left = int(base_conn_c0 - conn_c0_plug)

    plug_sprite = _build_plug_sprite(
        plug_h, plug_w, conn_c0_plug, profile, base_band_top, dock_left, base_sprite, bg, obj, ctrl
    )
    if plug_sprite is None:
        return None

    if int(np.sum(base_sprite != int(bg))) < 5 or int(np.sum(plug_sprite != int(bg))) < 5:
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    canvas_h = random.randint(max(7, bh + 1, ph + 1), 11)
    canvas_w = random.randint(max(5, bw + 1, pw + 1), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=1)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=bg)

    if np.sum(canvas == int(bg)) <= canvas.size // 3:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != int(bg)):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
