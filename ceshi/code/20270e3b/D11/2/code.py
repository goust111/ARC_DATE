# Variant of 20270e3b
# Dimension: D11 — 对接时的旋转/翻转 (transform)
# Change: plug仅做平移对接，无旋转或翻转... → 旋转方向由连接器位置编码

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D11 variant
# ---------------------------------------------------------------------------
#
# source_task:
#   none
#
# variant_spec:
#   core_change:
#     original: plug connector is always on the top edge of the plug sprite, so
#       the plug only needs translation before docking.
#     new: the input plug is a rotated version of that canonical sprite. The
#       connector side encodes the required rotation:
#         top edge    -> rotate 0 deg
#         right edge  -> rotate 90 deg CCW
#         bottom edge -> rotate 180 deg
#         left edge   -> rotate 270 deg CCW
#       After restoring the canonical orientation, docking is exactly the same
#       as the original task: place the plug so its connector row is one cell
#       above the base connector row.
#   pixel_level_behavior:
#     base:
#       unchanged from the seed task: one 8-connected sprite anchored at (0,0),
#       with a horizontal ctrl bar whose body is above and free side is below.
#     plug:
#       canonical form still has a horizontal ctrl bar on the top edge and body
#       below it, but the input shows one of four rotated versions. Therefore
#       the ctrl bar may appear horizontal on the top/bottom edge, or vertical
#       on the left/right edge.
#   concrete_trace:
#     original_flow:
#       - find base and plug
#       - assume plug connector is already on the top side
#       - translate plug upward so plug row 0 aligns one row above the base
#         connector row
#     new_flow:
#       - find base and plug
#       - detect which bbox edge contains the ctrl bar
#       - rotate plug back so the ctrl bar is on the top edge
#       - reuse the original translation-and-dock logic
#     difference:
#       the transformation step moves from "pure translation" to
#       "side-detected rotation + translation".
#   complexity_calibration:
#     - Only four rotation states are used, small enough to infer from a few
#       examples.
#     - To keep the rotation visually observable, the generator rejects plugs
#       whose cropped sprite is rotationally symmetric.
#     - To avoid ambiguity for 1-pixel connectors, the connector is forced to be
#       interior to the canonical plug bbox horizontally; after rotation it stays
#       away from corners, so the side is unique.
#   implicit_assumption_audit:
#     - Original `assert shpA == shpB` is no longer valid before rotation,
#       because base has a horizontal connector while a rotated plug may have a
#       vertical connector. The new solver compares connector shapes only after
#       canonicalizing the plug.
#     - Original plug-side checks `r-1 not in plug_body` assume the connector is
#       already on top. Those checks are now applied after rotation recovery.
#     - Original shift uses the connector's normalized top-left anchor. This is
#       still valid after plug canonicalization because the connector is again a
#       horizontal row.
#     - Original overlap rule "only ctrl may overlap" remains valid, because the
#       docking geometry itself is unchanged.
#   placement_legality:
#     - Generator still builds the assembled target first in canonical
#       coordinates, verifying that the plug body overlaps the base only at the
#       connector seam.
#     - Only after that is the plug rotated and placed elsewhere on the canvas.
#     - Placement uses padding=1 under 8-connectivity so the two sprites remain
#       distinct components in the input.
#   visual_observability:
#     - The connector side is a visible structural cue, not a hidden numeric
#       attribute.
#     - The plug body is required to be rotationally asymmetric, so a 90/180/270
#       degree correction produces a visibly different object.
#   visual_cleanliness:
#     - Still exactly two sprites, still only three colors in input and two in
#       output, and no extra markers are added.
#     - The connector remains a contiguous 1-3 cell bar, so the rotated cue is
#       easy to isolate by eye.
#   color_inference:
#     - No new color rule is introduced. `ctrl` remains the only special color
#       and is recolored to `obj` after docking, just like the original task.
#   new_solver_assumptions_and_generator_guards:
#     - Solver assumes the plug connector occupies exactly one bbox edge and is
#       either a straight row or a straight column. Generator enforces this by
#       rotating a canonical top-edge bar only.
#     - Solver assumes that after canonicalization, base and plug connector
#       lengths match. Generator constructs both from the same chosen length.
#     - Solver assumes the restored plug does not extend above or left of the
#       base after docking. Generator checks this in canonical assembled space
#       before rotating the plug away.

BG = 1
OBJ = 4
CTRL = 7

SIDE_TO_RESTORE_K = {
    "top": 0,
    "right": 1,
    "bottom": 2,
    "left": 3,
}

SIDE_TO_INPUT_K = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _component_records(grid: np.ndarray) -> list[dict]:
    lab, n = label_mask(grid != BG, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        sprite = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(BG), dtype=int)
        for r, c in coords:
            sprite[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": {(int(r), int(c)) for r, c in coords},
                "sprite": sprite,
            }
        )
    return comps


def _connector_points(sprite: np.ndarray) -> np.ndarray:
    return np.argwhere(sprite == int(CTRL))


def _norm_conn(conn_pts: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    assert conn_pts.size > 0
    r0, c0 = conn_pts.min(axis=0)
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _detect_connector_side(sprite: np.ndarray) -> tuple[str, int]:
    conn = _connector_points(sprite)
    assert conn.size > 0
    rows = sorted({int(r) for r, _c in conn})
    cols = sorted({int(c) for _r, c in conn})
    h, w = sprite.shape
    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[0] + len(cols)))
        if r == 0:
            return "top", int(conn.shape[0])
        if r == h - 1:
            return "bottom", int(conn.shape[0])
    if len(cols) == 1:
        c = cols[0]
        assert rows == list(range(rows[0], rows[0] + len(rows)))
        if c == 0:
            return "left", int(conn.shape[0])
        if c == w - 1:
            return "right", int(conn.shape[0])
    raise AssertionError("connector is not a unique straight edge bar")


def _restore_plug(sprite: np.ndarray) -> tuple[np.ndarray, str]:
    side, _conn_len = _detect_connector_side(sprite)
    return np.rot90(sprite, SIDE_TO_RESTORE_K[side]), side


def _rotationally_distinct(sprite: np.ndarray) -> bool:
    return all(not np.array_equal(sprite, np.rot90(sprite, k)) for k in (1, 2, 3))


def _valid_base_connector(sprite: np.ndarray) -> bool:
    conn = _connector_points(sprite)
    if conn.size == 0:
        return False
    rows = {int(r) for r, _c in conn}
    if len(rows) != 1:
        return False
    for r, c in conn:
        rr = int(r)
        cc = int(c)
        if rr == 0:
            return False
        if int(sprite[rr - 1, cc]) != int(OBJ):
            return False
        if rr + 1 < sprite.shape[0] and int(sprite[rr + 1, cc]) == int(OBJ):
            return False
    return True


def _valid_canonical_plug(sprite: np.ndarray) -> bool:
    try:
        side, _conn_len = _detect_connector_side(sprite)
    except AssertionError:
        return False
    if side != "top":
        return False
    conn = _connector_points(sprite)
    cols = sorted({int(c) for _r, c in conn})
    if cols[0] <= 0 or cols[-1] >= sprite.shape[1] - 1:
        return False
    for r, c in conn:
        rr = int(r)
        cc = int(c)
        if rr != 0:
            return False
        if sprite.shape[0] < 2 or int(sprite[1, cc]) != int(OBJ):
            return False
    return True


def _render_abs_points(points: dict[tuple[int, int], int]) -> np.ndarray:
    coords = np.array(list(points.keys()), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(BG), dtype=int)
    for (r, c), val in points.items():
        out[int(r - r0), int(c - c0)] = int(val)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the plug back to its canonical docking orientation, then dock it.
    """
    g = np.array(grid, dtype=int)

    comps = _component_records(g)
    assert len(comps) == 2

    comps.sort(key=lambda rec: (int(rec["bbox"][0] + rec["bbox"][1]), int(rec["bbox"][0]), int(rec["bbox"][1])))
    base_rec, plug_rec = comps[0], comps[1]
    r0b, c0b, _r1b, _c1b = base_rec["bbox"]
    assert int(r0b + c0b) == 0

    base_sprite = base_rec["sprite"]
    plug_input = plug_rec["sprite"]
    plug_sprite, _side = _restore_plug(plug_input)

    assert _valid_base_connector(base_sprite)
    assert _valid_canonical_plug(plug_sprite)

    conn_base = _connector_points(base_sprite)
    conn_plug = _connector_points(plug_sprite)
    shp_base, (brc0, bcc0) = _norm_conn(conn_base)
    shp_plug, (prc0, pcc0) = _norm_conn(conn_plug)
    assert shp_base == shp_plug

    base_abs = dict()
    br0, bc0, _br1, _bc1 = base_rec["bbox"]
    for rr in range(base_sprite.shape[0]):
        for cc in range(base_sprite.shape[1]):
            val = int(base_sprite[rr, cc])
            if val != int(BG):
                base_abs[(int(br0 + rr), int(bc0 + cc))] = val

    tr = int((br0 + brc0 - 1) - prc0)
    tc = int((bc0 + bcc0) - pcc0)

    merged = dict(base_abs)
    for rr in range(plug_sprite.shape[0]):
        for cc in range(plug_sprite.shape[1]):
            val = int(plug_sprite[rr, cc])
            if val == int(BG):
                continue
            ar = int(tr + rr)
            ac = int(tc + cc)
            cur = merged.get((ar, ac), int(BG))
            if cur != int(BG):
                assert cur == int(CTRL) or val == int(CTRL)
            merged[(ar, ac)] = val

    base_min_r = min(r for r, _c in base_abs)
    base_min_c = min(c for _r, c in base_abs)
    all_min_r = min(r for r, _c in merged)
    all_min_c = min(c for _r, c in merged)
    assert all_min_r == base_min_r and all_min_c == base_min_c

    out = _render_abs_points(merged)
    out[out == int(CTRL)] = int(OBJ)
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray | None = None,
) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = random.randint(0, len(frontier) - 1)
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
        if allowed is not None:
            nbrs = [(nr, nc) for nr, nc in nbrs if bool(allowed[int(nr), int(nc)])]
            if not nbrs:
                frontier[idx] = frontier[-1]
                frontier.pop()
                continue
        nr, nc = random.choice(nbrs)
        mask[int(nr), int(nc)] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    random.shuffle(pixels)
    for r, c in pixels:
        r = int(r)
        c = int(c)
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


def _build_base(conn_len: int) -> np.ndarray | None:
    base_h = random.randint(2, 7)
    base_w = random.randint(max(3, conn_len), 10)
    base_target = random.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    candidates = []
    for r in range(1, base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if c0 < 1:
                continue
            if not all(base[r, c] for c in cols):
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            if any(r + 1 < base_h and base[r + 1, c] for c in cols):
                continue
            candidates.append((r, cols))
    if not candidates:
        return None

    conn_row, conn_cols = random.choice(candidates)
    base_grid = np.full((base_h, base_w), int(BG), dtype=int)
    base_grid[base] = int(OBJ)
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = int(CTRL)
    if not is_contiguous(base_grid != int(BG), background=0, connectivity=8):
        return None
    return _crop_non_bg(base_grid)


def _build_canonical_plug(conn_len: int, base_sprite: np.ndarray) -> np.ndarray | None:
    base_h, base_w = base_sprite.shape
    conn_base = _connector_points(base_sprite)
    _shape, (base_conn_r0, base_conn_c0) = _norm_conn(conn_base)

    plug_h = random.randint(3, 7)
    min_w = max(conn_len + 2, 3)
    max_w = min(10, max(min_w, base_w + 4))
    if min_w > max_w:
        return None
    plug_w = random.randint(min_w, max_w)

    max_conn_c0 = min(base_conn_c0, plug_w - conn_len - 1)
    if max_conn_c0 < 1:
        return None
    plug_conn_c0 = random.randint(1, max_conn_c0)
    dc_offset = int(base_conn_c0 - plug_conn_c0)
    if dc_offset < 0:
        return None

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        plug[1, c] = True

    tr = int(base_conn_r0 - 1)
    tc = int(base_conn_c0 - plug_conn_c0)
    allowed = np.ones((plug_h, plug_w), dtype=bool)
    allowed[0, :] = False
    allowed[0, plug_conn_c0 : plug_conn_c0 + conn_len] = True
    for r in range(plug_h):
        for c in range(plug_w):
            dr = int(tr + r)
            dc = int(tc + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and int(base_sprite[dr, dc]) == int(OBJ):
                allowed[r, c] = False
    allowed[0, plug_conn_c0 : plug_conn_c0 + conn_len] = True

    keep = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    keep.update({(1, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)})

    def add_side_seed(side: str) -> bool:
        if side == "left":
            candidate_cols = list(range(plug_conn_c0 - 1, -1, -1))
        else:
            candidate_cols = list(range(plug_conn_c0 + conn_len, plug_w))
        for c in candidate_cols:
            for r in (1, 2, 3):
                if 0 <= r < plug_h and bool(allowed[r, c]):
                    plug[r, c] = True
                    keep.add((int(r), int(c)))
                    return True
        return False

    if not add_side_seed("left") or not add_side_seed("right"):
        return None

    plug_target = random.randint(5, min(29, plug_h * plug_w))
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))), allowed=allowed)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    _thin(plug, keep=keep)

    plug_grid = np.full((plug_h, plug_w), int(BG), dtype=int)
    plug_grid[plug] = int(OBJ)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = int(CTRL)
        if int(plug_grid[1, c]) != int(OBJ):
            return None
    if not is_contiguous(plug_grid != int(BG), background=0, connectivity=8):
        return None

    plug_sprite = _crop_non_bg(plug_grid)
    if plug_sprite is None or not _valid_canonical_plug(plug_sprite):
        return None
    if not _rotationally_distinct(plug_sprite):
        return None

    base_grid = np.array(base_sprite, dtype=int)
    conn_plug = _connector_points(plug_sprite)
    _shape_p, (plug_conn_r0, plug_conn_c0_after_crop) = _norm_conn(conn_plug)
    tr = int((base_conn_r0 - 1) - plug_conn_r0)
    tc = int(base_conn_c0 - plug_conn_c0_after_crop)

    for r in range(plug_sprite.shape[0]):
        for c in range(plug_sprite.shape[1]):
            val = int(plug_sprite[r, c])
            if val == int(BG):
                continue
            dr = int(tr + r)
            dc = int(tc + c)
            if 0 <= dr < base_grid.shape[0] and 0 <= dc < base_grid.shape[1]:
                cur = int(base_grid[dr, dc])
                if cur == int(OBJ):
                    if val == int(CTRL):
                        continue
                    return None
                if cur == int(CTRL):
                    continue
    if tc < 0 or tr < 0:
        return None

    return plug_sprite


def _gen_attempt() -> np.ndarray | None:
    conn_len = random.randint(1, 3)
    base_sprite = _build_base(conn_len)
    if base_sprite is None or not _valid_base_connector(base_sprite):
        return None

    conn_base = _connector_points(base_sprite)
    if int(conn_base.shape[0]) != conn_len:
        return None

    plug_sprite = _build_canonical_plug(conn_len, base_sprite)
    if plug_sprite is None:
        return None

    input_side = random.choice(["top", "right", "bottom", "left"])
    plug_input = np.rot90(plug_sprite, SIDE_TO_INPUT_K[input_side])
    side_check, side_len = _detect_connector_side(plug_input)
    if side_check != input_side or side_len != conn_len:
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_input.shape

    canvas_h_min = max(7, max(bh, ph) + 2)
    canvas_w_min = max(3, max(bw, pw) + 2)
    if canvas_h_min > 11 or canvas_w_min > 13:
        return None
    canvas_h = random.randint(canvas_h_min, 11)
    canvas_w = random.randint(canvas_w_min, 13)

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(BG))
    try:
        pr, pc = random_free_location(canvas, plug_input, transparent=int(BG), padding=1, padding_connectivity=8)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_input, pr, pc, transparent=int(BG))

    if np.sum(canvas == int(BG)) <= canvas.size // 2:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None
    if out.size == 0 or not np.any(out != int(BG)):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
