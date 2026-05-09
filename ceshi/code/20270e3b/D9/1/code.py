# Variant of 20270e3b
# Dimension: D9 — 连接器形状参数 (meta)
# Change: 连接器为1-3像素长的水平条，位于单行上 -> 2D矩形连接器（2x2 或 2x3）

from __future__ import annotations

import random
import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


# -----------------------------------------------------------------------------
# Design analysis for the D9 variant
# -----------------------------------------------------------------------------
#
# variant_spec:
#   core_change:
#     original_connector: a 1xL horizontal ctrl bar, L in {1,2,3}
#     new_connector: a 2xW solid ctrl rectangle, W in {2,3}
#   concrete_appearance:
#     - Each sprite still has one object-color body and one ctrl-color connector.
#     - The connector is now a full 2-row block instead of a single highlighted row.
#     - Base sprite: connector block is attached to the underside of the body.
#     - Plug sprite: connector block is attached to the topside of the body.
#     - Both sprites share the same normalized connector shape inside one puzzle:
#       either both are 2x2 or both are 2x3.
#   docking_rule:
#     - Match the two connector rectangles by exact normalized shape.
#     - Translate the plug so its connector block is directly above the base block:
#       same columns, and the plug block's bottom edge is one row above the base
#       block's top edge.
#     - This is the exact rectangle-height generalization of the original formula.
#       For height=1 it becomes the old "one row above" rule.
#     - After stamping, all ctrl cells recolor to obj and the result is tight-cropped.
#   behavior_trace:
#     example:
#       connector_shape: 2x3
#       base_connector_bbox: rows 4..5, cols 2..4
#       plug_connector_bbox: rows 0..1, cols 1..3
#       original_logic:
#         - The old solver would only track the top row of a connector.
#         - Its shift used tr = base_top - 1 - plug_top, which assumes height=1.
#         - With a 2-row block this would place the plug too low by one row.
#       new_logic:
#         - Height is part of the connector geometry.
#         - We use tr = base_top - connector_height - plug_top.
#         - Here tr = 4 - 2 - 0 = 2, tc = 2 - 1 = 1.
#         - The plug block lands at rows 2..3, directly above the base block
#           at rows 4..5, with no ctrl-ctrl overlap.
#       visible_difference:
#         - In input, orange connectors are visibly 2D blocks, not thin bars.
#         - In output, the merged seam is thicker because two connector rows on
#           each sprite are recolored into the yellow body.
#   complexity_calibration:
#     - Only two connector widths are allowed: 2 and 3.
#     - Height is fixed at 2, so the variant is visually obvious but still easy
#       to compare across the two sprites.
#     - The connector shape is repeated twice per example (base + plug), which
#       gives the human solver direct support for the matching rule.
#     - No extra transformations (rotation/flip/color remapping) are added.
#   implicit_assumption_audit_against_original_main:
#     - original lines 78..81 checked per-pixel empty space above/below a 1-row
#       connector. That assumption breaks for a 2-row connector because internal
#       ctrl cells do not have obj directly above/below them.
#       New rule: treat the connector as a block; require a full obj support row
#       above the base block and below the plug block, and require the opposite
#       outer side of the block to be empty.
#     - original line 87 used dr0 - 1 - pr0. The hard-coded "1" is the old
#       connector height. New rule uses connector_height instead.
#     - original generator searched single rows for ctrl placement. That no
#       longer guarantees a valid rectangle, so the generator now explicitly
#       constructs a 2x2 or 2x3 block.
#     - original plug growth seeded one connector row plus one body row. The new
#       generator seeds two connector rows plus one body-support row.
#   placement_legality:
#     - Base is still pasted at canvas origin so the solver can identify it by
#       bbox top-left.
#     - Plug is placed strictly to the right or strictly below the base with a
#       gap of 1..5 cells, so the two sprites never overlap or touch 8-neighbor.
#     - During assembled-state verification, any overlap is allowed only when at
#       least one side is ctrl. This guards against accidental obj-obj collisions.
#   visual_observability:
#     - The D9 change is visible in every training/test instance because the
#       connector occupies 4 or 6 cells and sits as a salient colored block on
#       the docking edge.
#     - The solver does not rely on invisible color ordering or metadata.
#   cleanliness:
#     - Connectors are kept at the bottom edge of the base and the top edge of
#       the plug. This avoids clutter from unrelated pixels mixed into the
#       connector rows and keeps the block easy to see.
#   new_main_assumptions_and_generator_guarantees:
#     - Assumption: each sprite contains exactly one solid ctrl rectangle.
#       Guarantee: generator constructs exactly one ctrl block per sprite.
#     - Assumption: the two ctrl rectangles have identical shape.
#       Guarantee: one shared (height,width) is chosen per puzzle.
#     - Assumption: base support row exists directly above the rectangle and plug
#       support row exists directly below the rectangle.
#       Guarantee: generator seeds and preserves those rows.
#     - Assumption: the plug does not extend above or left of the base after
#       docking.
#       Guarantee: base connector starts low enough, and plug connector column is
#       chosen so the docking translation has non-negative row/col offsets.
# -----------------------------------------------------------------------------


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bg: int) -> np.ndarray | None:
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
    rect_shape = {(dr, dc) for dr in range(h) for dc in range(w)}
    assert shape == rect_shape
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(h),
        "w": int(w),
        "shape": shape,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching 2D rectangular connector blocks, then crop.

    Variant rule:
    - ctrl is a solid 2x2 or 2x3 rectangle on each sprite.
    - Base is the sprite anchored at the grid origin.
    - Translate the plug so its connector rectangle is directly above the base
      connector rectangle, aligned by columns.
    - Recolor all ctrl cells to obj and tight-crop the merged result.
    """
    g = np.array(grid, dtype=int)

    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
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
    assert infoA["h"] == infoB["h"]
    assert infoA["w"] == infoB["w"]

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
        assert (int(infoD["r1"]) + 1, int(c)) not in base_body
        assert (int(infoD["r0"]) - 1, int(c)) in base_body
    for c in range(int(infoP["c0"]), int(infoP["c1"]) + 1):
        assert (int(infoP["r0"]) - 1, int(c)) not in plug_body
        assert (int(infoP["r1"]) + 1, int(c)) in plug_body

    tr = int(infoD["r0"] - infoP["h"] - infoP["r0"])
    tc = int(infoD["c0"] - infoP["c0"])
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


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]], allowed: np.ndarray) -> None:
    """Grow a connected shape within the allowed region."""
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
    """Remove interior pixels while preserving connectivity and kept cells."""
    h, w = mask.shape
    keep = keep or set()
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


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Draw a random Manhattan path between two cells."""
    r, c = map(int, start)
    tr, tc = map(int, end)
    mask[r, c] = True
    while (r, c) != (tr, tc):
        steps = []
        if r != tr:
            steps.append((r + (1 if tr > r else -1), c))
        if c != tc:
            steps.append((r, c + (1 if tc > c else -1)))
        r, c = _rng.choice(steps)
        mask[r, c] = True


def _build_base(conn_h: int, conn_w: int) -> tuple[np.ndarray, dict[str, int]] | None:
    bg, obj, ctrl = 1, 4, 7
    base_h = _rng.randint(4, 6)
    base_w = _rng.randint(max(conn_w + 1, 3), 8)
    conn_r0 = int(base_h - conn_h)
    conn_c0 = _rng.randint(0, base_w - conn_w)

    allowed = np.zeros((base_h, base_w), dtype=bool)
    allowed[:conn_r0, :] = True
    allowed[conn_r0 : conn_r0 + conn_h, conn_c0 : conn_c0 + conn_w] = True

    mask = np.zeros((base_h, base_w), dtype=bool)
    mask[conn_r0 : conn_r0 + conn_h, conn_c0 : conn_c0 + conn_w] = True
    mask[conn_r0 - 1, conn_c0 : conn_c0 + conn_w] = True
    mask[0, 0] = True
    _draw_path(mask, (0, 0), (conn_r0 - 1, _rng.randint(conn_c0, conn_c0 + conn_w - 1)))

    target_min = max(int(mask.sum()) + 1, 7)
    target_max = min(24, int(allowed.sum()))
    if target_max < target_min:
        return None
    target = _rng.randint(target_min, target_max)
    _grow_shape(mask, target, list(zip(*np.where(mask))), allowed)

    keep = {(0, 0)}
    keep |= {
        (r, c)
        for r in range(conn_r0, conn_r0 + conn_h)
        for c in range(conn_c0, conn_c0 + conn_w)
    }
    keep |= {(conn_r0 - 1, c) for c in range(conn_c0, conn_c0 + conn_w)}
    _thin(mask, keep=keep)

    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if not all(mask[conn_r0 - 1, c] for c in range(conn_c0, conn_c0 + conn_w)):
        return None

    grid = np.full((base_h, base_w), int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[conn_r0 : conn_r0 + conn_h, conn_c0 : conn_c0 + conn_w] = int(ctrl)

    return grid, {"r0": conn_r0, "c0": conn_c0, "h": conn_h, "w": conn_w}


def _build_plug(conn_h: int, conn_w: int, max_conn_c0: int) -> tuple[np.ndarray, dict[str, int]] | None:
    bg, obj, ctrl = 1, 4, 7
    plug_h = _rng.randint(conn_h + 1, 6)
    plug_w = _rng.randint(conn_w, 8)
    max_plug_conn_c0 = min(int(max_conn_c0), int(plug_w - conn_w))
    if max_plug_conn_c0 < 0:
        return None
    conn_c0 = _rng.randint(0, max_plug_conn_c0)

    allowed = np.zeros((plug_h, plug_w), dtype=bool)
    allowed[conn_h:, :] = True
    allowed[:conn_h, conn_c0 : conn_c0 + conn_w] = True

    mask = np.zeros((plug_h, plug_w), dtype=bool)
    mask[:conn_h, conn_c0 : conn_c0 + conn_w] = True
    mask[conn_h, conn_c0 : conn_c0 + conn_w] = True
    stem_col = _rng.randint(conn_c0, conn_c0 + conn_w - 1)
    _draw_path(mask, (conn_h, stem_col), (plug_h - 1, _rng.randint(0, plug_w - 1)))

    target_min = max(int(mask.sum()) + 1, 7)
    target_max = min(24, int(allowed.sum()))
    if target_max < target_min:
        return None
    target = _rng.randint(target_min, target_max)
    _grow_shape(mask, target, list(zip(*np.where(mask))), allowed)

    keep = {
        (r, c)
        for r in range(0, conn_h)
        for c in range(conn_c0, conn_c0 + conn_w)
    }
    keep |= {(conn_h, c) for c in range(conn_c0, conn_c0 + conn_w)}
    _thin(mask, keep=keep)

    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if not all(mask[conn_h, c] for c in range(conn_c0, conn_c0 + conn_w)):
        return None

    grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[:conn_h, conn_c0 : conn_c0 + conn_w] = int(ctrl)

    return grid, {"r0": 0, "c0": conn_c0, "h": conn_h, "w": conn_w}


def _can_dock_cleanly(base_grid: np.ndarray, plug_grid: np.ndarray, base_info: dict[str, int], plug_info: dict[str, int]) -> bool:
    bg, ctrl = 1, 7
    tr = int(base_info["r0"] - plug_info["h"] - plug_info["r0"])
    tc = int(base_info["c0"] - plug_info["c0"])
    if tr < 0 or tc < 0:
        return False

    bh, bw = base_grid.shape
    ph, pw = plug_grid.shape
    for r in range(ph):
        for c in range(pw):
            val = int(plug_grid[r, c])
            if val == int(bg):
                continue
            dr = int(r + tr)
            dc = int(c + tc)
            if 0 <= dr < bh and 0 <= dc < bw:
                cur = int(base_grid[dr, dc])
                if cur != int(bg) and cur != int(ctrl) and val != int(ctrl):
                    return False
    return True


def _place_input(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray | None:
    bg = 1
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    placements = []
    for gap in range(1, 6):
        if bh + gap + ph <= 13:
            placements.append(("below", gap))
        if bw + gap + pw <= 13:
            placements.append(("right", gap))
    if not placements:
        return None

    orient, gap = _rng.choice(placements)
    if orient == "below":
        canvas_h_min = bh + gap + ph
        canvas_h = _rng.randint(max(7, canvas_h_min), 13)
        canvas_w = _rng.randint(max(4, bw, pw), 13)
        pr = bh + gap
        pc = _rng.randint(0, canvas_w - pw)
    else:
        canvas_h = _rng.randint(max(7, bh, ph), 13)
        canvas_w_min = bw + gap + pw
        canvas_w = _rng.randint(max(4, canvas_w_min), 13)
        pr = _rng.randint(0, canvas_h - ph)
        pc = bw + gap

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    lab, n = label_mask(canvas != int(bg), connectivity=8)
    if int(n) != 2:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg = 1
    conn_h = 2
    conn_w = _rng.choice((2, 3))

    built_base = _build_base(conn_h, conn_w)
    if built_base is None:
        return None
    base_grid, base_info = built_base

    built_plug = _build_plug(conn_h, conn_w, max_conn_c0=int(base_info["c0"]))
    if built_plug is None:
        return None
    plug_grid, plug_info = built_plug

    if not _can_dock_cleanly(base_grid, plug_grid, base_info, plug_info):
        return None

    base_sprite = _crop(base_grid, bg)
    plug_sprite = _crop(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None

    canvas = _place_input(base_sprite, plug_sprite)
    if canvas is None:
        return None
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
