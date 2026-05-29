# Variant of 20270e3b
# Dimension: D15 — 连接器自由侧约束 (meta)
# Change: base连接器自由侧朝下（下方无body像素），plug连接器自由侧朝上（上方无body像素）... → 自由侧方向相同（均朝下）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


# Design analysis for the D15 variant
# -----------------------------------
# a) Concrete behavior and trace:
#    - Input still contains exactly two 8-connected sprites in colors bg=1, obj=4, ctrl=7.
#    - The base sprite remains anchored at the global top-left and still has a horizontal
#      ctrl connector whose free side faces downward.
#    - The plug sprite now also has its connector free side facing downward, so in the input
#      both sprites have obj pixels directly above every ctrl pixel.
#    - To dock them with the same final geometry as the original task (plug above base,
#      connector rows separated by one cell), the solver must first vertically flip the plug
#      inside its own bbox. After that flip, the plug connector faces upward and can be placed
#      one row above the base connector.
#    - Small trace:
#        original task: plug connector at local row 0, body below -> translate upward to dock.
#        variant task: input plug is the vertical flip of that shape, so its connector sits near
#        the bottom with body above -> flip it back -> then translate upward to dock.
#      The only semantic change is the extra flip step before translation.
# b) Complexity calibration:
#    - The rule still uses one connector bar per sprite with length 1..3, so correspondence
#      stays unique and human-readable.
#    - To make the flip visually observable, the generator rejects vertically symmetric plug
#      shapes; otherwise the new rule would degenerate to the original visually.
#    - The transform family is intentionally kept to one case only: vertical flip of the plug.
#      This keeps the task inferable from a few examples instead of branching across many
#      unrelated transforms.
# c) Hidden assumptions in the original main():
#    - The original code assumes `plug` already faces upward and asserts that directly.
#    - It also translates raw plug coordinates without any normalization transform.
#    - Those assumptions break in D15 because the input plug faces downward. The revised main()
#      therefore:
#        1. checks both input connectors face downward,
#        2. extracts the plug sprite as a local grid,
#        3. flips that local grid vertically,
#        4. recomputes the transformed connector location before translation.
# d) Placement legality:
#    - Generation is still output-first. We first build a valid docked state using the flipped
#      plug orientation, then derive the input by vertically flipping the plug back.
#    - This guarantees no out-of-bounds paste, no non-connector overlap in the assembled output,
#      and preserves the invariant that the base remains the top-left corner after tight crop.
#    - When placing the two input sprites on the canvas, padding=1 prevents accidental 8-neighbor
#      contact that would merge them into one component.
# e) Visual observability:
#    - Across examples, both input connectors visibly sit on the same side of their bodies.
#    - In the output, the plug appears vertically flipped relative to its input appearance and
#      then attached above the base. Because the plug is forced to be vertically asymmetric,
#      this change is visible with only 2-3 examples.
# f) Visual cleanliness:
#    - Only one overlap layer is allowed in the final assembly: connector-vs-body at the docking
#      seam. The input canvas still contains just two separated sprites on a plain background.
# g) Color-change inferability:
#    - Colors are unchanged from the original task. ctrl (7) still marks docking bars and is
#      recolored to obj (4) after docking. No new color mapping logic is introduced.


BG = 1
OBJ = 4
CTRL = 7
_rng = random


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _free_side_direction(
    conn_pts: set[tuple[int, int]], body_pts: set[tuple[int, int]]
) -> str | None:
    up_free = all((int(r - 1), int(c)) not in body_pts for r, c in conn_pts)
    down_free = all((int(r + 1), int(c)) not in body_pts for r, c in conn_pts)
    body_above = all((int(r - 1), int(c)) in body_pts for r, c in conn_pts)
    body_below = all((int(r + 1), int(c)) in body_pts for r, c in conn_pts)
    if down_free and body_above and not up_free:
        return "down"
    if up_free and body_below and not down_free:
        return "up"
    return None


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _sprite_local_grid(grid: np.ndarray, pts: set[tuple[int, int]], bg: int = BG) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_of_pts(pts)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        out[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return out, (int(r0), int(c0))


def _grid_points(local_grid: np.ndarray, bg: int = BG) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(local_grid != int(bg))}


def _grid_connector(local_grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(local_grid == int(CTRL))}


def _render_assembled(base_grid: np.ndarray, plug_grid: np.ndarray, plug_top: int, plug_left: int) -> np.ndarray | None:
    base_pts = _grid_points(base_grid)
    plug_pts = _grid_points(plug_grid)
    shifted_plug = {(int(r + plug_top), int(c + plug_left)) for r, c in plug_pts}
    pts_all = base_pts | shifted_plug
    if not pts_all:
        return None

    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(BG), dtype=int)

    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(base_grid[int(r), int(c)])
    for r, c in plug_pts:
        rr = int(r + plug_top - rr0)
        cc = int(c + plug_left - cc0)
        val = int(plug_grid[int(r), int(c)])
        cur = int(out[rr, cc])
        if cur != int(BG) and val != int(BG):
            assert cur == int(CTRL) or val == int(CTRL)
        out[rr, cc] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    return _crop_non_bg(out, bg=BG)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, after vertically flipping the plug.

    Variant logic:
    - Both input connectors face downward (body directly above ctrl, no body directly below).
    - The base sprite is the one anchored at the grid top-left.
    - Vertically flip the plug sprite within its own bbox so its connector faces upward.
    - Translate the flipped plug so its connector sits one cell above the base connector.
    - Stamp, recolor ctrl->obj, and tight-crop.
    """
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != int(BG), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(CTRL)}

    connA = conn(A)
    connB = conn(B)
    assert connA and connB
    shpA, _ = _norm(connA)
    shpB, _ = _norm(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, conn_base = A, connA
        plug, conn_plug = B, connB
    else:
        base, conn_base = B, connB
        plug, conn_plug = A, connA

    base_body = base - conn_base
    plug_body = plug - conn_plug
    assert _free_side_direction(conn_base, base_body) == "down"
    assert _free_side_direction(conn_plug, plug_body) == "down"

    plug_grid, _plug_origin = _sprite_local_grid(g, plug, bg=BG)
    plug_flipped = np.flipud(plug_grid)
    conn_flip = _grid_connector(plug_flipped)
    body_flip = _grid_points(plug_flipped) - conn_flip
    assert conn_flip
    assert _free_side_direction(conn_flip, body_flip) == "up"

    _shape_flip, (pr0, pc0) = _norm(conn_flip)
    _shape_base, (dr0, dc0) = _norm(conn_base)

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)

    plug_shift_vals: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(plug_flipped != int(BG)):
        rr = int(r + tr)
        cc = int(c + tc)
        plug_shift_vals[(rr, cc)] = int(plug_flipped[int(r), int(c)])

    pts_all = base | set(plug_shift_vals)
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    base_coords = np.array(list(base), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(BG), dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for (r, c), val in plug_shift_vals.items():
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(BG) and val != int(BG):
            assert cur == int(CTRL) or val == int(CTRL)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    cropped = _crop_non_bg(out, bg=BG)
    if cropped is None:
        return np.full((0, 0), int(BG), dtype=int)
    return cropped


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
    # Step 1: build base body anchored at (0,0)
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

    base_grid = np.full((base_h, base_w), int(BG), dtype=int)
    base_grid[base] = int(OBJ)
    for c in conn_cols:
        base_grid[conn_row, c] = int(CTRL)
    if not is_contiguous(base_grid != int(BG), background=0, connectivity=8):
        return None

    # Step 2: build the docked plug orientation first (connector faces up).
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

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    dock_top = int(conn_row - 1)
    dock_left = int(dc_offset)
    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = int(dock_top + r)
            dc = int(dock_left + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == int(OBJ):
                if r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len:
                    continue
                return None

    plug_docked = np.full((plug_h, plug_w), int(BG), dtype=int)
    plug_docked[plug] = int(OBJ)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_docked[0, c] = int(CTRL)
        if plug_h < 2 or plug_docked[1, c] != int(OBJ):
            return None
    if not is_contiguous(plug_docked != int(BG), background=0, connectivity=8):
        return None
    if np.array_equal(plug_docked, np.flipud(plug_docked)):
        return None

    expected = _render_assembled(base_grid, plug_docked, dock_top, dock_left)
    if expected is None or expected.size == 0:
        return None

    # Step 3: derive input plug by flipping vertically, so both inputs face down.
    plug_input = np.flipud(plug_docked)
    if np.array_equal(plug_input, plug_docked):
        return None
    conn_input = _grid_connector(plug_input)
    body_input = _grid_points(plug_input) - conn_input
    if _free_side_direction(conn_input, body_input) != "down":
        return None
    if _free_side_direction(_grid_connector(base_grid), _grid_points(base_grid) - _grid_connector(base_grid)) != "down":
        return None

    base_sprite = _crop_non_bg(base_grid, bg=BG)
    plug_sprite = _crop_non_bg(plug_input, bg=BG)
    if base_sprite is None or plug_sprite is None:
        return None
    if int(np.sum(base_sprite != int(BG))) < 5 or int(np.sum(plug_sprite != int(BG))) < 5:
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(3, max(bw, pw)), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(BG))
    try:
        pr, pc = random_free_location(
            canvas,
            plug_sprite,
            transparent=int(BG),
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    if pr + pc == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=int(BG))

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    if np.sum(canvas == int(BG)) <= canvas.size // 2:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
