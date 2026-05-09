# Combo variant of 20270e3b (index 5)
# combo_spec: D4/0+D13/0
# Name: 嵌入式重叠对接
#
# Design analysis
# a) Concrete behavior / trace
#    - Original 20270e3b translates the plug so its ctrl bar lands exactly one row above the base
#      ctrl bar. The merged shape usually grows outside the base bbox, and non-connector overlap is
#      forbidden.
#    - This combo keeps the same two-sprite / same-color regime, but changes both the docking
#      geometry and the overlap rule at once:
#        1. The base has an internal socket directly below its ctrl bar.
#        2. The plug is translated so both ctrl bars coincide on the same row.
#        3. The plug body drops into the socket, but some deeper body pixels are also allowed to
#           cover internal base ribs. Plug stamping wins on overlap.
#    - Small trace:
#      original: base ctrl at row 4, plug ctrl moves to row 3, only the seam may overlap.
#      combo:    base ctrl at row 3, plug ctrl also moves to row 3, plug body fills the cavity in
#                rows 4..k and additionally hides 1-3 rib pixels that were visible inside the cavity.
#      The output keeps the base bbox, turns ctrl->obj, and removes the visible rib because the plug
#      overwrites it.
#
# b) Complexity calibration
#    - A pure D4 cavity would make the plug body equal the socket exactly; that would not exercise
#      D13. A pure D13 overlap could degenerate into "just overwrite deeper rows", which would not
#      look embedded. The generator therefore forces both signals:
#        * socket_fill > 0: the plug covers background cavity pixels below the base connector.
#        * deep_overlap > 0: the plug also overwrites at least one deeper obj rib pixel.
#    - Connector length stays in 1..3 and the rib is only 1..3 pixels, keeping the rule legible.
#
# c) Hidden assumptions from the original solver
#    - `tr = (dr0 - 1) - pr0` is invalid now; connectors must align on the same row.
#    - `assert cur == ctrl or val == ctrl` on overlap is invalid now; body-body overlap is required.
#    - The original task expected the union bbox to extend from the base. Here a legal assembled
#      state stays fully inside the base bbox, so the solver now asserts that invariant instead.
#    - The D4-only equality `socket == plug_body` is also too strict now. In this combo, the correct
#      relation is `socket ⊆ plug_body`, and the remainder of the plug body must land on base obj
#      cells that get overwritten.
#
# d) Placement legality
#    - The generator is output-first. It builds the docked plug body, removes a boundary-connected
#      rib subset to obtain the socket, wraps the socket with a thin base shell, then separates the
#      two sprites and places them apart on the input canvas.
#    - Placement still guarantees exactly two 8-connected components in the input and keeps a
#      Manhattan/box gap of 1..5 so the sprites are clearly separate before docking.
#
# e) Visual observability
#    - Across examples, humans can see three consistent cues:
#        * the base contains a cavity under the ctrl mouth,
#        * the loose plug has the same outer body shape as that cavity region,
#        * some cavity cells already contain obj ribs that disappear after insertion.
#      That makes the combo visually distinguishable from both the original task and plain D4.
#
# f) Visual cleanliness
#    - Only three colors appear in the input and only two in the output.
#    - The rib is thin and connected to the shell, avoiding floating noise pixels.
#    - The plug never extends outside the base bbox after docking, so the merged output stays compact.
#
# g) New solver assumptions that the generator must satisfy
#    - The base has exactly one 4-connected socket component reachable from directly below every base
#      connector cell.
#    - The full socket is covered by the shifted plug body.
#    - At least one shifted plug body pixel overwrites a deeper base obj rib.
#    - Every shifted plug body pixel lies strictly inside the base bbox.
#      The generator constructs all samples to satisfy these assumptions.

from __future__ import annotations

import random as _rng
import sys

import numpy as np

try:
    from arc_utils import bbox_coords, is_contiguous, label_mask, paste
except ImportError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_coords, is_contiguous, label_mask, paste


BG = 1
OBJ = 4
CTRL = 7


def _component_points(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    lab, n = label_mask(mask, connectivity=8)
    comps = []
    for cid in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(cid))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    return comps


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn(sprite: set[tuple[int, int]], grid: np.ndarray) -> set[tuple[int, int]]:
    return {(r, c) for (r, c) in sprite if int(grid[int(r), int(c)]) == int(CTRL)}


def _norm(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    return {(int(r - r0), int(c - c0)) for r, c in pts}, (int(r0), int(c0))


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        out[r, c] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return int(_gap1(a[0], a[2], b[0], b[2]) + _gap1(a[1], a[3], b[1], b[3]))


def _place_with_gap(canvas: np.ndarray, patch: np.ndarray, bg: int = BG) -> tuple[int, int] | None:
    h, w = patch.shape
    occ = canvas != int(bg)
    patch_mask = patch != int(bg)
    dil = _dilate4(occ)
    base_bbox = bbox_coords(np.argwhere(occ))
    patch_pts = np.argwhere(patch_mask)
    candidates: list[tuple[int, int]] = []
    for r0 in range(0, canvas.shape[0] - h + 1):
        for c0 in range(0, canvas.shape[1] - w + 1):
            if r0 == 0 and c0 == 0:
                continue
            view = occ[r0 : r0 + h, c0 : c0 + w]
            if bool((view & patch_mask).any()):
                continue
            touch = dil[r0 : r0 + h, c0 : c0 + w]
            if bool((touch & patch_mask).any()):
                continue
            bbox = (
                int(r0 + patch_pts[:, 0].min()),
                int(c0 + patch_pts[:, 1].min()),
                int(r0 + patch_pts[:, 0].max()),
                int(c0 + patch_pts[:, 1].max()),
            )
            if 1 <= _bbox_gap(base_bbox, bbox) <= 5:
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return _rng.choice(candidates)


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray | None = None,
) -> None:
    allowed = np.ones_like(mask, dtype=bool) if allowed is None else np.array(allowed, dtype=bool)
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and allowed[rr, cc] and not mask[rr, cc]:
                nbrs.append((rr, cc))
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = int(r + dr), int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_boundary_cell(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = int(r + dr), int(c + dc)
        if not (0 <= rr < h and 0 <= cc < w) or not bool(mask[rr, cc]):
            return True
    return False


def _select_overlap_path(body: np.ndarray) -> set[tuple[int, int]] | None:
    body_pts = {(int(r), int(c)) for r, c in np.argwhere(body)}
    if len(body_pts) < 6:
        return None

    starts = [(r, c) for r, c in body_pts if r > 0 and _is_boundary_cell(body, r, c)]
    if not starts:
        return None

    max_len = min(3, len(body_pts) - 4)
    if max_len < 1:
        return None

    for _ in range(80):
        start = _rng.choice(starts)
        target_len = _rng.randint(1, max_len)
        path = {start}
        frontier = [start]
        while len(path) < target_len and frontier:
            cell = _rng.choice(frontier)
            cand = []
            for rr, cc in _neighbors4(body.shape, cell[0], cell[1]):
                if rr == 0:
                    continue
                if (rr, cc) in body_pts and (rr, cc) not in path:
                    cand.append((rr, cc))
            if not cand:
                frontier.remove(cell)
                continue
            nxt = _rng.choice(cand)
            path.add(nxt)
            frontier.append(nxt)

        socket = np.array(body, copy=True)
        for r, c in path:
            socket[r, c] = False

        if not socket[0].any():
            continue
        if not is_contiguous(socket, background=0, connectivity=4):
            continue
        if not any(_is_boundary_cell(body, r, c) for r, c in path):
            continue
        ok = True
        for r, c in path:
            if not any(bool(socket[rr, cc]) for rr, cc in _neighbors4(body.shape, r, c)):
                ok = False
                break
        if ok:
            return {(int(r), int(c)) for r, c in path}
    return None


def _socket_mask_from_base(base_grid: np.ndarray, conn_row: int, conn_cols: list[int]) -> np.ndarray | None:
    hole_mask = base_grid == int(BG)
    hole_lab, _ = label_mask(hole_mask, connectivity=4)
    socket_ids = set()
    for c in conn_cols:
        rr = int(conn_row + 1)
        if not (0 <= rr < base_grid.shape[0] and 0 <= c < base_grid.shape[1]):
            return None
        if not bool(hole_mask[rr, c]):
            return None
        socket_ids.add(int(hole_lab[rr, c]))
    if len(socket_ids) != 1:
        return None
    sid = socket_ids.pop()
    if sid <= 0:
        return None
    return hole_lab == int(sid)


def _fusion_metrics_from_assembled(base_grid: np.ndarray, plug_grid: np.ndarray) -> dict[str, int]:
    conn_pts = np.argwhere(base_grid == int(CTRL))
    assert conn_pts.size > 0
    conn_row = int(conn_pts[:, 0].min())
    conn_cols = sorted(int(c) for c in conn_pts[:, 1])
    socket_mask = _socket_mask_from_base(base_grid, conn_row, conn_cols)
    assert socket_mask is not None

    plug_body = plug_grid == int(OBJ)
    base_body = base_grid == int(OBJ)
    return {
        "socket_fill": int(np.sum(socket_mask & plug_body)),
        "deep_overlap": int(np.sum(base_body & plug_body)),
        "plug_body_pixels": int(np.sum(plug_body)),
        "socket_pixels": int(np.sum(socket_mask)),
    }


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    comps = _component_points(g != int(BG))
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    connA = _conn(A, g)
    connB = _conn(B, g)
    assert connA and connB
    assert len({r for r, _c in connA}) == 1
    assert len({r for r, _c in connB}) == 1

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
    for r, c in conn_base:
        assert (int(r - 1), int(c)) in base_body
        assert (int(r + 1), int(c)) not in base_body
    for r, c in conn_plug:
        assert (int(r + 1), int(c)) in plug_body
        assert (int(r - 1), int(c)) not in plug_body

    _sp, (pr0, pc0) = _norm(conn_plug)
    _sd, (dr0, dc0) = _norm(conn_base)
    tr = int(dr0 - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
    plug_shift_body = plug_shift - conn_base

    br0, bc0, br1, bc1 = _bbox_of_pts(base)
    rr0, cc0, rr1, cc1 = _bbox_of_pts(set(base) | set(plug_shift))
    assert (rr0, cc0, rr1, cc1) == (br0, bc0, br1, bc1)

    base_sprite = np.full((int(br1 - br0 + 1), int(bc1 - bc0 + 1)), int(BG), dtype=int)
    for r, c in base:
        base_sprite[int(r - br0), int(c - bc0)] = int(g[int(r), int(c)])

    conn_cols = sorted(int(c) for _r, c in conn_base)
    socket_mask = _socket_mask_from_base(base_sprite, int(dr0 - br0), [int(c - bc0) for c in conn_cols])
    assert socket_mask is not None
    socket = {
        (int(br0 + rr), int(bc0 + cc))
        for rr, cc in np.argwhere(socket_mask)
    }
    assert socket
    for r, c in socket:
        assert br0 < r < br1
        assert bc0 < c < bc1

    assert socket <= plug_shift_body
    overlap_cells = {(r, c) for (r, c) in plug_shift_body if (r, c) not in socket}
    assert overlap_cells
    assert all((r, c) in base_body for (r, c) in overlap_cells)
    assert all((r, c) not in base_body for (r, c) in socket)
    assert all(br0 < r < br1 and bc0 < c < bc1 for r, c in plug_shift_body)

    return {
        "grid": g,
        "base": base,
        "plug": plug,
        "conn_base": conn_base,
        "plug_shift": plug_shift,
        "shift": (tr, tc),
        "base_bbox": (br0, bc0, br1, bc1),
        "socket": socket,
        "overlap_cells": overlap_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert the plug into the base's internal socket, allowing deeper body overwrite.

    The connector bars still determine the unique horizontal alignment, but the plug connector lands
    on the same row as the base connector. The plug body fills the cavity and may cover internal ribs
    of the base; plug pixels have priority wherever they overlap.
    """
    info = _analyze_input(grid)
    g = info["grid"]
    base = info["base"]
    plug_shift = info["plug_shift"]
    tr, tc = info["shift"]
    br0, bc0, br1, bc1 = info["base_bbox"]

    out = np.full((int(br1 - br0 + 1), int(bc1 - bc0 + 1)), int(BG), dtype=int)
    for r, c in base:
        out[int(r - br0), int(c - bc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        out[int(r - br0), int(c - bc0)] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    return _crop_non_bg(out, BG)


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify that a sample truly uses both the embedded socket and the overwrite rule."""
    info = _analyze_input(grid)
    return {
        "socket_fill": len(info["socket"]),
        "deep_overlap": len(info["overlap_cells"]),
        "plug_body": int(len(info["socket"]) + len(info["overlap_cells"])),
    }


def _build_plug() -> tuple[np.ndarray, np.ndarray, int, int, set[tuple[int, int]]] | None:
    conn_len = _rng.randint(1, 3)
    body_h = _rng.randint(3, 5)
    body_w = _rng.randint(max(2, conn_len), min(6, conn_len + 3))
    body = np.zeros((body_h, body_w), dtype=bool)
    conn_c0 = _rng.randint(0, body_w - conn_len)
    body[0, conn_c0 : conn_c0 + conn_len] = True

    allowed = np.ones_like(body, dtype=bool)
    allowed[0, :] = False
    allowed[0, conn_c0 : conn_c0 + conn_len] = True
    target = _rng.randint(max(conn_len + 4, 6), min(int(body.size), 12))
    frontier = [(0, c) for c in range(conn_c0, conn_c0 + conn_len)]
    _grow_shape(body, target, frontier, allowed=allowed)

    if not body[1:].any():
        return None
    if not is_contiguous(body, background=0, connectivity=4):
        return None

    overlap = _select_overlap_path(body)
    if overlap is None:
        return None

    socket = np.array(body, copy=True)
    for r, c in overlap:
        socket[r, c] = False
    if not socket[0].any():
        return None
    if not is_contiguous(socket, background=0, connectivity=4):
        return None

    plug = np.full((int(body_h + 1), int(body_w)), int(BG), dtype=int)
    plug[1:][body] = int(OBJ)
    plug[0, conn_c0 : conn_c0 + conn_len] = int(CTRL)
    if not is_contiguous(plug != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(plug != int(BG))) < 5 or int(np.sum(plug != int(BG))) > 29:
        return None
    return plug, body, conn_c0, conn_len, overlap


def _build_base_and_expected(
    body: np.ndarray,
    conn_c0: int,
    conn_len: int,
    overlap: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    body_h, body_w = body.shape
    top_pad, left_pad, right_pad, bottom_pad = 2, 1, 1, 1
    H = int(top_pad + body_h + bottom_pad)
    W = int(left_pad + body_w + right_pad)
    if not (5 <= H <= 8 and 3 <= W <= 12):
        return None

    socket = np.zeros((H, W), dtype=bool)
    socket[top_pad : top_pad + body_h, left_pad : left_pad + body_w] = body
    for r, c in overlap:
        socket[int(top_pad + r), int(left_pad + c)] = False

    if not is_contiguous(socket, background=0, connectivity=4):
        return None

    shell = _dilate4(socket) & (~socket)
    conn_row = int(top_pad - 1)
    conn_cols = [int(left_pad + c) for c in range(conn_c0, conn_c0 + conn_len)]
    shell[int(top_pad - 2), conn_cols] = True

    base = np.full((H, W), int(BG), dtype=int)
    base[shell] = int(OBJ)
    base[conn_row, conn_cols] = int(CTRL)
    if not is_contiguous(base != int(BG), background=0, connectivity=8):
        return None

    plug_assembled = np.full((H, W), int(BG), dtype=int)
    plug_assembled[conn_row, conn_cols] = int(CTRL)
    plug_assembled[top_pad : top_pad + body_h, left_pad : left_pad + body_w][body] = int(OBJ)
    if not is_contiguous(plug_assembled != int(BG), background=0, connectivity=8):
        return None

    metrics = _fusion_metrics_from_assembled(base, plug_assembled)
    if metrics["socket_fill"] <= 0 or metrics["deep_overlap"] <= 0:
        return None
    if metrics["socket_fill"] + metrics["deep_overlap"] != metrics["plug_body_pixels"]:
        return None

    expected = np.array(base, copy=True)
    mask = plug_assembled != int(BG)
    expected[mask] = plug_assembled[mask]
    expected[expected == int(CTRL)] = int(OBJ)

    base_sprite = _crop_non_bg(base, BG)
    expected = _crop_non_bg(expected, BG)
    if int(np.sum(base_sprite != int(BG))) < 5 or int(np.sum(base_sprite != int(BG))) > 29:
        return None
    if int(np.sum(expected != int(BG))) > 29:
        return None
    return base_sprite, expected, metrics


def _place_input(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray | None:
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    min_h = max(7, bh + 1, ph + 1)
    min_w = max(5, bw + pw)
    if min_h > 11 or min_w > 13:
        return None

    canvas_h = _rng.randint(min_h, 11)
    canvas_w = _rng.randint(min_w, 13)
    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    pos = _place_with_gap(canvas, plug_sprite, BG)
    if pos is None:
        return None
    paste(canvas, plug_sprite, pos[0], pos[1], transparent=BG)

    lab, n = label_mask(canvas != int(BG), connectivity=8)
    if int(n) != 2:
        return None
    if np.sum(canvas == int(BG)) <= canvas.size // 2:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    built_plug = _build_plug()
    if built_plug is None:
        return None
    plug_sprite, body, conn_c0, conn_len, overlap = built_plug

    built_base = _build_base_and_expected(body, conn_c0, conn_len, overlap)
    if built_base is None:
        return None
    base_sprite, expected, metrics = built_base
    if metrics["socket_fill"] <= metrics["deep_overlap"]:
        return None

    canvas = _place_input(base_sprite, plug_sprite)
    if canvas is None:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None

    sample_metrics = fusion_metrics(canvas)
    if sample_metrics["socket_fill"] <= 0 or sample_metrics["deep_overlap"] <= 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(800):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 800 attempts")
