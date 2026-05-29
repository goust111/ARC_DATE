# Combo variant of 20270e3b (index 12)
# combo_spec: D2/2+D9/2
# Name: L形角落T形连接器
#
# Design analysis
# ===============
# a) Concrete form and behavior trace
# - Original task: both sprites carry identical 1D horizontal ctrl bars. The plug is
#   translated so its bar sits one row above the base bar, then all ctrl pixels are
#   recolored to obj and the merged result is tightly cropped.
# - This combo keeps the two-sprite / recolor / tight-crop scaffold, but changes both
#   the connector geometry and the docking rule:
#   1. Each connector is now a 4-pixel 2D T shape rather than a 1-row bar.
#   2. The two connectors are not identical. They are complementary T shapes living
#      in the same small bbox.
#   3. Docking is corner-style: the base connector sits at the base sprite's lower-right
#      corner and the plug connector sits at the plug sprite's upper-left corner. When
#      the connector bboxes are aligned, the two T shapes fill a solid rectangle.
# - We use two controlled families:
#   * horizontal family: a 2x3 bbox filled by top-T + bottom-T
#   * vertical family:   a 3x2 bbox filled by left-T + right-T
# - Small trace:
#   old bar: {(0,0),(0,1),(0,2)} + same bar translated by (-1,0) -> a horizontal seam.
#   new T:   base {(0,1),(1,0),(1,1),(1,2)} and plug {(0,0),(0,1),(0,2),(1,1)}.
#   Align their connector bboxes directly (no extra row offset): the union becomes the
#   full 2x3 rectangle. This is visibly different from the original one-row connector.
#
# b) Complexity calibration
# - The connector family is small on purpose: only two complementary T-pairs. This keeps
#   the puzzle inferable from a few examples while still forcing 2D shape recognition.
# - Restricting to 4-pixel T shapes avoids an over-large family of arbitrary 2D marks and
#   makes the "two different T's combine into a rectangle" cue stable across examples.
#
# c) Broken assumptions from the original solver
# - `shpA == shpB` is no longer valid. The new solver instead checks an allowed
#   complementary mapping from base-shape to plug-shape.
# - `tr = (dr0 - 1) - pr0` hard-coded vertical-above docking. Here the connector bboxes
#   are aligned directly: `tr = dr0 - pr0`, `tc = dc0 - pc0`.
# - The old overlap rule allowed ctrl-vs-obj overlap along a seam. Here valid overlap is
#   strictly ctrl-vs-ctrl inside the shared connector rectangle.
# - The original "body above / body below" checks become "base body approaches from the
#   north/west, plug body approaches from the south/east", which is the actual corner cue.
#
# d) Placement legality
# - Generator builds the assembled logic first. Base connectors are placed at the local
#   lower-right corner; plug connectors at the local upper-left corner.
# - Bodies are forbidden from occupying the connector rectangle itself. Therefore, after
#   alignment, any overlap outside the connector rectangle is impossible by construction.
# - Random input placement uses `padding=1` with 8-connectivity to keep the two sprites as
#   separate connected components in the input.
#
# e/f) Visual observability and cleanliness
# - Every connector has bbox height and width > 1, so the D9-style 2D connector change is
#   obvious at a glance.
# - The corner docking is also directly visible: the ctrl region is not a horizontal seam;
#   it is a compact corner block where two different T shapes fuse into a rectangle.
# - Only three colors appear in the input and only ctrl changes color in the output, so the
#   examples stay visually clean.
#
# g) Non-degeneracy target
# - Generated samples must satisfy:
#   * connector bbox is 2x3 or 3x2, never 1xL;
#   * base/plug connector shapes differ but are complementary;
#   * the solver uses zero extra docking offset beyond bbox alignment.
#   These conditions make the combo behavior materially different from the original task.

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


BG = 1
OBJ = 4
CTRL = 7

TOP_T = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
BOTTOM_T = frozenset({(0, 1), (1, 0), (1, 1), (1, 2)})
RIGHT_T = frozenset({(0, 1), (1, 0), (1, 1), (2, 1)})
LEFT_T = frozenset({(0, 0), (1, 0), (1, 1), (2, 0)})

BASE_TO_PLUG = {
    BOTTOM_T: TOP_T,
    RIGHT_T: LEFT_T,
}


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_points(
    pts: set[tuple[int, int]],
) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (int(r0), int(c0))


def _shape_hw(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    h = max(r for r, _c in shape) + 1
    w = max(c for _r, c in shape) + 1
    return int(h), int(w)


def _connector_info(
    grid: np.ndarray, sprite: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], frozenset[tuple[int, int]], tuple[int, int]]:
    conn = {(int(r), int(c)) for r, c in sprite if int(grid[int(r), int(c)]) == CTRL}
    assert len(conn) == 4
    shape, origin = _normalize_points(conn)
    assert shape in BASE_TO_PLUG or shape in BASE_TO_PLUG.values()
    return conn, shape, origin


def _full_rect_cells(origin: tuple[int, int], shape: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0 = map(int, origin)
    h, w = _shape_hw(shape)
    return {(int(r0 + r), int(c0 + c)) for r in range(h) for c in range(w)}


def _validate_corner_attachment(
    conn: set[tuple[int, int]], body: set[tuple[int, int]], *, base_role: bool
) -> None:
    if base_role:
        need_dirs = [(-1, 0), (0, -1)]
        forbid_dirs = [(1, 0), (0, 1)]
    else:
        need_dirs = [(1, 0), (0, 1)]
        forbid_dirs = [(-1, 0), (0, -1)]

    touched = {d: False for d in need_dirs}
    for r, c in conn:
        for dr, dc in forbid_dirs:
            assert (int(r + dr), int(c + dc)) not in body
        for dr, dc in need_dirs:
            if (int(r + dr), int(c + dc)) in body:
                touched[(dr, dc)] = True
    assert all(touched.values())


def _crop_non_bg(grid: np.ndarray) -> np.ndarray | None:
    nz = np.argwhere(grid != BG)
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _connector_points(grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid == CTRL)}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != BG, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        if pts:
            comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, plug = A, B
    else:
        base, plug = B, A

    conn_base, shape_base, origin_base = _connector_info(g, base)
    conn_plug, shape_plug, origin_plug = _connector_info(g, plug)
    assert shape_base in BASE_TO_PLUG
    assert BASE_TO_PLUG[shape_base] == shape_plug

    base_body = base - conn_base
    plug_body = plug - conn_plug
    _validate_corner_attachment(conn_base, base_body, base_role=True)
    _validate_corner_attachment(conn_plug, plug_body, base_role=False)

    tr = int(origin_base[0] - origin_plug[0])
    tc = int(origin_base[1] - origin_plug[1])
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
    shifted_conn = {(int(r + tr), int(c + tc)) for r, c in conn_plug}

    full_rect = _full_rect_cells(origin_base, shape_base)
    assert conn_base | shifted_conn == full_rect

    rr0, cc0, rr1, cc1 = _bbox_of_pts(base | plug_shift)
    base_coords = np.array(list(base), dtype=int)
    assert rr0 == int(base_coords[:, 0].min())
    assert cc0 == int(base_coords[:, 1].min())

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        src_r = int(r - tr)
        src_c = int(c - tc)
        val = int(g[src_r, src_c])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != BG:
            assert (int(r), int(c)) in conn_base
            assert (int(src_r), int(src_c)) in conn_plug
            assert cur == CTRL and val == CTRL
        out[int(r - rr0), int(c - cc0)] = val

    out[out == CTRL] = OBJ
    nz = np.argwhere(out != BG)
    if nz.size == 0:
        return np.full((0, 0), BG, dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    *,
    allowed: np.ndarray,
) -> None:
    count = int(mask.sum())
    h, w = mask.shape
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if mask[nr, nc] or not bool(allowed[nr, nc]):
                continue
            nbrs.append((nr, nc))
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
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        rc = (int(r), int(c))
        if rc in keep:
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


def _monotone_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, start)
    er, ec = map(int, end)
    path = [(int(r), int(c))]
    while (r, c) != (er, ec):
        options = []
        if r < er:
            options.append((r + 1, c))
        elif r > er:
            options.append((r - 1, c))
        if c < ec:
            options.append((r, c + 1))
        elif c > ec:
            options.append((r, c - 1))
        r, c = _rng.choice(options)
        path.append((int(r), int(c)))
    return path


def _candidate_adjacent_cells(
    shape: frozenset[tuple[int, int]],
    conn_r0: int,
    conn_c0: int,
    *,
    direction: tuple[int, int],
    allowed: np.ndarray,
) -> list[tuple[int, int]]:
    out = set()
    dr, dc = map(int, direction)
    h, w = allowed.shape
    for r, c in shape:
        rr = int(conn_r0 + r + dr)
        cc = int(conn_c0 + c + dc)
        if 0 <= rr < h and 0 <= cc < w and bool(allowed[rr, cc]):
            out.add((rr, cc))
    return sorted(out)


def _build_base_sprite(shape: frozenset[tuple[int, int]], height: int, width: int) -> np.ndarray | None:
    h = int(height)
    w = int(width)
    shape_h, shape_w = _shape_hw(shape)
    conn_r0 = h - shape_h
    conn_c0 = w - shape_w

    allowed = np.ones((h, w), dtype=bool)
    allowed[conn_r0:, conn_c0:] = False
    north = _candidate_adjacent_cells(shape, conn_r0, conn_c0, direction=(-1, 0), allowed=allowed)
    west = _candidate_adjacent_cells(shape, conn_r0, conn_c0, direction=(0, -1), allowed=allowed)
    if not north or not west:
        return None

    keep = set(_monotone_path(_rng.choice(north), (0, 0)))
    keep |= set(_monotone_path(_rng.choice(west), (0, 0)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in keep:
        mask[int(r), int(c)] = True

    max_target = min(29 - len(shape), int(allowed.sum()))
    min_target = max(len(keep), 5 - len(shape))
    if max_target < min_target:
        return None
    target = _rng.randint(min_target, max_target)
    _grow_shape(mask, target, list(keep), allowed=allowed)
    _thin(mask, keep=keep)

    grid = np.full((h, w), BG, dtype=int)
    grid[mask] = OBJ
    for r, c in shape:
        grid[int(conn_r0 + r), int(conn_c0 + c)] = CTRL

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    if int(np.sum(grid != BG)) < 5:
        return None
    return _crop_non_bg(grid)


def _build_plug_sprite(shape: frozenset[tuple[int, int]], height: int, width: int) -> np.ndarray | None:
    h = int(height)
    w = int(width)
    shape_h, shape_w = _shape_hw(shape)

    allowed = np.ones((h, w), dtype=bool)
    allowed[:shape_h, :shape_w] = False
    south = _candidate_adjacent_cells(shape, 0, 0, direction=(1, 0), allowed=allowed)
    east = _candidate_adjacent_cells(shape, 0, 0, direction=(0, 1), allowed=allowed)
    if not south or not east:
        return None

    target_corner = (h - 1, w - 1)
    keep = set(_monotone_path(_rng.choice(south), target_corner))
    keep |= set(_monotone_path(_rng.choice(east), target_corner))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in keep:
        mask[int(r), int(c)] = True

    max_target = min(29 - len(shape), int(allowed.sum()))
    min_target = max(len(keep), 5 - len(shape))
    if max_target < min_target:
        return None
    target = _rng.randint(min_target, max_target)
    _grow_shape(mask, target, list(keep), allowed=allowed)
    _thin(mask, keep=keep)

    grid = np.full((h, w), BG, dtype=int)
    grid[mask] = OBJ
    for r, c in shape:
        grid[int(r), int(c)] = CTRL

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    if int(np.sum(grid != BG)) < 5:
        return None
    return _crop_non_bg(grid)


def _assemble_output(base_grid: np.ndarray, plug_grid: np.ndarray) -> np.ndarray:
    base_conn = _connector_points(base_grid)
    plug_conn = _connector_points(plug_grid)
    shape_base, origin_base = _normalize_points(base_conn)
    shape_plug, origin_plug = _normalize_points(plug_conn)
    assert shape_base in BASE_TO_PLUG
    assert BASE_TO_PLUG[shape_base] == shape_plug

    tr = int(origin_base[0] - origin_plug[0])
    tc = int(origin_base[1] - origin_plug[1])
    shifted_plug = {(int(r + tr), int(c + tc)) for r, c in np.argwhere(plug_grid != BG)}
    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_grid != BG)}
    full_rect = _full_rect_cells(origin_base, shape_base)
    assert _connector_points(base_grid) | {(int(r + tr), int(c + tc)) for r, c in plug_conn} == full_rect

    rr0, cc0, rr1, cc1 = _bbox_of_pts(base_pts | shifted_plug)
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)
    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(base_grid[int(r), int(c)])
    for r, c in np.argwhere(plug_grid != BG):
        rr = int(r + tr - rr0)
        cc = int(c + tc - cc0)
        cur = int(out[rr, cc])
        val = int(plug_grid[int(r), int(c)])
        if cur != BG:
            assert cur == CTRL and val == CTRL
        out[rr, cc] = val

    out[out == CTRL] = OBJ
    return out


def _valid_docking(base_grid: np.ndarray, plug_grid: np.ndarray) -> bool:
    try:
        base_conn, shape_base, origin_base = _connector_info(
            base_grid, {(int(r), int(c)) for r, c in np.argwhere(base_grid != BG)}
        )
        plug_conn, shape_plug, origin_plug = _connector_info(
            plug_grid, {(int(r), int(c)) for r, c in np.argwhere(plug_grid != BG)}
        )
        if shape_base not in BASE_TO_PLUG or BASE_TO_PLUG[shape_base] != shape_plug:
            return False

        base_body = {(int(r), int(c)) for r, c in np.argwhere(base_grid == OBJ)}
        plug_body = {(int(r), int(c)) for r, c in np.argwhere(plug_grid == OBJ)}
        _validate_corner_attachment(base_conn, base_body, base_role=True)
        _validate_corner_attachment(plug_conn, plug_body, base_role=False)

        tr = int(origin_base[0] - origin_plug[0])
        tc = int(origin_base[1] - origin_plug[1])
        shifted_conn = {(int(r + tr), int(c + tc)) for r, c in plug_conn}
        if base_conn | shifted_conn != _full_rect_cells(origin_base, shape_base):
            return False

        for r, c in np.argwhere(plug_grid != BG):
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < base_grid.shape[0] and 0 <= cc < base_grid.shape[1]:
                if int(base_grid[rr, cc]) == OBJ and int(plug_grid[int(r), int(c)]) == OBJ:
                    return False
                if int(base_grid[rr, cc]) == CTRL and int(plug_grid[int(r), int(c)]) != CTRL:
                    return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    base_shape = _rng.choice(list(BASE_TO_PLUG))
    plug_shape = BASE_TO_PLUG[base_shape]
    shape_h, shape_w = _shape_hw(base_shape)

    for _ in range(20):
        base_h = _rng.randint(shape_h + 1, 6)
        base_w = _rng.randint(shape_w + 1, 7)
        plug_h = _rng.randint(shape_h + 1, 6)
        plug_w = _rng.randint(shape_w + 1, 7)
        out_h = base_h + plug_h - shape_h
        out_w = base_w + plug_w - shape_w
        if 5 <= out_h <= 8 and 5 <= out_w <= 12:
            break
    else:
        return None

    base_sprite = _build_base_sprite(base_shape, base_h, base_w)
    if base_sprite is None:
        return None
    plug_sprite = _build_plug_sprite(plug_shape, plug_h, plug_w)
    if plug_sprite is None:
        return None
    if not _valid_docking(base_sprite, plug_sprite):
        return None

    expected = _assemble_output(base_sprite, plug_sprite)
    if expected.size == 0 or not np.any(expected != BG):
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    min_canvas_h = max(7, max(bh, ph) + 2)
    min_canvas_w = max(6, max(bw, pw) + 2)
    if min_canvas_h > 11 or min_canvas_w > 13:
        return None
    canvas_h = _rng.randint(min_canvas_h, 11)
    canvas_w = _rng.randint(min_canvas_w, 13)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    try:
        pr, pc = random_free_location(
            canvas,
            plug_sprite,
            transparent=BG,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=BG)

    lab, n = label_mask(canvas != BG, connectivity=8)
    if int(n) != 2:
        return None
    if np.sum(canvas == BG) <= canvas.size // 2:
        return None

    try:
        got = main(canvas)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
