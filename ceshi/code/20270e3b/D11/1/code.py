# Variant of 20270e3b
# Dimension: D11 — 对接时的旋转/翻转 (transform)
# Change: plug仅做平移对接，无旋转或翻转... → plug需水平翻转后对接

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


# Design analysis for the D11 variant
# -----------------------------------
# variant_spec:
#   unchanged_core:
#     colors: bg=1, obj=4, ctrl=7
#     scene: exactly 2 foreground 8-connected sprites; base remains the sprite whose
#       bbox touches the grid origin; output remains a tight crop of the merged result.
#     connector: each sprite has one horizontal ctrl bar of length 1-3; base connector
#       has body above and free space below; plug connector has body below and free
#       space above.
#   d11_change:
#     original: move plug by translation so its connector sits one row above the
#       base connector.
#     new: horizontally reflect the entire plug sprite first, then translate it to
#       the same docking position.
#   behavior_trace:
#     small_example:
#       base_bbox: width 6 with connector on its right edge at cols 4-5
#       input_plug: width 4 with connector on its left edge at cols 0-1
#       old_logic: direct translation aligns cols 0-1 to 4-5; the plug body stays on
#         the wrong side and yields a different merged silhouette.
#       new_logic: fliplr(input_plug) moves its connector to cols 2-3; then translation
#         aligns those cols to 4-5, placing the mirrored body on the intended side.
#     visible_difference:
#       the plug is deliberately made horizontally asymmetric, and the connector is
#       forced onto one bbox edge in the docked state. Therefore the input shows the
#       connector on the opposite edge from the output-side docking pose.
#   complexity_calibration:
#     controls:
#       - connector side is only left or right, never an arbitrary orientation
#       - transform family is only one fixed action: horizontal reflection
#       - connector length still 1-3 and shared by both sprites
#     anti_degeneracy:
#       - reject plugs whose tight sprite is horizontally symmetric
#       - reject samples where docking with and without reflection produces the same
#         output, so the variant cannot collapse back to the original task
#   implicit_assumption_audit:
#     original_solver_dependencies:
#       - connector matching used identical normalized ctrl bars only; this still
#         holds after fliplr because a horizontal bar stays a horizontal bar
#       - plug placement used the original connector anchor directly; this no longer
#         holds, so the new solver must recompute the connector anchor after fliplr
#       - overlap checks assumed ctrl-vs-obj collisions only at the seam; this still
#         holds, but now it must be checked against mirrored plug pixels
#       - base staying at output top-left relied on the plug never extending left/up;
#         the generator now enforces that for the reflected docking pose
#   placement_legality:
#     generator_strategy:
#       - build the base sprite first
#       - build the *docked* plug in its post-flip orientation
#       - mirror that docked plug to obtain the separated input plug
#       - place base at (0,0) and place the mirrored input plug elsewhere with
#         bbox gap 1-5 and no 8-neighborhood contact
#     safety:
#       - final input is oracle-checked by the new solver
#       - exact-2-component structure is preserved by the placement search
#   visual_observability:
#     - base connector is forced to lie on the left or right edge of the base bbox
#     - docked plug connector is forced onto the same edge, so the separated input
#       plug necessarily shows the connector on the opposite edge
#     - plug masks are thin and asymmetric, making the mirror visually obvious in
#       2-3 examples
#   cleanliness:
#     - still only two objects and one connector per object
#     - no extra marker colors or nested overlays are introduced
#   color_inference:
#     - unchanged from the original task: ctrl disappears into obj after docking


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _extract_patch(grid: np.ndarray, pts: set[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_pts(pts)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        patch[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return patch


def _conn_points(patch: np.ndarray, ctrl: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(patch == int(ctrl))
    }


def _norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _connector_side(patch: np.ndarray, ctrl: int) -> str:
    conn = _conn_points(patch, ctrl)
    assert conn
    cols = [c for _r, c in conn]
    if int(min(cols)) == 0:
        return "left"
    if int(max(cols)) == int(patch.shape[1] - 1):
        return "right"
    return "interior"


def _dock_from_patches(
    base_patch: np.ndarray,
    plug_patch: np.ndarray,
    *,
    bg: int,
    obj: int,
    ctrl: int,
    flip_plug: bool,
) -> np.ndarray:
    base = np.array(base_patch, dtype=int)
    plug = np.array(plug_patch, dtype=int)
    assert base.size > 0 and plug.size > 0

    conn_base = _conn_points(base, ctrl)
    conn_plug = _conn_points(plug, ctrl)
    assert conn_base and conn_plug

    shp_base, (dr0, dc0) = _norm(conn_base)
    shp_plug_in, _ = _norm(conn_plug)
    assert shp_base == shp_plug_in

    base_body = {
        (int(r), int(c))
        for r, c in np.argwhere(base == int(obj))
    }
    for r, c in conn_base:
        assert (int(r + 1), int(c)) not in base_body

    work_plug = np.fliplr(plug) if flip_plug else plug.copy()
    conn_work = _conn_points(work_plug, ctrl)
    shp_plug, (pr0, pc0) = _norm(conn_work)
    assert shp_base == shp_plug

    plug_body = {
        (int(r), int(c))
        for r, c in np.argwhere(work_plug == int(obj))
    }
    for r, c in conn_work:
        assert (int(r - 1), int(c)) not in plug_body

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)

    plug_pixels: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(work_plug != int(bg)):
        plug_pixels[(int(r + tr), int(c + tc))] = int(work_plug[int(r), int(c)])

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base != int(bg))}
    pts_all = base_pts | set(plug_pixels)
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    assert rr0 == 0 and cc0 == 0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)
    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(base[int(r), int(c)])

    for (r, c), val in plug_pixels.items():
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    return _crop_non_bg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    connA = {(r, c) for (r, c) in A if int(g[int(r), int(c)]) == int(ctrl)}
    connB = {(r, c) for (r, c) in B if int(g[int(r), int(c)]) == int(ctrl)}
    assert connA and connB
    shpA, _ = _norm(connA)
    shpB, _ = _norm(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base_pts, plug_pts = A, B
    else:
        base_pts, plug_pts = B, A

    base_patch = _extract_patch(g, base_pts, bg)
    plug_patch = _extract_patch(g, plug_pts, bg)
    return _dock_from_patches(
        base_patch,
        plug_patch,
        bg=bg,
        obj=obj,
        ctrl=ctrl,
        flip_plug=True,
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
_rng = random


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= int(r + dr) < h and 0 <= int(c + dc) < w and not mask[int(r + dr), int(c + dc)]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r, c = int(r), int(c)
        if (r, c) in keep:
            continue
        adj_bg = any(
            not (0 <= int(r + dr) < h and 0 <= int(c + dc) < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _build_base() -> tuple[np.ndarray, str] | None:
    bg, obj, ctrl = 1, 4, 7
    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, int(base_h * base_w)))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    side = _rng.choice(["left", "right"])
    candidates: list[tuple[int, list[int]]] = []
    for r in range(1, base_h):
        if side == "left":
            c0 = 0
        else:
            c0 = int(base_w - conn_len)
        cols = list(range(int(c0), int(c0 + conn_len)))
        if not all(base[int(r), int(c)] for c in cols):
            continue
        if not all(base[int(r - 1), int(c)] for c in cols):
            continue
        if int(r + 1) < base_h and not all(not base[int(r + 1), int(c)] for c in cols):
            continue
        candidates.append((int(r), cols))
    if not candidates:
        return None

    conn_row, conn_cols = _rng.choice(candidates)
    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base] = int(obj)
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = int(ctrl)

    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None
    base_sprite = _crop_non_bg(base_grid, bg)
    if base_sprite.size == 0:
        return None
    if _connector_side(base_sprite, ctrl) != side:
        return None
    if _rng.random() < 0.5:
        base_sprite = np.fliplr(base_sprite)
        side = "right" if side == "left" else "left"
    return base_sprite, side


def _build_docked_plug(base_sprite: np.ndarray, side: str) -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7
    base_h, base_w = map(int, base_sprite.shape)
    conn_base = _conn_points(base_sprite, ctrl)
    _shape, (base_conn_r0, base_conn_c0) = _norm(conn_base)
    conn_len = int(len(conn_base))

    plug_h = _rng.randint(2, 7)
    if side == "right":
        max_w = base_w
        if max_w < max(conn_len, 2):
            return None
        plug_w = _rng.randint(max(conn_len, 2), max_w)
        dock_conn_c0 = int(plug_w - conn_len)
    else:
        plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
        dock_conn_c0 = 0

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(dock_conn_c0, dock_conn_c0 + conn_len):
        plug[0, int(c)] = True
        if plug_h > 1:
            plug[1, int(c)] = True

    plug_max = min(29, int(plug_h * plug_w))
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, [(int(r), int(c)) for r, c in zip(*np.where(plug))])
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    keep_plug = {(0, int(c)) for c in range(dock_conn_c0, dock_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    if np.array_equal(plug, np.fliplr(plug)):
        return None

    plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    plug_grid[plug] = int(obj)
    for c in range(dock_conn_c0, dock_conn_c0 + conn_len):
        plug_grid[0, int(c)] = int(ctrl)
        if plug_h < 2 or plug_grid[1, int(c)] != int(obj):
            return None

    if _connector_side(plug_grid, ctrl) != side:
        return None

    tc = int(base_conn_c0 - dock_conn_c0)
    tr = int((base_conn_r0 - 1) - 0)
    if tc < 0 or tr < 0:
        return None

    for r, c in np.argwhere(plug_grid != int(bg)):
        r, c = int(r), int(c)
        dr = int(r + tr)
        dc = int(c + tc)
        if 0 <= dr < base_h and 0 <= dc < base_w:
            cur = int(base_sprite[dr, dc])
            val = int(plug_grid[r, c])
            if cur != int(bg) and val != int(bg) and cur != int(ctrl) and val != int(ctrl):
                return None

    try:
        expected = _dock_from_patches(
            base_sprite,
            np.fliplr(plug_grid),
            bg=bg,
            obj=obj,
            ctrl=ctrl,
            flip_plug=True,
        )
    except AssertionError:
        return None
    try:
        direct = _dock_from_patches(
            base_sprite,
            np.fliplr(plug_grid),
            bg=bg,
            obj=obj,
            ctrl=ctrl,
            flip_plug=False,
        )
    except AssertionError:
        direct = None
    if direct is not None and np.array_equal(expected, direct):
        return None
    return plug_grid


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return int(_gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1))


def _placement_candidates(
    canvas_h: int,
    canvas_w: int,
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    bg: int,
) -> list[tuple[int, int]]:
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return []

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    base_dilated = set(base_pts)
    for r, c in list(base_pts):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                base_dilated.add((int(r + dr), int(c + dc)))
    plug_local = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}

    out: list[tuple[int, int]] = []
    base_bbox = (0, 0, int(bh - 1), int(bw - 1))
    for pr in range(0, int(canvas_h - ph + 1)):
        for pc in range(0, int(canvas_w - pw + 1)):
            if pr == 0 and pc == 0:
                continue
            shifted = {(int(r + pr), int(c + pc)) for r, c in plug_local}
            if any(pt in base_dilated for pt in shifted):
                continue
            gap = _bbox_gap(base_bbox, (int(pr), int(pc), int(pr + ph - 1), int(pc + pw - 1)))
            if 1 <= int(gap) <= 5:
                out.append((int(pr), int(pc)))
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    built = _build_base()
    if built is None:
        return None
    base_sprite, side = built

    docked_plug = _build_docked_plug(base_sprite, side)
    if docked_plug is None:
        return None

    plug_input = np.fliplr(docked_plug)
    if _connector_side(plug_input, ctrl) == side:
        return None

    try:
        expected = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            obj=obj,
            ctrl=ctrl,
            flip_plug=True,
        )
    except AssertionError:
        return None
    try:
        direct = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            obj=obj,
            ctrl=ctrl,
            flip_plug=False,
        )
    except AssertionError:
        direct = None
    if direct is not None and np.array_equal(expected, direct):
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_input.shape)
    min_h = max(7, bh, ph)
    min_w = max(3, bw, pw)
    if min_h > 11 or min_w > 13:
        return None

    for _ in range(20):
        canvas_h = _rng.randint(min_h, 11)
        canvas_w = _rng.randint(min_w, 13)
        placements = _placement_candidates(canvas_h, canvas_w, base_sprite, plug_input, bg)
        if not placements:
            continue
        pr, pc = _rng.choice(placements)
        canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
        paste(canvas, base_sprite, 0, 0, transparent=bg)
        paste(canvas, plug_input, pr, pc, transparent=bg)
        if np.sum(canvas == int(bg)) <= canvas.size // 2:
            continue
        try:
            result = main(canvas)
        except Exception:
            continue
        if not np.array_equal(result, expected):
            continue
        return canvas
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
