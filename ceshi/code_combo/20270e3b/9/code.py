# Combo variant of 20270e3b (index 9)
# combo_spec: D6/0+D8/2
# Name: 面积判定一对多对接
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace
#    Original task:
#    - There are exactly 2 sprites.
#    - The sprite touching the grid origin is base.
#    - Each sprite has exactly 1 horizontal ctrl connector.
#    - Move the other sprite so its connector docks one row above the base connector.
#
#    Fused task:
#    - There are 3-4 sprites total: 1 base + 2-3 plugs.
#    - The base is the unique largest sprite by foreground area, not the one nearest (0,0).
#    - The base has 2-3 separate horizontal ctrl bars at different horizontal positions.
#    - Each plug has exactly 1 ctrl bar; it docks to the unique base connector with the same shape.
#
#    Small trace:
#    - Input components have sizes 18, 9, 7.
#    - The 7-pixel plug is placed at the input origin; the 18-pixel sprite is elsewhere.
#    - Old logic would privilege the origin component; fused logic must instead count pixels,
#      choose the 18-pixel component as base, split its ctrl pixels into multiple ports, then
#      move both smaller plugs independently onto their matching ports.
#
# b) Complexity calibration
#    - Matching uses only connector lengths in {1,2,3}; within each sample these lengths are
#      unique, so every plug has exactly one visible target.
#    - There are only 2 or 3 plugs, enough to make the one-to-many rule obvious without making
#      the grid noisy or underconstrained.
#    - The generator enforces a strict area gap: base area exceeds every plug by at least 4 px.
#
# c) Implicit-assumption audit versus the original solver
#    - `len(components) == 2` breaks: fused inputs have 3-4 components.
#    - "base = component nearest origin" breaks: the origin component is intentionally a plug.
#    - "one connector per sprite" breaks for the base: ctrl pixels inside the base must be
#      re-labeled into separate 4-connected connector bars.
#    - "global connector shapes are equal" breaks: matching is now per-plug against one base port.
#    - "one translation" breaks: there is one translation per plug.
#    - Overlap checks must be stricter because multiple shifted plugs now coexist.
#
# d) Placement legality
#    - The generator builds a valid assembled state first, verifying that each plug only overlaps
#      the base on its own 2-row docking seam and never overlaps another plug.
#    - After cropping, it places one plug at the origin and scatters the base plus remaining plugs
#      with padding=1 under 8-connectivity, so the input stays split into separate components.
#
# e) Visual observability
#    - In training examples, solvers can see one large multi-port sprite plus multiple smaller
#      single-port sprites. The origin sprite is visibly not the stationary piece.
#    - This makes both changed dimensions legible from a few examples: "largest wins" and
#      "dock all plugs, not just one".
#
# f) Visual cleanliness
#    - Only the original three colors are used: bg=1, obj=4, ctrl=7.
#    - Base connectors are spaced apart horizontally and plugs are generated to avoid overlap
#      in the assembled state, so the final seam is readable instead of cluttered.
#
# g) Color inference
#    - No new color semantics are introduced. Ctrl always marks docking bars and is recolored
#      to obj after docking, exactly as in the source task.
#
# Post-change solver assumptions and generator guarantees
#    - There are exactly 3 or 4 foreground components.
#    - Exactly one component has strictly largest area; that component is the base.
#    - The base contains exactly as many 4-connected ctrl bars as there are plug components.
#    - Each plug contains exactly one ctrl bar, and base connector shapes are unique per sample.
#    - In the docked state, plugs do not extend above or left of the base and shifted plugs do
#      not overlap one another. The generator explicitly enforces all of these assumptions.

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (int(r0), int(c0))


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _ctrl_components(
    sprite: set[tuple[int, int]], grid: np.ndarray, ctrl: int
) -> list[set[tuple[int, int]]]:
    mask = np.zeros(grid.shape, dtype=bool)
    for r, c in sprite:
        if int(grid[int(r), int(c)]) == int(ctrl):
            mask[int(r), int(c)] = True
    lab, n = label_mask(mask, connectivity=4)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the unique largest sprite to multiple smaller plugs using matching ctrl bars.
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    assert 3 <= len(comps) <= 4

    sizes = [len(pts) for pts in comps]
    max_size = max(sizes)
    assert sizes.count(max_size) == 1
    base_idx = sizes.index(max_size)
    base = comps[base_idx]
    plugs = [pts for i, pts in enumerate(comps) if i != base_idx]
    assert 2 <= len(plugs) <= 3

    base_conns = _ctrl_components(base, g, ctrl)
    assert len(base_conns) == len(plugs)
    base_body = base - set().union(*base_conns)

    base_map: dict[frozenset[tuple[int, int]], tuple[set[tuple[int, int]], tuple[int, int]]] = {}
    for conn in base_conns:
        rows = {r for r, _ in conn}
        assert len(rows) == 1
        for r, c in conn:
            assert (int(r - 1), int(c)) in base_body
            assert (int(r + 1), int(c)) not in base_body
        shape, anchor = _norm_pts(conn)
        assert shape not in base_map
        base_map[shape] = (conn, anchor)

    shifted_plugs: list[tuple[set[tuple[int, int]], int, int]] = []
    used_shapes: set[frozenset[tuple[int, int]]] = set()
    shifted_union: set[tuple[int, int]] = set()

    for plug in plugs:
        plug_conns = _ctrl_components(plug, g, ctrl)
        assert len(plug_conns) == 1
        plug_conn = plug_conns[0]
        plug_body = plug - plug_conn
        rows = {r for r, _ in plug_conn}
        assert len(rows) == 1
        for r, c in plug_conn:
            assert (int(r + 1), int(c)) in plug_body
            assert (int(r - 1), int(c)) not in plug_body

        shape, (pr0, pc0) = _norm_pts(plug_conn)
        assert shape in base_map
        assert shape not in used_shapes
        used_shapes.add(shape)

        base_conn, (dr0, dc0) = base_map[shape]
        tr = int((dr0 - 1) - pr0)
        tc = int(dc0 - pc0)
        plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

        match_cols = {c for _, c in base_conn}
        dock_zone = {(int(dr0 - 1), int(c)) for c in match_cols}
        dock_zone |= {(int(dr0), int(c)) for c in match_cols}

        for pt in plug_shift:
            if pt in base:
                assert pt in dock_zone
            assert pt not in shifted_union
        shifted_union |= plug_shift
        shifted_plugs.append((plug, tr, tc))

    assert len(used_shapes) == len(base_map)

    pts_all = set(base) | shifted_union
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    base_coords = np.array(list(base), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for plug, tr, tc in shifted_plugs:
        for r, c in plug:
            rr = int(r + tr)
            cc = int(c + tc)
            val = int(g[int(r), int(c)])
            cur = int(out[int(rr - rr0), int(cc - cc0)])
            if cur != int(bg) and val != int(bg):
                assert cur == int(ctrl) or val == int(ctrl)
            out[int(rr - rr0), int(cc - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray | None = None,
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
            if mask[nr, nc]:
                continue
            if allowed is not None and not bool(allowed[nr, nc]):
                continue
            nbrs.append((nr, nc))
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    h, w = mask.shape
    for r, c in pixels:
        rr = int(r)
        cc = int(c)
        if (rr, cc) in keep:
            continue
        adj_bg = any(
            not (0 <= rr + dr < h and 0 <= cc + dc < w) or not mask[int(rr + dr), int(cc + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[rr, cc] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[rr, cc] = True


def _build_base() -> tuple[np.ndarray, list[tuple[int, list[int]]]] | None:
    bg, obj, ctrl = 1, 4, 7

    n_plugs = _rng.randint(2, 3)
    lengths = _rng.sample([1, 2, 3], n_plugs)
    _rng.shuffle(lengths)

    left_pad = _rng.randint(0, 1)
    right_pad = _rng.randint(0, 2)
    gaps = [_rng.randint(1, 2) for _ in range(n_plugs - 1)]
    base_w = left_pad + sum(lengths) + sum(gaps) + right_pad
    if base_w > 10:
        return None

    conn_row = _rng.randint(2, 5)
    base_h = conn_row + 1

    body = np.zeros((base_h, base_w), dtype=bool)
    body[0, 0] = True
    connectors: list[tuple[int, list[int]]] = []
    keep = {(0, 0)}

    c = left_pad
    for idx, conn_len in enumerate(lengths):
        cols = list(range(c, c + conn_len))
        connectors.append((conn_row, cols))
        bend = _rng.randint(0, conn_row - 1)
        body[0 : bend + 1, 0] = True
        body[bend, 0 : cols[-1] + 1] = True
        for col in cols:
            body[bend:conn_row, col] = True
            keep.add((int(conn_row - 1), int(col)))
        c += conn_len
        if idx < len(gaps):
            c += gaps[idx]

    allowed = np.ones_like(body, dtype=bool)
    allowed[conn_row, :] = False
    target_lo = max(12, int(body.sum()) + 2)
    target_hi = min(26, int(allowed.sum()))
    if target_lo > target_hi:
        return None
    frontier = [(int(r), int(c)) for r, c in np.argwhere(body)]
    _grow_shape(body, _rng.randint(target_lo, target_hi), frontier, allowed=allowed)
    _thin(body, keep=keep)

    for _row, cols in connectors:
        if not all(bool(body[int(conn_row - 1), int(col)]) for col in cols):
            return None
    if not is_contiguous(body, background=0, connectivity=8):
        return None

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[body] = obj
    for _row, cols in connectors:
        for col in cols:
            base_grid[int(conn_row), int(col)] = ctrl

    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None
    if int(np.sum(base_grid != bg)) < 12:
        return None
    return base_grid, connectors


def _build_plug_for_connector(
    base_grid: np.ndarray,
    conn_row: int,
    conn_cols: list[int],
    occupied: set[tuple[int, int]],
    base_count: int,
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    bg, obj, ctrl = 1, 4, 7
    base_h, base_w = base_grid.shape
    conn_len = len(conn_cols)
    conn_start = int(conn_cols[0])

    allowed_overlap = {(int(conn_row - 1), int(c)) for c in conn_cols}
    allowed_overlap |= {(int(conn_row), int(c)) for c in conn_cols}

    for _ in range(140):
        plug_h = _rng.randint(2, 7)
        plug_w = _rng.randint(max(conn_len, 2), min(8, conn_len + 4))
        max_conn_c0 = min(int(plug_w - conn_len), int(conn_start))
        if max_conn_c0 < 0:
            continue
        plug_conn_c0 = _rng.randint(0, max_conn_c0)

        plug = np.zeros((plug_h, plug_w), dtype=bool)
        keep: set[tuple[int, int]] = set()
        for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
            plug[0, c] = True
            plug[1, c] = True
            keep.add((0, int(c)))
            keep.add((1, int(c)))

        target_lo = max(5, int(plug.sum()))
        target_hi = min(17, int(plug_h * plug_w), int(base_count - 4))
        if target_lo > target_hi:
            continue
        frontier = [(int(r), int(c)) for r, c in np.argwhere(plug)]
        _grow_shape(plug, _rng.randint(target_lo, target_hi), frontier)
        if not is_contiguous(plug, background=0, connectivity=8):
            continue
        _thin(plug, keep=keep)

        if not all(bool(plug[0, c]) and bool(plug[1, c]) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)):
            continue

        tr = int(conn_row - 1)
        tc = int(conn_start - plug_conn_c0)
        global_pts: set[tuple[int, int]] = set()
        ok = True
        for pr, pc in np.argwhere(plug):
            gr = int(tr + pr)
            gc = int(tc + pc)
            if gr < 0 or gc < 0:
                ok = False
                break
            if (gr, gc) in occupied:
                ok = False
                break
            global_pts.add((gr, gc))
            if 0 <= gr < base_h and 0 <= gc < base_w and int(base_grid[gr, gc]) != int(bg):
                if (gr, gc) not in allowed_overlap:
                    ok = False
                    break
                val = int(ctrl) if int(pr) == 0 and plug_conn_c0 <= int(pc) < plug_conn_c0 + conn_len else int(obj)
                cur = int(base_grid[gr, gc])
                if cur != int(ctrl) and val != int(ctrl):
                    ok = False
                    break
        if not ok:
            continue

        plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
        plug_grid[plug] = obj
        for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
            plug_grid[0, c] = ctrl

        plug_count = int(np.sum(plug_grid != bg))
        if plug_count < 5 or plug_count >= base_count - 3:
            continue
        if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
            continue
        return plug_grid, global_pts

    return None


def _place_input_sprites(base_sprite: np.ndarray, plug_sprites: list[np.ndarray]) -> np.ndarray | None:
    bg = 1
    origin_idx = _rng.randrange(len(plug_sprites))
    origin_plug = plug_sprites[origin_idx]
    other_plugs = [sp for i, sp in enumerate(plug_sprites) if i != origin_idx]

    max_h = max([base_sprite.shape[0], origin_plug.shape[0]] + [sp.shape[0] for sp in other_plugs])
    max_w = max([base_sprite.shape[1], origin_plug.shape[1]] + [sp.shape[1] for sp in other_plugs])
    total_w = int(base_sprite.shape[1] + sum(sp.shape[1] for sp in plug_sprites))

    min_h = max(10, int(max_h + 2))
    min_w = max(12, int(max_w + 4), int(min(18, total_w // 2 + base_sprite.shape[1] + 2)))
    if min_h > 14 or min_w > 18:
        return None

    for _ in range(60):
        canvas_h = _rng.randint(min_h, 14)
        canvas_w = _rng.randint(min_w, 18)
        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        paste(canvas, origin_plug, 0, 0, transparent=bg)

        try:
            br, bc = random_free_location(
                canvas, base_sprite, transparent=bg, padding=1, padding_connectivity=8
            )
        except ValueError:
            continue
        if br + bc == 0:
            continue
        paste(canvas, base_sprite, br, bc, transparent=bg)

        ok = True
        for sprite in other_plugs:
            try:
                pr, pc = random_free_location(
                    canvas, sprite, transparent=bg, padding=1, padding_connectivity=8
                )
            except ValueError:
                ok = False
                break
            if pr + pc == 0:
                ok = False
                break
            paste(canvas, sprite, pr, pc, transparent=bg)
        if not ok:
            continue

        lab, n = label_mask(canvas != bg, connectivity=8)
        if int(n) != int(1 + len(plug_sprites)):
            continue
        sizes = []
        origin_sizes = []
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(k))
            r0, c0, _r1, _c1 = bbox_coords(pts)
            area = int(pts.shape[0])
            sizes.append(area)
            if int(r0 + c0) == 0:
                origin_sizes.append(area)
        if len(origin_sizes) != 1:
            continue
        if origin_sizes[0] >= max(sizes):
            continue
        if sizes.count(max(sizes)) != 1:
            continue
        return canvas
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = 1

    built_base = _build_base()
    if built_base is None:
        return None
    base_grid, connectors = built_base
    base_count = int(np.sum(base_grid != bg))

    occupied: set[tuple[int, int]] = set()
    plug_grids: list[np.ndarray] = []
    plug_counts: list[int] = []
    for conn_row, conn_cols in sorted(connectors, key=lambda item: (-len(item[1]), item[1][0])):
        built = _build_plug_for_connector(base_grid, conn_row, conn_cols, occupied, base_count)
        if built is None:
            return None
        plug_grid, global_pts = built
        occupied |= global_pts
        plug_grids.append(plug_grid)
        plug_counts.append(int(np.sum(plug_grid != bg)))

    if not plug_counts:
        return None
    if max(plug_counts) >= base_count - 3:
        return None

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprites = [_crop_non_bg(plug_grid, bg) for plug_grid in plug_grids]
    if base_sprite is None or any(sprite is None for sprite in plug_sprites):
        return None

    canvas = _place_input_sprites(base_sprite, [sprite for sprite in plug_sprites if sprite is not None])
    if canvas is None:
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
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
