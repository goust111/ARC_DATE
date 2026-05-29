# Variant of 20270e3b
# Dimension: D8 — 精灵数量与配对基数 (multi_step)
# Change: 恰好2个精灵，1:1配对对接... → 一个base多个plug

from __future__ import annotations

import random
import numpy as np
from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


# Design analysis for the D8 variant
# ----------------------------------
# concretization:
# - Input now contains 1 base sprite anchored at the grid origin and 2-3 separate plug
#   sprites elsewhere on the canvas.
# - The base contains 2-3 orange connector bars (ctrl=7), each a horizontal 1/2/3-cell
#   segment on the same sprite. Each plug contains exactly one connector bar whose length
#   matches exactly one base connector. The lengths are unique inside one puzzle, so the
#   matching is visible and unambiguous.
# - Each plug still docks using the original local rule: its connector row is translated so
#   it lands one row above the matched base connector, making a 2-row seam where ctrl meets
#   obj. After stamping, all ctrl pixels recolor to obj=4, then the result is tight-cropped.
#
# behavior trace:
# - Original task: find 2 components -> choose the origin component as base -> compute one
#   connector translation -> render once.
# - New task: find 3-4 components -> choose the origin component as base -> split the base's
#   ctrl pixels into multiple connector components -> for each non-base sprite, match its sole
#   connector to one base connector by normalized horizontal-bar shape -> compute one shift per
#   plug -> render all shifted plugs into the same assembled output.
# - The critical changed step is the matching/render loop: the old code depended on "exactly one
#   base connector and one plug connector", while the new rule needs a connector map and multiple
#   independent translations.
#
# complexity calibration:
# - If all connectors had the same length, a plug could match several base ports and the puzzle
#   would become ambiguous. To keep the D8 change visible but not underconstrained, each sample
#   uses 2 or 3 connectors with distinct lengths chosen from {1,2,3}.
# - The generator keeps all connectors visually separated along the base underside, so humans can
#   see a one-to-many docking interface rather than a noisy pile of orange pixels.
#
# implicit-assumption audit against the original main():
# - `assert len(best_comps) == 2`: no longer valid; the new solver expects 1 base + multiple plugs.
# - `connA/connB` as one set per sprite: no longer valid for the base, which now has multiple ctrl
#   components, so ctrl pixels must be re-labeled inside the base component.
# - `assert shpA == shpB`: no longer valid globally; matching becomes per-plug by connector shape.
# - Single `(tr, tc)` translation: replaced with one translation per plug.
# - Original overlap check only guaranteed "some ctrl is involved", which was harmless with one plug.
#   With several plugs, that would accidentally allow a plug to intersect another base connector or a
#   previous plug. The new solver therefore asserts plug-plug disjointness and only allows overlap on
#   the matched 2-row docking zone for that plug.
#
# placement legality:
# - The generator constructs the assembled state first, ensuring each plug only overlaps the base on
#   its own docking seam and never overlaps another plug.
# - It then crops the sprites and scatters the plugs back onto a larger canvas with padding=1 under
#   8-connectivity, guaranteeing separate connected components in the input.
#
# visual observability and neatness:
# - The changed rule is apparent from 2-4 examples because the input visibly has one port-rich base
#   and several independent plugs, and the output visibly contains all of them docked at once.
# - Colors stay fixed (`bg=1`, `obj=4`, `ctrl=7`), so there is no hidden color remapping burden.


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (int(r0), int(c0))


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _ctrl_components(sprite: set[tuple[int, int]], grid: np.ndarray, ctrl: int) -> list[set[tuple[int, int]]]:
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
    Connect one base sprite to multiple plug sprites using matching connector bars.

    Input structure:
    - Fixed colors: bg=1, obj=4, ctrl=7.
    - Exactly one base component anchored at the grid origin.
    - 2-3 additional plug components, each with one connector bar.
    - The base contains the same number of connector bars as there are plugs.

    Rule:
    - Match each plug to the unique base connector with the same normalized ctrl-bar shape.
    - Translate every plug independently using the original docking rule.
    - Stamp all plugs onto the base, recolor ctrl->obj, then tight-crop.
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    assert len(comps) >= 3

    boxes = [_bbox_of_pts(pts) for pts in comps]
    base_idx = min(range(len(comps)), key=lambda i: int(boxes[i][0] + boxes[i][1]))
    base = comps[base_idx]
    base_box = boxes[base_idx]
    assert int(base_box[0] + base_box[1]) == 0
    plugs = [pts for i, pts in enumerate(comps) if i != base_idx]
    assert 2 <= len(plugs) <= 3

    base_conns = _ctrl_components(base, g, ctrl)
    assert len(base_conns) == len(plugs)
    base_body = base - set().union(*base_conns)

    base_map: dict[frozenset[tuple[int, int]], tuple[set[tuple[int, int]], tuple[int, int]]] = {}
    for conn in base_conns:
        rows = {r for r, _c in conn}
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
        rows = {r for r, _c in plug_conn}
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

        match_cols = {c for _r, c in base_conn}
        dock_zone = (
            {(int(dr0 - 1), int(c)) for c in match_cols}
            | {(int(dr0), int(c)) for c in match_cols}
        )
        for pt in plug_shift:
            if pt in base:
                assert pt in dock_zone
        assert shifted_union.isdisjoint(plug_shift)
        shifted_union |= plug_shift
        shifted_plugs.append((plug, tr, tc))

    assert len(used_shapes) == len(base_map)

    pts_all = set(base) | shifted_union
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))

    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
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


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
import random as _rng


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]], allowed: np.ndarray | None = None) -> None:
    """Grow a connected shape from frontier up to target pixels, optionally within `allowed`."""
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
    """Remove interior pixels while preserving 8-connectivity."""
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    h, w = mask.shape
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


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
    target_lo = max(8, int(body.sum()) + 1)
    target_hi = min(24, int(allowed.sum()))
    if target_lo > target_hi:
        return None
    _grow_shape(body, _rng.randint(target_lo, target_hi), list(zip(*np.where(body))), allowed=allowed)
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
    if int(np.sum(base_grid != bg)) < 7:
        return None
    return base_grid, connectors


def _build_plug_for_connector(
    base_grid: np.ndarray,
    conn_row: int,
    conn_cols: list[int],
    occupied: set[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    bg, obj, ctrl = 1, 4, 7
    base_h, base_w = base_grid.shape
    conn_len = len(conn_cols)
    conn_start = int(conn_cols[0])

    allowed_overlap = {(int(conn_row - 1), int(c)) for c in conn_cols}
    allowed_overlap |= {(int(conn_row), int(c)) for c in conn_cols}

    for _ in range(120):
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
        target_hi = min(24, int(plug_h * plug_w))
        if target_lo > target_hi:
            continue
        _grow_shape(plug, _rng.randint(target_lo, target_hi), list(zip(*np.where(plug))))
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

        if int(np.sum(plug_grid != bg)) < 5:
            continue
        if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
            continue
        return plug_grid, global_pts

    return None


def _place_input_sprites(base_sprite: np.ndarray, plug_sprites: list[np.ndarray]) -> np.ndarray | None:
    bg = 1
    sprites = list(plug_sprites)
    _rng.shuffle(sprites)

    max_h = max([base_sprite.shape[0]] + [sp.shape[0] for sp in sprites])
    max_w = max([base_sprite.shape[1]] + [sp.shape[1] for sp in sprites])
    total_w = int(base_sprite.shape[1] + sum(sp.shape[1] for sp in sprites))

    min_h = max(10, int(max_h + 2))
    min_w = max(12, int(max_w + 3), int(min(17, total_w // 2 + base_sprite.shape[1] + 2)))
    if min_h > 14 or min_w > 17:
        return None

    for _ in range(40):
        canvas_h = _rng.randint(min_h, 14)
        canvas_w = _rng.randint(min_w, 17)
        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        paste(canvas, base_sprite, 0, 0, transparent=bg)

        ok = True
        for sprite in sprites:
            try:
                pr, pc = random_free_location(canvas, sprite, transparent=bg, padding=1, padding_connectivity=8)
            except ValueError:
                ok = False
                break
            if pr + pc == 0:
                ok = False
                break
            paste(canvas, sprite, pr, pc, transparent=bg)
        if not ok:
            continue

        _lab, n = label_mask(canvas != bg, connectivity=8)
        if int(n) != int(1 + len(sprites)):
            continue
        return canvas
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = 1

    built_base = _build_base()
    if built_base is None:
        return None
    base_grid, connectors = built_base

    occupied: set[tuple[int, int]] = set()
    plug_pairs: list[tuple[np.ndarray, set[tuple[int, int]]]] = []
    for conn_row, conn_cols in sorted(connectors, key=lambda item: (-len(item[1]), item[1][0])):
        built = _build_plug_for_connector(base_grid, conn_row, conn_cols, occupied)
        if built is None:
            return None
        plug_grid, global_pts = built
        occupied |= global_pts
        plug_pairs.append((plug_grid, global_pts))

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprites = [_crop_non_bg(plug_grid, bg) for plug_grid, _pts in plug_pairs]
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


def generate_input():
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
