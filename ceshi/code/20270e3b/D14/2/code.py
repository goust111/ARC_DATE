# Variant of 20270e3b
# Dimension: D14 — 精灵间距约束 (meta)
# Change: 输入中两个精灵的曼哈顿间距为1-5像素... → 精灵由分隔线划分区域

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste

# -----------------------------------------------------------------------------
# Design analysis for the D14 variant
#
# a) Concretization and behavior trace
#    - Keep the core task: two sprites dock using matching ctrl-colored connectors;
#      output recolors ctrl -> obj and tight-crops the merged result.
#    - Replace the old "two nearby sprites in one shared field" input layout with a
#      frame-separation layout: a single full ctrl-colored separator line splits the
#      input into two regions, and each region contains exactly one sprite.
#    - The separator's orientation carries the new meta signal:
#         * horizontal separator  => vertical docking (original geometry)
#         * vertical separator    => horizontal docking
#    - To avoid inventing a second docking algorithm, vertical-separator cases are
#      treated as a 90-degree-rotated copy of the horizontal-separator case. Solver
#      rotates the input to a canonical form, solves there, then rotates the output
#      back. This keeps the "connector overlap at docking seam" logic identical.
#    - Small trace:
#         original: plug is found by proximity and moved above base connector.
#         variant : top/left region sprite is plug; bottom/right region sprite is
#                   base; separator tells whether the merge happens vertically or
#                   horizontally. After canonicalization, plug is again moved above
#                   base connector, but the input evidence is the separator, not a
#                   1-5 Manhattan gap.
#
# b) Complexity calibration
#    - The new cue is visually strong: a full-row/full-column ctrl line is obvious.
#    - To keep the new branch visible, generator samples both separator directions.
#    - Connector lengths remain 1-3, as in the original, so there are still repeated
#      local regularities across examples rather than too many degrees of freedom.
#
# c) Review of old assumptions that no longer hold
#    - Old: foreground had exactly 2 connected components. Now it has 3: plug, base,
#      and the separator line. Solver must detect/remove the separator before object
#      extraction.
#    - Old: base was anchored at the grid origin. No longer true; region membership
#      now determines roles. Any top-left-based ranking would be wrong.
#    - Old: plug could be anywhere with small Manhattan gap. No longer true; legal
#      inputs require one sprite per side of the separator and a blank margin so the
#      line stays a distinct component.
#    - Old: docking direction was always vertical. Variant adds horizontal docking,
#      implemented by rotating the whole puzzle into the original vertical form.
#
# d) Placement legality
#    - Generator enforces:
#         * exactly one full ctrl separator row/col
#         * exactly one sprite in each region
#         * at least one background row/col between each sprite and the separator
#         * no overlap and no out-of-bounds placement
#    - Vertical-separator samples are produced by rotating a valid horizontal sample,
#      so legality is inherited instead of reimplemented.
#
# e) Visual observability
#    - A human can see that the separator partitions the scene into two zones.
#    - Across examples, horizontal lines yield tall stacked outputs; vertical lines
#      yield wide side-by-side outputs. The line orientation is therefore meaningful
#      in 2-3 examples.
#
# f) Visual cleanliness
#    - Separator is isolated by background, so it does not merge with connectors.
#    - Only three visible structures exist in the input: plug sprite, separator, base
#      sprite. No extra fragments are introduced.
#
# g) Color inferability
#    - Colors stay fixed to the original palette: bg=1, obj=4, ctrl=7.
#    - ctrl serves two visible roles with the same structural cue family:
#         * short straight connector on each sprite
#         * one full straight separator line across the canvas
#      The distinction is geometric, not color-id-dependent.
#
# h) New solver assumptions after the change
#    - There is exactly one full-row or full-column ctrl separator.
#    - Each side of the separator contains exactly one 8-connected sprite.
#    - Canonicalized sprites still use the original connector semantics:
#         base connector has body above and free space below;
#         plug connector has body below and free space above.
#    - Generator is written to guarantee all of these.
# -----------------------------------------------------------------------------


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _find_separator(grid: np.ndarray, ctrl: int) -> tuple[str, int]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[int(r), :] == int(ctrl))]
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, int(c)] == int(ctrl))]
    assert (len(rows) == 1) ^ (len(cols) == 1)
    if rows:
        return "h", rows[0]
    return "v", cols[0]


def _norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (r0, c0)


def _sprite_from_region(
    grid: np.ndarray, row_slice: slice, col_slice: slice, bg: int
) -> set[tuple[int, int]]:
    sub = grid[row_slice, col_slice]
    lab, n = label_mask(sub != int(bg), connectivity=8)
    assert int(n) == 1
    r_off = int(row_slice.start or 0)
    c_off = int(col_slice.start or 0)
    return {(int(r + r_off), int(c + c_off)) for r, c in np.argwhere(lab == 1)}


def _solve_canonical(grid: np.ndarray) -> np.ndarray:
    """
    Canonical form for this variant:
    - one horizontal full-ctrl separator row
    - plug sprite in the top region
    - base sprite in the bottom region
    - docking is the original vertical docking from 20270e3b
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    orient, sep = _find_separator(g, ctrl)
    assert orient == "h"
    assert 0 < int(sep) < g.shape[0] - 1

    plug = _sprite_from_region(g, slice(0, int(sep)), slice(0, g.shape[1]), bg)
    base = _sprite_from_region(g, slice(int(sep) + 1, g.shape[0]), slice(0, g.shape[1]), bg)

    connP = {(r, c) for (r, c) in plug if int(g[int(r), int(c)]) == int(ctrl)}
    connD = {(r, c) for (r, c) in base if int(g[int(r), int(c)]) == int(ctrl)}
    assert connP and connD

    shpP, (pr0, pc0) = _norm(connP)
    shpD, (dr0, dc0) = _norm(connD)
    assert shpP == shpD
    assert len({r for r, _c in connP}) == 1
    assert len({r for r, _c in connD}) == 1

    plug_body = plug - connP
    base_body = base - connD
    for r, c in connP:
        assert (int(r - 1), int(c)) not in plug_body
        assert (int(r + 1), int(c)) in plug_body
    for r, c in connD:
        assert (int(r + 1), int(c)) not in base_body
        assert (int(r - 1), int(c)) in base_body

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
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
    return _crop_non_bg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    D14 variant rule:
    - Detect the full ctrl separator line.
    - Horizontal separator: solve directly (top plug, bottom base).
    - Vertical separator: rotate CW so left/right becomes top/bottom, solve, rotate back.
    """
    g = np.array(grid, dtype=int)
    _bg, _obj, ctrl = 1, 4, 7
    orient, _sep = _find_separator(g, ctrl)
    if orient == "h":
        return _solve_canonical(g)
    solved = _solve_canonical(np.rot90(g, -1))
    return _crop_non_bg(np.rot90(solved, 1), 1)


# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------


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
        frontier.append((nr, nc))
        count += 1


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
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


def _crop_sprite(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _build_canonical_sprites() -> tuple[np.ndarray, np.ndarray] | None:
    bg, obj, ctrl = 1, 4, 7

    base_h = _rng.randint(3, 6)
    base_w = _rng.randint(3, 8)
    base_target = _rng.randint(5, min(22, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates: list[tuple[int, list[int]]] = []
    for r in range(1, base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            if any(r + 1 < base_h and base[r + 1, c] for c in cols):
                continue
            candidates.append((r, cols))
    if not candidates:
        return None
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[conn_row, c] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_h = _rng.randint(2, 6)
    plug_w = _rng.randint(max(conn_len, 2), min(8, base_w + 3))
    if plug_h * plug_w < 5:
        return None
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = conn_cols[0] - plug_conn_c0

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        plug[1, c] = True

    plug_target = _rng.randint(5, min(22, plug_h * plug_w))
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep)

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if not plug[1, c]:
            return None

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = conn_row + r
            dc = dc_offset + c
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == obj:
                if r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len:
                    continue
                return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if plug_grid[1, c] != obj:
            return None
        if c > 0 and c - 1 >= plug_conn_c0 and c - 1 < plug_conn_c0 + conn_len:
            pass
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    base_sprite = _crop_sprite(base_grid, bg)
    plug_sprite = _crop_sprite(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None
    if int(np.sum(base_sprite != bg)) < 5 or int(np.sum(plug_sprite != bg)) < 5:
        return None
    return plug_sprite, base_sprite


def _place_canonical_input(plug_sprite: np.ndarray, base_sprite: np.ndarray) -> np.ndarray | None:
    bg, ctrl = 1, 7
    ph, pw = map(int, plug_sprite.shape)
    bh, bw = map(int, base_sprite.shape)

    canvas_h = _rng.randint(max(9, ph + bh + 3), min(15, ph + bh + 6))
    canvas_w = _rng.randint(max(4, max(pw, bw)), min(14, max(pw, bw) + 4))
    if ph + bh + 3 > canvas_h or max(pw, bw) > canvas_w:
        return None

    sep_min = ph + 1
    sep_max = canvas_h - bh - 2
    if sep_min > sep_max:
        return None
    sep = _rng.randint(sep_min, sep_max)

    plug_r_max = sep - ph - 1
    base_r_min = sep + 2
    if plug_r_max < 0 or base_r_min > canvas_h - bh:
        return None

    plug_r = _rng.randint(0, plug_r_max)
    base_r = _rng.randint(base_r_min, canvas_h - bh)
    plug_c = _rng.randint(0, canvas_w - pw)
    base_c = _rng.randint(0, canvas_w - bw)

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    canvas[sep, :] = int(ctrl)
    paste(canvas, plug_sprite, plug_r, plug_c, transparent=bg)
    paste(canvas, base_sprite, base_r, base_c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical_sprites()
    if canonical is None:
        return None
    plug_sprite, base_sprite = canonical
    canvas = _place_canonical_input(plug_sprite, base_sprite)
    if canvas is None:
        return None

    if _rng.random() < 0.5:
        sample = canvas
    else:
        sample = np.rot90(canvas, 1)

    try:
        out = main(sample)
    except Exception:
        return None
    if out.size == 0 or not np.any(out != 1):
        return None
    return sample


def generate_input() -> np.ndarray:
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
