# Variant of bf45cf4b
# Dimension: D8 — 输入中对象的数量与布局 (src)
# Change: 恰好2个前景连通分量（8-连通），在网格中随机放置，8-连通不相邻（至少1像素间隔）... → 控制精灵和瓦片相邻放置（4-连通相邻但颜色不同）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for the D8 variant
# ----------------------------------
# design_analysis:
#   a_specific_behavior:
#     input_now:
#       - Foreground is no longer split into two separated multicolor components.
#       - The monochrome control sprite and the multicolor square tile are flush-adjacent:
#         one tile bbox side touches one control bbox side with zero gap.
#       - They touch by 4-neighborhood only; colors differ, so the union is one foreground
#         component under multicolor 8-connectivity, but the control color is still isolatable.
#     output:
#       - Unchanged from the base task: every filled control pixel expands into one full copy
#         of the tile, every hole expands into a tile-sized background block.
#     traced_example:
#       - Original: multicolor foreground components = [tile, ctrl], so sorting the two
#         components is enough.
#       - Variant: multicolor foreground components = [tile+ctrl merged together], so the old
#         `len(comps) == 2` assumption fails immediately.
#       - New logic: enumerate foreground colors, find the unique color whose tight bbox contains
#         only itself + bg and has at least one hole; treat that bbox as ctrl. Remove that color,
#         and the remaining foreground collapses to exactly one square multicolor tile.
#   b_complexity_calibration:
#     - The variant visibly changes the input because the two objects are glued together.
#     - We keep only one control color and 2-3 tile colors; this is enough to force color-based
#       segmentation without introducing arbitrary color mappings.
#     - The tile remains square and multicolor, so the old disambiguation signal still exists.
#   c_implicit_assumptions_in_original_main:
#     - `get_bg(g)`: still valid only if background is most common. Generator keeps a roomy
#       12-18 canvas so bg dominates.
#     - `find_components(... monochromatic=False)` returning exactly 2 components: no longer
#       valid, because adjacency merges tile and ctrl into one foreground component.
#     - `crop_bbox` on a chosen object: still valid if the two objects only touch externally and
#       never overlap. Generator places them flush side-to-side/up-down with no overlap.
#     - `ctrl` being recoverable from a single foreground color: valid only if tile colors never
#       reuse the control color. Generator enforces disjoint palettes.
#     - Remaining foreground after removing ctrl color forming one square object: valid only if
#       the tile is itself connected as a multicolor object. Generator keeps the tile border
#       non-background so the tile stays connected even when it has internal bg holes.
#   d_placement_legality:
#     - Objects are first assembled in a tight local canvas with exact side contact.
#     - The touch side is chosen from a control boundary that actually has filled pixels, so at
#       least one ctrl cell touches a non-bg tile border cell.
#     - The local pair is then embedded into a 12-18 full grid without clipping or overlap.
#   e_visual_observability:
#     - Across examples, the solver-visible change is obvious: the source no longer has a gap.
#     - Humans can still infer the same output rule because the control remains monochrome and the
#       tile remains a compact multicolor square; only the segmentation cue changes from space to color.
#   f_visual_cleanliness:
#     - No overlapping paint layers are used; contact is external only.
#     - The pair is tightly cropped before embedding, so the touch relationship reads clearly.
#   g_color_inference:
#     - No recoloring rule is added. Colors are only structural cues.
#     - Control uses exactly one color not present in the tile; tile uses 2-3 colors.


def _extract_ctrl_and_block(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover ctrl by color, then recover the tile from the remaining foreground."""
    fg_cols = colors_present(grid, exclude={bg})
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]] = []

    for col in fg_cols:
        comps = find_components(
            grid,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[col],
            return_format="dict",
        )
        if len(comps) != 1:
            continue

        ctrl_bbox = comps[0]["bbox"]
        ctrl_patch = crop_bbox(grid, ctrl_bbox)
        if colors_present(ctrl_patch, exclude={bg}) != [int(col)]:
            continue
        if ctrl_patch.shape[0] < 3 or ctrl_patch.shape[1] < 3:
            continue
        if not np.any(ctrl_patch == bg):
            continue

        rem = np.array(grid, copy=True)
        rem[rem == int(col)] = int(bg)
        rem_comps = find_components(rem, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
        if len(rem_comps) != 1:
            continue

        block = crop_bbox(rem, rem_comps[0]["bbox"])
        block_cols = colors_present(block, exclude={bg})
        if len(block_cols) < 2:
            continue
        if block.shape[0] != block.shape[1]:
            continue

        # Prefer the strongest base-task signatures if multiple colors somehow pass:
        # monochrome ctrl with holes, and a tile with more colors and square shape.
        score = (
            len(block_cols),
            int(ctrl_patch.shape[0] * ctrl_patch.shape[1]),
            int(block.shape[0]),
            -int(col),
        )
        candidates.append((score, ctrl_patch, block))

    if not candidates:
        raise ValueError("no valid control/tile split found")

    candidates.sort(reverse=True, key=lambda item: item[0])
    _score, ctrl, block = candidates[0]
    return ctrl, block


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver for the touching-object version of bf45cf4b.

    The transformation itself is unchanged. Only object extraction changes:
    the control sprite and the tile may be flush-adjacent, so we separate them
    by color identity rather than by overall foreground connected-components.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrl, block = _extract_ctrl_and_block(g, bg)
    ctrl_cols = colors_present(ctrl, exclude={bg})
    if len(ctrl_cols) != 1:
        raise ValueError("control sprite must be monochrome")

    ctrl_col = int(ctrl_cols[0])
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    mask = (ctrl == int(ctrl_col))
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, block, r * bh, c * bw)
    return out


def _dihedral_orbits(size: int) -> list[tuple[list[tuple[int, int]], bool]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[list[tuple[int, int]], bool]] = []
    s = int(size)
    for r in range(s):
        for c in range(s):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (c, s - 1 - r),
                (s - 1 - r, s - 1 - c),
                (s - 1 - c, r),
                (r, s - 1 - c),
                (s - 1 - r, c),
                (c, r),
                (s - 1 - c, s - 1 - r),
            }
            cells = sorted(orbit)
            seen.update(cells)
            border = any(rr in (0, s - 1) or cc in (0, s - 1) for rr, cc in cells)
            out.append((cells, border))
    return out


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    n_tile_colors = int(rng.integers(2, min(4, len(tile_colors) + 1)))
    palette = list(rng.choice(tile_colors, size=n_tile_colors, replace=False))
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.full((size, size), int(bg), dtype=int)
    assignments: dict[int, int] = {i: int(color) for i, color in enumerate(palette)}

    interior_idxs = [i for i, (_cells, border) in enumerate(orbits) if not border and i not in assignments]
    if interior_idxs and rng.random() < 0.25:
        assignments[int(rng.choice(interior_idxs))] = int(bg)

    for i, (_cells, border) in enumerate(orbits):
        if i in assignments:
            continue
        fill_choices = palette if border else palette + ([bg] if interior_idxs else [])
        assignments[i] = int(rng.choice(fill_choices))

    for i, (cells, _border) in enumerate(orbits):
        for r, c in cells:
            tile[r, c] = assignments[i]
    return tile


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    area = h * w
    min_fill = (area + 1) // 2
    max_fill = int(np.floor(0.72 * area))
    target = int(rng.integers(min_fill, max_fill + 1))

    ar = int(rng.integers(0, h))
    ac = int(rng.integers(0, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, ac] = True
    mask[ar, :] = True

    while int(mask.sum()) < target:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                    frontier.append((r, c))
        pick = frontier[int(rng.integers(len(frontier)))]
        mask[pick] = True

    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(color)
    return ctrl


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _ctrl_touch_count(grid: np.ndarray, bg: int, ctrl_color: int) -> int:
    H, W = grid.shape
    total = 0
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != int(ctrl_color):
                continue
            if r + 1 < H:
                other = int(grid[r + 1, c])
                if other != int(bg) and other != int(ctrl_color):
                    total += 1
            if c + 1 < W:
                other = int(grid[r, c + 1])
                if other != int(bg) and other != int(ctrl_color):
                    total += 1
    return total


def _make_touching_pair(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape

    side_rows = {
        "left": np.where(ctrl[:, 0] != bg)[0],
        "right": np.where(ctrl[:, cw - 1] != bg)[0],
    }
    side_cols = {
        "up": np.where(ctrl[0, :] != bg)[0],
        "down": np.where(ctrl[ch - 1, :] != bg)[0],
    }
    sides = [name for name, rows in side_rows.items() if len(rows) > 0]
    sides += [name for name, cols in side_cols.items() if len(cols) > 0]
    side = str(rng.choice(sides))

    ctrl_r0 = 0
    ctrl_c0 = 0
    if side == "left":
        target_row = int(rng.choice(side_rows[side]))
        tile_hit = int(rng.integers(0, th))
        tile_r0 = target_row - tile_hit
        tile_c0 = -tw
    elif side == "right":
        target_row = int(rng.choice(side_rows[side]))
        tile_hit = int(rng.integers(0, th))
        tile_r0 = target_row - tile_hit
        tile_c0 = cw
    elif side == "up":
        target_col = int(rng.choice(side_cols[side]))
        tile_hit = int(rng.integers(0, tw))
        tile_r0 = -th
        tile_c0 = target_col - tile_hit
    else:
        target_col = int(rng.choice(side_cols[side]))
        tile_hit = int(rng.integers(0, tw))
        tile_r0 = ch
        tile_c0 = target_col - tile_hit

    min_r = min(ctrl_r0, tile_r0)
    min_c = min(ctrl_c0, tile_c0)
    ctrl_r0 -= min_r
    ctrl_c0 -= min_c
    tile_r0 -= min_r
    tile_c0 -= min_c

    pair_h = max(ctrl_r0 + ch, tile_r0 + th)
    pair_w = max(ctrl_c0 + cw, tile_c0 + tw)
    pair = np.full((pair_h, pair_w), int(bg), dtype=int)
    paste(pair, tile, tile_r0, tile_c0)
    paste(pair, ctrl, ctrl_r0, ctrl_c0, transparent=bg)
    return pair


def _embed_pair(pair: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    ph, pw = pair.shape
    H = int(rng.integers(max(12, ph), 19))
    W = int(rng.integers(max(12, pw), 19))
    r0 = int(rng.integers(0, H - ph + 1))
    c0 = int(rng.integers(0, W - pw + 1))

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, pair, r0, c0)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = list(rng.choice(np.arange(10, dtype=int), size=5, replace=False))
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_palette = [int(x) for x in palette[2:]]

    tile = _make_tile(bg, tile_palette, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    expected = _compose_output(tile, ctrl, bg)

    pair = _make_touching_pair(tile, ctrl, bg, rng)
    inp = _embed_pair(pair, bg, rng)

    fg_union = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(fg_union) != 1:
        return None
    if _ctrl_touch_count(inp, bg, ctrl_color) <= 0:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
