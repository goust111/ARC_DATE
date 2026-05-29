# Variant of bf45cf4b
# Dimension: D3 — 瓦片的内部结构 (transform)
# Change: 瓦片为正方形（2-5×2-5），包含≥2种前景色，具有左右和上下对称性... → 瓦片为非正方形矩形

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior:
#    - Input still contains exactly two 8-connected foreground components:
#      a monochrome control sprite with holes, and one multicolor tile patch.
#    - The only semantic change is that the tile patch is now always a rectangle
#      with unequal height and width, e.g. 2x3, 3x5, 4x2. It remains LR+UD
#      symmetric and uses >=2 foreground colors.
#    - Output construction is unchanged at a high level: every filled control
#      pixel expands to one full tile copy, every hole expands to a bg rectangle.
#      The visible difference is that row scaling and column scaling are now
#      distinct and must be inferred separately.
#    - Small trace:
#      original: 3x3 square tile + 2x4 control -> output size (2*3)x(4*3)=6x12.
#      variant: 2x4 rectangular tile + 2x4 control -> output size (2*2)x(4*4)=4x16.
#      The stamping loop is the same, but height and width multipliers diverge.
#
# b) Complexity calibration:
#    - The task stays recognizable because the control object is still monochrome
#      and the tile is still polychrome. The extra freedom is limited to a small
#      catalog of rectangle sizes (each side in 2..5, but h!=w), so the rule is
#      observable rather than arbitrary.
#    - Samples include both tall and wide rectangles so humans see that there is
#      no single square scale factor.
#
# c) Implicit assumptions in the original main():
#    - `find_components(... monochromatic=False)` assumes the tile and control are
#      disconnected as whole foreground components. This remains valid and the
#      generator enforces a >=1-cell bg gap.
#    - `colors_present(patch, exclude={bg})` assumes the control has one fg color
#      and the tile has >=2 fg colors. This remains valid and is enforced.
#    - The old tie-break preferred square patches. That is no longer aligned with
#      the variant; the new solver instead prefers non-square patches if a tie
#      ever occurs.
#    - `bh, bw = block.shape` was already general enough. The hidden risk was not
#      the stamping loop, but misidentifying the block because "square" used to be
#      part of the ranking. That ranking is updated.
#
# d) Placement legality:
#    - Generator builds tile/control first, then places them into a 12..18 grid
#      with horizontal or vertical separation and at least one full bg column/row
#      between them. This prevents overlap and accidental 8-connectivity merging.
#
# e) Visual observability:
#    - Every generated tile is guaranteed non-square, so the output's repeated
#      motif has rectangular cells, not square cells.
#    - Control shapes are at least 3x3 with holes, making repeated row-vs-column
#      scaling visible across multiple copies.
#
# f) Visual cleanliness:
#    - Objects are still isolated and non-overlapping in the input.
#    - The tile uses a symmetric border-first construction so its bbox reads as
#      one clean rectangle rather than a ragged cluster.
#
# g) Color inferability:
#    - Colors are not remapped. The tile is copied verbatim, so the only thing to
#      infer is geometry. Keeping the tile at 2-3 colors avoids visual clutter.
#
# h) Variant metric for post-test auditing:
#    - `tile_h != tile_w` must hold for 100% of generated samples.
#    - We also track whether the rectangle is tall or wide; both branches should
#      appear over a batch so the variant does not collapse to one aspect family.
#
# i) New solver assumptions and generator guarantees:
#    - Assumption: there are exactly two foreground components. Guaranteed by
#      placement validation in `_gen_attempt`.
#    - Assumption: one component is multicolor and one is monochrome. Guaranteed
#      by tile/control construction and checked before accepting a sample.
#    - Assumption: the tile bbox is the intended rectangle. Guaranteed by forcing
#      the tile's outer border to be foreground, so cropping recovers full size.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale up a monochrome control sprite by stamping a multicolor rectangular tile.

    Variant-specific cue:
    - The block is no longer restricted to be square; output height and width use
      the tile's height and width independently.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"patch": patch, "cols": cols})

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["patch"].shape[0] != d["patch"].shape[1]),
        ),
        reverse=True,
    )
    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), bg, dtype=int)

    ctrl_col = int(ctrl_cols[0])
    mask = (ctrl == ctrl_col)

    out = np.full((ch * bh, cw * bw), bg, dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, block, r * bh, c * bw)
    return out


def _reflection_orbits(h: int, w: int) -> list[tuple[list[tuple[int, int]], bool]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }
            cells = sorted(orbit)
            seen.update(cells)
            border = any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in cells)
            out.append((cells, border))
    return out


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    dims = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4)]
    h, w = dims[int(rng.integers(len(dims)))]
    n_tile_colors = int(rng.integers(2, min(3, len(tile_colors)) + 1))
    colors = [int(x) for x in rng.choice(tile_colors, size=n_tile_colors, replace=False)]
    orbits = _reflection_orbits(h, w)

    tile = np.full((h, w), bg, dtype=int)
    assignments: dict[int, int] = {}

    border_idxs = [i for i, (_cells, border) in enumerate(orbits) if border]
    inner_idxs = [i for i, (_cells, border) in enumerate(orbits) if not border]
    assert len(border_idxs) >= 2

    assignments[border_idxs[0]] = colors[0]
    assignments[border_idxs[1]] = colors[1]

    if inner_idxs and rng.random() < 0.3:
        assignments[int(rng.choice(inner_idxs))] = bg

    for i, (_cells, border) in enumerate(orbits):
        if i in assignments:
            continue
        palette = colors if border else colors + [bg]
        assignments[i] = int(rng.choice(palette))

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
    h = int(rng.integers(3, 7))
    w = int(rng.integers(3, 7))
    area = h * w
    min_fill = max(3, int(np.ceil(0.45 * area)))
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

    ctrl = np.full((h, w), bg, dtype=int)
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


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    if horizontal:
        if rng.random() < 0.5:
            left_obj, right_obj = tile, ctrl
        else:
            left_obj, right_obj = ctrl, tile
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        H = int(rng.integers(max(12, max(lh, rh)), 19))
        W = int(rng.integers(max(12, lw + rw + gap), 19))
        slack = W - (lw + rw + gap)
        c_left = int(rng.integers(0, slack + 1))
        c_right = int(rng.integers(c_left + lw + gap, W - rw + 1))
        r_left = int(rng.integers(0, H - lh + 1))
        r_right = int(rng.integers(0, H - rh + 1))
        if left_obj is tile:
            tile_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            tile_pos = (r_right, c_right)
    else:
        if rng.random() < 0.5:
            top_obj, bot_obj = tile, ctrl
        else:
            top_obj, bot_obj = ctrl, tile
        th, tw = top_obj.shape
        bh, bw = bot_obj.shape
        H = int(rng.integers(max(12, th + bh + gap), 19))
        W = int(rng.integers(max(12, max(tw, bw)), 19))
        slack = H - (th + bh + gap)
        r_top = int(rng.integers(0, slack + 1))
        r_bot = int(rng.integers(r_top + th + gap, H - bh + 1))
        c_top = int(rng.integers(0, W - tw + 1))
        c_bot = int(rng.integers(0, W - bw + 1))
        if top_obj is tile:
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    fg = palette[1:]

    ctrl_color = int(fg[int(rng.integers(len(fg)))])
    tile_pool = [int(c) for c in fg if int(c) != ctrl_color]

    tile = _make_tile(bg, tile_pool, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)

    if tile.shape[0] == tile.shape[1]:
        return None

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    shapes = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))
        shapes.append(tuple(map(int, patch.shape)))

    if sorted(len(cols) for cols in colors_per_comp) != [1, len(set(int(x) for x in tile.flatten()) - {bg})]:
        return None
    if not any(h != w for h, w in shapes):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
