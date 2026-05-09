# Variant of bf45cf4b
# Dimension: D7 — 空洞区域的填充方式 (transform)
# Change: 控制精灵中的空洞位置填充瓦片大小的纯背景色区域... → 空洞位置填充瓦片的单色版本（使用控制精灵颜色）

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

"""
Design analysis for the D7 variant
==================================

spec_variant:
  unchanged_input:
    grid: 10-18 rows x 10-18 cols
    objects: exactly 2 disconnected foreground components under 8-connectivity
    control_sprite:
      type: monochrome rectangular bbox with bg holes
      size: 3-5 rows x 3-5 cols
      color: single foreground color, distinct from tile colors
      holes: guaranteed >= 1
    building_block:
      type: mostly solid multicolor square patch
      size: 3-5 x 3-5
      colors: 2-3 foreground colors, all distinct from control color
      may_contain_bg_inside: rare, but if present the monochrome hole-fill must preserve that silhouette
  changed_transform:
    old:
      filled control cell -> paste original multicolor tile
      hole control cell -> leave an all-bg tile-sized region
    new:
      filled control cell -> paste original multicolor tile
      hole control cell -> paste a monochrome version of the tile:
        same tile silhouette (all non-bg cells kept in place)
        every non-bg tile cell recolored to the control color
  output:
    size: (control_h * tile_h) x (control_w * tile_w)
    crop: tight by construction

a) Concrete behavior trace
--------------------------
Small example:
  block =
    2 3
    3 2
  ctrl =
    8 0
    0 8

Old output:
  [block][bg bg]
  [bg bg][block]

New output:
  [block][8 8]
  [8 8][block]

If the tile contains internal bg cells, "monochrome version" means:
  mono_block = where(block != bg, ctrl_color, bg)
So the hole cell still receives the tile's shape, just recolored to the control color.

b) Complexity calibration
-------------------------
- The input grammar stays the same, so object detection is not made harder in unrelated ways.
- The tile colors are forced to exclude the control color. Without this, the new hole-fill could visually blend
  with the original tile and make the D7 change hard to notice from few examples.
- Control masks still contain several holes, so the new branch executes repeatedly instead of being a rare corner case.

c) Old implicit assumptions audited
-----------------------------------
Relevant old solver lines and their assumptions:
- `out = full(..., bg)`: assumed hole positions should remain blank. This is no longer valid.
- `if mask[r, c]: paste(out, block, ...)`: assumed only filled cells produce visible output. This is no longer valid.
- Component finding, sorting by color-count/squareness, bbox cropping, and control-color extraction remain valid because
  the input format itself does not change in this variant.

New safe replacement:
- Precompute `mono_block = where(block != bg, ctrl_col, bg)`.
- For every control-bbox cell, paste `block` if filled else `mono_block`.

d) Placement legality
---------------------
- Generator still places exactly two disconnected input components with >= 1 bg gap.
- Output is assembled directly in a pre-sized array, so no overlap ambiguity exists there.
- The tile is cropped from its own bbox, and the control sprite is pasted transparently on input so its internal holes
  remain bg and do not create extra components.

e) Visual observability
-----------------------
- The crucial observable difference is that output regions corresponding to control holes are no longer empty.
- To make that obvious in 2-4 examples, the generator guarantees at least one hole and uses a control color not present
  in the tile. Human solvers can then see two repeated fill modes: multicolor tile vs single-color tile.

f) Visual cleanliness
---------------------
- No extra markers or overlays are added; the grid still contains only one repeated tile lattice.
- Because the control color is absent from the original tile palette, monochrome hole-tiles remain visually separable.

g) Color inference
------------------
- The recolor rule is simple: every hole tile uses exactly the control sprite color.
- The control sprite is the only monochrome input object, so the color source is localized and repeatable.

Post-change solver assumptions that generator must satisfy
----------------------------------------------------------
1. Exactly two multicolor-connected foreground components exist in the input.
2. One component is monochrome; the other has >= 2 foreground colors and is square-preferred.
3. The control sprite contains at least one bg hole so the D7 branch is exercised.
4. The tile has at least one non-bg cell, and its colors exclude the control color so the two output modes stay visible.
"""


def _mono_tile(block: np.ndarray, bg: int, ctrl_color: int) -> np.ndarray:
    return np.where(block != int(bg), int(ctrl_color), int(bg)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale up a monochrome control sprite by stamping either:
    - the original multicolor tile at filled control pixels, or
    - a monochrome version of that tile at control-hole pixels.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"bbox": comp["bbox"], "patch": patch, "cols": cols})

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)

    ctrl_col = int(ctrl_cols[0])
    mono_block = _mono_tile(block, bg, ctrl_col)

    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = block if int(ctrl[r, c]) == ctrl_col else mono_block
            paste(out, patch, r * bh, c * bw)
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
    chosen = [int(x) for x in rng.choice(tile_colors, size=n_tile_colors, replace=False)]
    orbits = _dihedral_orbits(size)
    rng.shuffle(orbits)

    tile = np.full((size, size), int(bg), dtype=int)
    assignments: dict[int, int] = {i: color for i, color in enumerate(chosen)}

    interior_idxs = [i for i, (_cells, border) in enumerate(orbits) if not border and i not in assignments]
    allow_bg_orbit = bool(interior_idxs) and bool(rng.random() < 0.15)
    if allow_bg_orbit:
        assignments[int(rng.choice(interior_idxs))] = int(bg)

    for i, (_cells, border) in enumerate(orbits):
        if i in assignments:
            continue
        palette = chosen if border or not allow_bg_orbit else chosen + [int(bg)]
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
    ctrl_colors = colors_present(ctrl, exclude={bg})
    assert len(ctrl_colors) == 1
    ctrl_color = int(ctrl_colors[0])
    mono_block = _mono_tile(tile, bg, ctrl_color)

    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = tile if int(ctrl[r, c]) == ctrl_color else mono_block
            paste(out, patch, r * th, c * tw)
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
        H = int(rng.integers(max(10, max(lh, rh)), 19))
        W = int(rng.integers(max(10, lw + rw + gap), 19))
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
        H = int(rng.integers(max(10, th + bh + gap), 19))
        W = int(rng.integers(max(10, max(tw, bw)), 19))
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
    ctrl_color = int(palette[1])
    tile_palette = [int(x) for x in palette[2:]]

    tile = _make_tile(bg, tile_palette, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))

    tile_fg_colors = sorted(set(int(x) for x in tile.flatten()) - {bg})
    if sorted(len(cols) for cols in colors_per_comp) != [1, len(tile_fg_colors)]:
        return None
    if ctrl_color in tile_fg_colors:
        return None
    if int(np.sum(ctrl == bg)) == 0:
        return None
    if not np.any(_mono_tile(tile, bg, ctrl_color) != bg):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
