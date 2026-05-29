# Combo variant of bf45cf4b (index 28)
# combo_spec: D14/1+D3/1
# Name: 非正方形瓦片显式非等比放大

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Input still has exactly two 8-connected foreground components:
#      one monochrome control sprite with holes, and one multicolor tile.
#    - D3/1 changes the tile itself: it must be a non-square rectangle, such as
#      2x4 or 3x5, so row and column evidence are already asymmetric.
#    - D14/1 then changes how one control pixel expands: we do not paste the raw
#      tile directly. Instead, we make an explicitly stretched tile whose height
#      stays `tile_h` while width becomes `2 * tile_w`.
#    - Small trace:
#      tile = 2x4, ctrl = 3x2.
#      Original task: each filled ctrl cell contributes one 2x4 copy, so output
#      size would be 6x8.
#      This combo: each filled ctrl cell contributes one 2x8 copy obtained by
#      duplicating every tile column horizontally, so output size is 6x16.
#      Hole cells still contribute equally sized bg rectangles.
#    - The difference is therefore not only "tile is rectangular", but also
#      "the repeated output atom is wider than the input tile".
#
# b) Complexity calibration:
#    - The rule remains human-solvable because there are still only two objects
#      with the same roles as the base task.
#    - The extra degree of freedom is controlled: the stretch is always a simple
#      horizontal x2 replication, not a large family of unrelated ratios.
#    - Every sample contains multiple tile copies in the output via the control
#      mask, so the wider repeated atom is visually reinforced.
#
# c) Original implicit assumptions and what changes:
#    - `find_components(..., monochromatic=False)` assumes the two semantic
#      objects stay disconnected as whole foreground components. Generator keeps
#      a >=1-cell bg gap, so this remains valid.
#    - Ranking the multicolor component above the monochrome component still
#      works. The old "prefer square" tie-break is wrong now; the solver instead
#      prefers non-square when color evidence ties.
#    - `block.shape` is no longer the output atom shape. The solver must derive a
#      second shape, `stretched.shape == (block_h, 2 * block_w)`, before tiling.
#    - `ctrl` is still read as a monochrome mask inside its bbox, so bg holes are
#      preserved exactly as before.
#
# d) Placement legality:
#    - Generator constructs tile/control first, then places them with horizontal
#      or vertical separation into a 12..18 by 12..18 bg canvas.
#    - The control is pasted transparently over bg only; the tile is pasted as a
#      solid rectangular patch. This prevents overlap and accidental merging.
#
# e) Visual observability:
#    - Non-square tiles guarantee that the input already exposes different row
#      and column extents.
#    - The output makes the second change obvious because each copied motif is
#      twice as wide as the tile visible in the input, not merely the same size.
#    - Using both tall and wide rectangles across samples prevents humans from
#      overfitting to a single aspect family.
#
# f) Visual cleanliness:
#    - The tile is generated with a fully foreground border, so its bbox is a
#      clean rectangle instead of a ragged multicolor cluster.
#    - The control remains one monochrome connected sprite with bg holes, so the
#      two input roles remain visually separable.
#
# g) Inferability of changes:
#    - No color remapping occurs. Colors are copied from the tile verbatim; only
#      geometry changes.
#    - Keeping the tile at 2-3 colors avoids unnecessary clutter while still
#      making the repeated horizontal duplication recognizable.
#
# h) Variant metrics for auditing after tests:
#    - `tile_h != tile_w` must hold for all accepted samples.
#    - `output_width == ctrl_w * 2 * tile_w` must hold for all accepted samples.
#    - The combo must not collapse to the D3-only rule; therefore the stretched
#      output must differ from the raw-tile output for every accepted sample.
#
# i) New solver assumptions and generator guarantees:
#    - Assumption: one component is monochrome and one is multicolor. Generator
#      enforces distinct control/tile palettes and validates accepted samples.
#    - Assumption: the multicolor component's cropped bbox is the full tile bbox.
#      Generator forces a foreground border to make cropping stable.
#    - Assumption: the intended stretch factor is horizontal x2. Generator audits
#      accepted samples against the unstretched baseline to ensure the fusion
#      behavior is actually present.


def _stretch_tile(block: np.ndarray) -> np.ndarray:
    """Explicit nonuniform scaling: keep height, double width."""
    return np.repeat(np.array(block, dtype=int), 2, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale up a monochrome control sprite by stamping a stretched rectangular tile.

    Variant-specific rule:
    - The tile is non-square.
    - Each filled control cell expands to a horizontally stretched tile copy with
      shape (tile_h, 2 * tile_w), not to the raw tile bbox.
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
    block = np.array(infos[0]["patch"], dtype=int)
    ctrl = np.array(infos[1]["patch"], dtype=int)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    assert len(ctrl_cols) == 1
    ctrl_col = int(ctrl_cols[0])
    mask = ctrl == ctrl_col

    stretched = _stretch_tile(block)
    sh, sw = map(int, stretched.shape)
    ch, cw = map(int, ctrl.shape)

    out = np.full((ch * sh, cw * sw), bg, dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, stretched, r * sh, c * sw)
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
    dims = [
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
        (3, 5),
        (5, 3),
        (4, 5),
        (5, 4),
    ]
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
    stretched = _stretch_tile(tile)
    sh, sw = map(int, stretched.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * sh, cw * sw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                paste(out, stretched, r * sh, c * sw)
    return out


def _compose_unstretched_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
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
    unstretched = _compose_unstretched_output(tile, ctrl, bg)

    if tile.shape[0] == tile.shape[1]:
        return None
    if expected.shape == unstretched.shape:
        return None
    if np.array_equal(expected, unstretched):
        return None
    if int(get_bg(inp)) != bg:
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
    if expected.shape[1] != ctrl.shape[1] * 2 * tile.shape[1]:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
