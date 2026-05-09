# Combo variant of bf45cf4b (index 10)
# combo_spec: D3/1+D6/0
# Name: 矩形瓦片奇偶翻转铺贴

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for combo D3/1 + D6/0
# --------------------------------------
# a) Concrete behavior and trace:
#    - Original task: identify one monochrome control sprite and one multicolor
#      tile, then stamp the same tile into every filled control cell.
#    - Fusion task: the tile is now always a non-square rectangle, and the copy
#      stamped at control cell (r,c) depends on parity:
#        * r odd  -> horizontal mirror (left/right)
#        * c odd  -> vertical mirror (up/down)
#        * both odd -> both mirrors
#      Using 0-based indexing gives four observable variants over the control
#      lattice: original / LR / UD / LR+UD.
#    - Small trace with a 2x3 tile T and a 3x3 control:
#        ctrl(0,0)=fill -> place T
#        ctrl(0,1)=fill -> place flipud(T)
#        ctrl(1,0)=fill -> place fliplr(T)
#        ctrl(1,1)=fill -> place flipud(fliplr(T))
#      Compared with the original, output size is still
#      (ctrl_h*tile_h, ctrl_w*tile_w), but repeated motifs alternate by row/col.
#
# b) Complexity calibration:
#    - The task stays learnable because object roles remain clean:
#      one monochrome rectangle-with-holes vs one multicolor rectangle.
#    - Freedom is deliberately bounded:
#      tile sizes come from a small catalog of non-square rectangles;
#      only two parity bits control orientation; the tile uses 2-3 colors.
#    - Generator also forces filled control cells to appear in all four parity
#      classes so the alternation is supported by repetition inside one sample.
#
# c) Original implicit assumptions that break and how they are repaired:
#    - "Prefer square tile" is invalid: tile is now always non-square, so solver
#      ranks candidates by multicolor-ness first and non-squareness second.
#    - "Every stamped copy is identical" is invalid: stamping now goes through
#      `_tile_variant(tile, r, c)` before `paste(...)`.
#    - `block.shape` was already general; the original solver did not assume
#      square scaling in the loop, only in object identification.
#
# d) Placement legality:
#    - Generator still creates exactly two disconnected 8-connected foreground
#      components and places them with at least one bg row/column gap.
#    - The tile is fully filled, so its cropped bbox is exactly the intended
#      rectangle; the control is pasted transparently so holes remain bg.
#
# e) Visual observability:
#    - Non-square tiles make row/column scale factors visibly different.
#    - The tile is required to have four distinct parity transforms, otherwise
#      the checkerboard rule could collapse visually.
#    - The control must use all four parity classes among filled cells, so all
#      four orientations appear in the output rather than only one or two.
#
# f) Visual cleanliness:
#    - Only two isolated objects appear in the input.
#    - The tile is a compact rectangle, not a ragged cluster, so the repeated
#      motif reads as a clean tile rather than an arbitrary blob.
#
# g) Change inferability:
#    - No color remapping is introduced. Colors are copied directly from the
#      tile; only orientation changes with position.
#    - Restricting the tile to 2-3 colors keeps mirrored differences legible.
#
# h) Quantitative anti-degeneration target:
#    - Every accepted sample must satisfy:
#        1. tile_h != tile_w
#        2. len({T, LR(T), UD(T), LR+UD(T)}) == 4
#        3. filled control cells cover parity classes {(0,0),(0,1),(1,0),(1,1)}
#        4. fused output != "plain repeated tile" output
#      Condition (4) proves the generator really exercises the new D6 rule.
#
# i) New solver assumptions audited against the generator:
#    - Assumption: exactly two fg components exist.
#      Guarantee: `_gen_attempt` rejects anything else.
#    - Assumption: one component is monochrome and one has >=2 fg colors.
#      Guarantee: tile construction and validation enforce this.
#    - Assumption: the control bbox is the intended lattice.
#      Guarantee: control is generated directly as a rectangular patch with holes.
#    - Assumption: parity is interpreted on control-bbox coordinates, not global
#      input coordinates.
#      Guarantee: solver crops the control component before iterating.


def _tile_variant(tile: np.ndarray, r: int, c: int) -> np.ndarray:
    """Apply the fused parity rule at control-lattice position (r,c)."""
    out = tile
    if int(r) % 2 == 1:
        out = np.fliplr(out)
    if int(c) % 2 == 1:
        out = np.flipud(out)
    return out


def _identify_tile_and_ctrl(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover the multicolor rectangular tile and the monochrome control patch.

    Ranking assumptions:
    - The tile has >=2 foreground colors.
    - The tile is non-square in this combo variant.
    - There are exactly two foreground components.
    """
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
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
    return infos[0]["patch"], infos[1]["patch"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a monochrome control sprite by stamping a multicolor rectangular tile.

    Fusion rule:
    - The tile is non-square.
    - Filled control cells stamp tile variants determined by control-row/control-
      column parity: odd rows flip horizontally, odd columns flip vertically.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    block, ctrl = _identify_tile_and_ctrl(g, bg)

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    mask = ctrl != int(bg)

    out = np.full((ch * bh, cw * bw), bg, dtype=int)
    for r in range(ch):
        for c in range(cw):
            if not bool(mask[r, c]):
                continue
            paste(out, _tile_variant(block, r, c), r * bh, c * bw)
    return out


def _tile_dims() -> list[tuple[int, int]]:
    return [(2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4)]


def _all_tile_variants(tile: np.ndarray) -> list[np.ndarray]:
    return [
        tile,
        np.fliplr(tile),
        np.flipud(tile),
        np.flipud(np.fliplr(tile)),
    ]


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    h, w = _tile_dims()[int(rng.integers(len(_tile_dims())))]
    n_colors = int(rng.integers(2, min(3, len(tile_colors)) + 1))
    colors = [int(x) for x in rng.choice(tile_colors, size=n_colors, replace=False)]

    for _ in range(200):
        tile = rng.choice(np.array(colors, dtype=int), size=(h, w), replace=True)

        # Force all chosen colors to appear at least once.
        anchors = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(anchors)
        for color, (r, c) in zip(colors, anchors):
            tile[r, c] = int(color)

        variants = _all_tile_variants(tile)
        if len({v.tobytes() for v in variants}) != 4:
            continue
        if len(colors_present(tile, exclude={bg})) < 2:
            continue
        return tile.astype(int)

    raise RuntimeError("failed to make a visibly asymmetric rectangular tile")


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _parity_classes(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r) % 2, int(c) % 2) for r, c in np.argwhere(mask)}


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        area = h * w
        min_fill = max(4, int(np.ceil(0.45 * area)))
        max_fill = min(area - 1, int(np.floor(0.72 * area)))
        if min_fill > max_fill:
            continue
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
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            mask[pick] = True

        if int(mask.sum()) != target:
            continue
        if int(mask.sum()) >= area:
            continue
        if len(_parity_classes(mask)) != 4:
            continue

        ctrl = np.full((h, w), bg, dtype=int)
        ctrl[mask] = int(color)
        return ctrl

    raise RuntimeError("failed to make control sprite with all parity classes")


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == int(bg):
                continue
            paste(out, _tile_variant(tile, r, c), r * th, c * tw)
    return out


def _compose_plain_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
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
        lh, lw = map(int, left_obj.shape)
        rh, rw = map(int, right_obj.shape)
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
        th, tw = map(int, top_obj.shape)
        bh, bw = map(int, bot_obj.shape)
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


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=6, replace=False)]
    bg = int(palette[0])
    fg = palette[1:]

    ctrl_color = int(fg[int(rng.integers(len(fg)))])
    tile_pool = [int(c) for c in fg if int(c) != ctrl_color]

    tile = _make_tile(bg, tile_pool, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)
    plain = _compose_plain_output(tile, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    shapes = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        colors_per_comp.append(colors_present(patch, exclude={bg}))
        shapes.append(tuple(map(int, patch.shape)))

    if sorted(len(cols) for cols in colors_per_comp) != [1, len(colors_present(tile, exclude={bg}))]:
        return None
    if not any(h != w for h, w in shapes):
        return None
    if len({v.tobytes() for v in _all_tile_variants(tile)}) != 4:
        return None
    if len(_parity_classes(ctrl != bg)) != 4:
        return None
    if np.array_equal(expected, plain):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
