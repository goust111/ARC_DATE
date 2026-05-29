# Variant of bf45cf4b
# Dimension: D6 — 瓦片放置时的变换操作 (transform)
# Change: 每个填充位置放置完全相同的瓦片副本，无任何变换... → 边缘位置和内部位置使用不同的瓦片变体

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for this D6 variant
# -----------------------------------
# a) Concrete behavior:
#    - Input still has exactly two 8-connected foreground components:
#      one monochrome control sprite with bg holes, and one polychrome square tile.
#    - Output still scales each filled control cell into one tile-sized block.
#    - The only rule change is conditional tile choice:
#        * if a filled control cell lies on the outer border of the control bbox
#          (top row, bottom row, left col, or right col), place a horizontally
#          flipped copy of the tile;
#        * otherwise place the original tile.
#    - Example on a 4x4 control bbox:
#        control fill at (0,2) used to stamp block B; now it stamps fliplr(B).
#        control fill at (2,2) used to stamp block B; it still stamps B.
#      The difference is localized to the output assembly step only.
# b) Complexity calibration:
#    - A pure color swap would be hard to infer from 2-4 examples, so the variant
#      uses a geometric transform with a visible relation to the input tile.
#    - To keep the transform observable, the generator makes the tile explicitly
#      left-right asymmetric and guarantees the control has both edge fills and
#      interior fills, so both tile variants appear in one sample.
# c) Implicit assumptions changed from the original solver:
#    - Original `paste(out, block, ...)` assumed every filled control pixel used
#      the same tile. That is no longer valid.
#    - Boundary detection is now meaningful only because `ctrl = crop_bbox(...)`
#      makes the control patch equal to its bbox; then `r in {0,ch-1}` / `c in
#      {0,cw-1}` really means "control bbox edge".
#    - The solver now assumes the tile transform is visible; the generator
#      guarantees `tile != fliplr(tile)` to prevent degeneration back to the
#      original task.
# d) Placement legality:
#    - The input generator still places exactly two disconnected components with
#      at least one background cell gap and at least one background border cell.
#    - The output-first oracle checks `main(inp) == expected`.
# e) Visual observability:
#    - Edge-vs-interior is easy to see because the control sprite itself exposes
#      which filled pixels touch the bbox border.
#    - The repeated mirrored blocks along the scaled perimeter create multiple
#      supports for the rule, instead of a one-off special case.
# f) Visual cleanliness:
#    - No extra markers or overlapping objects are introduced; only the repeated
#      tile copy changes.
#    - The tile remains a compact square patch, and the control remains a single
#      monochrome component with holes.
# g) Color inference:
#    - No new color mapping is introduced; the same tile colors are reused.
#      Human inference only needs to track a mirror transform, not a palette rule.
# Reference-task note:
#    - A solver file for source task 71e489b6 was not present anywhere in the
#      provided workspace, so this implementation calibrates only the stated
#      "boundary_detection" idea and does not import any other variant behavior.


def _is_edge_cell(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _transform_tile(tile: np.ndarray) -> np.ndarray:
    return np.fliplr(tile)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale up a monochrome control sprite by stamping a tile per filled pixel.

    D6 variant rule:
    - filled control cells on the control bbox border stamp a horizontally
      mirrored tile;
    - filled interior control cells stamp the original tile.
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
    assert len(ctrl_cols) == 1
    ctrl_col = int(ctrl_cols[0])
    mask = ctrl == ctrl_col

    edge_block = _transform_tile(block)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if not bool(mask[r, c]):
                continue
            patch = edge_block if _is_edge_cell(r, c, ch, cw) else block
            paste(out, patch, r * bh, c * bw)
    return out


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    n_tile_colors = int(rng.integers(2, min(4, len(fg_colors) + 1)))
    tile_colors = [int(x) for x in rng.choice(fg_colors, size=n_tile_colors, replace=False)]

    tile = rng.choice(np.array(tile_colors, dtype=int), size=(size, size)).astype(int)

    # Force a readable left-right asymmetry so fliplr(tile) is visibly different.
    tile[:, 0] = int(tile_colors[1])
    tile[:, -1] = int(tile_colors[0])
    if n_tile_colors >= 3:
        tile[0, 1:-1] = int(tile_colors[2])
        tile[-1, max(1, size - 2)] = int(tile_colors[2])
    else:
        tile[0, 1:-1] = int(tile_colors[1])
        tile[-1, 1::2] = int(tile_colors[1])

    assert len(colors_present(tile)) >= 2
    assert not np.array_equal(tile, _transform_tile(tile))
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
    h = int(rng.integers(4, 7))
    w = int(rng.integers(4, 7))
    area = h * w

    base_fill = h + w - 1
    min_fill = max(base_fill, int(np.ceil(0.45 * area)))
    max_fill = int(np.floor(0.72 * area))
    target = int(rng.integers(min_fill, max_fill + 1))

    ar = int(rng.integers(1, h - 1))
    ac = int(rng.integers(1, w - 1))
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

    interior = mask[1:-1, 1:-1]
    edge_fill_count = int(mask[0, :].sum() + mask[-1, :].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())
    assert interior.any()
    assert edge_fill_count > 0

    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(color)
    return ctrl


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    edge_tile = _transform_tile(tile)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] == bg:
                continue
            patch = edge_tile if _is_edge_cell(r, c, ch, cw) else tile
            paste(out, patch, r * th, c * tw)
    return out


def _compose_plain_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
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
        H = int(rng.integers(max(12, max(lh, rh) + 2), 19))
        W = int(rng.integers(max(12, lw + rw + gap + 2), 19))
        slack = W - (lw + rw + gap) - 2
        c_left = int(rng.integers(1, slack + 2))
        c_right = int(rng.integers(c_left + lw + gap, W - rw))
        r_left = int(rng.integers(1, H - lh))
        r_right = int(rng.integers(1, H - rh))
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
        H = int(rng.integers(max(12, th + bh + gap + 2), 19))
        W = int(rng.integers(max(12, max(tw, bw) + 2), 19))
        slack = H - (th + bh + gap) - 2
        r_top = int(rng.integers(1, slack + 2))
        r_bot = int(rng.integers(r_top + th + gap, H - bh))
        c_top = int(rng.integers(1, W - tw))
        c_bot = int(rng.integers(1, W - bw))
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
    tile_palette = palette[1:4]
    ctrl_color = int(palette[4])

    tile = _make_tile(bg, tile_palette, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)

    expected = _compose_output(tile, ctrl, bg)
    plain = _compose_plain_output(tile, ctrl, bg)
    if np.array_equal(expected, plain):
        return None

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    infos = []
    for comp in comps:
        patch = crop_bbox(inp, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append((patch, cols))

    color_counts = sorted(len(cols) for _patch, cols in infos)
    if color_counts[0] != 1 or color_counts[1] < 2:
        return None

    edge_mask = np.zeros_like(ctrl, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    filled = ctrl != bg
    if not np.any(filled & edge_mask):
        return None
    if not np.any(filled & ~edge_mask):
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
