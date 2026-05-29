# Variant of bf45cf4b
# Dimension: D13 — 瓦片颜色在放大输出中的处理 (transform)
# Change: 瓦片颜色原样保留，输出中每个放置位置的瓦片颜色完全相同... → 瓦片颜色被控制精灵颜色替换其中一种

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for the D13 variant
# -----------------------------------
# a) Concrete behavior:
#    - Input still has exactly two disconnected foreground components:
#      one monochrome control sprite with bg holes, and one multicolor square tile.
#    - Output still expands each filled control cell into one tile-sized block.
#    - The only semantic change is tile recoloring before stamping:
#      replace the tile's corner color (the color occupying tile[0, 0]) with the
#      control sprite color, then stamp that recolored tile everywhere.
#    - This keeps D13 as a pure output-color-processing change rather than a new
#      recognition or layout rule.
# b) Why corner color:
#    - The YAML only requires that one tile color be replaced by the control color.
#    - Using the corner color makes the changed subset structurally salient across
#      samples: a human can compare input tile corners against output tile corners
#      and infer the recoloring rule without extra markers.
# c) Hidden assumptions audited:
#    - The original component identification still works unchanged.
#    - The new solver assumes the tile corner is a foreground color and appears
#      multiple times. The generator guarantees a symmetric tile with all four
#      corners sharing the same non-background accent color.
# d) Degeneration avoidance:
#    - The generator picks the control color from outside the tile palette.
#    - The corner color occurs on all four corners, so recoloring is always visible.
# e) Visual cleanliness:
#    - No extra symbols or objects are introduced; only one coherent tile color
#      class changes in the output.


def _identify_parts(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
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
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    block = np.array(infos[0]["patch"], dtype=int)
    ctrl = np.array(infos[1]["patch"], dtype=int)
    return bg, block, ctrl


def _recolored_tile(tile: np.ndarray, ctrl_color: int) -> np.ndarray:
    src_color = int(tile[0, 0])
    assert src_color != int(ctrl_color)
    out = np.array(tile, copy=True)
    out[out == src_color] = int(ctrl_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    bg, block, ctrl = _identify_parts(grid)
    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)
    ctrl_color = int(ctrl_cols[0])
    mask = (ctrl == ctrl_color)
    tile = _recolored_tile(block, ctrl_color)

    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if bool(mask[r, c]):
                paste(out, tile, r * th, c * tw)
    return out


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


def _make_tile(bg: int, ctrl_color: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    tile_colors = list(rng.choice(fg_colors, size=3, replace=False))
    corner_color = int(tile_colors[0])
    base_color = int(tile_colors[1])
    stripe_color = int(tile_colors[2])

    tile = np.full((size, size), base_color, dtype=int)
    tile[0, 0] = corner_color
    tile[0, -1] = corner_color
    tile[-1, 0] = corner_color
    tile[-1, -1] = corner_color

    mid = size // 2
    tile[mid, :] = stripe_color
    tile[:, mid] = stripe_color

    # Restore the corner accent after drawing the cross.
    tile[0, 0] = corner_color
    tile[0, -1] = corner_color
    tile[-1, 0] = corner_color
    tile[-1, -1] = corner_color

    assert int(tile[0, 0]) != int(ctrl_color)
    assert len(colors_present(tile, exclude={bg})) >= 2
    assert np.count_nonzero(tile == tile[0, 0]) >= 4
    return tile


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, ctrl_color: int) -> np.ndarray:
    recolored = _recolored_tile(tile, ctrl_color)
    th, tw = recolored.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] == ctrl_color:
                paste(out, recolored, r * th, c * tw)
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
    palette = list(rng.choice(np.arange(10, dtype=int), size=6, replace=False))
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    fg_colors = [int(x) for x in palette[2:]]

    tile = _make_tile(bg, ctrl_color, fg_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg, ctrl_color)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None
    colors_per_comp = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))
    if sorted(len(cols) for cols in colors_per_comp) != [1, len(set(tile.flatten()) - {bg})]:
        return None

    original = np.full_like(expected, bg)
    th, tw = tile.shape
    for r in range(ctrl.shape[0]):
        for c in range(ctrl.shape[1]):
            if ctrl[r, c] == ctrl_color:
                paste(original, tile, r * th, c * tw)
    if np.array_equal(original, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid recolor variant input")
