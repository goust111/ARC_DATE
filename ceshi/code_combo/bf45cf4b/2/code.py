# Combo variant of bf45cf4b (index 2)
# combo_spec: D1/0+D13/0
# Name: 填充替色空洞反色双模态
#
# Design analysis
# ===============
# concrete_behavior:
#   - Preserve the original input schema: exactly two disconnected 8-connected
#     foreground components on a bg canvas.
#     1) ctrl: a monochrome rectangular sprite with bg holes inside its bbox
#     2) block: a multicolor square tile
#   - Fusion rule:
#       a) infer the ctrl binary mask from ctrl vs bg
#       b) first recolor the tile by replacing one salient tile foreground color
#          with the ctrl color
#       c) stamp that recolored tile at filled ctrl cells
#       d) stamp the foreground-color-inverted version of that recolored tile at
#          hole cells
#   - The replaced tile color is concretized as the corner accent `tile[0, 0]`.
#     The inversion rule is concretized as foreground-only `c -> 9 - c`, while
#     bg would stay bg if it were present.
# behavior_trace:
#   - Example:
#       bg = 0
#       input tile =
#         [[2, 4, 2],
#          [4, 7, 4],
#          [2, 4, 2]]
#       ctrl color = 6
#       ctrl =
#         [[6, 0],
#          [0, 6]]
#   - Original task:
#       filled -> original tile
#       hole   -> blank bg tile
#   - This combo:
#       recolored(tile, 6) =
#         [[6, 4, 6],
#          [4, 7, 4],
#          [6, 4, 6]]
#       inverted(recolored) =
#         [[3, 5, 3],
#          [5, 2, 5],
#          [3, 5, 3]]
#       output =
#         [[recolored, inverted],
#          [inverted, recolored]]
#   - Difference from the original is visible in both branches:
#       1) filled blocks are not raw copies of the input tile
#       2) holes are no longer empty, but a second stable palette
# complexity_calibration:
#   - The rule still has only two objects and one spatial mask, so recognition
#     remains clean.
#   - The two semantic changes reinforce each other instead of acting on separate
#     regions independently: hole tiles are not inversion of the raw input tile,
#     but inversion of the already recolored fill tile.
#   - Generator constraints keep the hypothesis space small:
#       1) the replaced color is always the corner accent, repeated on all corners
#       2) ctrl has at least two filled cells and two holes
#       3) tile uses exactly three foreground colors, so the recolor is obvious
#       4) fill-palette and hole-palette are drawn from opposite complement pairs
#          and stay disjoint
# implicit_assumption_review:
#   - `find_components(...)=2`: still required; generator places only tile+ctrl.
#   - sorting by `(multicolor, n_colors, square)` still works because ctrl has
#     one foreground color and tile has three.
#   - `ctrl_cols[0]`: still assumes ctrl is monochrome; generator guarantees it.
#   - `tile[0,0]`: now meaningful; generator guarantees the corner is a repeated
#     foreground accent, never bg and never ctrl color.
#   - output assembly can no longer "skip holes"; every ctrl-bbox cell must
#     receive either the recolored tile or its inverted counterpart.
#   - inversion now depends on the recolored tile, not the original tile; the
#     solver must apply the transforms in the right order.
# placement_legality:
#   - Input placement remains horizontal or vertical with >=1 bg gap.
#   - Objects never overlap or leave the grid.
#   - ctrl is pasted transparently over bg so its holes remain genuine bg cells.
# visual_observability:
#   - Human solvers can compare:
#       1) input tile vs filled output blocks to infer the corner-color replacement
#       2) filled output blocks vs hole output blocks to infer the complement map
#   - Because all output blocks keep identical geometry, only color changes must
#     be inferred, which is practical from 2-3 examples.
# cleanliness_and_inference:
#   - No extra marker objects are introduced.
#   - Tile palettes are deliberately limited and structured, avoiding noisy
#     many-to-many recolor hypotheses.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


INV_PAIRS = ((0, 9), (1, 8), (2, 7), (3, 6), (4, 5))


def _invert_tile(tile: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(tile, dtype=int, copy=True)
    mask = (out != int(bg))
    out[mask] = 9 - out[mask]
    return out


def _identify_parts(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append({"patch": np.array(patch, dtype=int), "cols": cols})

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
    return bg, block, ctrl


def _recolored_tile(tile: np.ndarray, ctrl_color: int) -> np.ndarray:
    src_color = int(tile[0, 0])
    assert src_color != int(ctrl_color)
    out = np.array(tile, dtype=int, copy=True)
    out[out == src_color] = int(ctrl_color)
    return out


def _compose_combo_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, ctrl_color: int) -> np.ndarray:
    fill_tile = _recolored_tile(tile, ctrl_color)
    hole_tile = _invert_tile(fill_tile, bg)
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = fill_tile if int(ctrl[r, c]) == int(ctrl_color) else hole_tile
            paste(out, patch, r * th, c * tw)
    return out


def _compose_original_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, ctrl_color: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == int(ctrl_color):
                paste(out, tile, r * th, c * tw)
    return out


def _compose_d1_only_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, ctrl_color: int) -> np.ndarray:
    inv_tile = _invert_tile(tile, bg)
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            patch = tile if int(ctrl[r, c]) == int(ctrl_color) else inv_tile
            paste(out, patch, r * th, c * tw)
    return out


def _compose_d13_only_output(tile: np.ndarray, ctrl: np.ndarray, bg: int, ctrl_color: int) -> np.ndarray:
    fill_tile = _recolored_tile(tile, ctrl_color)
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == int(ctrl_color):
                paste(out, fill_tile, r * th, c * tw)
    return out


def _combo_behavior_metrics(tile: np.ndarray, ctrl: np.ndarray, bg: int, ctrl_color: int) -> dict[str, int]:
    combo = _compose_combo_output(tile, ctrl, bg, ctrl_color)
    original = _compose_original_output(tile, ctrl, bg, ctrl_color)
    d1_only = _compose_d1_only_output(tile, ctrl, bg, ctrl_color)
    d13_only = _compose_d13_only_output(tile, ctrl, bg, ctrl_color)
    fill_tile = _recolored_tile(tile, ctrl_color)
    hole_tile = _invert_tile(fill_tile, bg)
    fill_palette = set(int(x) for x in fill_tile.flatten()) - {int(bg)}
    hole_palette = set(int(x) for x in hole_tile.flatten()) - {int(bg)}
    return {
        "diff_vs_original": int(np.sum(combo != original)),
        "diff_vs_d1_only": int(np.sum(combo != d1_only)),
        "diff_vs_d13_only": int(np.sum(combo != d13_only)),
        "hole_count": int(np.sum(ctrl == bg)),
        "fill_count": int(np.sum(ctrl == ctrl_color)),
        "recolored_pixels": int(np.sum(tile == tile[0, 0])),
        "fill_palette_size": int(len(fill_palette)),
        "hole_palette_size": int(len(hole_palette)),
        "palette_overlap": int(len(fill_palette & hole_palette)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    bg, block, ctrl = _identify_parts(grid)
    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)
    ctrl_color = int(ctrl_cols[0])
    return _compose_combo_output(block, ctrl, bg, ctrl_color)


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


def _connected_mask(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, connectivity=8, monochromatic=False, return_format="dict")
    return len(comps) == 1


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    area = h * w
    min_fill = max(2, (area + 1) // 2)
    max_fill = min(area - 2, int(np.floor(0.72 * area)))
    if min_fill > max_fill:
        max_fill = min_fill
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
        pick_r, pick_c = frontier[int(rng.integers(len(frontier)))]
        mask[pick_r, pick_c] = True

    while area - int(mask.sum()) < 2:
        filled = np.argwhere(mask)
        rng.shuffle(filled)
        changed = False
        for rr, cc in filled:
            trial = mask.copy()
            trial[int(rr), int(cc)] = False
            if int(trial.sum()) >= 2 and _connected_mask(trial):
                mask = trial
                changed = True
                break
        if not changed:
            break

    assert int(mask.sum()) >= 2
    assert area - int(mask.sum()) >= 2
    assert _connected_mask(mask)

    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(color)
    return ctrl


def _pick_side(pair: tuple[int, int], rng: np.random.Generator) -> tuple[int, int]:
    a, b = map(int, pair)
    if bool(rng.integers(0, 2)):
        return a, b
    return b, a


def _choose_color_roles(bg: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    usable_pairs = [pair for pair in INV_PAIRS if int(bg) not in pair]
    assert len(usable_pairs) == 4
    rng.shuffle(usable_pairs)
    chosen_pairs = usable_pairs[:4]
    roles = []
    for pair in chosen_pairs:
        chosen, complement = _pick_side(pair, rng)
        roles.append((int(chosen), int(complement)))

    ctrl_color, ctrl_inv = roles[0]
    src_color, src_inv = roles[1]
    base_color, base_inv = roles[2]
    stripe_color, stripe_inv = roles[3]

    fill_colors = {ctrl_color, base_color, stripe_color}
    hole_colors = {ctrl_inv, base_inv, stripe_inv}
    input_tile_colors = {src_color, base_color, stripe_color}

    assert len(fill_colors) == 3
    assert len(hole_colors) == 3
    assert len(input_tile_colors) == 3
    assert fill_colors.isdisjoint(hole_colors)
    assert int(bg) not in fill_colors
    assert int(bg) not in hole_colors

    return ctrl_color, src_color, base_color, stripe_color


def _make_tile(bg: int, ctrl_color: int, src_color: int, base_color: int, stripe_color: int, rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(4, 6))
    tile = np.full((size, size), int(base_color), dtype=int)

    tile[0, 0] = int(src_color)
    tile[0, -1] = int(src_color)
    tile[-1, 0] = int(src_color)
    tile[-1, -1] = int(src_color)

    mid = size // 2
    tile[mid, :] = int(stripe_color)
    tile[:, mid] = int(stripe_color)

    if size >= 5 and rng.random() < 0.5:
        tile[1, 1] = int(base_color)
        tile[1, -2] = int(base_color)
        tile[-2, 1] = int(base_color)
        tile[-2, -2] = int(base_color)

    tile[0, 0] = int(src_color)
    tile[0, -1] = int(src_color)
    tile[-1, 0] = int(src_color)
    tile[-1, -1] = int(src_color)

    assert int(tile[0, 0]) == int(src_color)
    assert int(src_color) != int(ctrl_color)
    assert len(colors_present(tile, exclude={bg})) == 3
    assert np.count_nonzero(tile == int(src_color)) >= 4
    return tile


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
    bg = int(rng.integers(0, 10))
    ctrl_color, src_color, base_color, stripe_color = _choose_color_roles(bg, rng)

    tile = _make_tile(bg, ctrl_color, src_color, base_color, stripe_color, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)

    expected = _compose_combo_output(tile, ctrl, bg, ctrl_color)
    metrics = _combo_behavior_metrics(tile, ctrl, bg, ctrl_color)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    colors_per_comp = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        colors_per_comp.append(sorted(set(int(x) for x in patch.flatten()) - {bg}))
    if sorted(len(cols) for cols in colors_per_comp) != [1, 3]:
        return None

    if metrics["diff_vs_original"] <= 0:
        return None
    if metrics["diff_vs_d1_only"] <= 0:
        return None
    if metrics["diff_vs_d13_only"] <= 0:
        return None
    if metrics["hole_count"] < 2:
        return None
    if metrics["fill_count"] < 2:
        return None
    if int(tile[0, 0]) == int(ctrl_color):
        return None
    if metrics["recolored_pixels"] < 4:
        return None
    if metrics["fill_palette_size"] != 3:
        return None
    if metrics["hole_palette_size"] != 3:
        return None
    if metrics["palette_overlap"] != 0:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid combo variant input")
