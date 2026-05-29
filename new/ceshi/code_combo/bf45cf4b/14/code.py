# Combo variant of bf45cf4b (index 14)
# combo_spec: D7/2+D13/1
# Name: 行色移位填充棋盘空洞
#
# Design analysis
# - Concrete behavior:
#   Original task: each filled control pixel stamps the same multicolor tile; each hole expands to plain bg.
#   Fusion here: filled pixels still stamp the tile, but row r of the control uses a color-cycled version of the
#   tile; holes stamp a checkerboard built from the tile's two dominant colors.
#   Example with tile colors [A,B,C] in row-major first-appearance order and control row index r:
#   row 0 filled -> A/B/C unchanged; row 1 filled -> A->B, B->C, C->A; row 2 filled -> A->C, B->A, C->B.
#   Any hole, regardless of row, becomes an A/B checkerboard instead of plain bg.
# - Complexity calibration:
#   The generator fixes the tile to exactly 3 foreground colors so the cycle is small and repeated, and the control
#   sprite has at least 3 rows with holes in every row so both "row shift" and "checkerboard hole" are observable.
#   The tile is generated with unique dominant color counts, so "two main colors" is visually grounded rather than
#   chosen by an invisible numeric tiebreak.
# - Hidden assumptions audited against the original solver:
#   1. Component identification by "multicolor square vs monochrome rectangle" still holds; generator preserves that.
#   2. `ctrl == ctrl_color` previously meant "filled pixel"; here that still holds, but holes now map to a non-bg tile.
#   3. A new assumption is needed for cycle order. To keep it human-visible, the cycle order is the row-major first
#      appearance order of the tile's foreground colors, which the input tile itself exposes.
#   4. Another new assumption is that the checkerboard colors are the tile's two dominant foreground colors; the
#      generator ensures their frequencies are strictly ordered.
# - Placement legality:
#   Input placement keeps the two objects inside a 12-18 by 12-18 grid with at least one full bg row/column gap,
#   so the 8-connected foreground still splits into exactly two components.
# - Visual observability / cleanliness:
#   Each control row contains both fills and holes, so the output shows shifted tile copies and checkerboard cavities
#   side by side. The tile uses only 3 foreground colors, preventing the row-shift rule from becoming noisy.

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste


def _scan_order_colors(tile: np.ndarray, bg: int) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for value in tile.flatten():
        color = int(value)
        if color == bg or color in seen:
            continue
        seen.add(color)
        order.append(color)
    return order


def _dominant_pair(tile: np.ndarray, bg: int) -> tuple[int, int]:
    order = _scan_order_colors(tile, bg)
    counts = {color: int(np.sum(tile == color)) for color in order}
    rank = {color: idx for idx, color in enumerate(order)}
    ordered = sorted(order, key=lambda color: (-counts[color], rank[color]))
    assert len(ordered) >= 2
    return int(ordered[0]), int(ordered[1])


def _shift_tile(tile: np.ndarray, bg: int, cycle: list[int], shift: int) -> np.ndarray:
    if not cycle:
        return tile.copy()
    step = int(shift) % len(cycle)
    if step == 0:
        return tile.copy()

    mapping = {
        int(cycle[idx]): int(cycle[(idx + step) % len(cycle)])
        for idx in range(len(cycle))
    }
    out = tile.copy()
    for src, dst in mapping.items():
        out[tile == src] = dst
    out[tile == bg] = int(bg)
    return out


def _checkerboard(h: int, w: int, color_a: int, color_b: int) -> np.ndarray:
    rr, cc = np.indices((int(h), int(w)))
    return np.where((rr + cc) % 2 == 0, int(color_a), int(color_b)).astype(int)


def _extract_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append(
            {
                "patch": patch,
                "cols": cols,
                "is_square": int(patch.shape[0]) == int(patch.shape[1]),
            }
        )

    infos.sort(
        key=lambda info: (
            len(info["cols"]) >= 2,
            len(info["cols"]),
            info["is_square"],
        ),
        reverse=True,
    )

    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]
    assert len(colors_present(block, exclude={bg})) >= 2
    assert len(colors_present(ctrl, exclude={bg})) == 1
    return block, ctrl


def _compose_fused_output(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)

    cycle = _scan_order_colors(block, bg)
    assert len(cycle) >= 2
    main_a, main_b = _dominant_pair(block, bg)
    hole_tile = _checkerboard(bh, bw, main_a, main_b)

    shifted_rows = [_shift_tile(block, bg, cycle, r) for r in range(ch)]
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        shifted = shifted_rows[r]
        for c in range(cw):
            patch = shifted if int(ctrl[r, c]) != bg else hole_tile
            paste(out, patch, r * bh, c * bw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    block, ctrl = _extract_objects(g, bg)
    return _compose_fused_output(block, ctrl, bg)


def _tile_template(size: int, colors: list[int]) -> np.ndarray:
    a, b, c = map(int, colors)
    if int(size) == 3:
        return np.array(
            [
                [a, a, b],
                [a, a, b],
                [b, c, a],
            ],
            dtype=int,
        )
    if int(size) == 4:
        return np.array(
            [
                [a, a, b, a],
                [a, a, b, a],
                [b, b, c, b],
                [a, a, b, a],
            ],
            dtype=int,
        )
    return np.array(
        [
            [a, a, b, a, a],
            [a, a, b, a, a],
            [b, b, c, b, b],
            [a, a, b, a, a],
            [a, a, b, a, a],
        ],
        dtype=int,
    )


def _random_dihedral(tile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(tile, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    return np.array(out, dtype=int)


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    tile = _tile_template(size, tile_colors)
    tile = _random_dihedral(tile, rng)
    assert len(colors_present(tile, exclude={bg})) == 3
    dom_a, dom_b = _dominant_pair(tile, bg)
    counts = {color: int(np.sum(tile == color)) for color in colors_present(tile, exclude={bg})}
    assert counts[dom_a] > counts[dom_b] > min(counts.values())
    return tile


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 7))
    w = int(rng.integers(3, 7))
    phase = int(rng.integers(0, 3))

    rr, cc = np.indices((h, w))
    mask = ((rr + cc + phase) % 3 != 0)
    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(color)

    # Every row has both fills and holes, which makes the fused behavior visible row by row.
    assert all(np.any(ctrl[r] == color) and np.any(ctrl[r] == bg) for r in range(h))
    comps = find_components(ctrl, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 1
    return ctrl


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    horizontal = bool(rng.integers(0, 2))
    gap = 1

    if horizontal:
        if bool(rng.integers(0, 2)):
            left_obj, right_obj = tile, ctrl
            left_is_tile = True
        else:
            left_obj, right_obj = ctrl, tile
            left_is_tile = False
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        H = int(rng.integers(max(12, max(lh, rh)), 19))
        W = int(rng.integers(max(12, lw + rw + gap), 19))
        slack = W - (lw + rw + gap)
        c_left = int(rng.integers(0, slack + 1))
        c_right = int(rng.integers(c_left + lw + gap, W - rw + 1))
        r_left = int(rng.integers(0, H - lh + 1))
        r_right = int(rng.integers(0, H - rh + 1))
        if left_is_tile:
            tile_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            tile_pos = (r_right, c_right)
    else:
        if bool(rng.integers(0, 2)):
            top_obj, bot_obj = tile, ctrl
            top_is_tile = True
        else:
            top_obj, bot_obj = ctrl, tile
            top_is_tile = False
        th, tw = top_obj.shape
        bh, bw = bot_obj.shape
        H = int(rng.integers(max(12, th + bh + gap), 19))
        W = int(rng.integers(max(12, max(tw, bw)), 19))
        slack = H - (th + bh + gap)
        r_top = int(rng.integers(0, slack + 1))
        r_bot = int(rng.integers(r_top + th + gap, H - bh + 1))
        c_top = int(rng.integers(0, W - tw + 1))
        c_bot = int(rng.integers(0, W - bw + 1))
        if top_is_tile:
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _plain_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    tile_colors = palette[1:4]
    ctrl_color = int(palette[4])

    tile = _make_tile(bg, tile_colors, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_fused_output(tile, ctrl, bg)

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    block, extracted_ctrl = _extract_objects(inp, bg)
    if not np.array_equal(main(inp), expected):
        return None

    # Anti-degeneration checks: the fused behavior must differ from the original plain expansion.
    plain = _plain_output(block, extracted_ctrl, bg)
    if np.array_equal(expected, plain):
        return None

    cycle = _scan_order_colors(block, bg)
    if len(cycle) < 3:
        return None
    hole_count = int(np.sum(extracted_ctrl == bg))
    shifted_rows = sum(
        1
        for r in range(extracted_ctrl.shape[0])
        if np.any(extracted_ctrl[r] != bg) and (r % len(cycle)) != 0
    )
    if hole_count == 0 or shifted_rows == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
