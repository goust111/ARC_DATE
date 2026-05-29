# Combo variant of bf45cf4b (index 21)
# combo_spec: D6/2+D7/2
# Name: 边缘内部分化棋盘空洞

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior:
#    - Input still contains exactly two 8-connected foreground components:
#      a monochrome control sprite with bg holes, and a polychrome square tile.
#    - Output is still the control bbox scaled by the tile size, but each control
#      cell now selects among three patch types:
#        * filled edge cell: original tile;
#        * filled strict interior cell: color-inverted tile;
#        * hole cell: checkerboard made from {bg, ctrl_color}.
#    - "Edge" is not just bbox border membership. A filled cell is edge iff at
#      least one of its 8 neighboring positions is empty or outside the bbox.
#      Interior means filled and all 8 neighbors are also filled.
#    - Small example with a 5x5 control mask:
#        11111
#        11011
#        11111
#        11111
#        11111
#      Original task: every 1 stamps tile T, every 0 becomes bg.
#      Combo task: cells around the hole and outer perimeter stamp T, cells in the
#      fully surrounded core stamp invert(T), and the 0 cell becomes checkerboard.
#      So the difference is visible in both filled and empty regions.
# b) Complexity calibration:
#    - The generator uses exactly two tile colors, so "color inversion" becomes a
#      direct color swap on the same geometry instead of a hard-to-infer remap.
#    - The control is forced to contain at least one strict interior fill and at
#      least one hole, so all three output patch types repeat within one sample.
# c) Implicit assumptions changed from the original solver:
#    - Original `paste(out, block, ...)` assumed one tile for all filled cells.
#      Now tile choice depends on the local 8-neighborhood of the control mask.
#    - Original hole handling assumed pure bg rectangles. That assumption is
#      invalid; hole regions now carry structure and a second color (ctrl color).
#    - New solver assumption: the block colors are distinguishable enough that the
#      inverted tile differs from the original. The generator guarantees exactly
#      two non-bg tile colors and avoids a checkerboard tile that would blur the
#      distinction with hole patches.
# d) Placement legality:
#    - The generator still places tile and control as two disconnected objects with
#      at least one background gap and at least one background border cell.
#    - Sizes are restricted so both objects fit inside a 12-18 by 12-18 input.
# e) Visual observability:
#    - Humans can see the control holes in the input, then see matching checkerboard
#      patches at those scaled positions in the output.
#    - They can also compare filled cells near empty space versus fully surrounded
#      filled cells and notice the same tile geometry with swapped colors inside.
# f) Visual cleanliness:
#    - No extra markers are added. The three patch types occupy disjoint tile-sized
#      cells in the scaled output, so there is no overlap ambiguity.
#    - Hole checkerboards use only bg and ctrl color, keeping them visually separate
#      from the tile palette.
# g) Color inference:
#    - Tile inversion is structural: the same pattern keeps its geometry while the
#      two tile colors swap roles. Hole filling is also structural: it always uses
#      the input bg and control color in a checkerboard, with enough repetitions to
#      infer the rule from a few examples.


def _extract_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
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
            int(d["patch"].shape[0]) == int(d["patch"].shape[1]),
        ),
        reverse=True,
    )
    return infos[0]["patch"], infos[1]["patch"]


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w or not bool(mask[rr, cc]):
                        out[r, c] = True
                        break
                if bool(out[r, c]):
                    break
    return out


def _invert_tile(tile: np.ndarray, bg: int) -> np.ndarray:
    cols = colors_present(tile, exclude={bg})
    if len(cols) <= 1:
        return tile.copy()
    swapped = tile.copy()
    rev = list(reversed(cols))
    mapping = {int(src): int(dst) for src, dst in zip(cols, rev)}
    for src, dst in mapping.items():
        swapped[tile == int(src)] = int(dst)
    return swapped


def _checker_patch(h: int, w: int, color_a: int, color_b: int, phase: int = 0) -> np.ndarray:
    rr, cc = np.indices((h, w))
    return np.where((rr + cc + int(phase)) % 2 == 0, int(color_a), int(color_b)).astype(int)


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)

    ctrl_cols = colors_present(ctrl, exclude={bg})
    assert len(ctrl_cols) == 1
    ctrl_col = int(ctrl_cols[0])
    mask = ctrl == ctrl_col
    edge = _edge_mask(mask)
    interior = mask & ~edge

    inv_tile = _invert_tile(tile, bg)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            rr = r * th
            cc = c * tw
            if bool(edge[r, c]):
                paste(out, tile, rr, cc)
            elif bool(interior[r, c]):
                paste(out, inv_tile, rr, cc)
            else:
                hole_patch = _checker_patch(th, tw, bg, ctrl_col, phase=(rr + cc) % 2)
                paste(out, hole_patch, rr, cc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tile, ctrl = _extract_objects(g, bg)
    return _compose_output(tile, ctrl, bg)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(int(x) for x in coords[0])}
    stack = [tuple(int(x) for x in coords[0])]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _make_tile(bg: int, fg_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    size = int(rng.integers(3, 6))
    a, b = [int(x) for x in rng.choice(fg_colors, size=2, replace=False)]
    tile = np.full((size, size), int(a), dtype=int)

    # Keep geometry fixed and non-checkerboard so color-swapping is the only change.
    tile[0, :] = int(b)
    tile[:, -1] = int(b)
    tile[size // 2, : max(2, size - 1)] = int(b)
    tile[1:size - 1, 1] = int(b)
    tile[size // 2, 1] = int(a)

    checker_a = _checker_patch(size, size, a, b, phase=0)
    checker_b = _checker_patch(size, size, a, b, phase=1)
    assert not np.array_equal(tile, checker_a)
    assert not np.array_equal(tile, checker_b)
    assert not np.array_equal(tile, _invert_tile(tile, bg))
    return tile


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(6, 9))
    w = int(rng.integers(6, 9))

    center_r = int(rng.integers(2, h - 2))
    center_c = int(rng.integers(2, w - 2))
    mask = np.ones((h, w), dtype=bool)
    protected = np.zeros_like(mask, dtype=bool)
    protected[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = True

    interior_cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if not bool(protected[r, c])]
    assert interior_cells
    rng.shuffle(interior_cells)
    forced_hole = interior_cells[0]
    mask[forced_hole] = False

    border_cells = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
    rng.shuffle(border_cells)
    for r, c in border_cells[: int(rng.integers(0, max(1, len(border_cells) // 4) + 1))]:
        if not bool(mask[r, c]):
            continue
        trial = mask.copy()
        trial[r, c] = False
        if _is_connected(trial):
            mask = trial

    extra_holes = int(rng.integers(0, max(2, len(interior_cells) // 3)))
    for r, c in interior_cells[1:]:
        if extra_holes <= 0:
            break
        if rng.random() < 0.45:
            trial = mask.copy()
            trial[r, c] = False
            if _is_connected(trial) and np.any((trial & ~_edge_mask(trial))):
                mask = trial
                extra_holes -= 1

    edge = _edge_mask(mask)
    interior = mask & ~edge
    assert np.any(interior)
    assert np.any(~mask)

    ctrl = np.full((h, w), int(bg), dtype=int)
    ctrl[mask] = int(color)
    return ctrl


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
        th, tw = map(int, top_obj.shape)
        bh, bw = map(int, bot_obj.shape)
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


def _plain_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if ctrl[r, c] != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=4, replace=False)]
    bg = int(palette[0])
    tile_palette = palette[1:3]
    ctrl_color = int(palette[3])

    tile = _make_tile(bg, tile_palette, rng)
    ctrl = _make_ctrl(bg, ctrl_color, rng)
    inp = _make_input(tile, ctrl, bg, rng)
    expected = _compose_output(tile, ctrl, bg)
    plain = _plain_output(tile, ctrl, bg)

    if np.array_equal(expected, plain):
        return None

    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return None

    block, ctrl_patch = _extract_objects(inp, bg)
    if len(colors_present(block, exclude={bg})) != 2:
        return None
    ctrl_cols = colors_present(ctrl_patch, exclude={bg})
    if len(ctrl_cols) != 1:
        return None

    mask = ctrl_patch == int(ctrl_cols[0])
    edge = _edge_mask(mask)
    interior = mask & ~edge
    if not np.any(edge):
        return None
    if not np.any(interior):
        return None
    if not np.any(~mask):
        return None

    inv_tile = _invert_tile(block, bg)
    if np.array_equal(block, inv_tile):
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
